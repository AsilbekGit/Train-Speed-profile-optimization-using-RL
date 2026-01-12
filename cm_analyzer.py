import numpy as np
import matplotlib.pyplot as plt
import os
import config
from utils import discretize_state
import time

class CMAnalyzer:
    def __init__(self, env):
        self.env = env
        # Q-Table dimensions: [Segments, Velocity_Bins, Actions]
        self.q_shape = (env.n_segments, 100, 4)
        self.q_curr = np.zeros(self.q_shape)
        self.q_prev = np.zeros(self.q_shape)
        
        # History tracking
        self.delta_history = []      # ΔQi values
        self.cm_history = []          # ΔQi/ΔQi-1 (ratio)
        self.ln_cm_history = []       # ln(ΔQi/ΔQi-1) - FOR PLOTTING like Fig 5
        
        # Hyperparams (using SARSA for CM analysis - see paper Section 3.4)
        self.alpha = 0.1
        self.gamma = 0.95
        self.epsilon = 0.15
        
        # Debug settings
        self.debug_mode = config.DEBUG_MODE
        self.print_every_step = config.PRINT_EVERY_STEP
        
    def detect_local_optimum_threshold(self):
        """
        Automatically detect YOUR local optimum threshold from ln(CM) data
        
        Paper methodology (Section 3.4):
        "the algorithm starts to fail to local optimums as the ln(ΔQi/ΔQi−1) 
         goes below −3.21"
        
        We detect this by:
        1. Taking last 20% of episodes (stable convergence region)
        2. Calculating median ln(CM) in that region
        3. This is where YOUR algorithm starts failing to local optimum
        
        Returns:
            ln_threshold: YOUR ln(CM) threshold value
            phi_threshold: Corresponding φ = e^(ln_threshold)
        """
        ln_cm_array = np.array(self.ln_cm_history)
        
        # Take last 20% of episodes (stable region)
        stable_start = int(len(ln_cm_array) * 0.8)
        stable_region = ln_cm_array[stable_start:]
        
        if len(stable_region) < 10:
            # Not enough data, use all data
            stable_region = ln_cm_array
        
        # Calculate threshold as median of stable region
        ln_threshold = np.median(stable_region)
        
        # Convert to φ ratio
        phi_threshold = np.exp(ln_threshold)
        
        return ln_threshold, phi_threshold
    
    def run(self, episodes=25000):  # Paper used 25,000 scenarios
        """
        Run CM Analysis using SARSA algorithm (Equation 27)
        Purpose: Find YOUR φ threshold where local optimum occurs
        
        Paper methodology (Section 3.4):
        - Run SARSA for 25,000 scenarios
        - Calculate ln(ΔQi/ΔQi-1) for each episode
        - Find where ln(CM) stabilizes → that's YOUR threshold
        """
        print(f"\n{'='*70}")
        print(f"CM ANALYSIS - Finding YOUR φ Threshold")
        print(f"{'='*70}")
        print(f"Methodology: Paper Section 3.4, Figure 5")
        print(f"Algorithm: SARSA (Equation 27)")
        print(f"Total Episodes: {episodes}")
        print(f"Expected: ln(CM) decreases from ~0 to stable negative value")
        print(f"YOUR threshold will be detected automatically")
        print(f"{'='*70}\n")
        
        start_time = time.time()
        success_count = 0
        
        for ep in range(1, episodes + 1):
            episode_start = time.time()
            
            # Reset environment
            self.env.reset()
            state = self.env._get_state()
            s_idx, v_idx = discretize_state(state)
            
            total_reward = 0.0
            steps = 0
            stuck_counter = 0
            prev_position = 0.0
            
            # Initial Action (ε-greedy)
            if np.random.rand() < self.epsilon:
                action = np.random.randint(4)
            else:
                action = np.argmax(self.q_curr[s_idx, v_idx])
            
            if self.debug_mode and ep <= 2:  # Debug only first 2 episodes
                print(f"\n{'='*70}")
                print(f"EPISODE {ep} START")
                print(f"{'='*70}")
                print(f"Initial State: seg={s_idx}, v={self.env.v:.2f}m/s")
                print(f"Initial Action: {config.ACTION_NAMES[action]}")
            
            # Episode loop
            while steps < config.MAX_STEPS_PER_EPISODE:
                current_position = self.env.seg_idx + (self.env.pos_in_seg / config.DX)
                
                # Step in environment
                next_state_raw, reward, done, _ = self.env.step(action)
                ns_idx, nv_idx = discretize_state(next_state_raw)
                
                total_reward += reward
                steps += 1
                
                # Print step details if enabled
                if self.print_every_step and ep <= 2:
                    print(f"  Step {steps:4d} | "
                          f"Seg: {s_idx:3d} | "
                          f"V: {self.env.v:6.2f}m/s | "
                          f"Action: {config.ACTION_NAMES[action]:8s} | "
                          f"Reward: {reward:8.2f}")
                
                # Stuck detection
                if abs(current_position - prev_position) < config.POSITION_EPSILON:
                    stuck_counter += 1
                    if stuck_counter >= config.STUCK_THRESHOLD:
                        if self.debug_mode:
                            print(f"\n  ⚠️ STUCK at step {steps}, ending episode")
                        done = True
                else:
                    stuck_counter = 0
                
                prev_position = current_position
                
                # Next Action (SARSA-style: ε-greedy)
                if np.random.rand() < self.epsilon:
                    next_action = np.random.randint(4)
                else:
                    next_action = np.argmax(self.q_curr[ns_idx, nv_idx])
                
                # SARSA Update (Equation 27 from paper)
                # Q(s,a) ← Q(s,a) + η[r + γQ(s',a') - Q(s,a)]
                target = reward + self.gamma * self.q_curr[ns_idx, nv_idx, next_action]
                if done:
                    target = reward
                
                old_q = self.q_curr[s_idx, v_idx, action]
                self.q_curr[s_idx, v_idx, action] += self.alpha * (target - old_q)
                
                # Move to next state
                s_idx, v_idx = ns_idx, nv_idx
                action = next_action
                
                if done:
                    break
            
            episode_time = time.time() - episode_start
            
            # --- Calculate ΔQi (sum of absolute Q-table changes) ---
            delta_i = np.sum(np.abs(self.q_curr - self.q_prev))
            self.delta_history.append(delta_i)
            
            # Update previous Q for next iteration
            self.q_prev = self.q_curr.copy()
            
            # --- Calculate CM: ΔQi / ΔQi-1 (Equations 28, 29) ---
            if len(self.delta_history) >= 2:
                delta_n = self.delta_history[-1]
                delta_n_minus_1 = self.delta_history[-2]
                
                if delta_n_minus_1 > 1e-9:
                    cm_ratio = delta_n / delta_n_minus_1
                else:
                    cm_ratio = 1.0
                
                # Store ratio (for actual Q-SARSA use later)
                self.cm_history.append(cm_ratio)
                
                # --- Calculate ln(ΔQi/ΔQi-1) for plotting (Figure 5) ---
                if cm_ratio > 1e-9:
                    ln_cm = np.log(cm_ratio)  # Natural logarithm
                else:
                    ln_cm = -10.0  # Very small ratio → large negative ln
                
                self.ln_cm_history.append(ln_cm)
            else:
                # First episode: no previous delta to compare
                self.cm_history.append(1.0)
                self.ln_cm_history.append(0.0)
            
            # Track success rate
            if self.env.seg_idx >= self.env.n_segments - 1:
                success_count += 1
            
            # --- Print Episode Summary ---
            if ep % 100 == 0 or ep <= 10:  # Print first 10, then every 100
                elapsed = time.time() - start_time
                avg_time_per_ep = elapsed / ep
                remaining = (episodes - ep) * avg_time_per_ep
                rem_str = time.strftime("%H:%M:%S", time.gmtime(remaining))
                
                success_marker = "✓" if self.env.seg_idx >= self.env.n_segments - 1 else "✗"
                
                # Get current values
                current_ln_cm = self.ln_cm_history[-1] if self.ln_cm_history else 0.0
                current_cm = self.cm_history[-1] if self.cm_history else 1.0
                
                print(f"{success_marker} Ep {ep:05d}/{episodes} | "
                      f"ln(CM): {current_ln_cm:7.3f} | "
                      f"CM: {current_cm:6.4f} | "
                      f"ΔQ: {delta_i:10.2f} | "
                      f"Success: {success_count}/{ep} | "
                      f"ETA: {rem_str}")
        
        print(f"\n{'='*70}")
        print(f"CM ANALYSIS COMPLETE")
        print(f"{'='*70}")
        elapsed_total = time.time() - start_time
        print(f"Total time: {time.strftime('%H:%M:%S', time.gmtime(elapsed_total))}")
        print(f"Success rate: {success_count}/{episodes} ({success_count/episodes*100:.1f}%)")
        
        # Detect YOUR threshold
        print(f"\n🔍 Detecting YOUR local optimum threshold...")
        ln_threshold, phi_threshold = self.detect_local_optimum_threshold()
        
        print(f"\n{'='*70}")
        print(f"YOUR THRESHOLD DETECTED")
        print(f"{'='*70}")
        print(f"  ln(CM) threshold = {ln_threshold:.4f}")
        print(f"  φ threshold = {phi_threshold:.4f}")
        print(f"\n  This is YOUR φ value to use in Q-SARSA training!")
        print(f"  (Paper found φ = 0.04 for Tehran/Shiraz Metro)")
        print(f"{'='*70}\n")
        
        print(f"Generating plots with YOUR threshold...")
        self.save_plot(ln_threshold, phi_threshold)
        self.save_data(ln_threshold, phi_threshold)
    
    def save_plot(self, ln_threshold, phi_threshold):
        """
        Generate plot matching Paper's Figure 5
        Shows YOUR detected threshold (not paper's)
        
        Args:
            ln_threshold: YOUR detected ln(CM) threshold
            phi_threshold: YOUR detected φ threshold
        """
        print("\n📊 Generating CM analysis plot (Figure 5 style)...")
        
        ln_cm_array = np.array(self.ln_cm_history)
        
        # Create figure matching paper's Figure 5
        fig, ax = plt.subplots(1, 1, figsize=(14, 8))
        
        # Plot ln(ΔQi/ΔQi-1) - EXACTLY like Figure 5
        ax.plot(ln_cm_array, linewidth=1.0, color='#4682B4', alpha=0.8, label='Your ln(CM) values')
        
        # Add YOUR threshold line (RED - detected from your data)
        ax.axhline(y=ln_threshold, color='red', linestyle='--', linewidth=2.5, 
                   label=f'YOUR threshold: ln(φ) = {ln_threshold:.3f} (φ = {phi_threshold:.4f})')
        
        # Calculate where to place annotation (at ~70% of x-axis, at threshold height)
        annotation_x = len(ln_cm_array) * 0.7
        annotation_y_arrow = ln_threshold
        annotation_y_text = ln_threshold + (max(ln_cm_array) - ln_threshold) * 0.3
        
        # Add annotation box like in Figure 5
        ax.annotate('Starts failing\nto local\noptimum', 
                    xy=(annotation_x, annotation_y_arrow), 
                    xytext=(annotation_x * 1.05, annotation_y_text),
                    fontsize=14, fontweight='bold',
                    bbox=dict(boxstyle='round,pad=0.8', facecolor='#4682B4', 
                             alpha=0.7, edgecolor='black', linewidth=2),
                    arrowprops=dict(arrowstyle='->', color='red', lw=2.5),
                    color='white',
                    ha='left')
        
        # Styling to match Figure 5
        ax.set_xlabel('Iteration', fontsize=14, fontweight='bold')
        ax.set_ylabel('Ln(ΔQi/ΔQi-1)', fontsize=14, fontweight='bold')
        ax.set_title(f'Diagram of averaged ln(ΔQi/ΔQi-1) changes for local optimum cases\n' + 
                     f'YOUR Threshold: φ = {phi_threshold:.4f}', 
                     fontsize=16, fontweight='bold', pad=20)
        
        ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
        ax.legend(fontsize=12, loc='upper right')
        
        # Set y-axis limits
        y_min = max(min(ln_cm_array), -4.5)
        y_max = max(max(ln_cm_array), 1.5)
        ax.set_ylim([y_min, y_max])
        
        plt.tight_layout()
        
        # Save main plot
        save_path = os.path.join(config.OUTPUT_DIR, "cm_plot_figure5.png")
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✓ Figure 5 style plot saved to: {save_path}")
        
        # --- Additional Analysis Plot ---
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))
        
        # Top: ln(CM) with YOUR threshold
        ax1.plot(ln_cm_array, linewidth=1.0, color='#4682B4', alpha=0.8, label='Your ln(CM)')
        ax1.axhline(y=ln_threshold, color='red', linestyle='--', linewidth=2, 
                   label=f'YOUR threshold: {ln_threshold:.3f}')
        
        # Add shaded region below threshold
        ax1.fill_between(range(len(ln_cm_array)), ln_threshold, y_min, 
                        alpha=0.1, color='red', label='Local optimum region')
        
        ax1.set_xlabel('Episode', fontsize=12)
        ax1.set_ylabel('ln(ΔQi/ΔQi-1)', fontsize=12)
        ax1.set_title('YOUR ln(CM) Analysis with Detected Threshold', fontsize=14, fontweight='bold')
        ax1.legend(fontsize=10)
        ax1.grid(True, alpha=0.3)
        ax1.set_ylim([y_min, y_max])
        
        # Bottom: Raw ΔQ values (for reference)
        ax2.plot(self.delta_history, linewidth=1.0, color='green', alpha=0.7)
        ax2.set_xlabel('Episode', fontsize=12)
        ax2.set_ylabel('ΔQi (Q-table change)', fontsize=12)
        ax2.set_title('Q-table Changes per Episode (should decrease)', fontsize=12)
        ax2.set_yscale('log')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        analysis_path = os.path.join(config.OUTPUT_DIR, "cm_analysis_detailed.png")
        plt.savefig(analysis_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✓ Detailed analysis plot saved to: {analysis_path}")
        
        # Print statistics
        print(f"\n📈 YOUR CM ANALYSIS:")
        print(f"   ln(CM) statistics:")
        print(f"     Mean:   {np.mean(ln_cm_array):.4f}")
        print(f"     Median: {np.median(ln_cm_array):.4f}")
        print(f"     Min:    {np.min(ln_cm_array):.4f}")
        print(f"     Max:    {np.max(ln_cm_array):.4f}")
        print(f"\n   YOUR DETECTED THRESHOLD:")
        print(f"     ln(φ) = {ln_threshold:.4f}")
        print(f"     φ = {phi_threshold:.4f}")
        
        # Compare with paper
        print(f"\n   For comparison:")
        print(f"     Paper's φ = 0.04 (Tehran/Shiraz Metro)")
        print(f"     YOUR φ = {phi_threshold:.4f}")
        
        # Count episodes below threshold
        below_threshold = np.sum(ln_cm_array < ln_threshold)
        print(f"\n   Episodes below threshold: {below_threshold}/{len(ln_cm_array)} "
              f"({below_threshold/len(ln_cm_array)*100:.1f}%)")
    
    def save_data(self, ln_threshold, phi_threshold):
        """Save CM data with YOUR detected threshold"""
        print("\n💾 Saving CM analysis data...")
        
        data_path = os.path.join(config.OUTPUT_DIR, "cm_data.npz")
        np.savez(data_path,
                 ln_cm_history=np.array(self.ln_cm_history),
                 cm_history=np.array(self.cm_history),
                 delta_history=np.array(self.delta_history),
                 q_final=self.q_curr,
                 ln_threshold=ln_threshold,
                 phi_threshold=phi_threshold)
        
        print(f"✓ Data saved to: {data_path}")
        
        # Save text summary
        txt_path = os.path.join(config.OUTPUT_DIR, "cm_summary.txt")
        with open(txt_path, 'w') as f:
            f.write("="*70 + "\n")
            f.write("CM ANALYSIS SUMMARY - YOUR Results\n")
            f.write("="*70 + "\n\n")
            
            f.write("METHODOLOGY (Paper Section 3.4, Figure 5):\n")
            f.write("  1. Run SARSA algorithm for many episodes\n")
            f.write("  2. Calculate ln(ΔQi/ΔQi-1) for each episode\n")
            f.write("  3. Find where ln(CM) stabilizes (last 20% of episodes)\n")
            f.write("  4. Median of stable region = YOUR threshold\n\n")
            
            f.write("="*70 + "\n")
            f.write("YOUR DETECTED THRESHOLD:\n")
            f.write("="*70 + "\n")
            f.write(f"  ln(φ) = {ln_threshold:.4f}\n")
            f.write(f"  φ = {phi_threshold:.4f}\n\n")
            f.write(f"  👉 USE THIS VALUE (φ = {phi_threshold:.4f}) in Q-SARSA training!\n\n")
            
            ln_cm_array = np.array(self.ln_cm_history)
            cm_array = np.array(self.cm_history)
            
            f.write("="*70 + "\n")
            f.write("STATISTICS:\n")
            f.write("="*70 + "\n")
            f.write(f"Total Episodes: {len(ln_cm_array)}\n\n")
            
            f.write("ln(CM) Statistics:\n")
            f.write(f"  Mean:      {np.mean(ln_cm_array):.4f}\n")
            f.write(f"  Median:    {np.median(ln_cm_array):.4f}\n")
            f.write(f"  Std Dev:   {np.std(ln_cm_array):.4f}\n")
            f.write(f"  Min:       {np.min(ln_cm_array):.4f}\n")
            f.write(f"  Max:       {np.max(ln_cm_array):.4f}\n\n")
            
            # Analyze stable region (last 20%)
            stable_start = int(len(ln_cm_array) * 0.8)
            stable_ln = ln_cm_array[stable_start:]
            stable_cm = cm_array[stable_start:]
            
            f.write("Stable Region (last 20% of episodes):\n")
            f.write(f"  Episodes: {stable_start} to {len(ln_cm_array)}\n")
            f.write(f"  ln(CM) mean:   {np.mean(stable_ln):.4f}\n")
            f.write(f"  ln(CM) median: {np.median(stable_ln):.4f}\n")
            f.write(f"  CM ratio mean: {np.mean(stable_cm):.4f}\n\n")
            
            # Comparison with paper
            f.write("="*70 + "\n")
            f.write("COMPARISON WITH PAPER:\n")
            f.write("="*70 + "\n")
            f.write(f"  Paper (Tehran/Shiraz Metro):\n")
            f.write(f"    ln(φ) = -3.21\n")
            f.write(f"    φ = 0.04\n\n")
            f.write(f"  YOUR Route:\n")
            f.write(f"    ln(φ) = {ln_threshold:.4f}\n")
            f.write(f"    φ = {phi_threshold:.4f}\n\n")
            
            diff_percent = ((phi_threshold - 0.04) / 0.04) * 100
            f.write(f"  Difference: {diff_percent:+.1f}%\n")
            if abs(diff_percent) < 20:
                f.write(f"  → Similar to paper's threshold\n")
            elif phi_threshold > 0.04:
                f.write(f"  → Higher threshold (more stable, switches to Q-learning earlier)\n")
            else:
                f.write(f"  → Lower threshold (more SARSA updates, slower switching)\n")
            
            f.write("\n" + "="*70 + "\n")
            f.write("NEXT STEPS:\n")
            f.write("="*70 + "\n")
            f.write("1. ✓ CM analysis complete\n")
            f.write("2. Check cm_plot_figure5.png - verify threshold looks correct\n")
            f.write(f"3. Use φ = {phi_threshold:.4f} in Q-SARSA training:\n")
            f.write("   python train_qsarsa.py\n")
            f.write(f"   (Enter {phi_threshold:.4f} when prompted for φ)\n")
        
        print(f"✓ Summary saved to: {txt_path}")
        print(f"\n📋 YOUR φ = {phi_threshold:.4f}")
        print(f"   Read {txt_path} for full details!")