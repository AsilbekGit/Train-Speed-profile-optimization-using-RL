import numpy as np
import matplotlib.pyplot as plt
import os
import config
from utils import discretize_state
import time
from scipy.ndimage import uniform_filter1d

class CMAnalyzer:
    def __init__(self, env):
        self.env = env
        # Q-Table dimensions: [Segments, Velocity_Bins, Actions]
        self.q_shape = (env.n_segments, 100, 4)
        self.q_curr = np.zeros(self.q_shape)
        self.q_prev = np.zeros(self.q_shape)
        
        # History tracking
        self.delta_history = []
        self.cm_history = []
        self.ln_cm_history = []
        self.success_history = []  # Track which episodes succeeded
        
        # Hyperparams (SARSA - Equation 27)
        self.alpha = 0.1
        self.gamma = 0.95
        self.epsilon = 0.15
        
        # Debug settings
        self.debug_mode = config.DEBUG_MODE
        self.print_every_step = config.PRINT_EVERY_STEP
        
    def smooth_curve(self, data, window_size=50):
        """
        Apply moving average smoothing to match paper's Figure 5
        Paper's curve is smooth because they averaged multiple runs
        """
        if len(data) < window_size:
            window_size = max(len(data) // 10, 1)
        
        return uniform_filter1d(data, size=window_size, mode='nearest')
    
    def detect_threshold_from_trend(self, ln_cm_smooth):
        """
        Detect threshold where ln(CM) stabilizes at negative value
        
        Paper's Figure 5 shows:
        - Starts around +1
        - Decreases smoothly
        - Stabilizes around -3.21 after ~1000 iterations
        
        Our method:
        1. Find where smoothed curve first goes consistently negative
        2. Take median of last 30% after that point
        """
        # Find first point where curve goes negative and stays negative
        for i in range(len(ln_cm_smooth) - 100):
            # Check if next 100 points are mostly negative
            if np.mean(ln_cm_smooth[i:i+100]) < -0.1:
                # Found start of stable region
                stable_start = i
                break
        else:
            # Didn't find stable region, use last 30%
            stable_start = int(len(ln_cm_smooth) * 0.7)
        
        # Take median of stable region
        stable_region = ln_cm_smooth[stable_start:]
        ln_threshold = np.median(stable_region)
        phi_threshold = np.exp(ln_threshold)
        
        return ln_threshold, phi_threshold, stable_start
    
    def run(self, episodes=25000):
        """
        Run CM Analysis using SARSA (Equation 27)
        
        IMPORTANT: If success rate is 0%, training is meaningless!
        We'll check after first 100 episodes and warn user.
        """
        print(f"\n{'='*70}")
        print(f"CM ANALYSIS - SARSA Algorithm (Paper Equation 27)")
        print(f"{'='*70}")
        print(f"Total Episodes: {episodes}")
        print(f"WARNING: If success rate is 0%, CM analysis won't work!")
        print(f"         Train must be able to complete route at least sometimes")
        print(f"{'='*70}\n")
        
        start_time = time.time()
        success_count = 0
        
        for ep in range(1, episodes + 1):
            # Reset environment
            self.env.reset()
            state = self.env._get_state()
            s_idx, v_idx = discretize_state(state)
            
            total_reward = 0.0
            steps = 0
            stuck_counter = 0
            prev_position = 0.0
            episode_success = False
            
            # Initial Action
            if np.random.rand() < self.epsilon:
                action = np.random.randint(4)
            else:
                action = np.argmax(self.q_curr[s_idx, v_idx])
            
            # Episode loop
            while steps < config.MAX_STEPS_PER_EPISODE:
                current_position = self.env.seg_idx + (self.env.pos_in_seg / config.DX)
                
                # Step
                next_state_raw, reward, done, info = self.env.step(action)
                ns_idx, nv_idx = discretize_state(next_state_raw)
                
                total_reward += reward
                steps += 1
                
                # Stuck detection
                if abs(current_position - prev_position) < config.POSITION_EPSILON:
                    stuck_counter += 1
                    if stuck_counter >= config.STUCK_THRESHOLD:
                        done = True
                else:
                    stuck_counter = 0
                
                prev_position = current_position
                
                # Next Action (SARSA)
                if np.random.rand() < self.epsilon:
                    next_action = np.random.randint(4)
                else:
                    next_action = np.argmax(self.q_curr[ns_idx, nv_idx])
                
                # SARSA Update (Equation 27)
                target = reward + self.gamma * self.q_curr[ns_idx, nv_idx, next_action]
                if done:
                    target = reward
                
                old_q = self.q_curr[s_idx, v_idx, action]
                self.q_curr[s_idx, v_idx, action] += self.alpha * (target - old_q)
                
                # Move to next state
                s_idx, v_idx = ns_idx, nv_idx
                action = next_action
                
                if done:
                    if self.env.seg_idx >= self.env.n_segments - 1:
                        episode_success = True
                    break
            
            # Track success
            if episode_success:
                success_count += 1
            self.success_history.append(episode_success)
            
            # Calculate ΔQi (Equation 28)
            delta_i = np.sum(np.abs(self.q_curr - self.q_prev))
            self.delta_history.append(delta_i)
            self.q_prev = self.q_curr.copy()
            
            # Calculate CM (Equation 29)
            if len(self.delta_history) >= 2:
                delta_n = self.delta_history[-1]
                delta_n_minus_1 = self.delta_history[-2]
                
                if delta_n_minus_1 > 1e-9:
                    cm_ratio = delta_n / delta_n_minus_1
                else:
                    cm_ratio = 1.0
                
                self.cm_history.append(cm_ratio)
                
                # ln(CM) for Figure 5
                if cm_ratio > 1e-9:
                    ln_cm = np.log(cm_ratio)
                else:
                    ln_cm = -10.0
                
                self.ln_cm_history.append(ln_cm)
            else:
                self.cm_history.append(1.0)
                self.ln_cm_history.append(0.0)
            
            # Print progress
            if ep % 100 == 0 or ep <= 10:
                elapsed = time.time() - start_time
                avg_time = elapsed / ep
                remaining = (episodes - ep) * avg_time
                rem_str = time.strftime("%H:%M:%S", time.gmtime(remaining))
                
                success_marker = "✓" if episode_success else "✗"
                current_ln_cm = self.ln_cm_history[-1] if self.ln_cm_history else 0.0
                
                print(f"{success_marker} Ep {ep:05d}/{episodes} | "
                      f"ln(CM): {current_ln_cm:7.3f} | "
                      f"Success: {success_count}/{ep} ({success_count/ep*100:5.1f}%) | "
                      f"ETA: {rem_str}")
            
            # Early warning if 0% success after 100 episodes
            if ep == 100 and success_count == 0:
                print(f"\n{'='*70}")
                print(f"⚠️  WARNING: 0% success rate after 100 episodes!")
                print(f"{'='*70}")
                print(f"This means the train NEVER completes the route.")
                print(f"Possible causes:")
                print(f"  1. Route is too difficult (too long, steep grades)")
                print(f"  2. Initial velocity too low")
                print(f"  3. Time limit (MAX_STEPS_PER_EPISODE) too small")
                print(f"  4. Reward function prevents learning")
                print(f"\nCM analysis will not be meaningful with 0% success.")
                print(f"Recommendations:")
                print(f"  - Increase initial velocity in environment.py reset()")
                print(f"  - Check if route is reasonable (run diagnose.py)")
                print(f"  - Simplify reward function temporarily")
                print(f"{'='*70}\n")
                
                response = input("Continue anyway? (y/n): ")
                if response.lower() != 'y':
                    print("Stopping CM analysis.")
                    return
        
        print(f"\n{'='*70}")
        print(f"CM ANALYSIS COMPLETE")
        print(f"{'='*70}")
        elapsed_total = time.time() - start_time
        print(f"Total time: {time.strftime('%H:%M:%S', time.gmtime(elapsed_total))}")
        print(f"Success rate: {success_count}/{episodes} ({success_count/episodes*100:.1f}%)")
        
        if success_count == 0:
            print(f"\n⚠️  0% SUCCESS RATE - CM analysis is not valid!")
            print(f"    Fix environment issues before using threshold.")
        
        # Smooth and detect threshold
        print(f"\n🔍 Processing CM data...")
        ln_cm_array = np.array(self.ln_cm_history)
        ln_cm_smooth = self.smooth_curve(ln_cm_array, window_size=100)
        
        ln_threshold, phi_threshold, stable_start = self.detect_threshold_from_trend(ln_cm_smooth)
        
        print(f"\n{'='*70}")
        print(f"YOUR THRESHOLD DETECTED")
        print(f"{'='*70}")
        print(f"  ln(φ) = {ln_threshold:.4f}")
        print(f"  φ = {phi_threshold:.4f}")
        print(f"  Stable region starts at episode: {stable_start}")
        print(f"\n  Paper's φ = 0.04 (ln = -3.21) for comparison")
        
        if success_count > 0:
            print(f"  ✓ Use φ = {phi_threshold:.4f} in Q-SARSA training")
        else:
            print(f"  ⚠️  Do NOT use this φ - fix environment first!")
        print(f"{'='*70}\n")
        
        self.save_plot(ln_threshold, phi_threshold, ln_cm_smooth, stable_start)
        self.save_data(ln_threshold, phi_threshold, success_count, episodes)
    
    def save_plot(self, ln_threshold, phi_threshold, ln_cm_smooth, stable_start):
        """Generate Figure 5 style plot with smoothed curve"""
        print("\n📊 Generating plots...")
        
        ln_cm_raw = np.array(self.ln_cm_history)
        
        # Main plot - Figure 5 style
        fig, ax = plt.subplots(1, 1, figsize=(14, 8))
        
        # Plot smoothed ln(CM) - like paper's Figure 5
        ax.plot(ln_cm_smooth, linewidth=2.0, color='#4682B4', alpha=0.9, 
                label='Smoothed ln(CM) (moving average)', zorder=3)
        
        # Plot raw data (lighter, background)
        ax.plot(ln_cm_raw, linewidth=0.5, color='#4682B4', alpha=0.2, 
                label='Raw ln(CM)', zorder=1)
        
        # YOUR threshold line (RED)
        ax.axhline(y=ln_threshold, color='red', linestyle='--', linewidth=2.5, 
                   label=f'YOUR threshold: ln(φ) = {ln_threshold:.3f} (φ = {phi_threshold:.4f})',
                   zorder=4)
        
        # Mark stable region start
        ax.axvline(x=stable_start, color='orange', linestyle=':', linewidth=1.5,
                   label=f'Stable region starts (ep {stable_start})', zorder=2)
        
        # Annotation
        annotation_x = len(ln_cm_smooth) * 0.7
        annotation_y = ln_threshold
        
        ax.annotate('Starts failing\nto local\noptimum', 
                    xy=(annotation_x, annotation_y), 
                    xytext=(annotation_x * 1.05, annotation_y + 1.0),
                    fontsize=14, fontweight='bold',
                    bbox=dict(boxstyle='round,pad=0.8', facecolor='#4682B4', 
                             alpha=0.7, edgecolor='black', linewidth=2),
                    arrowprops=dict(arrowstyle='->', color='red', lw=2.5),
                    color='white', ha='left', zorder=5)
        
        ax.set_xlabel('Iteration', fontsize=14, fontweight='bold')
        ax.set_ylabel('Ln(ΔQi/ΔQi-1)', fontsize=14, fontweight='bold')
        ax.set_title(f'Diagram of averaged ln(ΔQi/ΔQi-1) changes for local optimum cases\n' +
                     f'YOUR Threshold: φ = {phi_threshold:.4f}', 
                     fontsize=16, fontweight='bold', pad=20)
        
        ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
        ax.legend(fontsize=10, loc='upper right')
        
        # Set limits
        y_min = max(min(ln_cm_smooth) - 0.5, -4.5)
        y_max = max(max(ln_cm_smooth) + 0.5, 1.5)
        ax.set_ylim([y_min, y_max])
        
        plt.tight_layout()
        save_path = os.path.join(config.OUTPUT_DIR, "cm_plot_figure5.png")
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✓ Figure 5 plot saved: {save_path}")
        
        # Detailed analysis plot
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))
        
        # Top: Smoothed vs Raw
        ax1.plot(ln_cm_smooth, linewidth=2, color='blue', label='Smoothed', alpha=0.8)
        ax1.plot(ln_cm_raw, linewidth=0.3, color='gray', label='Raw', alpha=0.3)
        ax1.axhline(y=ln_threshold, color='red', linestyle='--', linewidth=2, 
                   label=f'Threshold: {ln_threshold:.3f}')
        ax1.set_xlabel('Episode')
        ax1.set_ylabel('ln(ΔQi/ΔQi-1)')
        ax1.set_title('Smoothed vs Raw ln(CM) - Smoothing reveals trend')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Bottom: Success rate over time
        success_cumulative = np.cumsum(self.success_history) / np.arange(1, len(self.success_history) + 1)
        ax2.plot(success_cumulative * 100, color='green', linewidth=2)
        ax2.set_xlabel('Episode')
        ax2.set_ylabel('Success Rate (%)')
        ax2.set_title('Cumulative Success Rate (should increase over time)')
        ax2.grid(True, alpha=0.3)
        ax2.set_ylim([0, 100])
        
        plt.tight_layout()
        detail_path = os.path.join(config.OUTPUT_DIR, "cm_analysis_detailed.png")
        plt.savefig(detail_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✓ Detailed analysis saved: {detail_path}")
        
        # Statistics
        print(f"\n📈 CM STATISTICS:")
        print(f"   Raw ln(CM): mean={np.mean(ln_cm_raw):.3f}, std={np.std(ln_cm_raw):.3f}")
        print(f"   Smoothed ln(CM): mean={np.mean(ln_cm_smooth):.3f}, std={np.std(ln_cm_smooth):.3f}")
        print(f"   Threshold: ln(φ)={ln_threshold:.3f}, φ={phi_threshold:.4f}")
    
    def save_data(self, ln_threshold, phi_threshold, success_count, total_episodes):
        """Save data and summary"""
        print("\n💾 Saving data...")
        
        data_path = os.path.join(config.OUTPUT_DIR, "cm_data.npz")
        np.savez(data_path,
                 ln_cm_history=np.array(self.ln_cm_history),
                 cm_history=np.array(self.cm_history),
                 delta_history=np.array(self.delta_history),
                 success_history=np.array(self.success_history),
                 q_final=self.q_curr,
                 ln_threshold=ln_threshold,
                 phi_threshold=phi_threshold)
        
        print(f"✓ Data saved: {data_path}")
        
        # Summary
        txt_path = os.path.join(config.OUTPUT_DIR, "cm_summary.txt")
        with open(txt_path, 'w') as f:
            f.write("="*70 + "\n")
            f.write("CM ANALYSIS SUMMARY\n")
            f.write("="*70 + "\n\n")
            
            f.write(f"Success Rate: {success_count}/{total_episodes} ({success_count/total_episodes*100:.1f}%)\n\n")
            
            if success_count == 0:
                f.write("⚠️  WARNING: 0% SUCCESS RATE!\n")
                f.write("="*70 + "\n")
                f.write("The train never completed the route in any episode.\n")
                f.write("This means CM analysis is NOT VALID.\n\n")
                f.write("YOU MUST FIX THE ENVIRONMENT FIRST:\n")
                f.write("  1. Run: python diagnose.py\n")
                f.write("  2. Check if train can complete with full power\n")
                f.write("  3. Increase initial velocity in environment.py\n")
                f.write("  4. Simplify reward function\n")
                f.write("  5. Check route difficulty\n\n")
                f.write("DO NOT USE THE DETECTED THRESHOLD UNTIL SUCCESS RATE > 0%\n")
                f.write("="*70 + "\n\n")
            
            f.write("YOUR DETECTED THRESHOLD:\n")
            f.write(f"  ln(φ) = {ln_threshold:.4f}\n")
            f.write(f"  φ = {phi_threshold:.4f}\n\n")
            
            f.write("Paper's threshold (for comparison):\n")
            f.write(f"  ln(φ) = -3.21\n")
            f.write(f"  φ = 0.04\n\n")
            
            if success_count > 0:
                f.write("✓ Use φ = {:.4f} in Q-SARSA training\n".format(phi_threshold))
            else:
                f.write("⚠️  DO NOT USE THIS φ - Fix environment first!\n")
        
        print(f"✓ Summary saved: {txt_path}")