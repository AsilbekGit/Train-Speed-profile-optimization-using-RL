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
        
        # History
        self.delta_history = []
        self.cm_history = []
        
        # Hyperparams
        self.alpha = 0.1
        self.gamma = 0.95
        self.epsilon = 0.15
        
        # Debug settings
        self.debug_mode = config.DEBUG_MODE
        self.print_every_step = config.PRINT_EVERY_STEP
        
    def run(self, episodes=2000):
        print(f"\n{'='*70}")
        print(f"STARTING CM ANALYSIS")
        print(f"{'='*70}")
        print(f"Total Episodes: {episodes}")
        print(f"Max Steps per Episode: {config.MAX_STEPS_PER_EPISODE}")
        print(f"Segments: {self.env.n_segments}")
        print(f"Q-table shape: {self.q_shape}")
        print(f"Debug Mode: {self.debug_mode}")
        print(f"Step-by-step printing: {self.print_every_step}")
        print(f"{'='*70}\n")
        
        start_time = time.time()
        
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
            
            # Initial Action
            if np.random.rand() < self.epsilon:
                action = np.random.randint(4)
            else:
                action = np.argmax(self.q_curr[s_idx, v_idx])
            
            if self.debug_mode:
                print(f"\n{'='*70}")
                print(f"EPISODE {ep} START")
                print(f"{'='*70}")
                print(f"Initial State: seg={s_idx}, v={self.env.v:.2f}m/s")
                print(f"Initial Action: {config.ACTION_NAMES[action]}")
            
            # Episode loop with max steps
            while steps < config.MAX_STEPS_PER_EPISODE:
                current_position = self.env.seg_idx + (self.env.pos_in_seg / config.DX)
                
                # Step in environment
                next_state_raw, reward, done, _ = self.env.step(action)
                ns_idx, nv_idx = discretize_state(next_state_raw)
                
                total_reward += reward
                steps += 1
                
                # Print step details if enabled
                if self.print_every_step:
                    print(f"  Step {steps:4d} | "
                          f"Seg: {s_idx:3d} | "
                          f"V: {self.env.v:6.2f}m/s | "
                          f"Action: {config.ACTION_NAMES[action]:8s} | "
                          f"Reward: {reward:8.2f} | "
                          f"Pos: {current_position:6.2f}")
                
                # Debug detailed info
                if self.debug_mode and steps % 100 == 0:
                    print(f"\n  [DEBUG Step {steps}]")
                    print(f"    Position in segment: {self.env.pos_in_seg:.2f}m")
                    print(f"    Total time: {self.env.t:.1f}s")
                    print(f"    Total energy: {self.env.energy_kwh:.3f}kWh")
                    print(f"    Cumulative reward: {total_reward:.2f}")
                
                # Stuck detection
                if abs(current_position - prev_position) < 0.001:
                    stuck_counter += 1
                    if stuck_counter >= config.STUCK_THRESHOLD:
                        print(f"\n  ⚠️  STUCK DETECTED at step {steps}")
                        print(f"     Velocity: {self.env.v:.3f}m/s")
                        print(f"     Position: seg={s_idx}, pos_in_seg={self.env.pos_in_seg:.2f}m")
                        print(f"     No movement for {stuck_counter} consecutive steps")
                        print(f"     Ending episode early...\n")
                        done = True
                else:
                    stuck_counter = 0
                
                prev_position = current_position
                
                # Next Action (SARSA Logic)
                if np.random.rand() < self.epsilon:
                    next_action = np.random.randint(4)
                else:
                    next_action = np.argmax(self.q_curr[ns_idx, nv_idx])
                
                # Update Q (SARSA update rule)
                target = reward + self.gamma * self.q_curr[ns_idx, nv_idx, next_action]
                if done:
                    target = reward
                
                old_q = self.q_curr[s_idx, v_idx, action]
                self.q_curr[s_idx, v_idx, action] += self.alpha * (target - old_q)
                
                if self.debug_mode and abs(self.q_curr[s_idx, v_idx, action] - old_q) > 0.01:
                    print(f"    Q-value updated: {old_q:.4f} -> {self.q_curr[s_idx, v_idx, action]:.4f}")
                
                # Move to next state
                s_idx, v_idx = ns_idx, nv_idx
                action = next_action
                
                if done:
                    if self.debug_mode or self.print_every_step:
                        print(f"\n  ✓ Episode completed at step {steps}")
                        print(f"    Final position: seg={self.env.seg_idx}")
                        print(f"    Success: {'YES' if self.env.seg_idx >= self.env.n_segments - 1 else 'NO'}")
                    break
            
            # Check if max steps reached
            if steps >= config.MAX_STEPS_PER_EPISODE and not done:
                print(f"\n  ⏱️  MAX STEPS REACHED ({config.MAX_STEPS_PER_EPISODE})")
                print(f"     Episode terminated early")
                print(f"     Final position: seg={self.env.seg_idx}/{self.env.n_segments}")
            
            episode_time = time.time() - episode_start
            
            # --- Calculate Delta and CM (as per paper) ---
            # Delta_i: Sum of absolute differences between Q_curr and Q_prev
            diff = np.sum(np.abs(self.q_curr - self.q_prev))
            self.delta_history.append(diff)
            
            # Update Previous Q for next iteration
            self.q_prev = self.q_curr.copy()
            
            # Calculate CM: ΔQ_i / ΔQ_{i-1} (Equation 28, 29 from paper)
            if len(self.delta_history) >= 2:
                delta_n = self.delta_history[-1]
                delta_n_minus_1 = self.delta_history[-2]
                
                if delta_n_minus_1 > 1e-9:
                    cm = delta_n / delta_n_minus_1
                else:
                    cm = 0.0
                
                # Cap for plotting safety (values > 3 are outliers)
                cm = min(cm, 3.0)
                self.cm_history.append(cm)
            else:
                cm = 1.0
                self.cm_history.append(cm)
            
            # --- Calculate ETA ---
            elapsed = time.time() - start_time
            avg_time_per_ep = elapsed / ep
            remaining = (episodes - ep) * avg_time_per_ep
            rem_str = time.strftime("%H:%M:%S", time.gmtime(remaining))
            
            # --- PRINT EPISODE SUMMARY ---
            success_marker = "✓" if self.env.seg_idx >= self.env.n_segments - 1 else "✗"
            print(f"\n{success_marker} Ep {ep:04d}/{episodes} | "
                  f"Steps: {steps:5d} | "
                  f"Reward: {total_reward:9.2f} | "
                  f"Delta: {diff:10.4f} | "
                  f"CM: {cm:6.4f} | "
                  f"Time: {episode_time:5.1f}s | "
                  f"ETA: {rem_str}")
            
            if self.debug_mode:
                print(f"  Final Stats:")
                print(f"    Segments completed: {self.env.seg_idx}/{self.env.n_segments}")
                print(f"    Final velocity: {self.env.v:.2f}m/s")
                print(f"    Total time: {self.env.t:.1f}s")
                print(f"    Total energy: {self.env.energy_kwh:.3f}kWh")
                print(f"  Q-table Stats:")
                print(f"    Non-zero Q-values: {np.count_nonzero(self.q_curr)}")
                print(f"    Max Q-value: {np.max(self.q_curr):.4f}")
                print(f"    Min Q-value: {np.min(self.q_curr):.4f}")
            
            print("-" * 70)
        
        print(f"\n{'='*70}")
        print(f"CM ANALYSIS COMPLETE")
        print(f"{'='*70}")
        print(f"Total time: {time.strftime('%H:%M:%S', time.gmtime(time.time() - start_time))}")
        print(f"Total episodes: {episodes}")
        print(f"CM history length: {len(self.cm_history)}")
        print(f"Delta history length: {len(self.delta_history)}")
        
        self.save_plot()
        self.save_data()
    
    def save_plot(self):
        """
        Generate and save CM plot (similar to Figure 5 in the paper)
        The φ reference line is from the paper - YOUR φ may be different!
        """
        print("\n📊 Generating CM plot...")
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
        
        # Plot 1: CM over episodes
        ax1.plot(self.cm_history, linewidth=1.5, color='blue', alpha=0.7, label='Your CM values')
        
        # Add reference line from paper (for comparison only)
        ax1.axhline(y=config.PHI_REFERENCE, color='red', linestyle='--', 
                   linewidth=2, label=f'Paper φ = {config.PHI_REFERENCE} (reference only)')
        
        ax1.set_title("Convergence Measurement (CM) Analysis\n"
                     "Find YOUR optimal φ threshold by analyzing where CM stabilizes", 
                     fontsize=14, fontweight='bold')
        ax1.set_xlabel("Episodes", fontsize=12)
        ax1.set_ylabel("CM Ratio (ΔQ_i / ΔQ_{i-1})", fontsize=12)
        ax1.legend(fontsize=10)
        ax1.grid(True, alpha=0.3)
        
        # Add text annotation
        ax1.text(0.02, 0.98, 
                'Note: The red line shows the paper\'s φ value.\n'
                'YOUR φ should be determined from this plot.',
                transform=ax1.transAxes, fontsize=9,
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        # Plot 2: Delta over episodes (log scale for better visualization)
        ax2.plot(self.delta_history, linewidth=1.5, color='green', alpha=0.7)
        ax2.set_xlabel("Episodes", fontsize=12)
        ax2.set_ylabel("Delta (ΔQ_i)", fontsize=12)
        ax2.set_title("Q-table Change per Episode", fontsize=12)
        ax2.set_yscale('log')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save plot
        save_path = os.path.join(config.OUTPUT_DIR, "cm_plot.png")
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✓ Plot saved to: {save_path}")
        
        # Also save a zoomed version for better detail
        fig, ax = plt.subplots(1, 1, figsize=(14, 6))
        ax.plot(self.cm_history, linewidth=1.5, color='blue', alpha=0.7, label='Your CM values')
        ax.axhline(y=config.PHI_REFERENCE, color='red', linestyle='--', 
                  linewidth=2, label=f'Paper φ = {config.PHI_REFERENCE} (reference)')
        ax.set_ylim([0, 1.5])  # Zoom to 0-1.5 range
        ax.set_title("CM Analysis (Zoomed: 0-1.5 range)\nLook for stable convergence region", fontsize=14)
        ax.set_xlabel("Episodes", fontsize=12)
        ax.set_ylabel("CM Ratio", fontsize=12)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Add horizontal lines at common thresholds for reference
        for threshold in [0.02, 0.04, 0.06, 0.08, 0.10]:
            ax.axhline(y=threshold, color='gray', linestyle=':', linewidth=0.5, alpha=0.5)
            ax.text(len(self.cm_history)*0.98, threshold, f'{threshold:.2f}', 
                   fontsize=8, va='center', ha='right', color='gray')
        
        zoom_path = os.path.join(config.OUTPUT_DIR, "cm_plot_zoomed.png")
        plt.savefig(zoom_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✓ Zoomed plot saved to: {zoom_path}")
        print(f"\n💡 NEXT STEP: Analyze the plots to find YOUR optimal φ value")
        print(f"   Look for the region where CM stabilizes/converges.")
        print(f"   This will be different from the paper's φ = {config.PHI_REFERENCE}")
    
    def save_data(self):
        """
        Save CM and Delta data to file for further analysis
        """
        print("\n💾 Saving data...")
        
        data_path = os.path.join(config.OUTPUT_DIR, "cm_data.npz")
        np.savez(data_path,
                 cm_history=np.array(self.cm_history),
                 delta_history=np.array(self.delta_history),
                 q_final=self.q_curr)
        
        print(f"✓ Data saved to: {data_path}")
        
        # Also save as text for easy inspection
        txt_path = os.path.join(config.OUTPUT_DIR, "cm_summary.txt")
        with open(txt_path, 'w') as f:
            f.write("="*70 + "\n")
            f.write("CM ANALYSIS SUMMARY\n")
            f.write("="*70 + "\n\n")
            
            f.write("PURPOSE: Find optimal φ threshold for YOUR route data\n")
            f.write("The paper used φ = 0.04 for Tehran/Shiraz Metro\n")
            f.write("YOUR φ will likely be different!\n\n")
            
            f.write("-"*70 + "\n")
            f.write(f"Total Episodes: {len(self.cm_history)}\n")
            f.write(f"Paper's φ (reference): {config.PHI_REFERENCE}\n\n")
            
            f.write("="*70 + "\n")
            f.write("CM STATISTICS (for your data):\n")
            f.write("="*70 + "\n")
            f.write(f"  Mean:      {np.mean(self.cm_history):.4f}\n")
            f.write(f"  Median:    {np.median(self.cm_history):.4f}\n")
            f.write(f"  Std Dev:   {np.std(self.cm_history):.4f}\n")
            f.write(f"  Min:       {np.min(self.cm_history):.4f}\n")
            f.write(f"  Max:       {np.max(self.cm_history):.4f}\n\n")
            
            # Calculate percentiles
            f.write("Percentiles:\n")
            for p in [10, 25, 50, 75, 90]:
                val = np.percentile(self.cm_history, p)
                f.write(f"  {p}th:      {val:.4f}\n")
            
            f.write("\n" + "="*70 + "\n")
            f.write("ANALYSIS (comparing to paper's φ = 0.04):\n")
            f.write("="*70 + "\n")
            
            # Compare with paper's threshold
            below_paper = np.sum(np.array(self.cm_history) < config.PHI_REFERENCE)
            above_paper = len(self.cm_history) - below_paper
            
            f.write(f"Episodes below paper's φ (0.04): {below_paper} "
                   f"({below_paper/len(self.cm_history)*100:.1f}%)\n")
            f.write(f"Episodes above paper's φ (0.04): {above_paper} "
                   f"({above_paper/len(self.cm_history)*100:.1f}%)\n\n")
            
            # Suggest potential φ values
            f.write("="*70 + "\n")
            f.write("SUGGESTED φ VALUES TO CONSIDER:\n")
            f.write("="*70 + "\n")
            f.write("(Analyze the plot to confirm!)\n\n")
            
            # Last 500 episodes (should be more stable)
            if len(self.cm_history) > 500:
                recent_cm = self.cm_history[-500:]
                f.write("Based on last 500 episodes (more stable):\n")
                f.write(f"  Mean:      {np.mean(recent_cm):.4f}\n")
                f.write(f"  Median:    {np.median(recent_cm):.4f}\n")
                f.write(f"  25th percentile: {np.percentile(recent_cm, 25):.4f}\n")
                f.write(f"  75th percentile: {np.percentile(recent_cm, 75):.4f}\n\n")
                
                f.write(f"💡 Consider φ ≈ {np.median(recent_cm):.4f} (median of last 500)\n")
                f.write(f"   or range [{np.percentile(recent_cm, 25):.4f}, "
                       f"{np.percentile(recent_cm, 75):.4f}]\n\n")
            
            f.write("="*70 + "\n")
            f.write("NEXT STEPS:\n")
            f.write("="*70 + "\n")
            f.write("1. Look at cm_plot.png and cm_plot_zoomed.png\n")
            f.write("2. Identify where CM stabilizes/converges\n")
            f.write("3. Choose YOUR φ value from the stable region\n")
            f.write("4. Use that φ in Q-SARSA implementation\n")
            f.write("5. Compare your results with the paper's φ = 0.04\n")
        
        print(f"✓ Summary saved to: {txt_path}")
        print(f"\n📋 Open {txt_path} for suggested φ values based on your data!")