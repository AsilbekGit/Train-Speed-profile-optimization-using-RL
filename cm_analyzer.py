import numpy as np
import matplotlib.pyplot as plt
import os
import config
from utils import discretize_state
import time
from scipy.ndimage import uniform_filter1d

class CMAnalyzer:
    """
    CM Analysis based on Section 3.4, Equations 28-29, Figure 5 from paper:
    "A comprehensive study on reinforcement learning application for train speed profile optimization"
    
    Goal: Find YOUR φ threshold by running SARSA and tracking Q-table convergence.
    
    Paper's Result: φ = 0.04 for Tehran/Shiraz Metro
    YOUR φ: Will be different based on your route!
    """
    
    def __init__(self, env):
        self.env = env
        self.q_shape = (env.n_segments, 100, 4)
        
        # Initialize Q-table with zeros (no bias - pure learning)
        self.q_curr = np.zeros(self.q_shape)
        self.q_prev = np.zeros(self.q_shape)
        
        # Light initial bias to help exploration (not too strong!)
        # Actions: 0=Brake, 1=Coast, 2=Cruise, 3=Power
        self.q_curr[:, :, 3] = 1.0   # Power: slight boost
        self.q_curr[:, :, 2] = 0.5   # Cruise: tiny boost
        
        print(f"Q-table shape: {self.q_shape}")
        print(f"  Segments: {env.n_segments}")
        print(f"  Velocity bins: 100")
        print(f"  Actions: 4 (Brake, Coast, Cruise, Power)")
        
        # History
        self.delta_history = []
        self.cm_history = []
        self.ln_cm_history = []
        self.success_history = []
        self.episode_max_progress = []
        
        # ============================================================
        # KEY FIX: Learning rate between 0.01 and 0.1
        # ============================================================
        # 0.1 = too fast, diverges
        # 0.01 = too slow, doesn't converge in reasonable time
        # 0.05 = balanced for 75km route
        self.alpha = 0.05  # BALANCED learning rate
        
        self.gamma = 0.95  # Discount factor
        
        # Exploration parameters
        self.epsilon = 0.20       # Start with more exploration
        self.epsilon_decay = 0.9995  # Slow decay
        self.epsilon_min = 0.05
        
        # Additional tracking
        self.recent_successes = []  # Track recent success rate
        
        self.debug_mode = config.DEBUG_MODE
        self.print_every_step = config.PRINT_EVERY_STEP
        
    def smooth_curve(self, data, window_size=100):
        """Apply moving average smoothing"""
        if len(data) < window_size:
            window_size = max(len(data) // 10, 1)
        return uniform_filter1d(data, size=window_size, mode='nearest')
    
    def detect_threshold_from_trend(self, ln_cm_smooth):
        """
        Detect threshold where ln(CM) stabilizes (Figure 5 methodology)
        
        Paper shows: ln(CM) starts at ~0, decreases to ~-3.21 where φ = 0.04
        """
        # Find stable region (where curve stops decreasing rapidly)
        stable_start = int(len(ln_cm_smooth) * 0.7)  # Last 30% is usually stable
        
        # Look for where derivative becomes small
        for i in range(len(ln_cm_smooth) - 200):
            window = ln_cm_smooth[i:i+200]
            if np.std(window) < 0.3:  # Low variance = stable
                stable_start = i
                break
        
        # Get threshold from stable region
        stable_region = ln_cm_smooth[stable_start:]
        if len(stable_region) > 0:
            ln_threshold = np.median(stable_region)
        else:
            ln_threshold = np.median(ln_cm_smooth)
        
        phi_threshold = np.exp(ln_threshold)
        
        return ln_threshold, phi_threshold, stable_start
    
    def run(self, episodes=25000):
        """
        Run CM Analysis with SARSA to find φ threshold
        """
        print(f"\n{'='*70}")
        print(f"CM ANALYSIS - Finding YOUR φ Threshold")
        print(f"{'='*70}")
        print(f"Paper's methodology: Section 3.4, Figure 5")
        print(f"Paper's result: φ = 0.04 for Tehran/Shiraz Metro")
        print(f"YOUR φ will be different based on your route!")
        print(f"{'='*70}")
        print(f"Total Episodes: {episodes}")
        print(f"Route: {self.env.n_segments} segments ({self.env.n_segments * config.DX / 1000:.1f} km)")
        print(f"Learning rate: α = {self.alpha}")
        print(f"{'='*70}\n")
        
        start_time = time.time()
        success_count = 0
        best_progress = 0
        
        for ep in range(1, episodes + 1):
            # Reset environment
            self.env.reset()
            state = self.env._get_state()
            s_idx, v_idx = discretize_state(state)
            
            total_reward = 0.0
            steps = 0
            stuck_counter = 0
            last_segment = 0
            episode_success = False
            max_segment_this_episode = 0
            
            # Initial action (ε-greedy)
            if np.random.rand() < self.epsilon:
                action = np.random.randint(4)
            else:
                action = np.argmax(self.q_curr[s_idx, v_idx])
            
            # Episode loop
            while steps < config.MAX_STEPS_PER_EPISODE:
                # Take action
                next_state_raw, reward, done, info = self.env.step(action)
                ns_idx, nv_idx = discretize_state(next_state_raw)
                
                total_reward += reward
                steps += 1
                max_segment_this_episode = max(max_segment_this_episode, self.env.seg_idx)
                
                # Stuck detection (position-based)
                if self.env.seg_idx == last_segment:
                    stuck_counter += 1
                    if stuck_counter >= 800:  # Allow time for recovery
                        done = True
                else:
                    stuck_counter = 0
                    last_segment = self.env.seg_idx
                
                # Select next action (ε-greedy for SARSA)
                if np.random.rand() < self.epsilon:
                    next_action = np.random.randint(4)
                else:
                    next_action = np.argmax(self.q_curr[ns_idx, nv_idx])
                
                # SARSA Update (Equation 27 from paper)
                if done:
                    target = reward
                else:
                    target = reward + self.gamma * self.q_curr[ns_idx, nv_idx, next_action]
                
                old_q = self.q_curr[s_idx, v_idx, action]
                td_error = target - old_q
                self.q_curr[s_idx, v_idx, action] += self.alpha * td_error
                
                # Move to next state
                s_idx, v_idx = ns_idx, nv_idx
                action = next_action
                
                if done:
                    # Check if successfully completed
                    if self.env.seg_idx >= self.env.n_segments - 1:
                        episode_success = True
                    break
            
            # Track success
            if episode_success:
                success_count += 1
            self.success_history.append(episode_success)
            self.episode_max_progress.append(max_segment_this_episode)
            
            # Track recent success rate (last 500 episodes)
            self.recent_successes.append(1 if episode_success else 0)
            if len(self.recent_successes) > 500:
                self.recent_successes.pop(0)
            
            if max_segment_this_episode > best_progress:
                best_progress = max_segment_this_episode
            
            # Calculate ΔQ (Equation 28 from paper)
            delta_i = np.sum(np.abs(self.q_curr - self.q_prev))
            self.delta_history.append(delta_i)
            self.q_prev = self.q_curr.copy()
            
            # Calculate CM = ΔQ_i / ΔQ_{i-1} (Equation 29)
            if len(self.delta_history) >= 2:
                delta_n = self.delta_history[-1]
                delta_n_minus_1 = self.delta_history[-2]
                
                if delta_n_minus_1 > 1e-9:
                    cm_ratio = delta_n / delta_n_minus_1
                else:
                    cm_ratio = 1.0
                
                self.cm_history.append(cm_ratio)
                
                # ln(CM) for Figure 5 style plot
                if cm_ratio > 1e-9:
                    ln_cm = np.log(cm_ratio)
                else:
                    ln_cm = -10.0
                
                self.ln_cm_history.append(ln_cm)
            else:
                self.cm_history.append(1.0)
                self.ln_cm_history.append(0.0)
            
            # Decay epsilon
            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
            
            # Print progress
            if ep % 100 == 0 or ep <= 10:
                elapsed = time.time() - start_time
                avg_time = elapsed / ep
                remaining = (episodes - ep) * avg_time
                rem_str = time.strftime("%H:%M:%S", time.gmtime(remaining))
                
                success_marker = "✓" if episode_success else "✗"
                current_ln_cm = self.ln_cm_history[-1] if self.ln_cm_history else 0.0
                progress_pct = (max_segment_this_episode / self.env.n_segments) * 100
                best_pct = (best_progress / self.env.n_segments) * 100
                success_rate = (success_count / ep) * 100
                recent_rate = (sum(self.recent_successes) / len(self.recent_successes)) * 100 if self.recent_successes else 0
                
                print(f"{success_marker} Ep {ep:05d}/{episodes} | "
                      f"ln(CM): {current_ln_cm:7.3f} | "
                      f"Success: {success_count}/{ep} ({success_rate:5.1f}%) | "
                      f"Recent: {recent_rate:5.1f}% | "
                      f"Best: {best_pct:5.1f}% | "
                      f"ε: {self.epsilon:.3f} | "
                      f"ETA: {rem_str}")
            
            # Early check at episode 1000
            if ep == 1000:
                success_rate = (success_count / ep) * 100
                recent_rate = (sum(self.recent_successes) / len(self.recent_successes)) * 100
                print(f"\n{'='*70}")
                print(f"📊 CHECKPOINT at Episode 1000:")
                print(f"   Overall success: {success_rate:.1f}%")
                print(f"   Recent success (last 500): {recent_rate:.1f}%")
                print(f"   Best progress: {(best_progress/self.env.n_segments)*100:.1f}%")
                
                if recent_rate < success_rate and success_rate > 1:
                    print(f"\n   ⚠️  Recent rate dropping! Q-table may be diverging.")
                    print(f"   Consider: lower α or more episodes")
                elif success_count == 0:
                    print(f"\n   ⚠️  No successes yet. Problem is challenging.")
                    print(f"   This is OK - continue training.")
                else:
                    print(f"\n   ✓ Training appears stable. Continue.")
                print(f"{'='*70}\n")
        
        # Final analysis
        print(f"\n{'='*70}")
        print(f"CM ANALYSIS COMPLETE")
        print(f"{'='*70}")
        elapsed_total = time.time() - start_time
        print(f"Total time: {time.strftime('%H:%M:%S', time.gmtime(elapsed_total))}")
        
        success_rate_final = (success_count / episodes) * 100
        best_pct = (best_progress / self.env.n_segments) * 100
        
        print(f"Success rate: {success_count}/{episodes} ({success_rate_final:.1f}%)")
        print(f"Best progress: {best_progress}/{self.env.n_segments} ({best_pct:.1f}%)")
        
        # Process CM data
        print(f"\n🔍 Processing CM data...")
        ln_cm_array = np.array(self.ln_cm_history)
        ln_cm_smooth = self.smooth_curve(ln_cm_array, window_size=100)
        
        ln_threshold, phi_threshold, stable_start = self.detect_threshold_from_trend(ln_cm_smooth)
        
        print(f"\n{'='*70}")
        print(f"DETECTED THRESHOLD (Figure 5 Analysis)")
        print(f"{'='*70}")
        print(f"  YOUR ln(φ) = {ln_threshold:.4f}")
        print(f"  YOUR φ = {phi_threshold:.4f}")
        print(f"  Paper's φ = 0.04 (for reference)")
        
        # Proper interpretation
        self._interpret_results(phi_threshold, success_rate_final, best_pct, ln_cm_smooth)
        
        # Save plots and data
        self.save_plot(ln_threshold, phi_threshold, ln_cm_smooth, stable_start)
        self.save_data(ln_threshold, phi_threshold, success_count, episodes, best_progress)
    
    def _interpret_results(self, phi, success_rate, best_progress_pct, ln_cm_smooth):
        """Properly interpret the results"""
        
        print(f"\n{'='*70}")
        print(f"INTERPRETATION")
        print(f"{'='*70}")
        
        # Check ln(CM) trend
        early_ln_cm = np.mean(ln_cm_smooth[:1000]) if len(ln_cm_smooth) > 1000 else np.mean(ln_cm_smooth[:len(ln_cm_smooth)//3])
        late_ln_cm = np.mean(ln_cm_smooth[-1000:]) if len(ln_cm_smooth) > 1000 else np.mean(ln_cm_smooth[-len(ln_cm_smooth)//3:])
        ln_cm_decreasing = late_ln_cm < early_ln_cm - 0.3
        
        print(f"\n  ln(CM) Trend:")
        print(f"    Early (first 1000 eps): {early_ln_cm:.3f}")
        print(f"    Late (last 1000 eps): {late_ln_cm:.3f}")
        print(f"    Decreasing: {'✓ Yes' if ln_cm_decreasing else '✗ No'}")
        
        if phi > 0.8:
            print(f"\n  ⚠️  φ = {phi:.4f} is close to 1.0")
            print(f"     This means: Q-table NOT converging properly")
            
            if success_rate < 5:
                print(f"\n     Cause: Problem is TOO HARD (success rate {success_rate:.1f}%)")
                print(f"     Agent reaches {best_progress_pct:.1f}% but can't complete")
                print(f"\n     Recommendations:")
                print(f"     1. Increase rewards for near-completion")
                print(f"     2. Check terminal segment for issues")
                print(f"     3. Run more episodes (try 50000)")
                print(f"     4. Adjust learning rate (try α = 0.03)")
            elif success_rate > 80:
                print(f"\n     Cause: Problem is TOO EASY (success rate {success_rate:.1f}%)")
                print(f"     Agent completes without learning optimal policy")
                print(f"\n     Recommendations:")
                print(f"     1. Reduce success bonus")
                print(f"     2. Increase energy penalty")
            else:
                print(f"\n     Cause: Q-table oscillating, not converging")
                print(f"\n     Recommendations:")
                print(f"     1. Lower learning rate (try α = 0.02)")
                print(f"     2. Run more episodes")
                
        elif phi > 0.3:
            print(f"\n  ⚠️  φ = {phi:.4f} is moderate")
            print(f"     Q-table converging slowly")
            print(f"\n     Recommendations:")
            print(f"     1. Run more episodes (try 50000)")
            print(f"     2. This φ is usable but not optimal")
            
        elif phi > 0.01:
            print(f"\n  ✓ φ = {phi:.4f} is in good range!")
            print(f"     Q-table is converging properly")
            print(f"\n     You can use this φ in Q-SARSA training")
            
        else:
            print(f"\n  ⚠️  φ = {phi:.4f} is very small")
            print(f"     Q-table might be over-converging (stuck in local optimum)")
        
        print(f"{'='*70}\n")
    
    def save_plot(self, ln_threshold, phi_threshold, ln_cm_smooth, stable_start):
        """Generate Figure 5 style plot from the paper"""
        print("\n📊 Generating plots...")
        
        ln_cm_raw = np.array(self.ln_cm_history)
        
        # ============================================================
        # FIGURE 5 STYLE PLOT
        # ============================================================
        fig, ax = plt.subplots(1, 1, figsize=(14, 8))
        
        # Plot smoothed and raw data
        ax.plot(ln_cm_smooth, linewidth=2.0, color='#4682B4', alpha=0.9, 
                label='Smoothed ln(CM)', zorder=3)
        ax.plot(ln_cm_raw, linewidth=0.5, color='#4682B4', alpha=0.15, 
                label='Raw ln(CM)', zorder=1)
        
        # YOUR threshold line
        ax.axhline(y=ln_threshold, color='red', linestyle='--', linewidth=2.5, 
                   label=f'YOUR threshold: ln(φ) = {ln_threshold:.3f} (φ = {phi_threshold:.4f})',
                   zorder=4)
        
        # Paper's reference line
        ax.axhline(y=-3.21, color='green', linestyle=':', linewidth=1.5, 
                   label=f"Paper's threshold: ln(φ) = -3.21 (φ = 0.04)",
                   zorder=2)
        
        # Stable region marker
        ax.axvline(x=stable_start, color='orange', linestyle=':', linewidth=1.5,
                   label=f'Stable region (ep {stable_start})', zorder=2)
        
        # Annotation like Figure 5
        annotation_x = len(ln_cm_smooth) * 0.7
        annotation_y = ln_threshold + 0.5 if ln_threshold < 0 else ln_threshold + 1.0
        ax.annotate('Starts failing\nto local\noptimum', 
                    xy=(annotation_x, ln_threshold), 
                    xytext=(annotation_x * 1.05, annotation_y),
                    fontsize=12, fontweight='bold',
                    bbox=dict(boxstyle='round,pad=0.5', facecolor='lightyellow', 
                             alpha=0.9, edgecolor='red', linewidth=2),
                    arrowprops=dict(arrowstyle='->', color='red', lw=2.0),
                    color='black', ha='left', zorder=5)
        
        ax.set_xlabel('Iteration', fontsize=14, fontweight='bold')
        ax.set_ylabel('Ln(ΔQi/ΔQi-1)', fontsize=14, fontweight='bold')
        ax.set_title(f'Figure 5 Style: Convergence Measurement Analysis\n'
                     f'YOUR φ = {phi_threshold:.4f} vs Paper φ = 0.04', 
                     fontsize=16, fontweight='bold', pad=20)
        
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=10, loc='upper right')
        
        # Set appropriate y-limits
        y_min = min(min(ln_cm_smooth) - 0.5, -4.0)
        y_max = max(max(ln_cm_smooth) + 0.5, 2.0)
        ax.set_ylim([y_min, y_max])
        
        plt.tight_layout()
        save_path = os.path.join(config.OUTPUT_DIR, "cm_plot_figure5.png")
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✓ Figure 5 plot saved: {save_path}")
        
        # ============================================================
        # DETAILED ANALYSIS PLOT
        # ============================================================
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(14, 12))
        
        # Top: ln(CM) over time
        ax1.plot(ln_cm_smooth, linewidth=2, color='blue', label='Smoothed ln(CM)')
        ax1.plot(ln_cm_raw, linewidth=0.3, color='gray', label='Raw', alpha=0.3)
        ax1.axhline(y=ln_threshold, color='red', linestyle='--', linewidth=2, label=f'YOUR φ={phi_threshold:.4f}')
        ax1.axhline(y=-3.21, color='green', linestyle=':', linewidth=1.5, label="Paper's φ=0.04")
        ax1.set_xlabel('Episode')
        ax1.set_ylabel('ln(ΔQi/ΔQi-1)')
        ax1.set_title('ln(CM) - Should DECREASE from ~0 to negative (like Figure 5)')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Middle: Success rate over time
        success_cumulative = np.cumsum(self.success_history) / np.arange(1, len(self.success_history) + 1)
        ax2.plot(success_cumulative * 100, color='green', linewidth=2, label='Cumulative success rate')
        
        # Also plot recent success rate
        recent_rates = []
        window = 500
        for i in range(len(self.success_history)):
            start_idx = max(0, i - window)
            rate = sum(self.success_history[start_idx:i+1]) / (i - start_idx + 1) * 100
            recent_rates.append(rate)
        ax2.plot(recent_rates, color='blue', linewidth=1, alpha=0.7, label='Recent rate (500 ep window)')
        
        ax2.set_xlabel('Episode')
        ax2.set_ylabel('Success Rate (%)')
        ax2.set_title('Success Rate (Recent rate should not drop below overall)')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        ax2.set_ylim([0, max(100, max(recent_rates) + 10) if recent_rates else 100])
        
        # Bottom: Progress over time
        progress_pct = [(seg / self.env.n_segments) * 100 for seg in self.episode_max_progress]
        ax3.plot(progress_pct, color='purple', linewidth=1, alpha=0.7)
        ax3.axhline(y=100, color='red', linestyle='--', label='Complete (100%)')
        ax3.set_xlabel('Episode')
        ax3.set_ylabel('Max Progress (%)')
        ax3.set_title('Learning Progress (Max segment reached per episode)')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        ax3.set_ylim([0, 105])
        
        plt.tight_layout()
        detail_path = os.path.join(config.OUTPUT_DIR, "cm_analysis_detailed.png")
        plt.savefig(detail_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✓ Detailed analysis saved: {detail_path}")
    
    def save_data(self, ln_threshold, phi_threshold, success_count, total_episodes, best_progress):
        """Save analysis data"""
        print("\n💾 Saving data...")
        
        data_path = os.path.join(config.OUTPUT_DIR, "cm_data.npz")
        np.savez(data_path,
                 ln_cm_history=np.array(self.ln_cm_history),
                 cm_history=np.array(self.cm_history),
                 delta_history=np.array(self.delta_history),
                 success_history=np.array(self.success_history),
                 episode_max_progress=np.array(self.episode_max_progress),
                 q_final=self.q_curr,
                 ln_threshold=ln_threshold,
                 phi_threshold=phi_threshold)
        
        print(f"✓ Data saved: {data_path}")
        
        # Summary file
        success_rate = (success_count / total_episodes) * 100
        txt_path = os.path.join(config.OUTPUT_DIR, "cm_summary.txt")
        with open(txt_path, 'w') as f:
            f.write("="*70 + "\n")
            f.write("CM ANALYSIS SUMMARY\n")
            f.write("="*70 + "\n\n")
            
            f.write(f"Route: {self.env.n_segments} segments ({self.env.n_segments * config.DX / 1000:.1f} km)\n")
            f.write(f"Episodes: {total_episodes}\n")
            f.write(f"Learning rate: α = {self.alpha}\n\n")
            
            f.write(f"Success Rate: {success_count}/{total_episodes} ({success_rate:.1f}%)\n")
            f.write(f"Best Progress: {best_progress}/{self.env.n_segments} segments ({(best_progress/self.env.n_segments)*100:.1f}%)\n\n")
            
            f.write("YOUR THRESHOLD:\n")
            f.write(f"  ln(φ) = {ln_threshold:.4f}\n")
            f.write(f"  φ = {phi_threshold:.4f}\n\n")
            
            f.write("PAPER'S THRESHOLD (for reference):\n")
            f.write(f"  ln(φ) = -3.21\n")
            f.write(f"  φ = 0.04\n\n")
            
            # Recommendations
            f.write("RECOMMENDATION:\n")
            if 0.01 < phi_threshold < 0.3 and success_rate > 5:
                f.write(f"✓ Your φ = {phi_threshold:.4f} looks good!\n")
                f.write(f"  Use this value in Q-SARSA and Deep-Q training.\n")
            elif phi_threshold > 0.5:
                f.write(f"⚠️  φ = {phi_threshold:.4f} indicates incomplete convergence\n")
                if success_rate < 5:
                    f.write(f"   Problem is challenging (success rate {success_rate:.1f}%)\n")
                    f.write(f"   Try: Run more episodes (50000+) or adjust rewards\n")
                else:
                    f.write(f"   Try: Lower learning rate or more episodes\n")
            else:
                f.write(f"  φ = {phi_threshold:.4f} - results need review\n")
        
        print(f"✓ Summary saved: {txt_path}")