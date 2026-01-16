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
        self.q_shape = (env.n_segments, 100, 4)
        
        # LIGHT initialization (not too strong!)
        self.q_curr = np.zeros(self.q_shape)
        self.q_prev = np.zeros(self.q_shape)
        
        # Small bias to guide initial exploration (not overwhelming)
        self.q_curr[:, :, 3] = 2.0   # Power: slight boost (was 10)
        self.q_curr[:, :, 2] = 1.0   # Cruise: tiny boost (was 5)
        # Coast and Brake stay at 0
        
        print(f"Q-table initialized with light bias:")
        print(f"  Power action: +2.0 (light guidance)")
        print(f"  Cruise action: +1.0 (tiny boost)")
        
        # History
        self.delta_history = []
        self.cm_history = []
        self.ln_cm_history = []
        self.success_history = []
        self.episode_max_progress = []
        
        # NORMAL hyperparameters (not too aggressive!)
        self.alpha = 0.1
        self.gamma = 0.95
        self.epsilon = 0.15  # Standard ε-greedy (was 0.25)
        self.epsilon_decay = 0.999  # Slower than before
        self.epsilon_min = 0.05
        
        self.debug_mode = config.DEBUG_MODE
        self.print_every_step = config.PRINT_EVERY_STEP
        
    def smooth_curve(self, data, window_size=100):
        """Apply moving average smoothing"""
        if len(data) < window_size:
            window_size = max(len(data) // 10, 1)
        return uniform_filter1d(data, size=window_size, mode='nearest')
    
    def detect_threshold_from_trend(self, ln_cm_smooth):
        """Detect threshold where ln(CM) stabilizes"""
        # Find where curve goes consistently negative
        for i in range(len(ln_cm_smooth) - 100):
            if np.mean(ln_cm_smooth[i:i+100]) < -0.1:
                stable_start = i
                break
        else:
            stable_start = int(len(ln_cm_smooth) * 0.7)
        
        stable_region = ln_cm_smooth[stable_start:]
        ln_threshold = np.median(stable_region)
        phi_threshold = np.exp(ln_threshold)
        
        return ln_threshold, phi_threshold, stable_start
    
    def run(self, episodes=25000):
        print(f"\n{'='*70}")
        print(f"CM ANALYSIS - SARSA with Balanced Configuration")
        print(f"{'='*70}")
        print(f"Target: 10-40% success rate for valid CM analysis")
        print(f"Total Episodes: {episodes}")
        print(f"Route: {self.env.n_segments * config.DX / 1000:.1f} km")
        print(f"{'='*70}\n")
        
        start_time = time.time()
        success_count = 0
        best_progress = 0
        
        for ep in range(1, episodes + 1):
            # Reset
            self.env.reset()
            state = self.env._get_state()
            s_idx, v_idx = discretize_state(state)
            
            total_reward = 0.0
            steps = 0
            stuck_counter = 0
            episode_success = False
            max_segment_this_episode = 0
            
            # ε-greedy action selection
            if np.random.rand() < self.epsilon:
                action = np.random.randint(4)
            else:
                action = np.argmax(self.q_curr[s_idx, v_idx])
            
            # Episode loop
            while steps < config.MAX_STEPS_PER_EPISODE:
                # Step
                next_state_raw, reward, done, info = self.env.step(action)
                ns_idx, nv_idx = discretize_state(next_state_raw)
                
                total_reward += reward
                steps += 1
                max_segment_this_episode = max(max_segment_this_episode, self.env.seg_idx)
                
                # Stuck detection (velocity-based, relaxed)
                if self.env.v < 0.5:
                    stuck_counter += 1
                    if stuck_counter >= 500:  # Moderate threshold
                        done = True
                else:
                    stuck_counter = 0
                
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
                
                # Move to next
                s_idx, v_idx = ns_idx, nv_idx
                action = next_action
                
                if done:
                    if self.env.seg_idx >= self.env.n_segments - 1:
                        episode_success = True
                    break
            
            # Track
            if episode_success:
                success_count += 1
            self.success_history.append(episode_success)
            self.episode_max_progress.append(max_segment_this_episode)
            
            if max_segment_this_episode > best_progress:
                best_progress = max_segment_this_episode
            
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
                
                print(f"{success_marker} Ep {ep:05d}/{episodes} | "
                      f"ln(CM): {current_ln_cm:7.3f} | "
                      f"Success: {success_count}/{ep} ({success_rate:5.1f}%) | "
                      f"Best: {best_pct:5.1f}% | "
                      f"ε: {self.epsilon:.3f} | "
                      f"ETA: {rem_str}")
            
            # Check success rate milestones
            if ep == 100:
                success_rate = (success_count / ep) * 100
                print(f"\n{'='*70}")
                if success_rate < 5:
                    print(f"⚠️  {success_rate:.1f}% success - on the low side")
                    print(f"   Best progress: {best_pct:.1f}%")
                    print(f"   Continuing...")
                elif success_rate > 60:
                    print(f"⚠️  {success_rate:.1f}% success - problem might be too easy")
                    print(f"   But continuing for CM analysis...")
                else:
                    print(f"✓ {success_rate:.1f}% success rate - good for learning!")
                print(f"{'='*70}\n")
        
        print(f"\n{'='*70}")
        print(f"CM ANALYSIS COMPLETE")
        print(f"{'='*70}")
        elapsed_total = time.time() - start_time
        print(f"Total time: {time.strftime('%H:%M:%S', time.gmtime(elapsed_total))}")
        
        success_rate_final = (success_count / episodes) * 100
        best_pct = (best_progress / self.env.n_segments) * 100
        
        print(f"Success rate: {success_count}/{episodes} ({success_rate_final:.1f}%)")
        print(f"Best progress: {best_progress}/{self.env.n_segments} ({best_pct:.1f}%)")
        
        # Evaluate success rate
        if success_rate_final > 80:
            print(f"\n⚠️  SUCCESS RATE TOO HIGH ({success_rate_final:.1f}%)")
            print(f"   Problem is too easy - CM analysis may not be meaningful")
            print(f"   Consider: reducing rewards, removing smart init")
        elif success_rate_final < 5:
            print(f"\n⚠️  SUCCESS RATE VERY LOW ({success_rate_final:.1f}%)")
            print(f"   Problem is very hard - but agent learned ({best_pct:.1f}% progress)")
        else:
            print(f"\n✓ Success rate ({success_rate_final:.1f}%) is good for CM analysis!")
        
        # Process CM data
        print(f"\n🔍 Processing CM data...")
        ln_cm_array = np.array(self.ln_cm_history)
        ln_cm_smooth = self.smooth_curve(ln_cm_array, window_size=100)
        
        ln_threshold, phi_threshold, stable_start = self.detect_threshold_from_trend(ln_cm_smooth)
        
        print(f"\n{'='*70}")
        print(f"DETECTED THRESHOLD")
        print(f"{'='*70}")
        print(f"  ln(φ) = {ln_threshold:.4f}")
        print(f"  φ = {phi_threshold:.4f}")
        print(f"  Paper's φ = 0.04 (for comparison)")
        
        # Validate threshold
        if phi_threshold > 0.8:
            print(f"\n  ⚠️  φ = {phi_threshold:.4f} is close to 1.0")
            print(f"     This means Q-table not converging properly")
            print(f"     Likely cause: Problem too easy (success rate {success_rate_final:.1f}%)")
        elif phi_threshold < 0.001:
            print(f"\n  ⚠️  φ = {phi_threshold:.4f} is very small")
            print(f"     Q-table might be over-converging")
        else:
            print(f"\n  ✓ φ = {phi_threshold:.4f} looks reasonable!")
            print(f"    Use this in Q-SARSA training")
        print(f"{'='*70}\n")
        
        self.save_plot(ln_threshold, phi_threshold, ln_cm_smooth, stable_start)
        self.save_data(ln_threshold, phi_threshold, success_count, episodes, best_progress)
    
    def save_plot(self, ln_threshold, phi_threshold, ln_cm_smooth, stable_start):
        """Generate Figure 5 style plot"""
        print("\n📊 Generating plots...")
        
        ln_cm_raw = np.array(self.ln_cm_history)
        
        # Main plot
        fig, ax = plt.subplots(1, 1, figsize=(14, 8))
        
        ax.plot(ln_cm_smooth, linewidth=2.0, color='#4682B4', alpha=0.9, 
                label='Smoothed ln(CM)', zorder=3)
        ax.plot(ln_cm_raw, linewidth=0.5, color='#4682B4', alpha=0.2, 
                label='Raw ln(CM)', zorder=1)
        
        ax.axhline(y=ln_threshold, color='red', linestyle='--', linewidth=2.5, 
                   label=f'YOUR threshold: ln(φ) = {ln_threshold:.3f} (φ = {phi_threshold:.4f})',
                   zorder=4)
        
        ax.axvline(x=stable_start, color='orange', linestyle=':', linewidth=1.5,
                   label=f'Stable region (ep {stable_start})', zorder=2)
        
        annotation_x = len(ln_cm_smooth) * 0.7
        ax.annotate('Starts failing\nto local\noptimum', 
                    xy=(annotation_x, ln_threshold), 
                    xytext=(annotation_x * 1.05, ln_threshold + 1.0),
                    fontsize=14, fontweight='bold',
                    bbox=dict(boxstyle='round,pad=0.8', facecolor='#4682B4', 
                             alpha=0.7, edgecolor='black', linewidth=2),
                    arrowprops=dict(arrowstyle='->', color='red', lw=2.5),
                    color='white', ha='left', zorder=5)
        
        ax.set_xlabel('Iteration', fontsize=14, fontweight='bold')
        ax.set_ylabel('Ln(ΔQi/ΔQi-1)', fontsize=14, fontweight='bold')
        ax.set_title(f'CM Analysis - YOUR Threshold: φ = {phi_threshold:.4f}', 
                     fontsize=16, fontweight='bold', pad=20)
        
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=10, loc='upper right')
        
        y_min = max(min(ln_cm_smooth) - 0.5, -4.5)
        y_max = max(max(ln_cm_smooth) + 0.5, 1.5)
        ax.set_ylim([y_min, y_max])
        
        plt.tight_layout()
        save_path = os.path.join(config.OUTPUT_DIR, "cm_plot_figure5.png")
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✓ Figure 5 plot saved: {save_path}")
        
        # Detailed plot
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(14, 12))
        
        # Top: ln(CM)
        ax1.plot(ln_cm_smooth, linewidth=2, color='blue', label='Smoothed')
        ax1.plot(ln_cm_raw, linewidth=0.3, color='gray', label='Raw', alpha=0.3)
        ax1.axhline(y=ln_threshold, color='red', linestyle='--', linewidth=2)
        ax1.set_xlabel('Episode')
        ax1.set_ylabel('ln(ΔQi/ΔQi-1)')
        ax1.set_title('ln(CM) - Should DECREASE from ~0 to negative')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Middle: Success rate over time
        success_cumulative = np.cumsum(self.success_history) / np.arange(1, len(self.success_history) + 1)
        ax2.plot(success_cumulative * 100, color='green', linewidth=2)
        ax2.axhline(y=20, color='orange', linestyle='--', label='20% (good)')
        ax2.axhline(y=50, color='red', linestyle='--', label='50% (high)')
        ax2.set_xlabel('Episode')
        ax2.set_ylabel('Success Rate (%)')
        ax2.set_title('Cumulative Success Rate (10-40% ideal for CM analysis)')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        ax2.set_ylim([0, 100])
        
        # Bottom: Progress
        progress_pct = [(seg / self.env.n_segments) * 100 for seg in self.episode_max_progress]
        ax3.plot(progress_pct, color='purple', linewidth=1, alpha=0.7)
        ax3.axhline(y=100, color='red', linestyle='--', label='Complete')
        ax3.set_xlabel('Episode')
        ax3.set_ylabel('Max Progress (%)')
        ax3.set_title('Learning Progress')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        ax3.set_ylim([0, 105])
        
        plt.tight_layout()
        detail_path = os.path.join(config.OUTPUT_DIR, "cm_analysis_detailed.png")
        plt.savefig(detail_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✓ Detailed analysis saved: {detail_path}")
    
    def save_data(self, ln_threshold, phi_threshold, success_count, total_episodes, best_progress):
        """Save data and summary"""
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
        
        # Summary
        success_rate = (success_count / total_episodes) * 100
        txt_path = os.path.join(config.OUTPUT_DIR, "cm_summary.txt")
        with open(txt_path, 'w') as f:
            f.write("="*70 + "\n")
            f.write("CM ANALYSIS SUMMARY\n")
            f.write("="*70 + "\n\n")
            
            f.write(f"Episodes: {total_episodes}\n")
            f.write(f"Success Rate: {success_count}/{total_episodes} ({success_rate:.1f}%)\n")
            f.write(f"Best Progress: {best_progress}/{self.env.n_segments} segments\n\n")
            
            f.write("DETECTED THRESHOLD:\n")
            f.write(f"  ln(φ) = {ln_threshold:.4f}\n")
            f.write(f"  φ = {phi_threshold:.4f}\n\n")
            
            f.write("COMPARISON WITH PAPER:\n")
            f.write(f"  Paper's φ = 0.04 (Tehran/Shiraz Metro)\n")
            f.write(f"  YOUR φ = {phi_threshold:.4f}\n\n")
            
            if success_rate > 80:
                f.write("⚠️  WARNING: Success rate very high ({:.1f}%)\n".format(success_rate))
                f.write("   Problem might be too easy\n")
                f.write("   φ close to 1.0 means Q-table not converging\n\n")
            elif success_rate < 5:
                f.write("⚠️  WARNING: Success rate very low ({:.1f}%)\n".format(success_rate))
                f.write("   Problem is very hard\n\n")
            else:
                f.write("✓ Success rate ({:.1f}%) is reasonable\n\n".format(success_rate))
            
            if phi_threshold > 0.8:
                f.write("⚠️  φ = {:.4f} close to 1.0 - Q-table not converging\n".format(phi_threshold))
                f.write("   Do NOT use this for Q-SARSA training\n")
                f.write("   Reduce rewards and try again\n")
            else:
                f.write("✓ Use φ = {:.4f} in Q-SARSA training\n".format(phi_threshold))
        
        print(f"✓ Summary saved: {txt_path}")