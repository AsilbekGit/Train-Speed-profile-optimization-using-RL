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
        
        # SMART INITIALIZATION: Bias Q-table toward Power action
        # This helps agent discover successful paths faster
        self.q_curr = np.zeros(self.q_shape)
        self.q_prev = np.zeros(self.q_shape)
        
        # Initialize Power action (action 3) with small positive values
        # This encourages trying Power more in early episodes
        self.q_curr[:, :, 3] = 10.0  # Power action gets initial boost
        self.q_curr[:, :, 2] = 5.0   # Cruise gets smaller boost
        self.q_curr[:, :, 1] = 2.0   # Coast gets tiny boost
        # Brake (action 0) stays at 0
        
        print(f"Q-table initialized with smart bias:")
        print(f"  Power action (3): +10.0 initial value")
        print(f"  Cruise action (2): +5.0 initial value")
        print(f"  Coast action (1): +2.0 initial value")
        print(f"  Brake action (0): 0.0 initial value")
        
        # History
        self.delta_history = []
        self.cm_history = []
        self.ln_cm_history = []
        self.success_history = []
        self.episode_max_progress = []  # Track furthest reached
        
        # Hyperparams
        self.alpha = 0.1
        self.gamma = 0.95
        self.epsilon = 0.25  # Start with HIGHER epsilon (was 0.15)
        self.epsilon_decay = 0.9995  # Slow decay
        self.epsilon_min = 0.10
        
        self.debug_mode = config.DEBUG_MODE
        self.print_every_step = config.PRINT_EVERY_STEP
        
    def smooth_curve(self, data, window_size=50):
        if len(data) < window_size:
            window_size = max(len(data) // 10, 1)
        return uniform_filter1d(data, size=window_size, mode='nearest')
    
    def detect_threshold_from_trend(self, ln_cm_smooth):
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
        print(f"CM ANALYSIS - SARSA with Smart Initialization")
        print(f"{'='*70}")
        print(f"Total Episodes: {episodes}")
        print(f"Initial epsilon: {self.epsilon}")
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
            
            # Initial Action (ε-greedy with higher initial epsilon)
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
                
                # Relaxed stuck detection (velocity-based, not position-based)
                if self.env.v < 0.5:  # Nearly stopped
                    stuck_counter += 1
                    if stuck_counter >= 1000:  # Very high threshold
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
            
            # Track success and progress
            if episode_success:
                success_count += 1
            self.success_history.append(episode_success)
            self.episode_max_progress.append(max_segment_this_episode)
            
            if max_segment_this_episode > best_progress:
                best_progress = max_segment_this_episode
            
            # Calculate ΔQi
            delta_i = np.sum(np.abs(self.q_curr - self.q_prev))
            self.delta_history.append(delta_i)
            self.q_prev = self.q_curr.copy()
            
            # Calculate CM
            if len(self.delta_history) >= 2:
                delta_n = self.delta_history[-1]
                delta_n_minus_1 = self.delta_history[-2]
                
                if delta_n_minus_1 > 1e-9:
                    cm_ratio = delta_n / delta_n_minus_1
                else:
                    cm_ratio = 1.0
                
                self.cm_history.append(cm_ratio)
                
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
                
                print(f"{success_marker} Ep {ep:05d}/{episodes} | "
                      f"Progress: {progress_pct:5.1f}% | "
                      f"Best: {best_pct:5.1f}% | "
                      f"Success: {success_count}/{ep} ({success_count/ep*100:5.1f}%) | "
                      f"ε: {self.epsilon:.3f} | "
                      f"ETA: {rem_str}")
            
            # Early check at episode 100
            if ep == 100:
                print(f"\n{'='*70}")
                if success_count == 0:
                    print(f"⚠️  WARNING: Still 0% success after 100 episodes")
                    print(f"   Best progress: {best_pct:.1f}% of route")
                    print(f"   Continuing to see if agent learns...")
                else:
                    print(f"✓ Good! {success_count} successes in first 100 episodes")
                    print(f"   Agent is learning!")
                print(f"{'='*70}\n")
                
            # Check at episode 500
            if ep == 500 and success_count == 0:
                print(f"\n{'='*70}")
                print(f"⚠️  STILL 0% after 500 episodes!")
                print(f"   Best progress: {best_pct:.1f}%")
                print(f"   This route might be too difficult for current setup.")
                print(f"{'='*70}\n")
                
                response = input("Continue training? (y/n): ")
                if response.lower() != 'y':
                    print("Stopping early.")
                    episodes = ep  # Update episodes count for summary
                    break
        
        print(f"\n{'='*70}")
        print(f"CM ANALYSIS COMPLETE")
        print(f"{'='*70}")
        elapsed_total = time.time() - start_time
        print(f"Total time: {time.strftime('%H:%M:%S', time.gmtime(elapsed_total))}")
        print(f"Success rate: {success_count}/{episodes} ({success_count/episodes*100:.1f}%)")
        print(f"Best progress: {best_progress}/{self.env.n_segments} ({best_pct:.1f}%)")
        
        if success_count == 0:
            print(f"\n⚠️  0% SUCCESS - CM analysis may not be meaningful")
            print(f"   But agent did reach {best_pct:.1f}% of route")
            print(f"   This shows learning is happening, just not complete")
        
        # Process data
        print(f"\n🔍 Processing CM data...")
        ln_cm_array = np.array(self.ln_cm_history)
        ln_cm_smooth = self.smooth_curve(ln_cm_array, window_size=100)
        
        ln_threshold, phi_threshold, stable_start = self.detect_threshold_from_trend(ln_cm_smooth)
        
        print(f"\n{'='*70}")
        print(f"DETECTED THRESHOLD")
        print(f"{'='*70}")
        print(f"  ln(φ) = {ln_threshold:.4f}")
        print(f"  φ = {phi_threshold:.4f}")
        
        if success_count > 0:
            print(f"  ✓ Use this value in Q-SARSA training")
        else:
            print(f"  ⚠️  With 0% success, consider:")
            print(f"     - Shorter route for initial training")
            print(f"     - Higher initial velocity")
            print(f"     - Curriculum learning (start easy)")
        print(f"{'='*70}\n")
        
        self.save_plot(ln_threshold, phi_threshold, ln_cm_smooth, stable_start)
        self.save_data(ln_threshold, phi_threshold, success_count, episodes, best_progress)
    
    def save_plot(self, ln_threshold, phi_threshold, ln_cm_smooth, stable_start):
        print("\n📊 Generating plots...")
        
        ln_cm_raw = np.array(self.ln_cm_history)
        
        # Main Figure 5 style plot
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
        
        # Progress plot
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))
        
        # Top: ln(CM)
        ax1.plot(ln_cm_smooth, linewidth=2, color='blue', label='Smoothed', alpha=0.8)
        ax1.plot(ln_cm_raw, linewidth=0.3, color='gray', label='Raw', alpha=0.3)
        ax1.axhline(y=ln_threshold, color='red', linestyle='--', linewidth=2)
        ax1.set_xlabel('Episode')
        ax1.set_ylabel('ln(ΔQi/ΔQi-1)')
        ax1.set_title('ln(CM) Analysis')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Bottom: Progress over time
        progress_pct = [(seg / self.env.n_segments) * 100 for seg in self.episode_max_progress]
        ax2.plot(progress_pct, color='green', linewidth=1, alpha=0.7)
        ax2.axhline(y=100, color='red', linestyle='--', label='Complete')
        ax2.set_xlabel('Episode')
        ax2.set_ylabel('Max Progress (%)')
        ax2.set_title('Learning Progress (% of route reached)')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        ax2.set_ylim([0, 105])
        
        plt.tight_layout()
        detail_path = os.path.join(config.OUTPUT_DIR, "cm_analysis_detailed.png")
        plt.savefig(detail_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✓ Detailed analysis saved: {detail_path}")
    
    def save_data(self, ln_threshold, phi_threshold, success_count, total_episodes, best_progress):
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
        txt_path = os.path.join(config.OUTPUT_DIR, "cm_summary.txt")
        with open(txt_path, 'w') as f:
            f.write("="*70 + "\n")
            f.write("CM ANALYSIS SUMMARY\n")
            f.write("="*70 + "\n\n")
            
            f.write(f"Episodes: {total_episodes}\n")
            f.write(f"Success Rate: {success_count}/{total_episodes} ({success_count/total_episodes*100:.1f}%)\n")
            f.write(f"Best Progress: {best_progress}/{self.env.n_segments} segments ({best_progress/self.env.n_segments*100:.1f}%)\n\n")
            
            f.write("DETECTED THRESHOLD:\n")
            f.write(f"  ln(φ) = {ln_threshold:.4f}\n")
            f.write(f"  φ = {phi_threshold:.4f}\n\n")
            
            if success_count > 0:
                f.write(f"✓ Use φ = {phi_threshold:.4f} in Q-SARSA training\n")
            else:
                f.write("⚠️  0% SUCCESS RATE\n")
                f.write(f"Agent reached {best_progress/self.env.n_segments*100:.1f}% of route\n")
                f.write("Consider: shorter route, curriculum learning, or higher initial velocity\n")
        
        print(f"✓ Summary saved: {txt_path}")