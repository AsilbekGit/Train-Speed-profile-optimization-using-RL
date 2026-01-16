"""
CM Analyzer - STABLE VERSION with Learning Rate Decay
=====================================================
Key fix: Learning rate DECAYS from 0.15 → 0.01 over training
This prevents Q-table divergence!
"""

import numpy as np
import matplotlib.pyplot as plt
import os
import env_settings.config as config
from data.utils import discretize_state
import time
from scipy.ndimage import uniform_filter1d

class CMAnalyzer:
    def __init__(self, env):
        self.env = env
        self.q_shape = (env.n_segments, 100, 4)
        
        # Initialize Q-table
        self.q_curr = np.zeros(self.q_shape)
        self.q_prev = np.zeros(self.q_shape)
        
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
        self.recent_successes = []
        self.q_stats_history = []
        
        # ============================================================
        # KEY FIX: Learning rate with DECAY
        # ============================================================
        self.alpha_start = 0.15     # Start higher
        self.alpha_end = 0.01       # End lower
        self.alpha = self.alpha_start
        
        self.gamma = 0.95
        
        # Exploration schedule
        self.epsilon_start = 0.30
        self.epsilon_end = 0.05
        self.epsilon = self.epsilon_start
        
        # Q-value bounds - prevent explosion
        self.q_min = -500.0
        self.q_max = 500.0
        
        # Visit counting
        self.visit_count = np.zeros(self.q_shape, dtype=np.int32)
        
        print(f"\nSTABLE Configuration:")
        print(f"  α: {self.alpha_start} → {self.alpha_end} (LINEAR DECAY)")
        print(f"  ε: {self.epsilon_start} → {self.epsilon_end}")
        print(f"  Q-bounds: [{self.q_min}, {self.q_max}]")
        print(f"Learning rate: α = {self.alpha_start} (will decay to {self.alpha_end})")
        
    def get_learning_rate(self, episode, total_episodes):
        """Linear decay of learning rate"""
        progress = episode / total_episodes
        return self.alpha_start - (self.alpha_start - self.alpha_end) * progress
    
    def get_epsilon(self, episode, total_episodes):
        """Exponential decay of exploration"""
        progress = episode / total_episodes
        epsilon = self.epsilon_start * (self.epsilon_end / self.epsilon_start) ** progress
        return max(self.epsilon_end, epsilon)
    
    def smooth_curve(self, data, window_size=100):
        if len(data) < window_size:
            window_size = max(len(data) // 10, 1)
        return uniform_filter1d(data, size=window_size, mode='nearest')
    
    def detect_threshold_from_trend(self, ln_cm_smooth):
        stable_start = int(len(ln_cm_smooth) * 0.7)
        
        for i in range(len(ln_cm_smooth) - 200):
            window = ln_cm_smooth[i:i+200]
            if np.std(window) < 0.3:
                stable_start = i
                break
        
        stable_region = ln_cm_smooth[stable_start:]
        ln_threshold = np.median(stable_region) if len(stable_region) > 0 else np.median(ln_cm_smooth)
        phi_threshold = np.exp(ln_threshold)
        
        return ln_threshold, phi_threshold, stable_start
    
    def run(self, episodes=25000):
        print(f"\n{'='*70}")
        print(f"CM ANALYSIS - STABLE VERSION (Learning Rate Decay)")
        print(f"{'='*70}")
        print(f"Paper's methodology: Section 3.4, Figure 5")
        print(f"Paper's result: φ = 0.04 for Tehran/Shiraz Metro")
        print(f"YOUR φ will be different based on your route!")
        print(f"{'='*70}")
        print(f"Total Episodes: {episodes}")
        print(f"Route: {self.env.n_segments} segments ({self.env.n_segments * config.DX / 1000:.1f} km)")
        print(f"Learning rate: α = {self.alpha_start} → {self.alpha_end} (DECAYING)")
        print(f"{'='*70}\n")
        
        start_time = time.time()
        success_count = 0
        best_progress = 0
        
        for ep in range(1, episodes + 1):
            # KEY: Update learning rate and epsilon
            self.alpha = self.get_learning_rate(ep, episodes)
            self.epsilon = self.get_epsilon(ep, episodes)
            
            # Reset
            self.env.reset()
            state = self.env._get_state()
            s_idx, v_idx = discretize_state(state)
            
            steps = 0
            stuck_counter = 0
            last_segment = 0
            episode_success = False
            max_segment = 0
            
            # Initial action
            if np.random.rand() < self.epsilon:
                action = np.random.randint(4)
            else:
                action = np.argmax(self.q_curr[s_idx, v_idx])
            
            # Episode loop
            while steps < config.MAX_STEPS_PER_EPISODE:
                next_state_raw, reward, done, info = self.env.step(action)
                ns_idx, nv_idx = discretize_state(next_state_raw)
                
                steps += 1
                max_segment = max(max_segment, self.env.seg_idx)
                
                # Stuck detection
                if self.env.seg_idx == last_segment:
                    stuck_counter += 1
                    if stuck_counter >= 1000:
                        done = True
                else:
                    stuck_counter = 0
                    last_segment = self.env.seg_idx
                
                # Next action (SARSA)
                if np.random.rand() < self.epsilon:
                    next_action = np.random.randint(4)
                else:
                    next_action = np.argmax(self.q_curr[ns_idx, nv_idx])
                
                # SARSA update with clipping
                if done:
                    target = reward
                else:
                    target = reward + self.gamma * self.q_curr[ns_idx, nv_idx, next_action]
                
                old_q = self.q_curr[s_idx, v_idx, action]
                td_error = target - old_q
                
                # Visit-based adjustment
                self.visit_count[s_idx, v_idx, action] += 1
                visits = self.visit_count[s_idx, v_idx, action]
                effective_alpha = self.alpha / (1.0 + 0.01 * visits)
                
                # Update and clip
                new_q = old_q + effective_alpha * td_error
                self.q_curr[s_idx, v_idx, action] = np.clip(new_q, self.q_min, self.q_max)
                
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
            self.episode_max_progress.append(max_segment)
            
            self.recent_successes.append(1 if episode_success else 0)
            if len(self.recent_successes) > 500:
                self.recent_successes.pop(0)
            
            if max_segment > best_progress:
                best_progress = max_segment
            
            # CM calculation
            delta_i = np.sum(np.abs(self.q_curr - self.q_prev))
            self.delta_history.append(delta_i)
            self.q_prev = self.q_curr.copy()
            
            if len(self.delta_history) >= 2:
                delta_n = self.delta_history[-1]
                delta_n_minus_1 = self.delta_history[-2]
                cm_ratio = delta_n / delta_n_minus_1 if delta_n_minus_1 > 1e-9 else 1.0
                self.cm_history.append(cm_ratio)
                ln_cm = np.log(cm_ratio) if cm_ratio > 1e-9 else -10.0
                self.ln_cm_history.append(ln_cm)
            else:
                self.cm_history.append(1.0)
                self.ln_cm_history.append(0.0)
            
            # Q stats
            self.q_stats_history.append((np.mean(self.q_curr), np.max(self.q_curr), np.min(self.q_curr)))
            
            # Print progress
            if ep % 100 == 0 or ep <= 10:
                elapsed = time.time() - start_time
                remaining = (episodes - ep) * (elapsed / ep)
                rem_str = time.strftime("%H:%M:%S", time.gmtime(remaining))
                
                marker = "✓" if episode_success else "✗"
                ln_cm = self.ln_cm_history[-1]
                best_pct = (best_progress / self.env.n_segments) * 100
                rate = (success_count / ep) * 100
                recent = (sum(self.recent_successes) / len(self.recent_successes)) * 100 if self.recent_successes else 0
                
                print(f"{marker} Ep {ep:05d}/{episodes} | "
                      f"ln(CM): {ln_cm:7.3f} | "
                      f"Success: {success_count}/{ep} ({rate:5.1f}%) | "
                      f"Recent: {recent:5.1f}% | "
                      f"Best: {best_pct:5.1f}% | "
                      f"α: {self.alpha:.4f} | "
                      f"ε: {self.epsilon:.3f} | "
                      f"ETA: {rem_str}")
            
            # Checkpoint
            if ep == 1000:
                rate = (success_count / ep) * 100
                recent = (sum(self.recent_successes) / len(self.recent_successes)) * 100
                print(f"\n{'='*70}")
                print(f"📊 CHECKPOINT at Episode 1000:")
                print(f"   Overall success: {rate:.1f}%")
                print(f"   Recent success (last 500): {recent:.1f}%")
                print(f"   Best progress: {(best_progress/self.env.n_segments)*100:.1f}%")
                print(f"   Current α: {self.alpha:.4f} (decaying)")
                
                if recent < rate * 0.3 and rate > 2:
                    print(f"\n   ⚠️  Recent rate dropping! But α is decaying, should stabilize.")
                    print(f"   Continue and watch if recent rate recovers.")
                print(f"{'='*70}\n")
        
        # Finish
        self._finish(success_count, episodes, best_progress, start_time)
    
    def _finish(self, success_count, episodes, best_progress, start_time):
        print(f"\n{'='*70}")
        print(f"CM ANALYSIS COMPLETE")
        print(f"{'='*70}")
        print(f"Time: {time.strftime('%H:%M:%S', time.gmtime(time.time() - start_time))}")
        print(f"Success: {success_count}/{episodes} ({(success_count/episodes)*100:.1f}%)")
        print(f"Best progress: {best_progress}/{self.env.n_segments} ({(best_progress/self.env.n_segments)*100:.1f}%)")
        
        # Process
        ln_cm_array = np.array(self.ln_cm_history)
        ln_cm_smooth = self.smooth_curve(ln_cm_array, 100)
        ln_threshold, phi_threshold, stable_start = self.detect_threshold_from_trend(ln_cm_smooth)
        
        print(f"\n{'='*70}")
        print(f"DETECTED THRESHOLD")
        print(f"{'='*70}")
        print(f"  YOUR ln(φ) = {ln_threshold:.4f}")
        print(f"  YOUR φ = {phi_threshold:.4f}")
        print(f"  Paper's φ = 0.04")
        
        # Check recent rate
        recent = self.recent_successes[-500:] if len(self.recent_successes) >= 500 else self.recent_successes
        final_recent = sum(recent) / len(recent) * 100 if recent else 0
        overall = (success_count / episodes) * 100
        
        print(f"\n  Final Success Analysis:")
        print(f"    Overall: {overall:.1f}%")
        print(f"    Final recent (last 500): {final_recent:.1f}%")
        
        if final_recent >= overall * 0.5:
            print(f"    ✓ Learning stabilized - recent rate held up")
        else:
            print(f"    ⚠️  Recent rate dropped - learning partially diverged")
        
        print(f"\n  Use φ = {phi_threshold:.4f} in train_qsarsa.py and train_dqn.py")
        print(f"{'='*70}\n")
        
        self._save_plot(ln_threshold, phi_threshold, ln_cm_smooth, stable_start)
        self._save_data(ln_threshold, phi_threshold, success_count, episodes, best_progress)
    
    def _save_plot(self, ln_threshold, phi_threshold, ln_cm_smooth, stable_start):
        print("📊 Generating plots...")
        
        ln_cm_raw = np.array(self.ln_cm_history)
        
        # Figure 5 style
        fig, ax = plt.subplots(figsize=(14, 8))
        
        ax.plot(ln_cm_smooth, linewidth=2, color='#4682B4', alpha=0.9, label='Smoothed ln(CM)')
        ax.plot(ln_cm_raw, linewidth=0.5, color='#4682B4', alpha=0.15, label='Raw')
        
        ax.axhline(y=ln_threshold, color='red', linestyle='--', linewidth=2.5,
                   label=f'YOUR: ln(φ) = {ln_threshold:.3f} (φ = {phi_threshold:.4f})')
        ax.axhline(y=-3.21, color='green', linestyle=':', linewidth=1.5,
                   label="Paper: ln(φ) = -3.21 (φ = 0.04)")
        
        ax.annotate('Starts failing\nto local\noptimum', 
                    xy=(len(ln_cm_smooth)*0.7, ln_threshold),
                    xytext=(len(ln_cm_smooth)*0.75, ln_threshold + 1),
                    fontsize=12, fontweight='bold',
                    bbox=dict(boxstyle='round', facecolor='lightyellow', edgecolor='red'),
                    arrowprops=dict(arrowstyle='->', color='red', lw=2))
        
        ax.set_xlabel('Iteration', fontsize=14, fontweight='bold')
        ax.set_ylabel('Ln(ΔQi/ΔQi-1)', fontsize=14, fontweight='bold')
        ax.set_title(f'Figure 5: CM Analysis\nYOUR φ = {phi_threshold:.4f}', fontsize=16, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=10)
        ax.set_ylim([min(min(ln_cm_smooth)-0.5, -4), max(max(ln_cm_smooth)+0.5, 2)])
        
        plt.tight_layout()
        plt.savefig(os.path.join(config.OUTPUT_DIR, "cm_plot_figure5.png"), dpi=300)
        plt.close()
        print(f"✓ Figure 5 saved")
        
        # Detailed plot
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        axes[0,0].plot(ln_cm_smooth, linewidth=2, color='blue')
        axes[0,0].axhline(y=ln_threshold, color='red', linestyle='--')
        axes[0,0].set_title('ln(CM)')
        axes[0,0].grid(True, alpha=0.3)
        
        if self.success_history:
            cumulative = np.cumsum(self.success_history) / np.arange(1, len(self.success_history)+1) * 100
            axes[0,1].plot(cumulative, color='green', linewidth=2, label='Cumulative')
            window = 500
            recent = [sum(self.success_history[max(0,i-window):i+1])/(min(i+1,window))*100 
                     for i in range(len(self.success_history))]
            axes[0,1].plot(recent, color='blue', alpha=0.7, label='Recent')
            axes[0,1].legend()
        axes[0,1].set_title('Success Rate')
        axes[0,1].grid(True, alpha=0.3)
        
        if self.q_stats_history:
            means = [h[0] for h in self.q_stats_history]
            maxs = [h[1] for h in self.q_stats_history]
            mins = [h[2] for h in self.q_stats_history]
            axes[1,0].plot(means, label='Mean')
            axes[1,0].plot(maxs, alpha=0.5, label='Max')
            axes[1,0].plot(mins, alpha=0.5, label='Min')
            axes[1,0].axhline(y=self.q_max, color='red', linestyle=':', alpha=0.3)
            axes[1,0].axhline(y=self.q_min, color='green', linestyle=':', alpha=0.3)
            axes[1,0].legend()
        axes[1,0].set_title('Q-values (should stay bounded)')
        axes[1,0].grid(True, alpha=0.3)
        
        if self.episode_max_progress:
            progress = [(s/self.env.n_segments)*100 for s in self.episode_max_progress]
            axes[1,1].plot(progress, alpha=0.7, color='purple')
            axes[1,1].axhline(y=100, color='red', linestyle='--')
            axes[1,1].set_ylim([0, 105])
        axes[1,1].set_title('Progress per Episode')
        axes[1,1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(config.OUTPUT_DIR, "cm_analysis_detailed.png"), dpi=300)
        plt.close()
        print(f"✓ Detailed plot saved")
    
    def _save_data(self, ln_threshold, phi_threshold, success_count, episodes, best_progress):
        print("💾 Saving data...")
        
        np.savez(os.path.join(config.OUTPUT_DIR, "cm_data.npz"),
                 ln_cm_history=np.array(self.ln_cm_history),
                 cm_history=np.array(self.cm_history),
                 success_history=np.array(self.success_history),
                 q_final=self.q_curr,
                 ln_threshold=ln_threshold,
                 phi_threshold=phi_threshold)
        
        with open(os.path.join(config.OUTPUT_DIR, "cm_summary.txt"), 'w') as f:
            f.write("="*70 + "\n")
            f.write("CM ANALYSIS SUMMARY\n")
            f.write("="*70 + "\n\n")
            f.write(f"Route: {self.env.n_segments} segments\n")
            f.write(f"Episodes: {episodes}\n")
            f.write(f"Learning rate: {self.alpha_start} → {self.alpha_end} (decayed)\n\n")
            f.write(f"Success: {success_count}/{episodes} ({(success_count/episodes)*100:.1f}%)\n\n")
            f.write(f"YOUR φ = {phi_threshold:.4f}\n")
            f.write(f"YOUR ln(φ) = {ln_threshold:.4f}\n\n")
            f.write(f"Paper's φ = 0.04\n\n")
            f.write("Use this φ in train_qsarsa.py and train_dqn.py\n")
        
        print("✓ Data saved")