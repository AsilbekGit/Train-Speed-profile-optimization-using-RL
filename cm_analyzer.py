"""
CM Analyzer - FIXED Q-table Initialization
==========================================
Problem: Q-table starts at zeros → argmax returns 0 (Brake) → train stops
Fix: Initialize with bias toward Power action
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
        
        # ============================================================
        # FIX: Initialize Q-table with bias toward POWER action
        # ============================================================
        # Actions: 0=Brake, 1=Coast, 2=Cruise, 3=Power
        # If all zeros, argmax returns 0 (Brake) → train stops!
        self.q_curr = np.zeros(self.q_shape)
        self.q_curr[:, :, 3] = 1.0  # Bias toward Power action
        self.q_curr[:, :, 2] = 0.5  # Second choice: Cruise
        
        self.q_prev = self.q_curr.copy()
        
        print(f"Q-table shape: {self.q_shape}")
        print(f"  Segments: {env.n_segments}")
        print(f"  Velocity bins: 100")
        print(f"  Actions: 4 (Brake, Coast, Cruise, Power)")
        print(f"  Initialization: Biased toward Power action")
        
        # History
        self.delta_history = []
        self.cm_history = []
        self.ln_cm_history = []
        self.success_history = []
        self.episode_max_progress = []
        
        # Fixed hyperparameters
        self.alpha = 0.1
        self.gamma = 0.95
        self.epsilon = 0.3  # Higher exploration
        
        print(f"\nHyperparameters:")
        print(f"  α (learning rate): {self.alpha}")
        print(f"  γ (discount): {self.gamma}")
        print(f"  ε (exploration): {self.epsilon}")
    
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
            self.env.reset()
            state = self.env._get_state()
            s_idx, v_idx = discretize_state(state)
            
            steps = 0
            stuck_counter = 0
            last_segment = 0
            episode_success = False
            max_segment = 0
            
            if np.random.rand() < self.epsilon:
                action = np.random.randint(4)
            else:
                action = np.argmax(self.q_curr[s_idx, v_idx])
            
            while steps < config.MAX_STEPS_PER_EPISODE:
                next_state_raw, reward, done, info = self.env.step(action)
                ns_idx, nv_idx = discretize_state(next_state_raw)
                
                steps += 1
                max_segment = max(max_segment, self.env.seg_idx)
                
                if self.env.seg_idx == last_segment:
                    stuck_counter += 1
                    if stuck_counter >= 1000:
                        done = True
                else:
                    stuck_counter = 0
                    last_segment = self.env.seg_idx
                
                if np.random.rand() < self.epsilon:
                    next_action = np.random.randint(4)
                else:
                    next_action = np.argmax(self.q_curr[ns_idx, nv_idx])
                
                if done:
                    target = reward
                else:
                    target = reward + self.gamma * self.q_curr[ns_idx, nv_idx, next_action]
                
                old_q = self.q_curr[s_idx, v_idx, action]
                self.q_curr[s_idx, v_idx, action] += self.alpha * (target - old_q)
                
                s_idx, v_idx = ns_idx, nv_idx
                action = next_action
                
                if done:
                    if self.env.seg_idx >= self.env.n_segments - 1:
                        episode_success = True
                    break
            
            if episode_success:
                success_count += 1
            self.success_history.append(episode_success)
            self.episode_max_progress.append(max_segment)
            
            if max_segment > best_progress:
                best_progress = max_segment
            
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
            
            if ep % 100 == 0 or ep <= 10:
                elapsed = time.time() - start_time
                remaining = (episodes - ep) * (elapsed / ep)
                rem_str = time.strftime("%H:%M:%S", time.gmtime(remaining))
                
                marker = "✓" if episode_success else "✗"
                ln_cm = self.ln_cm_history[-1]
                best_pct = (best_progress / self.env.n_segments) * 100
                rate = (success_count / ep) * 100
                
                print(f"{marker} Ep {ep:05d}/{episodes} | "
                      f"ln(CM): {ln_cm:7.3f} | "
                      f"Success: {success_count}/{ep} ({rate:5.1f}%) | "
                      f"Best: {best_pct:5.1f}% | "
                      f"ε: {self.epsilon:.3f} | "
                      f"ETA: {rem_str}")
            
            if ep == 1000:
                rate = (success_count / ep) * 100
                print(f"\n{'='*70}")
                print(f"📊 CHECKPOINT at Episode 1000:")
                print(f"   Overall success: {rate:.1f}%")
                print(f"   Best progress: {(best_progress/self.env.n_segments)*100:.1f}%")
                if best_progress < self.env.n_segments * 0.9:
                    print(f"\n   ⚠️  Best progress < 90% - physics may not be working")
                print(f"{'='*70}\n")
        
        self._finish(success_count, episodes, best_progress, start_time)
    
    def _finish(self, success_count, episodes, best_progress, start_time):
        print(f"\n{'='*70}")
        print(f"CM ANALYSIS COMPLETE")
        print(f"{'='*70}")
        print(f"Time: {time.strftime('%H:%M:%S', time.gmtime(time.time() - start_time))}")
        print(f"Success: {success_count}/{episodes} ({(success_count/episodes)*100:.1f}%)")
        print(f"Best progress: {best_progress}/{self.env.n_segments} ({(best_progress/self.env.n_segments)*100:.1f}%)")
        
        ln_cm_array = np.array(self.ln_cm_history)
        ln_cm_smooth = self.smooth_curve(ln_cm_array, 100)
        ln_threshold, phi_threshold, stable_start = self.detect_threshold_from_trend(ln_cm_smooth)
        
        print(f"\n{'='*70}")
        print(f"DETECTED THRESHOLD")
        print(f"{'='*70}")
        print(f"  YOUR ln(φ) = {ln_threshold:.4f}")
        print(f"  YOUR φ = {phi_threshold:.4f}")
        print(f"  Paper's φ = 0.04")
        print(f"\n  Use φ = {phi_threshold:.4f} in train_qsarsa.py and train_dqn.py")
        print(f"{'='*70}\n")
        
        self._save_plot(ln_threshold, phi_threshold, ln_cm_smooth, stable_start)
        self._save_data(ln_threshold, phi_threshold, success_count, episodes, best_progress)
    
    def _save_plot(self, ln_threshold, phi_threshold, ln_cm_smooth, stable_start):
        print("📊 Generating plots...")
        ln_cm_raw = np.array(self.ln_cm_history)
        
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
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        axes[0,0].plot(ln_cm_smooth, linewidth=2, color='blue')
        axes[0,0].axhline(y=ln_threshold, color='red', linestyle='--')
        axes[0,0].set_title('ln(CM)')
        axes[0,0].grid(True, alpha=0.3)
        
        if self.success_history:
            cumulative = np.cumsum(self.success_history) / np.arange(1, len(self.success_history)+1) * 100
            axes[0,1].plot(cumulative, color='green', linewidth=2)
        axes[0,1].set_title('Success Rate (%)')
        axes[0,1].grid(True, alpha=0.3)
        
        axes[1,0].plot(self.delta_history, alpha=0.7)
        axes[1,0].set_title('ΔQ')
        axes[1,0].grid(True, alpha=0.3)
        
        if self.episode_max_progress:
            progress = [(s/self.env.n_segments)*100 for s in self.episode_max_progress]
            axes[1,1].plot(progress, alpha=0.7, color='purple')
            axes[1,1].axhline(y=100, color='red', linestyle='--')
            axes[1,1].set_ylim([0, 105])
        axes[1,1].set_title('Max Progress (%)')
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
                 delta_history=np.array(self.delta_history),
                 success_history=np.array(self.success_history),
                 q_final=self.q_curr,
                 ln_threshold=ln_threshold,
                 phi_threshold=phi_threshold)
        
        with open(os.path.join(config.OUTPUT_DIR, "cm_summary.txt"), 'w') as f:
            f.write("="*70 + "\n")
            f.write("CM ANALYSIS SUMMARY\n")
            f.write("="*70 + "\n\n")
            f.write(f"Route: {self.env.n_segments} segments\n")
            f.write(f"Episodes: {episodes}\n\n")
            f.write(f"Success: {success_count}/{episodes} ({(success_count/episodes)*100:.1f}%)\n")
            f.write(f"Best progress: {best_progress}/{self.env.n_segments}\n\n")
            f.write(f"YOUR φ = {phi_threshold:.4f}\n")
            f.write(f"YOUR ln(φ) = {ln_threshold:.4f}\n\n")
            f.write(f"Paper's φ = 0.04\n")
        print("✓ Data saved")