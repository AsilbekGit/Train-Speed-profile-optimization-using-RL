"""
Q-SARSA Algorithm Implementation
================================
Based on Section 3.4 and Equations 26, 27, 30 from the paper
"A comprehensive study on reinforcement learning application for train speed profile optimization"

Key Innovation (Equation 30):
- When CM > φ: Use SARSA (faster convergence)
- When CM < φ: Use Q-learning (escape local optima)
"""

import numpy as np
import matplotlib.pyplot as plt
import os
import env_settings.config as config
from data.utils import discretize_state
import time

class QSARSA:
    def __init__(self, env, phi_threshold=0.10):
        """
        Initialize Q-SARSA algorithm
        
        Args:
            env: Training environment
            phi_threshold: φ threshold from CM analysis (default 0.10)
        """
        self.env = env
        self.phi = phi_threshold
        
        # Q-Table: [Segments, Velocity_Bins, Actions]
        self.q_shape = (env.n_segments, 100, 4)
        
        # Initialize with forward-motion bias
        self.q_curr = np.zeros(self.q_shape)
        self.q_curr[:, :, 3] = 5.0   # Power
        self.q_curr[:, :, 2] = 2.0   # Cruise
        self.q_curr[:, :, 1] = 0.5   # Coast
        self.q_curr[:, :, 0] = -2.0  # Brake
        
        self.q_prev = self.q_curr.copy()
        
        # History tracking
        self.delta_history = []
        self.cm_history = []
        self.reward_history = []
        self.energy_history = []
        self.time_history = []
        self.success_history = []
        
        # Hyperparameters
        self.alpha = 0.1    # Learning rate
        self.gamma = 0.95   # Discount factor
        self.epsilon = 0.10 # Exploration rate
        
        # Algorithm statistics
        self.use_sarsa_count = 0
        self.use_qlearning_count = 0
        
        print(f"\nQ-SARSA Initialized:")
        print(f"  φ threshold: {self.phi}")
        print(f"  Q-table shape: {self.q_shape}")
        print(f"  α (learning rate): {self.alpha}")
        print(f"  γ (discount): {self.gamma}")
        print(f"  ε (exploration): {self.epsilon}")
    
    def epsilon_greedy(self, s_idx, v_idx):
        """ε-greedy action selection"""
        if np.random.rand() < self.epsilon:
            return np.random.randint(4)
        else:
            return np.argmax(self.q_curr[s_idx, v_idx])
    
    def convergence_measurement(self):
        """
        Calculate CM as per Equations 28, 29 from paper
        CM(i) = ΔQ_i / ΔQ_{i-1}
        """
        if len(self.delta_history) < 2:
            return float('inf')
        
        delta_n = self.delta_history[-1]
        delta_n_minus_1 = self.delta_history[-2]
        
        if delta_n_minus_1 > 1e-9:
            return delta_n / delta_n_minus_1
        else:
            return 1.0
    
    def update(self, s_idx, v_idx, action, reward, ns_idx, nv_idx, next_action, done):
        """
        Q-SARSA hybrid update rule (Equation 30)
        
        Switches between:
        - SARSA (Eq. 27) when CM > φ: follows current policy
        - Q-learning (Eq. 26) when CM < φ: more exploration
        """
        old_q = self.q_curr[s_idx, v_idx, action]
        cm = self.convergence_measurement()
        
        if cm > self.phi:
            # SARSA update - faster convergence
            if done:
                target = reward
            else:
                target = reward + self.gamma * self.q_curr[ns_idx, nv_idx, next_action]
            self.use_sarsa_count += 1
        else:
            # Q-learning update - escape local optimum
            if done:
                target = reward
            else:
                target = reward + self.gamma * np.max(self.q_curr[ns_idx, nv_idx])
            # Increase exploration when switching to Q-learning
            self.epsilon = min(0.20, self.epsilon * 1.05)
            self.use_qlearning_count += 1
        
        # Update Q-value
        self.q_curr[s_idx, v_idx, action] += self.alpha * (target - old_q)
        
        return abs(self.q_curr[s_idx, v_idx, action] - old_q)
    
    def train(self, episodes=5000):
        """Train Q-SARSA for specified number of episodes"""
        print(f"\n{'='*70}")
        print(f"STARTING Q-SARSA TRAINING")
        print(f"{'='*70}")
        print(f"Total Episodes: {episodes}")
        print(f"φ threshold: {self.phi}")
        print(f"Goal: Minimize energy while completing route")
        print(f"{'='*70}\n")
        
        start_time = time.time()
        success_count = 0
        best_energy = float('inf')
        best_time = float('inf')
        
        for ep in range(1, episodes + 1):
            # Reset environment
            self.env.reset()
            state = self.env._get_state()
            s_idx, v_idx = discretize_state(state)
            
            # Initial action
            action = self.epsilon_greedy(s_idx, v_idx)
            
            total_reward = 0.0
            steps = 0
            
            # Episode loop
            while steps < config.MAX_STEPS_PER_EPISODE:
                # Take action
                next_state, reward, done, info = self.env.step(action)
                ns_idx, nv_idx = discretize_state(next_state)
                
                total_reward += reward
                steps += 1
                
                # Select next action (SARSA-style)
                next_action = self.epsilon_greedy(ns_idx, nv_idx)
                
                # Update Q-table
                self.update(s_idx, v_idx, action, reward, 
                           ns_idx, nv_idx, next_action, done)
                
                if done:
                    break
                
                # Emergency exit if stuck
                if self.env.v < 0.1 and steps > 100:
                    break
                
                s_idx, v_idx = ns_idx, nv_idx
                action = next_action
            
            # Calculate delta (Q-table change)
            delta_sum = np.sum(np.abs(self.q_curr - self.q_prev))
            self.delta_history.append(delta_sum)
            self.q_prev = self.q_curr.copy()
            
            # Calculate CM
            cm = self.convergence_measurement()
            self.cm_history.append(cm)
            
            # Check success
            episode_success = self.env.seg_idx >= self.env.n_segments - 2
            if episode_success:
                success_count += 1
                # Track best performance
                if info['energy'] < best_energy:
                    best_energy = info['energy']
                if info['time'] < best_time:
                    best_time = info['time']
            
            self.success_history.append(episode_success)
            self.reward_history.append(total_reward)
            self.energy_history.append(info.get('energy', 0))
            self.time_history.append(info.get('time', 0))
            
            # Decay exploration
            self.epsilon = max(0.02, self.epsilon * 0.999)
            
            # Print progress
            if ep % 100 == 0 or ep <= 10:
                elapsed = time.time() - start_time
                eta = (episodes - ep) * (elapsed / ep) if ep > 0 else 0
                eta_str = time.strftime("%H:%M:%S", time.gmtime(eta))
                
                marker = "✓" if episode_success else "✗"
                update_type = "SARSA" if cm > self.phi else "Q-Learn"
                rate = (success_count / ep) * 100
                
                print(f"{marker} Ep {ep:04d}/{episodes} | "
                      f"Success: {rate:5.1f}% | "
                      f"Energy: {info.get('energy', 0):6.1f} kWh | "
                      f"Time: {info.get('time', 0):6.0f}s | "
                      f"CM: {cm:.4f} | "
                      f"Mode: {update_type:8s} | "
                      f"ε: {self.epsilon:.3f} | "
                      f"ETA: {eta_str}")
        
        elapsed_total = time.time() - start_time
        print(f"\n{'='*70}")
        print(f"Q-SARSA TRAINING COMPLETE")
        print(f"{'='*70}")
        print(f"Total time: {time.strftime('%H:%M:%S', time.gmtime(elapsed_total))}")
        print(f"Final success rate: {(success_count/episodes)*100:.1f}%")
        print(f"SARSA updates: {self.use_sarsa_count} ({self.use_sarsa_count/(self.use_sarsa_count+self.use_qlearning_count+1)*100:.1f}%)")
        print(f"Q-learning updates: {self.use_qlearning_count}")
        print(f"\nBest Performance:")
        print(f"  Best energy: {best_energy:.1f} kWh")
        print(f"  Best time: {best_time:.0f} s")
        
        self.save_results()
        return self.q_curr
    
    def save_results(self):
        """Save training results and plots"""
        print("\n💾 Saving Q-SARSA results...")
        
        output_dir = os.path.join(config.OUTPUT_DIR, "qsarsa")
        os.makedirs(output_dir, exist_ok=True)
        
        # Save Q-table and history
        np.savez(os.path.join(output_dir, "qsarsa_data.npz"),
                 q_table=self.q_curr,
                 cm_history=np.array(self.cm_history),
                 delta_history=np.array(self.delta_history),
                 reward_history=np.array(self.reward_history),
                 energy_history=np.array(self.energy_history),
                 time_history=np.array(self.time_history),
                 success_history=np.array(self.success_history))
        
        # Plot results
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # Success rate
        if self.success_history:
            cumulative = np.cumsum(self.success_history) / np.arange(1, len(self.success_history)+1) * 100
            axes[0, 0].plot(cumulative, linewidth=2, color='green')
        axes[0, 0].set_title('Success Rate (%)')
        axes[0, 0].set_xlabel('Episode')
        axes[0, 0].set_ylabel('Success %')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Energy over time
        successful_episodes = [i for i, s in enumerate(self.success_history) if s]
        if successful_episodes:
            successful_energy = [self.energy_history[i] for i in successful_episodes]
            axes[0, 1].plot(successful_episodes, successful_energy, 'b.', alpha=0.5, markersize=2)
            # Moving average
            if len(successful_energy) > 50:
                window = 50
                ma = np.convolve(successful_energy, np.ones(window)/window, mode='valid')
                axes[0, 1].plot(successful_episodes[window-1:], ma, 'r-', linewidth=2, label='Moving Avg')
                axes[0, 1].legend()
        axes[0, 1].set_title('Energy (Successful Episodes)')
        axes[0, 1].set_xlabel('Episode')
        axes[0, 1].set_ylabel('Energy (kWh)')
        axes[0, 1].grid(True, alpha=0.3)
        
        # CM over time
        axes[1, 0].plot(self.cm_history, linewidth=1, alpha=0.7)
        axes[1, 0].axhline(y=self.phi, color='red', linestyle='--', label=f'φ={self.phi}')
        axes[1, 0].set_title('Convergence Measurement')
        axes[1, 0].set_xlabel('Episode')
        axes[1, 0].set_ylabel('CM')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # Travel time
        if successful_episodes:
            successful_time = [self.time_history[i] for i in successful_episodes]
            axes[1, 1].plot(successful_episodes, successful_time, 'g.', alpha=0.5, markersize=2)
            if len(successful_time) > 50:
                window = 50
                ma = np.convolve(successful_time, np.ones(window)/window, mode='valid')
                axes[1, 1].plot(successful_episodes[window-1:], ma, 'r-', linewidth=2, label='Moving Avg')
                axes[1, 1].legend()
        axes[1, 1].set_title('Travel Time (Successful Episodes)')
        axes[1, 1].set_xlabel('Episode')
        axes[1, 1].set_ylabel('Time (s)')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "qsarsa_training.png"), dpi=300)
        plt.close()
        
        print(f"✓ Results saved to: {output_dir}/")
    
    def get_optimal_action(self, segment, velocity):
        """Get best action for given state"""
        s_idx = int(segment)
        v_idx = int(velocity / config.VEL_BIN_SIZE)
        v_idx = max(0, min(v_idx, 99))
        return np.argmax(self.q_curr[s_idx, v_idx])
    
    def generate_speed_profile(self):
        """Generate optimal speed profile using learned policy"""
        print("\n📊 Generating optimal speed profile...")
        
        self.env.reset()
        
        segments = []
        velocities = []
        actions = []
        energies = []
        
        steps = 0
        while steps < config.MAX_STEPS_PER_EPISODE:
            state = self.env._get_state()
            s_idx, v_idx = discretize_state(state)
            
            # Get best action (no exploration)
            action = np.argmax(self.q_curr[s_idx, v_idx])
            
            segments.append(self.env.seg_idx)
            velocities.append(self.env.v)
            actions.append(action)
            energies.append(self.env.energy_kwh)
            
            _, _, done, _ = self.env.step(action)
            steps += 1
            
            if done or self.env.v < 0.1:
                break
        
        # Save profile
        output_dir = os.path.join(config.OUTPUT_DIR, "qsarsa")
        np.savez(os.path.join(output_dir, "speed_profile.npz"),
                 segments=np.array(segments),
                 velocities=np.array(velocities),
                 actions=np.array(actions),
                 energies=np.array(energies))
        
        # Plot profile
        fig, axes = plt.subplots(3, 1, figsize=(14, 10))
        
        # Speed profile
        axes[0].plot(segments, np.array(velocities) * 3.6, 'b-', linewidth=1)
        axes[0].set_ylabel('Speed (km/h)')
        axes[0].set_title('Optimal Speed Profile (Q-SARSA)')
        axes[0].grid(True, alpha=0.3)
        
        # Actions
        action_names = ['Brake', 'Coast', 'Cruise', 'Power']
        colors = ['red', 'yellow', 'blue', 'green']
        for i, name in enumerate(action_names):
            mask = np.array(actions) == i
            if np.any(mask):
                axes[1].scatter(np.array(segments)[mask], [i]*np.sum(mask), 
                              c=colors[i], s=1, label=name, alpha=0.5)
        axes[1].set_ylabel('Action')
        axes[1].set_yticks([0, 1, 2, 3])
        axes[1].set_yticklabels(action_names)
        axes[1].legend(loc='right')
        axes[1].grid(True, alpha=0.3)
        
        # Energy
        axes[2].plot(segments, energies, 'g-', linewidth=1)
        axes[2].set_xlabel('Segment')
        axes[2].set_ylabel('Cumulative Energy (kWh)')
        axes[2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "speed_profile.png"), dpi=300)
        plt.close()
        
        print(f"✓ Speed profile saved")
        print(f"  Final energy: {energies[-1]:.1f} kWh")
        print(f"  Final segment: {segments[-1]}/{self.env.n_segments}")
        
        return segments, velocities, actions, energies