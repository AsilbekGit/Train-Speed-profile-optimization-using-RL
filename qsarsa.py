"""
Q-SARSA Algorithm Implementation
Based on Section 3.4 and Equations 26, 27, 30 from the paper
"A comprehensive study on reinforcement learning application for train speed profile optimization"
by Sandidzadeh & Havaei (2023)
"""

import numpy as np
import matplotlib.pyplot as plt
import os
import config
from utils import discretize_state
import time

class QSARSA:
    def __init__(self, env, phi_threshold):
        """
        Initialize Q-SARSA algorithm
        
        Args:
            env: Training environment
            phi_threshold: φ threshold determined from CM analysis
                         (use YOUR φ value from CM analysis results!)
        """
        self.env = env
        self.phi = phi_threshold
        
        # Q-Table dimensions: [Segments, Velocity_Bins, Actions]
        self.q_shape = (env.n_segments, 100, 4)
        self.q_curr = np.zeros(self.q_shape)
        self.q_prev = np.zeros(self.q_shape)
        
        # History tracking
        self.delta_history = []
        self.cm_history = []
        self.reward_history = []
        self.energy_history = []
        self.time_history = []
        
        # Hyperparameters (from paper)
        self.alpha = 0.1    # Learning rate (η)
        self.gamma = 0.95   # Discount factor (γ)
        self.epsilon = 0.15 # Initial exploration rate
        
        # Algorithm state
        self.use_qlearning_count = 0
        self.use_sarsa_count = 0
        
        print(f"\nQ-SARSA Initialized:")
        print(f"  φ threshold: {self.phi}")
        print(f"  Q-table shape: {self.q_shape}")
        print(f"  α (learning rate): {self.alpha}")
        print(f"  γ (discount): {self.gamma}")
        print(f"  ε (exploration): {self.epsilon}")
    
    def epsilon_greedy(self, s_idx, v_idx):
        """
        ε-greedy action selection
        """
        if np.random.rand() < self.epsilon:
            return np.random.randint(4)
        else:
            return np.argmax(self.q_curr[s_idx, v_idx])
    
    def convergence_measurement(self):
        """
        Calculate CM as per Equations 28, 29 from paper
        cm(i) = ΔQ_i / ΔQ_{i-1}
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
        Q-SARSA hybrid update rule (Equation 30 from paper)
        
        Switches between:
        - SARSA update (Eq. 27) when cm(i) > φ
        - Q-learning update (Eq. 26) when cm(i) < φ
        """
        old_q = self.q_curr[s_idx, v_idx, action]
        
        # Calculate convergence measurement
        cm = self.convergence_measurement()
        
        # Hybrid update logic (Equation 30)
        if cm > self.phi:
            # Use SARSA update (Equation 27)
            # Faster convergence, follows current policy
            if done:
                target = reward
            else:
                target = reward + self.gamma * self.q_curr[ns_idx, nv_idx, next_action]
            self.use_sarsa_count += 1
            
        else:
            # Use Q-learning update (Equation 26)
            # More exploration, escapes local optimum
            if done:
                target = reward
            else:
                target = reward + self.gamma * np.max(self.q_curr[ns_idx, nv_idx])
            
            # Increase epsilon to explore more (as per paper)
            self.epsilon = min(0.3, self.epsilon * 1.1)
            self.use_qlearning_count += 1
        
        # Update Q-value (Equation 32)
        self.q_curr[s_idx, v_idx, action] += self.alpha * (target - old_q)
        
        return abs(self.q_curr[s_idx, v_idx, action] - old_q)
    
    def train(self, episodes=1000):
        """
        Train Q-SARSA for specified number of episodes
        """
        print(f"\n{'='*70}")
        print(f"STARTING Q-SARSA TRAINING")
        print(f"{'='*70}")
        print(f"Total Episodes: {episodes}")
        print(f"φ threshold: {self.phi}")
        print(f"{'='*70}\n")
        
        start_time = time.time()
        
        for ep in range(1, episodes + 1):
            # Reset environment
            self.env.reset()
            state = self.env._get_state()
            s_idx, v_idx = discretize_state(state)
            
            # Initial action (ε-greedy)
            action = self.epsilon_greedy(s_idx, v_idx)
            
            total_reward = 0.0
            steps = 0
            episode_deltas = []
            
            # Episode loop
            while steps < config.MAX_STEPS_PER_EPISODE:
                # Take action
                next_state_raw, reward, done, info = self.env.step(action)
                ns_idx, nv_idx = discretize_state(next_state_raw)
                
                total_reward += reward
                steps += 1
                
                # Select next action (SARSA-style)
                next_action = self.epsilon_greedy(ns_idx, nv_idx)
                
                # Update Q-table using Q-SARSA
                delta = self.update(s_idx, v_idx, action, reward, 
                                   ns_idx, nv_idx, next_action, done)
                episode_deltas.append(delta)
                
                if done:
                    break
                
                # Move to next state
                s_idx, v_idx = ns_idx, nv_idx
                action = next_action
            
            # Calculate Delta (sum of Q-table changes)
            delta_sum = np.sum(np.abs(self.q_curr - self.q_prev))
            self.delta_history.append(delta_sum)
            
            # Update previous Q
            self.q_prev = self.q_curr.copy()
            
            # Calculate CM
            cm = self.convergence_measurement()
            self.cm_history.append(cm)
            
            # Store metrics
            self.reward_history.append(total_reward)
            self.energy_history.append(info.get('energy', 0))
            self.time_history.append(info.get('time', 0))
            
            # Print progress
            if ep % 10 == 0 or ep == 1:
                elapsed = time.time() - start_time
                avg_time = elapsed / ep
                eta = (episodes - ep) * avg_time
                eta_str = time.strftime("%H:%M:%S", time.gmtime(eta))
                
                success = "✓" if self.env.seg_idx >= self.env.n_segments - 1 else "✗"
                update_type = "SARSA" if cm > self.phi else "Q-Learn"
                
                print(f"{success} Ep {ep:04d}/{episodes} | "
                      f"Steps: {steps:5d} | "
                      f"Reward: {total_reward:9.2f} | "
                      f"CM: {cm:6.4f} | "
                      f"Update: {update_type:8s} | "
                      f"ε: {self.epsilon:.3f} | "
                      f"ETA: {eta_str}")
        
        elapsed_total = time.time() - start_time
        print(f"\n{'='*70}")
        print(f"Q-SARSA TRAINING COMPLETE")
        print(f"{'='*70}")
        print(f"Total time: {time.strftime('%H:%M:%S', time.gmtime(elapsed_total))}")
        print(f"SARSA updates: {self.use_sarsa_count} ({self.use_sarsa_count/(self.use_sarsa_count+self.use_qlearning_count)*100:.1f}%)")
        print(f"Q-learning updates: {self.use_qlearning_count} ({self.use_qlearning_count/(self.use_sarsa_count+self.use_qlearning_count)*100:.1f}%)")
        
        self.save_results()
    
    def save_results(self):
        """Save training results and plots"""
        print("\n💾 Saving Q-SARSA results...")
        
        output_dir = os.path.join(config.OUTPUT_DIR, "qsarsa")
        os.makedirs(output_dir, exist_ok=True)
        
        # Save data
        np.savez(os.path.join(output_dir, "qsarsa_data.npz"),
                 q_table=self.q_curr,
                 cm_history=np.array(self.cm_history),
                 delta_history=np.array(self.delta_history),
                 reward_history=np.array(self.reward_history),
                 energy_history=np.array(self.energy_history),
                 time_history=np.array(self.time_history))
        
        # Plot results
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # CM plot
        axes[0, 0].plot(self.cm_history, linewidth=1, alpha=0.7)
        axes[0, 0].axhline(y=self.phi, color='red', linestyle='--', label=f'φ={self.phi}')
        axes[0, 0].set_title('Convergence Measurement')
        axes[0, 0].set_xlabel('Episode')
        axes[0, 0].set_ylabel('CM')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Reward plot
        axes[0, 1].plot(self.reward_history, linewidth=1, alpha=0.7)
        axes[0, 1].set_title('Total Reward per Episode')
        axes[0, 1].set_xlabel('Episode')
        axes[0, 1].set_ylabel('Reward')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Energy plot
        axes[1, 0].plot(self.energy_history, linewidth=1, alpha=0.7)
        axes[1, 0].set_title('Energy Consumption per Episode')
        axes[1, 0].set_xlabel('Episode')
        axes[1, 0].set_ylabel('Energy (kWh)')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Time plot
        axes[1, 1].plot(self.time_history, linewidth=1, alpha=0.7)
        axes[1, 1].set_title('Travel Time per Episode')
        axes[1, 1].set_xlabel('Episode')
        axes[1, 1].set_ylabel('Time (s)')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "qsarsa_training.png"), dpi=300)
        plt.close()
        
        print(f"✓ Results saved to: {output_dir}/")
    
    def get_optimal_policy(self):
        """
        Extract optimal policy from learned Q-table
        Returns dictionary: {(seg_idx, vel_idx): best_action}
        """
        policy = {}
        for s in range(self.q_shape[0]):
            for v in range(self.q_shape[1]):
                policy[(s, v)] = np.argmax(self.q_curr[s, v])
        return policy