"""
Deep-Q Network Implementation
Based on Section 3.6 and Figure 10 from the paper
"A comprehensive study on reinforcement learning application for train speed profile optimization"
by Sandidzadeh & Havaei (2023)

Network Architecture (Fig. 10): Input → 128 → 64 → 16 → 4 (Output)
Activation: tanh for hidden layers, sigmoid for output
"""

import numpy as np
import matplotlib.pyplot as plt
import os
import config
import time

class DeepQNetwork:
    def __init__(self, env, phi_threshold, state_dim=3):
        """
        Initialize Deep-Q Network
        
        Args:
            env: Training environment
            phi_threshold: φ threshold from CM analysis
            state_dim: State dimension (position, velocity, distance_to_end)
        """
        self.env = env
        self.phi = phi_threshold
        self.state_dim = state_dim
        self.n_actions = 4
        
        # Network architecture (Figure 10)
        # Input(3) → 128 → 64 → 16 → Output(4)
        self.layer_sizes = [state_dim, 128, 64, 16, self.n_actions]
        
        # Initialize weights and biases with Xavier initialization
        self.weights = []
        self.biases = []
        
        for i in range(len(self.layer_sizes) - 1):
            # Xavier initialization: scale by sqrt(2 / fan_in)
            w = np.random.randn(self.layer_sizes[i], self.layer_sizes[i+1]) * np.sqrt(2.0 / self.layer_sizes[i])
            b = np.zeros(self.layer_sizes[i+1])
            self.weights.append(w)
            self.biases.append(b)
        
        # Store previous weights for CM calculation
        self.prev_weights = [w.copy() for w in self.weights]
        
        # Hyperparameters
        self.alpha = 0.001  # Learning rate (smaller for neural network)
        self.gamma = 0.95   # Discount factor
        self.epsilon = 0.15 # Exploration rate
        
        # History tracking
        self.delta_history = []
        self.cm_history = []
        self.reward_history = []
        self.energy_history = []
        self.time_history = []
        
        # Algorithm state
        self.use_qlearning_count = 0
        self.use_sarsa_count = 0
        
        print(f"\nDeep-Q Network Initialized:")
        print(f"  Architecture: {' → '.join(map(str, self.layer_sizes))}")
        print(f"  φ threshold: {self.phi}")
        print(f"  Total parameters: {sum(w.size for w in self.weights)}")
        print(f"  α (learning rate): {self.alpha}")
        print(f"  γ (discount): {self.gamma}")
    
    def normalize_state(self, raw_state):
        """
        Convert raw state to normalized network input
        [segment_idx, velocity] → [pos_norm, vel_norm, dist_to_end]
        """
        seg_idx = int(raw_state[0])
        velocity = raw_state[1]
        
        pos_norm = seg_idx / self.env.n_segments
        vel_norm = velocity / config.MAX_SPEED_MS
        dist_to_end = (self.env.n_segments - seg_idx) / self.env.n_segments
        
        return np.array([pos_norm, vel_norm, dist_to_end], dtype=np.float32)
    
    def tanh(self, x):
        """Hyperbolic tangent activation"""
        return np.tanh(x)
    
    def tanh_derivative(self, x):
        """Derivative of tanh"""
        return 1.0 - np.tanh(x) ** 2
    
    def sigmoid(self, x):
        """Sigmoid activation for output layer"""
        return 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))
    
    def forward(self, state):
        """
        Forward pass through network
        Architecture: 3 tanh layers + 1 sigmoid layer
        
        Returns:
            q_values: Q-values for all actions
            activations: List of activations (for backprop)
            z_values: List of pre-activation values (for backprop)
        """
        activations = [state]
        z_values = []
        
        # Hidden layers with tanh activation
        for i in range(len(self.weights) - 1):
            z = activations[-1] @ self.weights[i] + self.biases[i]
            z_values.append(z)
            a = self.tanh(z)
            activations.append(a)
        
        # Output layer with sigmoid (scaled to meaningful Q-values)
        z = activations[-1] @ self.weights[-1] + self.biases[-1]
        z_values.append(z)
        q_values = self.sigmoid(z) * 200 - 100  # Scale to [-100, 100] range
        activations.append(q_values)
        
        return q_values, activations, z_values
    
    def epsilon_greedy(self, state):
        """ε-greedy action selection"""
        if np.random.rand() < self.epsilon:
            return np.random.randint(self.n_actions)
        else:
            q_values, _, _ = self.forward(state)
            return np.argmax(q_values)
    
    def convergence_measurement(self):
        """
        Calculate CM based on weight changes
        (Similar to Q-table delta in Q-SARSA)
        """
        if len(self.delta_history) < 2:
            return float('inf')
        
        delta_n = self.delta_history[-1]
        delta_n_minus_1 = self.delta_history[-2]
        
        if delta_n_minus_1 > 1e-9:
            return delta_n / delta_n_minus_1
        else:
            return 1.0
    
    def backprop_update(self, state, action, reward, next_state, next_action, done):
        """
        Backpropagation update using Q-SARSA logic
        """
        # Forward pass for current state
        q_values, activations, z_values = self.forward(state)
        old_q = q_values[action]
        
        # Calculate target using Q-SARSA hybrid
        next_q_values, _, _ = self.forward(next_state)
        cm = self.convergence_measurement()
        
        if cm > self.phi:
            # SARSA update
            if done:
                target = reward
            else:
                target = reward + self.gamma * next_q_values[next_action]
            self.use_sarsa_count += 1
        else:
            # Q-learning update
            if done:
                target = reward
            else:
                target = reward + self.gamma * np.max(next_q_values)
            self.epsilon = min(0.3, self.epsilon * 1.1)
            self.use_qlearning_count += 1
        
        # Calculate error
        error = np.zeros(self.n_actions)
        error[action] = target - q_values[action]
        
        # Backpropagation
        deltas = [error]
        
        # Backward through layers
        for i in range(len(self.weights) - 1, 0, -1):
            # Gradient through tanh
            delta = (deltas[0] @ self.weights[i].T) * self.tanh_derivative(z_values[i-1])
            deltas.insert(0, delta)
        
        # Update weights and biases (gradient ascent)
        for i in range(len(self.weights)):
            self.weights[i] += self.alpha * np.outer(activations[i], deltas[i])
            self.biases[i] += self.alpha * deltas[i]
        
        return abs(target - old_q)
    
    def train(self, episodes=1000):
        """
        Train Deep-Q Network
        """
        print(f"\n{'='*70}")
        print(f"STARTING DEEP-Q NETWORK TRAINING")
        print(f"{'='*70}")
        print(f"Total Episodes: {episodes}")
        print(f"φ threshold: {self.phi}")
        print(f"{'='*70}\n")
        
        start_time = time.time()
        
        for ep in range(1, episodes + 1):
            # Reset environment
            self.env.reset()
            raw_state = self.env._get_state()
            state = self.normalize_state(raw_state)
            
            # Initial action
            action = self.epsilon_greedy(state)
            
            total_reward = 0.0
            steps = 0
            
            # Episode loop
            while steps < config.MAX_STEPS_PER_EPISODE:
                # Take action
                next_raw_state, reward, done, info = self.env.step(action)
                next_state = self.normalize_state(next_raw_state)
                
                total_reward += reward
                steps += 1
                
                # Select next action
                next_action = self.epsilon_greedy(next_state)
                
                # Update network
                self.backprop_update(state, action, reward, next_state, next_action, done)
                
                if done:
                    break
                
                state = next_state
                action = next_action
            
            # Calculate weight change (delta)
            delta_sum = sum(np.sum(np.abs(w - wp)) 
                           for w, wp in zip(self.weights, self.prev_weights))
            self.delta_history.append(delta_sum)
            
            # Update previous weights
            self.prev_weights = [w.copy() for w in self.weights]
            
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
        print(f"DEEP-Q NETWORK TRAINING COMPLETE")
        print(f"{'='*70}")
        print(f"Total time: {time.strftime('%H:%M:%S', time.gmtime(elapsed_total))}")
        print(f"SARSA updates: {self.use_sarsa_count} ({self.use_sarsa_count/(self.use_sarsa_count+self.use_qlearning_count)*100:.1f}%)")
        print(f"Q-learning updates: {self.use_qlearning_count} ({self.use_qlearning_count/(self.use_sarsa_count+self.use_qlearning_count)*100:.1f}%)")
        
        self.save_results()
    
    # dqn.py - Fix save_results()

    def save_results(self):
        """Save training results"""
        print("\n💾 Saving Deep-Q results...")
        
        output_dir = os.path.join(config.OUTPUT_DIR, "deep_q")
        os.makedirs(output_dir, exist_ok=True)
        
        # FIXED: Save weights as separate arrays
        weights_dict = {f'w{i}': w for i, w in enumerate(self.weights)}
        biases_dict = {f'b{i}': b for i, b in enumerate(self.biases)}
        
        np.savez(os.path.join(output_dir, "deep_q_weights.npz"),
                **weights_dict,  # Unpack dictionary
                **biases_dict,
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
        plt.savefig(os.path.join(output_dir, "deep_q_training.png"), dpi=300)
        plt.close()
        
        print(f"✓ Results saved to: {output_dir}/")
    
    def predict(self, state):
        """Get best action for a given state"""
        q_values, _, _ = self.forward(state)
        return np.argmax(q_values)