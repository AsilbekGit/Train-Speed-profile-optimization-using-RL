"""
Deep Q-Network Implementation
=============================
Based on Section 3.6 and Figure 10 from the paper

Network Architecture (Figure 10): Input(3) → 128 → 64 → 16 → Output(4)
Uses Q-SARSA hybrid switching based on φ threshold
"""

import numpy as np
import matplotlib.pyplot as plt
import os
import env_settings.config as config
from data.utils import discretize_state
import time

class DeepQNetwork:
    def __init__(self, env, phi_threshold=0.10, state_dim=3):
        """
        Initialize Deep Q-Network
        
        Args:
            env: Training environment
            phi_threshold: φ threshold from CM analysis
            state_dim: State dimension (position, velocity, distance_to_end)
        """
        self.env = env
        self.phi = phi_threshold
        self.state_dim = state_dim
        self.n_actions = 4
        
        # Network architecture (Figure 10): 3 → 128 → 64 → 16 → 4
        self.layer_sizes = [state_dim, 128, 64, 16, self.n_actions]
        
        # Initialize weights with Xavier initialization
        self.weights = []
        self.biases = []
        
        np.random.seed(42)  # For reproducibility
        for i in range(len(self.layer_sizes) - 1):
            scale = np.sqrt(2.0 / self.layer_sizes[i])
            w = np.random.randn(self.layer_sizes[i], self.layer_sizes[i+1]) * scale
            b = np.zeros(self.layer_sizes[i+1])
            self.weights.append(w)
            self.biases.append(b)
        
        # Previous weights for CM calculation
        self.prev_weights = [w.copy() for w in self.weights]
        
        # Experience replay buffer
        self.replay_buffer = []
        self.buffer_size = 10000
        self.batch_size = 32
        
        # Hyperparameters
        self.alpha = 0.001   # Learning rate
        self.gamma = 0.95    # Discount factor
        self.epsilon = 0.15  # Exploration rate
        self.epsilon_min = 0.02
        self.epsilon_decay = 0.999
        
        # History tracking
        self.delta_history = []
        self.cm_history = []
        self.reward_history = []
        self.energy_history = []
        self.time_history = []
        self.success_history = []
        self.loss_history = []
        
        # Algorithm statistics
        self.use_sarsa_count = 0
        self.use_qlearning_count = 0
        
        total_params = sum(w.size + b.size for w, b in zip(self.weights, self.biases))
        
        print(f"\nDeep Q-Network Initialized:")
        print(f"  Architecture: {' → '.join(map(str, self.layer_sizes))}")
        print(f"  Total parameters: {total_params}")
        print(f"  φ threshold: {self.phi}")
        print(f"  α (learning rate): {self.alpha}")
        print(f"  γ (discount): {self.gamma}")
        print(f"  ε (exploration): {self.epsilon} → {self.epsilon_min}")
        print(f"  Replay buffer: {self.buffer_size}")
        print(f"  Batch size: {self.batch_size}")
    
    def normalize_state(self, raw_state):
        """Convert raw state to normalized network input"""
        seg_idx = int(raw_state[0])
        velocity = raw_state[1]
        
        pos_norm = seg_idx / self.env.n_segments
        vel_norm = velocity / config.MAX_SPEED_MS
        dist_to_end = (self.env.n_segments - seg_idx) / self.env.n_segments
        
        return np.array([pos_norm, vel_norm, dist_to_end], dtype=np.float32)
    
    def relu(self, x):
        """ReLU activation"""
        return np.maximum(0, x)
    
    def relu_derivative(self, x):
        """Derivative of ReLU"""
        return (x > 0).astype(float)
    
    def forward(self, state):
        """
        Forward pass through network
        
        Returns:
            q_values: Q-values for all actions
            activations: List of activations (for backprop)
            z_values: List of pre-activation values (for backprop)
        """
        activations = [state]
        z_values = []
        
        # Hidden layers with ReLU
        for i in range(len(self.weights) - 1):
            z = activations[-1] @ self.weights[i] + self.biases[i]
            z_values.append(z)
            a = self.relu(z)
            activations.append(a)
        
        # Output layer (linear)
        z = activations[-1] @ self.weights[-1] + self.biases[-1]
        z_values.append(z)
        q_values = z  # Linear output
        activations.append(q_values)
        
        return q_values, activations, z_values
    
    def predict(self, state):
        """Get Q-values for a state"""
        q_values, _, _ = self.forward(state)
        return q_values
    
    def epsilon_greedy(self, state):
        """ε-greedy action selection"""
        if np.random.rand() < self.epsilon:
            return np.random.randint(self.n_actions)
        else:
            q_values = self.predict(state)
            return np.argmax(q_values)
    
    def convergence_measurement(self):
        """Calculate CM based on weight changes"""
        if len(self.delta_history) < 2:
            return float('inf')
        
        delta_n = self.delta_history[-1]
        delta_n_minus_1 = self.delta_history[-2]
        
        if delta_n_minus_1 > 1e-9:
            return delta_n / delta_n_minus_1
        else:
            return 1.0
    
    def store_transition(self, state, action, reward, next_state, done):
        """Store transition in replay buffer"""
        if len(self.replay_buffer) >= self.buffer_size:
            self.replay_buffer.pop(0)
        
        self.replay_buffer.append({
            'state': state,
            'action': action,
            'reward': reward,
            'next_state': next_state,
            'done': done
        })
    
    def sample_batch(self):
        """Sample random batch from replay buffer"""
        indices = np.random.choice(len(self.replay_buffer), 
                                   min(self.batch_size, len(self.replay_buffer)), 
                                   replace=False)
        return [self.replay_buffer[i] for i in indices]
    
    def train_step(self):
        """Perform one training step using experience replay"""
        if len(self.replay_buffer) < self.batch_size:
            return 0.0
        
        batch = self.sample_batch()
        total_loss = 0.0
        
        cm = self.convergence_measurement()
        use_sarsa = cm > self.phi
        
        for transition in batch:
            state = transition['state']
            action = transition['action']
            reward = transition['reward']
            next_state = transition['next_state']
            done = transition['done']
            
            # Forward pass
            q_values, activations, z_values = self.forward(state)
            
            # Calculate target
            if done:
                target = reward
            else:
                next_q = self.predict(next_state)
                if use_sarsa:
                    # SARSA: use action that would be selected
                    next_action = np.argmax(next_q)
                    target = reward + self.gamma * next_q[next_action]
                    self.use_sarsa_count += 1
                else:
                    # Q-learning: use max Q-value
                    target = reward + self.gamma * np.max(next_q)
                    self.use_qlearning_count += 1
            
            # Calculate error
            td_error = target - q_values[action]
            total_loss += td_error ** 2
            
            # Backpropagation
            # Output layer gradient
            output_grad = np.zeros(self.n_actions)
            output_grad[action] = -2 * td_error  # Gradient of MSE
            
            # Backward through layers
            grad = output_grad
            for i in range(len(self.weights) - 1, -1, -1):
                # Gradient w.r.t. weights and biases
                dw = np.outer(activations[i], grad)
                db = grad
                
                # Gradient w.r.t. previous layer
                if i > 0:
                    grad = (grad @ self.weights[i].T) * self.relu_derivative(z_values[i-1])
                
                # Update weights (gradient descent)
                self.weights[i] -= self.alpha * dw
                self.biases[i] -= self.alpha * db
        
        return total_loss / len(batch)
    
    def train(self, episodes=5000):
        """Train Deep Q-Network"""
        print(f"\n{'='*70}")
        print(f"STARTING DEEP Q-NETWORK TRAINING")
        print(f"{'='*70}")
        print(f"Total Episodes: {episodes}")
        print(f"φ threshold: {self.phi}")
        print(f"{'='*70}\n")
        
        start_time = time.time()
        success_count = 0
        best_energy = float('inf')
        best_time = float('inf')
        
        for ep in range(1, episodes + 1):
            # Reset environment
            self.env.reset()
            raw_state = self.env._get_state()
            state = self.normalize_state(raw_state)
            
            total_reward = 0.0
            steps = 0
            episode_loss = 0.0
            
            while steps < config.MAX_STEPS_PER_EPISODE:
                # Select action
                action = self.epsilon_greedy(state)
                
                # Take action
                next_raw_state, reward, done, info = self.env.step(action)
                next_state = self.normalize_state(next_raw_state)
                
                total_reward += reward
                steps += 1
                
                # Store transition
                self.store_transition(state, action, reward, next_state, done)
                
                # Train
                loss = self.train_step()
                episode_loss += loss
                
                if done:
                    break
                
                if self.env.v < 0.1 and steps > 100:
                    break
                
                state = next_state
            
            # Calculate weight delta
            delta_sum = sum(np.sum(np.abs(w - wp)) 
                           for w, wp in zip(self.weights, self.prev_weights))
            self.delta_history.append(delta_sum)
            self.prev_weights = [w.copy() for w in self.weights]
            
            # Calculate CM
            cm = self.convergence_measurement()
            self.cm_history.append(cm)
            
            # Check success
            episode_success = self.env.seg_idx >= self.env.n_segments - 2
            if episode_success:
                success_count += 1
                if info['energy'] < best_energy:
                    best_energy = info['energy']
                if info['time'] < best_time:
                    best_time = info['time']
            
            self.success_history.append(episode_success)
            self.reward_history.append(total_reward)
            self.energy_history.append(info.get('energy', 0))
            self.time_history.append(info.get('time', 0))
            self.loss_history.append(episode_loss / max(1, steps))
            
            # Decay epsilon
            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
            
            # Print progress
            if ep % 100 == 0 or ep <= 10:
                elapsed = time.time() - start_time
                eta = (episodes - ep) * (elapsed / ep) if ep > 0 else 0
                eta_str = time.strftime("%H:%M:%S", time.gmtime(eta))
                
                marker = "✓" if episode_success else "✗"
                rate = (success_count / ep) * 100
                avg_loss = self.loss_history[-1] if self.loss_history else 0
                
                print(f"{marker} Ep {ep:04d}/{episodes} | "
                      f"Success: {rate:5.1f}% | "
                      f"Energy: {info.get('energy', 0):6.1f} kWh | "
                      f"Loss: {avg_loss:.4f} | "
                      f"CM: {cm:.4f} | "
                      f"ε: {self.epsilon:.3f} | "
                      f"ETA: {eta_str}")
        
        elapsed_total = time.time() - start_time
        print(f"\n{'='*70}")
        print(f"DEEP Q-NETWORK TRAINING COMPLETE")
        print(f"{'='*70}")
        print(f"Total time: {time.strftime('%H:%M:%S', time.gmtime(elapsed_total))}")
        print(f"Final success rate: {(success_count/episodes)*100:.1f}%")
        print(f"\nBest Performance:")
        print(f"  Best energy: {best_energy:.1f} kWh")
        print(f"  Best time: {best_time:.0f} s")
        
        self.save_results()
    
    def save_results(self):
        """Save training results"""
        print("\n💾 Saving Deep-Q results...")
        
        output_dir = os.path.join(config.OUTPUT_DIR, "deep_q")
        os.makedirs(output_dir, exist_ok=True)
        
        # Save weights
        weights_dict = {f'w{i}': w for i, w in enumerate(self.weights)}
        biases_dict = {f'b{i}': b for i, b in enumerate(self.biases)}
        
        np.savez(os.path.join(output_dir, "dqn_weights.npz"),
                 **weights_dict,
                 **biases_dict,
                 cm_history=np.array(self.cm_history),
                 delta_history=np.array(self.delta_history),
                 reward_history=np.array(self.reward_history),
                 energy_history=np.array(self.energy_history),
                 time_history=np.array(self.time_history),
                 success_history=np.array(self.success_history),
                 loss_history=np.array(self.loss_history))
        
        # Plot results
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # Success rate
        if self.success_history:
            cumulative = np.cumsum(self.success_history) / np.arange(1, len(self.success_history)+1) * 100
            axes[0, 0].plot(cumulative, linewidth=2, color='green')
        axes[0, 0].set_title('Success Rate (%)')
        axes[0, 0].set_xlabel('Episode')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Loss
        axes[0, 1].plot(self.loss_history, linewidth=1, alpha=0.7)
        axes[0, 1].set_title('Training Loss')
        axes[0, 1].set_xlabel('Episode')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Energy
        successful_episodes = [i for i, s in enumerate(self.success_history) if s]
        if successful_episodes:
            successful_energy = [self.energy_history[i] for i in successful_episodes]
            axes[1, 0].plot(successful_episodes, successful_energy, 'b.', alpha=0.5, markersize=2)
        axes[1, 0].set_title('Energy (Successful Episodes)')
        axes[1, 0].set_xlabel('Episode')
        axes[1, 0].set_ylabel('Energy (kWh)')
        axes[1, 0].grid(True, alpha=0.3)
        
        # CM
        axes[1, 1].plot(self.cm_history, linewidth=1, alpha=0.7)
        axes[1, 1].axhline(y=self.phi, color='red', linestyle='--', label=f'φ={self.phi}')
        axes[1, 1].set_title('Convergence Measurement')
        axes[1, 1].set_xlabel('Episode')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "dqn_training.png"), dpi=300)
        plt.close()
        
        print(f"✓ Results saved to: {output_dir}/")
    
    def generate_speed_profile(self):
        """Generate optimal speed profile using learned network"""
        print("\n📊 Generating optimal speed profile (DQN)...")
        
        self.env.reset()
        
        segments = []
        velocities = []
        actions = []
        energies = []
        
        steps = 0
        while steps < config.MAX_STEPS_PER_EPISODE:
            raw_state = self.env._get_state()
            state = self.normalize_state(raw_state)
            
            # Get best action (no exploration)
            q_values = self.predict(state)
            action = np.argmax(q_values)
            
            segments.append(self.env.seg_idx)
            velocities.append(self.env.v)
            actions.append(action)
            energies.append(self.env.energy_kwh)
            
            _, _, done, _ = self.env.step(action)
            steps += 1
            
            if done or self.env.v < 0.1:
                break
        
        # Save profile
        output_dir = os.path.join(config.OUTPUT_DIR, "deep_q")
        np.savez(os.path.join(output_dir, "speed_profile.npz"),
                 segments=np.array(segments),
                 velocities=np.array(velocities),
                 actions=np.array(actions),
                 energies=np.array(energies))
        
        print(f"✓ Speed profile saved")
        print(f"  Final energy: {energies[-1]:.1f} kWh")
        print(f"  Final segment: {segments[-1]}/{self.env.n_segments}")
        
        return segments, velocities, actions, energies