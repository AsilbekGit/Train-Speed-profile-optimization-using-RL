"""
Deep Q-Network - STABLE VERSION
===============================
Key Fixes:
1. Gradient clipping (prevents explosion)
2. Target network (stability)
3. Reward normalization
4. Lower learning rate
5. Network output bias toward Power action
6. Huber loss instead of MSE
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
        Initialize Deep Q-Network with stability improvements
        """
        self.env = env
        self.phi = phi_threshold
        self.state_dim = state_dim
        self.n_actions = 4
        
        # Network architecture: 3 → 128 → 64 → 16 → 4
        self.layer_sizes = [state_dim, 128, 64, 16, self.n_actions]
        
        # Initialize weights
        np.random.seed(42)
        self.weights = []
        self.biases = []
        
        for i in range(len(self.layer_sizes) - 1):
            fan_in = self.layer_sizes[i]
            fan_out = self.layer_sizes[i+1]
            # Xavier initialization
            scale = np.sqrt(2.0 / (fan_in + fan_out))
            w = np.random.randn(fan_in, fan_out) * scale
            b = np.zeros(fan_out)
            self.weights.append(w)
            self.biases.append(b)
        
        # Bias the output layer toward Power action (index 3)
        self.biases[-1][3] = 2.0   # Power - preferred
        self.biases[-1][2] = 1.0   # Cruise - second choice
        self.biases[-1][1] = 0.0   # Coast
        self.biases[-1][0] = -1.0  # Brake - discouraged
        
        # Target network (updated slowly for stability)
        self.target_weights = [w.copy() for w in self.weights]
        self.target_biases = [b.copy() for b in self.biases]
        self.target_update_freq = 50
        
        # For CM calculation
        self.prev_weights = [w.copy() for w in self.weights]
        
        # Experience replay
        self.replay_buffer = []
        self.buffer_size = 10000
        self.batch_size = 32
        self.min_replay_size = 500
        
        # Hyperparameters - tuned for stability
        self.alpha = 0.0005    # Low learning rate
        self.gamma = 0.95
        self.epsilon = 0.15
        self.epsilon_min = 0.02
        self.epsilon_decay = 0.998
        
        # Stability parameters
        self.grad_clip = 1.0
        self.reward_scale = 0.01  # Scale down rewards
        
        # History
        self.delta_history = []
        self.cm_history = []
        self.reward_history = []
        self.energy_history = []
        self.time_history = []
        self.success_history = []
        self.loss_history = []
        
        self.use_sarsa_count = 0
        self.use_qlearning_count = 0
        
        total_params = sum(w.size + b.size for w, b in zip(self.weights, self.biases))
        
        print(f"\nDeep Q-Network Initialized (STABLE):")
        print(f"  Architecture: {' → '.join(map(str, self.layer_sizes))}")
        print(f"  Total parameters: {total_params}")
        print(f"  φ threshold: {self.phi}")
        print(f"  α (learning rate): {self.alpha}")
        print(f"  γ (discount): {self.gamma}")
        print(f"  ε (exploration): {self.epsilon} → {self.epsilon_min}")
        print(f"  Gradient clipping: {self.grad_clip}")
        print(f"  Reward scaling: {self.reward_scale}")
        print(f"  Target update: every {self.target_update_freq} episodes")
    
    def normalize_state(self, raw_state):
        """Normalize state to [0, 1] range"""
        seg_idx = int(raw_state[0])
        velocity = raw_state[1]
        
        pos_norm = seg_idx / self.env.n_segments
        vel_norm = velocity / config.MAX_SPEED_MS
        dist_to_end = 1.0 - pos_norm
        
        return np.array([pos_norm, vel_norm, dist_to_end], dtype=np.float32)
    
    def relu(self, x):
        return np.maximum(0, x)
    
    def relu_derivative(self, x):
        return (x > 0).astype(float)
    
    def forward(self, state, use_target=False):
        """Forward pass"""
        weights = self.target_weights if use_target else self.weights
        biases = self.target_biases if use_target else self.biases
        
        activations = [state]
        z_values = []
        
        for i in range(len(weights) - 1):
            z = activations[-1] @ weights[i] + biases[i]
            z_values.append(z)
            a = self.relu(z)
            activations.append(a)
        
        # Output layer (linear)
        z = activations[-1] @ weights[-1] + biases[-1]
        z_values.append(z)
        q_values = z
        activations.append(q_values)
        
        return q_values, activations, z_values
    
    def predict(self, state, use_target=False):
        """Get Q-values"""
        q_values, _, _ = self.forward(state, use_target)
        return q_values
    
    def epsilon_greedy(self, state):
        """Action selection"""
        if np.random.rand() < self.epsilon:
            return np.random.randint(self.n_actions)
        return np.argmax(self.predict(state))
    
    def convergence_measurement(self):
        """Calculate CM"""
        if len(self.delta_history) < 2:
            return 1.0
        
        delta_n = self.delta_history[-1]
        delta_n_minus_1 = self.delta_history[-2]
        
        if delta_n_minus_1 > 1e-9:
            return delta_n / delta_n_minus_1
        return 1.0
    
    def store_transition(self, state, action, reward, next_state, done):
        """Store with reward scaling"""
        scaled_reward = reward * self.reward_scale
        scaled_reward = np.clip(scaled_reward, -10, 10)
        
        if len(self.replay_buffer) >= self.buffer_size:
            self.replay_buffer.pop(0)
        
        self.replay_buffer.append({
            'state': state.copy(),
            'action': action,
            'reward': scaled_reward,
            'next_state': next_state.copy(),
            'done': done
        })
    
    def update_target_network(self):
        """Soft update target network"""
        tau = 0.1  # Soft update rate
        for i in range(len(self.weights)):
            self.target_weights[i] = tau * self.weights[i] + (1 - tau) * self.target_weights[i]
            self.target_biases[i] = tau * self.biases[i] + (1 - tau) * self.target_biases[i]
    
    def huber_loss(self, td_error, delta=1.0):
        """Huber loss - more robust than MSE"""
        if abs(td_error) <= delta:
            return 0.5 * td_error ** 2
        return delta * (abs(td_error) - 0.5 * delta)
    
    def huber_loss_grad(self, td_error, delta=1.0):
        """Gradient of Huber loss"""
        if abs(td_error) <= delta:
            return td_error
        return delta * np.sign(td_error)
    
    def train_step(self):
        """Training step with stability"""
        if len(self.replay_buffer) < self.min_replay_size:
            return 0.0
        
        # Sample batch
        indices = np.random.choice(len(self.replay_buffer), 
                                   min(self.batch_size, len(self.replay_buffer)), 
                                   replace=False)
        batch = [self.replay_buffer[i] for i in indices]
        
        total_loss = 0.0
        cm = self.convergence_measurement()
        use_sarsa = cm > self.phi
        
        # Accumulate gradients
        grad_w = [np.zeros_like(w) for w in self.weights]
        grad_b = [np.zeros_like(b) for b in self.biases]
        
        for transition in batch:
            state = transition['state']
            action = transition['action']
            reward = transition['reward']
            next_state = transition['next_state']
            done = transition['done']
            
            # Forward pass
            q_values, activations, z_values = self.forward(state)
            
            # Target using TARGET network
            if done:
                target = reward
            else:
                next_q_target = self.predict(next_state, use_target=True)
                if use_sarsa:
                    next_q_main = self.predict(next_state, use_target=False)
                    next_action = np.argmax(next_q_main)
                    target = reward + self.gamma * next_q_target[next_action]
                    self.use_sarsa_count += 1
                else:
                    target = reward + self.gamma * np.max(next_q_target)
                    self.use_qlearning_count += 1
            
            # TD error with clipping
            td_error = target - q_values[action]
            td_error = np.clip(td_error, -10, 10)
            
            total_loss += self.huber_loss(td_error)
            
            # Backpropagation with Huber loss gradient
            output_grad = np.zeros(self.n_actions)
            output_grad[action] = -self.huber_loss_grad(td_error) / self.batch_size
            
            grad = output_grad
            for i in range(len(self.weights) - 1, -1, -1):
                dw = np.outer(activations[i], grad)
                db = grad.copy()
                
                # Clip gradients
                dw = np.clip(dw, -self.grad_clip, self.grad_clip)
                db = np.clip(db, -self.grad_clip, self.grad_clip)
                
                grad_w[i] += dw
                grad_b[i] += db
                
                if i > 0:
                    grad = (grad @ self.weights[i].T) * self.relu_derivative(z_values[i-1])
                    grad = np.clip(grad, -self.grad_clip, self.grad_clip)
        
        # Apply gradients
        for i in range(len(self.weights)):
            self.weights[i] -= self.alpha * grad_w[i]
            self.biases[i] -= self.alpha * grad_b[i]
        
        return total_loss / len(batch)
    
    def train(self, episodes=5000):
        """Train DQN"""
        print(f"\n{'='*70}")
        print(f"STARTING DEEP Q-NETWORK TRAINING")
        print(f"{'='*70}")
        print(f"Total Episodes: {episodes}")
        print(f"φ threshold: {self.phi}")
        print(f"Training starts after {self.min_replay_size} transitions")
        print(f"{'='*70}\n")
        
        start_time = time.time()
        success_count = 0
        best_energy = float('inf')
        best_time = float('inf')
        
        for ep in range(1, episodes + 1):
            self.env.reset()
            raw_state = self.env._get_state()
            state = self.normalize_state(raw_state)
            
            total_reward = 0.0
            steps = 0
            episode_loss = 0.0
            loss_count = 0
            
            while steps < config.MAX_STEPS_PER_EPISODE:
                action = self.epsilon_greedy(state)
                
                next_raw_state, reward, done, info = self.env.step(action)
                next_state = self.normalize_state(next_raw_state)
                
                total_reward += reward
                steps += 1
                
                self.store_transition(state, action, reward, next_state, done)
                
                loss = self.train_step()
                if loss > 0:
                    episode_loss += loss
                    loss_count += 1
                
                if done:
                    break
                
                if self.env.v < 0.1 and steps > 100:
                    break
                
                state = next_state
            
            # Update target network
            if ep % self.target_update_freq == 0:
                self.update_target_network()
            
            # Calculate delta for CM
            delta_sum = sum(np.sum(np.abs(w - wp)) 
                           for w, wp in zip(self.weights, self.prev_weights))
            self.delta_history.append(delta_sum)
            self.prev_weights = [w.copy() for w in self.weights]
            
            cm = self.convergence_measurement()
            self.cm_history.append(cm)
            
            # Success check
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
            avg_loss = episode_loss / max(1, loss_count)
            self.loss_history.append(avg_loss)
            
            # Decay epsilon
            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
            
            # Print progress
            if ep % 100 == 0 or ep <= 10:
                elapsed = time.time() - start_time
                eta = (episodes - ep) * (elapsed / ep) if ep > 0 else 0
                eta_str = time.strftime("%H:%M:%S", time.gmtime(eta))
                
                marker = "✓" if episode_success else "✗"
                rate = (success_count / ep) * 100
                mode = "SARSA" if cm > self.phi else "Q-Learn"
                
                print(f"{marker} Ep {ep:04d}/{episodes} | "
                      f"Success: {rate:5.1f}% | "
                      f"Energy: {info.get('energy', 0):7.1f} kWh | "
                      f"Time: {info.get('time', 0):5.0f}s | "
                      f"Loss: {avg_loss:7.4f} | "
                      f"Mode: {mode:7s} | "
                      f"ε: {self.epsilon:.3f} | "
                      f"ETA: {eta_str}")
        
        elapsed_total = time.time() - start_time
        print(f"\n{'='*70}")
        print(f"DEEP Q-NETWORK TRAINING COMPLETE")
        print(f"{'='*70}")
        print(f"Total time: {time.strftime('%H:%M:%S', time.gmtime(elapsed_total))}")
        print(f"Final success rate: {(success_count/episodes)*100:.1f}%")
        print(f"SARSA updates: {self.use_sarsa_count}")
        print(f"Q-learning updates: {self.use_qlearning_count}")
        if best_energy < float('inf'):
            print(f"\nBest Performance:")
            print(f"  Best energy: {best_energy:.1f} kWh")
            print(f"  Best time: {best_time:.0f} s")
        
        self.save_results()
    
    def save_results(self):
        """Save results"""
        print("\n💾 Saving Deep-Q results...")
        
        output_dir = os.path.join(config.OUTPUT_DIR, "deep_q")
        os.makedirs(output_dir, exist_ok=True)
        
        weights_dict = {f'w{i}': w for i, w in enumerate(self.weights)}
        biases_dict = {f'b{i}': b for i, b in enumerate(self.biases)}
        
        np.savez(os.path.join(output_dir, "dqn_weights.npz"),
                 **weights_dict, **biases_dict,
                 cm_history=np.array(self.cm_history),
                 reward_history=np.array(self.reward_history),
                 energy_history=np.array(self.energy_history),
                 time_history=np.array(self.time_history),
                 success_history=np.array(self.success_history),
                 loss_history=np.array(self.loss_history))
        
        # Plot
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        if self.success_history:
            cumulative = np.cumsum(self.success_history) / np.arange(1, len(self.success_history)+1) * 100
            axes[0, 0].plot(cumulative, linewidth=2, color='green')
        axes[0, 0].set_title('Success Rate (%)')
        axes[0, 0].set_xlabel('Episode')
        axes[0, 0].grid(True, alpha=0.3)
        
        axes[0, 1].plot(self.loss_history, linewidth=1, alpha=0.7)
        axes[0, 1].set_title('Training Loss')
        axes[0, 1].set_xlabel('Episode')
        axes[0, 1].grid(True, alpha=0.3)
        
        successful_eps = [i for i, s in enumerate(self.success_history) if s]
        if successful_eps:
            successful_energy = [self.energy_history[i] for i in successful_eps]
            axes[1, 0].plot(successful_eps, successful_energy, 'b.', alpha=0.5, markersize=2)
        axes[1, 0].set_title('Energy (Successful Episodes)')
        axes[1, 0].set_xlabel('Episode')
        axes[1, 0].set_ylabel('Energy (kWh)')
        axes[1, 0].grid(True, alpha=0.3)
        
        if successful_eps:
            successful_time = [self.time_history[i] for i in successful_eps]
            axes[1, 1].plot(successful_eps, successful_time, 'g.', alpha=0.5, markersize=2)
        axes[1, 1].set_title('Time (Successful Episodes)')
        axes[1, 1].set_xlabel('Episode')
        axes[1, 1].set_ylabel('Time (s)')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "dqn_training.png"), dpi=300)
        plt.close()
        
        print(f"✓ Results saved to: {output_dir}/")
    
    def generate_speed_profile(self):
        """Generate optimal speed profile"""
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
        
        output_dir = os.path.join(config.OUTPUT_DIR, "deep_q")
        np.savez(os.path.join(output_dir, "speed_profile.npz"),
                 segments=np.array(segments),
                 velocities=np.array(velocities),
                 actions=np.array(actions),
                 energies=np.array(energies))
        
        print(f"✓ Speed profile saved")
        print(f"  Final energy: {energies[-1] if energies else 0:.1f} kWh")
        print(f"  Final segment: {segments[-1] if segments else 0}/{self.env.n_segments}")
        
        return segments, velocities, actions, energies