"""
Deep Q-Network - FIXED VERSION (Paper-Compliant)
==================================================
Based on: "A comprehensive study on reinforcement learning application 
for train speed profile optimization" - Sandidzadeh & Havaei (2023)
Section 3.6, Figure 10

Key Fixes from Original:
1. Gradient clipping to ±1.0 (prevents explosion - was reaching 271M)
2. Huber loss instead of MSE (robust to outliers)
3. Target network with soft update τ=0.1 (stability)
4. Reward scaling ×0.01 (normalized gradients)
5. Learning rate reduced to 0.0005
6. TD error clipping to ±10
7. Output bias initialization: Power=+2, Brake=-1 (forward motion preference)
8. Paper-compliant architecture: 128→64→16→4 with tanh hidden + sigmoid output
9. 2D state representation (position, velocity) per paper
10. Three-phase training: Q-SARSA data generation → supervised training → online fine-tuning

Architecture (Figure 10):
    Input (x, v) → 128 tanh → 64 tanh → 16 tanh → 4 sigmoid → action probabilities
"""

import numpy as np
import os
import sys

# Import project modules
try:
    import env_settings.config
except ImportError:
    # Fallback config
    class config:
        OUTPUT_DIR = "results_cm"
        MAX_STEPS_PER_EPISODE = 2000
        N_ACTIONS = 4
        GAMMA = 0.99
        ALPHA = 0.5
        EPSILON_START = 1.0
        EPSILON_MIN = 0.01
        EPSILON_DECAY = 0.999


class DeepQNetwork:
    """
    Deep Q-Network with paper-compliant architecture and stability fixes.
    
    Training follows three phases per the paper (Section 3.6):
    1. Generate training data using Q-SARSA episodes
    2. Train neural network on collected data (supervised)
    3. Online fine-tuning with the network
    """
    
    def __init__(self, env, phi_threshold=0.10):
        self.env = env
        self.phi = phi_threshold
        
        # Network architecture per Figure 10
        self.n_inputs = 2  # (position_normalized, velocity_normalized)
        self.n_actions = getattr(config, 'N_ACTIONS', 4)
        self.hidden_sizes = [128, 64, 16]  # Paper: 3 hidden layers
        
        # Hyperparameters - FIXED
        self.gamma = getattr(config, 'GAMMA', 0.99)
        self.lr = 0.0005  # Reduced from 0.001 (prevents gradient explosion)
        self.epsilon = getattr(config, 'EPSILON_START', 1.0)
        self.epsilon_min = getattr(config, 'EPSILON_MIN', 0.01)
        self.epsilon_decay = getattr(config, 'EPSILON_DECAY', 0.999)
        self.reward_scale = 0.01  # Scale rewards to prevent huge gradients
        self.grad_clip = 1.0  # Gradient clipping threshold
        self.td_clip = 10.0  # TD error clipping
        self.tau = 0.1  # Target network soft update rate
        
        # Experience replay buffer
        self.replay_buffer = []
        self.buffer_size = 10000
        self.batch_size = 64
        self.min_replay = 200  # Minimum samples before training
        
        # Initialize networks (online + target)
        self.weights = self._init_weights()
        self.target_weights = self._deep_copy_weights(self.weights)
        
        # Q-table for Phase 1 (Q-SARSA data generation)
        n_segments = getattr(env, 'n_segments', 749)
        n_speeds = 50  # Discretized speed bins
        self.q_table = np.zeros((n_segments, n_speeds, self.n_actions))
        # Initialize with forward-motion bias
        self.q_table[:, :, 0] = 2.0   # Power: strongly preferred initially
        self.q_table[:, :, 1] = 1.0   # Cruise: good default
        self.q_table[:, :, 2] = 0.0   # Coast: neutral
        self.q_table[:, :, 3] = -1.0  # Brake: discouraged initially
        self.prev_q_table = self.q_table.copy()
        
        # CM-based switching
        self.cm_history = []
        
        # Training history
        self.success_history = []
        self.energy_history = []
        self.time_history = []
        self.loss_history = []
        self.reward_history = []
        
        # Phase tracking
        self.training_data = []  # Collected from Q-SARSA phase
        
        print(f"DQN initialized:")
        print(f"  Architecture: {self.n_inputs} → {' → '.join(map(str, self.hidden_sizes))} → {self.n_actions}")
        print(f"  Activations: tanh (hidden) + sigmoid (output)")
        print(f"  Learning rate: {self.lr}")
        print(f"  Gradient clip: ±{self.grad_clip}")
        print(f"  Reward scale: ×{self.reward_scale}")
        print(f"  Target network: soft update τ={self.tau}")
        print(f"  φ threshold: {self.phi}")
    
    # =====================================================
    # Neural Network (NumPy implementation)
    # =====================================================
    
    def _init_weights(self):
        """Initialize network weights with He initialization + output bias."""
        weights = {}
        layer_sizes = [self.n_inputs] + self.hidden_sizes + [self.n_actions]
        
        for i in range(len(layer_sizes) - 1):
            fan_in = layer_sizes[i]
            fan_out = layer_sizes[i + 1]
            # He initialization for tanh
            std = np.sqrt(2.0 / fan_in)
            weights[f'W{i}'] = np.random.randn(fan_in, fan_out) * std
            weights[f'b{i}'] = np.zeros(fan_out)
        
        # Output bias: favor forward motion (Power=+2, Cruise=+1, Coast=0, Brake=-1)
        last_idx = len(layer_sizes) - 2
        weights[f'b{last_idx}'] = np.array([2.0, 1.0, 0.0, -1.0])
        
        return weights
    
    def _deep_copy_weights(self, weights):
        """Deep copy weights dictionary."""
        return {k: v.copy() for k, v in weights.items()}
    
    def _tanh(self, x):
        """Numerically stable tanh."""
        return np.tanh(np.clip(x, -20, 20))
    
    def _sigmoid(self, x):
        """Numerically stable sigmoid."""
        x = np.clip(x, -20, 20)
        return 1.0 / (1.0 + np.exp(-x))
    
    def _tanh_deriv(self, tanh_output):
        """Derivative of tanh given its output."""
        return 1.0 - tanh_output ** 2
    
    def _sigmoid_deriv(self, sigmoid_output):
        """Derivative of sigmoid given its output."""
        return sigmoid_output * (1.0 - sigmoid_output)
    
    def _forward(self, x, weights=None):
        """Forward pass through the network.
        
        Architecture (Figure 10):
            Input → tanh → tanh → tanh → sigmoid → output
        """
        if weights is None:
            weights = self.weights
        
        n_layers = len(self.hidden_sizes) + 1
        activations = [x]
        pre_activations = []
        
        for i in range(n_layers):
            z = activations[-1] @ weights[f'W{i}'] + weights[f'b{i}']
            pre_activations.append(z)
            
            if i < n_layers - 1:
                # Hidden layers: tanh activation (per paper)
                a = self._tanh(z)
            else:
                # Output layer: sigmoid activation (per paper - "logistic")
                a = self._sigmoid(z)
            
            activations.append(a)
        
        return activations, pre_activations
    
    def predict(self, state, use_target=False):
        """Predict Q-values for a state."""
        weights = self.target_weights if use_target else self.weights
        x = np.array(state, dtype=np.float64).reshape(1, -1)
        activations, _ = self._forward(x, weights)
        return activations[-1].flatten()
    
    def _backward(self, state, target_q, learning_rate=None):
        """Backward pass with gradient clipping.
        
        Returns loss value.
        """
        if learning_rate is None:
            learning_rate = self.lr
        
        x = np.array(state, dtype=np.float64).reshape(1, -1)
        target = np.array(target_q, dtype=np.float64).reshape(1, -1)
        
        # Forward pass
        activations, pre_activations = self._forward(x)
        output = activations[-1]
        
        # Huber loss (robust to outliers, prevents gradient explosion)
        error = output - target
        abs_error = np.abs(error)
        huber_delta = 1.0
        loss = np.where(
            abs_error <= huber_delta,
            0.5 * error ** 2,
            huber_delta * (abs_error - 0.5 * huber_delta)
        ).mean()
        
        # Gradient of Huber loss
        d_output = np.where(
            abs_error <= huber_delta,
            error,
            huber_delta * np.sign(error)
        ) / self.n_actions
        
        # Clip output gradient
        d_output = np.clip(d_output, -self.grad_clip, self.grad_clip)
        
        # Backpropagation
        n_layers = len(self.hidden_sizes) + 1
        delta = d_output * self._sigmoid_deriv(activations[-1])  # Output layer sigmoid
        
        for i in range(n_layers - 1, -1, -1):
            # Gradient for weights and biases
            dW = activations[i].T @ delta
            db = delta.sum(axis=0)
            
            # Clip gradients
            dW = np.clip(dW, -self.grad_clip, self.grad_clip)
            db = np.clip(db, -self.grad_clip, self.grad_clip)
            
            # Update weights
            self.weights[f'W{i}'] -= learning_rate * dW
            self.weights[f'b{i}'] -= learning_rate * db
            
            # Propagate gradient to previous layer
            if i > 0:
                delta = (delta @ self.weights[f'W{i}'].T) * self._tanh_deriv(activations[i])
                # Clip propagated gradient
                delta = np.clip(delta, -self.grad_clip, self.grad_clip)
        
        return loss
    
    def _soft_update_target(self):
        """Soft update target network: θ_target = τ·θ + (1-τ)·θ_target."""
        for key in self.weights:
            self.target_weights[key] = (
                self.tau * self.weights[key] + 
                (1 - self.tau) * self.target_weights[key]
            )
    
    # =====================================================
    # State Processing
    # =====================================================
    
    def normalize_state(self, raw_state):
        """Normalize state to 2D (position, velocity) per paper.
        
        Paper uses (x, v) where:
        - x = position (normalized to [0, 1])
        - v = velocity (normalized to [0, 1])
        """
        if hasattr(raw_state, '__len__') and len(raw_state) >= 2:
            pos = raw_state[0]  # Position
            vel = raw_state[1]  # Velocity
        else:
            # Fallback: use environment state directly
            pos = self.env.seg_idx / max(self.env.n_segments, 1)
            vel = self.env.v / 120.0  # Normalize by max speed (km/h → fraction)
        
        # Ensure [0, 1] range
        pos_norm = np.clip(pos if pos <= 1.0 else pos / max(self.env.n_segments, 1), 0, 1)
        vel_norm = np.clip(vel if vel <= 1.0 else vel / 120.0, 0, 1)
        
        return np.array([pos_norm, vel_norm], dtype=np.float64)
    
    def _discretize_speed(self, velocity):
        """Discretize velocity for Q-table lookup."""
        v_max = 120.0  # km/h
        n_bins = 50
        bin_idx = int(np.clip(velocity / v_max * (n_bins - 1), 0, n_bins - 1))
        return bin_idx
    
    # =====================================================
    # Reward Function (Equation 45)
    # =====================================================
    
    def compute_reward(self, info, done):
        """Compute reward per paper's Equation 45.
        
        R = R_End     if reaching endpoint
        R = δ·ΔT + ρ·E   for forward progress (penalize time & energy)
        R = -C        for violations (backward, speed limit)
        """
        delta_coeff = 0.5   # Time penalty weight (δ)
        rho_coeff = 0.5     # Energy penalty weight (ρ)
        R_end = 100.0       # Completion bonus
        C_penalty = 10.0    # Violation penalty
        
        if done and info.get('completed', False):
            reward = R_end
        elif info.get('violation', False) or info.get('backward', False):
            reward = -C_penalty
        else:
            # Forward progress: penalize time and energy consumption
            dt = info.get('dt', 1.0)
            energy = info.get('energy_step', 0.0)
            progress = info.get('progress', 0.0)
            
            # Reward for forward progress, penalize time and energy
            reward = progress * 10.0 - delta_coeff * dt * 0.01 - rho_coeff * energy * 0.001
        
        # Scale reward to prevent gradient explosion
        return reward * self.reward_scale
    
    # =====================================================
    # Q-SARSA Hybrid (Equation 30 - CM-based switching)
    # =====================================================
    
    def _compute_cm(self, episode):
        """Compute Convergence Measurement."""
        if episode < 2:
            return 0.0
        
        delta_q = np.sum(np.abs(self.q_table - self.prev_q_table))
        self.prev_q_table = self.q_table.copy()
        
        if len(self.cm_history) > 0 and self.cm_history[-1] > 1e-10:
            cm = delta_q / self.cm_history[-1] if self.cm_history[-1] > 0 else 1.0
        else:
            cm = 1.0
        
        self.cm_history.append(delta_q)
        return cm
    
    def _qsarsa_update(self, seg, v_bin, action, reward, next_seg, next_v_bin, next_action, cm):
        """Q-SARSA hybrid update (Equation 30).
        
        If cm > φ: SARSA update (follow current policy)
        If cm ≤ φ: Q-learning update (explore optimal)
        """
        alpha = getattr(config, 'ALPHA', 0.5)
        
        if cm > self.phi:
            # SARSA: use next action's Q-value
            td_target = reward + self.gamma * self.q_table[next_seg, next_v_bin, next_action]
        else:
            # Q-learning: use max Q-value
            td_target = reward + self.gamma * np.max(self.q_table[next_seg, next_v_bin, :])
        
        td_error = td_target - self.q_table[seg, v_bin, action]
        self.q_table[seg, v_bin, action] += alpha * td_error
    
    # =====================================================
    # Experience Replay
    # =====================================================
    
    def _store_experience(self, state, action, reward, next_state, done):
        """Store experience in replay buffer."""
        self.replay_buffer.append((state, action, reward, next_state, done))
        if len(self.replay_buffer) > self.buffer_size:
            self.replay_buffer.pop(0)
    
    def _sample_batch(self):
        """Sample random batch from replay buffer."""
        indices = np.random.choice(len(self.replay_buffer), self.batch_size, replace=False)
        return [self.replay_buffer[i] for i in indices]
    
    def _train_batch(self):
        """Train on a batch from replay buffer.
        
        Uses target network for stable Q-value estimates.
        """
        if len(self.replay_buffer) < self.min_replay:
            return 0.0
        
        batch = self._sample_batch()
        total_loss = 0.0
        
        for state, action, reward, next_state, done in batch:
            # Current Q-values
            current_q = self.predict(state)
            
            # Target Q-values (from target network)
            if done:
                target_value = reward
            else:
                next_q = self.predict(next_state, use_target=True)
                target_value = reward + self.gamma * np.max(next_q)
            
            # Clip TD error
            td_error = target_value - current_q[action]
            td_error = np.clip(td_error, -self.td_clip, self.td_clip)
            
            # Build target Q-values
            target_q = current_q.copy()
            target_q[action] = current_q[action] + td_error
            
            # Update network
            loss = self._backward(state, target_q)
            total_loss += loss
        
        # Soft update target network
        self._soft_update_target()
        
        return total_loss / self.batch_size
    
    # =====================================================
    # Training (Three-Phase per Paper Section 3.6)
    # =====================================================
    
    def train(self, episodes=5000):
        """Three-phase training per paper methodology.
        
        Phase 1 (40%): Q-SARSA data generation + Q-table training
        Phase 2 (10%): Supervised network training on collected data
        Phase 3 (50%): Online DQN fine-tuning with experience replay
        """
        phase1_eps = int(episodes * 0.4)
        phase2_batches = 100  # Supervised training iterations
        phase3_eps = episodes - phase1_eps
        
        print(f"\n{'='*70}")
        print(f"PHASE 1: Q-SARSA Data Generation ({phase1_eps} episodes)")
        print(f"{'='*70}")
        
        self._phase1_qsarsa(phase1_eps)
        
        print(f"\n{'='*70}")
        print(f"PHASE 2: Supervised Network Training ({phase2_batches} batches)")
        print(f"{'='*70}")
        
        self._phase2_supervised(phase2_batches)
        
        print(f"\n{'='*70}")
        print(f"PHASE 3: Online DQN Fine-tuning ({phase3_eps} episodes)")
        print(f"{'='*70}")
        
        self._phase3_online(phase3_eps, start_episode=phase1_eps)
        
        # Save results
        self._save_results()
    
    def _phase1_qsarsa(self, episodes):
        """Phase 1: Generate training data using Q-SARSA."""
        successes = 0
        
        for ep in range(episodes):
            self.env.reset()
            episode_data = []
            total_reward = 0
            done = False
            steps = 0
            
            # Compute CM for switching
            cm = self._compute_cm(ep)
            
            while not done and steps < getattr(config, 'MAX_STEPS_PER_EPISODE', 2000):
                # Get state
                raw_state = self.env._get_state()
                state = self.normalize_state(raw_state)
                seg = min(self.env.seg_idx, self.q_table.shape[0] - 1)
                v_bin = self._discretize_speed(self.env.v)
                
                # Epsilon-greedy from Q-table
                if np.random.random() < self.epsilon:
                    action = np.random.randint(self.n_actions)
                else:
                    action = np.argmax(self.q_table[seg, v_bin, :])
                
                # Step environment
                next_raw_state, env_reward, done, info = self.env.step(action)
                next_state = self.normalize_state(next_raw_state)
                
                # Compute paper reward
                reward = self.compute_reward(info, done)
                
                # Store for network training
                episode_data.append((state, action, reward))
                total_reward += reward
                
                # Q-SARSA update
                next_seg = min(self.env.seg_idx, self.q_table.shape[0] - 1)
                next_v_bin = self._discretize_speed(self.env.v)
                
                if np.random.random() < self.epsilon:
                    next_action = np.random.randint(self.n_actions)
                else:
                    next_action = np.argmax(self.q_table[next_seg, next_v_bin, :])
                
                self._qsarsa_update(seg, v_bin, action, reward, 
                                   next_seg, next_v_bin, next_action, cm)
                
                # Store in replay buffer too
                self._store_experience(state, action, reward, next_state, done)
                
                steps += 1
            
            # Track success
            completed = info.get('completed', False) if isinstance(info, dict) else False
            if not completed:
                # Check if we reached the end based on segment
                completed = self.env.seg_idx >= self.env.n_segments - 1
            
            if completed:
                successes += 1
            
            self.success_history.append(completed)
            self.energy_history.append(getattr(self.env, 'energy_kwh', 0))
            self.reward_history.append(total_reward)
            
            # Store episode training data with return
            G = total_reward
            for state, action, reward in episode_data:
                weight = abs(G / max(abs(reward), 1e-6))
                weight = min(weight, 100.0)  # Clamp weight
                self.training_data.append((state, action, weight, completed))
            
            # Decay epsilon
            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
            
            # Progress report
            if (ep + 1) % 100 == 0:
                recent_success = sum(self.success_history[-100:]) / min(100, len(self.success_history))
                recent_energy = np.mean([self.energy_history[i] for i in range(-min(100, len(self.energy_history)), 0) 
                                        if self.success_history[i]])  if any(self.success_history[-100:]) else 0
                print(f"  Phase 1 Ep {ep+1}/{episodes}: "
                      f"Success={recent_success:.0%}, "
                      f"Energy={recent_energy:.0f} kWh, "
                      f"ε={self.epsilon:.3f}, "
                      f"Data={len(self.training_data)} samples")
        
        print(f"\n  Phase 1 Complete: {successes}/{episodes} successful "
              f"({successes/max(episodes,1):.0%}), "
              f"{len(self.training_data)} training samples collected")
    
    def _phase2_supervised(self, n_batches):
        """Phase 2: Train network on Q-SARSA collected data."""
        if not self.training_data:
            print("  No training data collected! Skipping Phase 2.")
            return
        
        # Filter to successful episodes' data (prioritize good trajectories)
        good_data = [d for d in self.training_data if d[3]]  # completed=True
        if len(good_data) < 50:
            good_data = self.training_data  # Use all if few successes
        
        print(f"  Training on {len(good_data)} samples from successful episodes")
        
        states = np.array([d[0] for d in good_data])
        actions = np.array([d[1] for d in good_data])
        weights = np.array([d[2] for d in good_data])
        
        # Normalize weights
        weights = weights / max(weights.max(), 1e-6)
        
        for batch_idx in range(n_batches):
            # Sample weighted batch
            probs = weights / weights.sum()
            indices = np.random.choice(len(good_data), min(self.batch_size, len(good_data)), 
                                      replace=True, p=probs)
            
            batch_loss = 0.0
            for idx in indices:
                state = states[idx]
                action = actions[idx]
                weight = weights[idx]
                
                # Create target: boost chosen action's probability
                current_q = self.predict(state)
                target_q = current_q.copy()
                
                # Increase Q-value for the chosen action proportional to weight
                target_q[action] = current_q[action] + weight * 0.1
                
                loss = self._backward(state, target_q, learning_rate=self.lr * 0.5)
                batch_loss += loss
            
            if (batch_idx + 1) % 20 == 0:
                avg_loss = batch_loss / len(indices)
                print(f"  Batch {batch_idx+1}/{n_batches}: Loss={avg_loss:.6f}")
        
        # Sync target network after supervised training
        self.target_weights = self._deep_copy_weights(self.weights)
        print("  Phase 2 Complete: Network trained on Q-SARSA data")
    
    def _phase3_online(self, episodes, start_episode=0):
        """Phase 3: Online DQN fine-tuning with experience replay."""
        successes = 0
        
        for ep in range(episodes):
            self.env.reset()
            total_reward = 0
            total_loss = 0
            done = False
            steps = 0
            n_updates = 0
            
            while not done and steps < getattr(config, 'MAX_STEPS_PER_EPISODE', 2000):
                # Get state
                raw_state = self.env._get_state()
                state = self.normalize_state(raw_state)
                
                # Epsilon-greedy from network
                if np.random.random() < self.epsilon:
                    action = np.random.randint(self.n_actions)
                else:
                    q_values = self.predict(state)
                    action = np.argmax(q_values)
                
                # Step
                next_raw_state, env_reward, done, info = self.env.step(action)
                next_state = self.normalize_state(next_raw_state)
                reward = self.compute_reward(info, done)
                
                total_reward += reward
                
                # Store experience
                self._store_experience(state, action, reward, next_state, done)
                
                # Train on batch
                if len(self.replay_buffer) >= self.min_replay and steps % 4 == 0:
                    loss = self._train_batch()
                    total_loss += loss
                    n_updates += 1
                
                steps += 1
            
            # Track success
            completed = info.get('completed', False) if isinstance(info, dict) else False
            if not completed:
                completed = self.env.seg_idx >= self.env.n_segments - 1
            
            if completed:
                successes += 1
            
            self.success_history.append(completed)
            self.energy_history.append(getattr(self.env, 'energy_kwh', 0))
            self.time_history.append(steps)
            avg_loss = total_loss / max(n_updates, 1)
            self.loss_history.append(avg_loss)
            self.reward_history.append(total_reward)
            
            # Decay epsilon
            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
            
            # Progress report
            global_ep = start_episode + ep + 1
            if (ep + 1) % 100 == 0:
                recent = self.success_history[-(ep+1):][-100:]
                recent_success = sum(recent) / len(recent)
                recent_energy_vals = [self.energy_history[-(ep+1)+i] 
                                     for i in range(max(0, len(recent)-100), len(recent))
                                     if self.success_history[-(ep+1)+i]]
                recent_energy = np.mean(recent_energy_vals) if recent_energy_vals else 0
                print(f"  Phase 3 Ep {ep+1}/{episodes} (Global {global_ep}): "
                      f"Success={recent_success:.0%}, "
                      f"Energy={recent_energy:.0f} kWh, "
                      f"Loss={avg_loss:.6f}, "
                      f"ε={self.epsilon:.3f}")
        
        total_success = sum(self.success_history)
        total_eps = len(self.success_history)
        print(f"\n  Phase 3 Complete: {successes}/{episodes} successful")
        print(f"  Overall: {total_success}/{total_eps} ({total_success/max(total_eps,1):.0%})")
    
    # =====================================================
    # Results & Visualization
    # =====================================================
    
    def _save_results(self):
        """Save training results and plots."""
        output_dir = os.path.join(getattr(config, 'OUTPUT_DIR', 'results_cm'), "deep_q")
        os.makedirs(output_dir, exist_ok=True)
        
        # Save weights and history
        save_dict = {
            'success_history': np.array(self.success_history),
            'energy_history': np.array(self.energy_history),
            'reward_history': np.array(self.reward_history),
            'loss_history': np.array(self.loss_history),
        }
        # Save weights
        for key, val in self.weights.items():
            save_dict[f'weight_{key}'] = val
        
        np.savez(os.path.join(output_dir, "dqn_weights.npz"), **save_dict)
        
        # Plot training curves
        try:
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
            
            fig, axes = plt.subplots(2, 2, figsize=(14, 10))
            fig.suptitle('DQN Training Results (Paper-Compliant)', fontsize=14)
            
            episodes = range(len(self.success_history))
            
            # Success rate (rolling 100)
            if len(self.success_history) > 100:
                rolling_success = [
                    sum(self.success_history[max(0,i-100):i+1]) / min(i+1, 100)
                    for i in range(len(self.success_history))
                ]
                axes[0, 0].plot(episodes, rolling_success, 'b-', linewidth=1)
            else:
                axes[0, 0].plot(episodes, [float(s) for s in self.success_history], 'b.', markersize=2)
            axes[0, 0].set_title('Success Rate (Rolling 100)')
            axes[0, 0].set_xlabel('Episode')
            axes[0, 0].set_ylabel('Success Rate')
            axes[0, 0].set_ylim(-0.05, 1.05)
            axes[0, 0].grid(True, alpha=0.3)
            
            # Loss
            if self.loss_history:
                axes[0, 1].plot(range(len(self.loss_history)), self.loss_history, 'r-', 
                               alpha=0.5, linewidth=0.5)
                axes[0, 1].set_title('Training Loss')
                axes[0, 1].set_xlabel('Episode (Phase 3)')
                axes[0, 1].set_ylabel('Loss')
                axes[0, 1].set_yscale('log')
                axes[0, 1].grid(True, alpha=0.3)
            
            # Energy (successful episodes only)
            successful_eps = [i for i, s in enumerate(self.success_history) if s]
            if successful_eps:
                successful_energy = [self.energy_history[i] for i in successful_eps]
                axes[1, 0].plot(successful_eps, successful_energy, 'g.', alpha=0.5, markersize=2)
                axes[1, 0].set_title('Energy (Successful Episodes)')
                axes[1, 0].set_xlabel('Episode')
                axes[1, 0].set_ylabel('Energy (kWh)')
                axes[1, 0].grid(True, alpha=0.3)
            
            # Reward
            axes[1, 1].plot(episodes, self.reward_history, 'm-', alpha=0.3, linewidth=0.5)
            if len(self.reward_history) > 100:
                rolling_reward = [
                    np.mean(self.reward_history[max(0,i-100):i+1])
                    for i in range(len(self.reward_history))
                ]
                axes[1, 1].plot(episodes, rolling_reward, 'm-', linewidth=2, label='Rolling avg')
                axes[1, 1].legend()
            axes[1, 1].set_title('Total Reward per Episode')
            axes[1, 1].set_xlabel('Episode')
            axes[1, 1].set_ylabel('Reward')
            axes[1, 1].grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, "dqn_training.png"), dpi=300)
            plt.close()
            print(f"\n✓ Training plots saved to: {output_dir}/dqn_training.png")
        except ImportError:
            print("  (matplotlib not available - skipping plots)")
        
        print(f"✓ Results saved to: {output_dir}/")
    
    def generate_speed_profile(self):
        """Generate optimal speed profile using trained network."""
        print("\n📊 Generating optimal speed profile (DQN)...")
        
        self.env.reset()
        
        segments = []
        velocities = []
        actions_taken = []
        energies = []
        
        steps = 0
        while steps < getattr(config, 'MAX_STEPS_PER_EPISODE', 2000):
            raw_state = self.env._get_state()
            state = self.normalize_state(raw_state)
            
            # Greedy action from network
            q_values = self.predict(state)
            action = np.argmax(q_values)
            
            segments.append(self.env.seg_idx)
            velocities.append(self.env.v)
            actions_taken.append(action)
            energies.append(getattr(self.env, 'energy_kwh', 0))
            
            _, _, done, info = self.env.step(action)
            steps += 1
            
            if done or self.env.v < 0.1:
                break
        
        # Save speed profile
        output_dir = os.path.join(getattr(config, 'OUTPUT_DIR', 'results_cm'), "deep_q")
        os.makedirs(output_dir, exist_ok=True)
        
        np.savez(os.path.join(output_dir, "speed_profile.npz"),
                 segments=np.array(segments),
                 velocities=np.array(velocities),
                 actions=np.array(actions_taken),
                 energies=np.array(energies))
        
        final_energy = energies[-1] if energies else 0
        final_seg = segments[-1] if segments else 0
        n_segs = getattr(self.env, 'n_segments', 749)
        completed = final_seg >= n_segs - 1
        
        print(f"✓ Speed profile saved")
        print(f"  Final segment: {final_seg}/{n_segs} {'✓ COMPLETED' if completed else '✗ INCOMPLETE'}")
        print(f"  Final energy: {final_energy:.1f} kWh")
        print(f"  Steps taken: {steps}")
        
        # Plot speed profile
        try:
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
            
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
            fig.suptitle('DQN Optimal Speed Profile', fontsize=14)
            
            positions = np.array(segments) * 0.1  # Convert to km (100m segments)
            
            ax1.plot(positions, velocities, 'b-', linewidth=1.5)
            ax1.set_ylabel('Speed (km/h)')
            ax1.set_title('Speed Profile')
            ax1.grid(True, alpha=0.3)
            
            ax2.plot(positions, energies, 'r-', linewidth=1.5)
            ax2.set_xlabel('Position (km)')
            ax2.set_ylabel('Energy (kWh)')
            ax2.set_title('Cumulative Energy')
            ax2.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, "speed_profile.png"), dpi=300)
            plt.close()
            print(f"✓ Speed profile plot saved")
        except ImportError:
            pass
        
        return segments, velocities, actions_taken, energies