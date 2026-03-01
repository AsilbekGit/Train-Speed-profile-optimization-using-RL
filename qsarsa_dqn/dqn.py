"""
Deep Q-Network - PARALLELIZED VERSION
=======================================
Optimized for multi-core systems (DGX Spark 20 cores)

Key optimizations:
1. VECTORIZED batch training - process all 64 samples in one matrix multiply
   (was: 64 individual forward+backward passes → now: 1 batched pass)
2. PARALLEL episode collection - multiprocessing across N cores
3. Reduced training frequency for better throughput
4. NumPy BLAS multi-threading support

Architecture (Figure 10):
    Input (x, v) → 128 tanh → 64 tanh → 16 tanh → 4 sigmoid → action probabilities
"""

import numpy as np
import os
import sys
import time
from multiprocessing import Pool, cpu_count

# Import project modules
try:
    import env_settings.config 
except ImportError:
    class config:
        OUTPUT_DIR = "results_cm"
        MAX_STEPS_PER_EPISODE = 2000
        N_ACTIONS = 4
        GAMMA = 0.99
        ALPHA = 0.5
        EPSILON_START = 1.0
        EPSILON_MIN = 0.01
        EPSILON_DECAY = 0.999


# =========================================================
# Worker function for parallel episode collection
# =========================================================

def _run_episode_worker(args):
    """
    Run a single episode in a separate process.
    Returns collected experiences and episode stats.
    """
    (weights, env_args, epsilon, n_actions, gamma, reward_scale, 
     max_steps, hidden_sizes, n_inputs) = args
    
    # Reconstruct environment in this process
    try:
        # Try different import paths
        for phys_mod, env_mod in [('env_settings.physics', 'env_settings.environment'),
                                   ('physics', 'environment')]:
            try:
                phys_module = __import__(phys_mod, fromlist=['TrainPhysics'])
                env_module = __import__(env_mod, fromlist=['TrainEnv'])
                TrainPhysics = phys_module.TrainPhysics
                TrainEnv = env_module.TrainEnv
                break
            except ImportError:
                continue
        else:
            # Try config
            import env_settings.config as cfg
            TrainPhysics = cfg.TrainPhysics
            TrainEnv = cfg.TrainEnv
        
        grades, limits, curves = env_args
        physics = TrainPhysics()
        env = TrainEnv(physics, grades, limits, curves)
    except Exception as e:
        return None
    
    # Build lightweight forward function
    def predict(state, w):
        x = np.array(state, dtype=np.float64).reshape(1, -1)
        n_layers = len(hidden_sizes) + 1
        for i in range(n_layers):
            x = x @ w[f'W{i}'] + w[f'b{i}']
            if i < n_layers - 1:
                x = np.tanh(np.clip(x, -20, 20))
            else:
                x = 1.0 / (1.0 + np.exp(-np.clip(x, -20, 20)))
        return x.flatten()
    
    def normalize_state(env):
        pos = env.seg_idx / max(env.n_segments, 1)
        vel = env.v / 120.0
        return np.array([np.clip(pos, 0, 1), np.clip(vel, 0, 1)], dtype=np.float64)
    
    def compute_reward(info, done, env):
        if done and (info.get('completed', False) or env.seg_idx >= env.n_segments - 1):
            return 100.0 * reward_scale
        elif info.get('violation', False) or info.get('backward', False):
            return -10.0 * reward_scale
        else:
            dt = info.get('dt', 1.0)
            energy = info.get('energy_step', 0.0)
            progress = info.get('progress', 0.0)
            return (progress * 10.0 - 0.5 * dt * 0.01 - 0.5 * energy * 0.001) * reward_scale
    
    # Run episode
    env.reset()
    experiences = []
    total_reward = 0
    done = False
    steps = 0
    
    while not done and steps < max_steps:
        state = normalize_state(env)
        
        if np.random.random() < epsilon:
            action = np.random.randint(n_actions)
        else:
            q_values = predict(state, weights)
            action = np.argmax(q_values)
        
        _, env_reward, done, info = env.step(action)
        next_state = normalize_state(env)
        reward = compute_reward(info, done, env)
        total_reward += reward
        
        experiences.append((state, action, reward, next_state, done))
        steps += 1
    
    completed = (info.get('completed', False) if isinstance(info, dict) else False) or \
                env.seg_idx >= env.n_segments - 1
    energy = getattr(env, 'energy_kwh', 0)
    
    return {
        'experiences': experiences,
        'completed': completed,
        'energy': energy,
        'reward': total_reward,
        'steps': steps,
    }


class DeepQNetwork:
    """
    Deep Q-Network with VECTORIZED training and PARALLEL episode collection.
    """
    
    def __init__(self, env, phi_threshold=0.10, n_workers=None):
        self.env = env
        self.phi = phi_threshold
        
        # Detect available cores
        self.n_workers = n_workers or min(cpu_count(), 20)
        
        # Network architecture per Figure 10
        self.n_inputs = 2
        self.n_actions = getattr(config, 'N_ACTIONS', 4)
        self.hidden_sizes = [128, 64, 16]
        
        # Hyperparameters
        self.gamma = getattr(config, 'GAMMA', 0.99)
        self.lr = 0.0005
        self.epsilon = getattr(config, 'EPSILON_START', 1.0)
        self.epsilon_min = getattr(config, 'EPSILON_MIN', 0.01)
        self.epsilon_decay = getattr(config, 'EPSILON_DECAY', 0.999)
        self.reward_scale = 0.01
        self.grad_clip = 1.0
        self.td_clip = 10.0
        self.tau = 0.1
        
        # Experience replay
        self.replay_buffer = []
        self.buffer_size = 50000  # Larger buffer for parallel collection
        self.batch_size = 128     # Larger batch for vectorized training
        self.min_replay = 500
        
        # Initialize networks
        self.weights = self._init_weights()
        self.target_weights = self._deep_copy_weights(self.weights)
        
        # Q-table for Phase 1
        n_segments = getattr(env, 'n_segments', 749)
        n_speeds = 50
        self.q_table = np.zeros((n_segments, n_speeds, self.n_actions))
        self.q_table[:, :, 0] = 2.0
        self.q_table[:, :, 1] = 1.0
        self.q_table[:, :, 2] = 0.0
        self.q_table[:, :, 3] = -1.0
        self.prev_q_table = self.q_table.copy()
        
        self.cm_history = []
        
        # Training history
        self.success_history = []
        self.energy_history = []
        self.time_history = []
        self.loss_history = []
        self.reward_history = []
        self.training_data = []
        
        # Store env constructor args for workers
        self._env_args = (
            getattr(env, '_grades', None) or np.zeros(n_segments),
            getattr(env, '_limits', None) or np.full(n_segments, 120.0),
            getattr(env, '_curves', None) or np.zeros(n_segments),
        )
        # Try to get actual data from env
        self._try_extract_env_data()
        
        print(f"DQN initialized (PARALLELIZED):")
        print(f"  Architecture: {self.n_inputs} → {' → '.join(map(str, self.hidden_sizes))} → {self.n_actions}")
        print(f"  Activations: tanh (hidden) + sigmoid (output)")
        print(f"  Learning rate: {self.lr}")
        print(f"  Gradient clip: ±{self.grad_clip}")
        print(f"  Reward scale: ×{self.reward_scale}")
        print(f"  Target network: soft update τ={self.tau}")
        print(f"  Batch size: {self.batch_size} (vectorized)")
        print(f"  Workers: {self.n_workers} cores")
        print(f"  φ threshold: {self.phi}")
    
    def _try_extract_env_data(self):
        """Try to extract route data from the environment for workers."""
        env = self.env
        # Try common attribute names
        for g_attr in ['grades', '_grades', 'grade_data', 'track_grades']:
            if hasattr(env, g_attr):
                grades = getattr(env, g_attr)
                if isinstance(grades, np.ndarray) and len(grades) > 0:
                    self._env_args = (grades, self._env_args[1], self._env_args[2])
                    break
        
        for l_attr in ['limits', '_limits', 'speed_limits', 'track_limits']:
            if hasattr(env, l_attr):
                limits = getattr(env, l_attr)
                if isinstance(limits, np.ndarray) and len(limits) > 0:
                    self._env_args = (self._env_args[0], limits, self._env_args[2])
                    break
        
        for c_attr in ['curves', '_curves', 'curvatures', 'track_curves']:
            if hasattr(env, c_attr):
                curves = getattr(env, c_attr)
                if isinstance(curves, np.ndarray) and len(curves) > 0:
                    self._env_args = (self._env_args[0], self._env_args[1], curves)
                    break
    
    # =====================================================
    # Neural Network (NumPy - VECTORIZED)
    # =====================================================
    
    def _init_weights(self):
        weights = {}
        layer_sizes = [self.n_inputs] + self.hidden_sizes + [self.n_actions]
        for i in range(len(layer_sizes) - 1):
            fan_in = layer_sizes[i]
            fan_out = layer_sizes[i + 1]
            std = np.sqrt(2.0 / fan_in)
            weights[f'W{i}'] = np.random.randn(fan_in, fan_out) * std
            weights[f'b{i}'] = np.zeros(fan_out)
        last_idx = len(layer_sizes) - 2
        weights[f'b{last_idx}'] = np.array([2.0, 1.0, 0.0, -1.0])
        return weights
    
    def _deep_copy_weights(self, weights):
        return {k: v.copy() for k, v in weights.items()}
    
    def _tanh(self, x):
        return np.tanh(np.clip(x, -20, 20))
    
    def _sigmoid(self, x):
        x = np.clip(x, -20, 20)
        return 1.0 / (1.0 + np.exp(-x))
    
    def _tanh_deriv(self, tanh_output):
        return 1.0 - tanh_output ** 2
    
    def _sigmoid_deriv(self, sigmoid_output):
        return sigmoid_output * (1.0 - sigmoid_output)
    
    def _forward_batch(self, X, weights=None):
        """
        VECTORIZED forward pass for batch of inputs.
        X: (batch_size, n_inputs)
        Returns all activations for backprop.
        """
        if weights is None:
            weights = self.weights
        
        n_layers = len(self.hidden_sizes) + 1
        activations = [X]
        
        for i in range(n_layers):
            Z = activations[-1] @ weights[f'W{i}'] + weights[f'b{i}']
            if i < n_layers - 1:
                A = self._tanh(Z)
            else:
                A = self._sigmoid(Z)
            activations.append(A)
        
        return activations
    
    def predict(self, state, use_target=False):
        """Predict Q-values for a single state."""
        weights = self.target_weights if use_target else self.weights
        x = np.array(state, dtype=np.float64).reshape(1, -1)
        activations = self._forward_batch(x, weights)
        return activations[-1].flatten()
    
    def predict_batch(self, states, use_target=False):
        """Predict Q-values for a batch of states."""
        weights = self.target_weights if use_target else self.weights
        X = np.array(states, dtype=np.float64)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        activations = self._forward_batch(X, weights)
        return activations[-1]
    
    def _backward_batch(self, states, targets, learning_rate=None):
        """
        VECTORIZED backward pass for a batch.
        states: (batch_size, n_inputs)
        targets: (batch_size, n_actions)
        
        Processes ALL samples in ONE pass using matrix operations.
        This is ~50-100x faster than looping over individual samples.
        """
        if learning_rate is None:
            learning_rate = self.lr
        
        batch_size = states.shape[0]
        
        # Forward pass (batched)
        activations = self._forward_batch(states)
        output = activations[-1]
        
        # Huber loss
        error = output - targets
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
        
        d_output = np.clip(d_output, -self.grad_clip, self.grad_clip)
        
        # Backpropagation (batched)
        n_layers = len(self.hidden_sizes) + 1
        delta = d_output * self._sigmoid_deriv(activations[-1])
        
        for i in range(n_layers - 1, -1, -1):
            # Batched gradient: average over batch
            dW = (activations[i].T @ delta) / batch_size
            db = delta.mean(axis=0)
            
            # Clip
            dW = np.clip(dW, -self.grad_clip, self.grad_clip)
            db = np.clip(db, -self.grad_clip, self.grad_clip)
            
            # Update
            self.weights[f'W{i}'] -= learning_rate * dW
            self.weights[f'b{i}'] -= learning_rate * db
            
            if i > 0:
                delta = (delta @ self.weights[f'W{i}'].T) * self._tanh_deriv(activations[i])
                delta = np.clip(delta, -self.grad_clip, self.grad_clip)
        
        return loss
    
    def _soft_update_target(self):
        for key in self.weights:
            self.target_weights[key] = (
                self.tau * self.weights[key] + 
                (1 - self.tau) * self.target_weights[key]
            )
    
    # =====================================================
    # State Processing
    # =====================================================
    
    def normalize_state(self, raw_state):
        if hasattr(raw_state, '__len__') and len(raw_state) >= 2:
            pos = raw_state[0]
            vel = raw_state[1]
        else:
            pos = self.env.seg_idx / max(self.env.n_segments, 1)
            vel = self.env.v / 120.0
        
        pos_norm = np.clip(pos if pos <= 1.0 else pos / max(self.env.n_segments, 1), 0, 1)
        vel_norm = np.clip(vel if vel <= 1.0 else vel / 120.0, 0, 1)
        return np.array([pos_norm, vel_norm], dtype=np.float64)
    
    def _discretize_speed(self, velocity):
        v_max = 120.0
        n_bins = 50
        return int(np.clip(velocity / v_max * (n_bins - 1), 0, n_bins - 1))
    
    # =====================================================
    # Reward Function (Equation 45)
    # =====================================================
    
    def compute_reward(self, info, done):
        delta_coeff = 0.5
        rho_coeff = 0.5
        R_end = 100.0
        C_penalty = 10.0
        
        if done and info.get('completed', False):
            reward = R_end
        elif info.get('violation', False) or info.get('backward', False):
            reward = -C_penalty
        else:
            dt = info.get('dt', 1.0)
            energy = info.get('energy_step', 0.0)
            progress = info.get('progress', 0.0)
            reward = progress * 10.0 - delta_coeff * dt * 0.01 - rho_coeff * energy * 0.001
        
        return reward * self.reward_scale
    
    # =====================================================
    # Q-SARSA Hybrid (Equation 30)
    # =====================================================
    
    def _compute_cm(self, episode):
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
        alpha = getattr(config, 'ALPHA', 0.5)
        if cm > self.phi:
            td_target = reward + self.gamma * self.q_table[next_seg, next_v_bin, next_action]
        else:
            td_target = reward + self.gamma * np.max(self.q_table[next_seg, next_v_bin, :])
        td_error = td_target - self.q_table[seg, v_bin, action]
        self.q_table[seg, v_bin, action] += alpha * td_error
    
    # =====================================================
    # VECTORIZED Experience Replay Training
    # =====================================================
    
    def _store_experience(self, state, action, reward, next_state, done):
        self.replay_buffer.append((state, action, reward, next_state, done))
        if len(self.replay_buffer) > self.buffer_size:
            self.replay_buffer.pop(0)
    
    def _store_experiences_bulk(self, experiences):
        """Store a list of experiences at once."""
        self.replay_buffer.extend(experiences)
        # Trim to buffer size
        if len(self.replay_buffer) > self.buffer_size:
            self.replay_buffer = self.replay_buffer[-self.buffer_size:]
    
    def _train_batch_vectorized(self):
        """
        VECTORIZED batch training - processes all samples in ONE matrix multiply.
        ~50-100x faster than the old loop-based approach.
        """
        if len(self.replay_buffer) < self.min_replay:
            return 0.0
        
        # Sample batch
        indices = np.random.choice(len(self.replay_buffer), self.batch_size, replace=False)
        batch = [self.replay_buffer[i] for i in indices]
        
        # Unpack into arrays (vectorized)
        states = np.array([b[0] for b in batch], dtype=np.float64)
        actions = np.array([b[1] for b in batch], dtype=np.int32)
        rewards = np.array([b[2] for b in batch], dtype=np.float64)
        next_states = np.array([b[3] for b in batch], dtype=np.float64)
        dones = np.array([b[4] for b in batch], dtype=bool)
        
        # Batch predict current Q-values
        current_q = self.predict_batch(states)
        
        # Batch predict next Q-values (from target network)
        next_q = self.predict_batch(next_states, use_target=True)
        
        # Compute targets
        target_values = rewards.copy()
        target_values[~dones] += self.gamma * np.max(next_q[~dones], axis=1)
        
        # Build target Q-values
        target_q = current_q.copy()
        for i in range(self.batch_size):
            td_error = target_values[i] - current_q[i, actions[i]]
            td_error = np.clip(td_error, -self.td_clip, self.td_clip)
            target_q[i, actions[i]] = current_q[i, actions[i]] + td_error
        
        # ONE backward pass for entire batch
        loss = self._backward_batch(states, target_q)
        
        # Soft update target
        self._soft_update_target()
        
        return loss
    
    # =====================================================
    # Training (Three-Phase)
    # =====================================================
    
    def train(self, episodes=5000):
        phase1_eps = int(episodes * 0.4)
        phase2_batches = 100
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
        print(f"PHASE 3: Online DQN Fine-tuning ({phase3_eps} episodes, {self.n_workers} workers)")
        print(f"{'='*70}")
        self._phase3_parallel(phase3_eps, start_episode=phase1_eps)
        
        self._save_results()
    
    def _phase1_qsarsa(self, episodes):
        """Phase 1: Generate training data using Q-SARSA."""
        successes = 0
        t_start = time.time()
        
        for ep in range(episodes):
            self.env.reset()
            episode_data = []
            total_reward = 0
            done = False
            steps = 0
            cm = self._compute_cm(ep)
            
            while not done and steps < getattr(config, 'MAX_STEPS_PER_EPISODE', 2000):
                raw_state = self.env._get_state()
                state = self.normalize_state(raw_state)
                seg = min(self.env.seg_idx, self.q_table.shape[0] - 1)
                v_bin = self._discretize_speed(self.env.v)
                
                if np.random.random() < self.epsilon:
                    action = np.random.randint(self.n_actions)
                else:
                    action = np.argmax(self.q_table[seg, v_bin, :])
                
                next_raw_state, env_reward, done, info = self.env.step(action)
                next_state = self.normalize_state(next_raw_state)
                reward = self.compute_reward(info, done)
                
                episode_data.append((state, action, reward))
                total_reward += reward
                
                next_seg = min(self.env.seg_idx, self.q_table.shape[0] - 1)
                next_v_bin = self._discretize_speed(self.env.v)
                
                if np.random.random() < self.epsilon:
                    next_action = np.random.randint(self.n_actions)
                else:
                    next_action = np.argmax(self.q_table[next_seg, next_v_bin, :])
                
                self._qsarsa_update(seg, v_bin, action, reward, 
                                   next_seg, next_v_bin, next_action, cm)
                
                self._store_experience(state, action, reward, next_state, done)
                steps += 1
            
            completed = info.get('completed', False) if isinstance(info, dict) else False
            if not completed:
                completed = self.env.seg_idx >= self.env.n_segments - 1
            if completed:
                successes += 1
            
            self.success_history.append(completed)
            self.energy_history.append(getattr(self.env, 'energy_kwh', 0))
            self.reward_history.append(total_reward)
            
            G = total_reward
            for state, action, reward in episode_data:
                weight = abs(G / max(abs(reward), 1e-6))
                weight = min(weight, 100.0)
                self.training_data.append((state, action, weight, completed))
            
            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
            
            if (ep + 1) % 100 == 0:
                elapsed = time.time() - t_start
                eps_per_sec = (ep + 1) / elapsed
                eta = (episodes - ep - 1) / eps_per_sec
                recent_success = sum(self.success_history[-100:]) / min(100, len(self.success_history))
                recent_energy = np.mean([self.energy_history[i] for i in range(-min(100, len(self.energy_history)), 0) 
                                        if self.success_history[i]]) if any(self.success_history[-100:]) else 0
                print(f"  Phase 1 Ep {ep+1}/{episodes}: "
                      f"Success={recent_success:.0%}, "
                      f"Energy={recent_energy:.0f} kWh, "
                      f"ε={self.epsilon:.3f}, "
                      f"Speed={eps_per_sec:.1f} ep/s, "
                      f"ETA={eta:.0f}s")
        
        print(f"\n  Phase 1 Complete: {successes}/{episodes} successful "
              f"({successes/max(episodes,1):.0%}), "
              f"{len(self.training_data)} training samples, "
              f"{time.time()-t_start:.0f}s")
    
    def _phase2_supervised(self, n_batches):
        """Phase 2: Train network on Q-SARSA collected data (VECTORIZED)."""
        if not self.training_data:
            print("  No training data! Skipping Phase 2.")
            return
        
        good_data = [d for d in self.training_data if d[3]]
        if len(good_data) < 50:
            good_data = self.training_data
        
        print(f"  Training on {len(good_data)} samples from successful episodes")
        
        states = np.array([d[0] for d in good_data])
        actions = np.array([d[1] for d in good_data])
        weights = np.array([d[2] for d in good_data])
        weights = weights / max(weights.max(), 1e-6)
        
        for batch_idx in range(n_batches):
            probs = weights / weights.sum()
            indices = np.random.choice(len(good_data), min(self.batch_size, len(good_data)), 
                                      replace=True, p=probs)
            
            batch_states = states[indices]
            batch_actions = actions[indices]
            batch_weights = weights[indices]
            
            # Vectorized: predict all at once
            current_q = self.predict_batch(batch_states)
            target_q = current_q.copy()
            
            for i in range(len(indices)):
                target_q[i, batch_actions[i]] += batch_weights[i] * 0.1
            
            loss = self._backward_batch(batch_states, target_q, learning_rate=self.lr * 0.5)
            
            if (batch_idx + 1) % 20 == 0:
                print(f"  Batch {batch_idx+1}/{n_batches}: Loss={loss:.6f}")
        
        self.target_weights = self._deep_copy_weights(self.weights)
        print("  Phase 2 Complete: Network trained on Q-SARSA data")
    
    def _phase3_parallel(self, episodes, start_episode=0):
        """
        Phase 3: PARALLEL online DQN fine-tuning.
        
        Runs N episodes in parallel across workers, collects experiences,
        then trains on the main process with vectorized batch updates.
        """
        successes = 0
        t_start = time.time()
        
        # How many episodes to run in each parallel batch
        parallel_batch = self.n_workers
        n_rounds = episodes // parallel_batch
        remainder = episodes % parallel_batch
        
        max_steps = getattr(config, 'MAX_STEPS_PER_EPISODE', 2000)
        ep_counter = 0
        
        print(f"  Running {n_rounds} rounds × {parallel_batch} parallel episodes")
        print(f"  + {remainder} remaining episodes")
        
        # Check if multiprocessing works by testing with a small batch
        use_multiprocessing = True
        try:
            test_args = [(
                self._deep_copy_weights(self.weights),
                self._env_args,
                1.0,  # full exploration for test
                self.n_actions,
                self.gamma,
                self.reward_scale,
                10,  # very few steps
                self.hidden_sizes,
                self.n_inputs,
            )]
            with Pool(1) as pool:
                test_result = pool.map(_run_episode_worker, test_args, chunksize=1)
            if test_result[0] is None:
                raise RuntimeError("Worker returned None")
            print(f"  ✓ Multiprocessing test passed")
        except Exception as e:
            print(f"  ⚠ Multiprocessing failed ({e}), falling back to sequential + vectorized batches")
            use_multiprocessing = False
        
        if use_multiprocessing:
            self._phase3_multiprocess(n_rounds, parallel_batch, remainder, 
                                     max_steps, start_episode, t_start)
        else:
            self._phase3_sequential_fast(episodes, start_episode, t_start)
    
    def _phase3_multiprocess(self, n_rounds, parallel_batch, remainder, 
                             max_steps, start_episode, t_start):
        """Phase 3 with multiprocessing."""
        successes = 0
        ep_counter = 0
        total_rounds = n_rounds + (1 if remainder > 0 else 0)
        
        for round_idx in range(total_rounds):
            n_eps = parallel_batch if round_idx < n_rounds else remainder
            if n_eps == 0:
                continue
            
            # Prepare worker arguments
            worker_args = []
            for _ in range(n_eps):
                worker_args.append((
                    self._deep_copy_weights(self.weights),
                    self._env_args,
                    self.epsilon,
                    self.n_actions,
                    self.gamma,
                    self.reward_scale,
                    max_steps,
                    self.hidden_sizes,
                    self.n_inputs,
                ))
            
            # Run episodes in parallel
            with Pool(min(n_eps, self.n_workers)) as pool:
                results = pool.map(_run_episode_worker, worker_args, chunksize=1)
            
            # Process results
            for result in results:
                if result is None:
                    self.success_history.append(False)
                    self.energy_history.append(0)
                    self.time_history.append(0)
                    self.reward_history.append(0)
                    self.loss_history.append(0)
                    ep_counter += 1
                    continue
                
                # Store experiences
                self._store_experiences_bulk(result['experiences'])
                
                if result['completed']:
                    successes += 1
                
                self.success_history.append(result['completed'])
                self.energy_history.append(result['energy'])
                self.time_history.append(result['steps'])
                self.reward_history.append(result['reward'])
                
                ep_counter += 1
                self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
            
            # Train on collected experiences (multiple batches)
            n_train_batches = max(1, n_eps * 2)  # 2 training batches per episode
            total_loss = 0
            for _ in range(n_train_batches):
                loss = self._train_batch_vectorized()
                total_loss += loss
            avg_loss = total_loss / max(n_train_batches, 1)
            self.loss_history.extend([avg_loss] * n_eps)
            
            # Progress report
            if (round_idx + 1) % max(1, total_rounds // 30) == 0 or round_idx == total_rounds - 1:
                elapsed = time.time() - t_start
                eps_per_sec = ep_counter / max(elapsed, 1)
                total_eps = start_episode + ep_counter
                remaining = (n_rounds * parallel_batch + remainder) - ep_counter
                eta = remaining / max(eps_per_sec, 0.01)
                
                recent_n = min(100, len(self.success_history))
                recent_success = sum(self.success_history[-recent_n:]) / recent_n
                recent_energy_vals = [self.energy_history[-recent_n+i] 
                                     for i in range(recent_n)
                                     if self.success_history[-recent_n+i]]
                recent_energy = np.mean(recent_energy_vals) if recent_energy_vals else 0
                
                print(f"  Phase 3 Ep {ep_counter} (Global {total_eps}): "
                      f"Success={recent_success:.0%}, "
                      f"Energy={recent_energy:.0f} kWh, "
                      f"Loss={avg_loss:.6f}, "
                      f"ε={self.epsilon:.3f}, "
                      f"Speed={eps_per_sec:.1f} ep/s, "
                      f"ETA={eta:.0f}s")
        
        total_s = sum(self.success_history)
        total_e = len(self.success_history)
        print(f"\n  Phase 3 Complete: {successes} successful ({successes}/{ep_counter})")
        print(f"  Overall: {total_s}/{total_e} ({total_s/max(total_e,1):.0%})")
        print(f"  Total time: {time.time()-t_start:.0f}s")
    
    def _phase3_sequential_fast(self, episodes, start_episode, t_start):
        """Fallback: sequential but with vectorized batch training."""
        successes = 0
        
        for ep in range(episodes):
            self.env.reset()
            total_reward = 0
            total_loss = 0
            done = False
            steps = 0
            n_updates = 0
            
            while not done and steps < getattr(config, 'MAX_STEPS_PER_EPISODE', 2000):
                raw_state = self.env._get_state()
                state = self.normalize_state(raw_state)
                
                if np.random.random() < self.epsilon:
                    action = np.random.randint(self.n_actions)
                else:
                    q_values = self.predict(state)
                    action = np.argmax(q_values)
                
                next_raw_state, env_reward, done, info = self.env.step(action)
                next_state = self.normalize_state(next_raw_state)
                reward = self.compute_reward(info, done)
                total_reward += reward
                
                self._store_experience(state, action, reward, next_state, done)
                
                # Train less frequently but with vectorized batches
                if len(self.replay_buffer) >= self.min_replay and steps % 8 == 0:
                    loss = self._train_batch_vectorized()
                    total_loss += loss
                    n_updates += 1
                
                steps += 1
            
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
            
            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
            
            if (ep + 1) % 100 == 0:
                elapsed = time.time() - t_start
                eps_per_sec = (ep + 1) / elapsed
                eta = (episodes - ep - 1) / eps_per_sec
                recent = self.success_history[-100:]
                recent_success = sum(recent) / len(recent)
                recent_energy_vals = [self.energy_history[-100+i] 
                                     for i in range(len(recent)) if recent[i]]
                recent_energy = np.mean(recent_energy_vals) if recent_energy_vals else 0
                print(f"  Phase 3 Ep {ep+1}/{episodes}: "
                      f"Success={recent_success:.0%}, "
                      f"Energy={recent_energy:.0f} kWh, "
                      f"Loss={avg_loss:.6f}, "
                      f"ε={self.epsilon:.3f}, "
                      f"Speed={eps_per_sec:.1f} ep/s, "
                      f"ETA={eta:.0f}s")
        
        total_s = sum(self.success_history)
        total_e = len(self.success_history)
        print(f"\n  Phase 3 Complete: {successes}/{episodes}")
        print(f"  Overall: {total_s}/{total_e} ({total_s/max(total_e,1):.0%})")
        print(f"  Total time: {time.time()-t_start:.0f}s")
    
    # =====================================================
    # Results & Visualization
    # =====================================================
    
    def _save_results(self):
        output_dir = os.path.join(getattr(config, 'OUTPUT_DIR', 'results_cm'), "deep_q")
        os.makedirs(output_dir, exist_ok=True)
        
        save_dict = {
            'success_history': np.array(self.success_history),
            'energy_history': np.array(self.energy_history),
            'reward_history': np.array(self.reward_history),
            'loss_history': np.array(self.loss_history) if self.loss_history else np.array([]),
        }
        for key, val in self.weights.items():
            save_dict[f'weight_{key}'] = val
        
        np.savez(os.path.join(output_dir, "dqn_weights.npz"), **save_dict)
        
        try:
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
            
            fig, axes = plt.subplots(2, 2, figsize=(14, 10))
            fig.suptitle('DQN Training Results (Parallelized)', fontsize=14)
            
            episodes_range = range(len(self.success_history))
            
            if len(self.success_history) > 100:
                rolling_success = [
                    sum(self.success_history[max(0,i-100):i+1]) / min(i+1, 100)
                    for i in range(len(self.success_history))
                ]
                axes[0, 0].plot(episodes_range, rolling_success, 'b-', linewidth=1)
            axes[0, 0].set_title('Success Rate (Rolling 100)')
            axes[0, 0].set_ylim(-0.05, 1.05)
            axes[0, 0].grid(True, alpha=0.3)
            
            if self.loss_history:
                valid_loss = [l for l in self.loss_history if l > 0]
                if valid_loss:
                    axes[0, 1].plot(range(len(valid_loss)), valid_loss, 'r-', alpha=0.5, linewidth=0.5)
                    axes[0, 1].set_title('Training Loss')
                    axes[0, 1].set_yscale('log')
                    axes[0, 1].grid(True, alpha=0.3)
            
            successful_eps = [i for i, s in enumerate(self.success_history) if s]
            if successful_eps:
                successful_energy = [self.energy_history[i] for i in successful_eps]
                axes[1, 0].plot(successful_eps, successful_energy, 'g.', alpha=0.5, markersize=2)
                axes[1, 0].set_title('Energy (Successful Episodes)')
                axes[1, 0].grid(True, alpha=0.3)
            
            axes[1, 1].plot(episodes_range, self.reward_history, 'm-', alpha=0.3, linewidth=0.5)
            if len(self.reward_history) > 100:
                rolling_reward = [np.mean(self.reward_history[max(0,i-100):i+1])
                                 for i in range(len(self.reward_history))]
                axes[1, 1].plot(episodes_range, rolling_reward, 'm-', linewidth=2)
            axes[1, 1].set_title('Reward per Episode')
            axes[1, 1].grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, "dqn_training.png"), dpi=300)
            plt.close()
            print(f"\n✓ Training plots saved")
        except ImportError:
            pass
        
        print(f"✓ Results saved to: {output_dir}/")
    
    def generate_speed_profile(self):
        print("\n📊 Generating optimal speed profile (DQN)...")
        
        self.env.reset()
        segments, velocities, actions_taken, energies = [], [], [], []
        steps = 0
        
        while steps < getattr(config, 'MAX_STEPS_PER_EPISODE', 2000):
            raw_state = self.env._get_state()
            state = self.normalize_state(raw_state)
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
        
        output_dir = os.path.join(getattr(config, 'OUTPUT_DIR', 'results_cm'), "deep_q")
        os.makedirs(output_dir, exist_ok=True)
        
        np.savez(os.path.join(output_dir, "speed_profile.npz"),
                 segments=np.array(segments), velocities=np.array(velocities),
                 actions=np.array(actions_taken), energies=np.array(energies))
        
        final_seg = segments[-1] if segments else 0
        n_segs = getattr(self.env, 'n_segments', 749)
        completed = final_seg >= n_segs - 1
        
        print(f"✓ Final segment: {final_seg}/{n_segs} {'✓ COMPLETED' if completed else '✗ INCOMPLETE'}")
        print(f"  Final energy: {energies[-1] if energies else 0:.1f} kWh")
        
        try:
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
            
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
            fig.suptitle('DQN Optimal Speed Profile', fontsize=14)
            positions = np.array(segments) * 0.1
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