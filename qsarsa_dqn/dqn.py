"""
Deep Q-Network - FULLY PARALLELIZED (All Phases)
==================================================
Uses ALL CPU cores on DGX Spark for:
  Phase 1: Parallel Q-SARSA episode collection (20 eps at once)
  Phase 2: Vectorized batch training (BLAS multi-threading)
  Phase 3: Parallel DQN episode collection + vectorized training

Architecture (Figure 10):
    Input (x, v) → 128 tanh → 64 tanh → 16 tanh → 4 sigmoid
"""

import numpy as np
import os
import sys
import time
from multiprocessing import Pool, cpu_count

# Import project modules
try:
    import env_settings.config as config
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


# =============================================================
# TOP-LEVEL WORKER (must be at module level for multiprocessing)
# =============================================================

def _episode_worker(args):
    """
    Run ONE episode in a separate process.
    Works for both Phase 1 (Q-table) and Phase 3 (Q-table + network).
    
    Returns transitions, experiences, and stats.
    """
    (env_args, q_table, net_weights, epsilon, gamma, n_actions,
     reward_scale, max_steps, hidden_sizes, phi, use_network_ratio,
     phase) = args
    
    # Reconstruct environment
    try:
        grades, limits, curves = env_args
        for phys_mod, env_mod in [('env_settings.physics', 'env_settings.environment'),
                                   ('physics', 'environment')]:
            try:
                pm = __import__(phys_mod, fromlist=['TrainPhysics'])
                em = __import__(env_mod, fromlist=['TrainEnv'])
                physics = pm.TrainPhysics()
                env = em.TrainEnv(physics, grades, limits, curves)
                break
            except ImportError:
                continue
        else:
            return None
    except Exception:
        return None
    
    # Neural network forward pass (lightweight, no class needed)
    def net_predict(state, weights):
        x = np.array(state, dtype=np.float64).reshape(1, -1)
        n_layers = len(hidden_sizes) + 1
        for i in range(n_layers):
            x = x @ weights[f'W{i}'] + weights[f'b{i}']
            if i < n_layers - 1:
                x = np.tanh(np.clip(x, -20, 20))
            else:
                x = 1.0 / (1.0 + np.exp(-np.clip(x, -20, 20)))
        return x.flatten()
    
    def normalize_state(env_obj):
        pos = env_obj.seg_idx / max(env_obj.n_segments, 1)
        vel = env_obj.v / 120.0
        return np.array([np.clip(pos, 0, 1), np.clip(vel, 0, 1)], dtype=np.float64)
    
    def discretize_speed(velocity):
        return int(np.clip(velocity / 120.0 * 49, 0, 49))
    
    def compute_reward(info, done, env_obj):
        """Energy-focused reward."""
        C_penalty = 10.0
        if done and info.get('completed', False):
            total_energy = info.get('total_energy', getattr(env_obj, 'energy_kwh', 2500))
            energy_bonus = max(0, (2500 - total_energy) / 10.0)
            return (30.0 + energy_bonus) * reward_scale
        elif info.get('violation', False) or info.get('backward', False):
            return -C_penalty * reward_scale
        else:
            energy_step = info.get('energy_step', 0.0)
            progress = info.get('progress', 1.0 / max(env_obj.n_segments, 1))
            action = info.get('action', -1)
            seg_idx = min(env_obj.seg_idx, len(grades) - 1)
            grade = grades[seg_idx]
            current_v = env_obj.v
            lim_val = limits[seg_idx] if seg_idx < len(limits) else 22.0
            limit = lim_val if lim_val > 1 else 22.0
            speed_ratio = current_v / max(limit, 1.0)
            
            progress_reward = progress * 3.0
            energy_penalty = energy_step * 0.05
            
            action_bonus = 0.0
            if grade < -1.0:
                if action == 2: action_bonus = 0.02
                elif action == 3: action_bonus = 0.01
                elif action == 0: action_bonus = -0.03
            elif grade < 0.5:
                if speed_ratio > 0.5:
                    if action == 2: action_bonus = 0.015
                    elif action == 1: action_bonus = 0.005
                    elif action == 0: action_bonus = -0.01
            else:
                if action == 0 and speed_ratio < 0.3:
                    action_bonus = 0.005
            
            speed_bonus = 0.0
            if 0.6 <= speed_ratio <= 0.85: speed_bonus = 0.005
            elif speed_ratio > 0.95: speed_bonus = -0.005
            
            return (progress_reward - energy_penalty + action_bonus + speed_bonus) * reward_scale
    
    # Run episode
    env.reset()
    transitions = []   # For Q-table updates (Phase 1)
    experiences = []   # For replay buffer (Phase 3)
    episode_data = []  # For training data collection
    total_reward = 0
    done = False
    steps = 0
    prev_energy = 0.0
    info = {}
    
    n_segs = q_table.shape[0]
    n_vbins = q_table.shape[1]
    
    while not done and steps < max_steps:
        state = normalize_state(env)
        seg = min(env.seg_idx, n_segs - 1)
        v_bin = discretize_speed(env.v)
        
        # Action selection
        if np.random.random() < epsilon:
            action = np.random.randint(n_actions)
        elif phase == 3 and np.random.random() < use_network_ratio and net_weights is not None:
            q_vals = net_predict(state, net_weights)
            action = np.argmax(q_vals)
        else:
            action = np.argmax(q_table[seg, v_bin, :])
        
        # Step
        _, env_reward, done, info = env.step(action)
        if not isinstance(info, dict):
            info = {}
        
        next_state = normalize_state(env)
        next_seg = min(env.seg_idx, n_segs - 1)
        next_v_bin = discretize_speed(env.v)
        
        # Compute energy step
        curr_energy = getattr(env, 'energy_kwh', 0)
        info['energy_step'] = max(0, curr_energy - prev_energy)
        info['total_energy'] = curr_energy
        info['action'] = action
        info['progress'] = info.get('progress', 1.0 / max(env.n_segments, 1))
        prev_energy = curr_energy
        
        reward = compute_reward(info, done, env)
        total_reward += reward
        
        # Choose next action for SARSA
        if np.random.random() < epsilon:
            next_action = np.random.randint(n_actions)
        else:
            next_action = np.argmax(q_table[next_seg, next_v_bin, :])
        
        # Store transition for Q-table update
        transitions.append((seg, v_bin, action, reward / reward_scale,
                           next_seg, next_v_bin, next_action))
        
        # Store experience for replay buffer
        experiences.append((state.copy(), action, reward, next_state.copy(), done))
        
        # Store for training data
        episode_data.append((state.copy(), action, reward))
        
        steps += 1
    
    completed = (info.get('completed', False) if isinstance(info, dict) else False) or \
                env.seg_idx >= env.n_segments - 1
    energy = getattr(env, 'energy_kwh', 0)
    
    return {
        'transitions': transitions,
        'experiences': experiences,
        'episode_data': episode_data,
        'completed': completed,
        'energy': energy,
        'reward': total_reward,
        'steps': steps,
    }


class DeepQNetwork:
    """
    Deep Q-Network with FULL PARALLELIZATION across all CPU cores.
    """
    
    def __init__(self, env, phi_threshold=0.10, n_workers=None):
        self.env = env
        self.phi = phi_threshold
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
        self.buffer_size = 50000
        self.batch_size = 128
        self.min_replay = 500
        
        # Initialize networks
        self.weights = self._init_weights()
        self.target_weights = self._deep_copy_weights(self.weights)
        
        # Q-table
        n_segments = getattr(env, 'n_segments', 749)
        n_speeds = 50
        self.q_table = np.zeros((n_segments, n_speeds, self.n_actions))
        self.q_table[:, :, 0] = 1.0
        self.q_table[:, :, 1] = 1.2
        self.q_table[:, :, 2] = 0.8
        self.q_table[:, :, 3] = -0.5
        self.prev_q_table = self.q_table.copy()
        self.cm_history = []
        
        # Training history
        self.success_history = []
        self.energy_history = []
        self.time_history = []
        self.loss_history = []
        self.reward_history = []
        self.training_data = []
        
        # Store env data for workers
        self._env_args = self._extract_env_data()
        
        print(f"DQN initialized (FULLY PARALLELIZED):")
        print(f"  Architecture: {self.n_inputs} → {' → '.join(map(str, self.hidden_sizes))} → {self.n_actions}")
        print(f"  Activations: tanh (hidden) + sigmoid (output)")
        print(f"  Learning rate: {self.lr}")
        print(f"  Gradient clip: ±{self.grad_clip}")
        print(f"  Reward scale: ×{self.reward_scale}")
        print(f"  Target network: soft update τ={self.tau}")
        print(f"  Batch size: {self.batch_size} (vectorized)")
        print(f"  Workers: {self.n_workers} cores")
        print(f"  φ threshold: {self.phi}")
    
    def _extract_env_data(self):
        """Extract route data arrays from environment for workers."""
        env = self.env
        n_seg = getattr(env, 'n_segments', 749)
        
        grades = np.zeros(n_seg)
        limits = np.full(n_seg, 22.0)
        curves = np.zeros(n_seg)
        
        for attr in ['grades', '_grades', 'grade_data', 'track_grades']:
            if hasattr(env, attr):
                val = getattr(env, attr)
                if isinstance(val, np.ndarray) and len(val) > 0:
                    grades = val; break
        
        for attr in ['limits', '_limits', 'speed_limits', 'track_limits']:
            if hasattr(env, attr):
                val = getattr(env, attr)
                if isinstance(val, np.ndarray) and len(val) > 0:
                    limits = val; break
        
        for attr in ['curves', '_curves', 'curvatures', 'track_curves']:
            if hasattr(env, attr):
                val = getattr(env, attr)
                if isinstance(val, np.ndarray) and len(val) > 0:
                    curves = val; break
        
        return (grades, limits, curves)
    
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
        weights[f'b{last_idx}'] = np.array([0.5, 1.0, 0.8, -0.3])
        return weights
    
    def _deep_copy_weights(self, weights):
        return {k: v.copy() for k, v in weights.items()}
    
    def _forward_batch(self, X, weights=None):
        if weights is None:
            weights = self.weights
        n_layers = len(self.hidden_sizes) + 1
        activations = [X]
        for i in range(n_layers):
            Z = activations[-1] @ weights[f'W{i}'] + weights[f'b{i}']
            if i < n_layers - 1:
                A = np.tanh(np.clip(Z, -20, 20))
            else:
                A = 1.0 / (1.0 + np.exp(-np.clip(Z, -20, 20)))
            activations.append(A)
        return activations
    
    def predict(self, state, use_target=False):
        weights = self.target_weights if use_target else self.weights
        x = np.array(state, dtype=np.float64).reshape(1, -1)
        return self._forward_batch(x, weights)[-1].flatten()
    
    def predict_batch(self, states, use_target=False):
        weights = self.target_weights if use_target else self.weights
        X = np.array(states, dtype=np.float64)
        if X.ndim == 1: X = X.reshape(1, -1)
        return self._forward_batch(X, weights)[-1]
    
    def _backward_batch(self, states, targets, learning_rate=None):
        if learning_rate is None:
            learning_rate = self.lr
        batch_size = states.shape[0]
        activations = self._forward_batch(states)
        output = activations[-1]
        
        error = output - targets
        abs_error = np.abs(error)
        loss = np.where(abs_error <= 1.0, 0.5 * error**2, abs_error - 0.5).mean()
        
        d_output = np.where(abs_error <= 1.0, error, np.sign(error)) / self.n_actions
        d_output = np.clip(d_output, -self.grad_clip, self.grad_clip)
        
        n_layers = len(self.hidden_sizes) + 1
        sig_out = activations[-1]
        delta = d_output * sig_out * (1.0 - sig_out)
        
        for i in range(n_layers - 1, -1, -1):
            dW = (activations[i].T @ delta) / batch_size
            db = delta.mean(axis=0)
            dW = np.clip(dW, -self.grad_clip, self.grad_clip)
            db = np.clip(db, -self.grad_clip, self.grad_clip)
            self.weights[f'W{i}'] -= learning_rate * dW
            self.weights[f'b{i}'] -= learning_rate * db
            if i > 0:
                tanh_out = activations[i]
                delta = (delta @ self.weights[f'W{i}'].T) * (1.0 - tanh_out**2)
                delta = np.clip(delta, -self.grad_clip, self.grad_clip)
        
        return loss
    
    def _soft_update_target(self):
        for key in self.weights:
            self.target_weights[key] = self.tau * self.weights[key] + (1 - self.tau) * self.target_weights[key]
    
    # =====================================================
    # State / Speed helpers
    # =====================================================
    
    def normalize_state(self, raw_state):
        if hasattr(raw_state, '__len__') and len(raw_state) >= 2:
            pos, vel = raw_state[0], raw_state[1]
        else:
            pos = self.env.seg_idx / max(self.env.n_segments, 1)
            vel = self.env.v / 120.0
        pos_norm = np.clip(pos if pos <= 1.0 else pos / max(self.env.n_segments, 1), 0, 1)
        vel_norm = np.clip(vel if vel <= 1.0 else vel / 120.0, 0, 1)
        return np.array([pos_norm, vel_norm], dtype=np.float64)
    
    def _discretize_speed(self, velocity):
        return int(np.clip(velocity / 120.0 * 49, 0, 49))
    
    # =====================================================
    # Reward (used by main process for Phase 2 / profile)
    # =====================================================
    
    def compute_reward(self, info, done):
        C_penalty = 10.0
        if done and info.get('completed', False):
            total_energy = info.get('total_energy', getattr(self.env, 'energy_kwh', 2500))
            energy_bonus = max(0, (2500 - total_energy) / 10.0)
            return (30.0 + energy_bonus) * self.reward_scale
        elif info.get('violation', False) or info.get('backward', False):
            return -C_penalty * self.reward_scale
        else:
            energy_step = info.get('energy_step', 0.0)
            progress = info.get('progress', 1.0 / max(self.env.n_segments, 1))
            action = info.get('action', -1)
            seg_idx = min(self.env.seg_idx, 748) if hasattr(self.env, 'seg_idx') else 0
            grade = self._env_args[0][seg_idx] if seg_idx < len(self._env_args[0]) else 0.0
            current_v = getattr(self.env, 'v', 0)
            lim_val = self._env_args[1][seg_idx] if seg_idx < len(self._env_args[1]) else 22.0
            limit = lim_val if lim_val > 1 else 22.0
            speed_ratio = current_v / max(limit, 1.0)
            
            progress_reward = progress * 3.0
            energy_penalty = energy_step * 0.05
            action_bonus = 0.0
            if grade < -1.0:
                if action == 2: action_bonus = 0.02
                elif action == 3: action_bonus = 0.01
                elif action == 0: action_bonus = -0.03
            elif grade < 0.5:
                if speed_ratio > 0.5:
                    if action == 2: action_bonus = 0.015
                    elif action == 1: action_bonus = 0.005
                    elif action == 0: action_bonus = -0.01
            else:
                if action == 0 and speed_ratio < 0.3: action_bonus = 0.005
            speed_bonus = 0.0
            if 0.6 <= speed_ratio <= 0.85: speed_bonus = 0.005
            elif speed_ratio > 0.95: speed_bonus = -0.005
            return (progress_reward - energy_penalty + action_bonus + speed_bonus) * self.reward_scale
    
    # =====================================================
    # Q-SARSA (main-process updates)
    # =====================================================
    
    def _apply_qtable_updates(self, transitions, cm):
        """Apply Q-table updates from worker-collected transitions."""
        alpha = getattr(config, 'ALPHA', 0.5)
        for (seg, v_bin, action, reward, next_seg, next_v_bin, next_action) in transitions:
            if cm > self.phi:
                td_target = reward + self.gamma * self.q_table[next_seg, next_v_bin, next_action]
            else:
                td_target = reward + self.gamma * np.max(self.q_table[next_seg, next_v_bin, :])
            td_error = td_target - self.q_table[seg, v_bin, action]
            self.q_table[seg, v_bin, action] += alpha * td_error
    
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
    
    # =====================================================
    # Experience Replay (vectorized)
    # =====================================================
    
    def _store_experiences_bulk(self, experiences):
        self.replay_buffer.extend(experiences)
        if len(self.replay_buffer) > self.buffer_size:
            self.replay_buffer = self.replay_buffer[-self.buffer_size:]
    
    def _train_batch_vectorized(self):
        if len(self.replay_buffer) < self.min_replay:
            return 0.0
        indices = np.random.choice(len(self.replay_buffer), self.batch_size, replace=False)
        batch = [self.replay_buffer[i] for i in indices]
        
        states = np.array([b[0] for b in batch], dtype=np.float64)
        actions = np.array([b[1] for b in batch], dtype=np.int32)
        rewards = np.array([b[2] for b in batch], dtype=np.float64)
        next_states = np.array([b[3] for b in batch], dtype=np.float64)
        dones = np.array([b[4] for b in batch], dtype=bool)
        
        current_q = self.predict_batch(states)
        next_q = self.predict_batch(next_states, use_target=True)
        
        target_values = rewards.copy()
        target_values[~dones] += self.gamma * np.max(next_q[~dones], axis=1)
        
        target_q = current_q.copy()
        for i in range(self.batch_size):
            td_error = np.clip(target_values[i] - current_q[i, actions[i]], -self.td_clip, self.td_clip)
            target_q[i, actions[i]] = current_q[i, actions[i]] + td_error
        
        loss = self._backward_batch(states, target_q)
        self._soft_update_target()
        return loss
    
    # =====================================================
    # PARALLEL Episode Runner
    # =====================================================
    
    def _run_parallel_episodes(self, n_episodes, phase, net_use_ratio=0.0):
        """Run n_episodes across all workers, return list of results."""
        worker_args = []
        for _ in range(n_episodes):
            worker_args.append((
                self._env_args,
                self.q_table.copy(),
                self._deep_copy_weights(self.weights) if phase == 3 else None,
                self.epsilon,
                self.gamma,
                self.n_actions,
                self.reward_scale,
                getattr(config, 'MAX_STEPS_PER_EPISODE', 2000),
                self.hidden_sizes,
                self.phi,
                net_use_ratio,
                phase,
            ))
        
        with Pool(min(n_episodes, self.n_workers)) as pool:
            results = pool.map(_episode_worker, worker_args, chunksize=1)
        
        return [r for r in results if r is not None]
    
    def _test_multiprocessing(self):
        """Test if multiprocessing works with this environment."""
        try:
            test_args = [(
                self._env_args, self.q_table.copy(), None,
                1.0, self.gamma, self.n_actions, self.reward_scale,
                10, self.hidden_sizes, self.phi, 0.0, 1,
            )]
            with Pool(1) as pool:
                result = pool.map(_episode_worker, test_args, chunksize=1)
            if result[0] is None:
                raise RuntimeError("Worker returned None")
            print(f"  ✓ Multiprocessing OK ({self.n_workers} cores)")
            return True
        except Exception as e:
            print(f"  ✗ Multiprocessing failed: {e}")
            print(f"    Falling back to sequential mode")
            return False
    
    # =====================================================
    # Training (Three-Phase, ALL PARALLEL)
    # =====================================================
    
    def train(self, episodes=5000):
        phase1_eps = int(episodes * 0.4)
        phase2_batches = 100  # placeholder, overridden inside
        phase3_eps = episodes - phase1_eps
        
        # Test multiprocessing first
        print(f"\nTesting multiprocessing...")
        self.use_mp = self._test_multiprocessing()
        
        print(f"\n{'='*70}")
        print(f"PHASE 1: Q-SARSA Data Generation ({phase1_eps} episodes, "
              f"{'parallel' if self.use_mp else 'sequential'})")
        print(f"{'='*70}")
        self._phase1_qsarsa(phase1_eps)
        
        print(f"\n{'='*70}")
        print(f"PHASE 2: Supervised Network Training (vectorized)")
        print(f"{'='*70}")
        self._phase2_supervised(phase2_batches)
        
        print(f"\n{'='*70}")
        print(f"PHASE 3: Online DQN Fine-tuning ({phase3_eps} episodes, "
              f"{'parallel' if self.use_mp else 'sequential'}, Q-table guided)")
        print(f"{'='*70}")
        self._phase3_online(phase3_eps, start_episode=phase1_eps)
        
        self._save_results()
    
    # =====================================================
    # PHASE 1: Parallel Q-SARSA
    # =====================================================
    
    def _phase1_qsarsa(self, episodes):
        successes = 0
        t_start = time.time()
        batch_size = self.n_workers if self.use_mp else 1
        ep_done = 0
        
        print(f"  Batch size: {batch_size} episodes per round")
        
        while ep_done < episodes:
            n_this_round = min(batch_size, episodes - ep_done)
            cm = self._compute_cm(ep_done)
            
            if self.use_mp and n_this_round > 1:
                results = self._run_parallel_episodes(n_this_round, phase=1)
            else:
                # Sequential fallback
                results = []
                for _ in range(n_this_round):
                    r = self._run_single_episode_phase1()
                    if r is not None:
                        results.append(r)
            
            # Process results on main process
            for result in results:
                # Apply Q-table updates
                self._apply_qtable_updates(result['transitions'], cm)
                
                # Store experiences for replay buffer
                self._store_experiences_bulk(result['experiences'])
                
                # Store training data
                G = result['reward']
                for state, action, reward in result['episode_data']:
                    weight = min(abs(G / max(abs(reward), 1e-6)), 100.0)
                    self.training_data.append((state, action, weight, result['completed']))
                
                if result['completed']:
                    successes += 1
                self.success_history.append(result['completed'])
                self.energy_history.append(result['energy'])
                self.reward_history.append(result['reward'])
                
                self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
                ep_done += 1
            
            # Progress report
            if ep_done % 100 < batch_size or ep_done >= episodes:
                elapsed = time.time() - t_start
                eps_per_sec = ep_done / max(elapsed, 1)
                eta = (episodes - ep_done) / max(eps_per_sec, 0.01)
                n = min(100, len(self.success_history))
                recent_success = sum(self.success_history[-n:]) / n
                recent_e = [self.energy_history[-n+i] for i in range(n) if self.success_history[-n+i]]
                recent_energy = np.mean(recent_e) if recent_e else 0
                print(f"  Phase 1 Ep {ep_done}/{episodes}: "
                      f"Success={recent_success:.0%}, "
                      f"Energy={recent_energy:.0f} kWh, "
                      f"ε={self.epsilon:.3f}, "
                      f"Speed={eps_per_sec:.1f} ep/s, "
                      f"ETA={eta:.0f}s")
        
        print(f"\n  Phase 1 Complete: {successes}/{episodes} ({successes/max(episodes,1):.0%}), "
              f"{len(self.training_data)} samples, {time.time()-t_start:.0f}s")
    
    def _run_single_episode_phase1(self):
        """Sequential fallback for Phase 1."""
        self.env.reset()
        transitions, experiences, episode_data = [], [], []
        total_reward, prev_energy = 0, 0.0
        done, steps = False, 0
        info = {}
        max_steps = getattr(config, 'MAX_STEPS_PER_EPISODE', 2000)
        
        while not done and steps < max_steps:
            raw_state = self.env._get_state()
            state = self.normalize_state(raw_state)
            seg = min(self.env.seg_idx, self.q_table.shape[0] - 1)
            v_bin = self._discretize_speed(self.env.v)
            
            if np.random.random() < self.epsilon:
                action = np.random.randint(self.n_actions)
            else:
                action = np.argmax(self.q_table[seg, v_bin, :])
            
            _, env_reward, done, info = self.env.step(action)
            if not isinstance(info, dict): info = {}
            next_state = self.normalize_state(self.env._get_state() if hasattr(self.env, '_get_state') else [0,0])
            next_seg = min(self.env.seg_idx, self.q_table.shape[0] - 1)
            next_v_bin = self._discretize_speed(self.env.v)
            
            curr_energy = getattr(self.env, 'energy_kwh', 0)
            info['energy_step'] = max(0, curr_energy - prev_energy)
            info['total_energy'] = curr_energy
            info['action'] = action
            info['progress'] = info.get('progress', 1.0 / max(self.env.n_segments, 1))
            prev_energy = curr_energy
            
            reward = self.compute_reward(info, done)
            total_reward += reward
            
            next_action = np.random.randint(self.n_actions) if np.random.random() < self.epsilon \
                          else np.argmax(self.q_table[next_seg, next_v_bin, :])
            
            transitions.append((seg, v_bin, action, reward / self.reward_scale,
                               next_seg, next_v_bin, next_action))
            experiences.append((state.copy(), action, reward, next_state.copy(), done))
            episode_data.append((state.copy(), action, reward))
            steps += 1
        
        completed = (info.get('completed', False) if isinstance(info, dict) else False) or \
                    self.env.seg_idx >= self.env.n_segments - 1
        
        return {
            'transitions': transitions, 'experiences': experiences,
            'episode_data': episode_data, 'completed': completed,
            'energy': getattr(self.env, 'energy_kwh', 0),
            'reward': total_reward, 'steps': steps,
        }
    
    # =====================================================
    # PHASE 2: Vectorized Distillation
    # =====================================================
    
    def _phase2_supervised(self, n_batches):
        n_batches = 2000
        print(f"  Distilling Q-table into network ({n_batches} batches)...")
        print(f"  Q-table shape: {self.q_table.shape}")
        
        visited_states, visited_q_values = [], []
        n_segs, n_vbins = self.q_table.shape[0], self.q_table.shape[1]
        init_vals = np.array([1.0, 1.2, 0.8, -0.5])
        
        for seg in range(n_segs):
            for v_bin in range(n_vbins):
                q_vals = self.q_table[seg, v_bin, :]
                if not np.allclose(q_vals, init_vals, atol=0.01):
                    pos_norm = seg / max(n_segs, 1)
                    vel_norm = v_bin / max(n_vbins - 1, 1)
                    visited_states.append([pos_norm, vel_norm])
                    visited_q_values.append(q_vals)
        
        print(f"  Found {len(visited_states)} visited Q-table states")
        
        if len(visited_states) < 10:
            print("  Too few states, using episode data fallback...")
            good_data = [d for d in self.training_data if d[3]]
            if len(good_data) < 50: good_data = self.training_data
            states = np.array([d[0] for d in good_data])
            actions = np.array([d[1] for d in good_data])
            
            for batch_idx in range(n_batches):
                idx = np.random.choice(len(good_data), min(self.batch_size, len(good_data)), replace=True)
                batch_states = states[idx]
                target_q = np.full((len(idx), self.n_actions), 0.2)
                for i in range(len(idx)):
                    target_q[i, actions[idx[i]]] = 0.9
                loss = self._backward_batch(batch_states, target_q, learning_rate=self.lr)
                if (batch_idx + 1) % 200 == 0:
                    print(f"  Batch {batch_idx+1}/{n_batches}: Loss={loss:.6f}")
        else:
            all_states = np.array(visited_states, dtype=np.float64)
            all_q = np.array(visited_q_values, dtype=np.float64)
            q_min, q_max = all_q.min(), all_q.max()
            all_q_norm = (all_q - q_min) / max(q_max - q_min, 1e-6) * 0.8 + 0.1
            
            for batch_idx in range(n_batches):
                idx = np.random.choice(len(all_states), min(self.batch_size, len(all_states)), replace=True)
                loss = self._backward_batch(all_states[idx], all_q_norm[idx], learning_rate=self.lr * 3)
                
                if (batch_idx + 1) % 200 == 0:
                    test_idx = np.random.choice(len(all_states), min(200, len(all_states)), replace=False)
                    net_act = np.argmax(self.predict_batch(all_states[test_idx]), axis=1)
                    qt_act = np.argmax(all_q_norm[test_idx], axis=1)
                    agreement = np.mean(net_act == qt_act)
                    print(f"  Batch {batch_idx+1}/{n_batches}: Loss={loss:.6f}, Agreement={agreement:.0%}")
                    if agreement > 0.85:
                        print(f"  ✓ Early stop at {agreement:.0%} agreement")
                        break
        
        self.target_weights = self._deep_copy_weights(self.weights)
        print("  Phase 2 Complete: Network distilled from Q-table")
    
    # =====================================================
    # PHASE 3: Parallel Online DQN
    # =====================================================
    
    def _phase3_online(self, episodes, start_episode=0):
        successes = 0
        t_start = time.time()
        batch_size = self.n_workers if self.use_mp else 1
        ep_done = 0
        net_use_ratio = 0.0
        
        print(f"  Strategy: Q-table guided → gradual network takeover")
        print(f"  Batch: {batch_size} episodes/round, vectorized training")
        
        while ep_done < episodes:
            n_this_round = min(batch_size, episodes - ep_done)
            
            if self.use_mp and n_this_round > 1:
                results = self._run_parallel_episodes(n_this_round, phase=3,
                                                      net_use_ratio=net_use_ratio)
            else:
                results = []
                for _ in range(n_this_round):
                    r = self._run_single_episode_phase3(net_use_ratio)
                    if r is not None:
                        results.append(r)
            
            # Process results
            for result in results:
                self._store_experiences_bulk(result['experiences'])
                if result['completed']:
                    successes += 1
                self.success_history.append(result['completed'])
                self.energy_history.append(result['energy'])
                self.time_history.append(result['steps'])
                self.reward_history.append(result['reward'])
                self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
                ep_done += 1
            
            # Train network on collected experiences (multiple batches per round)
            n_train = max(1, n_this_round * 2)
            total_loss = 0
            for _ in range(n_train):
                total_loss += self._train_batch_vectorized()
            avg_loss = total_loss / max(n_train, 1)
            self.loss_history.append(avg_loss)
            
            # Adjust network usage based on success rate
            if ep_done >= 100 and ep_done % 50 < batch_size:
                recent_50 = sum(self.success_history[-50:]) / 50
                if recent_50 > 0.8:
                    net_use_ratio = min(net_use_ratio + 0.1, 0.9)
                elif recent_50 < 0.5:
                    net_use_ratio = max(net_use_ratio - 0.1, 0.0)
            
            # Progress report
            if ep_done % 100 < batch_size or ep_done >= episodes:
                elapsed = time.time() - t_start
                eps_sec = ep_done / max(elapsed, 1)
                eta = (episodes - ep_done) / max(eps_sec, 0.01)
                n = min(100, len(self.success_history))
                recent_s = sum(self.success_history[-n:]) / n
                recent_e = [self.energy_history[-n+i] for i in range(n) if self.success_history[-n+i]]
                r_energy = np.mean(recent_e) if recent_e else 0
                global_ep = start_episode + ep_done
                print(f"  Phase 3 Ep {ep_done}/{episodes} (Global {global_ep}): "
                      f"Success={recent_s:.0%}, Energy={r_energy:.0f} kWh, "
                      f"Loss={avg_loss:.6f}, ε={self.epsilon:.3f}, "
                      f"NetUse={net_use_ratio:.0%}, "
                      f"Speed={eps_sec:.1f} ep/s, ETA={eta:.0f}s")
        
        total_s = sum(self.success_history)
        total_e = len(self.success_history)
        print(f"\n  Phase 3 Complete: {successes}/{episodes}")
        print(f"  Overall: {total_s}/{total_e} ({total_s/max(total_e,1):.0%})")
        print(f"  Final network usage: {net_use_ratio:.0%}")
        print(f"  Total time: {time.time()-t_start:.0f}s")
    
    def _run_single_episode_phase3(self, net_use_ratio):
        """Sequential fallback for Phase 3."""
        self.env.reset()
        experiences = []
        total_reward, done, steps = 0, False, 0
        info = {}
        prev_energy = 0.0
        max_steps = getattr(config, 'MAX_STEPS_PER_EPISODE', 2000)
        
        while not done and steps < max_steps:
            raw_state = self.env._get_state()
            state = self.normalize_state(raw_state)
            seg = min(self.env.seg_idx, self.q_table.shape[0] - 1)
            v_bin = self._discretize_speed(self.env.v)
            
            if np.random.random() < self.epsilon:
                action = np.random.randint(self.n_actions)
            elif np.random.random() < net_use_ratio:
                action = np.argmax(self.predict(state))
            else:
                action = np.argmax(self.q_table[seg, v_bin, :])
            
            _, _, done, info = self.env.step(action)
            if not isinstance(info, dict): info = {}
            next_state = self.normalize_state(self.env._get_state() if hasattr(self.env, '_get_state') else [0,0])
            
            curr_energy = getattr(self.env, 'energy_kwh', 0)
            info['energy_step'] = max(0, curr_energy - prev_energy)
            info['total_energy'] = curr_energy
            info['action'] = action
            info['progress'] = info.get('progress', 1.0 / max(self.env.n_segments, 1))
            prev_energy = curr_energy
            
            reward = self.compute_reward(info, done)
            total_reward += reward
            experiences.append((state.copy(), action, reward, next_state.copy(), done))
            steps += 1
        
        completed = (info.get('completed', False) if isinstance(info, dict) else False) or \
                    self.env.seg_idx >= self.env.n_segments - 1
        
        return {
            'transitions': [], 'experiences': experiences, 'episode_data': [],
            'completed': completed, 'energy': getattr(self.env, 'energy_kwh', 0),
            'reward': total_reward, 'steps': steps,
        }
    
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
            fig.suptitle('DQN Training Results (Fully Parallelized)', fontsize=14)
            ep_range = range(len(self.success_history))
            
            if len(self.success_history) > 100:
                rolling = [sum(self.success_history[max(0,i-100):i+1]) / min(i+1,100)
                          for i in range(len(self.success_history))]
                axes[0,0].plot(ep_range, rolling, 'b-', linewidth=1)
            axes[0,0].set_title('Success Rate (Rolling 100)')
            axes[0,0].set_ylim(-0.05, 1.05)
            axes[0,0].grid(True, alpha=0.3)
            
            if self.loss_history:
                valid = [l for l in self.loss_history if l > 0]
                if valid:
                    axes[0,1].plot(range(len(valid)), valid, 'r-', alpha=0.5, linewidth=0.5)
                    axes[0,1].set_title('Training Loss')
                    axes[0,1].set_yscale('log')
                    axes[0,1].grid(True, alpha=0.3)
            
            succ_eps = [i for i, s in enumerate(self.success_history) if s]
            if succ_eps:
                succ_e = [self.energy_history[i] for i in succ_eps]
                axes[1,0].plot(succ_eps, succ_e, 'g.', alpha=0.5, markersize=2)
                axes[1,0].set_title('Energy (Successful Episodes)')
                axes[1,0].grid(True, alpha=0.3)
            
            axes[1,1].plot(ep_range, self.reward_history, 'm-', alpha=0.3, linewidth=0.5)
            if len(self.reward_history) > 100:
                rr = [np.mean(self.reward_history[max(0,i-100):i+1])
                     for i in range(len(self.reward_history))]
                axes[1,1].plot(ep_range, rr, 'm-', linewidth=2)
            axes[1,1].set_title('Reward per Episode')
            axes[1,1].grid(True, alpha=0.3)
            
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
            seg = min(self.env.seg_idx, self.q_table.shape[0] - 1)
            v_bin = self._discretize_speed(self.env.v)
            
            q_table_action = np.argmax(self.q_table[seg, v_bin, :])
            net_action = np.argmax(self.predict(state))
            q_vals = self.q_table[seg, v_bin, :]
            action = net_action if (q_vals.max() - q_vals.min()) < 0.1 else q_table_action
            
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
            ax1.set_ylabel('Speed (m/s)')
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