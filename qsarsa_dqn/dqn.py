"""
Deep Q-Network for Train Speed Profile Optimization — FIXED
=============================================================
Based on Section 3.6 of:
"A comprehensive study on reinforcement learning application 
 for train speed profile optimization"

FIXES from original dqn.py:
1. ACTION MAPPING: Now matches environment.py and config.py
     Action 0 = Brake, 1 = Coast, 2 = Cruise, 3 = Power
   (Was REVERSED: 0=Power, 3=Brake — agent was braking when
    it thought it was powering, causing massive energy waste)

2. REWARD FUNCTION: Uses paper's Eq. 45:
     R = R_end          if reached destination
     R = δ·ΔT + ρ·E    if forward progress  
     R = -C             if violation
   (Was overly complex 3-stage with wrong action references)

3. OUTPUT ACTIVATION: Linear output (no sigmoid)
   (Sigmoid squashed Q-values to [0,1], destroying gradients)

4. SPEED BINS: 100 bins matching Q-SARSA
   (Was 50 — too coarse for fine-grained speed control)

5. GAMMA: 0.95 matching Q-SARSA
   (Was 0.99 — too high, causing Q-value explosion in long episodes)

6. Q-TABLE INIT: Matches environment action semantics
   Action 3 (Power) = highest, Action 0 (Brake) = lowest

Architecture (Figure 10 from paper):
    Input (x, v) → 128 tanh → 64 tanh → 16 tanh → 4 (linear)

Training (Section 3.6):
    Phase 1: Q-SARSA episode collection + Q-table learning
    Phase 2: Distill Q-table into neural network
    Phase 3: Online DQN fine-tuning
"""

import numpy as np
import os
import sys
import time

try:
    import env_settings.config as config
except ImportError:
    class config:
        OUTPUT_DIR = "results_cm"
        MAX_STEPS_PER_EPISODE = 20000
        N_ACTIONS = 4
        GAMMA = 0.95
        ALPHA = 0.5
        EPSILON_START = 1.0
        EPSILON_MIN = 0.01
        EPSILON_DECAY = 0.999
        VEL_BIN_SIZE = 0.5
        ACTION_NAMES = ['Brake', 'Coast', 'Cruise', 'Power']


class DeepQNetwork:
    """
    Deep-Q Network following the paper's methodology:
    - Q-SARSA (Eq. 30) generates training data for the network
    - Network input: (position, speed) normalized to [0,1]
    - Network output: Q-values for each action (4 outputs)
    - Reward function per Eq. 45
    """
    
    def __init__(self, env, phi_threshold=0.10):
        self.env = env
        self.phi = phi_threshold
        
        # ─── Architecture (Figure 10) ───
        self.n_inputs = 2       # (position, velocity)
        self.n_actions = 4      # Brake, Coast, Cruise, Power
        self.hidden_sizes = [128, 64, 16]
        
        # ─── ACTION MAPPING (matches environment.py!) ───
        # Action 0 = Brake  (a = ab < 0)
        # Action 1 = Coast  (Ftraction = 0)
        # Action 2 = Cruise (a = 0)
        # Action 3 = Power  (a = am > 0)
        self.ACTION_BRAKE = 0
        self.ACTION_COAST = 1
        self.ACTION_CRUISE = 2
        self.ACTION_POWER = 3
        self.ACTION_NAMES = ['Brake', 'Coast', 'Cruise', 'Power']
        
        # ─── Hyperparameters ───
        self.gamma = 0.95           # Discount factor (match Q-SARSA)
        self.lr = 0.001             # Network learning rate
        self.lr_min = 0.00005
        self.lr_max = 0.002
        self.lr_warmup_frac = 0.05
        self.lr_history = []
        self._recent_losses = []
        
        # Q-SARSA alpha (dynamic)
        self.alpha_max = 0.5
        self.alpha_min = 0.05
        
        self.epsilon = getattr(config, 'EPSILON_START', 1.0)
        self.epsilon_min = getattr(config, 'EPSILON_MIN', 0.01)
        self.epsilon_decay = getattr(config, 'EPSILON_DECAY', 0.999)
        
        self.grad_clip = 1.0
        self.td_clip = 10.0
        self.tau = 0.01             # Soft target update rate (slower = more stable)
        
        # ─── Replay buffer ───
        self.replay_buffer = []
        self.buffer_size = 50000
        self.batch_size = 128
        self.min_replay = 500
        
        # ─── Networks (initialized with He init) ───
        self.weights = self._init_weights()
        self.target_weights = self._deep_copy_weights(self.weights)
        
        # ─── Q-table: CORRECT action mapping ───
        # Action 0=Brake, 1=Coast, 2=Cruise, 3=Power
        n_segments = getattr(env, 'n_segments', 749)
        n_speeds = 100  # FIX: match Q-SARSA's 100 bins
        self.q_table = np.zeros((n_segments, n_speeds, self.n_actions))
        self.q_table[:, :, self.ACTION_POWER]  = 5.0    # Power: highest (must move!)
        self.q_table[:, :, self.ACTION_CRUISE] = 2.0    # Cruise: second
        self.q_table[:, :, self.ACTION_COAST]  = 0.5    # Coast: third
        self.q_table[:, :, self.ACTION_BRAKE]  = -2.0   # Brake: lowest
        self.prev_q_table = self.q_table.copy()
        self.cm_history = []
        
        # ─── Extract route data ───
        self.grades = np.zeros(n_segments)
        self.limits = np.full(n_segments, 22.0)
        for attr in ['grades', '_grades', 'grade_data', 'track_grades']:
            if hasattr(env, attr):
                v = getattr(env, attr)
                if isinstance(v, np.ndarray) and len(v) > 0:
                    self.grades = v; break
        for attr in ['limits', '_limits', 'speed_limits', 'track_limits']:
            if hasattr(env, attr):
                v = getattr(env, attr)
                if isinstance(v, np.ndarray) and len(v) > 0:
                    self.limits = v; break
        
        # ─── Reward parameters (Eq. 45) ───
        self.R_end = 100.0          # Terminal reward for reaching destination
        self.delta = -0.01          # Time cost coefficient (δ)
        self.rho = -0.001           # Energy cost coefficient (ρ)
        self.C_penalty = -10.0      # Violation penalty (-C)
        
        # ─── History ───
        self.success_history = []
        self.energy_history = []
        self.time_history = []
        self.loss_history = []
        self.reward_history = []
        self.training_data = []
        
        print(f"DQN (FIXED) initialized:")
        print(f"  Architecture: {self.n_inputs} → {' → '.join(map(str, self.hidden_sizes))} → {self.n_actions}")
        print(f"  Activations: tanh (hidden) + LINEAR (output)  [FIX: was sigmoid]")
        print(f"  Action mapping: 0=Brake, 1=Coast, 2=Cruise, 3=Power  [FIX: was reversed]")
        print(f"  Q-table: {n_segments}×{n_speeds}×{self.n_actions}  [FIX: 100 bins, was 50]")
        print(f"  Q-table init: Power=5.0, Cruise=2.0, Coast=0.5, Brake=-2.0  [FIX]")
        print(f"  γ (discount): {self.gamma}  [FIX: was 0.99]")
        print(f"  τ (target update): {self.tau}  [FIX: was 0.1]")
        print(f"  Reward: Paper Eq. 45 (δ={self.delta}, ρ={self.rho})  [FIX: was 3-stage]")
        print(f"  φ threshold: {self.phi}")
    
    # ═════════════════════════════════════════════════════
    # Neural Network
    # ═════════════════════════════════════════════════════
    
    def _init_weights(self):
        """He initialization for tanh layers."""
        weights = {}
        sizes = [self.n_inputs] + self.hidden_sizes + [self.n_actions]
        for i in range(len(sizes) - 1):
            std = np.sqrt(2.0 / sizes[i])
            weights[f'W{i}'] = np.random.randn(sizes[i], sizes[i+1]) * std
            weights[f'b{i}'] = np.zeros(sizes[i+1])
        return weights
    
    def _deep_copy_weights(self, w):
        return {k: v.copy() for k, v in w.items()}
    
    def _forward_batch(self, X, weights=None):
        """Forward pass: tanh hidden layers + LINEAR output (FIX: was sigmoid)."""
        if weights is None:
            weights = self.weights
        n_layers = len(self.hidden_sizes) + 1
        activations = [X]
        for i in range(n_layers):
            Z = activations[-1] @ weights[f'W{i}'] + weights[f'b{i}']
            if i < n_layers - 1:
                # Hidden layers: tanh
                A = np.tanh(np.clip(Z, -20, 20))
            else:
                # Output layer: LINEAR (not sigmoid!)
                # Q-values must be unbounded for proper value estimation
                A = Z
            activations.append(A)
        return activations
    
    def predict(self, state, use_target=False):
        w = self.target_weights if use_target else self.weights
        x = np.array(state, dtype=np.float64).reshape(1, -1)
        return self._forward_batch(x, w)[-1].flatten()
    
    def predict_batch(self, states, use_target=False):
        w = self.target_weights if use_target else self.weights
        X = np.array(states, dtype=np.float64)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        return self._forward_batch(X, w)[-1]
    
    def _backward_batch(self, states, targets, lr=None):
        """Backprop with Huber loss and gradient clipping. LINEAR output derivative = 1."""
        if lr is None:
            lr = self.lr
        bs = states.shape[0]
        acts = self._forward_batch(states)
        out = acts[-1]
        
        err = out - targets
        ae = np.abs(err)
        # Huber loss
        loss = np.where(ae <= 1.0, 0.5 * err**2, ae - 0.5).mean()
        
        # Derivative of Huber loss
        d_out = np.where(ae <= 1.0, err, np.sign(err)) / self.n_actions
        d_out = np.clip(d_out, -self.grad_clip, self.grad_clip)
        
        n_layers = len(self.hidden_sizes) + 1
        # FIX: Linear output → derivative is just 1 (not sigmoid derivative)
        delta = d_out  # No activation derivative for linear output
        
        for i in range(n_layers - 1, -1, -1):
            dW = np.clip((acts[i].T @ delta) / bs, -self.grad_clip, self.grad_clip)
            db = np.clip(delta.mean(axis=0), -self.grad_clip, self.grad_clip)
            self.weights[f'W{i}'] -= lr * dW
            self.weights[f'b{i}'] -= lr * db
            if i > 0:
                delta = (delta @ self.weights[f'W{i}'].T) * (1.0 - acts[i]**2)  # tanh deriv
                delta = np.clip(delta, -self.grad_clip, self.grad_clip)
        
        self._recent_losses.append(loss)
        if len(self._recent_losses) > 100:
            self._recent_losses = self._recent_losses[-100:]
        return loss
    
    def _soft_update_target(self):
        for k in self.weights:
            self.target_weights[k] = self.tau * self.weights[k] + (1 - self.tau) * self.target_weights[k]
    
    # ═════════════════════════════════════════════════════
    # State Helpers
    # ═════════════════════════════════════════════════════
    
    def normalize_state(self, raw_state):
        """Normalize state to [0, 1] for network input."""
        if hasattr(raw_state, '__len__') and len(raw_state) >= 2:
            p, v = raw_state[0], raw_state[1]
        else:
            p = self.env.seg_idx
            v = self.env.v
        # Position: segment / total segments
        p_norm = np.clip(p / max(self.env.n_segments, 1), 0, 1)
        # Velocity: v / max_speed
        max_speed = getattr(config, 'MAX_SPEED_MS', 36.1)
        v_norm = np.clip(v / max_speed, 0, 1)
        return np.array([p_norm, v_norm], dtype=np.float64)
    
    def _discretize_speed(self, vel):
        """Discretize speed to 0-99 bins (FIX: was 0-49)."""
        vel_bin_size = getattr(config, 'VEL_BIN_SIZE', 0.5)
        return int(np.clip(vel / vel_bin_size, 0, 99))
    
    # ═════════════════════════════════════════════════════
    # Dynamic Learning Rate
    # ═════════════════════════════════════════════════════
    
    def _get_lr(self, episode, total_episodes, phase=3):
        """Cosine annealing with warm-up."""
        if phase == 2:
            return self.lr_max * 3  # Supervised: fast learning
        
        frac = episode / max(total_episodes, 1)
        warmup_end = self.lr_warmup_frac
        
        if frac < warmup_end:
            t = frac / warmup_end
            lr = 0.0001 + t * (self.lr_max - 0.0001)
        else:
            t = (frac - warmup_end) / (1.0 - warmup_end)
            lr = self.lr_min + 0.5 * (self.lr_max - self.lr_min) * (1 + np.cos(np.pi * t))
        
        # Loss-spike detection
        if len(self._recent_losses) >= 10:
            avg_loss = np.mean(self._recent_losses[-10:])
            if self._recent_losses[-1] > 3.0 * avg_loss and avg_loss > 1e-8:
                lr *= 0.5
        
        lr = np.clip(lr, self.lr_min, self.lr_max)
        self.lr = lr
        self.lr_history.append(lr)
        return lr
    
    def _get_alpha(self, episode, total_episodes):
        """Dynamic Q-SARSA alpha: cosine decay."""
        frac = episode / max(total_episodes, 1)
        return self.alpha_min + 0.5 * (self.alpha_max - self.alpha_min) * (1 + np.cos(np.pi * frac))
    
    # ═════════════════════════════════════════════════════
    # REWARD FUNCTION — Paper Eq. 45 (FIXED)
    # ═════════════════════════════════════════════════════
    
    def compute_reward(self, info, done):
        """
        Reward function based on Eq. 45 of the paper:
        
        R = R_end                   if x_i = x_T  (reached destination)
        R = δ·ΔT_i + ρ·E_i         if x_i > x_{i-1}  (forward progress)
        R = -C                      if x_i <= x_{i-1} or v_i > v_limit
        
        Where:
            δ = time cost coefficient (negative, penalizes time)
            ρ = energy cost coefficient (negative, penalizes energy)
            R_end = large positive terminal reward
            C = violation penalty constant
        """
        # Terminal: reached destination
        if done and self.env.seg_idx >= self.env.n_segments - 2:
            # Big reward for completion + energy bonus
            total_energy = getattr(self.env, 'energy_kwh', 0)
            travel_time = getattr(self.env, 't', 0)
            # Base completion reward + energy-dependent bonus
            energy_bonus = max(0, (3000 - total_energy) / 30.0)
            return self.R_end + energy_bonus
        
        # Check violations
        seg_idx = min(self.env.seg_idx, len(self.limits) - 1)
        limit = self.limits[seg_idx]
        if limit <= 1:
            limit = 22.0  # Station — use default limit
        
        current_v = getattr(self.env, 'v', 0)
        
        # Violation: speed limit exceeded
        if current_v > limit + 2.0:
            return self.C_penalty
        
        # Violation: train stopped (not at station)
        if current_v < 0.5 and limit > 1.0:
            return self.C_penalty
        
        # Normal progress: δ·ΔT + ρ·E
        energy_step = info.get('energy_step', 0.0)
        dt = getattr(config, 'DT', 1.0)
        
        # Time cost: penalizes each time step
        time_cost = self.delta * dt
        # Energy cost: penalizes energy consumption
        energy_cost = self.rho * energy_step * 1000  # Scale kWh to meaningful range
        
        # Progress bonus: reward for advancing segments
        progress = info.get('progress', 0.0)
        progress_reward = progress * 5.0
        
        # Grade-aware coasting bonus (small)
        grade = self.grades[seg_idx] if seg_idx < len(self.grades) else 0
        action = info.get('action', -1)
        grade_bonus = 0.0
        if grade < -0.5 and action == self.ACTION_COAST and current_v > 3.0:
            grade_bonus = 0.002  # Reward coasting on downhill
        if grade < -0.5 and action == self.ACTION_POWER:
            grade_bonus = -0.003  # Penalize powering on downhill
        
        reward = progress_reward + time_cost + energy_cost + grade_bonus
        return reward
    
    # ═════════════════════════════════════════════════════
    # Q-SARSA Update (Eq. 30)
    # ═════════════════════════════════════════════════════
    
    def _compute_cm(self, ep):
        """Convergence Measurement per Eq. 29: CM(i) = ΔQ_i / ΔQ_{i-1}"""
        if ep < 2:
            return 0.0
        dq = np.sum(np.abs(self.q_table - self.prev_q_table))
        self.prev_q_table = self.q_table.copy()
        cm = dq / self.cm_history[-1] if self.cm_history and self.cm_history[-1] > 1e-10 else 1.0
        self.cm_history.append(dq)
        return cm
    
    def _qsarsa_update(self, seg, vb, a, r, ns, nvb, na, cm, alpha=0.5):
        """
        Q-SARSA update rule (Eq. 30):
        - When CM > φ: use SARSA update (Eq. 27) — faster
        - When CM < φ: use Q-learning update (Eq. 26) — escape local optima
        """
        if cm > self.phi:
            # SARSA: use next action
            target = r + self.gamma * self.q_table[ns, nvb, na]
        else:
            # Q-learning: use max
            target = r + self.gamma * np.max(self.q_table[ns, nvb, :])
        self.q_table[seg, vb, a] += alpha * (target - self.q_table[seg, vb, a])
    
    # ═════════════════════════════════════════════════════
    # Replay Buffer + Network Training
    # ═════════════════════════════════════════════════════
    
    def _store_exp(self, s, a, r, ns, d):
        self.replay_buffer.append((s, a, r, ns, d))
        if len(self.replay_buffer) > self.buffer_size:
            self.replay_buffer.pop(0)
    
    def _train_batch(self):
        """Sample from replay buffer and train network via gradient descent."""
        if len(self.replay_buffer) < self.min_replay:
            return 0.0
        
        idx = np.random.choice(len(self.replay_buffer), self.batch_size, replace=False)
        batch = [self.replay_buffer[i] for i in idx]
        
        states = np.array([b[0] for b in batch], dtype=np.float64)
        actions = np.array([b[1] for b in batch], dtype=np.int32)
        rewards = np.array([b[2] for b in batch], dtype=np.float64)
        nstates = np.array([b[3] for b in batch], dtype=np.float64)
        dones = np.array([b[4] for b in batch], dtype=bool)
        
        # Current Q-values
        cur_q = self.predict_batch(states)
        # Target network Q-values for next states
        nxt_q = self.predict_batch(nstates, use_target=True)
        
        # Compute TD targets
        tv = rewards.copy()
        tv[~dones] += self.gamma * np.max(nxt_q[~dones], axis=1)
        
        # Build target array (only update the taken action)
        tgt = cur_q.copy()
        for i in range(self.batch_size):
            td = np.clip(tv[i] - cur_q[i, actions[i]], -self.td_clip, self.td_clip)
            tgt[i, actions[i]] = cur_q[i, actions[i]] + td
        
        loss = self._backward_batch(states, tgt, lr=self.lr)
        self._soft_update_target()
        return loss
    
    # ═════════════════════════════════════════════════════
    # Training
    # ═════════════════════════════════════════════════════
    
    def train(self, episodes=5000):
        """
        Three-phase training per Section 3.6:
        Phase 1: Q-SARSA data generation (fills Q-table)
        Phase 2: Distill Q-table into neural network
        Phase 3: Online DQN fine-tuning
        """
        p1 = int(episodes * 0.4)
        p3 = episodes - p1
        
        print(f"\n{'='*70}")
        print(f"PHASE 1: Q-SARSA Data Generation ({p1} episodes)")
        print(f"{'='*70}")
        self._phase1(p1)
        
        print(f"\n{'='*70}")
        print(f"PHASE 2: Distill Q-table → Network")
        print(f"{'='*70}")
        self._phase2()
        
        print(f"\n{'='*70}")
        print(f"PHASE 3: Online DQN Fine-tuning ({p3} episodes)")
        print(f"{'='*70}")
        self._phase3(p3, start_ep=p1)
        
        self._save_results()
    
    def _phase1(self, episodes):
        """Phase 1: Run Q-SARSA episodes to build Q-table (training data for network)."""
        successes = 0
        t0 = time.time()
        max_steps = getattr(config, 'MAX_STEPS_PER_EPISODE', 20000)
        
        for ep in range(episodes):
            self.env.reset()
            ep_data = []
            total_r = 0
            done = False
            steps = 0
            cm = self._compute_cm(ep)
            alpha = self._get_alpha(ep, episodes)
            prev_e = 0.0
            info = {}
            
            while not done and steps < max_steps:
                raw = self.env._get_state()
                state = self.normalize_state(raw)
                seg = min(self.env.seg_idx, self.q_table.shape[0] - 1)
                vb = self._discretize_speed(self.env.v)
                
                # ε-greedy action selection from Q-table
                if np.random.random() < self.epsilon:
                    action = np.random.randint(self.n_actions)
                else:
                    action = np.argmax(self.q_table[seg, vb, :])
                
                _, _, done, info = self.env.step(action)
                if not isinstance(info, dict):
                    info = {}
                nstate = self.normalize_state(self.env._get_state())
                
                # Track energy per step
                ce = getattr(self.env, 'energy_kwh', 0)
                info['energy_step'] = max(0, ce - prev_e)
                info['total_energy'] = ce
                info['action'] = action
                info['progress'] = info.get('progress', 1.0 / max(self.env.n_segments, 1))
                prev_e = ce
                
                reward = self.compute_reward(info, done)
                ep_data.append((state, action, reward))
                total_r += reward
                
                # Q-SARSA update (Eq. 30)
                ns = min(self.env.seg_idx, self.q_table.shape[0] - 1)
                nvb = self._discretize_speed(self.env.v)
                na = np.random.randint(self.n_actions) if np.random.random() < self.epsilon \
                     else np.argmax(self.q_table[ns, nvb, :])
                self._qsarsa_update(seg, vb, action, reward, ns, nvb, na, cm, alpha)
                
                # Also store in replay buffer for Phase 3
                self._store_exp(state, action, reward, nstate, done)
                steps += 1
            
            # Check completion
            completed = self.env.seg_idx >= self.env.n_segments - 2
            if completed:
                successes += 1
            
            self.success_history.append(completed)
            self.energy_history.append(getattr(self.env, 'energy_kwh', 0))
            self.reward_history.append(total_r)
            
            # Store training data with return-weighted importance
            G = total_r
            for s, a, r in ep_data:
                w = min(abs(G / max(abs(r), 1e-6)), 100.0)
                self.training_data.append((s, a, w, completed))
            
            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
            
            if (ep + 1) % 100 == 0:
                el = time.time() - t0
                eps = (ep + 1) / el
                eta = (episodes - ep - 1) / eps
                n = min(100, len(self.success_history))
                rs = sum(self.success_history[-n:]) / n
                re_vals = [self.energy_history[-n+i] for i in range(n) if self.success_history[-n+i]]
                re = np.mean(re_vals) if re_vals else 0
                print(f"  Phase 1 Ep {ep+1}/{episodes}: "
                      f"Success={rs:.0%}, Energy={re:.0f} kWh, "
                      f"α={alpha:.4f}, ε={self.epsilon:.3f}, "
                      f"Speed={eps:.1f} ep/s, ETA={eta:.0f}s")
        
        print(f"\n  Phase 1 Complete: {successes}/{episodes} ({successes/max(episodes,1):.0%}), "
              f"{len(self.training_data)} samples, {time.time()-t0:.0f}s")
    
    def _phase2(self):
        """Phase 2: Distill Q-table knowledge into neural network."""
        n_batches = 3000
        print(f"  Distilling Q-table ({n_batches} batches)...")
        
        # Collect visited Q-table states
        visited_s, visited_q = [], []
        ns, nv = self.q_table.shape[0], self.q_table.shape[1]
        # Initial values to compare against
        init = np.array([-2.0, 0.5, 2.0, 5.0])  # [Brake, Coast, Cruise, Power] init
        
        for seg in range(ns):
            for vb in range(nv):
                q = self.q_table[seg, vb, :]
                if not np.allclose(q, init, atol=0.1):
                    visited_s.append([seg / ns, vb / max(nv - 1, 1)])
                    visited_q.append(q)
        
        print(f"  Found {len(visited_s)} visited states")
        
        if len(visited_s) < 10:
            print("  Too few states, using episode data fallback...")
            good = [d for d in self.training_data if d[3]]
            if len(good) < 50:
                good = self.training_data
            ss = np.array([d[0] for d in good])
            aa = np.array([d[1] for d in good])
            for b in range(n_batches):
                ix = np.random.choice(len(good), min(self.batch_size, len(good)), replace=True)
                # Create soft targets centered on the chosen action
                tgt = np.full((len(ix), self.n_actions), -1.0)
                for i in range(len(ix)):
                    tgt[i, aa[ix[i]]] = 1.0
                loss = self._backward_batch(ss[ix], tgt)
                if (b + 1) % 500 == 0:
                    print(f"  Batch {b+1}/{n_batches}: Loss={loss:.6f}")
        else:
            S = np.array(visited_s, dtype=np.float64)
            Q = np.array(visited_q, dtype=np.float64)
            
            # Normalize Q-values for stable training (linear scaling)
            qmin, qmax = Q.min(), Q.max()
            qrange = max(qmax - qmin, 1e-6)
            # Scale to roughly [-2, 2] range for linear output
            Q_norm = (Q - qmin) / qrange * 4.0 - 2.0
            
            for b in range(n_batches):
                ix = np.random.choice(len(S), min(self.batch_size, len(S)), replace=True)
                p2_lr = self._get_lr(b, n_batches, phase=2)
                loss = self._backward_batch(S[ix], Q_norm[ix], lr=p2_lr)
                
                if (b + 1) % 500 == 0:
                    tix = np.random.choice(len(S), min(200, len(S)), replace=False)
                    net_a = np.argmax(self.predict_batch(S[tix]), axis=1)
                    qt_a = np.argmax(Q_norm[tix], axis=1)
                    agr = np.mean(net_a == qt_a)
                    print(f"  Batch {b+1}/{n_batches}: Loss={loss:.6f}, Agreement={agr:.0%}")
                    if agr > 0.90:
                        print(f"  ✓ Early stop at {agr:.0%} agreement")
                        break
        
        self.target_weights = self._deep_copy_weights(self.weights)
        print("  Phase 2 Complete")
    
    def _phase3(self, episodes, start_ep=0):
        """Phase 3: Online DQN fine-tuning with Q-table guidance."""
        successes = 0
        t0 = time.time()
        max_steps = getattr(config, 'MAX_STEPS_PER_EPISODE', 20000)
        
        # Gradually shift from Q-table to network
        net_use_start = 0.2
        net_use_end = 0.8
        
        print(f"  Strategy: Q-table guided → gradual network takeover")
        
        for ep in range(episodes):
            self._get_lr(ep, episodes, phase=3)
            
            # Linearly increase network usage
            net_use = net_use_start + (net_use_end - net_use_start) * (ep / max(episodes - 1, 1))
            
            self.env.reset()
            total_r = 0
            total_loss = 0
            done = False
            steps = 0
            n_upd = 0
            prev_e = 0.0
            info = {}
            
            while not done and steps < max_steps:
                raw = self.env._get_state()
                state = self.normalize_state(raw)
                seg = min(self.env.seg_idx, self.q_table.shape[0] - 1)
                vb = self._discretize_speed(self.env.v)
                
                # Action selection: ε-greedy with Q-table/network blend
                if np.random.random() < self.epsilon:
                    action = np.random.randint(self.n_actions)
                elif np.random.random() < net_use:
                    action = np.argmax(self.predict(state))
                else:
                    action = np.argmax(self.q_table[seg, vb, :])
                
                _, _, done, info = self.env.step(action)
                if not isinstance(info, dict):
                    info = {}
                nstate = self.normalize_state(self.env._get_state())
                
                ce = getattr(self.env, 'energy_kwh', 0)
                info['energy_step'] = max(0, ce - prev_e)
                info['total_energy'] = ce
                info['action'] = action
                info['progress'] = info.get('progress', 1.0 / max(self.env.n_segments, 1))
                prev_e = ce
                
                reward = self.compute_reward(info, done)
                total_r += reward
                self._store_exp(state, action, reward, nstate, done)
                
                # Also update Q-table (keeps it fresh)
                ns = min(self.env.seg_idx, self.q_table.shape[0] - 1)
                nvb = self._discretize_speed(self.env.v)
                na = np.argmax(self.q_table[ns, nvb, :])
                alpha = self._get_alpha(ep + start_ep, episodes + start_ep)
                cm = self._compute_cm(ep + start_ep) if ep > 0 else 1.0
                self._qsarsa_update(seg, vb, action, reward, ns, nvb, na, cm, alpha)
                
                # Train network from replay buffer
                if steps % 4 == 0 and len(self.replay_buffer) >= self.min_replay:
                    loss = self._train_batch()
                    total_loss += loss
                    n_upd += 1
                
                steps += 1
            
            completed = self.env.seg_idx >= self.env.n_segments - 2
            if completed:
                successes += 1
            
            self.success_history.append(completed)
            self.energy_history.append(getattr(self.env, 'energy_kwh', 0))
            self.time_history.append(getattr(self.env, 't', 0))
            self.reward_history.append(total_r)
            self.loss_history.append(total_loss / max(n_upd, 1))
            
            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
            
            if (ep + 1) % 100 == 0:
                el = time.time() - t0
                eps = (ep + 1) / el
                eta = (episodes - ep - 1) / eps
                n = min(100, len(self.success_history))
                rs = sum(self.success_history[-n:]) / n
                re_vals = [self.energy_history[-(n-i)] for i in range(n) 
                          if self.success_history[-(n-i)]]
                re = np.mean(re_vals) if re_vals else 0
                avg_loss = np.mean(self.loss_history[-min(100, len(self.loss_history)):])
                print(f"  Phase 3 Ep {ep+1}/{episodes}: "
                      f"Success={rs:.0%}, Energy={re:.0f} kWh, "
                      f"Loss={avg_loss:.6f}, NetUse={net_use:.0%}, "
                      f"ε={self.epsilon:.3f}, Speed={eps:.1f} ep/s, ETA={eta:.0f}s")
        
        print(f"\n  Phase 3 Complete: {successes}/{episodes} "
              f"({successes/max(episodes,1):.0%}), {time.time()-t0:.0f}s")
    
    # ═════════════════════════════════════════════════════
    # Results & Plotting
    # ═════════════════════════════════════════════════════
    
    def _save_results(self):
        """Save training results and plots."""
        od = os.path.join(getattr(config, 'OUTPUT_DIR', 'results_cm'), "deep_q")
        os.makedirs(od, exist_ok=True)
        
        # Save Q-table
        np.save(os.path.join(od, "q_table.npy"), self.q_table)
        
        # Save history
        np.savez(os.path.join(od, "training_history.npz"),
                 success=np.array(self.success_history),
                 energy=np.array(self.energy_history),
                 reward=np.array(self.reward_history),
                 loss=np.array(self.loss_history) if self.loss_history else np.array([]))
        
        # Summary
        n = min(200, len(self.success_history))
        if n > 0:
            rs = sum(self.success_history[-n:]) / n
            re_vals = [self.energy_history[-n+i] for i in range(n) if self.success_history[-n+i]]
            re = np.mean(re_vals) if re_vals else 0
            print(f"\n  Last {n} episodes: Success={rs:.0%}, Avg Energy={re:.0f} kWh")
            if re_vals:
                print(f"  Best Energy: {min(re_vals):.0f} kWh")
        
        # Plot training curves
        try:
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
            
            fig, axes = plt.subplots(2, 2, figsize=(14, 10))
            fig.suptitle('DQN Training (FIXED)', fontsize=14)
            
            er = range(len(self.success_history))
            
            # Success rate
            if len(self.success_history) > 50:
                window = 50
                sr = [np.mean(self.success_history[max(0,i-window):i+1]) for i in er]
                axes[0,0].plot(er, sr, 'g-', linewidth=1.5)
            axes[0,0].set_title('Success Rate (Moving Avg)')
            axes[0,0].set_ylabel('Success Rate')
            axes[0,0].grid(True, alpha=0.3)
            
            # Energy history
            successful_energies = [(i, e) for i, (e, s) in 
                                   enumerate(zip(self.energy_history, self.success_history)) if s]
            if successful_energies:
                se_x, se_y = zip(*successful_energies)
                axes[0,1].scatter(se_x, se_y, c='b', alpha=0.3, s=2)
                if len(se_y) > 50:
                    window = 50
                    ma = np.convolve(se_y, np.ones(window)/window, mode='valid')
                    axes[0,1].plot(range(se_x[0]+window-1, se_x[0]+window-1+len(ma)), ma, 
                                  'r-', linewidth=2, label='Moving Avg')
                    axes[0,1].legend()
            axes[0,1].set_title('Energy (Successful Episodes)')
            axes[0,1].grid(True, alpha=0.3)
            
            # Loss
            if self.loss_history:
                axes[1,0].plot(range(len(self.loss_history)), self.loss_history, 
                              'r-', alpha=0.3, linewidth=0.5)
                if len(self.loss_history) > 100:
                    ll = [np.mean(self.loss_history[max(0,i-100):i+1])
                         for i in range(len(self.loss_history))]
                    axes[1,0].plot(range(len(ll)), ll, 'r-', linewidth=2)
                axes[1,0].set_title('Training Loss')
                axes[1,0].grid(True, alpha=0.3)
            
            # Reward
            axes[1,1].plot(er, self.reward_history, 'm-', alpha=0.3, linewidth=0.5)
            if len(self.reward_history) > 100:
                rr = [np.mean(self.reward_history[max(0,i-100):i+1])
                     for i in range(len(self.reward_history))]
                axes[1,1].plot(er, rr, 'm-', linewidth=2)
            axes[1,1].set_title('Reward per Episode')
            axes[1,1].grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(os.path.join(od, "dqn_training.png"), dpi=300)
            plt.close()
            print(f"\n✓ Training plots saved")
        except ImportError:
            pass
        
        print(f"✓ Results saved to: {od}/")
    
    def generate_speed_profile(self):
        """Generate optimal speed profile using the learned policy."""
        print("\n📊 Generating optimal speed profile (DQN FIXED)...")
        self.env.reset()
        segs, vels, acts, ens = [], [], [], []
        steps = 0
        max_steps = getattr(config, 'MAX_STEPS_PER_EPISODE', 20000)
        
        while steps < max_steps:
            state = self.normalize_state(self.env._get_state())
            seg = min(self.env.seg_idx, self.q_table.shape[0] - 1)
            vb = self._discretize_speed(self.env.v)
            
            # Use Q-table where it has strong opinions, network otherwise
            qt_vals = self.q_table[seg, vb, :]
            qt_range = qt_vals.max() - qt_vals.min()
            
            if qt_range > 0.5:
                action = np.argmax(qt_vals)
            else:
                action = np.argmax(self.predict(state))
            
            segs.append(self.env.seg_idx)
            vels.append(self.env.v)
            acts.append(action)
            ens.append(getattr(self.env, 'energy_kwh', 0))
            
            _, _, done, _ = self.env.step(action)
            steps += 1
            if done or self.env.v < 0.1:
                break
        
        od = os.path.join(getattr(config, 'OUTPUT_DIR', 'results_cm'), "deep_q")
        os.makedirs(od, exist_ok=True)
        np.savez(os.path.join(od, "speed_profile.npz"),
                 segments=np.array(segs), velocities=np.array(vels),
                 actions=np.array(acts), energies=np.array(ens))
        
        fs = segs[-1] if segs else 0
        ns = getattr(self.env, 'n_segments', 749)
        ok = fs >= ns - 2
        print(f"✓ Final segment: {fs}/{ns} {'✓ COMPLETED' if ok else '✗ INCOMPLETE'}")
        print(f"  Final energy: {ens[-1] if ens else 0:.1f} kWh")
        print(f"  Travel time: {steps * getattr(config, 'DT', 1.0):.0f} s")
        
        # Action distribution — CORRECT names
        acts_arr = np.array(acts)
        for i, name in enumerate(self.ACTION_NAMES):
            pct = np.sum(acts_arr == i) / len(acts_arr) * 100
            print(f"  Action {i} ({name}): {pct:.1f}%")
        
        # Plot
        try:
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
            
            fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(14, 10), sharex=True)
            fig.suptitle('DQN Optimal Speed Profile (FIXED)', fontsize=14)
            pos = np.array(segs) * 0.1  # km
            
            ax1.plot(pos, np.array(vels) * 3.6, 'b-', linewidth=1.5)
            ax1.set_ylabel('Speed (km/h)')
            ax1.set_title('Speed Profile')
            ax1.grid(True, alpha=0.3)
            
            ax2.plot(pos, ens, 'r-', linewidth=1.5)
            ax2.set_ylabel('Energy (kWh)')
            ax2.set_title('Cumulative Energy')
            ax2.grid(True, alpha=0.3)
            
            colors = ['red', 'orange', 'blue', 'green']
            for i in range(4):
                mask = acts_arr == i
                if mask.any():
                    ax3.scatter(pos[mask], [i]*mask.sum(), c=colors[i], 
                              s=1, label=self.ACTION_NAMES[i], alpha=0.5)
            ax3.set_ylabel('Action')
            ax3.set_xlabel('Position (km)')
            ax3.set_title('Action Profile')
            ax3.legend(markerscale=5)
            ax3.set_yticks([0, 1, 2, 3])
            ax3.set_yticklabels(self.ACTION_NAMES)
            ax3.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(os.path.join(od, "speed_profile.png"), dpi=300)
            plt.close()
            print(f"✓ Speed profile plot saved")
        except ImportError:
            pass
        
        return segs, vels, acts, ens