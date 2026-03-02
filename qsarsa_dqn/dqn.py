"""
Deep Q-Network for Train Speed Profile Optimization
=====================================================
Sequential execution with vectorized batch training.
Aggressive energy optimization targeting <1500 kWh.

Architecture (Figure 10):
    Input (x, v) → 128 tanh → 64 tanh → 16 tanh → 4 sigmoid

Three-Phase Training (Section 3.6):
    Phase 1: Q-SARSA episode collection + Q-table learning
    Phase 2: Distill Q-table into neural network
    Phase 3: Online DQN fine-tuning (Q-table guided)
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
        MAX_STEPS_PER_EPISODE = 2000
        N_ACTIONS = 4
        GAMMA = 0.99
        ALPHA = 0.5
        EPSILON_START = 1.0
        EPSILON_MIN = 0.01
        EPSILON_DECAY = 0.999


class DeepQNetwork:
    
    def __init__(self, env, phi_threshold=0.10):
        self.env = env
        self.phi = phi_threshold
        
        # Architecture
        self.n_inputs = 2
        self.n_actions = getattr(config, 'N_ACTIONS', 4)
        self.hidden_sizes = [128, 64, 16]
        
        # Hyperparameters
        self.gamma = getattr(config, 'GAMMA', 0.99)
        
        # Dynamic Learning Rate
        self.lr_max = 0.002       # Peak LR (after warm-up)
        self.lr_min = 0.00005     # Floor LR (end of training)
        self.lr = self.lr_max     # Current LR (updated each episode)
        self.lr_warmup_frac = 0.05  # Warm-up over first 5% of episodes
        self.lr_history = []
        self._recent_losses = []  # For spike detection
        
        # Dynamic Q-SARSA alpha
        self.alpha_max = 0.7
        self.alpha_min = 0.1
        
        self.epsilon = getattr(config, 'EPSILON_START', 1.0)
        self.epsilon_min = getattr(config, 'EPSILON_MIN', 0.01)
        self.epsilon_decay = getattr(config, 'EPSILON_DECAY', 0.999)
        self.reward_scale = 0.01
        self.grad_clip = 1.0
        self.td_clip = 10.0
        self.tau = 0.1
        
        # Replay buffer
        self.replay_buffer = []
        self.buffer_size = 50000
        self.batch_size = 128
        self.min_replay = 500
        
        # Networks
        self.weights = self._init_weights()
        self.target_weights = self._deep_copy_weights(self.weights)
        
        # Q-table: balanced init — must COMPLETE route first, then optimize energy
        n_segments = getattr(env, 'n_segments', 749)
        n_speeds = 50
        self.q_table = np.zeros((n_segments, n_speeds, self.n_actions))
        # Action 0=Power, 1=Cruise, 2=Coast, 3=Brake
        self.q_table[:, :, 0] = 1.5   # Power: preferred initially (must learn to move)
        self.q_table[:, :, 1] = 1.0   # Cruise: moderate
        self.q_table[:, :, 2] = 0.5   # Coast: lower initially
        self.q_table[:, :, 3] = -0.5  # Brake: discouraged
        self.prev_q_table = self.q_table.copy()
        self.cm_history = []
        
        # Extract route data for reward function
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
        
        # History
        self.success_history = []
        self.energy_history = []
        self.time_history = []
        self.loss_history = []
        self.reward_history = []
        self.training_data = []
        
        print(f"DQN initialized:")
        print(f"  Architecture: {self.n_inputs} → {' → '.join(map(str, self.hidden_sizes))} → {self.n_actions}")
        print(f"  Activations: tanh (hidden) + sigmoid (output)")
        print(f"  Learning rate: dynamic (warm-up → {self.lr_max} → cosine → {self.lr_min})")
        print(f"  Q-SARSA alpha: dynamic ({self.alpha_max} → cosine → {self.alpha_min})")
        print(f"  Gradient clip: ±{self.grad_clip}")
        print(f"  Reward scale: ×{self.reward_scale}")
        print(f"  Target network: soft update τ={self.tau}")
        print(f"  Batch size: {self.batch_size} (vectorized)")
        print(f"  Q-table init: Balanced (Power=1.5, Cruise=1.0, Coast=0.5)")
        print(f"  φ threshold: {self.phi}")
    
    # =====================================================
    # Neural Network
    # =====================================================
    
    def _init_weights(self):
        weights = {}
        sizes = [self.n_inputs] + self.hidden_sizes + [self.n_actions]
        for i in range(len(sizes) - 1):
            std = np.sqrt(2.0 / sizes[i])
            weights[f'W{i}'] = np.random.randn(sizes[i], sizes[i+1]) * std
            weights[f'b{i}'] = np.zeros(sizes[i+1])
        # Output bias: balanced start
        last = len(sizes) - 2
        weights[f'b{last}'] = np.array([0.5, 0.3, 0.0, -0.5])
        return weights
    
    def _deep_copy_weights(self, w):
        return {k: v.copy() for k, v in w.items()}
    
    def _forward_batch(self, X, weights=None):
        if weights is None: weights = self.weights
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
        w = self.target_weights if use_target else self.weights
        x = np.array(state, dtype=np.float64).reshape(1, -1)
        return self._forward_batch(x, w)[-1].flatten()
    
    def predict_batch(self, states, use_target=False):
        w = self.target_weights if use_target else self.weights
        X = np.array(states, dtype=np.float64)
        if X.ndim == 1: X = X.reshape(1, -1)
        return self._forward_batch(X, w)[-1]
    
    def _backward_batch(self, states, targets, lr=None):
        if lr is None: lr = self.lr
        bs = states.shape[0]
        acts = self._forward_batch(states)
        out = acts[-1]
        
        err = out - targets
        ae = np.abs(err)
        loss = np.where(ae <= 1.0, 0.5 * err**2, ae - 0.5).mean()
        
        d_out = np.where(ae <= 1.0, err, np.sign(err)) / self.n_actions
        d_out = np.clip(d_out, -self.grad_clip, self.grad_clip)
        
        n_layers = len(self.hidden_sizes) + 1
        delta = d_out * acts[-1] * (1.0 - acts[-1])  # sigmoid deriv
        
        for i in range(n_layers - 1, -1, -1):
            dW = np.clip((acts[i].T @ delta) / bs, -self.grad_clip, self.grad_clip)
            db = np.clip(delta.mean(axis=0), -self.grad_clip, self.grad_clip)
            self.weights[f'W{i}'] -= lr * dW
            self.weights[f'b{i}'] -= lr * db
            if i > 0:
                delta = (delta @ self.weights[f'W{i}'].T) * (1.0 - acts[i]**2)  # tanh deriv
                delta = np.clip(delta, -self.grad_clip, self.grad_clip)
        return loss
    
    def _soft_update_target(self):
        for k in self.weights:
            self.target_weights[k] = self.tau * self.weights[k] + (1 - self.tau) * self.target_weights[k]
    
    # =====================================================
    # State helpers
    # =====================================================
    
    def normalize_state(self, raw_state):
        if hasattr(raw_state, '__len__') and len(raw_state) >= 2:
            p, v = raw_state[0], raw_state[1]
        else:
            p = self.env.seg_idx / max(self.env.n_segments, 1)
            v = self.env.v / 120.0
        p = np.clip(p if p <= 1 else p / max(self.env.n_segments, 1), 0, 1)
        v = np.clip(v if v <= 1 else v / 120.0, 0, 1)
        return np.array([p, v], dtype=np.float64)
    
    def _discretize_speed(self, vel):
        return int(np.clip(vel / 120.0 * 49, 0, 49))
    
    # =====================================================
    # Dynamic Learning Rate
    # =====================================================
    
    def _get_lr(self, episode, total_episodes, phase=3):
        """
        Cosine annealing with warm-up and loss-spike adaptation.
        
        Schedule:
          Warm-up (0→5%):   linear ramp 0.0001 → lr_max
          Cosine  (5%→100%): smooth decay lr_max → lr_min
          
        Loss spike: if current loss > 3× running average → halve LR temporarily
        Phase 2: uses fixed 3×lr_max for supervised distillation
        """
        if phase == 2:
            return self.lr_max * 3  # Supervised: fast learning
        
        frac = episode / max(total_episodes, 1)
        warmup_end = self.lr_warmup_frac
        
        if frac < warmup_end:
            # Linear warm-up: 0.0001 → lr_max
            t = frac / warmup_end
            lr = 0.0001 + t * (self.lr_max - 0.0001)
        else:
            # Cosine annealing: lr_max → lr_min
            t = (frac - warmup_end) / (1.0 - warmup_end)
            lr = self.lr_min + 0.5 * (self.lr_max - self.lr_min) * (1 + np.cos(np.pi * t))
        
        # Loss-spike detection: if loss spikes, reduce LR temporarily
        if len(self._recent_losses) >= 10:
            avg_loss = np.mean(self._recent_losses[-10:])
            if self._recent_losses[-1] > 3.0 * avg_loss and avg_loss > 1e-8:
                lr *= 0.5  # Halve on spike
        
        lr = np.clip(lr, self.lr_min, self.lr_max)
        self.lr = lr
        self.lr_history.append(lr)
        return lr
    
    def _get_alpha(self, episode, total_episodes):
        """
        Dynamic Q-SARSA learning rate: cosine decay from alpha_max → alpha_min.
        Higher alpha early = faster Q-table convergence.
        Lower alpha later = stable refinement.
        """
        frac = episode / max(total_episodes, 1)
        alpha = self.alpha_min + 0.5 * (self.alpha_max - self.alpha_min) * (1 + np.cos(np.pi * frac))
        return alpha
    
    # =====================================================
    # Reward Function — AGGRESSIVE energy minimization
    # =====================================================
    
    def compute_reward(self, info, done):
        """
        TWO-STAGE reward for <1500 kWh.
        
        Problem with v1: energy_penalty(0.10) was 36x progress → agent
        learned "never power" → 0% success → no learning.
        
        Fix: Track success rate. Early on, focus on completion.
        Once completing reliably (>70%), ramp up energy penalty.
        
        Stage 1 (success < 70%): energy × 0.02, progress × 8
        Stage 2 (success > 70%): energy × 0.06, progress × 4
        """
        # Determine stage based on recent success
        n_recent = min(200, len(self.success_history))
        if n_recent >= 50:
            recent_success = sum(self.success_history[-n_recent:]) / n_recent
        else:
            recent_success = 0.0
        
        # Stage parameters
        if recent_success > 0.70:
            # Stage 2: Agent can complete → push hard on energy
            energy_coeff = 0.06
            progress_coeff = 4.0
            completion_base = 20.0
        else:
            # Stage 1: Must learn to complete first
            energy_coeff = 0.02
            progress_coeff = 8.0
            completion_base = 50.0
        
        if done and info.get('completed', False):
            total_energy = info.get('total_energy', getattr(self.env, 'energy_kwh', 2500))
            # Energy bonus: 2000→+50, 1500→+100, 1200→+130
            energy_bonus = max(0, (2500 - total_energy) / 10.0)
            if total_energy < 1500:
                energy_bonus += 50.0
            elif total_energy < 1800:
                energy_bonus += 20.0
            return (completion_base + energy_bonus) * self.reward_scale
        
        if info.get('violation', False) or info.get('backward', False):
            return -10.0 * self.reward_scale
        
        energy_step = info.get('energy_step', 0.0)
        progress = info.get('progress', 1.0 / max(self.env.n_segments, 1))
        action = info.get('action', -1)
        
        seg_idx = min(self.env.seg_idx, len(self.grades) - 1)
        grade = self.grades[seg_idx]
        current_v = getattr(self.env, 'v', 0)
        lim_val = self.limits[seg_idx] if seg_idx < len(self.limits) else 22.0
        limit = lim_val if lim_val > 1 else 22.0
        speed_ratio = current_v / max(limit, 1.0)
        
        # 1. PROGRESS
        progress_r = progress * progress_coeff
        
        # 2. ENERGY PENALTY (scales with stage)
        energy_p = energy_step * energy_coeff
        
        # 3. COAST BONUS (always active — nudges toward energy saving)
        coast_r = 0.0
        if action == 2 and current_v > 3.0:
            coast_r = 0.02
            if 0.4 <= speed_ratio <= 0.85:
                coast_r = 0.04  # Sweet spot
        
        # 4. GRADE-AWARE
        grade_r = 0.0
        if grade < -1.0:  # Downhill
            if action == 2:   grade_r = 0.03
            elif action == 3: grade_r = 0.01
            elif action == 0: grade_r = -0.04
        elif grade < -0.3:  # Slight downhill
            if action == 2:   grade_r = 0.02
            elif action == 0: grade_r = -0.02
        elif grade > 2.0:   # Steep uphill
            if action == 0 and speed_ratio < 0.5:
                grade_r = 0.01
        
        # 5. UNNECESSARY POWER PENALTY
        power_p = 0.0
        if action == 0:
            if speed_ratio > 0.7:
                power_p = -0.03
            elif speed_ratio > 0.5 and grade <= 0:
                power_p = -0.015
        
        # 6. SPEED EFFICIENCY
        speed_r = 0.0
        if 0.5 <= speed_ratio <= 0.75:
            speed_r = 0.003
        elif speed_ratio > 0.95:
            speed_r = -0.008
        
        reward = progress_r - energy_p + coast_r + grade_r + power_p + speed_r
        return reward * self.reward_scale
    
    # =====================================================
    # Q-SARSA
    # =====================================================
    
    def _compute_cm(self, ep):
        if ep < 2: return 0.0
        dq = np.sum(np.abs(self.q_table - self.prev_q_table))
        self.prev_q_table = self.q_table.copy()
        cm = dq / self.cm_history[-1] if self.cm_history and self.cm_history[-1] > 1e-10 else 1.0
        self.cm_history.append(dq)
        return cm
    
    def _qsarsa_update(self, seg, vb, a, r, ns, nvb, na, cm, alpha=0.5):
        if cm > self.phi:
            target = r + self.gamma * self.q_table[ns, nvb, na]
        else:
            target = r + self.gamma * np.max(self.q_table[ns, nvb, :])
        self.q_table[seg, vb, a] += alpha * (target - self.q_table[seg, vb, a])
    
    # =====================================================
    # Replay buffer + vectorized training
    # =====================================================
    
    def _store_exp(self, s, a, r, ns, d):
        self.replay_buffer.append((s, a, r, ns, d))
        if len(self.replay_buffer) > self.buffer_size:
            self.replay_buffer.pop(0)
    
    def _train_batch(self):
        if len(self.replay_buffer) < self.min_replay:
            return 0.0
        idx = np.random.choice(len(self.replay_buffer), self.batch_size, replace=False)
        batch = [self.replay_buffer[i] for i in idx]
        
        states = np.array([b[0] for b in batch], dtype=np.float64)
        actions = np.array([b[1] for b in batch], dtype=np.int32)
        rewards = np.array([b[2] for b in batch], dtype=np.float64)
        nstates = np.array([b[3] for b in batch], dtype=np.float64)
        dones = np.array([b[4] for b in batch], dtype=bool)
        
        cur_q = self.predict_batch(states)
        nxt_q = self.predict_batch(nstates, use_target=True)
        
        tv = rewards.copy()
        tv[~dones] += self.gamma * np.max(nxt_q[~dones], axis=1)
        
        tgt = cur_q.copy()
        for i in range(self.batch_size):
            td = np.clip(tv[i] - cur_q[i, actions[i]], -self.td_clip, self.td_clip)
            tgt[i, actions[i]] = cur_q[i, actions[i]] + td
        
        loss = self._backward_batch(states, tgt, lr=self.lr)  # uses current dynamic LR
        self._soft_update_target()
        
        # Track loss for spike detection
        self._recent_losses.append(loss)
        if len(self._recent_losses) > 100:
            self._recent_losses = self._recent_losses[-100:]
        
        return loss
    
    # =====================================================
    # Training
    # =====================================================
    
    def train(self, episodes=5000):
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
        print(f"PHASE 3: Online DQN Fine-tuning ({p3} episodes, Q-table guided)")
        print(f"{'='*70}")
        self._phase3(p3, start_ep=p1)
        
        self._save_results()
    
    def _phase1(self, episodes):
        successes = 0
        t0 = time.time()
        
        for ep in range(episodes):
            self.env.reset()
            ep_data = []
            total_r = 0
            done = False
            steps = 0
            cm = self._compute_cm(ep)
            alpha = self._get_alpha(ep, episodes)  # Dynamic Q-SARSA alpha
            prev_e = 0.0
            info = {}
            
            while not done and steps < getattr(config, 'MAX_STEPS_PER_EPISODE', 2000):
                raw = self.env._get_state()
                state = self.normalize_state(raw)
                seg = min(self.env.seg_idx, self.q_table.shape[0] - 1)
                vb = self._discretize_speed(self.env.v)
                
                if np.random.random() < self.epsilon:
                    action = np.random.randint(self.n_actions)
                else:
                    action = np.argmax(self.q_table[seg, vb, :])
                
                _, _, done, info = self.env.step(action)
                if not isinstance(info, dict): info = {}
                nstate = self.normalize_state(self.env._get_state() if hasattr(self.env, '_get_state') else [0,0])
                
                ce = getattr(self.env, 'energy_kwh', 0)
                info['energy_step'] = max(0, ce - prev_e)
                info['total_energy'] = ce
                info['action'] = action
                info['progress'] = info.get('progress', 1.0 / max(self.env.n_segments, 1))
                prev_e = ce
                
                reward = self.compute_reward(info, done)
                ep_data.append((state, action, reward))
                total_r += reward
                
                ns = min(self.env.seg_idx, self.q_table.shape[0] - 1)
                nvb = self._discretize_speed(self.env.v)
                na = np.random.randint(self.n_actions) if np.random.random() < self.epsilon \
                     else np.argmax(self.q_table[ns, nvb, :])
                self._qsarsa_update(seg, vb, action, reward / self.reward_scale, ns, nvb, na, cm, alpha)
                self._store_exp(state, action, reward, nstate, done)
                steps += 1
            
            completed = (info.get('completed', False) if isinstance(info, dict) else False) or \
                        self.env.seg_idx >= self.env.n_segments - 1
            if completed: successes += 1
            
            self.success_history.append(completed)
            self.energy_history.append(getattr(self.env, 'energy_kwh', 0))
            self.reward_history.append(total_r)
            
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
        n_batches = 2000
        print(f"  Distilling Q-table ({n_batches} batches)...")
        
        visited_s, visited_q = [], []
        ns, nv = self.q_table.shape[0], self.q_table.shape[1]
        init = np.array([1.5, 1.0, 0.5, -0.5])  # match Q-table init
        
        for seg in range(ns):
            for vb in range(nv):
                q = self.q_table[seg, vb, :]
                # Check if visited (differs from init)
                if not np.allclose(q, init, atol=0.05):
                    visited_s.append([seg / ns, vb / max(nv - 1, 1)])
                    visited_q.append(q)
        
        print(f"  Found {len(visited_s)} visited states")
        
        if len(visited_s) < 10:
            print("  Too few states, using episode data...")
            good = [d for d in self.training_data if d[3]]
            if len(good) < 50: good = self.training_data
            ss = np.array([d[0] for d in good])
            aa = np.array([d[1] for d in good])
            for b in range(n_batches):
                ix = np.random.choice(len(good), min(self.batch_size, len(good)), replace=True)
                tgt = np.full((len(ix), self.n_actions), 0.2)
                for i in range(len(ix)):
                    tgt[i, aa[ix[i]]] = 0.9
                loss = self._backward_batch(ss[ix], tgt)
                if (b+1) % 200 == 0:
                    print(f"  Batch {b+1}/{n_batches}: Loss={loss:.6f}")
        else:
            S = np.array(visited_s, dtype=np.float64)
            Q = np.array(visited_q, dtype=np.float64)
            qmin, qmax = Q.min(), Q.max()
            Qn = (Q - qmin) / max(qmax - qmin, 1e-6) * 0.8 + 0.1
            
            for b in range(n_batches):
                ix = np.random.choice(len(S), min(self.batch_size, len(S)), replace=True)
                p2_lr = self._get_lr(b, n_batches, phase=2)
                loss = self._backward_batch(S[ix], Qn[ix], lr=p2_lr)
                if (b+1) % 200 == 0:
                    tix = np.random.choice(len(S), min(200, len(S)), replace=False)
                    net_a = np.argmax(self.predict_batch(S[tix]), axis=1)
                    qt_a = np.argmax(Qn[tix], axis=1)
                    agr = np.mean(net_a == qt_a)
                    print(f"  Batch {b+1}/{n_batches}: Loss={loss:.6f}, Agreement={agr:.0%}")
                    if agr > 0.85:
                        print(f"  ✓ Early stop at {agr:.0%}")
                        break
        
        self.target_weights = self._deep_copy_weights(self.weights)
        print("  Phase 2 Complete")
    
    def _phase3(self, episodes, start_ep=0):
        successes = 0
        t0 = time.time()
        net_use = 0.0
        max_steps = getattr(config, 'MAX_STEPS_PER_EPISODE', 2000)
        
        print(f"  Strategy: Q-table guided → gradual network takeover")
        
        for ep in range(episodes):
            # Update dynamic learning rate
            self._get_lr(ep, episodes, phase=3)
            
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
                
                if np.random.random() < self.epsilon:
                    action = np.random.randint(self.n_actions)
                elif np.random.random() < net_use:
                    action = np.argmax(self.predict(state))
                else:
                    action = np.argmax(self.q_table[seg, vb, :])
                
                _, _, done, info = self.env.step(action)
                if not isinstance(info, dict): info = {}
                nstate = self.normalize_state(self.env._get_state() if hasattr(self.env, '_get_state') else [0,0])
                
                ce = getattr(self.env, 'energy_kwh', 0)
                info['energy_step'] = max(0, ce - prev_e)
                info['total_energy'] = ce
                info['action'] = action
                info['progress'] = info.get('progress', 1.0 / max(self.env.n_segments, 1))
                prev_e = ce
                
                reward = self.compute_reward(info, done)
                total_r += reward
                self._store_exp(state, action, reward, nstate, done)
                
                if len(self.replay_buffer) >= self.min_replay and steps % 8 == 0:
                    loss = self._train_batch()
                    total_loss += loss
                    n_upd += 1
                
                steps += 1
            
            completed = (info.get('completed', False) if isinstance(info, dict) else False) or \
                        self.env.seg_idx >= self.env.n_segments - 1
            if completed: successes += 1
            
            self.success_history.append(completed)
            self.energy_history.append(getattr(self.env, 'energy_kwh', 0))
            self.time_history.append(steps)
            self.loss_history.append(total_loss / max(n_upd, 1))
            self.reward_history.append(total_r)
            
            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
            
            if (ep+1) % 50 == 0 and ep >= 100:
                r50 = sum(self.success_history[-50:]) / 50
                if r50 > 0.8: net_use = min(net_use + 0.1, 0.9)
                elif r50 < 0.5: net_use = max(net_use - 0.1, 0.0)
            
            if (ep+1) % 100 == 0:
                el = time.time() - t0
                eps = (ep+1) / el
                eta = (episodes - ep - 1) / max(eps, 0.01)
                n = min(100, len(self.success_history))
                rs = sum(self.success_history[-n:]) / n
                re_vals = [self.energy_history[-n+i] for i in range(n) if self.success_history[-n+i]]
                re = np.mean(re_vals) if re_vals else 0
                gep = start_ep + ep + 1
                print(f"  Phase 3 Ep {ep+1}/{episodes} (Global {gep}): "
                      f"Success={rs:.0%}, Energy={re:.0f} kWh, "
                      f"Loss={self.loss_history[-1]:.6f}, LR={self.lr:.6f}, "
                      f"ε={self.epsilon:.3f}, NetUse={net_use:.0%}, "
                      f"Speed={eps:.1f} ep/s, ETA={eta:.0f}s")
        
        ts = sum(self.success_history)
        te = len(self.success_history)
        print(f"\n  Phase 3 Complete: {successes}/{episodes}")
        print(f"  Overall: {ts}/{te} ({ts/max(te,1):.0%})")
        print(f"  Final network usage: {net_use:.0%}")
        print(f"  Total time: {time.time()-t0:.0f}s")
    
    # =====================================================
    # Save & Visualize
    # =====================================================
    
    def _save_results(self):
        od = os.path.join(getattr(config, 'OUTPUT_DIR', 'results_cm'), "deep_q")
        os.makedirs(od, exist_ok=True)
        
        sd = {'success_history': np.array(self.success_history),
              'energy_history': np.array(self.energy_history),
              'reward_history': np.array(self.reward_history),
              'loss_history': np.array(self.loss_history) if self.loss_history else np.array([])}
        for k, v in self.weights.items(): sd[f'weight_{k}'] = v
        np.savez(os.path.join(od, "dqn_weights.npz"), **sd)
        
        try:
            import matplotlib; matplotlib.use('Agg')
            import matplotlib.pyplot as plt
            
            fig, axes = plt.subplots(2, 2, figsize=(14, 10))
            fig.suptitle('DQN Training Results', fontsize=14)
            er = range(len(self.success_history))
            
            if len(self.success_history) > 100:
                roll = [sum(self.success_history[max(0,i-100):i+1]) / min(i+1,100)
                       for i in range(len(self.success_history))]
                axes[0,0].plot(er, roll, 'b-', linewidth=1)
            axes[0,0].set_title('Success Rate (Rolling 100)')
            axes[0,0].set_ylim(-0.05, 1.05); axes[0,0].grid(True, alpha=0.3)
            
            vl = [l for l in self.loss_history if l > 0]
            if vl:
                axes[0,1].plot(range(len(vl)), vl, 'r-', alpha=0.5, linewidth=0.5)
                axes[0,1].set_title('Training Loss & Learning Rate')
                axes[0,1].set_yscale('log')
                axes[0,1].grid(True, alpha=0.3)
                # LR on secondary axis
                if self.lr_history:
                    ax_lr = axes[0,1].twinx()
                    ax_lr.plot(range(len(self.lr_history)), self.lr_history, 
                              'b-', alpha=0.6, linewidth=1, label='LR')
                    ax_lr.set_ylabel('Learning Rate', color='blue')
                    ax_lr.tick_params(axis='y', labelcolor='blue')
            
            se = [i for i, s in enumerate(self.success_history) if s]
            if se:
                axes[1,0].plot(se, [self.energy_history[i] for i in se], 'g.', alpha=0.5, ms=2)
                axes[1,0].set_title('Energy (Successful Episodes)')
                axes[1,0].axhline(y=1500, color='r', linestyle='--', label='Target: 1500 kWh')
                axes[1,0].legend(); axes[1,0].grid(True, alpha=0.3)
            
            axes[1,1].plot(er, self.reward_history, 'm-', alpha=0.3, linewidth=0.5)
            if len(self.reward_history) > 100:
                rr = [np.mean(self.reward_history[max(0,i-100):i+1])
                     for i in range(len(self.reward_history))]
                axes[1,1].plot(er, rr, 'm-', linewidth=2)
            axes[1,1].set_title('Reward per Episode'); axes[1,1].grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(os.path.join(od, "dqn_training.png"), dpi=300)
            plt.close()
            print(f"\n✓ Training plots saved")
        except ImportError: pass
        
        print(f"✓ Results saved to: {od}/")
    
    def generate_speed_profile(self):
        print("\n📊 Generating optimal speed profile (DQN)...")
        self.env.reset()
        segs, vels, acts, ens = [], [], [], []
        steps = 0
        
        while steps < getattr(config, 'MAX_STEPS_PER_EPISODE', 2000):
            state = self.normalize_state(self.env._get_state())
            seg = min(self.env.seg_idx, self.q_table.shape[0] - 1)
            vb = self._discretize_speed(self.env.v)
            
            qt_a = np.argmax(self.q_table[seg, vb, :])
            net_a = np.argmax(self.predict(state))
            qr = self.q_table[seg, vb, :].max() - self.q_table[seg, vb, :].min()
            action = net_a if qr < 0.1 else qt_a
            
            segs.append(self.env.seg_idx)
            vels.append(self.env.v)
            acts.append(action)
            ens.append(getattr(self.env, 'energy_kwh', 0))
            
            _, _, done, _ = self.env.step(action)
            steps += 1
            if done or self.env.v < 0.1: break
        
        od = os.path.join(getattr(config, 'OUTPUT_DIR', 'results_cm'), "deep_q")
        os.makedirs(od, exist_ok=True)
        np.savez(os.path.join(od, "speed_profile.npz"),
                 segments=np.array(segs), velocities=np.array(vels),
                 actions=np.array(acts), energies=np.array(ens))
        
        fs = segs[-1] if segs else 0
        ns = getattr(self.env, 'n_segments', 749)
        ok = fs >= ns - 1
        print(f"✓ Final segment: {fs}/{ns} {'✓ COMPLETED' if ok else '✗ INCOMPLETE'}")
        print(f"  Final energy: {ens[-1] if ens else 0:.1f} kWh")
        
        # Action distribution
        acts_arr = np.array(acts)
        for i, name in enumerate(['Power', 'Cruise', 'Coast', 'Brake']):
            pct = np.sum(acts_arr == i) / len(acts_arr) * 100
            print(f"  Action {i} ({name}): {pct:.1f}%")
        
        try:
            import matplotlib; matplotlib.use('Agg')
            import matplotlib.pyplot as plt
            fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(14, 10), sharex=True)
            fig.suptitle('DQN Optimal Speed Profile', fontsize=14)
            pos = np.array(segs) * 0.1
            
            ax1.plot(pos, vels, 'b-', linewidth=1.5)
            ax1.set_ylabel('Speed (m/s)'); ax1.set_title('Speed Profile')
            ax1.grid(True, alpha=0.3)
            
            ax2.plot(pos, ens, 'r-', linewidth=1.5)
            ax2.axhline(y=1500, color='orange', linestyle='--', label='Target')
            ax2.set_ylabel('Energy (kWh)'); ax2.set_title('Cumulative Energy')
            ax2.legend(); ax2.grid(True, alpha=0.3)
            
            colors = ['red', 'orange', 'green', 'blue']
            labels = ['Power', 'Cruise', 'Coast', 'Brake']
            for i in range(4):
                mask = acts_arr == i
                if mask.any():
                    ax3.scatter(pos[mask], [i]*mask.sum(), c=colors[i], s=1, label=labels[i], alpha=0.5)
            ax3.set_ylabel('Action'); ax3.set_xlabel('Position (km)')
            ax3.set_title('Action Profile'); ax3.legend(markerscale=5)
            ax3.set_yticks([0,1,2,3]); ax3.set_yticklabels(labels)
            ax3.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(os.path.join(od, "speed_profile.png"), dpi=300)
            plt.close()
            print(f"✓ Speed profile plot saved")
        except ImportError: pass
        
        return segs, vels, acts, ens