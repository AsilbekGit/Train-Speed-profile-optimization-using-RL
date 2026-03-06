"""
Deep Q-Network for Train Speed Profile Optimization — v2 PRAGMATIC
====================================================================
Goal: Energy consumption < 1500 kWh

Strategy:
  Phase 1: Clone Q-SARSA EXACTLY (proven to work: eps=0.10, env reward)
           -> Gets ~90%+ success rate, ~2000-3000 kWh
  Phase 2: Distill Q-table into neural network  
  Phase 3: Energy optimization with network
           -> Penalize Power, reward Coast, target < 1500 kWh

Action mapping (matches environment.py):
  0 = Brake, 1 = Coast, 2 = Cruise, 3 = Power
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
        VEL_BIN_SIZE = 0.5
        DT = 1.0
        DX = 100.0
        MAX_SPEED_MS = 36.1
        ACTION_NAMES = ['Brake', 'Coast', 'Cruise', 'Power']

try:
    from data.utils import discretize_state
except ImportError:
    def discretize_state(state):
        s_idx = int(state[0])
        v_idx = int(state[1] / getattr(config, 'VEL_BIN_SIZE', 0.5))
        return s_idx, min(max(v_idx, 0), 99)


class DeepQNetwork:

    # Action constants (match environment.py)
    A_BRAKE  = 0
    A_COAST  = 1
    A_CRUISE = 2
    A_POWER  = 3
    A_NAMES  = ['Brake', 'Coast', 'Cruise', 'Power']

    def __init__(self, env, phi_threshold=0.10):
        self.env = env
        self.phi = phi_threshold
        self.n_segments = getattr(env, 'n_segments', 749)

        # -- Network architecture (Fig. 10): 2 -> 128 -> 64 -> 16 -> 4 --
        self.n_inputs  = 2
        self.n_actions = 4
        self.hidden    = [128, 64, 16]

        self.weights        = self._init_weights()
        self.target_weights = self._copy_w(self.weights)

        # -- Q-table (same shape & init as Q-SARSA) --
        self.q = np.zeros((self.n_segments, 100, 4))
        self.q[:, :, self.A_POWER]  = 5.0
        self.q[:, :, self.A_CRUISE] = 2.0
        self.q[:, :, self.A_COAST]  = 0.5
        self.q[:, :, self.A_BRAKE]  = -2.0
        self.q_prev = self.q.copy()

        # -- Q-SARSA hypers (MATCH qsarsa.py exactly) --
        self.alpha   = 0.1
        self.gamma   = 0.95
        self.epsilon = 0.10      # Low from start - trust the bias!

        # -- Network training hypers --
        self.lr       = 0.001
        self.lr_min   = 0.00005
        self.lr_max   = 0.002
        self.grad_clip = 1.0
        self.tau       = 0.005   # Very slow target update

        # -- Replay --
        self.replay  = []
        self.buf_max = 80000
        self.batch   = 128

        # -- Route data for energy-aware reward --
        self.grades = getattr(env, 'grades', np.zeros(self.n_segments))
        self.limits = getattr(env, 'limits', np.full(self.n_segments, 22.0))

        # -- History --
        self.success_history = []
        self.energy_history  = []
        self.time_history    = []
        self.reward_history  = []
        self.loss_history    = []

        # -- CM tracking --
        self.delta_history = []
        self.cm_history    = []
        self.use_sarsa = 0
        self.use_qlearn = 0

        print(f"DQN v2 (Pragmatic) initialized:")
        print(f"  Net: {self.n_inputs}->{'->'.join(map(str,self.hidden))}->{self.n_actions} (tanh+linear)")
        print(f"  Actions: 0=Brake 1=Coast 2=Cruise 3=Power")
        print(f"  Q-table: {self.n_segments}x100x4  init=[Br=-2, Co=0.5, Cr=2, Pw=5]")
        print(f"  Phase 1: a={self.alpha}, g={self.gamma}, e={self.epsilon} (clone Q-SARSA)")
        print(f"  phi={self.phi}")

    # ================================================================
    # Neural network (tanh hidden + LINEAR output)
    # ================================================================

    def _init_weights(self):
        w = {}
        sizes = [self.n_inputs] + self.hidden + [self.n_actions]
        for i in range(len(sizes)-1):
            std = np.sqrt(2.0 / sizes[i])
            w[f'W{i}'] = np.random.randn(sizes[i], sizes[i+1]) * std
            w[f'b{i}'] = np.zeros(sizes[i+1])
        return w

    def _copy_w(self, w):
        return {k: v.copy() for k, v in w.items()}

    def _forward(self, X, w=None):
        if w is None: w = self.weights
        nl = len(self.hidden) + 1
        a = [X]
        for i in range(nl):
            z = a[-1] @ w[f'W{i}'] + w[f'b{i}']
            if i < nl - 1:
                a.append(np.tanh(np.clip(z, -20, 20)))
            else:
                a.append(z)          # LINEAR output
        return a

    def predict(self, s, target=False):
        w = self.target_weights if target else self.weights
        x = np.atleast_2d(np.array(s, dtype=np.float64))
        return self._forward(x, w)[-1]

    def _backprop(self, S, T, lr=None):
        if lr is None: lr = self.lr
        bs = S.shape[0]
        a = self._forward(S)
        out = a[-1]
        err = out - T
        ae = np.abs(err)
        loss = np.where(ae<=1, 0.5*err**2, ae-0.5).mean()

        d = np.where(ae<=1, err, np.sign(err)) / self.n_actions
        d = np.clip(d, -self.grad_clip, self.grad_clip)

        nl = len(self.hidden) + 1
        delta = d  # linear output -> deriv = 1
        for i in range(nl-1, -1, -1):
            dW = np.clip((a[i].T @ delta)/bs, -self.grad_clip, self.grad_clip)
            db = np.clip(delta.mean(0), -self.grad_clip, self.grad_clip)
            self.weights[f'W{i}'] -= lr * dW
            self.weights[f'b{i}'] -= lr * db
            if i > 0:
                delta = (delta @ self.weights[f'W{i}'].T) * (1.0 - a[i]**2)
                delta = np.clip(delta, -self.grad_clip, self.grad_clip)
        return loss

    def _soft_update(self):
        for k in self.weights:
            self.target_weights[k] = self.tau*self.weights[k] + (1-self.tau)*self.target_weights[k]

    # ================================================================
    # State helpers
    # ================================================================

    def _norm(self, raw):
        """Normalize state to [0,1] for network."""
        p = raw[0] / max(self.n_segments, 1)
        v = raw[1] / getattr(config, 'MAX_SPEED_MS', 36.1)
        return np.array([np.clip(p, 0, 1), np.clip(v, 0, 1)], dtype=np.float64)

    def _vbin(self, vel):
        return int(np.clip(vel / getattr(config, 'VEL_BIN_SIZE', 0.5), 0, 99))

    # ================================================================
    # Q-SARSA update (Eq. 30)
    # ================================================================

    def _cm(self):
        if len(self.delta_history) < 2:
            return float('inf')
        d0 = self.delta_history[-2]
        d1 = self.delta_history[-1]
        return d1 / d0 if d0 > 1e-9 else 1.0

    def _qsarsa_step(self, s, vb, a, r, ns, nvb, na, done):
        """Single Q-SARSA update with CM switching."""
        cm = self._cm()
        if cm > self.phi:
            tgt = r if done else r + self.gamma * self.q[ns, nvb, na]
            self.use_sarsa += 1
        else:
            tgt = r if done else r + self.gamma * np.max(self.q[ns, nvb, :])
            self.epsilon = min(0.20, self.epsilon * 1.05)
            self.use_qlearn += 1
        self.q[s, vb, a] += self.alpha * (tgt - self.q[s, vb, a])

    # ================================================================
    # Replay buffer
    # ================================================================

    def _store(self, s, a, r, ns, d):
        self.replay.append((s, a, r, ns, d))
        if len(self.replay) > self.buf_max:
            self.replay.pop(0)

    def _train_net(self):
        if len(self.replay) < 1000:
            return 0.0
        idx = np.random.choice(len(self.replay), self.batch, replace=False)
        batch = [self.replay[i] for i in idx]

        S  = np.array([b[0] for b in batch], dtype=np.float64)
        A  = np.array([b[1] for b in batch], dtype=np.int32)
        R  = np.array([b[2] for b in batch], dtype=np.float64)
        NS = np.array([b[3] for b in batch], dtype=np.float64)
        D  = np.array([b[4] for b in batch], dtype=bool)

        cur = self.predict(S).copy()
        nxt = self.predict(NS, target=True)

        tv = R.copy()
        tv[~D] += self.gamma * np.max(nxt[~D], axis=1)

        tgt = cur.copy()
        for i in range(self.batch):
            td = np.clip(tv[i] - cur[i, A[i]], -10, 10)
            tgt[i, A[i]] = cur[i, A[i]] + td

        loss = self._backprop(S, tgt, lr=self.lr)
        self._soft_update()
        return loss

    # ================================================================
    # TRAINING
    # ================================================================

    def train(self, episodes=5000):
        p1_eps = int(episodes * 0.40)   # 40% Q-SARSA (learn to complete)
        p3_eps = episodes - p1_eps      # 60% DQN fine-tuning (minimize energy)

        print(f"\n{'='*70}")
        print(f"PHASE 1: Q-SARSA -- Learn to Complete Route ({p1_eps} episodes)")
        print(f"  Using SAME settings as your working Q-SARSA")
        print(f"  eps=0.10, alpha=0.1, gamma=0.95, env reward")
        print(f"{'='*70}")
        self._phase1(p1_eps)

        print(f"\n{'='*70}")
        print(f"PHASE 2: Distill Q-table -> Network")
        print(f"{'='*70}")
        self._phase2()

        print(f"\n{'='*70}")
        print(f"PHASE 3: DQN Energy Optimization ({p3_eps} episodes)")
        print(f"  Target: < 1500 kWh")
        print(f"{'='*70}")
        self._phase3(p3_eps)

        self._save()

    # -- PHASE 1: Exact Q-SARSA clone ----------------------------

    def _phase1(self, episodes):
        """
        Run Q-SARSA with EXACTLY the same settings as qsarsa.py.
        Uses the environment's built-in reward (env.step returns it).
        """
        successes = 0
        best_energy = float('inf')
        t0 = time.time()
        max_steps = getattr(config, 'MAX_STEPS_PER_EPISODE', 20000)

        for ep in range(1, episodes + 1):
            self.env.reset()
            state = self.env._get_state()
            s_idx, v_idx = discretize_state(state)

            # Pick first action
            if np.random.rand() < self.epsilon:
                action = np.random.randint(4)
            else:
                action = np.argmax(self.q[s_idx, v_idx])

            total_reward = 0.0
            steps = 0

            while steps < max_steps:
                # -- Take action, get ENVIRONMENT reward --
                next_state, reward, done, info = self.env.step(action)
                ns_idx, nv_idx = discretize_state(next_state)
                total_reward += reward
                steps += 1

                # Next action (SARSA-style)
                if np.random.rand() < self.epsilon:
                    next_action = np.random.randint(4)
                else:
                    next_action = np.argmax(self.q[ns_idx, nv_idx])

                # Q-SARSA update
                self._qsarsa_step(s_idx, v_idx, action, reward,
                                  ns_idx, nv_idx, next_action, done)

                # Also store normalized for network training later
                sn  = self._norm(state)
                nsn = self._norm(next_state)
                self._store(sn, action, reward * 0.01, nsn, done)

                if done:
                    break
                if self.env.v < 0.1 and steps > 100:
                    break

                s_idx, v_idx = ns_idx, nv_idx
                action = next_action
                state = next_state

            # Delta tracking (for CM)
            delta = np.sum(np.abs(self.q - self.q_prev))
            self.delta_history.append(delta)
            self.q_prev = self.q.copy()

            ep_success = self.env.seg_idx >= self.n_segments - 2
            if ep_success:
                successes += 1
                e = info.get('energy', getattr(self.env, 'energy_kwh', 0))
                if e < best_energy:
                    best_energy = e

            self.success_history.append(ep_success)
            self.energy_history.append(info.get('energy', getattr(self.env, 'energy_kwh', 0)))
            self.time_history.append(info.get('time', getattr(self.env, 't', 0)))
            self.reward_history.append(total_reward)

            self.epsilon = max(0.02, self.epsilon * 0.999)

            if ep % 200 == 0 or ep <= 10:
                el = time.time() - t0
                rate = successes / ep * 100
                cm = self._cm()
                mode = "SARSA" if cm > self.phi else "Q-Learn"
                recent_e = [self.energy_history[-i] for i in range(1, min(101, ep+1))
                           if self.success_history[-i]]
                avg_e = np.mean(recent_e) if recent_e else 0
                print(f"  Ep {ep:4d}/{episodes} | Success={rate:5.1f}% | "
                      f"Energy={avg_e:6.0f} kWh | CM={cm:.4f} ({mode}) | "
                      f"eps={self.epsilon:.3f} | Best={best_energy:.0f} kWh | "
                      f"{el:.0f}s")

        rate = successes / episodes * 100
        print(f"\n  Phase 1 done: {successes}/{episodes} ({rate:.1f}%) in {time.time()-t0:.0f}s")
        print(f"  Best energy: {best_energy:.0f} kWh")
        print(f"  SARSA: {self.use_sarsa}, Q-learn: {self.use_qlearn}")

    # -- PHASE 2: Distill Q-table -> Network --------------------

    def _phase2(self):
        # Collect visited Q-table entries
        visited_s, visited_q = [], []
        init = np.array([-2.0, 0.5, 2.0, 5.0])
        for seg in range(self.n_segments):
            for vb in range(100):
                q = self.q[seg, vb, :]
                if not np.allclose(q, init, atol=0.1):
                    visited_s.append([seg / self.n_segments, vb / 99.0])
                    visited_q.append(q)

        print(f"  Visited states: {len(visited_s)}")

        if len(visited_s) < 50:
            print("  WARNING: Too few visited states! Using random sampling.")
            for _ in range(5000):
                seg = np.random.randint(self.n_segments)
                vb  = np.random.randint(100)
                visited_s.append([seg / self.n_segments, vb / 99.0])
                visited_q.append(self.q[seg, vb, :])

        S = np.array(visited_s, dtype=np.float64)
        Q = np.array(visited_q, dtype=np.float64)

        # Scale Q-values to reasonable range for linear output
        qmin, qmax = Q.min(), Q.max()
        qrange = max(qmax - qmin, 1e-6)
        Qn = (Q - qmin) / qrange * 4.0 - 2.0   # Map to [-2, 2]

        n_batches = 5000
        best_agr = 0
        for b in range(n_batches):
            ix = np.random.choice(len(S), min(self.batch, len(S)), replace=True)
            frac = b / n_batches
            lr = self.lr_max * 3 if frac < 0.1 else self.lr_max * (1 - frac)
            lr = max(lr, self.lr_min)

            loss = self._backprop(S[ix], Qn[ix], lr=lr)

            if (b+1) % 1000 == 0:
                tix = np.random.choice(len(S), min(500, len(S)), replace=False)
                net_a = np.argmax(self.predict(S[tix]), axis=1)
                qt_a  = np.argmax(Qn[tix], axis=1)
                agr = np.mean(net_a == qt_a)
                best_agr = max(best_agr, agr)
                print(f"  Batch {b+1}/{n_batches}: Loss={loss:.6f}, Agreement={agr:.0%}")
                if agr > 0.92:
                    print(f"  Early stop at {agr:.0%}")
                    break

        self.target_weights = self._copy_w(self.weights)
        print(f"  Phase 2 done. Best agreement: {best_agr:.0%}")

    # -- PHASE 3: DQN energy optimization ----------------------

    def _phase3(self, episodes):
        """
        Fine-tune using DQN with energy-aware reward.
        Q-table still guides exploration; network gradually takes over.
        """
        successes = 0
        best_energy = float('inf')
        t0 = time.time()
        max_steps = getattr(config, 'MAX_STEPS_PER_EPISODE', 20000)

        # Reset epsilon for Phase 3 exploration
        self.epsilon = 0.08

        for ep in range(1, episodes + 1):
            self.env.reset()
            total_r = 0.0
            total_loss = 0.0
            n_upd = 0
            done = False
            steps = 0
            prev_energy = 0.0

            # Gradually shift from Q-table -> network
            net_frac = min(0.8, 0.1 + 0.7 * (ep / episodes))

            while not done and steps < max_steps:
                raw = self.env._get_state()
                state_n = self._norm(raw)
                seg = min(self.env.seg_idx, self.n_segments - 1)
                vb  = self._vbin(self.env.v)

                # -- Action selection: eps-greedy with Q-table/network blend --
                if np.random.rand() < self.epsilon:
                    action = np.random.randint(4)
                elif np.random.rand() < net_frac:
                    action = int(np.argmax(self.predict(state_n).flatten()))
                else:
                    action = int(np.argmax(self.q[seg, vb, :]))

                # Step
                _, env_reward, done, info = self.env.step(action)
                new_raw = self.env._get_state()
                new_state_n = self._norm(new_raw)

                cur_energy = getattr(self.env, 'energy_kwh', 0)
                energy_step = cur_energy - prev_energy
                prev_energy = cur_energy

                # -- ENERGY-AWARE REWARD --
                reward = self._energy_reward(env_reward, done, action,
                                              energy_step, seg, ep, episodes)
                total_r += reward

                self._store(state_n, action, reward, new_state_n, done)

                # Also keep Q-table updated
                ns = min(self.env.seg_idx, self.n_segments - 1)
                nvb = self._vbin(self.env.v)
                na = int(np.argmax(self.q[ns, nvb, :]))
                self._qsarsa_step(seg, vb, action, env_reward,
                                  ns, nvb, na, done)

                # Train network every 4 steps
                if steps % 4 == 0:
                    loss = self._train_net()
                    total_loss += loss
                    n_upd += 1

                steps += 1

            # Track delta for CM
            delta = np.sum(np.abs(self.q - self.q_prev))
            self.delta_history.append(delta)
            self.q_prev = self.q.copy()

            completed = self.env.seg_idx >= self.n_segments - 2
            ep_energy = getattr(self.env, 'energy_kwh', 0)
            if completed:
                successes += 1
                if ep_energy < best_energy:
                    best_energy = ep_energy

            self.success_history.append(completed)
            self.energy_history.append(ep_energy)
            self.time_history.append(getattr(self.env, 't', 0))
            self.reward_history.append(total_r)
            self.loss_history.append(total_loss / max(n_upd, 1))

            self.epsilon = max(0.01, self.epsilon * 0.998)

            # Cosine LR decay
            frac = ep / episodes
            self.lr = self.lr_min + 0.5*(self.lr_max - self.lr_min)*(1 + np.cos(np.pi*frac))

            if ep % 200 == 0:
                el = time.time() - t0
                rate = successes / ep * 100
                recent_e = [self.energy_history[-i] for i in range(1, min(201, ep+1))
                           if self.success_history[-i]]
                avg_e = np.mean(recent_e) if recent_e else 0
                avg_l = np.mean(self.loss_history[-200:]) if self.loss_history else 0
                print(f"  Ep {ep:4d}/{episodes} | Success={rate:5.1f}% | "
                      f"Energy={avg_e:6.0f} kWh | Best={best_energy:.0f} | "
                      f"Loss={avg_l:.5f} | Net={net_frac:.0%} | "
                      f"eps={self.epsilon:.3f} | {el:.0f}s")

        rate = successes / episodes * 100
        print(f"\n  Phase 3 done: {successes}/{episodes} ({rate:.1f}%) in {time.time()-t0:.0f}s")
        print(f"  Best energy: {best_energy:.0f} kWh")

    def _energy_reward(self, env_reward, done, action, energy_step, seg, ep, total_eps):
        """
        Energy-aware reward for Phase 3.
        
        Strategy: start with mostly env_reward (ensures completion),
        gradually increase energy penalties as training progresses.
        """
        # Completion is always good
        if done and self.env.seg_idx >= self.n_segments - 2:
            total_e = getattr(self.env, 'energy_kwh', 0)
            # Reward inversely proportional to energy
            if total_e < 1200:
                return 200.0
            elif total_e < 1500:
                return 150.0
            elif total_e < 2000:
                return 80.0
            elif total_e < 2500:
                return 40.0
            else:
                return 20.0

        # Phase progress: 0 -> 1 over training
        progress = min(1.0, ep / max(total_eps, 1))

        # Start with env_reward (proven), fade in energy penalty
        base = env_reward * (1.0 - 0.5 * progress)  # env reward fades to 50%

        # Energy penalty -- grows with training progress
        e_pen = -energy_step * 5.0 * progress   # Starts at 0, grows to -5x

        # Grade-aware action shaping
        grade = self.grades[seg] if seg < len(self.grades) else 0
        v = getattr(self.env, 'v', 0)
        limit = self.limits[seg] if seg < len(self.limits) else 22.0
        if limit <= 1: limit = 22.0
        speed_ratio = v / max(limit, 1.0)

        shape = 0.0
        # Reward coasting (zero energy, preserves momentum)
        if action == self.A_COAST and v > 3.0:
            shape += 0.5 * progress
            if grade < -0.3:       # Downhill: gravity provides free energy
                shape += 1.0 * progress
        
        # Penalize power when not needed
        if action == self.A_POWER:
            if speed_ratio > 0.75:
                shape -= 0.8 * progress   # Already fast
            if grade < -0.5:
                shape -= 1.5 * progress   # Downhill: coast instead!
        
        # Penalize unnecessary braking (wastes kinetic energy)
        if action == self.A_BRAKE and speed_ratio < 0.85:
            shape -= 0.5 * progress

        # Cruise is efficient for maintaining speed
        if action == self.A_CRUISE and 0.5 < speed_ratio < 0.95:
            shape += 0.3 * progress

        return (base + e_pen + shape) * 0.01  # Scale down for network stability

    # ================================================================
    # Save results & speed profile
    # ================================================================

    def _save(self):
        od = os.path.join(getattr(config, 'OUTPUT_DIR', 'results_cm'), "deep_q")
        os.makedirs(od, exist_ok=True)

        np.save(os.path.join(od, "q_table.npy"), self.q)
        np.savez(os.path.join(od, "history.npz"),
                 success=np.array(self.success_history),
                 energy=np.array(self.energy_history),
                 time=np.array(self.time_history),
                 reward=np.array(self.reward_history))

        n = min(200, len(self.success_history))
        if n > 0:
            rs = sum(self.success_history[-n:]) / n
            re = [self.energy_history[-n+i] for i in range(n) if self.success_history[-n+i]]
            if re:
                print(f"\n  Last {n} eps: Success={rs:.0%}, Avg Energy={np.mean(re):.0f} kWh")
                print(f"  Min Energy: {min(re):.0f} kWh")

        # Plot
        try:
            import matplotlib; matplotlib.use('Agg')
            import matplotlib.pyplot as plt

            fig, axes = plt.subplots(2, 2, figsize=(14, 10))
            fig.suptitle('DQN Training', fontsize=14)
            er = range(len(self.success_history))

            # Success rate
            if len(self.success_history) > 50:
                w = 50
                sr = [np.mean(self.success_history[max(0,i-w):i+1]) for i in er]
                axes[0,0].plot(er, sr, 'g-', lw=1.5)
            axes[0,0].set_title('Success Rate'); axes[0,0].grid(True, alpha=0.3)

            # Energy
            se = [(i,e) for i,(e,s) in enumerate(zip(self.energy_history, self.success_history)) if s]
            if se:
                sx, sy = zip(*se)
                axes[0,1].scatter(sx, sy, c='b', alpha=0.3, s=2)
                if len(sy) > 50:
                    ma = np.convolve(sy, np.ones(50)/50, mode='valid')
                    axes[0,1].plot(range(sx[0]+49, sx[0]+49+len(ma)), ma, 'r-', lw=2)
                axes[0,1].axhline(1500, c='orange', ls='--', label='1500 kWh target')
                axes[0,1].legend()
            axes[0,1].set_title('Energy (successful)'); axes[0,1].grid(True, alpha=0.3)

            # Loss
            if self.loss_history:
                axes[1,0].plot(self.loss_history, 'r-', alpha=0.3, lw=0.5)
                if len(self.loss_history) > 100:
                    ll = [np.mean(self.loss_history[max(0,i-100):i+1]) for i in range(len(self.loss_history))]
                    axes[1,0].plot(ll, 'r-', lw=2)
            axes[1,0].set_title('Loss'); axes[1,0].grid(True, alpha=0.3)

            # Reward
            axes[1,1].plot(er, self.reward_history, 'm-', alpha=0.3, lw=0.5)
            if len(self.reward_history) > 100:
                rr = [np.mean(self.reward_history[max(0,i-100):i+1]) for i in er]
                axes[1,1].plot(er, rr, 'm-', lw=2)
            axes[1,1].set_title('Reward'); axes[1,1].grid(True, alpha=0.3)

            plt.tight_layout()
            plt.savefig(os.path.join(od, "dqn_training.png"), dpi=300)
            plt.close()
            print(f"Training plot saved")
        except ImportError: pass

        print(f"Results -> {od}/")

    def generate_speed_profile(self):
        """Generate optimal speed profile with energy focus."""
        print("\nGenerating optimal speed profile (DQN)...")
        self.env.reset()
        segs, vels, acts, ens = [], [], [], []
        max_steps = getattr(config, 'MAX_STEPS_PER_EPISODE', 20000)

        for step in range(max_steps):
            raw = self.env._get_state()
            state_n = self._norm(raw)
            seg = min(self.env.seg_idx, self.n_segments - 1)
            vb  = self._vbin(self.env.v)

            # Use Q-table where confident, else network
            qt = self.q[seg, vb, :]
            qr = qt.max() - qt.min()
            if qr > 1.0:
                action = int(np.argmax(qt))
            else:
                action = int(np.argmax(self.predict(state_n).flatten()))

            segs.append(self.env.seg_idx)
            vels.append(self.env.v)
            acts.append(action)
            ens.append(getattr(self.env, 'energy_kwh', 0))

            _, _, done, _ = self.env.step(action)
            if done or self.env.v < 0.1:
                break

        od = os.path.join(getattr(config, 'OUTPUT_DIR', 'results_cm'), "deep_q")
        os.makedirs(od, exist_ok=True)
        np.savez(os.path.join(od, "speed_profile.npz"),
                 segments=np.array(segs), velocities=np.array(vels),
                 actions=np.array(acts), energies=np.array(ens))

        fs = segs[-1] if segs else 0
        ok = fs >= self.n_segments - 2
        final_e = ens[-1] if ens else 0
        final_t = len(segs) * getattr(config, 'DT', 1.0)

        print(f"  Segment: {fs}/{self.n_segments} {'COMPLETE' if ok else 'INCOMPLETE'}")
        print(f"  Energy:  {final_e:.1f} kWh")
        print(f"  Time:    {final_t:.0f} s")

        aa = np.array(acts)
        for i, nm in enumerate(self.A_NAMES):
            print(f"  {nm}: {np.sum(aa==i)/len(aa)*100:.1f}%")

        # Plot
        try:
            import matplotlib; matplotlib.use('Agg')
            import matplotlib.pyplot as plt
            fig, (a1,a2,a3) = plt.subplots(3,1, figsize=(14,10), sharex=True)
            fig.suptitle(f'DQN Speed Profile -- Energy={final_e:.0f} kWh', fontsize=14)
            pos = np.array(segs) * getattr(config, 'DX', 100) / 1000

            a1.plot(pos, np.array(vels)*3.6, 'b-', lw=1.5)
            lim_pos = np.arange(self.n_segments) * getattr(config, 'DX', 100) / 1000
            a1.plot(lim_pos, self.limits * 3.6, 'r--', lw=0.8, alpha=0.5, label='Speed limit')
            a1.set_ylabel('Speed (km/h)'); a1.set_title('Speed Profile')
            a1.legend(); a1.grid(True, alpha=0.3)

            a2.plot(pos, ens, 'r-', lw=1.5)
            a2.axhline(1500, c='orange', ls='--', label='1500 kWh target')
            a2.set_ylabel('Energy (kWh)'); a2.set_title('Cumulative Energy')
            a2.legend(); a2.grid(True, alpha=0.3)

            colors = ['red','orange','blue','green']
            for i in range(4):
                m = aa == i
                if m.any():
                    a3.scatter(pos[m], [i]*m.sum(), c=colors[i], s=1, label=self.A_NAMES[i], alpha=0.5)
            a3.set_ylabel('Action'); a3.set_xlabel('Position (km)')
            a3.set_yticks([0,1,2,3]); a3.set_yticklabels(self.A_NAMES)
            a3.legend(markerscale=5); a3.grid(True, alpha=0.3)

            plt.tight_layout()
            plt.savefig(os.path.join(od, "speed_profile.png"), dpi=300)
            plt.close()
            print(f"Speed profile plot saved")
        except ImportError: pass

        return segs, vels, acts, ens