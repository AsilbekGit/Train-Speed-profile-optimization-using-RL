"""
Enriched Gymnasium environment for the PPO experiment.

Builds on the existing TrainEnv (env_settings/environment.py), which calls
TrainPhysics.get_total_resistance — the Davis equation in env_settings/physics.py.
NOTHING in env_settings/ is touched; this module only adds an outer wrapper.

What this fixes vs. env_settings/gym_env.py.TrainGymEnv:

  1. Observation goes from 2-D `[seg_idx, v]` to 8-D with lookahead so the policy
     can see what's coming (current grade, current speed limit, mean grade over
     the next short window, max grade ahead, tightest speed limit ahead, distance
     to next station). Without this the policy is forced to memorize the route
     by position alone.

  2. Reward includes an energy term. The base TrainEnv reward optimizes for
     completion + progress and contains *no* energy signal at all. We add
     `-energy_coef * energy_kwh_step` so coasting and cruising are preferred
     over flooring the throttle.

  3. Speed-limit handling is firmer. The base env applies a flat -5 only when
     `v > limit + 2 m/s`; we add a *proportional* penalty that grows linearly
     with overshoot from the first m/s, and we *terminate* the episode if the
     train blows past the limit by more than `limit_overshoot_term` m/s.

The Davis-equation physics, the energy formula in TrainEnv.step, and the
underlying step dynamics are unchanged.
"""

import os
import sys
import numpy as np
import gymnasium as gym
from gymnasium import spaces

# Reach project root regardless of where this is imported from.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import env_settings.config as config
from env_settings.physics import TrainPhysics
from env_settings.environment import TrainEnv


GRADE_SCALE = 7.0   # grade % normalization; route data is roughly [-4.6, 6.3]


class PPOTrainEnv(gym.Env):
    """
    Rich-observation, energy-aware Gymnasium env for PPO.
    Delegates every physics step to TrainEnv (Davis untouched).
    """

    metadata = {"render_modes": []}

    LOOKAHEAD_SHORT = 5    # segments (= 500 m at DX=100)
    LOOKAHEAD_LONG = 20    # segments (= 2 km)
    LOOKAHEAD_LIMIT = 20   # segments

    OBS_DIM = 8

    def __init__(
        self,
        grades,
        limits,
        curves,
        max_steps=None,
        stall_limit=200,
        energy_coef=2.0,
        limit_pen_coef=2.0,
        limit_overshoot_term=5.0,
    ):
        super().__init__()
        self.grades_arr = np.asarray(grades, dtype=np.float64)
        self.limits_arr = np.asarray(limits, dtype=np.float64)
        self.curves_arr = np.asarray(curves, dtype=np.float64)

        self.physics = TrainPhysics()
        self.env = TrainEnv(self.physics, grades, limits, curves)

        self.n_segments = self.env.n_segments
        self.max_steps = max_steps or getattr(config, 'MAX_STEPS_PER_EPISODE', 20000)
        self.max_speed = getattr(config, 'MAX_SPEED_MS', 36.1)
        self.stall_limit = stall_limit

        self.energy_coef = energy_coef
        self.limit_pen_coef = limit_pen_coef
        self.limit_overshoot_term = limit_overshoot_term

        self.observation_space = spaces.Box(
            low=-3.0, high=3.0, shape=(self.OBS_DIM,), dtype=np.float32
        )
        self.action_space = spaces.Discrete(4)

        # Stations: paper convention = limits == 1.0 (m/s). Cache positions.
        self._station_segs = np.where(self.limits_arr <= 1.0)[0]

        self._steps = 0
        self._stall = 0
        self._last_energy = 0.0

    # ------------------------------------------------------------------
    # observation features
    # ------------------------------------------------------------------

    def _grade_mean(self, idx, k):
        end = min(idx + k, self.n_segments)
        if end <= idx:
            return 0.0
        return float(self.grades_arr[idx:end].mean())

    def _grade_max(self, idx, k):
        end = min(idx + k, self.n_segments)
        if end <= idx:
            return 0.0
        return float(self.grades_arr[idx:end].max())

    def _limit_min(self, idx, k):
        end = min(idx + k, self.n_segments)
        if end <= idx:
            return self.max_speed
        chunk = self.limits_arr[idx:end]
        return float(chunk.min())

    def _dist_to_next_station(self, idx):
        future = self._station_segs[self._station_segs > idx]
        if future.size == 0:
            return float(self.n_segments - idx)
        return float(future[0] - idx)

    def _obs(self):
        seg = int(self.env.seg_idx)
        seg_clamped = min(seg, self.n_segments - 1)
        v = float(self.env.v)

        grade_now = float(self.grades_arr[seg_clamped])
        limit_now = float(self.limits_arr[seg_clamped])
        limit_now_eff = max(limit_now, 1.0)

        feat = np.array([
            seg / max(self.n_segments, 1),                                    # 0: pos_frac     [0,1]
            v / self.max_speed,                                                # 1: v_frac       [0,1]
            grade_now / GRADE_SCALE,                                           # 2: grade_now    signed
            limit_now_eff / self.max_speed,                                    # 3: limit_now    [0,1]
            self._grade_mean(seg_clamped, self.LOOKAHEAD_SHORT) / GRADE_SCALE, # 4: grade_short  signed
            self._grade_max(seg_clamped, self.LOOKAHEAD_LONG) / GRADE_SCALE,   # 5: grade_long_max
            self._limit_min(seg_clamped, self.LOOKAHEAD_LIMIT) / self.max_speed,  # 6: limit_min_ahead
            self._dist_to_next_station(seg_clamped) / max(self.n_segments, 1), # 7: dist_to_station_frac
        ], dtype=np.float32)

        return np.clip(feat, -3.0, 3.0)

    # ------------------------------------------------------------------
    # gym API
    # ------------------------------------------------------------------

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.env.reset()
        self._steps = 0
        self._stall = 0
        self._last_energy = float(self.env.energy_kwh)
        return self._obs(), {}

    def step(self, action):
        action = int(action)
        seg_before = int(self.env.seg_idx)
        seg_clamped = min(seg_before, self.n_segments - 1)

        # Effective speed limit at the segment we're entering this step. The base
        # env uses the same value internally for its -5 penalty band.
        limit_at_seg = float(self.limits_arr[seg_clamped])
        if limit_at_seg <= 1.0:   # station; let env's own stall penalty handle stop
            limit_at_seg = 22.0

        _, base_reward, done, info = self.env.step(action)

        e_now = float(self.env.energy_kwh)
        e_step = max(0.0, e_now - self._last_energy)
        self._last_energy = e_now

        v_after = float(self.env.v)
        overshoot = max(0.0, v_after - limit_at_seg)

        # Reward shaping:
        #   - keep base env reward (progress + completion + small bonuses)
        #   - subtract proportional energy penalty per kWh of step energy
        #   - add proportional limit-overshoot penalty (linear from first m/s)
        energy_pen = -self.energy_coef * e_step
        limit_pen = -self.limit_pen_coef * overshoot
        reward = float(base_reward) + energy_pen + limit_pen

        self._steps += 1
        self._stall = self._stall + 1 if v_after < 0.1 else 0

        terminated = bool(done)
        truncated = (self._steps >= self.max_steps) or (self._stall >= self.stall_limit)

        # Hard termination on egregious limit violation.
        if overshoot > self.limit_overshoot_term:
            truncated = True
            reward -= 50.0

        info['success'] = bool(terminated and self.env.seg_idx >= self.n_segments - 1)
        info['energy'] = e_now
        info['energy_step'] = e_step
        info['time'] = float(self.env.t)
        info['overshoot'] = overshoot

        return self._obs(), reward, terminated, truncated, info
