"""
Enriched Gymnasium environment for the PPO experiment.

Builds on the existing TrainEnv (env_settings/environment.py), which calls
TrainPhysics.get_total_resistance — the Davis equation in env_settings/physics.py.
NOTHING in env_settings/ is touched; this module only adds an outer wrapper.

What this fixes vs. env_settings/gym_env.py.TrainGymEnv:

  1. Observation goes from 2-D `[seg_idx, v]` to 9-D with lookahead so the policy
     can see what's coming (current grade, current speed limit, mean grade over
     the next short window, max grade ahead, tightest speed limit ahead, *distance
     to that tightest limit*, and distance to next station). Without this the
     policy is forced to memorize the route by position alone. The distance to the
     tightest upcoming limit is what lets the policy TIME its braking — knowing a
     3 m/s zone is "somewhere in the next 2 km" is not enough; it needs to know
     whether to brake now or in fifteen segments.

  2. Reward includes an energy term. The base TrainEnv reward optimizes for
     completion + progress and contains *no* energy signal at all. We add
     `-energy_coef * energy_kwh_step` so coasting and cruising are preferred
     over flooring the throttle.

  3. Speed-limit handling is firmer. The base env applies a flat -5 only when
     `v > limit + 2 m/s`; we add a *proportional* penalty that grows linearly
     with overshoot from the first m/s, and we *optionally* terminate the
     episode if the train blows past the limit by more than
     `limit_overshoot_term` m/s. Pass `limit_overshoot_term=None` to DISABLE
     termination (episode continues; only the graded penalty applies) — the
     return trip uses this so a single tight-limit zone near the end of the
     reversed route cannot make completion unreachable.

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

    OBS_DIM = 9

    def __init__(
        self,
        grades,
        limits,
        curves,
        max_steps=None,
        stall_limit=200,
        energy_coef=2.0,
        limit_pen_coef=2.0,
        limit_overshoot_term=5.0,   # m/s overshoot that ends the episode; None disables
        jerk_pen_coef=0.0,          # penalty per unit of |action change| (smoother driving)
        cancel_speed_bonus=False,   # cancel the base env's +0.1 'v>5' speed bonus
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
        self.jerk_pen_coef = jerk_pen_coef
        self.cancel_speed_bonus = cancel_speed_bonus

        self.observation_space = spaces.Box(
            low=-3.0, high=3.0, shape=(self.OBS_DIM,), dtype=np.float32
        )
        self.action_space = spaces.Discrete(4)

        # Stations: paper convention = limits == 1.0 (m/s). Cache positions.
        self._station_segs = np.where(self.limits_arr <= 1.0)[0]

        self._steps = 0
        self._stall = 0
        self._last_energy = 0.0
        self._prev_action = None

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

    def _dist_to_min_limit(self, idx, k):
        """
        Segments from `idx` to the tightest speed limit in the next k segments
        (0 = the tightest limit is the current segment). This is the timing
        signal the policy needs to brake early enough: `_limit_min` says *how
        tight* the worst upcoming limit is, this says *how far away* it is.
        Returns k (the window size) when nothing is ahead, so the feature reads
        "no tight limit nearby". Stations (limit <= 1.0) are treated as
        ordinary low limits here so the policy also anticipates them.
        """
        end = min(idx + k, self.n_segments)
        if end <= idx:
            return float(k)
        chunk = self.limits_arr[idx:end]
        return float(int(np.argmin(chunk)))   # first (nearest) tightest segment

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
            self._dist_to_min_limit(seg_clamped, self.LOOKAHEAD_LIMIT)
                / self.LOOKAHEAD_LIMIT,                                        # 7: dist_to_tight_limit [0,1]
            self._dist_to_next_station(seg_clamped) / max(self.n_segments, 1), # 8: dist_to_station_frac
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
        self._prev_action = None
        return self._obs(), {}

    def step(self, action):
        action = int(action)
        seg_before = int(self.env.seg_idx)
        seg_clamped = min(seg_before, self.n_segments - 1)

        # Effective speed limit at the segment we're entering this step. The base
        # env uses the same value internally for its -5 penalty band.
        raw_limit = float(self.limits_arr[seg_clamped])
        is_station = raw_limit <= 1.0
        # At a station the train is meant to be creeping/stopping, so we don't
        # charge an overshoot penalty there — relax the effective limit.
        limit_at_seg = 22.0 if is_station else raw_limit

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

        # Optional gentler-driving shaping (off by default):
        #   - cancel the base env's flat +0.1 'v>5' speed bonus so the policy is
        #     not simply paid to go fast; speed must justify itself against energy.
        speed_bonus_cancel = (-0.1 if (self.cancel_speed_bonus and v_after > 5.0)
                              else 0.0)
        #   - penalize abrupt action changes to smooth the profile. Actions are
        #     ordered Brake(0) < Coast(1) < Cruise(2) < Power(3), so |Δaction| is a
        #     proxy for jerk — a Power<->Brake flip (|3-0|=3) is penalized most.
        if self.jerk_pen_coef > 0.0 and self._prev_action is not None:
            jerk_pen = -self.jerk_pen_coef * abs(action - self._prev_action)
        else:
            jerk_pen = 0.0
        self._prev_action = action

        reward = (float(base_reward) + energy_pen + limit_pen
                  + speed_bonus_cancel + jerk_pen)

        self._steps += 1
        self._stall = self._stall + 1 if v_after < 0.1 else 0

        terminated = bool(done)
        truncated = (self._steps >= self.max_steps) or (self._stall >= self.stall_limit)

        # Hard termination on egregious limit violation. Set
        # limit_overshoot_term=None to DISABLE this: the episode then continues
        # and only the graded limit penalty (limit_pen above) applies, so the
        # agent can still reach completion after a brief overspeed. The reverse
        # route places a 3.0 m/s zone (~seg 718) near the end that is reached at
        # high speed; killing the episode there made the return trip unable to
        # finish, so the return trainer passes None here.
        if self.limit_overshoot_term is not None and overshoot > self.limit_overshoot_term:
            truncated = True
            reward -= 50.0

        info['success'] = bool(terminated and self.env.seg_idx >= self.n_segments - 1)
        info['energy'] = e_now
        info['energy_step'] = e_step
        info['time'] = float(self.env.t)
        info['overshoot'] = overshoot

        return self._obs(), reward, terminated, truncated, info
