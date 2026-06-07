"""
Gymnasium adapter and route loader shared by train_dqn.py and the PPO/return-trip
trainers (load_data, require_gpu).

The wrapper delegates every step to the existing TrainEnv
(env_settings/environment.py), which calls
TrainPhysics.get_total_resistance() — the Davis equation
    r_t(V) = C0 + C1*V + C2*V^2
defined in env_settings/physics.py. Physics are unchanged; only the
API shape is adapted to Gymnasium so Tianshou's algorithms can drive it.
"""

import os
import sys
import numpy as np
import pandas as pd

import gymnasium as gym
from gymnasium import spaces

import torch

import env_settings.config as config
from env_settings.physics import TrainPhysics
from env_settings.environment import TrainEnv


def load_data():
    here = os.path.dirname(os.path.abspath(__file__))
    candidates = [
        'data/data.csv',
        os.path.join(here, '..', 'data', 'data.csv'),
        os.path.join(os.path.dirname(here), 'data', 'data.csv'),
    ]
    csv = next((p for p in candidates if os.path.exists(p)), None)
    if csv is None:
        raise FileNotFoundError("data/data.csv not found")
    df = pd.read_csv(csv)
    col = {c.lower().strip(): c for c in df.columns}
    grades = df[col[next(k for k in col if 'grade' in k)]].values.astype(float)
    limits = df[col[next(k for k in col if 'speed' in k)]].values.astype(float)
    curves = df[col[next(k for k in col if 'curv'  in k)]].values.astype(float)
    return grades, limits, curves


def require_gpu():
    """Abort hard if CUDA is not available — neural-net trainers must run on GPU."""
    if not torch.cuda.is_available():
        sys.exit("ERROR: CUDA is required. No NVIDIA GPU detected.\n"
                 "       Install a CUDA-enabled torch build, or run on a GPU host.")
    return torch.device('cuda')


class TrainGymEnv(gym.Env):
    """
    Gymnasium wrapper around TrainEnv (Davis physics intact).

    Observation : [seg_pos / N_segments, v / V_max] in [0, 1]
    Action      : Discrete(4) — 0=Brake, 1=Coast, 2=Cruise, 3=Power
    Truncates the episode on max_steps or after stall_limit consecutive
    near-zero-velocity steps.
    """
    metadata = {"render_modes": []}

    def __init__(self, grades, limits, curves, max_steps=None, stall_limit=200):
        super().__init__()
        self.physics = TrainPhysics()
        self.env = TrainEnv(self.physics, grades, limits, curves)
        self.n_segments = self.env.n_segments
        self.max_steps = max_steps or getattr(config, 'MAX_STEPS_PER_EPISODE', 20000)
        self.max_speed = getattr(config, 'MAX_SPEED_MS', 36.1)
        self.stall_limit = stall_limit

        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(2,), dtype=np.float32
        )
        self.action_space = spaces.Discrete(4)

        self._steps = 0
        self._stall = 0

    def _obs(self):
        seg, v = self.env._get_state()
        p = float(seg) / max(self.n_segments, 1)
        u = float(v) / self.max_speed
        return np.array([np.clip(p, 0.0, 1.0), np.clip(u, 0.0, 1.0)], dtype=np.float32)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.env.reset()
        self._steps = 0
        self._stall = 0
        return self._obs(), {}

    def step(self, action):
        action = int(action)
        _, reward, done, info = self.env.step(action)
        self._steps += 1
        self._stall = self._stall + 1 if self.env.v < 0.1 else 0

        terminated = bool(done)
        truncated = (self._steps >= self.max_steps) or (self._stall >= self.stall_limit)
        info['success'] = bool(terminated and self.env.seg_idx >= self.n_segments - 1)
        info['energy'] = float(self.env.energy_kwh)
        info['time'] = float(self.env.t)

        return self._obs(), float(reward), terminated, truncated, info
