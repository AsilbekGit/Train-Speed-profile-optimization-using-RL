"""
PPO trainer (Tianshou) for the train speed-profile problem.

This is the project's PPO implementation. Key features:

  * Uses ppo.env_wrapper.PPOTrainEnv (rich 8-D obs + energy reward + tougher limits)
  * VectorEnvNormObs around the train + test envs (running obs mean/std);
    test envs share the train obs_rms but stop updating it.
  * InMemoryLogger captures train / test / update stats so we can save a
    training-curves PNG at the end.
  * Best-checkpoint is saved whenever the test return improves.
  * Bigger default network ([256, 256]).
  * Hard GPU requirement (aborts if cuda is unavailable).

Run:
    python -m ppo.train --steps 1500000
    python -m ppo.train --steps 200000      # quick check
"""

import argparse
import os
import sys
import time
import csv

import numpy as np
import torch

from tianshou.data import Collector, VectorReplayBuffer
from tianshou.env import DummyVectorEnv, VectorEnvNormObs
from tianshou.utils.net.common import Net
from tianshou.utils.net.discrete import (
    DiscreteActor,
    DiscreteCritic,
    dist_fn_categorical_from_logits,
)
from tianshou.algorithm.modelfree.ppo import PPO
from tianshou.algorithm.modelfree.reinforce import ProbabilisticActorPolicy
from tianshou.algorithm.optim import AdamOptimizerFactory
from tianshou.trainer import OnPolicyTrainerParams
from tianshou.utils.logger.logger_base import BaseLogger

# Project-root imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import env_settings.config as config
from env_settings.gym_env import load_data, require_gpu

# Local
from ppo.env_wrapper import PPOTrainEnv
from ppo.plot import plot_training_curves, save_rollout_outputs

# Results live inside the ppo/ package (ppo/results/), anchored to this file so
# the output path is independent of the working directory you launch from.
PPO_DIR = os.path.dirname(os.path.abspath(__file__))


# ---------------------------------------------------------------------------
# In-memory logger — captures everything Tianshou writes during training
# ---------------------------------------------------------------------------

VALID_LOG_TYPES = (int, float, np.integer, np.floating)


def _flatten(d, prefix=""):
    """Flatten nested dicts; drop arrays / non-numeric values."""
    out = {}
    for k, v in d.items():
        key = f"{prefix}/{k}" if prefix else str(k)
        if isinstance(v, dict):
            out.update(_flatten(v, key))
        elif isinstance(v, np.ndarray):
            continue
        elif isinstance(v, VALID_LOG_TYPES):
            out[key] = float(v)
    return out


class InMemoryLogger(BaseLogger):
    """Captures train / test / update / info stats into Python lists."""

    def __init__(self,
                 train_interval=1000,
                 test_interval=1,
                 update_interval=1000,
                 info_interval=1,
                 save_interval=None):
        super().__init__(train_interval, test_interval, update_interval,
                         info_interval, save_interval)
        self.training_log = []
        self.test_log = []
        self.update_log = []
        self.info_log = []

    def prepare_dict_for_logging(self, log_data):
        return _flatten(log_data)

    def write(self, step_type, step, data):
        bucket = {
            "train/env_step":  self.training_log,
            "test/env_step":   self.test_log,
            "update/gradient_step": self.update_log,
            "info/epoch":      self.info_log,
        }.get(step_type)
        if bucket is None:
            return
        bucket.append({"step": step, **data})

    def log_training_data(self, log_data, step):
        self.write("train/env_step", step, self.prepare_dict_for_logging(log_data))

    def log_test_data(self, log_data, step):
        self.write("test/env_step", step, self.prepare_dict_for_logging(log_data))

    def log_update_data(self, log_data, step):
        self.write("update/gradient_step", step, self.prepare_dict_for_logging(log_data))

    def log_info_data(self, log_data, step):
        self.write("info/epoch", step, self.prepare_dict_for_logging(log_data))

    def save_data(self, epoch, env_step, update_step, save_checkpoint_fn=None):
        pass

    def restore_data(self):
        return 0, 0, 0

    def restore_logged_data(self, log_path):
        return {}

    def finalize(self):
        pass

    def to_csv(self, out_dir):
        for name, log in [
            ("train.csv", self.training_log),
            ("test.csv", self.test_log),
            ("update.csv", self.update_log),
            ("info.csv", self.info_log),
        ]:
            if not log:
                continue
            keys = sorted({k for row in log for k in row.keys()})
            with open(os.path.join(out_dir, name), "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=keys)
                w.writeheader()
                for row in log:
                    w.writerow({k: row.get(k, "") for k in keys})


# ---------------------------------------------------------------------------
# Greedy rollout for the final speed-profile chart
# ---------------------------------------------------------------------------

def greedy_rollout(actor, eval_env, device):
    obs, _ = eval_env.reset()
    segs, vels, acts, ens = [], [], [], []
    actor.eval()
    with torch.no_grad():
        while True:
            obs_t = torch.from_numpy(np.asarray(obs, dtype=np.float32)) \
                         .unsqueeze(0).to(device)
            out = actor(obs_t)
            logits = out[0] if isinstance(out, tuple) else out
            action = int(torch.argmax(logits, dim=-1).item())
            segs.append(eval_env.env.seg_idx)
            vels.append(eval_env.env.v)
            acts.append(action)
            ens.append(eval_env.env.energy_kwh)
            obs, _, term, trunc, _ = eval_env.step(action)
            if term or trunc:
                break
    return segs, vels, acts, ens


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--steps',           type=int,   default=1_000_000)
    p.add_argument('--n-envs',          type=int,   default=8)
    p.add_argument('--n-test-envs',     type=int,   default=2)
    p.add_argument('--lr',              type=float, default=3e-4)
    p.add_argument('--gamma',           type=float, default=0.99)
    p.add_argument('--gae-lambda',      type=float, default=0.95)
    p.add_argument('--clip',            type=float, default=0.2)
    p.add_argument('--ent-coef',        type=float, default=0.01)
    p.add_argument('--vf-coef',         type=float, default=0.5)
    p.add_argument('--collect-steps',   type=int,   default=2048)
    p.add_argument('--epoch-steps',     type=int,   default=20_000)
    p.add_argument('--batch-size',      type=int,   default=256)
    p.add_argument('--n-update-epochs', type=int,   default=10)
    p.add_argument('--hidden',          type=int,   nargs='+', default=[256, 256])
    # Forward and return MUST share these so the two legs are the same model and
    # only the route differs (the energy/profile comparison would otherwise be
    # confounded by reward shaping). Keep in sync with return_trip/train_ppo.py.
    p.add_argument('--energy-coef',     type=float, default=2.0,
                   help='reward penalty per kWh of step energy')
    p.add_argument('--limit-pen',       type=float, default=4.0,
                   help='reward penalty per m/s of speed-limit overshoot')
    p.add_argument('--limit-term',      type=float, default=-1.0,
                   help='m/s overshoot that ends the episode; <=0 disables termination')
    p.add_argument('--jerk-pen',        type=float, default=0.0,
                   help='optional penalty per unit of |action change| (0 = off)')
    p.add_argument('--seed',            type=int,   default=0)
    p.add_argument('--out-dir',         type=str,
                   default=os.path.join(PPO_DIR, 'results'))
    args = p.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    print('=' * 72)
    print('PPO — rich 9-D obs, energy-aware reward, obs normalization, GPU only')
    print('Physics: Davis equation (env_settings/physics.py — UNCHANGED)')
    print('=' * 72)

    device = require_gpu()
    print(f'Device: {device} ({torch.cuda.get_device_name(0)})')

    print('\n1. Loading route data...')
    grades, limits, curves = load_data()
    print(f'   {len(grades)} segments, {len(grades) * 0.1:.1f} km')

    # <=0 disables the hard limit-overshoot termination (env wants None). Same
    # convention and default as return_trip/train_ppo.py so both legs match.
    limit_term = None if args.limit_term <= 0 else args.limit_term
    print('\n2. Building envs (rich obs + energy reward)...')
    print(f'   energy_coef={args.energy_coef}, limit_pen={args.limit_pen}, '
          f'limit_term={"DISABLED" if limit_term is None else f"{limit_term} m/s"}, '
          f'jerk_pen={args.jerk_pen}')
    def make_env():
        return PPOTrainEnv(
            grades, limits, curves,
            energy_coef=args.energy_coef,
            limit_pen_coef=args.limit_pen,
            limit_overshoot_term=limit_term,
            jerk_pen_coef=args.jerk_pen,
        )
    sample_env = make_env()
    print(f'   obs dim = {sample_env.OBS_DIM}, action dim = {sample_env.action_space.n}')

    train_envs_raw = DummyVectorEnv([make_env for _ in range(args.n_envs)])
    test_envs_raw  = DummyVectorEnv([make_env for _ in range(args.n_test_envs)])
    # Running obs normalization on training envs; test envs share the rms
    # but freeze their copy so test-time stats reflect the actual policy.
    train_envs = VectorEnvNormObs(train_envs_raw, update_obs_rms=True)
    test_envs  = VectorEnvNormObs(test_envs_raw,  update_obs_rms=False)
    test_envs.set_obs_rms(train_envs.get_obs_rms())

    print('\n3. Building PPO networks...')
    obs_dim = sample_env.OBS_DIM
    n_actions = sample_env.action_space.n

    actor_trunk = Net(state_shape=obs_dim, hidden_sizes=list(args.hidden),
                      activation=torch.nn.Tanh)
    actor = DiscreteActor(preprocess_net=actor_trunk, action_shape=n_actions,
                          softmax_output=False)
    critic_trunk = Net(state_shape=obs_dim, hidden_sizes=list(args.hidden),
                       activation=torch.nn.Tanh)
    critic = DiscreteCritic(preprocess_net=critic_trunk)

    policy = ProbabilisticActorPolicy(
        actor=actor,
        dist_fn=dist_fn_categorical_from_logits,
        deterministic_eval=True,
        action_space=sample_env.action_space,
        observation_space=sample_env.observation_space,
        action_scaling=False,
    )
    optim = AdamOptimizerFactory(lr=args.lr)
    algorithm = PPO(
        policy=policy,
        critic=critic,
        optim=optim,
        eps_clip=args.clip,
        vf_coef=args.vf_coef,
        ent_coef=args.ent_coef,
        gae_lambda=args.gae_lambda,
        gamma=args.gamma,
        max_grad_norm=0.5,
        advantage_normalization=True,
    )
    algorithm.to(device)
    print(f'   Actor + Critic: {obs_dim} -> {"->".join(map(str, args.hidden))} '
          f'-> {n_actions} (tanh)')
    print(f'   PPO: clip={args.clip} gamma={args.gamma} gae={args.gae_lambda} '
          f'ent={args.ent_coef} vf={args.vf_coef} lr={args.lr}')

    print('\n4. Collectors + buffer + logger...')
    buffer = VectorReplayBuffer(
        total_size=args.collect_steps * args.n_envs,
        buffer_num=args.n_envs,
    )
    train_collector = Collector(algorithm, train_envs, buffer)
    test_collector  = Collector(algorithm, test_envs)

    out_dir = args.out_dir
    os.makedirs(out_dir, exist_ok=True)
    logger = InMemoryLogger()

    best_ckpt_path = os.path.join(out_dir, 'ppo_best.pt')
    last_ckpt_path = os.path.join(out_dir, 'ppo_last.pt')

    def save_best_fn(algo):
        torch.save(algo.state_dict(), best_ckpt_path)

    max_epochs = max(1, args.steps // args.epoch_steps)
    params = OnPolicyTrainerParams(
        max_epochs=max_epochs,
        epoch_num_steps=args.epoch_steps,
        training_collector=train_collector,
        test_collector=test_collector,
        collection_step_num_env_steps=args.collect_steps,
        update_step_num_repetitions=args.n_update_epochs,
        batch_size=args.batch_size,
        test_step_num_episodes=4,
        save_best_fn=save_best_fn,
        logger=logger,
        show_progress=True,
        verbose=False,
    )

    print(f'\n5. Training PPO for {args.steps} env steps ({max_epochs} epochs)...')
    t0 = time.time()
    info = algorithm.run_training(params)
    print(f'\n   Training done in {time.time() - t0:.0f} s')
    torch.save(algorithm.state_dict(), last_ckpt_path)
    print(f'   Last  checkpoint -> {last_ckpt_path}')
    if os.path.exists(best_ckpt_path):
        print(f'   Best  checkpoint -> {best_ckpt_path}')

    # Save logger to CSVs and PNG
    logger.to_csv(out_dir)
    plot_training_curves(os.path.join(out_dir, 'training_curves.png'), logger)
    print(f'   Curves -> {out_dir}/training_curves.png')

    # ----- final greedy rollout using best checkpoint when available -----
    print('\n6. Greedy rollout for speed profile...')
    if os.path.exists(best_ckpt_path):
        algorithm.load_state_dict(torch.load(best_ckpt_path, map_location=device))
        print('   (using best-checkpoint policy)')

    # Build a single eval env that mirrors the obs normalization applied in training.
    eval_env_raw = make_env()

    class _NormalizedSingleEnv:
        """Wraps a single env so obs are normalized identically to test_envs."""
        def __init__(self, env, rms):
            self.env = env.env  # underlying TrainEnv (so rollout can read seg_idx, v)
            self._w = env       # the PPOTrainEnv (for step/reset)
            self._rms = rms
            self.n_segments = env.n_segments

        def reset(self):
            obs, info = self._w.reset()
            return self._normalize(obs), info

        def step(self, action):
            obs, r, term, trunc, info = self._w.step(action)
            return self._normalize(obs), r, term, trunc, info

        def _normalize(self, obs):
            mean = self._rms.mean if self._rms is not None else 0.0
            std = np.sqrt(self._rms.var) if self._rms is not None else 1.0
            n = (np.asarray(obs, dtype=np.float64) - mean) / (std + 1e-8)
            return np.clip(n, -10.0, 10.0).astype(np.float32)

    rms = train_envs.get_obs_rms()
    eval_env = _NormalizedSingleEnv(eval_env_raw, rms)
    segs, vels, acts, ens = greedy_rollout(actor, eval_env, device)

    final_seg = segs[-1] if segs else 0
    success = final_seg >= eval_env.n_segments - 2
    final_e = ens[-1] if ens else 0.0
    final_t = len(segs) * getattr(config, 'DT', 1.0)
    print(f'   Segment: {final_seg}/{eval_env.n_segments} '
          f'{"COMPLETE" if success else "INCOMPLETE"}')
    print(f'   Energy:  {final_e:.1f} kWh')
    print(f'   Time:    {final_t:.0f} s')
    aa = np.array(acts) if acts else np.array([0])
    for i, nm in enumerate(['Brake', 'Coast', 'Cruise', 'Power']):
        print(f'   {nm:6s}: {(aa == i).mean() * 100:.1f}%')

    # All rollout artifacts (npz, per-timestep speed_log.csv, speed_profile.png
    # vs distance, speed_vs_time.png vs time) from one shared helper.
    save_rollout_outputs(
        out_dir, segs, vels, acts, ens, grades, limits,
        title=f'PPO — Energy={final_e:.0f} kWh '
              f'({"COMPLETE" if success else "INCOMPLETE"})',
        dx=getattr(config, 'DX', 100.0),
        dt=getattr(config, 'DT', 1.0),
    )
    print(f'   Outputs -> {out_dir}/ '
          f'(speed_profile.png, speed_vs_time.png, speed_log.csv)')

    print('\n' + '=' * 72)
    print('PPO TRAINING COMPLETE')
    print('=' * 72)


if __name__ == '__main__':
    main()
