"""
Train PPO on the *return* leg (B → A) using the improved PPO setup from ppo/.

Reuses ppo.env_wrapper.PPOTrainEnv (rich obs + energy reward + tougher limits)
and ppo.plot (speed profile + grade band + limit overlay; training curves).
The InMemoryLogger and most of the trainer body are imported from ppo.train so
nothing is duplicated.

Output: results_cm/return/ppo/

Usage:
    python -m return_trip.train_ppo
    python -m return_trip.train_ppo --steps 1500000
    python -m return_trip.train_ppo --steps 8000          # quick sanity
"""

import argparse
import os
import sys
import time

import numpy as np
import torch

from tianshou.data import Collector, VectorReplayBuffer
from tianshou.env import DummyVectorEnv, VectorEnvNormObs
from tianshou.utils.net.common import Net
from tianshou.utils.net.discrete import (
    DiscreteActor, DiscreteCritic, dist_fn_categorical_from_logits,
)
from tianshou.algorithm.modelfree.ppo import PPO
from tianshou.algorithm.modelfree.reinforce import ProbabilisticActorPolicy
from tianshou.algorithm.optim import AdamOptimizerFactory
from tianshou.trainer import OnPolicyTrainerParams

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import env_settings.config as config
from env_settings.gym_env import require_gpu

from ppo.env_wrapper import PPOTrainEnv
from ppo.plot import plot_training_curves, save_rollout_outputs
from ppo.train import InMemoryLogger, greedy_rollout

from return_trip.route import load_reverse_route, print_summary


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
    # Kept in sync with ppo/train.py so forward and return are the same model.
    p.add_argument('--energy-coef',     type=float, default=2.0,
                   help='reward penalty per kWh of step energy')
    p.add_argument('--jerk-pen',        type=float, default=0.0,
                   help='optional penalty per unit of |action change| (0 = off)')
    # Graded speed-limit penalty, stronger than the forward trip (2.0) but kept
    # moderate: with the hard termination disabled (below) this is the agent's
    # main pressure to slow for the tight-limit zone near the end of the reverse
    # route. Expert review flagged 10.0 as a fragile break-even regime (the
    # plow-through penalty ~rivals the completion bonus, inviting a stall-at-
    # station degenerate policy), so we use 4.0. Combined with the new
    # distance-to-tight-limit observation feature the agent can now time braking.
    p.add_argument('--limit-pen',       type=float, default=4.0)
    # <= 0 DISABLES the hard limit-overshoot termination (passes None to the
    # env). The forward trip keeps termination at 5.0 m/s; the return trip
    # disables it so the seg-~718 3.0 m/s zone (reached at high speed) cannot
    # kill the episode and make completion unreachable. See DEVELOPMENT_LOG.md.
    p.add_argument('--limit-term',      type=float, default=-1.0,
                   help='m/s overshoot that ends the episode; <=0 disables termination')
    p.add_argument('--seed',            type=int,   default=0)
    p.add_argument('--out-dir',         type=str,
                   # Hard-coded base so a same-session run after
                   # return_trip.train_qsarsa (which mutates config.OUTPUT_DIR)
                   # can't produce 'results_cm/return/return/ppo/'.
                   default=os.path.join('results_cm', 'return', 'ppo'))
    args = p.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    print('=' * 72)
    print('PPO — RETURN TRIP (B → A)  [grades sign-flipped + segment-reversed]')
    print('Physics: Davis equation (env_settings/physics.py — UNCHANGED)')
    print('=' * 72)

    device = require_gpu()
    print(f'Device: {device} ({torch.cuda.get_device_name(0)})')

    print('\n1. Loading reverse route...')
    grades, limits, curves = load_reverse_route()
    print_summary(grades, limits, curves)

    # <=0 disables the hard limit-overshoot termination (env wants None).
    limit_term = None if args.limit_term <= 0 else args.limit_term
    print('\n2. Building envs (rich obs + energy reward, reversed route)...')
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

    train_envs_raw = DummyVectorEnv([make_env for _ in range(args.n_envs)])
    test_envs_raw  = DummyVectorEnv([make_env for _ in range(args.n_test_envs)])
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
        actor=actor, dist_fn=dist_fn_categorical_from_logits,
        deterministic_eval=True,
        action_space=sample_env.action_space,
        observation_space=sample_env.observation_space,
        action_scaling=False,
    )
    optim = AdamOptimizerFactory(lr=args.lr)
    algorithm = PPO(
        policy=policy, critic=critic, optim=optim,
        eps_clip=args.clip, vf_coef=args.vf_coef, ent_coef=args.ent_coef,
        gae_lambda=args.gae_lambda, gamma=args.gamma,
        max_grad_norm=0.5, advantage_normalization=True,
    )
    algorithm.to(device)

    print('\n4. Collectors + buffer + logger...')
    buffer = VectorReplayBuffer(
        total_size=args.collect_steps * args.n_envs, buffer_num=args.n_envs,
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
    algorithm.run_training(params)
    print(f'\n   Training done in {time.time() - t0:.0f} s')
    torch.save(algorithm.state_dict(), last_ckpt_path)
    print(f'   Last checkpoint -> {last_ckpt_path}')
    if os.path.exists(best_ckpt_path):
        print(f'   Best checkpoint -> {best_ckpt_path}')

    logger.to_csv(out_dir)
    plot_training_curves(os.path.join(out_dir, 'training_curves.png'), logger)
    print(f'   Curves -> {out_dir}/training_curves.png')

    print('\n6. Greedy rollout for speed profile...')
    if os.path.exists(best_ckpt_path):
        algorithm.load_state_dict(torch.load(best_ckpt_path, map_location=device))
        print('   (using best-checkpoint policy)')

    # Mirror the obs normalization used in training when running the eval rollout.
    eval_env_raw = make_env()
    rms = train_envs.get_obs_rms()

    class _NormalizedSingleEnv:
        def __init__(self, env, rms):
            self.env = env.env
            self._w = env
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
        title=f'PPO RETURN — Energy={final_e:.0f} kWh '
              f'({"COMPLETE" if success else "INCOMPLETE"})',
        dx=getattr(config, 'DX', 100.0),
        dt=getattr(config, 'DT', 1.0),
    )
    print(f'   Outputs -> {out_dir}/ '
          f'(speed_profile.png, speed_vs_time.png, speed_log.csv)')

    print('\n' + '=' * 72)
    print('PPO RETURN TRIP COMPLETE')
    print('=' * 72)


if __name__ == '__main__':
    main()
