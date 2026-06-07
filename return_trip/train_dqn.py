"""
Train DQN (Tianshou) on the *return* leg (B → A).

Same algorithm as train_dqn.py — vanilla DQN with a target network, ε-greedy,
single env, GPU required. The only difference is the route: it loads the
reversed grades/limits/curves via return_trip.route.load_reverse_route().

Output: results_cm/return/dqn/

Usage:
    python -m return_trip.train_dqn
    python -m return_trip.train_dqn --steps 1500000 --target-update-freq 1000
"""

import argparse
import os
import sys
import time
import numpy as np
import torch

from tianshou.data import Collector, VectorReplayBuffer
from tianshou.env import DummyVectorEnv
from tianshou.utils.net.common import Net
from tianshou.algorithm.modelfree.dqn import DQN, DiscreteQLearningPolicy
from tianshou.algorithm.optim import AdamOptimizerFactory
from tianshou.trainer import OffPolicyTrainerParams

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import env_settings.config as config
from env_settings.gym_env import TrainGymEnv, require_gpu

from return_trip.route import load_reverse_route, print_summary


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--steps', type=int, default=1_000_000)
    p.add_argument('--n-envs', type=int, default=1)
    p.add_argument('--n-test-envs', type=int, default=1)
    p.add_argument('--lr', type=float, default=1e-3)
    p.add_argument('--gamma', type=float, default=0.99)
    p.add_argument('--buffer-size', type=int, default=100_000)
    p.add_argument('--batch-size', type=int, default=64)
    p.add_argument('--collect-steps', type=int, default=8)
    p.add_argument('--epoch-steps', type=int, default=20_000)
    p.add_argument('--eps-start', type=float, default=1.0)
    p.add_argument('--eps-end', type=float, default=0.05)
    p.add_argument('--eps-decay-frac', type=float, default=0.5)
    p.add_argument('--hidden', type=int, nargs='+', default=[128, 64])
    p.add_argument('--target-update-freq', type=int, default=500)
    p.add_argument('--seed', type=int, default=0)
    args = p.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    print('=' * 70)
    print('DQN TRAINING — RETURN TRIP (B → A)')
    print('=' * 70)

    device = require_gpu()
    print(f'Device: {device} ({torch.cuda.get_device_name(0)})')

    print('\n1. Loading reverse route...')
    grades, limits, curves = load_reverse_route()
    print_summary(grades, limits, curves)

    def make_env():
        return TrainGymEnv(grades, limits, curves)
    train_envs = DummyVectorEnv([make_env for _ in range(args.n_envs)])
    test_envs = DummyVectorEnv([make_env for _ in range(args.n_test_envs)])
    sample_env = make_env()

    print('\n2. Building DQN...')
    model = Net(state_shape=2, action_shape=4, hidden_sizes=list(args.hidden))
    policy = DiscreteQLearningPolicy(
        model=model,
        action_space=sample_env.action_space,
        observation_space=sample_env.observation_space,
        eps_training=args.eps_start,
        eps_inference=0.0,
    )
    optim = AdamOptimizerFactory(lr=args.lr)
    algorithm = DQN(
        policy=policy, optim=optim,
        gamma=args.gamma, n_step_return_horizon=1,
        target_update_freq=args.target_update_freq,
        is_double=False,
    )
    algorithm.to(device)

    buffer = VectorReplayBuffer(total_size=args.buffer_size, buffer_num=args.n_envs)
    train_collector = Collector(algorithm, train_envs, buffer)
    test_collector = Collector(algorithm, test_envs)

    decay_steps = max(1, int(args.steps * args.eps_decay_frac))
    def training_fn(epoch, env_step):
        frac = min(1.0, env_step / decay_steps)
        eps = args.eps_start + (args.eps_end - args.eps_start) * frac
        policy.set_eps_training(eps)

    def test_fn(epoch, env_step):
        policy.set_eps_inference(0.0)

    max_epochs = max(1, args.steps // args.epoch_steps)
    params = OffPolicyTrainerParams(
        max_epochs=max_epochs,
        epoch_num_steps=args.epoch_steps,
        training_collector=train_collector,
        test_collector=test_collector,
        collection_step_num_env_steps=args.collect_steps,
        update_step_num_gradient_steps_per_sample=1.0,
        batch_size=args.batch_size,
        test_step_num_episodes=2,
        training_fn=training_fn,
        test_fn=test_fn,
        show_progress=True,
        verbose=False,
    )

    print(f'\n3. Training {args.steps} env steps ({max_epochs} epochs)...')
    t0 = time.time()
    algorithm.run_training(params)
    print(f'\n   Done in {time.time() - t0:.0f} s')

    # Hard-code the base path here so that running this trainer after
    # return_trip.train_qsarsa in the same Python process can't produce
    # a doubled 'results_cm/return/return/dqn/' path.
    out_dir = os.path.join('results_cm', 'return', 'dqn')
    os.makedirs(out_dir, exist_ok=True)
    torch.save(algorithm.state_dict(), os.path.join(out_dir, 'dqn.pt'))

    print('\n4. Greedy rollout...')
    eval_env = make_env()
    obs, _ = eval_env.reset()
    segs, vels, acts, ens = [], [], [], []
    model.eval()
    with torch.no_grad():
        while True:
            obs_t = torch.from_numpy(np.asarray(obs, dtype=np.float32)) \
                         .unsqueeze(0).to(device)
            out = model(obs_t)
            qvals = out[0] if isinstance(out, tuple) else out
            action = int(torch.argmax(qvals, dim=-1).item())
            segs.append(eval_env.env.seg_idx)
            vels.append(eval_env.env.v)
            acts.append(action)
            ens.append(eval_env.env.energy_kwh)
            obs, _, term, trunc, _ = eval_env.step(action)
            if term or trunc:
                break

    np.savez(os.path.join(out_dir, 'speed_profile.npz'),
             segments=np.array(segs), velocities=np.array(vels),
             actions=np.array(acts), energies=np.array(ens))

    final_seg = segs[-1] if segs else 0
    success = final_seg >= eval_env.n_segments - 2
    final_e = ens[-1] if ens else 0.0
    print(f'   Segment: {final_seg}/{eval_env.n_segments} '
          f'{"COMPLETE" if success else "INCOMPLETE"}')
    print(f'   Energy:  {final_e:.1f} kWh')

    aa = np.array(acts) if acts else np.array([0])
    for i, nm in enumerate(['Brake', 'Coast', 'Cruise', 'Power']):
        print(f'   {nm:6s}: {(aa == i).mean() * 100:.1f}%')

    print('\n' + '=' * 70)
    print(f'COMPLETE — outputs in {out_dir}/')
    print('=' * 70)


if __name__ == '__main__':
    main()
