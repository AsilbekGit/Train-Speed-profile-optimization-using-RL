"""
Train DQN — Tianshou. Vanilla DQN with target network, GPU only.
=================================================================
Solid plain DQN. No phases, no Q-SARSA bootstrap, no distillation,
no greedy sweep.

  - target network synced to online net every --target-update-freq grad steps
    (canonical Mnih-et-al-2015 setup)
  - is_double=False (no double-DQN)
  - epsilon-greedy with linear decay
  - GPU is *required* (script aborts if CUDA is unavailable)

Physics: env_settings/physics.py (Davis equation) — UNCHANGED.

Usage:
    python train_dqn.py
    python train_dqn.py --steps 1500000 --n-envs 8 --hidden 128 64
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

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import env_settings.config as config
from env_settings.gym_env import TrainGymEnv, load_data, require_gpu


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--steps', type=int, default=1_000_000,
                   help='total environment steps')
    p.add_argument('--n-envs', type=int, default=1)
    p.add_argument('--n-test-envs', type=int, default=1)
    p.add_argument('--lr', type=float, default=1e-3)
    p.add_argument('--gamma', type=float, default=0.99)
    p.add_argument('--buffer-size', type=int, default=100_000)
    p.add_argument('--batch-size', type=int, default=64)
    p.add_argument('--collect-steps', type=int, default=8,
                   help='env steps between gradient-update cycles')
    p.add_argument('--epoch-steps', type=int, default=20_000)
    p.add_argument('--eps-start', type=float, default=1.0)
    p.add_argument('--eps-end', type=float, default=0.05)
    p.add_argument('--eps-decay-frac', type=float, default=0.5,
                   help='fraction of total steps over which eps decays')
    p.add_argument('--hidden', type=int, nargs='+', default=[128, 64])
    p.add_argument('--target-update-freq', type=int, default=500,
                   help='gradient steps between target-net syncs (0 disables target)')
    p.add_argument('--seed', type=int, default=0)
    args = p.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    print('=' * 70)
    print('DQN TRAINING (Tianshou) — vanilla DQN with target network, GPU only')
    print('Physics: Davis equation (env_settings/physics.py — UNCHANGED)')
    print('=' * 70)

    device = require_gpu()
    print(f'Device: {device} ({torch.cuda.get_device_name(0)})')

    print('\n1. Loading route data...')
    grades, limits, curves = load_data()
    print(f'   {len(grades)} segments, {len(grades) * 0.1:.1f} km')

    print('\n2. Building envs...')
    def make_env():
        return TrainGymEnv(grades, limits, curves)
    train_envs = DummyVectorEnv([make_env for _ in range(args.n_envs)])
    test_envs = DummyVectorEnv([make_env for _ in range(args.n_test_envs)])
    sample_env = make_env()

    print('\n3. Building DQN net...')
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
        policy=policy,
        optim=optim,
        gamma=args.gamma,
        n_step_return_horizon=1,
        target_update_freq=args.target_update_freq,  # 0 disables target net
        is_double=False,                             # vanilla DQN
    )
    algorithm.to(device)
    target_str = (f'target sync every {args.target_update_freq} grad steps'
                  if args.target_update_freq > 0 else 'no target')
    print(f'   Net: 2 -> {"->".join(map(str, args.hidden))} -> 4 (relu, on {device})')
    print(f'   gamma={args.gamma} lr={args.lr} buffer={args.buffer_size} '
          f'batch={args.batch_size}  ({target_str}, no double)')

    print('\n4. Collectors + replay buffer...')
    buffer = VectorReplayBuffer(total_size=args.buffer_size, buffer_num=args.n_envs)
    train_collector = Collector(algorithm, train_envs, buffer)
    test_collector = Collector(algorithm, test_envs)

    decay_steps = max(1, int(args.steps * args.eps_decay_frac))
    eps_start, eps_end = args.eps_start, args.eps_end

    def training_fn(epoch, env_step):
        frac = min(1.0, env_step / decay_steps)
        eps = eps_start + (eps_end - eps_start) * frac
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

    print(f'\n5. Training {args.steps} env steps ({max_epochs} epochs)...')
    t0 = time.time()
    algorithm.run_training(params)
    print(f'\n   Done in {time.time() - t0:.0f} s')

    out_dir = os.path.join(getattr(config, 'OUTPUT_DIR', 'results_cm'), 'dqn')
    os.makedirs(out_dir, exist_ok=True)
    torch.save(algorithm.state_dict(), os.path.join(out_dir, 'dqn.pt'))
    print(f'   Checkpoint -> {out_dir}/dqn.pt')

    print('\n6. Greedy rollout for speed profile...')
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
    final_t = len(segs) * getattr(config, 'DT', 1.0)
    print(f'   Segment: {final_seg}/{eval_env.n_segments} '
          f'{"COMPLETE" if success else "INCOMPLETE"}')
    print(f'   Energy:  {final_e:.1f} kWh')
    print(f'   Time:    {final_t:.0f} s')
    aa = np.array(acts) if acts else np.array([0])
    for i, nm in enumerate(['Brake', 'Coast', 'Cruise', 'Power']):
        print(f'   {nm:6s}: {(aa == i).mean() * 100:.1f}%')

    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig, (a1, a2, a3) = plt.subplots(3, 1, figsize=(14, 10), sharex=True)
        fig.suptitle(f'DQN Speed Profile — Energy={final_e:.0f} kWh '
                     f'({"COMPLETE" if success else "INCOMPLETE"})')
        pos = np.array(segs) * getattr(config, 'DX', 100) / 1000
        a1.plot(pos, np.array(vels) * 3.6, 'b-', lw=1.5)
        a1.set_ylabel('Speed (km/h)'); a1.grid(alpha=0.3); a1.set_title('Speed')
        a2.plot(pos, ens, 'r-', lw=1.5)
        a2.set_ylabel('Energy (kWh)'); a2.grid(alpha=0.3); a2.set_title('Cumulative Energy')
        colors = ['red', 'orange', 'blue', 'green']
        names = ['Brake', 'Coast', 'Cruise', 'Power']
        for i in range(4):
            m = aa == i
            if m.any():
                a3.scatter(pos[m], [i] * m.sum(), c=colors[i], s=2,
                           label=names[i], alpha=0.5)
        a3.set_yticks([0, 1, 2, 3]); a3.set_yticklabels(names)
        a3.set_xlabel('Position (km)'); a3.set_ylabel('Action')
        a3.legend(markerscale=4); a3.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, 'speed_profile.png'), dpi=200)
        plt.close()
        print(f'   Plot -> {out_dir}/speed_profile.png')
    except ImportError:
        pass

    print('\n' + '=' * 70)
    print('DQN TRAINING COMPLETE')
    print('=' * 70)


if __name__ == '__main__':
    main()
