"""
Train PPO (Proximal Policy Optimization) — Tianshou
====================================================
Same problem as train_dqn.py / train_qsarsa.py: pick a brake/coast/cruise/power
action per 100-m track segment to drive a train along the route while keeping
energy low.

Physics — UNCHANGED. The Gymnasium wrapper here delegates every step to the
existing TrainEnv (env_settings/environment.py), which calls
TrainPhysics.get_total_resistance(...) — the Davis equation
    r_t(V) = C0 + C1*V + C2*V^2
defined in env_settings/physics.py. Nothing about the dynamics is altered;
only the API around them is adapted to Gymnasium so Tianshou's PPO can drive it.

GPU: runs on CUDA when available (DGX Spark GB10 here).

Usage:
    python train_ppo.py                       # 1M steps, 8 parallel envs
    python train_ppo.py --steps 200000        # quick run
    python train_ppo.py --n-envs 16 --hidden 256 128
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
from tianshou.utils.net.discrete import (
    DiscreteActor,
    DiscreteCritic,
    dist_fn_categorical_from_logits,
)
from tianshou.algorithm.modelfree.ppo import PPO
from tianshou.algorithm.modelfree.reinforce import ProbabilisticActorPolicy
from tianshou.algorithm.optim import AdamOptimizerFactory
from tianshou.trainer import OnPolicyTrainerParams

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import env_settings.config as config
from env_settings.gym_env import TrainGymEnv, load_data, require_gpu


# ---------------------------------------------------------------------------
# Greedy rollout (used after training to produce the deterministic profile)
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
    parser = argparse.ArgumentParser()
    parser.add_argument('--steps', type=int, default=1_000_000,
                        help='total environment steps')
    parser.add_argument('--n-envs', type=int, default=8)
    parser.add_argument('--n-test-envs', type=int, default=2)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--gae-lambda', type=float, default=0.95)
    parser.add_argument('--clip', type=float, default=0.2)
    parser.add_argument('--ent-coef', type=float, default=0.01)
    parser.add_argument('--vf-coef', type=float, default=0.5)
    parser.add_argument('--collect-steps', type=int, default=2048,
                        help='env steps per rollout (per env)')
    parser.add_argument('--epoch-steps', type=int, default=20000)
    parser.add_argument('--batch-size', type=int, default=256)
    parser.add_argument('--n-update-epochs', type=int, default=10,
                        help='PPO epochs per collected rollout')
    parser.add_argument('--hidden', type=int, nargs='+', default=[128, 64])
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    print('=' * 70)
    print('PPO TRAINING (Tianshou) — Train Speed Profile Optimization')
    print('Physics: Davis equation (env_settings/physics.py — UNCHANGED)')
    print('=' * 70)

    device = require_gpu()
    print(f'Device: {device} ({torch.cuda.get_device_name(0)})')

    print('\n1. Loading route data...')
    grades, limits, curves = load_data()
    print(f'   Route: {len(grades)} segments, {len(grades) * 0.1:.1f} km')

    print('\n2. Building Gymnasium-wrapped envs...')
    def make_env():
        return TrainGymEnv(grades, limits, curves)

    train_envs = DummyVectorEnv([make_env for _ in range(args.n_envs)])
    test_envs = DummyVectorEnv([make_env for _ in range(args.n_test_envs)])
    sample_env = make_env()

    print('\n3. Building PPO networks...')
    obs_dim = 2
    n_actions = 4

    actor_trunk = Net(state_shape=obs_dim, hidden_sizes=list(args.hidden),
                      activation=torch.nn.Tanh)
    actor = DiscreteActor(preprocess_net=actor_trunk, action_shape=n_actions,
                          softmax_output=False)  # outputs raw logits

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

    print(f'  Actor + Critic: 2 -> {"->".join(map(str, args.hidden))} -> {n_actions} (tanh)')
    print(f'  Optimizer: Adam(lr={args.lr})')
    print(f'  PPO: clip={args.clip} gamma={args.gamma} gae={args.gae_lambda} '
          f'ent={args.ent_coef} vf={args.vf_coef}')

    print('\n4. Collectors + replay buffer...')
    buffer = VectorReplayBuffer(
        total_size=args.collect_steps * args.n_envs,
        buffer_num=args.n_envs,
    )
    train_collector = Collector(algorithm, train_envs, buffer)
    test_collector = Collector(algorithm, test_envs)

    max_epochs = max(1, args.steps // args.epoch_steps)
    params = OnPolicyTrainerParams(
        max_epochs=max_epochs,
        epoch_num_steps=args.epoch_steps,
        training_collector=train_collector,
        test_collector=test_collector,
        collection_step_num_env_steps=args.collect_steps,
        update_step_num_repetitions=args.n_update_epochs,
        batch_size=args.batch_size,
        test_step_num_episodes=2,
        show_progress=True,
        verbose=False,
    )

    print(f'\n5. Training PPO for {args.steps} env steps ({max_epochs} epochs)...')
    t0 = time.time()
    info = algorithm.run_training(params)
    elapsed = time.time() - t0
    print(f'\n   Training done in {elapsed:.0f} s')

    # -------- save + evaluate --------
    out_dir = os.path.join(getattr(config, 'OUTPUT_DIR', 'results_cm'), 'ppo')
    os.makedirs(out_dir, exist_ok=True)
    torch.save(algorithm.state_dict(), os.path.join(out_dir, 'ppo.pt'))
    print(f'   Checkpoint -> {out_dir}/ppo.pt')

    print('\n6. Greedy rollout for speed profile...')
    eval_env = make_env()
    segs, vels, acts, ens = greedy_rollout(actor, eval_env, device)

    np.savez(os.path.join(out_dir, 'speed_profile.npz'),
             segments=np.array(segs), velocities=np.array(vels),
             actions=np.array(acts), energies=np.array(ens))

    final_seg = segs[-1] if segs else 0
    success = final_seg >= eval_env.n_segments - 2
    final_e = ens[-1] if ens else 0.0
    final_t = len(segs) * getattr(config, 'DT', 1.0)

    print(f"   Segment: {final_seg}/{eval_env.n_segments} "
          f"{'COMPLETE' if success else 'INCOMPLETE'}")
    print(f"   Energy:  {final_e:.1f} kWh")
    print(f"   Time:    {final_t:.0f} s")
    aa = np.array(acts) if acts else np.array([0])
    for i, nm in enumerate(['Brake', 'Coast', 'Cruise', 'Power']):
        print(f"   {nm:6s}: {(aa == i).mean() * 100:.1f}%")

    # plot
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig, (a1, a2, a3) = plt.subplots(3, 1, figsize=(14, 10), sharex=True)
        fig.suptitle(f'PPO Speed Profile — Energy={final_e:.0f} kWh '
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
        print(f"   Plot -> {out_dir}/speed_profile.png")
    except ImportError:
        pass

    print('\n' + '=' * 70)
    print('PPO TRAINING COMPLETE')
    print('=' * 70)


if __name__ == '__main__':
    main()
