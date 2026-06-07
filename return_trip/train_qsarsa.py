"""
Train Q-SARSA on the *return* leg (B → A).

Identical to train_qsarsa.py except it loads the reversed route via
return_trip.route.load_reverse_route(). The algorithm code in
qsarsa_dqn/qsarsa.py is reused unchanged.

Output: results_cm/return/qsarsa/  — qsarsa.save_results joins
config.OUTPUT_DIR with "qsarsa". To redirect without modifying qsarsa.py
we mutate config.OUTPUT_DIR scoped to main() (try/finally restores it),
so running this script in the same Python session as the forward Q-SARSA
trainer doesn't cross-contaminate paths.

Usage:
    python -m return_trip.train_qsarsa
    python -m return_trip.train_qsarsa --phi 0.10 --episodes 5000
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import env_settings.config as config
from env_settings.physics import TrainPhysics
from env_settings.environment import TrainEnv
from qsarsa_dqn.qsarsa import QSARSA

from return_trip.route import load_reverse_route, print_summary


RETURN_OUTPUT_DIR = 'results_cm/return'


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--phi', type=float, default=0.10)
    p.add_argument('--episodes', type=int, default=5000)
    args = p.parse_args()

    print('=' * 70)
    print('Q-SARSA TRAINING — RETURN TRIP (B → A)')
    print('=' * 70)

    # Scope the config mutation to this function so back-to-back imports
    # in the same Python process don't bleed forward writes into the
    # return folder (or vice-versa).
    original_output_dir = config.OUTPUT_DIR
    config.OUTPUT_DIR = RETURN_OUTPUT_DIR
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)
    try:
        print('\n1. Loading reverse route...')
        grades, limits, curves = load_reverse_route()
        print_summary(grades, limits, curves)

        print('\n2. Building env (Davis physics — UNCHANGED)...')
        physics = TrainPhysics()
        env = TrainEnv(physics, grades, limits, curves)

        print(f'\n3. phi = {args.phi}')
        agent = QSARSA(env, phi_threshold=args.phi)

        print(f'\n4. Training {args.episodes} episodes...')
        agent.train(episodes=args.episodes)

        print('\n5. Speed profile...')
        agent.generate_speed_profile()
    finally:
        config.OUTPUT_DIR = original_output_dir

    print('\n' + '=' * 70)
    print(f'COMPLETE — outputs in {RETURN_OUTPUT_DIR}/qsarsa/')
    print('=' * 70)


if __name__ == '__main__':
    main()
