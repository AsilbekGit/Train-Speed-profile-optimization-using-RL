"""
Train Deep Q-Network v2 (Pragmatic)
=====================================
Usage:
    python train_dqn.py
    python train_dqn.py --phi 0.99 --episodes 5000

KEY DIFFERENCE from v1:
  Phase 1 now clones Q-SARSA exactly (eps=0.10, env reward).
  This guarantees high success rate before network training begins.
  
  v1 used eps=1.0 (fully random) -> only 2% success in Phase 1.
  v2 uses eps=0.10 (like Q-SARSA) -> should get 80%+ success.
"""

import sys
import os
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def load_data():
    search = [
        'data/data.csv', '../data/data.csv', '../../data/data.csv',
        os.path.join(os.path.dirname(__file__), 'data', 'data.csv'),
        os.path.join(os.path.dirname(__file__), '..', 'data', 'data.csv'),
    ]
    csv_path = None
    for p in search:
        if os.path.exists(p):
            csv_path = p; break
    if csv_path is None:
        for mod_name in ['config', 'env_settings.config']:
            try:
                cfg = __import__(mod_name, fromlist=['DATA_PATH'])
                if hasattr(cfg, 'DATA_PATH') and os.path.exists(cfg.DATA_PATH):
                    csv_path = cfg.DATA_PATH; break
            except ImportError: continue
    if csv_path is None:
        print("ERROR: Cannot find data/data.csv!"); sys.exit(1)

    print(f"   Loading from: {csv_path}")
    df = pd.read_csv(csv_path)
    print(f"   Columns: {list(df.columns)}, Segments: {len(df)}")

    col = {c.lower().strip(): c for c in df.columns}
    gk = next((k for k in col if 'grade' in k), None)
    grades = df[col[gk]].values.astype(float) if gk else np.zeros(len(df))
    lk = next((k for k in col if 'speed' in k), None)
    limits = df[col[lk]].values.astype(float) if lk else np.full(len(df), 22.0)
    ck = next((k for k in col if 'curv' in k), None)
    curves = df[col[ck]].values.astype(float) if ck else np.zeros(len(df))

    ns = limits[limits > 1]
    print(f"\n   Route: {len(grades)} segments, {len(grades)*0.1:.1f} km")
    print(f"   Grade: [{grades.min():.4f}%, {grades.max():.4f}%]")
    if len(ns) > 0:
        print(f"   Speed limit: [{ns.min():.1f}, {ns.max():.1f}] m/s")
    print(f"   Stations: {np.sum(limits == 1)}")
    return grades, limits, curves


def get_env(grades, limits, curves):
    for pm, em in [('env_settings.physics', 'env_settings.environment'),
                    ('physics', 'environment')]:
        try:
            P = __import__(pm, fromlist=['TrainPhysics']).TrainPhysics
            E = __import__(em, fromlist=['TrainEnv']).TrainEnv
            return P(), E(P(), grades, limits, curves)
        except (ImportError, AttributeError): continue
    print("ERROR: Cannot import TrainPhysics/TrainEnv"); sys.exit(1)


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--phi', type=float, default=0.99)
    parser.add_argument('--episodes', type=int, default=5000)
    args = parser.parse_args()

    print("=" * 70)
    print("DEEP Q-NETWORK v2 (Pragmatic)")
    print("Train Speed Profile Optimization")
    print("Target: < 1500 kWh energy consumption")
    print("=" * 70)
    print("\nKey fix: Phase 1 clones Q-SARSA exactly (eps=0.10, env reward)")
    print("  v1 had eps=1.0 -> 2% success. v2 uses eps=0.10 -> 80%+ success.")

    print("\n1. Loading route data...")
    grades, limits, curves = load_data()

    print("\n2. Initializing environment...")
    physics, env = get_env(grades, limits, curves)

    print(f"\n3. Using phi = {args.phi}")

    print("\n4. Initializing Deep Q-Network v2...")
    # Try multiple import paths
    dqn_class = None
    for mod_path in ['qsarsa_dqn.dqn', 'dqn']:
        try:
            mod = __import__(mod_path, fromlist=['DeepQNetwork'])
            dqn_class = mod.DeepQNetwork
            print(f"   Using: {mod_path}")
            break
        except ImportError:
            continue
    
    if dqn_class is None:
        print("ERROR: Cannot find dqn.py!")
        print("Place dqn.py in qsarsa_dqn/ or project root")
        sys.exit(1)

    dqn = dqn_class(env, phi_threshold=args.phi)

    print(f"\n5. Starting training ({args.episodes} episodes)...")
    dqn.train(episodes=args.episodes)

    print("\n6. Generating optimal speed profile...")
    dqn.generate_speed_profile()

    print("\n" + "=" * 70)
    print("TRAINING COMPLETE!")
    print("=" * 70)


if __name__ == "__main__":
    main()