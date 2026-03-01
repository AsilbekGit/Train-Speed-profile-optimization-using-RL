"""
Q-SARSA Training Script
========================
Usage:
    python train_qsarsa.py
    python train_qsarsa.py --phi 0.10 --episodes 5000
"""

import argparse
import os
import sys
import numpy as np

# Import project modules
try:
    import env_settings.config as config
    from env_settings.config import TrainPhysics, TrainEnv
except ImportError:
    try:
        from env_settings.environment import TrainEnv
        from env_settings.physics import TrainPhysics
        import env_settings.config
    except ImportError:
        print("ERROR: Cannot import project modules (config, TrainPhysics, TrainEnv)")
        print("Make sure you're running from the qsarsa_dqn/ directory")
        sys.exit(1)

try:
    from qsarsa_dqn.qsarsa import QSARSA
except ImportError:
    print("ERROR: Cannot import qsarsa module")
    print("Make sure qsarsa.py is in the current directory")
    sys.exit(1)


def load_route_data():
    """Load route data (grades, speed limits, curves)."""
    for path in ['route_data.npz', 'data/route_data.npz', '../route_data.npz']:
        if os.path.exists(path):
            data = np.load(path, allow_pickle=True)
            return data['grades'], data['limits'], data.get('curves', np.zeros_like(data['grades']))
    
    if hasattr(config, 'GRADES') and hasattr(config, 'SPEED_LIMITS'):
        grades = np.array(config.GRADES)
        limits = np.array(config.SPEED_LIMITS)
        curves = np.array(getattr(config, 'CURVES', np.zeros_like(grades)))
        return grades, limits, curves
    
    grade_files = ['grades.npy', 'data/grades.npy', 'grade_data.npy']
    for gf in grade_files:
        if os.path.exists(gf):
            grades = np.load(gf)
            limits = np.load(gf.replace('grade', 'speed_limit')) if os.path.exists(gf.replace('grade', 'speed_limit')) else np.full_like(grades, 120.0)
            curves = np.load(gf.replace('grade', 'curve')) if os.path.exists(gf.replace('grade', 'curve')) else np.zeros_like(grades)
            return grades, limits, curves
    
    print("WARNING: No route data found. Using default 749-segment route.")
    n_segments = 749
    return np.zeros(n_segments), np.full(n_segments, 120.0), np.zeros(n_segments)


def main():
    parser = argparse.ArgumentParser(description='Train Q-SARSA for speed profile optimization')
    parser.add_argument('--phi', type=float, default=0.10, help='CM threshold φ (default: 0.10)')
    parser.add_argument('--episodes', type=int, default=5000, help='Number of training episodes (default: 5000)')
    args = parser.parse_args()
    
    phi = args.phi
    episodes = args.episodes
    
    print("=" * 70)
    print("Q-SARSA TRAINING")
    print("Train Speed Profile Optimization")
    print("=" * 70)
    
    # 1. Load route data
    print("\n1. Loading route data...")
    grades, limits, curves = load_route_data()
    print(f"   Route: {len(grades)} segments ({len(grades) * 0.1:.1f} km)")
    
    # 2. Initialize environment
    print("\n2. Initializing environment...")
    physics = TrainPhysics()
    env = TrainEnv(physics, grades, limits, curves)
    
    # 3. Set φ
    print(f"\n3. Using φ = {phi}")
    
    # 4. Initialize Q-SARSA
    print("\n4. Initializing Q-SARSA...")
    agent = QSARSA(env, phi_threshold=phi)
    
    # 5. Train
    print(f"\n5. Starting training ({episodes} episodes)...")
    agent.train(episodes=episodes)
    
    # 6. Generate speed profile
    print("\n6. Generating optimal speed profile...")
    agent.generate_speed_profile()
    
    print("\n" + "=" * 70)
    print("TRAINING COMPLETE!")
    print("=" * 70)
    output_dir = os.path.join(getattr(config, 'OUTPUT_DIR', 'results_cm'), "qsarsa")
    print(f"\nResults saved to: {output_dir}/")


if __name__ == "__main__":
    main()