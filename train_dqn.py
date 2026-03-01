"""
DQN Training Script
===================
Usage:
    python train_dqn.py
    python train_dqn.py --phi 0.10 --episodes 5000
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
        import env_settings.config as config
    except ImportError:
        print("ERROR: Cannot import project modules (config, TrainPhysics, TrainEnv)")
        print("Make sure you're running from the qsarsa_dqn/ directory")
        sys.exit(1)

from qsarsa_dqn.dqn import DeepQNetwork


def load_route_data():
    """Load route data (grades, speed limits, curves)."""
    # Try loading from standard locations
    for path in ['route_data.npz', 'data/route_data.npz', '../route_data.npz']:
        if os.path.exists(path):
            data = np.load(path, allow_pickle=True)
            return data['grades'], data['limits'], data.get('curves', np.zeros_like(data['grades']))
    
    # Try loading from config
    if hasattr(config, 'GRADES') and hasattr(config, 'SPEED_LIMITS'):
        grades = np.array(config.GRADES)
        limits = np.array(config.SPEED_LIMITS)
        curves = np.array(getattr(config, 'CURVES', np.zeros_like(grades)))
        return grades, limits, curves
    
    # Try loading individual files
    grade_files = ['grades.npy', 'data/grades.npy', 'grade_data.npy']
    for gf in grade_files:
        if os.path.exists(gf):
            grades = np.load(gf)
            # Look for corresponding files
            limits = np.load(gf.replace('grade', 'speed_limit')) if os.path.exists(gf.replace('grade', 'speed_limit')) else np.full_like(grades, 120.0)
            curves = np.load(gf.replace('grade', 'curve')) if os.path.exists(gf.replace('grade', 'curve')) else np.zeros_like(grades)
            return grades, limits, curves
    
    print("WARNING: No route data found. Using default 749-segment route.")
    n_segments = 749
    grades = np.zeros(n_segments)
    limits = np.full(n_segments, 120.0)
    curves = np.zeros(n_segments)
    return grades, limits, curves


def main():
    parser = argparse.ArgumentParser(description='Train Deep Q-Network for speed profile optimization')
    parser.add_argument('--phi', type=float, default=0.10, help='CM threshold φ (default: 0.10)')
    parser.add_argument('--episodes', type=int, default=5000, help='Number of training episodes (default: 5000)')
    args = parser.parse_args()
    
    phi = args.phi
    episodes = args.episodes
    
    print("=" * 70)
    print("DEEP Q-NETWORK TRAINING")
    print("Train Speed Profile Optimization")
    print("=" * 70)
    
    # 1. Load route data
    print("\n1. Loading route data...")
    grades, limits, curves = load_route_data()
    print(f"   Route: {len(grades)} segments ({len(grades) * 0.1:.1f} km)")
    
    # 2. Initialize physics and environment
    print("\n2. Initializing environment...")
    physics = TrainPhysics()
    env = TrainEnv(physics, grades, limits, curves)
    
    # 3. Set φ value
    print(f"\n3. Using φ = {phi}")
    print("   (Change with: python train_dqn.py --phi 0.05)")
    
    # 4. Initialize DQN
    print("\n4. Initializing Deep Q-Network...")
    dqn = DeepQNetwork(env, phi_threshold=phi)
    
    # 5. Train
    print(f"\n5. Starting training ({episodes} episodes)...")
    dqn.train(episodes=episodes)
    
    # 6. Generate speed profile
    print("\n6. Generating optimal speed profile...")
    dqn.generate_speed_profile()
    
    print("\n" + "=" * 70)
    print("TRAINING COMPLETE!")
    print("=" * 70)
    output_dir = os.path.join(getattr(config, 'OUTPUT_DIR', 'results_cm'), "deep_q")
    print(f"\nResults saved to: {output_dir}/")
    print("Files:")
    print("  - dqn_weights.npz (network weights and history)")
    print("  - dqn_training.png (training plots)")
    print("  - speed_profile.npz (optimal trajectory)")
    print("  - speed_profile.png (speed profile visualization)")


if __name__ == "__main__":
    main()