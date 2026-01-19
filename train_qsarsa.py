"""
Train Q-SARSA Algorithm
=======================
Run this AFTER CM analysis to train with your φ value

Usage:
    python train_qsarsa.py
    
    Or with custom phi:
    python train_qsarsa.py --phi 0.10 --episodes 5000
"""

import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.utils import load_data
from env_settings.physics import TrainPhysics
from env_settings.environment import TrainEnv
from qsarsa_dqn.qsarsa import QSARSA

def main():
    print("="*70)
    print("Q-SARSA TRAINING")
    print("="*70)
    
    # Parse command line args
    phi = 0.10  # Default
    episodes = 5000
    
    for i, arg in enumerate(sys.argv):
        if arg == '--phi' and i+1 < len(sys.argv):
            phi = float(sys.argv[i+1])
        if arg == '--episodes' and i+1 < len(sys.argv):
            episodes = int(sys.argv[i+1])
    
    # 1. Load data
    print("\n1. Loading route data...")
    grades, limits, curves = load_data()
    
    # 2. Initialize physics and environment
    print("\n2. Initializing environment...")
    physics = TrainPhysics()
    env = TrainEnv(physics, grades, limits, curves)
    
    # 3. Set φ value
    print(f"\n3. Using φ = {phi}")
    print("   (Change with: python train_qsarsa.py --phi 0.05)")
    
    # 4. Initialize Q-SARSA
    print("\n4. Initializing Q-SARSA...")
    qsarsa = QSARSA(env, phi_threshold=phi)
    
    # 5. Train
    print(f"\n5. Starting training ({episodes} episodes)...")
    q_table = qsarsa.train(episodes=episodes)
    
    # 6. Generate speed profile
    print("\n6. Generating optimal speed profile...")
    qsarsa.generate_speed_profile()
    
    print("\n" + "="*70)
    print("TRAINING COMPLETE!")
    print("="*70)
    print(f"\nResults saved to: results_cm/qsarsa/")
    print("Files:")
    print("  - qsarsa_data.npz (Q-table and training history)")
    print("  - qsarsa_training.png (training plots)")
    print("  - speed_profile.npz (optimal trajectory)")
    print("  - speed_profile.png (trajectory visualization)")

if __name__ == "__main__":
    main()