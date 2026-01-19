"""
Train Deep Q-Network (GPU Version)
==================================
Uses PyTorch with CUDA for GPU acceleration

Requirements:
    pip install torch

Usage:
    python train_dqn.py
    python train_dqn.py --phi 0.10 --episodes 5000
"""

import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Check PyTorch and GPU
try:
    import torch
    print("="*70)
    print("DEEP Q-NETWORK TRAINING (PyTorch GPU)")
    print("="*70)
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"CUDA version: {torch.version.cuda}")
    print("="*70)
except ImportError:
    print("ERROR: PyTorch not installed!")
    print("Install with: pip install torch")
    sys.exit(1)

from data.utils import load_data
from env_settings.physics import TrainPhysics
from env_settings.environment import TrainEnv
from qsarsa_dqn.dqn import DeepQNetwork

def main():
    print("\n" + "="*70)
    print("INITIALIZING DQN TRAINING")
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
    
    print("\n" + "="*70)
    print("TRAINING COMPLETE!")
    print("="*70)
    print(f"\nResults saved to: results_cm/deep_q/")
    print("Files:")
    print("  - dqn_weights.npz (network weights and history)")
    print("  - dqn_training.png (training plots)")
    print("  - speed_profile.npz (optimal trajectory)")

if __name__ == "__main__":
    main()