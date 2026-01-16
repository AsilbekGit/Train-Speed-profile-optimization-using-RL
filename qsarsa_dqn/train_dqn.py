"""
Train Deep-Q Network
Run this AFTER completing CM analysis to find YOUR φ value
"""

from data.utils import load_data
from env_settings.physics import TrainPhysics
from env_settings.environment import TrainEnv
from dqn import DeepQNetwork

def main():
    print("="*70)
    print("DEEP-Q NETWORK TRAINING")
    print("="*70)
    
    # 1. Load data
    print("\n1. Loading route data...")
    grades, limits, curves = load_data()
    
    # 2. Initialize physics and environment
    print("\n2. Initializing environment...")
    physics = TrainPhysics()
    env = TrainEnv(physics, grades, limits, curves)
    
    # 3. Set YOUR φ value from CM analysis
    print("\n3. Setting φ threshold...")
    print("   ⚠️  IMPORTANT: Use YOUR φ value from CM analysis!")
    print("   Check results_cm/cm_summary.txt for suggested φ")
    
    YOUR_PHI = 0.04  # CHANGE THIS to your actual φ value!
    
    print(f"   Using φ = {YOUR_PHI}")
    confirm = input("   Is this correct? (y/n): ")
    
    if confirm.lower() != 'y':
        try:
            YOUR_PHI = float(input("   Enter YOUR φ value: "))
        except:
            print("   Invalid input, using default 0.04")
            YOUR_PHI = 0.04
    
    # 4. Initialize Deep-Q Network
    print("\n4. Initializing Deep-Q Network...")
    dqn = DeepQNetwork(env, phi_threshold=YOUR_PHI)
    
    # 5. Train
    print("\n5. Starting training...")
    episodes = 1000  # Adjust as needed
    dqn.train(episodes=episodes)
    
    print("\n" + "="*70)
    print("TRAINING COMPLETE!")
    print("="*70)
    print(f"\nResults saved to: results_cm/deep_q/")
    print("Check deep_q_training.png for performance plots")

if __name__ == "__main__":
    main()