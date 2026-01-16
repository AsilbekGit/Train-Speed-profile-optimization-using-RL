"""
Quick test script to verify setup before running full training

This will:
1. Test data loading
2. Test physics calculations
3. Run 2 quick episodes with full debug output
4. Show what's happening step-by-step

Run this BEFORE running the full training!
"""

from data.utils import load_data
from physics import TrainPhysics
from environment import TrainEnv
from cm_analyzer import CMAnalyzer
import config

def test_data_loading():
    """Test if data files can be loaded"""
    print("\n" + "="*70)
    print("TEST 1: Data Loading")
    print("="*70)
    
    try:
        grades, limits, curves = load_data()
        print(f"✓ Data loaded successfully!")
        print(f"  Segments: {len(grades)}")
        print(f"  Grade range: [{grades.min():.2f}, {grades.max():.2f}] %")
        print(f"  Speed limit range: [{limits.min():.2f}, {limits.max():.2f}] m/s")
        print(f"  Curvature range: [{curves.min():.4f}, {curves.max():.4f}] %")
        return grades, limits, curves
    except Exception as e:
        print(f"✗ ERROR loading data: {e}")
        return None, None, None

def test_physics(physics):
    """Test physics calculations"""
    print("\n" + "="*70)
    print("TEST 2: Physics Calculations")
    print("="*70)
    
    # Test at different speeds
    test_speeds = [0, 10, 20, 30, 40]  # m/s
    
    print("\nTraction Force vs Speed:")
    print("-" * 50)
    for v in test_speeds:
        f_trac = physics.get_max_traction_force(v)
        print(f"  v={v:2d}m/s: F_traction = {f_trac:10.2f} N")
    
    print("\nResistance Force (flat, no curve):")
    print("-" * 50)
    for v in test_speeds:
        r = physics.get_total_resistance(v, grade_pct=0, curv_pct=0)
        print(f"  v={v:2d}m/s: Resistance = {r:10.2f} N")
    
    print("\nResistance Force (5% grade, 0.1% curve):")
    print("-" * 50)
    for v in test_speeds:
        r = physics.get_total_resistance(v, grade_pct=5.0, curv_pct=0.1)
        print(f"  v={v:2d}m/s: Resistance = {r:10.2f} N")

def test_environment_step(env):
    """Test a single environment step"""
    print("\n" + "="*70)
    print("TEST 3: Environment Step")
    print("="*70)
    
    state = env.reset()
    print(f"\nInitial state: seg={state[0]}, v={state[1]:.2f}m/s")
    
    print("\nTesting each action for 5 steps:")
    print("-" * 70)
    
    for action_idx in range(4):
        env.reset()
        action_name = config.ACTION_NAMES[action_idx]
        print(f"\n  Action: {action_name}")
        
        for step in range(5):
            state, reward, done, info = env.step(action_idx)
            print(f"    Step {step+1}: v={info['velocity']:6.2f}m/s, "
                  f"seg={info['segment']:3d}, "
                  f"reward={reward:8.2f}, "
                  f"done={done}")
            if done:
                break

def test_quick_episodes():
    """Run 2 quick episodes with debug mode"""
    print("\n" + "="*70)
    print("TEST 4: Quick Episode Test (with debug output)")
    print("="*70)
    print("\nThis will run 2 episodes with full debug output.")
    print("You'll see exactly what's happening step-by-step.\n")
    
    # Temporarily enable debug mode
    original_debug = config.DEBUG_MODE
    original_print_step = config.PRINT_EVERY_STEP
    
    config.DEBUG_MODE = True
    config.PRINT_EVERY_STEP = True
    
    try:
        # Load data
        grades, limits, curves = load_data()
        
        # Initialize
        physics = TrainPhysics()
        env = TrainEnv(physics, grades, limits, curves)
        analyzer = CMAnalyzer(env)
        
        # Run 2 episodes
        print("\nRunning 2 test episodes...")
        analyzer.run(episodes=2)
        
        print("\n" + "="*70)
        print("TEST 4 COMPLETE")
        print("="*70)
        print("\nIf you saw detailed output above, everything is working!")
        print("Check the results_cm/ folder for the plot.")
        
    finally:
        # Restore original settings
        config.DEBUG_MODE = original_debug
        config.PRINT_EVERY_STEP = original_print_step

def main():
    """Run all tests"""
    print("\n" + "🚂"*35)
    print("\n  TRAIN SPEED OPTIMIZATION - SETUP TEST")
    print("\n" + "🚂"*35)
    
    print("\nThis script will test your setup before running full training.")
    print("It will take about 1-2 minutes.\n")
    
    input("Press Enter to start tests...")
    
    # Test 1: Data Loading
    grades, limits, curves = test_data_loading()
    if grades is None:
        print("\n❌ Data loading failed. Please check your data files.")
        return
    
    # Test 2: Physics
    physics = TrainPhysics()
    test_physics(physics)
    
    # Test 3: Environment
    env = TrainEnv(physics, grades, limits, curves)
    test_environment_step(env)
    
    # Test 4: Quick episodes
    print("\n" + "="*70)
    response = input("\nRun 2 test episodes with full debug? (y/n): ")
    if response.lower() == 'y':
        test_quick_episodes()
    else:
        print("Skipping episode test.")
    
    print("\n" + "="*70)
    print("ALL TESTS COMPLETE!")
    print("="*70)
    print("\n✓ If all tests passed, you can now run: python main.py")
    print("✓ For your first real run, try: episodes=10 to see if it works")
    print("✓ Then increase to episodes=100, 500, 2000 as needed")
    print("\n" + "🚂"*35 + "\n")

if __name__ == "__main__":
    main()