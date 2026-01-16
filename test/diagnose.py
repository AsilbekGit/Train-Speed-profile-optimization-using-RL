"""
Diagnostic script to check why episodes aren't completing
"""

from data.utils import load_data
from env_settings.physics import TrainPhysics
from env_settings.environment import TrainEnv
import env_settings.config as config

def diagnose():
    print("="*70)
    print("DIAGNOSTIC CHECK")
    print("="*70)
    
    grades, limits, curves = load_data()
    physics = TrainPhysics()
    env = TrainEnv(physics, grades, limits, curves)
    
    # Test: Can train complete route at full power?
    print("\nTest 1: Full power throughout")
    env.reset()
    
    for step in range(config.MAX_STEPS_PER_EPISODE):
        _, reward, done, info = env.step(3)  # Action 3 = Full power
        
        if done:
            print(f"✓ Completed in {step} steps!")
            print(f"  Time: {info['time']:.1f}s")
            print(f"  Energy: {info['energy']:.2f}kWh")
            break
        
        if step % 1000 == 0:
            print(f"  Step {step}: seg={info['segment']}/{env.n_segments}, "
                  f"v={info['velocity']:.1f}m/s")
    
    if not done:
        print(f"✗ Did NOT complete! Reached segment {info['segment']}/{env.n_segments}")
        print(f"  Need to increase MAX_STEPS_PER_EPISODE")
    
    print("\n" + "="*70)
    
    # Test 2: Route statistics
    print("\nTest 2: Route difficulty analysis")
    print(f"Total segments: {len(grades)}")
    print(f"Total distance: {len(grades) * 0.1:.1f} km")
    print(f"Grade range: {min(grades):.2f}% to {max(grades):.2f}%")
    print(f"Steep uphills (>3%): {sum(1 for g in grades if g > 3)}")
    print(f"Steep downhills (<-3%): {sum(1 for g in grades if g < -3)}")
    print(f"Speed limit range: {min(limits):.1f} to {max(limits):.1f} m/s")
    
    # Calculate if route is feasible
    avg_speed_needed = (len(grades) * 0.1) / (config.MAX_STEPS_PER_EPISODE / 3600)
    print(f"\nRequired average speed: {avg_speed_needed:.1f} km/h")
    print(f"Is this feasible? {'✓ Yes' if avg_speed_needed < 60 else '✗ Too fast!'}")

if __name__ == "__main__":
    diagnose()