"""
DIAGNOSTIC: Why is train stuck at ~57%?
========================================
This script analyzes the physics to find the blocking point.
"""

import numpy as np
from data.utils import load_data
from env_settings.physics import TrainPhysics
from env_settings.environment import TrainEnv
import env_settings.config as config

def analyze_physics():
    """Analyze why train can't complete route"""
    
    print("="*70)
    print("PHYSICS ANALYSIS: Why is train stuck at ~57%?")
    print("="*70)
    
    # Load data
    grades, limits, curves = load_data()
    physics = TrainPhysics()
    
    n_segments = len(grades)
    print(f"\nRoute: {n_segments} segments ({n_segments * config.DX / 1000:.1f} km)")
    print(f"Grade range: [{min(grades):.2f}%, {max(grades):.2f}%]")
    
    # Find the 57% point
    stuck_segment = int(n_segments * 0.57)
    print(f"\n57% mark = segment {stuck_segment}")
    
    # Analyze physics at different grades
    print(f"\n{'='*70}")
    print("FORCE ANALYSIS AT DIFFERENT GRADES")
    print("="*70)
    
    print(f"\nMax traction force (at low speed):")
    f_trac_low = physics.get_max_traction_force(1.0)
    f_trac_high = physics.get_max_traction_force(20.0)
    print(f"  At v=1 m/s:  {f_trac_low:,.0f} N")
    print(f"  At v=20 m/s: {f_trac_high:,.0f} N")
    
    print(f"\nResistance at different grades (v=5 m/s):")
    for grade in [0, 2, 4, 5, 6, 6.28]:
        r = physics.get_total_resistance(5.0, grade, 0)
        f_net = f_trac_low - r
        can_climb = "✓ CAN climb" if f_net > 0 else "✗ CANNOT climb"
        print(f"  Grade {grade:5.2f}%: Resistance = {r:>10,.0f} N | Net force = {f_net:>+10,.0f} N | {can_climb}")
    
    # Find maximum climbable grade
    print(f"\n{'='*70}")
    print("MAXIMUM CLIMBABLE GRADE")
    print("="*70)
    
    max_grade = 0
    for grade in np.arange(0, 10, 0.1):
        r = physics.get_total_resistance(1.0, grade, 0)
        if f_trac_low > r:
            max_grade = grade
        else:
            break
    
    print(f"\n  Train can climb up to: {max_grade:.1f}%")
    print(f"  Route has grades up to: {max(grades):.2f}%")
    
    if max(grades) > max_grade:
        print(f"\n  ⚠️  PROBLEM: Route has grades steeper than train can climb!")
        print(f"      Difference: {max(grades) - max_grade:.2f}%")
    
    # Find problem segments
    print(f"\n{'='*70}")
    print("PROBLEM SEGMENTS (grade > max climbable)")
    print("="*70)
    
    problem_segments = []
    for i, g in enumerate(grades):
        if g > max_grade:
            pct = i / n_segments * 100
            problem_segments.append((i, g, pct))
    
    if problem_segments:
        print(f"\nFound {len(problem_segments)} segments with unclimbable grades:")
        for seg, grade, pct in problem_segments[:20]:
            print(f"  Segment {seg:4d} ({pct:5.1f}%): grade = {grade:+.2f}%")
        if len(problem_segments) > 20:
            print(f"  ... and {len(problem_segments) - 20} more")
        
        # First problem segment
        first_problem = problem_segments[0]
        print(f"\n  FIRST BLOCKING POINT: Segment {first_problem[0]} ({first_problem[2]:.1f}%)")
    else:
        print(f"\n  No unclimbable segments found - problem is elsewhere")
    
    # Test actual completion
    print(f"\n{'='*70}")
    print("SIMULATION TEST: Try to complete route")
    print("="*70)
    
    env = TrainEnv(physics, grades, limits, curves)
    env.reset()
    
    max_segment = 0
    velocity_at_stuck = 0
    grade_at_stuck = 0
    
    for step in range(50000):
        # Use full power
        _, _, done, info = env.step(3)
        
        if env.seg_idx > max_segment:
            max_segment = env.seg_idx
            velocity_at_stuck = env.v
            if max_segment < len(grades):
                grade_at_stuck = grades[max_segment]
        
        if done or env.v < 0.1:
            break
    
    print(f"\n  Max segment reached: {max_segment} ({max_segment/n_segments*100:.1f}%)")
    print(f"  Velocity when stuck: {velocity_at_stuck:.2f} m/s")
    print(f"  Grade at stuck point: {grade_at_stuck:.2f}%")
    
    # Summary
    print(f"\n{'='*70}")
    print("DIAGNOSIS SUMMARY")
    print("="*70)
    
    print(f"""
The Problem:
  - Train max traction (at low speed): {f_trac_low:,.0f} N
  - Train can climb grades up to: {max_grade:.1f}%
  - Route has grades up to: {max(grades):.2f}%
  
  The train PHYSICALLY CANNOT climb the steep grades in your route!
  
The Fix:
  Replace env_settings/physics.py with the fixed version that:
  1. Uses adhesion-limited traction (~1,000,000 N) instead of comfort-limited (216,000 N)
  2. Allows train to produce more force at low speeds for hill climbing
  
After the fix:
  - Train will be able to climb grades up to ~25%
  - Route completion will be possible
  - Then RL learning can actually work
""")

if __name__ == "__main__":
    analyze_physics()