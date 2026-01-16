"""
DIAGNOSTIC: Find why agent is stuck at ~57% of route
Run this to identify the problem segment
"""

import numpy as np
import env_settings.config as config
from env_settings.environment import TrainEnv
from data.utils import load_data

def diagnose_route():
    """Analyze route to find physical barriers"""
    
    print("="*70)
    print("ROUTE DIAGNOSTIC")
    print("="*70)
    
    # Load environment
    env = TrainEnv()
    
    n_segments = env.n_segments
    total_distance = n_segments * config.DX / 1000  # km
    
    print(f"\nRoute: {n_segments} segments, {total_distance:.1f} km")
    print(f"57% mark: segment {int(n_segments * 0.57)} ({total_distance * 0.57:.1f} km)")
    
    # Analyze grades and speed limits
    grades = env.grades
    speed_limits = env.speed_limits
    
    print(f"\n{'='*70}")
    print("GRADE ANALYSIS (steeper = harder)")
    print("="*70)
    
    # Find steep grades
    steep_threshold = 3.0  # percent
    steep_segments = []
    for i, g in enumerate(grades):
        if abs(g) > steep_threshold:
            steep_segments.append((i, g, i/n_segments*100))
    
    print(f"\nSegments with grade > {steep_threshold}%:")
    if steep_segments:
        for seg, grade, pct in steep_segments[:20]:
            direction = "UPHILL ↑" if grade > 0 else "DOWNHILL ↓"
            print(f"  Segment {seg} ({pct:.1f}%): {grade:+.2f}% {direction}")
        if len(steep_segments) > 20:
            print(f"  ... and {len(steep_segments) - 20} more")
    else:
        print("  None found")
    
    # Check around 57% mark
    problem_start = int(n_segments * 0.50)
    problem_end = int(n_segments * 0.65)
    
    print(f"\n{'='*70}")
    print(f"DETAILED ANALYSIS: Segments {problem_start}-{problem_end} (50%-65%)")
    print("="*70)
    
    print(f"\n{'Seg':>5} | {'%':>5} | {'Grade':>7} | {'Limit':>7} | Notes")
    print("-" * 50)
    
    for i in range(problem_start, min(problem_end, n_segments)):
        pct = i / n_segments * 100
        grade = grades[i] if i < len(grades) else 0
        limit = speed_limits[i] if i < len(speed_limits) else 22
        
        notes = []
        if grade > 2.0:
            notes.append("STEEP UPHILL")
        elif grade < -2.0:
            notes.append("STEEP DOWN")
        if limit < 10:
            notes.append("LOW SPEED LIMIT")
        
        note_str = ", ".join(notes) if notes else ""
        print(f"{i:>5} | {pct:>5.1f} | {grade:>+7.2f} | {limit:>7.1f} | {note_str}")
    
    # Test if route is completable with simple policy
    print(f"\n{'='*70}")
    print("TESTING: Can route be completed with maximum power?")
    print("="*70)
    
    env.reset()
    max_segment_reached = 0
    steps = 0
    max_steps = 50000
    
    # Try constant power action
    while steps < max_steps:
        # Action 3 = Power (maximum acceleration)
        _, _, done, info = env.step(3)
        steps += 1
        
        if env.seg_idx > max_segment_reached:
            max_segment_reached = env.seg_idx
            if max_segment_reached % 100 == 0:
                pct = max_segment_reached / n_segments * 100
                print(f"  Reached segment {max_segment_reached} ({pct:.1f}%) at step {steps}")
        
        if done:
            if env.seg_idx >= n_segments - 1:
                print(f"\n✓ SUCCESS! Route completed at step {steps}")
            else:
                pct = env.seg_idx / n_segments * 100
                print(f"\n✗ FAILED at segment {env.seg_idx} ({pct:.1f}%)")
                print(f"  Velocity: {env.v:.2f} m/s")
                print(f"  Grade at failure: {grades[env.seg_idx] if env.seg_idx < len(grades) else 'N/A'}")
                print(f"  Speed limit: {speed_limits[env.seg_idx] if env.seg_idx < len(speed_limits) else 'N/A'}")
            break
    
    if steps >= max_steps:
        pct = max_segment_reached / n_segments * 100
        print(f"\n⚠️  Timeout after {max_steps} steps")
        print(f"  Best progress: segment {max_segment_reached} ({pct:.1f}%)")
    
    # Test with smarter policy
    print(f"\n{'='*70}")
    print("TESTING: Smart policy (power on uphills, coast on flats)")
    print("="*70)
    
    env.reset()
    max_segment_reached = 0
    steps = 0
    
    while steps < max_steps:
        # Get current grade
        current_grade = grades[env.seg_idx] if env.seg_idx < len(grades) else 0
        current_v = env.v
        current_limit = speed_limits[env.seg_idx] if env.seg_idx < len(speed_limits) else 22
        
        # Smart policy
        if current_v < 5:
            action = 3  # Power if too slow
        elif current_v > current_limit - 1:
            action = 2  # Cruise if near limit
        elif current_grade > 1.0:
            action = 3  # Power on uphill
        elif current_grade < -1.0:
            action = 1  # Coast on downhill
        else:
            action = 2  # Cruise on flat
        
        _, _, done, info = env.step(action)
        steps += 1
        
        if env.seg_idx > max_segment_reached:
            max_segment_reached = env.seg_idx
            if max_segment_reached % 100 == 0:
                pct = max_segment_reached / n_segments * 100
                print(f"  Reached segment {max_segment_reached} ({pct:.1f}%) - v={env.v:.1f}m/s")
        
        if done:
            if env.seg_idx >= n_segments - 1:
                print(f"\n✓ SUCCESS! Route completed at step {steps}")
            else:
                pct = env.seg_idx / n_segments * 100
                print(f"\n✗ FAILED at segment {env.seg_idx} ({pct:.1f}%)")
                print(f"  Velocity: {env.v:.2f} m/s")
                print(f"  Grade: {grades[env.seg_idx] if env.seg_idx < len(grades) else 'N/A'}%")
                print(f"  Speed limit: {speed_limits[env.seg_idx] if env.seg_idx < len(speed_limits) else 'N/A'} m/s")
            break
    
    if steps >= max_steps:
        pct = max_segment_reached / n_segments * 100
        print(f"\n⚠️  Timeout after {max_steps} steps")
        print(f"  Best progress: segment {max_segment_reached} ({pct:.1f}%)")
    
    # Summary
    print(f"\n{'='*70}")
    print("DIAGNOSIS SUMMARY")
    print("="*70)
    
    final_pct = max_segment_reached / n_segments * 100
    
    if final_pct < 60:
        print(f"\n⚠️  Route completion is BLOCKED at ~{final_pct:.0f}%")
        print("\nPossible causes:")
        print("  1. Steep uphill grade that train can't climb")
        print("  2. Speed drops to 0 and can't recover")
        print("  3. Done condition triggered incorrectly")
        print("\nCheck environment.py for:")
        print("  - How 'done' is determined")
        print("  - Minimum velocity threshold")
        print("  - Physics calculations")
    elif final_pct < 100:
        print(f"\n⚠️  Route partially completable ({final_pct:.0f}%)")
        print("  Some issue prevents full completion")
    else:
        print(f"\n✓ Route IS completable!")
        print("  Problem is in learning, not physics")

if __name__ == "__main__":
    diagnose_route()