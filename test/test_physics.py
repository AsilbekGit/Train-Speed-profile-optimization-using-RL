"""
PHYSICS TEST - Run this FIRST to confirm the problem
=====================================================
This script tests if your train can complete the route.
Run: python test_physics.py
"""

import numpy as np
from data.utils import load_data
import env_settings.config as config

# Load data
print("Loading route data...")
grades, limits, curves = load_data()
n_segments = len(grades)

print(f"\nRoute: {n_segments} segments ({n_segments * config.DX / 1000:.1f} km)")
print(f"Grade range: [{min(grades):.2f}%, {max(grades):.2f}%]")

# Current physics parameters
mass = config.MASS_KG
max_acc = config.MAX_ACC
g = 9.81

print(f"\n{'='*60}")
print("CURRENT PHYSICS LIMITS")
print(f"{'='*60}")

# Current max traction
f_traction_current = mass * max_acc
print(f"Max traction force (current): {f_traction_current:,.0f} N")
print(f"  Formula: mass × MAX_ACC = {mass:,.0f} × {max_acc} = {f_traction_current:,.0f} N")

# What's needed for 6% grade
grade_max = max(grades)
f_grade = mass * g * (grade_max / 100)
f_davis = 39221 + 420 * 2 + 38 * 4  # At low speed (~2 m/s)
f_total_resistance = f_grade + f_davis

print(f"\nResistance on {grade_max:.2f}% grade:")
print(f"  Grade resistance: {f_grade:,.0f} N")
print(f"  Davis resistance: {f_davis:,.0f} N")
print(f"  Total: {f_total_resistance:,.0f} N")

net_force = f_traction_current - f_total_resistance
print(f"\nNet force = {f_traction_current:,.0f} - {f_total_resistance:,.0f} = {net_force:,.0f} N")

if net_force < 0:
    print(f"\n❌ PROBLEM: Net force is NEGATIVE!")
    print(f"   The train CANNOT climb a {grade_max:.2f}% grade!")
    print(f"   This is why you're stuck at ~50-60%")
else:
    print(f"\n✓ Train can climb {grade_max:.2f}% grade")

# What the fix provides
print(f"\n{'='*60}")
print("WITH PHYSICS FIX")
print(f"{'='*60}")

adhesion_coeff = 0.30
f_adhesion = adhesion_coeff * mass * g
print(f"Max adhesion force: {f_adhesion:,.0f} N")
print(f"  Formula: 0.30 × {mass:,.0f} × 9.81 = {f_adhesion:,.0f} N")

net_force_fixed = f_adhesion - f_total_resistance
print(f"\nNet force (fixed) = {f_adhesion:,.0f} - {f_total_resistance:,.0f} = {net_force_fixed:,.0f} N")

if net_force_fixed > 0:
    print(f"\n✓ With fix: Train CAN climb {grade_max:.2f}% grade!")
else:
    print(f"\n❌ Still cannot climb (unusual)")

# Now test actual simulation
print(f"\n{'='*60}")
print("SIMULATION TEST")
print(f"{'='*60}")

# Import physics and environment
from env_settings.physics import TrainPhysics
from env_settings.environment import TrainEnv

physics = TrainPhysics()
env = TrainEnv(physics, grades, limits, curves)

# Test with full power
print("\nTesting route completion with full power...")
env.reset()
max_segment = 0
stuck_count = 0
last_segment = 0

for step in range(50000):
    _, _, done, info = env.step(3)  # Full power
    
    if env.seg_idx > max_segment:
        max_segment = env.seg_idx
        stuck_count = 0
    else:
        stuck_count += 1
    
    if stuck_count > 1000 or done:
        break
    
    last_segment = env.seg_idx

progress_pct = (max_segment / n_segments) * 100

print(f"\nResult: Reached segment {max_segment}/{n_segments} ({progress_pct:.1f}%)")

if max_segment >= n_segments - 1:
    print("✓ SUCCESS! Route is completable.")
    print("  You can run CM analysis now.")
else:
    print(f"❌ FAILED! Train stuck at {progress_pct:.1f}%")
    print(f"\n  The physics.py fix was NOT applied!")
    print(f"\n  TO FIX:")
    print(f"  1. Open env_settings/physics.py")
    print(f"  2. Find the get_max_traction_force() function")
    print(f"  3. Replace it with the version below:")
    print(f"""
    def get_max_traction_force(self, v):
        # Adhesion limit (realistic for hill climbing)
        adhesion_coeff = 0.30
        f_adhesion = adhesion_coeff * self.mass * 9.81  # ~1,059,480 N
        
        # Power limit
        if v < 0.5:
            f_power = f_adhesion
        else:
            f_power = (self.power * self.eta) / v
        
        # At low speeds, use adhesion limit for hill climbing
        if v < 10.0:
            return min(f_adhesion, f_power)
        else:
            # At higher speeds, also consider comfort
            f_comfort = self.mass * config.MAX_ACC
            return min(f_comfort, f_power, f_adhesion)
    """)

# Find the blocking segment
if max_segment < n_segments - 1:
    print(f"\n{'='*60}")
    print(f"BLOCKING POINT ANALYSIS")
    print(f"{'='*60}")
    
    print(f"\nSegments around the stuck point ({max_segment}):")
    start = max(0, max_segment - 5)
    end = min(n_segments, max_segment + 10)
    
    for i in range(start, end):
        grade = grades[i]
        limit = limits[i]
        marker = ">>> STUCK HERE" if i == max_segment else ""
        print(f"  Seg {i}: grade={grade:+.2f}%, limit={limit:.1f} m/s {marker}")