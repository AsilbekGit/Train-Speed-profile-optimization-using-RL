"""
Reverse-route loader.

Loads the same data/data.csv used by the forward trip and returns the data
re-oriented for the B → A traversal:

    reverse_grades[i]  = -1 * grades[N-1-i]    # reverse order AND flip sign
    reverse_limits[i]  =       limits[N-1-i]   # direction-agnostic
    reverse_curves[i]  =       curves[N-1-i]   # direction-agnostic

WHY GRADES FLIP SIGN
--------------------
The grade in data.csv is "what the train experiences as it traverses
segment i in the *forward* direction". A positive grade means the track
rises as you go from A toward B. When the train comes back, it traverses
the same physical segment in the *opposite* direction, so what was a rise
is now a descent — the sign flips.

WHY LIMITS AND CURVES DON'T
---------------------------
Speed limits are properties of a piece of track (often signed at fixed
mileposts). They apply regardless of which direction the train passes.
Same for curvature: a 1% curve is a 1% curve regardless of direction
(curve resistance uses |curvature| in the physics anyway).

WHY THE AGENT DOESN'T NEED A "DIRECTION" FLAG
---------------------------------------------
The RL agent observes [position, speed, grade, limit, ...]. It does NOT
observe "am I going forward or backward". By feeding it the reversed
route (with flipped grades), what it sees IS the return trip — a route
where formerly-uphills are now downhills. It will naturally learn to
coast more on those segments, because the energy formula in
env_settings/environment.py charges no energy when traction force is
zero, and on a downhill very little (or zero) traction is needed.
"""

import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from env_settings.gym_env import load_data as _load_forward_data


def load_reverse_route():
    """
    Returns (grades, limits, curves) for the B→A traversal.

    Order is reversed; grade signs are flipped; limits and curves are
    reversed without sign change.
    """
    f_grades, f_limits, f_curves = _load_forward_data()

    grades  = (-1.0 * f_grades[::-1]).copy()
    limits  =        f_limits[::-1].copy()
    curves  =        f_curves[::-1].copy()

    return grades, limits, curves


def print_summary(grades, limits, curves):
    """Convenience: print a small one-block summary of the route."""
    n = len(grades)
    print(f"   Reverse route: {n} segments, {n * 0.1:.1f} km")
    print(f"   Grade range (after sign-flip): "
          f"[{grades.min():.4f}%, {grades.max():.4f}%]")
    non_station = limits[limits > 1]
    if len(non_station):
        print(f"   Speed limit range: "
              f"[{non_station.min():.1f}, {non_station.max():.1f}] m/s")
    print(f"   Stations: {int(np.sum(limits == 1))}")
