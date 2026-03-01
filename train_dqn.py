"""
Train Deep Q-Network (Parallelized)
====================================
Optimized for multi-core systems (DGX Spark).
Uses all available CPU cores for episode collection + vectorized training.

Usage:
    python train_dqn.py
    python train_dqn.py --phi 0.10 --episodes 5000
    python train_dqn.py --phi 0.10 --episodes 5000 --workers 20
"""

import os
import sys

# CRITICAL: Set NumPy/BLAS threading BEFORE importing numpy
# This enables multi-core matrix operations
n_cores = str(os.cpu_count() or 20)
os.environ['OMP_NUM_THREADS'] = n_cores
os.environ['MKL_NUM_THREADS'] = n_cores
os.environ['OPENBLAS_NUM_THREADS'] = n_cores
os.environ['NUMEXPR_NUM_THREADS'] = n_cores
os.environ['VECLIB_MAXIMUM_THREADS'] = n_cores

import numpy as np
import pandas as pd

# Add parent directory to path (in case running from qsarsa_dqn/ subfolder)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def load_data():
    """
    Load route data from data/coordinates.dat and data/data.csv
    
    data.csv columns: Grade (%), Speed_limit (m/s), Curvature (%)
    coordinates.dat: node_id  x  y (whitespace separated)
    
    Speed_limit = 1 means station stop.
    """
    # Try multiple possible paths for data.csv
    search_paths = [
        'data/data.csv',
        '../data/data.csv',
        '../../data/data.csv',
        os.path.join(os.path.dirname(__file__), 'data', 'data.csv'),
        os.path.join(os.path.dirname(__file__), '..', 'data', 'data.csv'),
    ]
    
    csv_path = None
    for p in search_paths:
        if os.path.exists(p):
            csv_path = p
            break
    
    # Also try config paths
    if csv_path is None:
        for mod_name in ['config', 'env_settings.config']:
            try:
                cfg = __import__(mod_name, fromlist=['DATA_PATH'])
                if hasattr(cfg, 'DATA_PATH') and os.path.exists(cfg.DATA_PATH):
                    csv_path = cfg.DATA_PATH
                    break
            except ImportError:
                continue
    
    if csv_path is None:
        print("ERROR: Cannot find data/data.csv!")
        print("Searched in:", search_paths)
        print("Current directory:", os.getcwd())
        if os.path.exists('data'):
            print("Files in data/:", os.listdir('data'))
        sys.exit(1)
    
    print(f"   Loading track data from: {csv_path}")
    df = pd.read_csv(csv_path)
    print(f"   Columns found: {list(df.columns)}")
    print(f"   Segments: {len(df)}")
    
    # Handle different column naming (case-insensitive lookup)
    col_map = {}
    for c in df.columns:
        col_map[c.lower().strip()] = c
    
    # Grade (%)
    grade_key = next((k for k in col_map if 'grade' in k), None)
    if grade_key is None:
        print(f"ERROR: No 'Grade' column. Available: {list(df.columns)}")
        sys.exit(1)
    grades = df[col_map[grade_key]].values.astype(float)
    
    # Speed limit (m/s, where 1 = station)
    limit_key = next((k for k in col_map if 'speed' in k and 'limit' in k), None)
    if limit_key is None:
        limit_key = next((k for k in col_map if 'speed' in k), None)
    if limit_key is None:
        print(f"ERROR: No 'Speed_limit' column. Available: {list(df.columns)}")
        sys.exit(1)
    limits = df[col_map[limit_key]].values.astype(float)
    
    # Curvature (%)
    curve_key = next((k for k in col_map if 'curv' in k), None)
    if curve_key:
        curves = df[col_map[curve_key]].values.astype(float)
    else:
        curves = np.zeros(len(grades))
        print("   WARNING: No 'Curvature' column found, using zeros")
    
    # Load coordinates if available
    coord_search = [p.replace('data.csv', 'coordinates.dat') for p in search_paths]
    for p in coord_search:
        if os.path.exists(p):
            coords = np.loadtxt(p)
            print(f"   Loaded {len(coords)} coordinate nodes from: {p}")
            break
    
    # Print data summary
    non_station = limits[limits > 1]
    station_count = np.sum(limits == 1)
    print(f"\n   Route summary:")
    print(f"   - Segments: {len(grades)}")
    print(f"   - Total distance: {len(grades) * 0.1:.1f} km")
    print(f"   - Grade range: [{grades.min():.4f}%, {grades.max():.4f}%]")
    if len(non_station) > 0:
        print(f"   - Speed limit range: [{non_station.min():.1f}, {non_station.max():.1f}] m/s "
              f"([{non_station.min()*3.6:.1f}, {non_station.max()*3.6:.1f}] km/h)")
    print(f"   - Curvature range: [{curves.min():.6f}%, {curves.max():.6f}%]")
    print(f"   - Station stops (speed_limit=1): {station_count}")
    
    return grades, limits, curves


def get_physics_and_env(grades, limits, curves):
    """Try different import paths to get TrainPhysics and TrainEnv."""
    import_attempts = [
        ('env_settings.physics', 'env_settings.environment'),
        ('physics', 'environment'),
    ]
    
    for phys_mod, env_mod in import_attempts:
        try:
            phys_module = __import__(phys_mod, fromlist=['TrainPhysics'])
            env_module = __import__(env_mod, fromlist=['TrainEnv'])
            physics = phys_module.TrainPhysics()
            env = env_module.TrainEnv(physics, grades, limits, curves)
            print(f"   ✓ Loaded physics/environment from {phys_mod}")
            return physics, env
        except (ImportError, AttributeError):
            continue
    
    # Try config module
    for mod_name in ['config', 'env_settings.config']:
        try:
            cfg = __import__(mod_name, fromlist=['TrainPhysics', 'TrainEnv'])
            physics = cfg.TrainPhysics()
            env = cfg.TrainEnv(physics, grades, limits, curves)
            print(f"   ✓ Loaded from {mod_name}")
            return physics, env
        except (ImportError, AttributeError):
            continue
    
    print("ERROR: Cannot import TrainPhysics/TrainEnv")
    print("Make sure you're running from the project root directory")
    sys.exit(1)


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Train DQN for speed profile optimization')
    parser.add_argument('--phi', type=float, default=0.10, help='CM threshold φ (default: 0.10)')
    parser.add_argument('--episodes', type=int, default=5000, help='Training episodes (default: 5000)')
    parser.add_argument('--workers', type=int, default=None, help='Number of parallel workers (default: all cores)')
    args = parser.parse_args()
    
    print("=" * 70)
    print("DEEP Q-NETWORK TRAINING (PARALLELIZED)")
    print("Train Speed Profile Optimization")
    print(f"NumPy threads: {n_cores} | Workers: {args.workers or 'auto'}")
    print("=" * 70)
    
    # 1. Load route data
    print("\n1. Loading route data...")
    grades, limits, curves = load_data()
    
    # 2. Initialize physics and environment
    print("\n2. Initializing environment...")
    physics, env = get_physics_and_env(grades, limits, curves)
    
    # 3. Set φ
    print(f"\n3. Using φ = {args.phi}")
    
    # 4. Initialize DQN
    print("\n4. Initializing Deep Q-Network...")
    from qsarsa_dqn.dqn import DeepQNetwork
    dqn = DeepQNetwork(env, phi_threshold=args.phi, n_workers=args.workers)
    
    # Explicitly set route data for parallel workers
    dqn._env_args = (grades, limits, curves)
    
    # 5. Train
    print(f"\n5. Starting training ({args.episodes} episodes)...")
    dqn.train(episodes=args.episodes)
    
    # 6. Generate speed profile
    print("\n6. Generating optimal speed profile...")
    dqn.generate_speed_profile()
    
    print("\n" + "=" * 70)
    print("TRAINING COMPLETE!")
    print("=" * 70)


if __name__ == "__main__":
    main()