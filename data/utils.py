import pandas as pd
import numpy as np
import env_settings.config

def load_data():
    print(f"Loading data from {config.COORD_PATH} and {config.DATA_PATH}...")
    
    # 1. Load Coordinates
    # Handle the specific tab separation
    coords = pd.read_csv(config.COORD_PATH, sep=r'\t', header=None, engine='python')
    coords.columns = ['id', 'x', 'y']
    
    # 2. Load Segment Data
    data = pd.read_csv(config.DATA_PATH)
    
    # Cleaning 'Unnamed' columns if they exist
    if 'Unnamed: 0' in data.columns:
        data = data.drop(columns=['Unnamed: 0'])
    
    # Validation
    if len(coords) != 750:
        print(f"WARNING: Expected 750 coords, found {len(coords)}")
    if len(data) != 749:
        print(f"WARNING: Expected 749 data rows, found {len(data)}")

    # Extract clean numpy arrays
    # Grade is in percent in CSV -> keep as percent for physics calc
    grades = data['Grade'].values 
    
    # Speed limit: Check if km/h or m/s
    speed_limits = data['Speed limit'].values
    if np.max(speed_limits) > 60:
        print("Detected Speed Limits in km/h. Converting to m/s.")
        speed_limits = speed_limits / 3.6
        
    # Curvature is in percent
    curvatures = data['Curvature'].values
    
    return grades, speed_limits, curvatures

def discretize_state(state):
    """
    Maps continuous state [segment_index, velocity] to discrete indices.
    """
    s_idx = int(state[0])
    
    # Velocity binning
    v_idx = int(state[1] / config.VEL_BIN_SIZE)
    
    # Cap velocity index at 100 bins (0 to 50 m/s) to prevent out-of-bounds
    v_idx = min(v_idx, 99)
    v_idx = max(v_idx, 0)
    
    return s_idx, v_idx