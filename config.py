import os
import torch

# --- System Settings ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
OUTPUT_DIR = "results_cm"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# --- File Paths ---
COORD_PATH = "coordinates.dat"
DATA_PATH = "data.csv"

# --- Physics Constants ---
MASS_KG = 360000.0        # 360 Tons
POWER_WATTS = 3640000.0   # 3640 kW
EFFICIENCY = 0.85
MAX_ACC = 0.6             # m/s^2 (approx 2.16 km/h/s)
MAX_DEC = -0.8            # m/s^2 (approx 2.88 km/h/s)
MAX_SPEED_MS = 130.0 / 3.6

# --- Davis Coefficients (Uncalibrated base) ---
# C0 will be calibrated dynamically
C1 = 420.0
C2 = 38.0

# --- RL Settings ---
DX = 100.0                # Segment length (meters)
DT = 1.0                  # Time step (seconds)
VEL_BIN_SIZE = 0.5        # Discretization for Q-Table (m/s)