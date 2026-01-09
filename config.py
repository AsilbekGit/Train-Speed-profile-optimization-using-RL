import os
import torch

# --- System Settings ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
OUTPUT_DIR = "results_cm"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# --- File Paths ---
COORD_PATH = "data/coordinates.dat"
DATA_PATH = "data/data.csv"

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

# --- Episode Control ---
MAX_STEPS_PER_EPISODE = 20000  # Maximum steps per episode to prevent infinite loops
STUCK_THRESHOLD = 100          # Number of consecutive steps with no movement to detect stuck
                               # Increased from 50 to 100 - train might coast/brake temporarily
POSITION_EPSILON = 0.01        # Minimum position change (meters) to consider as movement

# --- Convergence Measurement (CM Analysis) ---
# NOTE: φ threshold is NOT fixed! It depends on your specific route data.
# The paper found φ = 0.04 for Tehran/Shiraz Metro, but YOUR φ will be different.
# 
# Purpose of this CM analysis: Find YOUR optimal φ by analyzing the plot
# 
# For plotting reference only (compare with paper's result):
PHI_REFERENCE = 0.04           # Paper's φ value (for comparison only)
                               
# You will determine YOUR actual φ threshold after seeing the CM plot:
# - Look for where CM stabilizes/converges in your plot
# - This will be YOUR φ value to use in Q-SARSA later

# --- Debug and Logging ---
DEBUG_MODE = False             # Set to True for detailed debugging info
PRINT_EVERY_STEP = False       # Set to True to print every single step (WARNING: very verbose!)
                               # Recommended: False for normal runs, True only for debugging first few episodes

# --- Action Names (for readable output) ---
ACTION_NAMES = ['Brake', 'Coast', 'Cruise', 'Power']

# --- Notes ---
# DEBUG_MODE: Shows detailed Q-value updates, position tracking, etc.
# PRINT_EVERY_STEP: Shows every single step in the episode (use with small episode counts)
# 
# For normal training runs:
#   DEBUG_MODE = False
#   PRINT_EVERY_STEP = False
#
# For debugging/understanding what's happening:
#   DEBUG_MODE = True
#   PRINT_EVERY_STEP = True
#   And run with episodes=1 or 2