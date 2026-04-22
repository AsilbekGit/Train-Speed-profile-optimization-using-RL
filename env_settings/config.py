import os

try:
    import torch
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
except ImportError:
    DEVICE = "cpu"
OUTPUT_DIR = "results_cm"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# --- File Paths ---
COORD_PATH = "data/coordinates.dat"
DATA_PATH = "data/data.csv"

# --- Physics Constants ---
MASS_KG = 360000.0        # 360 Tons (ER9E 6-car consist)
MASS_TONS = MASS_KG / 1000.0  # 360 tons
POWER_WATTS = 3640000.0   # 3640 kW
EFFICIENCY = 0.85
MAX_ACC = 0.6             # m/s^2 (approx 2.16 km/h/s)
MAX_DEC = -0.8            # m/s^2 (approx 2.88 km/h/s)
MAX_SPEED_MS = 130.0 / 3.6
GRAVITY = 9.81            # m/s^2

# --- Davis Coefficients (Uzbekistan ER9E) ---
# Davis equation: r_t = C0 + C1*V + C2*V^2
#   V in km/h
#   r_t in kg*s/ton (specific resistance per ton of train mass)
#
# To get total force in Newtons:
#   F_davis(N) = r_t * MASS_TONS * GRAVITY
#              = (C0 + C1*V + C2*V^2) * 360 * 9.81
C0 = 1.1                  # kg*s/ton
C1 = 0.01                 # kg*s/ton per (km/h)
C2 = 0.000227             # kg*s/ton per (km/h)^2

# --- RL Settings ---
DX = 100.0                # Segment length (meters)
DT = 1.0                  # Time step (seconds)
VEL_BIN_SIZE = 0.5        # Discretization for Q-Table (m/s)

# --- Episode Control ---
MAX_STEPS_PER_EPISODE = 20000  # Maximum steps per episode to prevent infinite loops
STUCK_THRESHOLD = 100          # Number of consecutive steps with no movement to detect stuck
POSITION_EPSILON = 0.01        # Minimum position change (meters) to consider as movement

# --- Convergence Measurement (CM Analysis) ---
# Paper Methodology (Section 3.4, Figure 5):
# 1. Run SARSA for 25,000 scenarios
# 2. Calculate ln(ΔQi/ΔQi-1) for each episode
# 3. Plot ln(CM) vs iteration (Figure 5)
# 4. Find where ln(CM) goes below threshold (indicates local optimum)
# 5. Paper found: ln(CM) → -3.21, which means φ = e^(-3.21) ≈ 0.04
#
# YOUR φ threshold will be different depending on your route data!
# The CM analysis will help you find it.

PHI_REFERENCE = 0.04           # Paper's φ for Tehran/Shiraz (reference only)
LN_PHI_REFERENCE = -3.21       # ln(0.04) = -3.21 (Figure 5 threshold line)

# CM Analysis settings
CM_ANALYSIS_EPISODES = 25000   # Paper used 25,000 scenarios

# --- Debug and Logging ---
DEBUG_MODE = False
PRINT_EVERY_STEP = False

# --- Action Names (for readable output) ---
ACTION_NAMES = ['Brake', 'Coast', 'Cruise', 'Power']