"""
Configuration file for limb regeneration simulation
"""

import os
from datetime import datetime
import numpy as np

# Create date-based output directory

def get_next_run_number(base_dir, date):
    """
    Find next available run number for today's date by getting highest existing number
    and incrementing, rather than counting directories
    """
    if not os.path.exists(base_dir):
        os.makedirs(base_dir, exist_ok=True)
        return 1
        
    pattern = f'output_{date}_run'
    existing = [d for d in os.listdir(base_dir) if d.startswith(pattern)]
    
    if not existing:
        return 1
        
    # Extract run numbers from directory names
    run_numbers = []
    for dirname in existing:
        try:
            num = int(dirname.split('run')[-1])
            run_numbers.append(num)
        except ValueError:
            continue
            
    # Return highest number + 1, or 1 if no valid numbers found
    return max(run_numbers, default=0) + 1

# Set up paths - config is now at root level
BASE_DIR = os.path.dirname(os.path.abspath(__file__))  # Root directory where config.py lives (...python/agents)
INPUT_DIR = os.path.join(BASE_DIR, 'data', 'input')
DATA_DIR = os.path.join(BASE_DIR, 'data')

# File paths
CELL_INIT_FILE_BONE = os.path.join(INPUT_DIR, 'pos500_bone.csv')
CELL_INIT_FILE_NO_BONE = os.path.join(INPUT_DIR, 'pos500.csv')
BOUNDARY_FILE = os.path.join(INPUT_DIR, 'epi200.csv')
BONE_FILE = os.path.join(INPUT_DIR, 'bone.csv')

# Ensure input directory exists (but not output)
os.makedirs(INPUT_DIR, exist_ok=True)

# Function to create and return output directory path

def get_output_dir():
    date_str = datetime.now().strftime('%m_%d_%y')
    run_num = get_next_run_number(os.path.join(DATA_DIR, 'output'), date_str)
    output_dir = os.path.join(DATA_DIR, 'output', f'output_{date_str}_run{run_num}')
    os.makedirs(output_dir, exist_ok=True)
    return output_dir

OUTPUT_DIR = 'data/output/migration30_no_soft'
# OUTPUT_DIR = get_output_dir()
# Time parameters
DT = 5e-6 # orginal was 1e-5
TMAX = 7.0
STEPS_TOTAL = int(TMAX/DT) + 1

# Add Bone/Softening?
BONE_VISUALIZATION = True
BONE_FORCES = False
BONE_ENABLED = BONE_VISUALIZATION or BONE_FORCES
ALLOW_SOFTENING = False ###
SPORATIC_SOFTENING = False ### leave this false in almost all cases.

# Physical parameters
CONVERSION_FACTOR_UM = 200 # 1 simulation unit = 200 um (micrometers)
EXP_OUTGROWTH_LENGTH = 238.5 # in um  (CTRL) ### change to new 
EXP_AREA = 124973.7 # in um^2 (CTRL)
EXP_OUTGROWTH_LENGTH_C59 = 84.626 # in um (C59) ### change to new (~88 um?)
EXP_AREA_C59 = 34399.84 # in um^2 (C59)

DL_CRIT = 0.1 # diameter of cells, repulsion range(0.1 sim units = 20 um)
K_RM = 0.005 # Random Motion Constant
XI = 1.5

K_BC_REP = 80.0 # Cell-Boundary Repulsion Constant
K_BC_ADH = 0.01 # Cell-Boundary Adhesion Constant
K_CC_REP = 30.0 # Cell-Cell Repulsion Force Constant (mesenchyme)
K_CC_ADH = 0.005 * K_CC_REP # Cell-Cell Adhesion Force Constant (mesenchyme)
K_BONE = 100.0 # Cell-Bone Repulsion Force Constant (mesenchyme)
BONE_PHASE_PERCENT = 80.0 # Percentage chance cells can phase through bone (0-100)

KBEND = 5e-5 ### delete
KB_MAX, KB_MID, KB_MIN = 150.0, 75.0, 1.0 # parameters for sigmoid kb function 

# Cell Cycle parameters
T_DORMANT = 1.0
KDEATH = 0.01
OFFSET = 0.1

M_LENGTH = 28/24
G_LENGTH = 28/24
G_LENGTH_MAX = 36/24 # for gradient
G_LENGTH_MIN = 18/24 # for gradient
KDIV = 1 / (M_LENGTH + G_LENGTH) # used in param optimization as a parameter
# ZONE_START = 0.0 + 2*DL_CRIT
GRADIENT = None # 'zone', 'linear',  or None (linear soon to be removed! ###) change it to true false..

# Case parameters
MIGRATION_ENABLED = True # Enable/disable cell migration
REGULATION_FRONT_FLAG = True
MIGRATION_DELAY = 2.0 # wait until this time before migration begins
WHICH_MIGRATION = 'random' # 'random' or 'anterior_posterior'
MIGRATION_PERCENT = 30.0 # Percentage of cells that migrate (0-100), 30% creates the most accurate growth.
MIGRATION_DIRECTION = 'x' # 'x', 'y' direction for migration
MIGRATION_STRENGTH = 0.5  # Controls migration strength/speed
MIGRATION_STD_SCALE = 0.005 #1000.0

DIRECTED_DIVISION_ANGLE = None # Angle in radians for directed division (None for no directed division)

INTERCALATION_ENABLED = False # Enable/disable intercalation
INTERCAL_STRENGTH = 0.5
INTERCAL_STD_SCALE = 0.005
INTERCAL_DELAY = 2.0
INTERCAL_PERCENT = 0.0

JAMMING_ENABLED = False  # Enable/disable jamming

JAMMING_ZONE_WIDTH = 0.9 # rename this, this is the non-jammed zone width
K_CC_REP_JAMMING = K_CC_REP # Cell-Cell Repulsion Force Constant for jamming (same as non-jamming for now)
K_CC_ADH_JAMMING = K_CC_ADH # Cell-Cell Adhesion Force Constant for jamming (same as non-jamming for now)
K_RM_JAMMING = 0.0 # Random Motion Constant for jamming

# External Stress Test (poking)
EXT_STRESS_FORCE = False
EXT_FORCE_DELAY = 3.0
K_EXT = 1.0
FORCE_PER_UNIT_LENGTH = 30.0
# Domain bounds
XMIN, XMAX = -2.0, 3.0
YMIN, YMAX = -2.5, 2.5

# Simulation flags
VIDEO_FLAG = True
PROFILING_FLAG = False
FRAME_SKIP = 2000
PRINT_STEPS_FLAG = True # print steps to console
PRINT_STEPS_INTERVAL = int(1/DT)  # print steps to console every PRINT_STEPS_INTERVAL steps (1 day). for old printing, set to FRAME_SKIP
SAVE_DATA_DICT = True
SAVE_FIGURES = True

# Video parameters
FPS = 30
FIGSIZE = (6, 6)
DPI = 200
VIDEO_PARAMS = {
    'fps': FPS,
    'figsize': FIGSIZE,
    'dpi': DPI
}