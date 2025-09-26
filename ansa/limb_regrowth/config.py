"""
Configuration file for limb regeneration simulation

LIMB REGEN!
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
CELL_INIT_FILE_BONE = os.path.join(INPUT_DIR, 'cellinitialization_500.mat') ###  'cellinitialization_500.mat' 'cellinitialization_1500_bone.mat'
CELL_INIT_FILE_NO_BONE = os.path.join(INPUT_DIR, 'cellinitialization_500.mat')
BOUNDARY_FILE = os.path.join(INPUT_DIR, 'epi200.csv') # 200!!!!

# Ensure input directory exists (but not output)
os.makedirs(INPUT_DIR, exist_ok=True)

# Function to create and return output directory path

def get_output_dir():
    date_str = datetime.now().strftime('%m_%d_%y')
    run_num = get_next_run_number(os.path.join(DATA_DIR, 'output'), date_str)
    output_dir = os.path.join(DATA_DIR, 'output', f'output_{date_str}_run{run_num}')
    os.makedirs(output_dir, exist_ok=True)
    return output_dir

# Time parameters
DT = 5e-6 # 1e-5
TMAX = 7.0 
STEPS_TOTAL = int(TMAX/DT) + 1

# Add Bone/Softening?
BONE_ENABLED = False
ALLOW_SOFTENING = True

# Physical parameters
DL_CRIT = 0.1 #0.1 # diameter of cells, repulsion range
K_RM = 1000.0 #* 40 # Random Motion Constant for jamming  1000.0, *40 for one of the jamming cases 
XI = 1.5
# KCOLL = 0.75
# KPROX = 1000
# XPROX = 0.0
K_INTERCAL = 0.4
# INTERCAL_DELAY = 0.0 # Time Until Intercalation Activates
K_BC = 10.0 #10.0#5.0 # Cell-Boundary Spring Constant  
K_CC_REP = 40.0  # Cell-Cell Repulsion Force Constant
K_CC_ADH = 0.01 * K_CC_REP # Cell-Cell Adhesion Force Constant
K_BONE = 100.0 # Cell-Bone Repulsion Force Constant
BONE_PHASE_PERCENT = 80.0 # Percentage chance cells can phase through bone (0-100)

KBEND = 0.002#5e-4
KB_MAX, KB_MID, KB_MIN = 150.0, 75.0, 1.0 #100, 0.1 # parameters for sigmoid kb function 

# Cell Cycle parameters
T_DORMANT = 1.0
KDEATH = 0.05
OFFSET = 0.1

M_LENGTH = 17/24 * (1/DT) # division phase length: 17 hours 
G_LENGTH_MAX = 70/24 * (1/DT) # 50/24  in steps
G_LENGTH_MIN = 35/24 * (1/DT) # 35/24  in steps
G_LENGTH = (G_LENGTH_MAX + G_LENGTH_MIN) / 2
GRADIENT = None # 'linear', 'zone' or None
ZONE_START = 1.0 + 2*DL_CRIT
# G_LENGTH = 50/24 * (1/DT)  G0/G1 phases length: 35 hours (now 50)

# Case parameters
MIGRATION_ENABLED = False # Enable/disable cell migration
WHICH_MIGRATION = 'random' # 'random' or 'anterior_posterior'
MIGRATION_PERCENT = 6.0 # Percentage of cells that migrate (0-100)
MIGRATION_DIRECTION = 'x' # 'x', 'y', 'center' direction for migration
MIGRATION_STRENGTH = 0.5  # Controls migration strength/speed
MIGRATION_STD_SCALE = 1000.0

DIRECTED_DIVISION_ANGLE = None # Angle in radians for directed division (None for no directed division)

INTERCALATION_ENABLED = False # Enable/disable intercalation
N_INTERCAL_PAIRS = 125 # Number of intercalation pairs to create

JAMMING_ENABLED = False  # Enable/disable jamming
# JAMMING_ZONE_X = (0.5, 1.0)
# JAMMING_ZONE_Y = (-0.5, 0.5)
JAMMING_ZONE_WIDTH = 0.5
K_CC_REP_JAMMING = 300.0 # Cell-Cell Repulsion Force Constant for jamming
K_CC_ADH_JAMMING = 0.05 * K_CC_REP_JAMMING # Cell-Cell Adhesion Force Constant for jamming
K_RM_JAMMING = 0.0 # Random Motion Constant for jamming

# Domain bounds
XMIN, XMAX = -1.0, 3.0
YMIN, YMAX = -2.5, 2.5

# Simulation flags
VIDEO_FLAG = False
PROFILING_FLAG = True
FRAME_SKIP = 2000 # 1000

# Video parameters
FPS = 30
FIGSIZE = (6, 6)
DPI = 200
VIDEO_PARAMS = {
    'fps': FPS,
    'figsize': FIGSIZE,
    'dpi': DPI
}