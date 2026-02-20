"""Configuration file for limb regeneration simulation"""
import os
from datetime import datetime
from pathlib import Path
import numpy as np

# Repository root
REPO_ROOT = Path(__file__).parent.parent
SRC_ROOT = Path(__file__).parent
OUTPUT_DIR = f'{REPO_ROOT}/simulation1'
INPUT_DIR = f"{SRC_ROOT}/input"

# File paths
CELL_INIT_FILE = os.path.join(INPUT_DIR, 'sim_data', 'pos0.csv')
BONE_FILE = os.path.join(INPUT_DIR, 'sim_data', 'bone.csv')
BOUNDARY_FILE = os.path.join(INPUT_DIR, 'sim_data', 'epi0.csv')
CTRL_AVG_FILE = os.path.join(INPUT_DIR, 'exp_curves', 'averages','control_avg_xy.csv')
C59_AVG_FILE = os.path.join(INPUT_DIR, 'exp_curves', 'averages','c59_avg_xy.csv')


# Time parameters
DT = 5e-6
TMAX = 7.0
STEPS_TOTAL = int(TMAX/DT) + 1

# Add Bone/Softening
BONE_VISUALIZATION = True
ALLOW_SOFTENING = True
SPORATIC_SOFTENING = False

# Physical parameters
CONVERSION_FACTOR_UM = 200 # 1 simulation unit = 200 um
EXP_OUTGROWTH_LENGTH = 237.37 # in um  (CTRL) 
EXP_AREA = 79590.10 # in um^2 (CTRL)
EXP_AR = 0.929  # Aspect ratio

EXP_OUTGROWTH_LENGTH_C59 = 88.11 # in um (C59)
EXP_AREA_C59 = 21880.93 # in um^2 (C59)
EXP_AR_C59 = 0.481 # Aspect ratio


D0 = 0.1 # Dell diameter
SIGMA = 0.005 # Diffusion 
XI = 1.5

K_BC_REP = 80.0 # Cell-Boundary Spring Constant
K_BC_ADH = 0.01
K_CC_REP = 30.0 # Cell-cell Repulsion Force Constant (mesenchyme)
K_CC_ADH = 0.005 * K_CC_REP # Cell-Cell Adhesion Force Constant (mesenchyme)

KB_MAX, KB_MID, KB_MIN = 150.0, 75.0, 1.0 # Epithelial stiffness 
KBEND = 0.4 ###
VISCOSITY = 0.5 ###
K_LATERAL = 0.008 ###

# Cell Cycle parameters
T_DORMANT = 1.0
KDEATH = 0.01
OFFSET = 0.1
M_LENGTH = 28/24
G_LENGTH = 28/24
KDIV = 1 / (M_LENGTH + G_LENGTH)

GRADIENT = False
G_LENGTH_MAX = 36/24
G_LENGTH_MIN = 18/24

# Case parameters
MIGRATION_ENABLED = True
REGULATION_FRONT_FLAG = True
MIGRATION_DELAY = 2.0
WHICH_MIGRATION = 'random' # 'random' or 'anterior_posterior'
MIGRATION_FRACTION = 0.30 # Fraction of cells that migrate (0-1)
MIGRATION_DIRECTION = 'x' # 'x', 'y', 'center' direction for migration
MU_MIGRATION = 0.5 ### 0.6
SIGMA_MIGRATION = 0.005

DIRECTED_DIVISION_ANGLE = None # Angle in radians for directed division (None for no directed division)

INTERCALATION_ENABLED = False
MU_INTERCAL = 0.5
SIGMA_INTERCAL = 0.005
INTERCAL_DELAY = 2.0
INTERCAL_FRACTION = 0.0

JAMMING_ENABLED = False  # Enable/disable jamming
FLUID_LIKE_ZONE_WIDTH = 0.9 # Width of unjammed/fluid-like region
K_CC_REP_JAMMED = K_CC_REP # Cell-cell repulsion Force Constant for jammed cells
K_CC_ADH_JAMMED = K_CC_ADH # Cell-cell adhesion Force Constant for jammed cells
SIGMA_JAMMED = 0.0 # Random Motion Constant for Jammed Cells
SIGMA_UNJAMMED = 10 * SIGMA # Random Motion Constant for Unjammed Cells

# External Stress Test
EXT_STRESS_FORCE = False
EXT_FORCE_DELAY = 3.0
K_EXT = 1.0
FORCE_PER_UNIT_LENGTH = 30.0

# Domain bounds
XMIN, XMAX = -2.0, 2.0
YMIN, YMAX = -2.0, 2.0

# Simulation flags
VIDEO_FLAG = True
PROFILING_FLAG = False
FRAME_SKIP = 2000
PRINT_STEPS_FLAG = True
PRINT_STEPS_INTERVAL = int(1/DT)
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

def get_output_dir():
    date_str = datetime.now().strftime('%m_%d_%y')
    run_num = get_next_run_number(os.path.join(REPO_ROOT, 'output'), date_str)
    output_dir = os.path.join(REPO_ROOT, 'output', f'output_{date_str}_run{run_num}')
    os.makedirs(output_dir, exist_ok=True)
    return output_dir