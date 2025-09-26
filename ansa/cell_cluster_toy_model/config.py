"""
Configuration file for limb regeneration simulation

TOY MODEL TOY MODEL TOY MODEL TOY MODEL TOY MODEL TOY MODEL TOY MODEL TOY MODEL TOY MODEL TOY MODEL TOY MODEL TOY MODEL TOY MODEL TOY MODEL TOY MODEL TOY MODEL TOY MODEL 
TOY MODEL TOY MODEL TOY MODEL TOY MODEL TOY MODEL TOY MODEL TOY MODEL TOY MODEL TOY MODEL TOY MODEL TOY MODEL TOY MODEL TOY MODEL TOY MODEL TOY MODEL TOY MODEL TOY MODEL 

"""

import os
from datetime import datetime

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

# Create date-based output directory with run number
date_str = datetime.now().strftime('%m_%d_%y')
data_dir = os.path.join(BASE_DIR, 'data')
run_num = get_next_run_number(data_dir, date_str)
OUTPUT_DIR = os.path.join(data_dir,'output', f'output_{date_str}_run{run_num}')

# File paths
CELL_INIT_FILE = os.path.join(INPUT_DIR, 'cellinitialization_500.mat')
BOUNDARY_FILE = os.path.join(INPUT_DIR, 'boundary0.csv')

# Ensure directories exist
os.makedirs(INPUT_DIR, exist_ok=True)
# os.makedirs(OUTPUT_DIR, exist_ok=True)

# Time parameters
DT = 1e-5
TMAX = 5.0# 5.0 for full sim
STEPS_TOTAL = int(TMAX/DT) + 1

# Physical parameters
DL_CRIT = 0.1 # diameter of cells, repulsion range
XI = 1.5
KB = 50.0#10.0
KBEND = 1e-2#2e-3
KCOLL = 0.75
KPROX = 1000
XPROX = 0.0
K_INTERCAL = 1.4#0.5 #5.0
INTERCAL_DELAY = 0.0 # Time Until Intercalation Activates
K_BC = 150.0 # Cell-Boundary Spring Constant
K_CC_REP = 500.0#40.0#4.0#2.0 # Cell-Cell Repulsion Force Constant
K_CC_ATTR = 0.005 * 40.0 # Cell-Cell Attraction Force Constant
# K_BONE = 1.0 # Cell-Bone Repulsion Force Constant

SOFT_RANGE = (-0.5, 0.5)
SOFT_FACTOR = 50 # 100

# Cell Cycle parameters
T_DORMANT = 0.0 # ---> 1.0 for full sim
K_MIGRATE = 0.397#0.2
KDEATH = 0.001#0.05 # ---> 0.05 for full sim
KDIV = 0.7#1.45 #1.45 #----> 0.4 for full sim
OFFSET = DL_CRIT

# Domain bounds
XMIN, XMAX = -3, 3
YMIN, YMAX = -3, 3

# Simulation flags
VIDEO_FLAG = False#True
PROFILING_FLAG = True
FRAME_SKIP = 1000

# Video parameters
FPS = 30
FIGSIZE = (6, 6)
DPI = 200
VIDEO_PARAMS = {
    'fps': FPS,
    'figsize': FIGSIZE,
    'dpi': DPI
}