"""
Limb Regeneration Simulation - Agents Based Model (2D) - Version 9
Ansa Brews-Smith, May 2025
Copos Lab, Northeastern University

"""
import warnings
warnings.filterwarnings('ignore')

import matplotlib
matplotlib.use('Agg')  # Set backend before importing pyplot
import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio 
import time
import os
# os.environ['NUMBA_THREADING_LAYER'] = 'tbb'  # Use TBB for better threading
from numba import njit, prange, set_num_threads, get_num_threads
from numpy import int64, float64
from scipy.sparse import spdiags
from scipy.special import expit

# Import configuration and utilities
from config import *
from utils.animations import AnimationManager
from utils.data_io import save_files
from utils.profiler import profiling
from utils.signal_handling import setup_signal_handler
from utils.post_process import cycle_plot2, runtime_plot, density_heatmap, trajectory_plot, phase_distribution_plot, boundary_plot, morphometrics, MSD_plot, save_first_last_frames
from config import get_output_dir
OUTPUT_DIR = get_output_dir()

# OUTPUT_DIR = 'data/output/C59'
# os.makedirs(OUTPUT_DIR, exist_ok=True)

set_num_threads(get_num_threads())
print(f"Using {get_num_threads()} threads")

# Load initial cell positions 
if BONE_ENABLED:
    matfile = sio.loadmat(CELL_INIT_FILE_BONE)
else:
    matfile = sio.loadmat(CELL_INIT_FILE_NO_BONE)
N0 = int(matfile['Ncells'][0,0])
pos0 = matfile['pos0']    # shape: (N,2)

# Pre‑allocate cell arrays 
CELLS_MAX = 6 * N0
pos = np.full((CELLS_MAX, 2), np.nan)
v = np.zeros((CELLS_MAX, 2))
pos[:N0, :] = pos0

# cell cycle set-up
division_status = np.zeros((CELLS_MAX,), dtype=bool)
cycle_phases = np.zeros((CELLS_MAX,), dtype=int) # 0 = G0/G1 (at rest), 1 = S/G2/M (division), start all at G0/G1
empty_slots = np.isnan(pos[:, 0])
cycle_phases[empty_slots] = -1 # empty slot
death_mask = np.zeros((CELLS_MAX,), dtype=bool)
phase_clocks = np.zeros((CELLS_MAX,), dtype=float)

# Alive mask & daughter count
n_daughter = 0

# Global variables for cases (and bone passage)
INTERCAL_PHASE_PERCENT = (N_INTERCAL_PAIRS * 2) / CELLS_MAX * 100.0 # Percentage of cells that can intercalate (0-100)
migrant_cells = np.zeros(CELLS_MAX, dtype=bool)
intercal_pairs = None
intercal_cells = np.zeros(CELLS_MAX, dtype=bool)
jammed_cells = np.zeros(CELLS_MAX, dtype=bool)
bone_passage_mask = np.zeros(CELLS_MAX, dtype=bool)
bone_passage_mask[np.random.choice(CELLS_MAX, size=int(CELLS_MAX*BONE_PHASE_PERCENT/100), replace=False)] = True ### set X% of cells to be able to phase through bone

def init_migration_cells(n_cells_max, migration_percent):
    """
    Initialize which cells are migrants at the start.
    
    Args:
        n_cells: Total number of cells
        migration_percent: Percentage of cells to make migrants (0-100)
        
    Returns:
        migrant_cells: Boolean array indicating migrant cells
    """
    n_migrants = int(N0 * migration_percent / 100.0)
    migrant_cells = np.zeros(n_cells_max, dtype=bool)
    if n_migrants > 0:
        if WHICH_MIGRATION == 'anterior_posterior':
            anterior_posterior = np.zeros(n_cells_max, dtype=bool)
            anterior_posterior[np.where((pos[:, 1] < -1.5 + 2*DL_CRIT) | (pos[:, 1] > 1.5 - 2*DL_CRIT))[0]] = True
            migrant_indices = np.random.choice(np.where(anterior_posterior)[0], size=n_migrants, replace=False)
            migrant_cells[migrant_indices] = True
            # print(f'migrant indices: {migrant_indices}')
        elif WHICH_MIGRATION == 'random':
            migrant_indices = np.random.choice(n_migrants, size=n_migrants, replace=False)
            
            migrant_cells[migrant_indices] = True
            # print(f"Migrants: {np.sum(migrant_cells)}")
        else:
            raise ValueError(f"Invalid migration type: {WHICH_MIGRATION}")
    return migrant_cells

def update_jammed_cells(Xe):
    global pos, jammed_cells
    # jamming_area = (pos[:, 0] > JAMMING_ZONE_X[0]) & (pos[:, 0] < JAMMING_ZONE_X[1]) & (pos[:, 1] > JAMMING_ZONE_Y[0]) & (pos[:, 1] < JAMMING_ZONE_Y[1])
    jammed_cells = np.zeros(CELLS_MAX, dtype=bool)
    x_max = Xe[:, 0].max()
    jamming_area = ((x_max - pos[:, 0]) < JAMMING_ZONE_WIDTH) & (pos[:, 1] > -0.7) & (pos[:, 1] < 0.7)
    jammed_cells[jamming_area] = True
    # return jammed_cells

def init_intercal_pairs(pos_active, n_pairs):
    """
    Create intercalation pairs between cells with y>0 and y<0.
    
    Args:
        pos_active: (n_cells, 2) array of positions
        n_pairs: Number of pairs to create
        
    Returns:
        intercal_pairs: list of (i,j) index pairs
    """
    n_cells = len(pos_active)
    indices = np.arange(n_cells)
    
    # Split into above and below y=0
    above = indices[pos_active[:, 1] > 0]
    below = indices[pos_active[:, 1] < 0]
    
    # Shuffle each group to randomize pairings
    np.random.shuffle(above)
    np.random.shuffle(below)
    
    # Create pairs (limited by smaller group and n_pairs)
    m = min(len(above), len(below), n_pairs)
    above = above[:m]
    below = below[:m]
    
    # Create pairs
    intercal_pairs = [(int(above[i]), int(below[i])) for i in range(m)]
    intercal_cells = np.zeros(n_cells, dtype=bool)
    intercal_cells[above] = True
    intercal_cells[below] = True
    return intercal_cells, intercal_pairs

def update_intercal_pairs_after_death(intercal_pairs, death_mask):
    """
    Remove pairs that contain dead cells and reindex remaining pairs.
    
    Args:
        intercal_pairs: Current list of pairs
        death_mask: Boolean array indicating dead cells
        
    Returns:
        updated_pairs: New list with dead cells removed
    """
    if intercal_pairs is None:
        return None
    
    updated_pairs = []
    for i, j in intercal_pairs:
        # Only keep pairs where both cells are alive
        if not death_mask[i] and not death_mask[j]:
            updated_pairs.append((i, j))
    
    return updated_pairs

def update_intercal_pairs_graduation(intercal_pairs, intercal_cells, pos, graduation_zone=0.3):
    """
    Remove intercalation status from cells that reach the y=0 line (within ±graduation_zone)
    and transfer their status to other eligible cells outside the graduation zone.
    
    Args:
        intercal_pairs: Current list of pairs
        intercal_cells: Boolean array indicating intercalation cells
        pos: Cell positions
        graduation_zone: Distance from y=0 where cells graduate from intercalation
        
    Returns:
        updated_pairs: New list with graduated cells replaced
        intercal_cells: Updated intercalation status
    """
    if intercal_pairs is None or len(intercal_pairs) == 0:
        return intercal_pairs, intercal_cells
    
    alive = ~np.isnan(pos[:, 0])
    graduated_cells = []
    
    # Find cells that have graduated (within ±graduation_zone of y=0)
    for i, j in intercal_pairs:
        if alive[i] and abs(pos[i, 1]) <= graduation_zone:
            graduated_cells.append(i)
        if alive[j] and abs(pos[j, 1]) <= graduation_zone:
            graduated_cells.append(j)
    
    if len(graduated_cells) == 0:
        return intercal_pairs, intercal_cells
    
    # Remove graduated cells from intercalation
    for cell_idx in graduated_cells:
        intercal_cells[cell_idx] = False
    
    # Find eligible cells for new intercalation (alive, not already intercalating, outside graduation zone)
    eligible_above = alive & (~intercal_cells) & (pos[:, 1] > graduation_zone)
    eligible_below = alive & (~intercal_cells) & (pos[:, 1] < -graduation_zone)
    
    above_indices = np.where(eligible_above)[0]
    below_indices = np.where(eligible_below)[0]
    
    # Shuffle to randomize selection
    np.random.shuffle(above_indices)
    np.random.shuffle(below_indices)
    
    # Create new pairs to replace graduated ones
    updated_pairs = []
    new_pairs_created = 0
    
    # First, keep pairs that don't have graduated cells
    for i, j in intercal_pairs:
        if i not in graduated_cells and j not in graduated_cells:
            updated_pairs.append((i, j))
    
    # Create new pairs for graduated cells
    n_new_pairs_needed = len(graduated_cells) // 2  # Each graduated cell was in a pair
    max_new_pairs = min(n_new_pairs_needed, len(above_indices), len(below_indices))
    
    for k in range(max_new_pairs):
        if k < len(above_indices) and k < len(below_indices):
            new_i = above_indices[k]
            new_j = below_indices[k]
            updated_pairs.append((new_i, new_j))
            intercal_cells[new_i] = True
            intercal_cells[new_j] = True
            new_pairs_created += 1
    
    # print(f"Intercalation update: {len(graduated_cells)} cells graduated, {new_pairs_created} new pairs created")
    
    return updated_pairs, intercal_cells

def map_intercal_pairs_to_local(intercal_pairs, active):
    """
    Map global indices in intercal_pairs to local indices for pos_active.
    
    Args:
        intercal_pairs: List of (i,j) global index pairs
        active: Array of global indices of active cells
        
    Returns:
        intercal_pairs_local: List of (i,j) local index pairs, or None
    """
    if intercal_pairs is None:
        return None
    
    global_to_local = {g: l for l, g in enumerate(active)}
    intercal_pairs_local = []
    for i, j in intercal_pairs:
        if i in global_to_local and j in global_to_local:
            intercal_pairs_local.append((global_to_local[i], global_to_local[j]))
    
    return intercal_pairs_local if intercal_pairs_local else None

#---------------------------------------------------------------------------------------------------
## CORE SIMULATION FUNCTIONS
#---------------------------------------------------------------------------------------------------

def build_boundaries(soft=True):
    """Construct the boundary shapes for the limb regeneration simulation (Xb is bone, Xe is epithlium, Xc is thick collagen region)"""

    ## EPITHELIUM ##
    Xe0 = np.loadtxt(BOUNDARY_FILE, delimiter=',') # epithelium (change to epi0.csv again) ###
    Xe = Xe0.copy()
    xe = Xe0[:, 0]
    ye = Xe0[:, 1]
    dsb = np.hypot(*(Xe0[1] - Xe0[0]))  # first segment length

    # Sparse forward‑difference matrix Db
    Ne = Xe.shape[0]
    e = np.ones(Ne)
    Db = spdiags([-e, e], [0,1], Ne, Ne, format='csr')
    Db[Ne-1, 0] = 1

    # rest lengths of boundary "springs"
    blp0 = np.hypot(*(Db @ Xe0).T)
    blm0 = np.hypot(*(Db.T @ Xe0).T)

    # Calculate soft boundary indices - keep sigmoid softness distribution
    center = Xe0[:, 0].max() - 0.05  # Center just before the tip  
    width = 0.01  # Extremely steep transition
    y_lo, y_hi = -1.0, 1.0              # vertical bounds for soft region

    # Sigmoid window in y
    sig_lo = expit((ye - y_lo) / 0.1)  # 0.05 controls sharpness of lower edge
    sig_hi = expit(-(ye - y_hi) / 0.1) # 0.05 controls sharpness of upper edge
    window = sig_lo * sig_hi            # 1 inside, 0 outside, smooth transition

    # Stiffness profile: soft at tip (large x), stiff at base (small x)

    k_b = lambda x: KB_MID + (KB_MIN - KB_MID) * expit((x - center) / width)
    if soft:
        kb_vals = KB_MID * (1 - window) + k_b(xe) * window

        # corner softening
        # kb_vals[np.where((0.75 < Xe[:, 1]) & (Xe[:, 1] < 1.25) | (-1.25 < Xe[:, 1]) & (Xe[:, 1] < -0.75))[0]] = KB_MIN ## softened corners!
        
    else:
        kb_vals = np.ones(Ne) * KB_MID
    kb_vals[np.where(Xe[:, 0] < 0)[0]] = KB_MAX
    # kb_vals = np.ones(Ne) * KB_MAX ####
    # kb_vals[np.where((ye >= -1) & (ye <= 1))[0]] = KB_MIN ####
    ## BONE ##
    if BONE_ENABLED:
        Xb = np.loadtxt('data/input/bone.csv', delimiter=',')
    else:
        Xb = None

    return Xe0, Xe, Ne, Db, blp0, blm0, dsb, kb_vals, Xb

# Initialize boundary
Xe0, Xe, Ne, Db, blp0, blm0, dsb, kb_vals, Xb  = build_boundaries(soft=ALLOW_SOFTENING)
left_wall = np.min(Xe0[:, 0])
x_cut = Xe0[:, 0].max() # This defines the cut-off for active epithelium interaction
active_epi_indices = np.where(Xe[:, 0] >= x_cut)[0]
n_active_epi = len(active_epi_indices)
# Create DbT for the epithelium_elasticity function
DbT = Db.T.tocsr()

@njit(parallel=True)
def hard_reset(pos_active, v_active, Xe, Xb, check_bone=False):
    """
    Reset cells that leave the growth region boundary or enter the bone region.
    
    Args:
        pos_active: (n_cells, 2) array of cell positions
        v_active: (n_cells, 2) array of cell velocities
        Xe: (n_boundary, 2) array of epithelium boundary points
        Xb: (n_bone, 2) array of bone boundary points (static)
        
    Returns:
        Updated pos_active and v_active
    """
    n_cells = len(pos_active)
    n_epithelium = len(Xe)
    n_bone = len(Xb)
    
    for i in prange(n_cells):
        x, y = pos_active[i]
        
        # Check if cell is inside epithelium boundary (allowed region)
        inside_epithelium = False
        j = n_epithelium - 1
        
        # Ray casting algorithm for epithelium
        for k in range(n_epithelium):
            xi, yi = Xe[k]
            xj, yj = Xe[j]
            
            if ((yi > y) != (yj > y)) and (x < (xj - xi) * (y - yi) / (yj - yi) + xi):
                inside_epithelium = not inside_epithelium
            j = k
        
        # Close the epithelium polygon properly
        xi, yi = Xe[0]
        xj, yj = Xe[-1]
        if ((yi > y) != (yj > y)) and (x < (xj - xi) * (y - yi) / (yj - yi) + xi):
            inside_epithelium = not inside_epithelium
        
        # Check if cell is inside bone region (forbidden region)
        inside_bone = False
        if BONE_ENABLED and check_bone:
            if n_bone > 0:  # Only check if bone boundary exists
                j = n_bone - 1
                
                # Ray casting algorithm for bone
                for k in range(n_bone):
                    xi, yi = Xb[k]
                    xj, yj = Xb[j]
                    
                    if ((yi > y) != (yj > y)) and (x < (xj - xi) * (y - yi) / (yj - yi) + xi):
                        inside_bone = not inside_bone
                    j = k
                
                # Close the bone polygon properly
                xi, yi = Xb[0]
                xj, yj = Xb[-1]
                if ((yi > y) != (yj > y)) and (x < (xj - xi) * (y - yi) / (yj - yi) + xi):
                    inside_bone = not inside_bone
        
        # Reset cell if it's outside epithelium OR inside bone
        needs_reset = not inside_epithelium or inside_bone
        
        if needs_reset:
            # Find the nearest valid position (on epithelium, away from bone)
            min_dist_sq = np.inf
            min_idx = 0
            reset_boundary = Xe  # Default to epithelium boundary
            
            # If inside bone, prioritize getting out of bone first

            if inside_bone and BONE_ENABLED:
                # Find nearest point on bone boundary to push away from
                for j in range(n_bone):
                    dx = Xb[j, 0] - x
                    dy = Xb[j, 1] - y
                    dist_sq = dx*dx + dy*dy
                    if dist_sq < min_dist_sq:
                        min_dist_sq = dist_sq
                        min_idx = j
                
                # Calculate outward normal from bone
                prev_idx = (min_idx - 1) % n_bone
                next_idx = (min_idx + 1) % n_bone
                
                # Tangent vector
                tx = Xb[next_idx, 0] - Xb[prev_idx, 0]
                ty = Xb[next_idx, 1] - Xb[prev_idx, 1]
                
                # Normal vector (rotated tangent 90 degrees CCW)
                nx = -ty
                ny = tx
                norm = np.sqrt(nx*nx + ny*ny)
                
                if norm > 0:
                    nx /= norm
                    ny /= norm
                    
                    # Make sure normal points outward from bone
                    test_x = Xb[min_idx, 0] + 0.01 * nx
                    test_y = Xb[min_idx, 1] + 0.01 * ny
                    
                    # Quick inside check for test point
                    test_inside_bone = False
                    j = n_bone - 1
                    for k in range(n_bone):
                        xi, yi = Xb[k]
                        xj, yj = Xb[j]
                        if ((yi > test_y) != (yj > test_y)) and (test_x < (xj - xi) * (test_y - yi) / (yj - yi) + xi):
                            test_inside_bone = not test_inside_bone
                        j = k
                    
                    # Flip normal if pointing inward to bone
                    if test_inside_bone:
                        nx = -nx
                        ny = -ny
                else:
                    # Fallback: direction away from bone center
                    bone_center_x = np.mean(Xb[:, 0])
                    bone_center_y = np.mean(Xb[:, 1])
                    nx = x - bone_center_x
                    ny = y - bone_center_y
                    norm = np.sqrt(nx*nx + ny*ny)
                    if norm > 0:
                        nx /= norm
                        ny /= norm
                    else:
                        nx = 1.0  # Default rightward
                        ny = 0.0
                
                # Reset position outside bone with larger offset
                jitter = 0.01 * (np.random.rand() - 0.5)
                pos_active[i, 0] = Xb[min_idx, 0] + 0.15 * nx + jitter  # Larger offset from bone
                pos_active[i, 1] = Xb[min_idx, 1] + 0.15 * ny + jitter
                
            else:
                # Outside epithelium - find nearest epithelium point
                # Optimized nearest point search (coarse + fine)
                step = max(1, n_epithelium // 50)
                for j in range(0, n_epithelium, step):
                    dx = Xe[j, 0] - x
                    dy = Xe[j, 1] - y
                    dist_sq = dx*dx + dy*dy
                    if dist_sq < min_dist_sq:
                        min_dist_sq = dist_sq
                        min_idx = j
                
                # Fine search (+-3 points around best candidate)
                start = max(0, min_idx - 3)
                end = min(n_epithelium, min_idx + 4)
                for j in range(start, end):
                    dx = Xe[j, 0] - x
                    dy = Xe[j, 1] - y
                    dist_sq = dx*dx + dy*dy
                    if dist_sq < min_dist_sq:
                        min_dist_sq = dist_sq
                        min_idx = j
                
                # Calculate inward normal at nearest epithelium point
                prev_idx = (min_idx - 1) % n_epithelium
                next_idx = (min_idx + 1) % n_epithelium
                
                # Tangent vector
                tx = Xe[next_idx, 0] - Xe[prev_idx, 0]
                ty = Xe[next_idx, 1] - Xe[prev_idx, 1]
                
                # Normal vector (rotated tangent 90 degrees CCW)
                nx = -ty
                ny = tx
                norm = np.sqrt(nx*nx + ny*ny)
                
                if norm > 0:
                    nx /= norm
                    ny /= norm
                    
                    # Verify normal direction points inward to epithelium
                    test_x = Xe[min_idx, 0] + 0.01 * nx
                    test_y = Xe[min_idx, 1] + 0.01 * ny
                    
                    # Quick inside check for test point
                    test_inside = False
                    j = n_epithelium - 1
                    for k in range(n_epithelium):
                        xi, yi = Xe[k]
                        xj, yj = Xe[j]
                        if ((yi > test_y) != (yj > test_y)) and (test_x < (xj - xi) * (test_y - yi) / (yj - yi) + xi):
                            test_inside = not test_inside
                        j = k
                    
                    # Flip normal if pointing outward from epithelium
                    if not test_inside:
                        nx = -nx
                        ny = -ny
                else:
                    # Fallback: direction toward epithelium center
                    epi_center_x = np.mean(Xe[:, 0])
                    epi_center_y = np.mean(Xe[:, 1])
                    nx = epi_center_x - Xe[min_idx, 0]
                    ny = epi_center_y - Xe[min_idx, 1]
                    norm = np.sqrt(nx*nx + ny*ny)
                    if norm > 0:
                        nx /= norm
                        ny /= norm
                    else:
                        nx = 1.0  # Default rightward
                        ny = 0.0
                
                # Reset position inside epithelium
                jitter = 0.01 * (np.random.rand() - 0.5)
                pos_active[i, 0] = Xe[min_idx, 0] + 0.07 * nx + jitter
                pos_active[i, 1] = Xe[min_idx, 1] + 0.07 * ny + jitter
            
            # Apply velocity damping for any reset
            v_active[i] *= 0.5  # Reduce velocity by 50%
    
    return pos_active, v_active

def cell_cycle(div_allowed=True, directed_angle=None, migrant_cells=None, gradient=None):
    global pos, cycle_phases, phase_clocks, death_mask, n_daughter, jammed_cells

    alive = ~np.isnan(pos[:,0])
    r_death = np.random.rand(cycle_phases.size)
    x_arr = pos[:, 0][alive]
    x_max = x_arr.max()
    x_min = x_arr.min()
    if gradient is None:
        G_lengths = np.zeros_like(cycle_phases)
        G_lengths[alive] = G_LENGTH
    elif gradient == 'linear':
        G_lengths_func = lambda x: G_LENGTH_MAX + (G_LENGTH_MIN - G_LENGTH_MAX) * (x - x_min) / (x_max - x_min)
        G_lengths = np.zeros_like(cycle_phases)
        G_lengths[alive] = G_lengths_func(x_arr)
    elif gradient == 'zone':
        G_lengths_func = lambda x: np.where((x < ZONE_START) | (x > x_max - 2 * DL_CRIT), G_LENGTH_MAX, G_LENGTH_MIN)
        G_lengths = np.zeros_like(cycle_phases)
        G_lengths[alive] = G_lengths_func(x_arr)
    # elif gradient == 'sigmoid':
    #     G_lengths_func = lambda x: G_LENGTH_MAX + (G_LENGTH_MIN - G_LENGTH_MAX) * (x - x_min) / (x_max - x_min)
    #     G_lengths = np.zeros_like(cycle_phases)
    #     G_lengths[alive] = G_lengths_func(x_arr)

    # cell death
    pD = KDEATH*DT
    die_mask = alive & (r_death < pD) # decide who dies
    pos[die_mask, :] = np.nan
    cycle_phases[die_mask] = -1      # or some sentinel
    death_mask[die_mask] = True
    jammed_cells[die_mask] = False
    if intercal_cells is not None:
        intercal_cells[die_mask] = False
    if migrant_cells is not None:
        migrant_cells[die_mask] = False  # Disable migration for dead cells

    if div_allowed:
        # masks for G0/G1 vs M
        r_div = np.random.rand(cycle_phases.size)
        maskG = alive & (cycle_phases == 0)
        maskM = alive & (cycle_phases == 1)

        # per‐cell transition probabilities (convert phase_clocks from time to steps)
        phase_steps = phase_clocks * DT
        pG = 1.0 / (G_lengths - phase_steps + 1) ###
        pM = 1.0 / (M_LENGTH - phase_steps + 1) ###

        
        # decide who enters M and who divides
        enterM = maskG & (r_div < pG)
        divideM = maskM & (r_div < pM)
        # G_tracker.append(np.nanmean(phase_steps[enterM])*DT)
        # M_tracker.append(np.nanmean(phase_steps[divideM])*DT)

        # update those phases/clocks
        cycle_phases[enterM]    = 1
        phase_clocks[enterM]    = 0.0

        cycle_phases[divideM]   = 0
        phase_clocks[divideM]   = 0.0

        # place daughters for each dividing mother
        mothers = np.where(divideM)[0]
        free_spots = np.where(np.isnan(pos[:,0]))[0]
        for mom, dau in zip(mothers, free_spots):
            if directed_angle is not None:
                # Use directed division with some noise
                ang = np.random.normal(loc=directed_angle, scale=np.pi/6)
            else:
                # Random division angle
                ang = 2*np.pi*np.random.rand()
            
            pos[dau, 0] = pos[mom, 0] + OFFSET * np.cos(ang)
            pos[dau, 1] = pos[mom, 1] + OFFSET * np.sin(ang)
            # initialize daughter
            division_status[dau] = True
            cycle_phases[dau]  = 0
            phase_clocks[dau]  = 0.0
            n_daughter += 1

            
            if MIGRATION_ENABLED and np.random.rand() < MIGRATION_PERCENT / 100.0:
                migrant_cells[dau] = True
            else:
                migrant_cells[dau] = False

            if jammed_cells[mom]:
                jammed_cells[dau] = True ####
            else: ####
                jammed_cells[dau] = False ####

            # Initialize intercalation status for daughter (daughters start as non-intercalating)
            if INTERCALATION_ENABLED and np.random.rand() < INTERCAL_PHASE_PERCENT / 100.0:
                intercal_cells[dau] = True
            else:
                intercal_cells[dau] = False

        # advance the clock for *all* living cells
        phase_clocks[alive] += DT

    return pos, cycle_phases, phase_clocks


def epithelium_elasticity(Xe, Db, DbT, blp0, blm0, dsb):
    """Compute elastic forces (stretch + bend) for the boundary"""
    global kb_vals
    # Xe: (Nb,2), Db, DbT: sparse matrices
    # 1) Stretch (spring) forces via matrix multiply
    fp = Db.dot(Xe)       # forward diffs
    fm = DbT.dot(Xe)      # backward diffs

    # 2) lengths
    lp = np.linalg.norm(fp, axis=1)
    lm = np.linalg.norm(fm, axis=1)
    lp_safe = np.where(lp>0, lp, 1e-12)
    lm_safe = np.where(lm>0, lm, 1e-12)

    # 3) spring tensions
    t1 = kb_vals * (lp/blp0 - 1)
    t2 = kb_vals * (lm/blm0 - 1)

    # 4) assemble stretch forces
    F_stretch = (fp * (t1/dsb)[:,None] / lp_safe[:,None] +
                 fm * (t2/dsb)[:,None] / lm_safe[:,None])

    # 5) Bending forces via discrete 4th-difference (vectorized)
    Nb = Xe.shape[0]
    Xp2 = np.roll(Xe, -2, axis=0)
    Xp1 = np.roll(Xe, -1, axis=0)
    Xm1 = np.roll(Xe, 1, axis=0)
    Xm2 = np.roll(Xe, 2, axis=0)
    fourth_diff = Xp2 - 4*Xp1 + 6*Xe - 4*Xm1 + Xm2
    Fbend = - (KBEND * kb_vals/dsb)[:,None] * fourth_diff
    
    # zero ends
    Fbend[:2,:] = 0
    Fbend[-2:,:] = 0

    Fbs = F_stretch + Fbend
    return Fbs

#---------------------------------------------------------------------------------------------------
## CELL OPERATIONS FUNCTIONS
#---------------------------------------------------------------------------------------------------

@njit(parallel=True, fastmath=True, nogil=True)
def cc_repulsion(pos_active):
    """
    Calculate cell-cell repulsion forces using spatial binning.
    Args:
        pos_active: (n_cells, 2) array of active cell positions
    Returns:
        F_cc: Cell-cell repulsion forces
    """
    n_cells = len(pos_active)
    F_cc = np.zeros((n_cells, 2), dtype=float64)
    
    # Grid setup for spatial binning
    interaction_range = 2.0 * DL_CRIT
    cell_size = interaction_range
    domain_width = XMAX - XMIN
    domain_height = YMAX - YMIN
    nx = max(1, int(domain_width / cell_size)) + 1
    ny = max(1, int(domain_height / cell_size)) + 1
    n_bins = nx * ny
    
    # Initialize spatial data structures
    head = np.full(n_bins, -1, dtype=int64)
    nxt = np.full(n_cells, -1, dtype=int64)
    
    # Assign particles to cells
    for i in range(n_cells):
        ix = int((pos_active[i, 0] - XMIN) / cell_size)
        iy = int((pos_active[i, 1] - YMIN) / cell_size)
        ix = max(0, min(ix, nx - 1))
        iy = max(0, min(iy, ny - 1))
        cell_idx = ix + iy * nx
        nxt[i] = head[cell_idx]
        head[cell_idx] = i
    
    # Calculate forces using cell lists
    interaction_range_sq = interaction_range * interaction_range
    
    for cell_y in prange(ny):
        for cell_x in range(nx):
            cell_idx = cell_x + cell_y * nx
            i = head[cell_idx]
            while i != -1:
                xi, yi = pos_active[i, 0], pos_active[i, 1]
                # Check all 8 neighbors (including self)
                for dy_cell in (-1, 0, 1):
                    for dx_cell in (-1, 0, 1):
                        neighbor_x = cell_x + dx_cell
                        neighbor_y = cell_y + dy_cell
                        if (neighbor_x < 0 or neighbor_x >= nx or 
                            neighbor_y < 0 or neighbor_y >= ny):
                            continue
                        neighbor_idx = neighbor_x + neighbor_y * nx
                        j = head[neighbor_idx]
                        while j != -1:
                            if i < j:  # Only compute force once per pair
                                dx = pos_active[j, 0] - xi
                                dy = pos_active[j, 1] - yi
                                r2 = dx * dx + dy * dy
                                if 0 < r2 < interaction_range_sq:
                                    r = np.sqrt(r2)
                                    if jammed_cells[i] and jammed_cells[j]:
                                        fx = (K_CC_REP_JAMMING * max((DL_CRIT - r, 0)) - K_CC_ADH_JAMMING * max((r - DL_CRIT, 0))) * (dx / r)
                                        fy = (K_CC_REP_JAMMING * max((DL_CRIT - r, 0)) - K_CC_ADH_JAMMING * max((r - DL_CRIT, 0))) * (dy / r)
                                    else:
                                        fx = (K_CC_REP * max((DL_CRIT - r, 0)) - K_CC_ADH * max((r - DL_CRIT, 0))) * (dx / r)
                                        fy = (K_CC_REP * max((DL_CRIT - r, 0)) - K_CC_ADH * max((r - DL_CRIT, 0))) * (dy / r)
                                    F_cc[i, 0] -= fx
                                    F_cc[i, 1] -= fy
                                    F_cc[j, 0] += fx
                                    F_cc[j, 1] += fy
                            j = nxt[j]
                i = nxt[i]
    return F_cc

# ORIGINAL BC_connect - COMMENTED OUT FOR PERFORMANCE TESTING
# @njit(parallel=True, fastmath=True, nogil=True)
# def BC_connect_old(pos_active, Xe):
#     # """
#     # Boundary-cell interaction using agentslimbreg1s.py approach with cell-list optimization.
#     # 1. Standard repulsion for nearby cells
#     # 2. Bidirectional spring system between each boundary point and its closest cell
#     
#     # Returns:
#     #     F_on_cell: Forces applied to cells (shape: n_cells x 2)
#     #     F_on_epi: Forces applied to epithelium boundary points (shape: n_boundary x 2)
#     # """
#     n_cells = len(pos_active)
#     n_boundary = len(Xe)
#     F_on_cell = np.zeros((n_cells, 2), dtype=float64)      # Forces ON cells
#     F_on_epi = np.zeros((n_boundary, 2), dtype=float64)   # Forces ON epithelium
#     
#     dl_crit_sq = DL_CRIT * DL_CRIT
#     spring_rest_length = DL_CRIT
#     
#     # Cell-list optimization for boundary points
#     cell_size = DL_CRIT
#     nx = max(1, int((XMAX - XMIN) / cell_size)) + 1
#     ny = max(1, int((YMAX - YMIN) / cell_size)) + 1
#     n_bins = nx * ny
#     
#     # Cell list data structures for boundary points
#     head_boundary = np.full(n_bins, -1, dtype=int64)
#     nxt_boundary = np.full(n_boundary, -1, dtype=int64)
#     
#     # Bin each boundary point
#     for j in range(n_boundary):
#         ix = int((Xe[j, 0] - XMIN) / cell_size)
#         iy = int((Xe[j, 1] - YMIN) / cell_size)
#         if ix < 0: ix = 0
#         elif ix >= nx: ix = nx-1
#         if iy < 0: iy = 0
#         elif iy >= ny: iy = ny-1
#         b = ix + iy*nx
#         nxt_boundary[j] = head_boundary[b]
#         head_boundary[b] = j
#     
#     # 1. STANDARD CELL-BOUNDARY REPULSION with cell-list
#     for i in prange(n_cells):
#         total_force_x = 0.0
#         total_force_y = 0.0
#         
#         xi = pos_active[i, 0]
#         yi = pos_active[i, 1]
#         
#         # Determine cell's bin
#         ix0 = int((xi - XMIN) / cell_size)
#         iy0 = int((yi - YMIN) / cell_size)
#         
#         # Check 3x3 neighboring bins for boundary points
#         for dix in (-1, 0, 1):
#             ix = ix0 + dix
#             if ix < 0 or ix >= nx: continue
#             for diy in (-1, 0, 1):
#                 iy = iy0 + diy
#                 if iy < 0 or iy >= ny: continue
#                 b = ix + iy*nx
#                 j = head_boundary[b]
#                 
#                 # Check all boundary points in this bin
#                 while j != -1:
#                     dx = xi - Xe[j, 0]
#                     dy = yi - Xe[j, 1]
#                     dist_sq = dx*dx + dy*dy
#                     
#                     if dist_sq < dl_crit_sq and dist_sq > 0:
#                         dist = np.sqrt(dist_sq)
#                         fx = dx / dist  # normalized direction from boundary to cell
#                         fy = dy / dist
#                         force_magnitude = 2.0 * (1.0 - dist/DL_CRIT) 
#                         
#                         total_force_x += force_magnitude * fx
#                         total_force_y += force_magnitude * fy
#                         
#                         # Apply equal and opposite force to epithelium point
#                         F_on_epi[j, 0] -= force_magnitude * fx  # Force on epithelium (Newton's 3rd law)
#                         F_on_epi[j, 1] -= force_magnitude * fy
#                     
#                     j = nxt_boundary[j]
#         
#         # Set total accumulated force on cell (AFTER loop)
#         F_on_cell[i, 0] = total_force_x
#         F_on_cell[i, 1] = total_force_y
# 
#     # 2. BIDIRECTIONAL SPRING SYSTEM with cell-list optimization
#     attachment_enabled = True
#     if attachment_enabled:
#         for j in prange(n_boundary):
#             min_dist_sq = 1e20
#             closest_cell = -1
#             
#             # Find closest cell to this boundary point within reasonable distance
#             max_spring_distance_sq = (3.0 * DL_CRIT)**2  # Only connect if reasonably close
#             for i in range(n_cells):
#                 dx = pos_active[i, 0] - Xe[j, 0]
#                 dy = pos_active[i, 1] - Xe[j, 1]
#                 dist_sq = dx*dx + dy*dy
#                 
#                 if dist_sq < min_dist_sq and dist_sq < max_spring_distance_sq:
#                     min_dist_sq = dist_sq
#                     closest_cell = i
#         
#             if closest_cell >= 0 and min_dist_sq > 0:
#                 current_dist = np.sqrt(min_dist_sq)
#                 
#                 # Spring force calculation: F = k * (current_length - rest_length)
#                 spring_extension = current_dist - spring_rest_length
#                 spring_force_magnitude = K_BC * spring_extension
#                 
#                 # Limit force magnitude to prevent instability
#                 # max_force_magnitude = K_BC * DL_CRIT  # Limit to spring constant * critical length
#                 # if abs(spring_force_magnitude) > max_force_magnitude:
#                 #     spring_force_magnitude = max_force_magnitude if spring_force_magnitude > 0 else -max_force_magnitude
#                 
#                 # Direction from boundary point to cell
#                 dx = pos_active[closest_cell, 0] - Xe[j, 0]
#                 dy = pos_active[closest_cell, 1] - Xe[j, 1]
#                 fx = dx / current_dist  # normalized direction
#                 fy = dy / current_dist
#                 
#                 if spring_extension > 0:
#                     # Cell is too far: pull boundary toward cell and cell toward boundary
#                     F_on_epi[j, 0] += spring_force_magnitude * fx   # Force ON epithelium
#                     F_on_epi[j, 1] += spring_force_magnitude * fy
#                     
#                     # Also pull cell toward boundary (add to existing forces)
#                     F_on_cell[closest_cell, 0] -= spring_force_magnitude * fx  # Force ON cell
#                     F_on_cell[closest_cell, 1] -= spring_force_magnitude * fy
#                     
#                 else:
#                     # Cell is too close: push boundary away from cell and cell away from boundary
#                     F_on_epi[j, 0] += spring_force_magnitude * fx   # Force ON epithelium
#                     F_on_epi[j, 1] += spring_force_magnitude * fy
#                     
#                     # Also push cell away from boundary
#                     F_on_cell[closest_cell, 0] -= spring_force_magnitude * fx  # Force ON cell
#                     F_on_cell[closest_cell, 1] -= spring_force_magnitude * fy
# 
#     return F_on_cell, F_on_epi

@njit(fastmath=True, nogil=True)  # Removed parallel=True to avoid race conditions
def BC_connect(pos_active, Xe):
    """
    Optimized boundary-cell interaction using cell-list approach similar to cc_repulsion.
    1. Standard repulsion for nearby cells
    2. Bidirectional spring system between each boundary point and its closest cell
    
    Returns:
        F_on_cell: Forces applied to cells (shape: n_cells x 2)
        F_on_epi: Forces applied to epithelium boundary points (shape: n_boundary x 2)
    """
    n_cells = len(pos_active)
    n_boundary = len(Xe)
    F_on_cell = np.zeros((n_cells, 2), dtype=float64)
    F_on_epi = np.zeros((n_boundary, 2), dtype=float64)
    
    # Physical parameters
    dl_crit_sq = DL_CRIT * DL_CRIT
    spring_rest_length = DL_CRIT
    max_spring_distance = 3.0 * DL_CRIT
    max_spring_distance_sq = max_spring_distance * max_spring_distance
    
    # Grid setup for spatial binning (similar to cc_repulsion)
    interaction_range = max(2.0 * DL_CRIT, max_spring_distance)
    cell_size = interaction_range
    nx = max(1, int((XMAX - XMIN) / cell_size)) + 1
    ny = max(1, int((YMAX - YMIN) / cell_size)) + 1
    n_bins = nx * ny
    
    # Cell list data structures for CELLS (not boundary points)
    head_cells = np.full(n_bins, -1, dtype=int64)
    nxt_cells = np.full(n_cells, -1, dtype=int64)
    
    # Cell list data structures for BOUNDARY points  
    head_boundary = np.full(n_bins, -1, dtype=int64)
    nxt_boundary = np.full(n_boundary, -1, dtype=int64)
    
    # Bin all cells
    for i in range(n_cells):
        ix = int((pos_active[i, 0] - XMIN) / cell_size)
        iy = int((pos_active[i, 1] - YMIN) / cell_size)
        ix = max(0, min(ix, nx - 1))
        iy = max(0, min(iy, ny - 1))
        cell_idx = ix + iy * nx
        nxt_cells[i] = head_cells[cell_idx]
        head_cells[cell_idx] = i
    
    # Bin all boundary points
    for j in range(n_boundary):
        ix = int((Xe[j, 0] - XMIN) / cell_size)
        iy = int((Xe[j, 1] - YMIN) / cell_size)
        ix = max(0, min(ix, nx - 1))
        iy = max(0, min(iy, ny - 1))
        cell_idx = ix + iy * nx
        nxt_boundary[j] = head_boundary[cell_idx]
        head_boundary[cell_idx] = j
    
    # 1. CELL-BOUNDARY REPULSION using spatial bins
    for cell_y in range(ny):
        for cell_x in range(nx):
            cell_idx = cell_x + cell_y * nx
            
            # Get all cells in this bin
            i = head_cells[cell_idx]
            while i != -1:
                xi, yi = pos_active[i, 0], pos_active[i, 1]
                
                # Check neighboring bins for boundary points
                for dy_cell in (-1, 0, 1):
                    for dx_cell in (-1, 0, 1):
                        neighbor_x = cell_x + dx_cell
                        neighbor_y = cell_y + dy_cell
                        if (neighbor_x < 0 or neighbor_x >= nx or 
                            neighbor_y < 0 or neighbor_y >= ny):
                            continue
                        
                        neighbor_idx = neighbor_x + neighbor_y * nx
                        j = head_boundary[neighbor_idx]
                        
                        # Check all boundary points in this neighboring bin
                        while j != -1:
                            dx = xi - Xe[j, 0]
                            dy = yi - Xe[j, 1]
                            dist_sq = dx*dx + dy*dy
                            
                            if dist_sq < dl_crit_sq and dist_sq > 0:
                                dist = np.sqrt(dist_sq)
                                fx = dx / dist
                                fy = dy / dist
                                force_magnitude = 2.0 * (1.0 - dist/DL_CRIT)
                                
                                # Apply forces (Newton's 3rd law)
                                F_on_cell[i, 0] += force_magnitude * fx
                                F_on_cell[i, 1] += force_magnitude * fy
                                F_on_epi[j, 0] -= force_magnitude * fx
                                F_on_epi[j, 1] -= force_magnitude * fy
                            
                            j = nxt_boundary[j]
                
                i = nxt_cells[i]
    
    # 2. SPRING SYSTEM using spatial optimization
    # For each boundary point, find closest cell using spatial bins
    for j in range(n_boundary):
        min_dist_sq = 1e20
        closest_cell = -1
        
        # Get boundary point's bin
        bx = int((Xe[j, 0] - XMIN) / cell_size)
        by = int((Xe[j, 1] - YMIN) / cell_size)
        bx = max(0, min(bx, nx - 1))
        by = max(0, min(by, ny - 1))
        
        # Search in expanding radius until we find a close enough cell
        max_search_radius = 2  # bins to search
        for search_radius in range(max_search_radius + 1):
            found_close_cell = False
            
            # Check all bins within search radius
            for dy in range(-search_radius, search_radius + 1):
                for dx in range(-search_radius, search_radius + 1):
                    if abs(dx) != search_radius and abs(dy) != search_radius and search_radius > 0:
                        continue  # Only check perimeter for efficiency
                    
                    cell_x = bx + dx
                    cell_y = by + dy
                    if cell_x < 0 or cell_x >= nx or cell_y < 0 or cell_y >= ny:
                        continue
                    
                    cell_idx = cell_x + cell_y * nx
                    i = head_cells[cell_idx]
                    
                    while i != -1:
                        dx_cell = pos_active[i, 0] - Xe[j, 0]
                        dy_cell = pos_active[i, 1] - Xe[j, 1]
                        dist_sq = dx_cell*dx_cell + dy_cell*dy_cell
                        
                        if dist_sq < min_dist_sq and dist_sq < max_spring_distance_sq:
                            min_dist_sq = dist_sq
                            closest_cell = i
                            if dist_sq < (1.5 * DL_CRIT)**2:  # Found a very close cell
                                found_close_cell = True
                        
                        i = nxt_cells[i]
            
            if found_close_cell:
                break
        
        # Apply spring force if we found a close enough cell
        if closest_cell >= 0 and min_dist_sq > 0:
            current_dist = np.sqrt(min_dist_sq)
            spring_extension = current_dist - spring_rest_length
            spring_force_magnitude = K_BC * spring_extension
            
            # Direction from boundary point to cell
            dx = pos_active[closest_cell, 0] - Xe[j, 0]
            dy = pos_active[closest_cell, 1] - Xe[j, 1]
            fx = dx / current_dist
            fy = dy / current_dist
            
            # Apply spring forces (Newton's 3rd law)
            F_on_epi[j, 0] += spring_force_magnitude * fx
            F_on_epi[j, 1] += spring_force_magnitude * fy
            F_on_cell[closest_cell, 0] -= spring_force_magnitude * fx
            F_on_cell[closest_cell, 1] -= spring_force_magnitude * fy

    return F_on_cell, F_on_epi

@njit(parallel=True, fastmath=True, nogil=True)
def bone_interactions(pos_active, Xb, bone_passage_mask, step=0):
    """
    Calculate forces from bone interactions using cell lists for speed.
    Bone points are static, so we organize them in cells and then
    for each active cell, we only check nearby bone points.
    """
    n_cells = len(pos_active)
    n_bone = len(Xb)
    F_bone = np.zeros((n_cells, 2), dtype=float64)

    # Build cell list for bone points
    cell_size = DL_CRIT
    nx = max(1, int((XMAX - XMIN) / cell_size)) + 1
    ny = max(1, int((YMAX - YMIN) / cell_size)) + 1
    n_bins = nx * ny
    
    # Cell list data structures
    head = np.full(n_bins, -1, dtype=int64)
    nxt = np.full(n_bone, -1, dtype=int64)

    # Put each bone point into its cell
    for j in range(n_bone):
        ix = int((Xb[j, 0] - XMIN) / cell_size)
        iy = int((Xb[j, 1] - YMIN) / cell_size)
        ix = max(0, min(ix, nx - 1))
        iy = max(0, min(iy, ny - 1))
        cell_idx = ix + iy * nx
        
        # Insert bone point j at head of linked list for this cell
        nxt[j] = head[cell_idx]
        head[cell_idx] = j

    # For each active cell, find closest bone point using cell lists 
    for i in prange(n_cells):
        x, y = pos_active[i, 0], pos_active[i, 1]
        
        # Find which cell this active cell is in
        ix = int((x - XMIN) / cell_size)
        iy = int((y - YMIN) / cell_size)
        ix = max(0, min(ix, nx - 1))
        iy = max(0, min(iy, ny - 1))
        
        min_dist_sq = 1e20
        closest = -1

        # Search current cell and neighboring cells for bone points
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                nix = ix + dx
                niy = iy + dy
                if 0 <= nix < nx and 0 <= niy < ny:
                    cell_idx = nix + niy * nx
                    j = head[cell_idx]  # First bone point in this cell
                    
                    # Loop through all bone points in this cell
                    while j != -1:
                        dx_ = x - Xb[j, 0]
                        dy_ = y - Xb[j, 1]
                        dist_sq = dx_ * dx_ + dy_ * dy_
                        
                        if dist_sq < min_dist_sq:
                            min_dist_sq = dist_sq
                            closest = j
                        
                        j = nxt[j]  # Move to next bone point in this cell

        # Calculate force from closest bone point
        if closest >= 0 and min_dist_sq > 0:
            r_min = np.sqrt(min_dist_sq)
            r_vec = np.array([x - Xb[closest, 0], y - Xb[closest, 1]])
            r_hat = r_vec / (np.linalg.norm(r_vec) + 1e-8)

            # Use proper ray-casting algorithm to detect if cell is inside bone (same as hard_reset)
            inside_bone = False
            if n_bone > 0:
                j = n_bone - 1
                # Ray casting algorithm for bone
                for k in range(n_bone):
                    xi, yi = Xb[k]
                    xj, yj = Xb[j]
                    if ((yi > y) != (yj > y)) and (x < (xj - xi) * (y - yi) / (yj - yi) + xi):
                        inside_bone = not inside_bone
                    j = k
                
                # Close the bone polygon properly
                xi, yi = Xb[0]
                xj, yj = Xb[-1]
                if ((yi > y) != (yj > y)) and (x < (xj - xi) * (y - yi) / (yj - yi) + xi):
                    inside_bone = not inside_bone
            
            # Reset bone passage mask if cell is too far from bone
            # if bone_passage_mask[i] and min_dist_sq > (2 * DL_CRIT)**2: ###
            #     bone_passage_mask[i] = False ###
            
            # Calculate bone forces for cells near bone or inside bone
            if (r_min < DL_CRIT/2) or inside_bone:
                if bone_passage_mask[i]:
                    # Cell phases through - no bone force applied
                    F_bone[i, :] = np.array([0.0, 0.0])
                elif inside_bone:
                    # Cell doesn't phase through - strong repulsive force
                    F_bone[i, :] = -10*K_BONE * r_min * r_hat
                elif not inside_bone:
                    # Cell is outside bone - normal repulsive force
                    F_bone[i, :] = (K_BONE / (r_min + 1e-5)) * r_hat

    return F_bone, bone_passage_mask

@njit(parallel=True, fastmath=True, nogil=True)
def intercal_force(pos_active, intercal_pairs):
    """
    Calculate intercalation forces between paired cells.
    
    Args:
        pos_active: (n_cells, 2) array of active cell positions
        intercal_pairs: list of (i,j) index pairs into pos_active
        
    Returns:
        F_intercal: Intercalation forces
    """
    n_cells = len(pos_active)
    F_intercal = np.zeros((n_cells, 2), dtype=float64)
    
    for k in prange(len(intercal_pairs)):
        i, j = intercal_pairs[k]
        dx = pos_active[i, 0] - pos_active[j, 0]
        dy = pos_active[i, 1] - pos_active[j, 1]
        r = np.sqrt(dx**2 + dy**2)
        if r > 0:  # Avoid division by zero
            fx = K_INTERCAL * dx / r
            fy = K_INTERCAL * dy / r
            F_intercal[i, 0] -= fx
            F_intercal[i, 1] -= fy
            F_intercal[j, 0] += fx
            F_intercal[j, 1] += fy
    
    return F_intercal

@njit(parallel=True, fastmath=True, nogil=True)
def calculate_velocities(F_cc, F_on_cell, F_on_epi, F_bone, F_intercal=None):
    # """
    # Calculate velocities from total forces.
    # Uses agentslimbreg1s.py scaling for epithelium forces.
    
    # Args:
    #     F_cc: Cell-cell forces
    #     F_on_epi: Boundary collision forces
    #     F_bone: Bone interaction forces
    #     F_intercal: Intercalation forces (optional)
    #     xi: Drag coefficient
        
    # Returns:
    #     v_active: Updated velocities
    # """
    n_cells = len(F_cc)
    v_active = np.zeros((n_cells, 2), dtype=float64)
    
    for i in prange(n_cells):
        total_force_x = F_cc[i, 0] + F_on_cell[i, 0] + F_bone[i, 0]
        total_force_y = F_cc[i, 1] + F_on_cell[i, 1] + F_bone[i, 1]
        
        # Add intercalation forces if provided
        if F_intercal is not None:
            total_force_x += F_intercal[i, 0]
            total_force_y += F_intercal[i, 1]

        v_active[i, 0] = total_force_x / XI
        v_active[i, 1] = total_force_y / XI
    
    return v_active


@njit(parallel=True, fastmath=True, nogil=True)
def extra_damping(pos_active, v_active, left_wall, xb_ymax, xb_ymin, xe_ymax, xe_ymin, bone_enabled):
    """
    Apply damping effects to velocities.
    
    Args:
        pos_active: Cell positions
        v_active: Cell velocities
        left_wall: Left boundary for rightward damping
        DL_CRIT: Critical distance for bone damping
        
    Returns:
        v_active: Damped velocities
    """
    n_cells = len(pos_active)
    damping_zone_width = 0.5
    # top_wall = 1.5
    # bottom_wall = -1.5
    for i in prange(n_cells):
        x = pos_active[i, 0]
        y = pos_active[i, 1]
        
        # Rightward damping
        if x < left_wall + damping_zone_width:
            distance_ratio = (x - left_wall) / damping_zone_width
            damping_factor = 0.1 + 0.9 * distance_ratio  # 10% to 100% speed
            v_active[i, 0] *= damping_factor
        if bone_enabled:
            if xb_ymax < y < xe_ymax and x < 0: # above bone
                distance_ratio = (y - xb_ymax) / damping_zone_width
                damping_factor = 0.1 + 0.9 * distance_ratio  # 10% to 100% speed
                v_active[i, 1] *= damping_factor
            if xe_ymin < y < xb_ymin and x < 0: # below bone
                distance_ratio = (y - xe_ymin) / damping_zone_width
                damping_factor = 0.1 + 0.9 * distance_ratio  # 10% to 100% speed
                v_active[i, 1] *= damping_factor 
    
    return v_active

@njit(parallel=True, fastmath=True, nogil=True)
def right_correction(pos_active, Xe):
    # """
    # Apply rightward correction force based on agentslimbreg1s.py (lines 674-680).
    
    # Args:
    #     pos_active: Cell positions
    #     Xe: Epithelium boundary points
        
    # Returns:
    #     F_correction: Rightward correction forces
    # """
    n_cells = len(pos_active)
    F_correction = np.zeros((n_cells, 2), dtype=float64)
    
    # Calculate left wall position from epithelium boundary
    left_wall = np.min(Xe[:, 0])
    
    for i in prange(n_cells):
        if pos_active[i, 0] < left_wall:
            F_correction[i, 0] = -8.0 * pos_active[i, 0]
        elif pos_active[i, 0] < (left_wall + 0.5):
            F_correction[i, 0] = 1.0 * (1.0 - (pos_active[i, 0] - left_wall))
    
    return F_correction

@njit(parallel=True, fastmath=True, nogil=True)
def update_positions(pos_active, v_active, Xe, migrant_cells=None, jammed_cells_active=None, migration_direction=None):
    """
    Update cell positions with velocity and random noise.
    
    Args:
        pos_active: Current cell positions
        v_active: Cell velocities
        migrant_cells: Boolean array indicating which cells are migrants
        jammed_cells_active: Boolean array indicating which cells are jammed (for active cells only)
        migration_direction: 'x' or 'y' for migration direction (now ignored)
        
    Returns:
        pos_active: Updated positions
    """
    n_cells = len(pos_active)
    # eta = -6 + 12*np.random.random((n_cells, 2))  # Random noise
    eta = np.random.normal(loc=0, scale=K_RM*np.sqrt(DT), size=(n_cells, 2)) # brownian motion
    eta_jammed = np.random.normal(loc=0, scale=K_RM_JAMMING*np.sqrt(DT), size=(n_cells, 2))

    for i in prange(n_cells):
        if jammed_cells_active is not None and jammed_cells_active[i]:
            eta[i, :] = eta_jammed[i, :]
        # Note: removed redundant "else" clause that was assigning eta[i] to itself
        # Apply migration if cell is a migrant
        if migrant_cells is not None and migrant_cells[i]:
            # Initialize migration variables
            eta_migration_x = 0.0
            eta_migration_y = 0.0
            
            # Generate biased normally distributed migration steps
            if migration_direction == 'x':
                eta_migration_x = np.random.normal(loc=MIGRATION_STRENGTH, scale=MIGRATION_STD_SCALE*np.sqrt(DT))
                eta_migration_y = 0.0
            elif migration_direction == 'y':
                eta_migration_x = 0.0
                eta_migration_y = np.random.normal(loc=MIGRATION_STRENGTH, scale=MIGRATION_STD_SCALE*np.sqrt(DT))
            elif migration_direction == 'center':
                # Calculate target point: x_max of epithelium, y=0
                target_x = np.max(Xe[:, 0])
                target_y = 0.0

                dx = target_x - pos_active[i, 0]
                dy = target_y - pos_active[i, 1]
                dist = np.sqrt(dx*dx + dy*dy)
                
                # Calculate bias direction (normalized)
                if dist > 0:
                    bias_x = dx / dist
                    bias_y = dy / dist
                else:
                    bias_x = 0.0
                    bias_y = 0.0
                eta_migration_x = np.random.normal(loc=MIGRATION_STRENGTH, scale=MIGRATION_STD_SCALE*np.sqrt(DT))
                eta_migration_y = np.random.normal(loc=MIGRATION_STRENGTH, scale=MIGRATION_STD_SCALE*np.sqrt(DT))
                # if GRADIENT !='zone':
                #     eta_migration_x = MIGRATION_STRENGTH * np.random.normal(loc=bias_x, scale=np.sqrt(DT))
                #     eta_migration_y = MIGRATION_STRENGTH * np.random.normal(loc=bias_y, scale=np.sqrt(DT))
            if GRADIENT == 'zone' and pos_active[i, 0] > ZONE_START:
                # if pos_active[i, 0] < ZONE_START: # stop migrating in proliferation zone
                #     eta_migration_x = MIGRATION_STRENGTH * np.random.normal(loc=bias_x, scale=np.sqrt(DT))
                #     eta_migration_y = MIGRATION_STRENGTH * np.random.normal(loc=bias_y, scale=np.sqrt(DT))
                # else:
                eta_migration_x = 0.0
                eta_migration_y = 0.0
            
            # Update position with biased random migration
            pos_active[i, 0] += v_active[i, 0]*DT + eta[i, 0]*DT + eta_migration_x*DT
            pos_active[i, 1] += v_active[i, 1]*DT + eta[i, 1]*DT + eta_migration_y*DT
        else:
            # Standard random motion
            pos_active[i, 0] += v_active[i, 0]*DT + eta[i, 0]*DT
            pos_active[i, 1] += v_active[i, 1]*DT + eta[i, 1]*DT

    return pos_active

def cell_operations(pos_active, v_active, Xe, Xb, left_wall, reset=False, 
                   migrant_cells=None, jammed_cells_active=None, migration_direction=None, intercal_pairs=None, step=0):
    """
    Cell operations function that calls individual components.
    
    Args:
        pos_active: (n_cells, 2) array of active cell positions
        v_active: (n_cells, 2) array of active cell velocities  
        Xe: (n_boundary, 2) array of epithelium boundary points
        Xb: (n_bone, 2) array of bone boundary points
        left_wall: x-location of left wall
        reset: Whether to reset cells
        migrant_cells: Boolean array indicating migrant cells
        migration_direction: 'x' or 'y' for migration direction
        intercal_pairs: List of (i,j) pairs for intercalation
        
    Returns:
        F_cc: Cell-cell repulsion forces
        F_on_cell: Total cell-boundary forces
        F_on_epi: Boundary collision forces
        pos_active: Updated cell positions
        v_active: Updated cell velocities
    """
    global pos, v, bone_passage_mask
    # Compute forces for active cells
    F_cc = cc_repulsion(pos_active)
    F_on_cell, F_on_epi = BC_connect(pos_active, Xe)  # Using agentslimbreg1s.py epithelium
    # F_on_cell, F_on_epi = np.zeros((len(pos_active), 2)), np.zeros((Ne, 2)) ### DEBUG
    if BONE_ENABLED:
        F_bone, bone_passage_mask = bone_interactions(pos_active, Xb, bone_passage_mask, step)
    else:
        F_bone = np.zeros((len(pos_active), 2))
    
    # Compute intercalation forces if enabled
    F_intercal = None
    if intercal_pairs is not None and len(intercal_pairs) > 0:
        F_intercal = intercal_force(pos_active, intercal_pairs)

    # Compute rightward correction force (from agentslimbreg1s.py)
    # F_correction = right_correction(pos_active, Xe)  ### COMMENTED OUT

    # Update velocities and positions for active cells
    v_active = calculate_velocities(F_cc, F_on_cell, F_on_epi, F_bone, F_intercal)
    xb_ymax = Xb[:, 1].max() if BONE_ENABLED else 0.0
    xb_ymin = Xb[:, 1].min() if BONE_ENABLED else 0.0
    xe_ymax = Xe[:, 1].max()
    xe_ymin = Xe[:, 1].min()
    v_active = extra_damping(pos_active, v_active, left_wall, xb_ymax, xb_ymin, xe_ymax, xe_ymin, BONE_ENABLED)
    pos_active = update_positions(pos_active, v_active, Xe, migrant_cells, jammed_cells_active, migration_direction)

    if reset:
        # Pass empty array instead of None for Xb when bone is disabled
        Xb_param = Xb if BONE_ENABLED and Xb is not None else np.empty((0, 2))
        pos_active, v_active = hard_reset(pos_active, v_active, Xe, Xb_param, check_bone=False)

    # Copy back to global arrays
    # pos[active] = pos_active
    # v[active] = v_active

    return pos_active, v_active, F_cc, F_on_cell, F_on_epi, F_intercal

def single_iteration(step, t):
    global pos, v, n_daughter, Xe, division_status, Ne, x_cut, T_DORMANT, migrant_cells, intercal_pairs, jammed_cells, intercal_cells

    F_cc_full = np.zeros((CELLS_MAX, 2))
    F_on_cell_full = np.zeros((CELLS_MAX, 2))
    F_intercal_full = np.zeros((CELLS_MAX, 2))
    # F_correction_full = np.zeros((CELLS_MAX, 2))  ### COMMENTED OUT
    F_on_epi = np.zeros((Ne, 2))
    active = np.where(~np.isnan(pos[:, 0]))[0]

    # Handle cell cycle events (division/death)
    if t >= T_DORMANT:
        # Use directed division if specified
        cell_cycle(div_allowed=True, gradient=GRADIENT, directed_angle=DIRECTED_DIVISION_ANGLE, migrant_cells=migrant_cells)
    else:
        cell_cycle(div_allowed=False, gradient=GRADIENT, migrant_cells=migrant_cells)

    # Update intercalation pairs after cell death
    if INTERCALATION_ENABLED and intercal_pairs is not None:
        intercal_pairs = update_intercal_pairs_after_death(intercal_pairs, death_mask)
        # Update intercalation pairs for graduation (cells reaching y=0 line)
        intercal_pairs, intercal_cells = update_intercal_pairs_graduation(intercal_pairs, intercal_cells, pos)

    if len(active) == 0:
        return np.zeros((CELLS_MAX, 2)), np.zeros((CELLS_MAX, 2)), division_status, 0
    
    pos_active = pos[active].copy()
    v_active = np.zeros((len(active), 2))

    # Get migrant cells for active cells
    migrant_cells_active = migrant_cells[active]
    
    # Get jammed cells for active cells
    jammed_cells_active = jammed_cells[active]

    # Call cell operations (in-place update of pos and v)
    # Map intercalation pairs to local indices
    intercal_pairs_local = map_intercal_pairs_to_local(intercal_pairs, active)
    
    if step % 100 == 0:
        pos_active, v_active, F_cc, F_on_cell, F_on_epi, F_intercal = cell_operations(
            pos_active, v_active, Xe, Xb, left_wall, reset=True,
            migrant_cells=migrant_cells_active, 
            jammed_cells_active=jammed_cells_active,
            migration_direction=MIGRATION_DIRECTION if MIGRATION_ENABLED else None,
            intercal_pairs=intercal_pairs_local, step=step
        )
    else:
        pos_active, v_active, F_cc, F_on_cell, F_on_epi, F_intercal = cell_operations(
            pos_active, v_active, Xe, Xb, left_wall, reset=False,
            migrant_cells=migrant_cells_active, 
            jammed_cells_active=jammed_cells_active,
            migration_direction=MIGRATION_DIRECTION if MIGRATION_ENABLED else None,
            intercal_pairs=intercal_pairs_local, step=step
        )
    if JAMMING_ENABLED:
        update_jammed_cells(Xe)
    # Copy forces to full arrays for data collection
    if step % FRAME_SKIP == 0:
        F_cc_full[active] = F_cc.copy()
        F_on_cell_full[active] = F_on_cell.copy()
        # F_correction_full[active] = F_correction.copy()  ### COMMENTED OUT
        if F_intercal is not None:
            F_intercal_full[active] = F_intercal.copy()
        F = F_cc_full + F_on_cell_full + F_intercal_full  # removed F_correction_full
        # if JAMMING_ENABLED and step % (10 * FRAME_SKIP) == 0:
        
    else:
        F = None

    # Copy updated positions and velocities back to main arrays
    pos[active] = pos_active.copy()
    v = np.zeros((CELLS_MAX, 2))  
    v[active] = v_active

    # Update boundary elasticity and positions
    F_elast = epithelium_elasticity(Xe, Db, DbT, blp0, blm0, dsb)
    Fb = F_elast + F_on_epi  # 5x scaling on epithelium forces (same as agentslimbreg1s.py)

    # update_mask = Xe[1:-1, 0] >= -2.0 # update everything for now
    # minimal_update_mask = ~update_mask
    Xe[1:-1] += (Fb[1:-1] / XI) * DT
    # Xe += (Fb / XI) * DT

    # fix coordinates of the first and last points
    Xe[0, 0] = Xe0[0, 0]
    Xe[-1, 0] = Xe0[-1, 0]
    Xe[0, 1] = Xe0[0, 1]
    Xe[-1, 1] = Xe0[-1, 1]
    # Xe[0, 0] = Xe0[0, 0] # fix x coordinates of the first and last points
    # Xe[-1, 0] = Xe0[-1, 0]

    # Recalculate active cells after cell cycle events (division/death)
    active_after_cycle = np.where(~np.isnan(pos[:, 0]))[0]
    N_active = len(active_after_cycle)
    return F, v, division_status, N_active

#---------------------------------------------------------------------------------------------------
## MAIN SIMULATION FUNCTION
#---------------------------------------------------------------------------------------------------

def run_simulation():
    """Main simulation loop with CSV data collection"""
    global pos, Xe, Ne, Db, blp0, blm0, dsb, soft_idx, T_DORMANT, migrant_cells, intercal_pairs, jammed_cells, intercal_cells
    N0 = np.where(~np.isnan(pos[:,0]))[0].size
    print(f"Running simulation for {STEPS_TOTAL} steps")
    print(f"Initial cell count: {N0}")
    print(f"Boundary points: {Ne}")
    
    # Initialize cases
    if MIGRATION_ENABLED:
        migrant_cells = init_migration_cells(CELLS_MAX, MIGRATION_PERCENT)
        print(f"Migration enabled: {int(N0 * MIGRATION_PERCENT / 100.0)} cells migrating ({MIGRATION_PERCENT}%) in {MIGRATION_DIRECTION} direction")
    else:
        migrant_cells = np.zeros(CELLS_MAX, dtype=bool)
        
    if INTERCALATION_ENABLED:
        intercal_cells, intercal_pairs = init_intercal_pairs(pos, N_INTERCAL_PAIRS)
        print(f"Intercalation enabled: {len(intercal_pairs)} pairs")
    else:
        intercal_cells = np.zeros(CELLS_MAX, dtype=bool)
        intercal_pairs = None
    
    if DIRECTED_DIVISION_ANGLE is not None:
        print(f"Directed division enabled: angle = {DIRECTED_DIVISION_ANGLE} radians")
    else:
        print("Directed division disabled: using uniform random division")

    if JAMMING_ENABLED:
        update_jammed_cells(Xe)
        print(f"Jammming enabled: {jammed_cells.sum()} cells initially jammed")
        # print(f"Jammming zone: {JAMMING_ZONE_X} x {JAMMING_ZONE_Y}")
        print(f"Jammming zone width: {JAMMING_ZONE_WIDTH}")
    else:
        jammed_cells = np.zeros(CELLS_MAX, dtype=bool)
    
    # Initialize animation manager
    animation_manager = AnimationManager(OUTPUT_DIR, VIDEO_PARAMS, VIDEO_FLAG)
    
    # Set up signal handling for clean shutdown
    setup_signal_handler(animation_manager)
    
    # Data collection - use lists for flexibility
    positions = []
    forces = []
    velocities = []
    boundaries = []
    divisions = []
    deaths = []
    Gphase = []
    Mphase = []
    times = []
    cell_count = []
    step_times = [] # for timing runtime
    
    # Additional cell status data
    phase_clocks_data = []
    cycle_phases_data = []
    migrant_cells_data = []
    intercal_cells_data = []
    jammed_cells_data = []
    
    # Save initial state
    positions.append(pos.copy())
    forces.append(np.zeros_like(pos))
    velocities.append(np.zeros_like(pos))
    boundaries.append(Xe.copy())
    Gphase.append(cycle_phases[cycle_phases==0].shape[0])
    Mphase.append(cycle_phases[cycle_phases==1].shape[0])
    divisions.append(division_status.copy())
    deaths.append(death_mask.copy())
    cell_count.append(N0)
    times.append(0.0)
    
    # Save initial cell status data
    phase_clocks_data.append(phase_clocks.copy())
    cycle_phases_data.append(cycle_phases.copy())
    migrant_cells_data.append(migrant_cells.copy())
    intercal_cells_data.append(intercal_cells.copy())
    jammed_cells_data.append(jammed_cells.copy())
    
    start = time.time()
    last_time = start  # For step time calculation
    
    # Main simulation loop
    for step, t in enumerate(np.arange(0, TMAX, DT)):
        # Use the refactored single_iteration function
        F, v, div_status, N_active = single_iteration(step, t)
        

            
        if step % FRAME_SKIP == 0 and step > 0:

            if VIDEO_FLAG:
                # Create boolean masks for intercalation and migration cells
                intercal_cells_anim = None
                if INTERCALATION_ENABLED and intercal_pairs is not None and len(intercal_pairs) > 0:
                    intercal_cells_anim = np.zeros(CELLS_MAX, dtype=bool)
                    for i, j in intercal_pairs:
                        intercal_cells_anim[i] = True
                        intercal_cells_anim[j] = True
                # migrant_cells is already a boolean mask of length CELLS_MAX
                animation_manager.animate_frame(step, t, pos, Xe, Xb, pos0, cycle_phases,
                                            kb_vals=kb_vals,
                                            migrant_cells=migrant_cells,
                                            intercal_cells=intercal_cells_anim,
                                            jammed_cells=jammed_cells)
                animation_manager.animate_density_heatmap(step, t, pos, Xe, Xb, kb_vals, x_cut,
                                                        bin_size=0.1)
            # Save this timestep's data
            positions.append(pos.copy())
            forces.append(F.copy())
            velocities.append(v.copy())
            boundaries.append(Xe.copy())
            Gphase.append(cycle_phases[cycle_phases==0].shape[0])
            Mphase.append(cycle_phases[cycle_phases==1].shape[0])
            divisions.append(division_status.copy())
            deaths.append(death_mask.copy())
            times.append(t)
            cell_count.append(N_active)
            
            # Save cell status data
            phase_clocks_data.append(phase_clocks.copy())
            cycle_phases_data.append(cycle_phases.copy())
            migrant_cells_data.append(migrant_cells.copy())
            intercal_cells_data.append(intercal_cells.copy())
            jammed_cells_data.append(jammed_cells.copy())

            current_time = time.time()
            avg_step_time = (current_time - last_time) / FRAME_SKIP
            step_times.append(avg_step_time)

            active_cells = np.where(~np.isnan(pos[:,0]))[0].size
            print(f"t = {t:.2f}, Step {step}/{STEPS_TOTAL}, cells: {active_cells}")
            last_time = current_time
    elapsed = time.time() - start
    readable_time = time.strftime('%H:%M:%S', time.gmtime(elapsed))
    print(f'time elapsed: {readable_time}')
    
    # Clean up animation resources
    animation_manager.close()
    
    # Save first and last frames as separate images
    if len(positions) > 1:
        # Get first and last frame data
        pos_first = positions[0]
        pos_last = positions[-1]
        Xe_first = boundaries[0] 
        Xe_last = boundaries[-1]
        
        # Get initial and final cycle phases
        cycle_phases_first = np.zeros_like(cycle_phases)  # All cells start in G0/G1
        cycle_phases_last = cycle_phases.copy()
        
        # Create intercalation cells mask if needed
        intercal_cells_mask = None
        if INTERCALATION_ENABLED and intercal_pairs is not None and len(intercal_pairs) > 0:
            intercal_cells_mask = np.zeros(CELLS_MAX, dtype=bool)
            for i, j in intercal_pairs:
                intercal_cells_mask[i] = True
                intercal_cells_mask[j] = True
        
        save_first_last_frames(
            pos_first, pos_last, Xe_first, Xe_last, Xb, kb_vals,
            cycle_phases_first, cycle_phases_last, x_cut, 
            OUTPUT_DIR=OUTPUT_DIR, 
            migrant_cells=migrant_cells, 
            intercal_cells=intercal_cells_mask
        )
    
    # Prepare data dictionary
    data_dict = {
        'positions': np.array(positions),
        'forces': np.array(forces),
        'velocities': np.array(velocities),
        'boundaries': np.array(boundaries),
        'divisions': np.array(divisions),
        'deaths': np.array(deaths),
        'times': np.array(times),
        'phase_clocks': np.array(phase_clocks_data),
        'cycle_phases': np.array(cycle_phases_data),
        'migrant_cells': np.array(migrant_cells_data),
        'intercal_cells': np.array(intercal_cells_data),
        'jammed_cells': np.array(jammed_cells_data),
        'kb_vals': kb_vals,
        'elapsed': readable_time
    }
    

    # Plots
    density_heatmap(kb_vals, pos=pos, Xe=Xe, x_cut=x_cut, OUTPUT_DIR=OUTPUT_DIR, bin_size=0.25, shading='gouraud', fig_mode=True)#0.1)
    density_heatmap(kb_vals, pos=pos, Xe=Xe, x_cut=x_cut, OUTPUT_DIR=OUTPUT_DIR, bin_size=0.25, shading='auto', fig_mode=True)#0.1)
    cycle_plot2(times, cell_count, N0, OUTPUT_DIR=OUTPUT_DIR)
    phase_distribution_plot(times, Gphase, Mphase, fit=False, OUTPUT_DIR=OUTPUT_DIR)
    if Xe is not None:
        Xe_growth = Xe[Xe[:, 0] > x_cut]
        
        # Calculate metrics
        area, perimeter, aspect_ratio, ellipticity, roundness, a , b, volume_fraction = morphometrics(Xe, pos=pos, x_cut=x_cut)
        boundary_plot(Xe0, Xe, Xe_growth, Xb, x_cut, 
                    aspect_ratio=aspect_ratio, area=area, roundness=roundness,
                    a=a, perimeter=perimeter, ellipticity=ellipticity, 
                    OUTPUT_DIR=OUTPUT_DIR)
    # if Xe is None:
    #     area, perimeter, aspect_ratio, ellipticity, roundness, a , b, volume_fraction = morphometrics(Xe=None, pos=pos)
    #     Xe_growth = None
    #     boundary_plot(Xe0, Xe, Xe_growth, Xb, x_cut, 
    #                 aspect_ratio=aspect_ratio, area=area, roundness=roundness,
    #                 a=a, perimeter=perimeter, ellipticity=ellipticity, 
    #                 OUTPUT_DIR=OUTPUT_DIR, pos0=pos0, pos_final=pos)

    if TMAX > T_DORMANT:
        runtime_plot(cell_count, step_times, OUTPUT_DIR=OUTPUT_DIR)

    trajectory_plot(positions, Xe, x_cut, deaths, kb_vals, OUTPUT_DIR=OUTPUT_DIR)
    _, _, slope, D = MSD_plot(np.array(positions), OUTPUT_DIR=OUTPUT_DIR)
    print(f'MSD slope: {slope}, D: {D}')
    # Prepare config parameters for saving
    config_params = {
        'Time Parameters': {
            'TMAX': TMAX,
            'DT': DT,
            'STEPS_TOTAL': STEPS_TOTAL,
            'ELAPSED': readable_time
        },
        'Physical Parameters': {
            'DL_CRIT': DL_CRIT,
            'XI': XI,
            'KB_MAX': KB_MAX,
            'KB_MIN': KB_MIN,
            'KBEND': KBEND,
            # 'XPROX': XPROX,
            'K_BC': K_BC,
            'K_INTERCAL': K_INTERCAL,
            'K_RM': K_RM,

            'SOFTENING_ENABLED': ALLOW_SOFTENING
        },
        'Cell Division Parameters': {
            'KDEATH': KDEATH,
            'M_LENGTH': M_LENGTH,
            # 'G_LENGTH': G_LENGTH,
            'G_LENGTH_MAX': G_LENGTH_MAX,
            'G_LENGTH_MIN': G_LENGTH_MIN,
            'GRADIENT': GRADIENT,
            'OFFSET': OFFSET
        },
        'Case Parameters': {
            'JAMMING_ENABLED': JAMMING_ENABLED,
            'JAMMING_ZONE_WIDTH': JAMMING_ZONE_WIDTH,
            'K_CC_REP_JAMMING': K_CC_REP_JAMMING,
            'K_CC_ADH_JAMMING': K_CC_ADH_JAMMING,
            'K_RM_JAMMING': K_RM_JAMMING,
            'MIGRATION_ENABLED': MIGRATION_ENABLED,
            'MIGRATION_PERCENT': MIGRATION_PERCENT,
            'MIGRATION_DIRECTION': MIGRATION_DIRECTION,
            'DIRECTED_DIVISION_ANGLE': DIRECTED_DIVISION_ANGLE,
            'INTERCALATION_ENABLED': INTERCALATION_ENABLED,
            'N_INTERCAL_PAIRS': N_INTERCAL_PAIRS
        },
        'Domain Bounds': {
            'XMIN': XMIN,
            'XMAX': XMAX,
            'YMIN': YMIN,
            'YMAX': YMAX
        },
        'Simulation Settings': {
            'VIDEO_FLAG': VIDEO_FLAG,
            'FRAME_SKIP': FRAME_SKIP,
            'PROFILING_FLAG': PROFILING_FLAG
        }
    }
    
    # Save simulation data
    save_files(data_dict, config_params, pos0, pos, Xe0, Xe, x_cut, n_daughter, N0, OUTPUT_DIR=OUTPUT_DIR, FRAME_SKIP=FRAME_SKIP)
    print("Simulation finished and data saved.")
    
    # Return data for profiler (required format)
    return data_dict, readable_time

if __name__ == "__main__":
    if PROFILING_FLAG:
        # Run with profiling
        data_dict, elapsed = profiling(run_simulation, OUTPUT_DIR)
    else:
        run_simulation()
