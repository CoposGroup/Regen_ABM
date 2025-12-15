"""
Limb Regeneration Simulation - Agents Based Model (2D) - Version 11
Ansa Brew-Smith, December 2025
Copos Lab, Northeastern University

TODO:
    - concise doc strings on all functions
    - clean up comments
    - remove dead code, unused imports/variables, etc.

cell types:
0 = normal
1 = migrant
2 = intercalating
3 = jammed
"""
import warnings
warnings.filterwarnings('ignore')

import matplotlib
matplotlib.use('Agg')  # Set backend before importing pyplot
import matplotlib.pyplot as plt
import numpy as np
import time
import os
from numba import njit, prange, set_num_threads, get_num_threads
from numpy import int64, float64
from scipy.sparse import spdiags
from scipy.special import expit
import shutil

# Import configuration and utilities
from config import *
from utils.animations import AnimationManager
from utils.data_io import save_files
from utils.profiler import profiling
from utils.signal_handling import setup_signal_handler
from utils.post_process import morphometrics
from utils.curve_comp import coefficients, distance_metric
from config import get_output_dir

# OUTPUT_DIR = get_output_dir()
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Load initial cell positions
if BONE_FORCES:
    pos0 = np.loadtxt(CELL_INIT_FILE_BONE, delimiter=',')
else:
    pos0 = np.loadtxt(CELL_INIT_FILE_NO_BONE, delimiter=',')
N0 = len(pos0)

# Pre-allocate cell arrays 
CELLS_MAX = 10 * N0
pos = np.full((CELLS_MAX, 2), np.nan)
v = np.zeros((CELLS_MAX, 2))
pos[:N0, :] = pos0

# Set up cell cycle
division_status = np.zeros((CELLS_MAX,), dtype=bool)
cycle_phases = np.zeros((CELLS_MAX,), dtype=int) # 0 = G1 (at rest), 1 = S/G2/M (division), start all cells at G1
empty_slots = np.isnan(pos[:, 0])
cycle_phases[empty_slots] = -1 # Empty slots are assigned -1
death_mask = np.zeros((CELLS_MAX,), dtype=bool)
phase_clocks = np.zeros((CELLS_MAX,), dtype=float)

n_daughter = 0
n_deaths = 0

# Global variables for hypotheses and bone passage

regulation_front = np.inf
# intercal_pairs = None
bone_passage_mask = np.zeros(CELLS_MAX, dtype=bool) ### remove
bone_passage_mask[np.random.choice(CELLS_MAX, size=int(CELLS_MAX*BONE_PHASE_PERCENT/100), replace=False)] = True ### (remove) set X% of cells to be able to phase through bone

cell_types = np.zeros(CELLS_MAX, dtype=int)

def init_migration_cells(n_cells_max):
    """
    Initialize which cells are migrants at the start by marking entries in cell_types.
    
    Args:
        n_cells_max: Total number of cells (capacity)
        migration_percent: Percentage of cells to make migrants (0-100)
        
    Returns:
        cell_types: Integer type array updated in place (0 normal, 1 migrant, 2 intercalating, 3 jammed)
    """
    global cell_types
    n_migrants = int(N0 * MIGRATION_PERCENT / 100.0)
    if n_migrants > 0:
        if WHICH_MIGRATION == 'anterior_posterior':
            anterior_posterior = np.zeros(n_cells_max, dtype=bool)
            anterior_posterior[np.where((pos[:, 1] < -1.5 + 2*DL_CRIT) | (pos[:, 1] > 1.5 - 2*DL_CRIT))[0]] = True
            migrant_indices = np.random.choice(np.where(anterior_posterior)[0], size=n_migrants, replace=False)
            cell_types[migrant_indices] = 1
        elif WHICH_MIGRATION == 'random':
            if JAMMING_ENABLED:
                migrant_indices = np.random.choice(np.where(cell_types != 3)[0], size=n_migrants, replace=False)
            else:
                migrant_indices = np.random.choice(N0, size=n_migrants, replace=False)
            
            cell_types[migrant_indices] = 1
            # print(f"Migrants: {np.sum(migrant_cells)}")
        else:
            raise ValueError(f"Invalid migration type: {WHICH_MIGRATION}")
    return cell_types

def update_jammed_cells(Xe):
    """Update jammed cells (type 3) based on position relative to epithelium."""
    global pos, cell_types
    x_max = Xe[:, 0].max()
    jamming_area = ~(((x_max - pos[:, 0]) < JAMMING_ZONE_WIDTH) & (pos[:, 1] > -0.9) & (pos[:, 1] < 0.9)) | (np.abs(pos[:, 1]) > 1.0) & ~np.isnan(pos[:, 0])
    cell_types[jamming_area] = 3

def init_intercal_cells():
    """

    """
    global cell_types
    n_intercal = int(N0 * INTERCAL_PERCENT / 100.0)
    intercal_indices = np.random.choice(N0, size=n_intercal, replace=False)
    cell_types[intercal_indices] = 2

    return cell_types

#---------------------------------------------------------------------------------------------------
## CORE SIMULATION FUNCTIONS
#---------------------------------------------------------------------------------------------------

def build_boundaries(soft=True):
    """Construct the boundary shapes for the limb regeneration simulation (Xb is bone, Xe is epithlium"""

    ## EPITHELIUM ##
    Xe0 = np.loadtxt(BOUNDARY_FILE, delimiter=',') # epithelium
    Xe = Xe0.copy()
    xe = Xe0[:, 0]
    ye = Xe0[:, 1]
    
    # Sparse forward-difference matrix Db
    Ne = Xe.shape[0]
    e = np.ones(Ne)
    Db = spdiags([-e, e], [0,1], Ne, Ne, format='csr')
    Db[Ne-1, 0] = 1

    # Load rest lengths from saved files if available, otherwise compute from current geometry
    dsb_file = os.path.join(INPUT_DIR, 'dsb.npy')
    blp0_file = os.path.join(INPUT_DIR, 'blp0.npy')
    blm0_file = os.path.join(INPUT_DIR, 'blm0.npy')
    ne_file = os.path.join(INPUT_DIR, 'Ne.npy')
    
    if all(os.path.exists(f) for f in [dsb_file, blp0_file, blm0_file, ne_file]):
        saved_Ne = int(np.load(ne_file))
        if saved_Ne == Ne:
            dsb = float(np.load(dsb_file))
            blp0 = np.load(blp0_file)
            blm0 = np.load(blm0_file)
            # print(f"Loaded rest lengths from saved files (Ne={Ne})")
        else:
            print(f"Warning: Saved rest lengths have Ne={saved_Ne} but current boundary has Ne={Ne}. Computing from current geometry.")
            dsb = np.hypot(*(Xe0[1] - Xe0[0]))
            blp0 = np.hypot(*(Db @ Xe0).T)
            blm0 = np.hypot(*(Db.T @ Xe0).T)
    else:
        dsb = np.hypot(*(Xe0[1] - Xe0[0]))  # first segment length
        blp0 = np.hypot(*(Db @ Xe0).T)  # rest length of edge from i to i+1
        blm0 = np.hypot(*(Db.T @ Xe0).T)  # rest length of edge from i-1 to i

    # blp0_mean = np.mean(np.hypot(*(Db @ Xe0).T)) ##
    # blm0_mean = np.mean(np.hypot(*(Db.T @ Xe0).T)) ##
    # blp0 = np.ones(Ne) * blp0_mean
    # blm0 = np.ones(Ne) * blm0_mean

    # Calculate soft boundary indices - keep sigmoid softness distribution
    center = Xe0[:, 0].max() - 0.05  # Center just before the tip  
    width = 0.01  # steep transition
    y_lo, y_hi = -1.0, 1.0              # vertical bounds for soft region

    # Sigmoid window in y
    sig_lo = expit((ye - y_lo) / 0.1)  # 0.05 controls sharpness of lower edge
    sig_hi = expit(-(ye - y_hi) / 0.1) # 0.05 controls sharpness of upper edge
    window = sig_lo * sig_hi            # 1 inside, 0 outside, smooth transition

    # Stiffness profile: soft at tip (large x), stiff at base (small x)

    k_b = lambda x: KB_MID + (KB_MIN - KB_MID) * expit((x - center) / width)
    if soft:
        kb_vals = KB_MID * (1 - window) + k_b(xe) * window        
    else:
        kb_vals = np.ones(Ne) * KB_MID
        # kb_vals[np.where((0.75 < Xe[:, 1]) & (Xe[:, 1] < 1.25) | (-1.25 < Xe[:, 1]) & (Xe[:, 1] < -0.75))[0]] = KB_MIN ### corner softening
    # Create a small linear transition zone around x=-1 (+/- 0.2)
    transition_center = -1.0
    transition_half_width = 0.2
    
    # Linear gradient from KB_MAX (at x <= -1.2) to existing kb_vals (at x >= -0.8)
    x_left = transition_center - transition_half_width   # -1.2
    x_right = transition_center + transition_half_width  # -0.8
    
    # calculate linear interpolation factor (0 at x_left, 1 at x_right)
    transition_factor = np.clip((xe - x_left) / (2 * transition_half_width), 0, 1)
    
    # apply transition
    kb_vals = np.where(xe < x_left, KB_MAX,
                      np.where(xe > x_right, kb_vals,
                              KB_MAX * (1 - transition_factor) + kb_vals * transition_factor))
    if SPORATIC_SOFTENING: # random patches of soft epithelium
        soft_patches_ranges = [range(10, 15), range(26, 32), range(57,62), range(120, 124), range(140,144), range(165,170), range(184, 190)]
        soft_indices = np.concatenate([np.array(list(r)) for r in soft_patches_ranges])
        kb_vals[soft_indices] = KB_MIN

    ## BONE ##
    if BONE_VISUALIZATION or BONE_FORCES:
        Xb = np.loadtxt(BONE_FILE, delimiter=',')
    else:
        Xb = None

    return Xe0, Xe, Ne, Db, blp0, blm0, dsb, kb_vals, Xb

# Initialize boundary
Xe0, Xe, Ne, Db, blp0, blm0, dsb, kb_vals, Xb = build_boundaries(soft=ALLOW_SOFTENING)
left_wall = np.min(Xe0[:, 0])
x_cut = 0 ## Xe0[:, 0].max() # amputation plane (should be at x=0)
active_epi_indices = np.where(Xe[:, 0] >= x_cut)[0]
n_active_epi = len(active_epi_indices)
DbT = Db.T.tocsr() # for elasticity function

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
        if BONE_FORCES and check_bone:
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

            if inside_bone and BONE_FORCES:
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

def cell_cycle(div_allowed=True, directed_angle=None, gradient=None):
    global pos, cycle_phases, phase_clocks, death_mask, n_daughter, cell_types, n_deaths, regulation_front#, jammed_cells

    alive = ~np.isnan(pos[:,0])
    r_death = np.random.rand(cycle_phases.size)
    x_arr = pos[:, 0][alive]

    if gradient is None:
        # G_lengths = np.zeros_like(cycle_phases)
        G_lengths = G_LENGTH # change to constant scalar
    # elif gradient == 'linear':
    #     G_lengths_func = lambda x: G_LENGTH_MAX + (G_LENGTH_MIN - G_LENGTH_MAX) * (x - x_min) / (x_max - x_min)
    #     G_lengths = np.zeros(cycle_phases.shape, dtype=float)  # must be float, not int! int gives infinite probability because it rounds G_LENGTH to 0
    #     G_lengths[alive] = G_lengths_func(x_arr)
    elif gradient == 'zone':
        G_lengths_func = lambda x: np.where(x < regulation_front, G_LENGTH_MAX, G_LENGTH_MIN) # max length when cells are left of migration front plane, min length to right (distal) of front
        G_lengths = np.zeros(cycle_phases.shape, dtype=float)  # MUST be float, not int!
        G_lengths[alive] = G_lengths_func(x_arr)

    # cell death
    pD = KDEATH*DT
    die_mask = alive & (r_death < pD) # decide who dies
    pos[die_mask, :] = np.nan
    cycle_phases[die_mask] = -1      # or some sentinel
    death_mask[die_mask] = True
    n_deaths += np.sum(die_mask)
    cell_types[die_mask] = 0 # if a new cell takes this index, it is normal by default
    
    # Recompute alive mask after deaths to exclude just-died cells
    alive = ~np.isnan(pos[:,0])

    if div_allowed:
        # masks for G1 vs M
        r_trans = np.random.rand(cycle_phases.size) # transition probabilities
        maskG = alive & (cycle_phases == 0)
        maskM = alive & (cycle_phases == 1)

        # per‐cell transition probabilities (convert phase_clocks from time to steps)
        pG = 1.0 / (G_lengths/DT) # geometric transition times with mean G_length or M_length
        pM = 1.0 / (M_LENGTH/DT)
        
        # decide who enters M and who divides
        enterM = maskG & (r_trans < pG)
        divideM = maskM & (r_trans < pM)

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
                r_div_angle = np.random.rand()
                if r_div_angle < 0.5:
                    ang += np.pi # 50% chance to flip direction
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

            
            # Assign daughter cell type with proper priority
            if cell_types[mom] == 3:
                # Jammed cells produce jammed daughters
                cell_types[dau] = 3
            elif MIGRATION_ENABLED and np.random.rand() < MIGRATION_PERCENT / 100.0:
                # Random chance to be a migrant cell
                cell_types[dau] = 1
            elif INTERCALATION_ENABLED and np.random.rand() < INTERCAL_PERCENT / 100.0:
                # Random chance to be an intercalating cell
                cell_types[dau] = 2
            else:
                # Default: normal cell
                cell_types[dau] = 0

        # advance the clock for *all* living cells (recompute alive to include daughters)
        alive = ~np.isnan(pos[:,0])
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

    # 5) Bending forces via discrete 4th-difference
    # Nb = Xe.shape[0]
    # Xp2 = np.roll(Xe, -2, axis=0)
    # Xp1 = np.roll(Xe, -1, axis=0)
    # Xm1 = np.roll(Xe, 1, axis=0)
    # Xm2 = np.roll(Xe, 2, axis=0)
    # fourth_diff = Xp2 - 4*Xp1 + 6*Xe - 4*Xm1 + Xm2
    # # Fbend = - (KBEND * kb_vals/dsb)[:,None] * fourth_diff
    # Fbend = - (KBEND * kb_vals/(dsb**4))[:,None] * fourth_diff
    # print(f'dsb: {dsb}')
    # zero ends to correct from rolling over (we do not have periodic boundary conditions)
    # Fbend[:2,:] = 0
    # Fbend[-2:,:] = 0

    Fbs = F_stretch #+ Fbend ### removed bending for now (unstable)
    return Fbs

#---------------------------------------------------------------------------------------------------
## CELL OPERATIONS FUNCTIONS
#---------------------------------------------------------------------------------------------------

@njit(parallel=True, fastmath=True, nogil=True)
def cc_repulsion(pos_active, cell_types_active):
    """
    Calculate cell-cell repulsion/adhesion forces using spatial binning.
    Jammed adhesion parameters are applied only when both interacting cells
    in the local active slice have type 3.
    Args:
        pos_active: (n_cells, 2) active positions
        cell_types_active: (n_cells,) active cell types aligned to pos_active
    Returns:
        F_cc: (n_cells,2) forces
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
                                    if cell_types_active[i] == 3 and cell_types_active[j] == 3:
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


############### clean this function up! uses same cc_repulsion forces as above.
@njit(fastmath=True, nogil=True, parallel=True)
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
    half_dl_crit_sq = (0.5 * DL_CRIT) * (0.5 * DL_CRIT)
    max_spring_distance = 2.0 * DL_CRIT
    # max_spring_distance_sq = max_spring_distance * max_spring_distance
    
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
    
    # using spatial bins
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
                            
                            if dist_sq < half_dl_crit_sq and dist_sq > 0:
                                dist = np.sqrt(dist_sq)
                                r=dist
                                fmagx = (K_BC_REP * max((DL_CRIT - r, 0)) - K_BC_ADH * max((r - DL_CRIT, 0))) * (dx / r)
                                fmagy = (K_BC_REP * max((DL_CRIT - r, 0)) - K_BC_ADH * max((r - DL_CRIT, 0))) * (dy / r)
                                
                                F_on_cell[i, 0] += fmagx 
                                F_on_cell[i, 1] += fmagy
                                F_on_epi[j, 0] -= fmagx
                                F_on_epi[j, 1] -= fmagy
                            
                            j = nxt_boundary[j]
                
                i = nxt_cells[i]
    
    return F_on_cell, F_on_epi

@njit(parallel=True, fastmath=True, nogil=True)
def ext_force(Xe, points=np.array([94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105])): # [96, 97, 98, 99, 100, 101, 102, 103] maybe change to 8 points?, also write as range()
    """
    Apply external force to the epithelium.
    Force is applied in -x direction to the specified boundary points.
    """
    Ne = len(Xe)
    F_ext = np.zeros((Ne, 2), dtype=float64)
    
    # Apply force K_EXT to each point
    for i in points:
        F_ext[i, 0] = -K_EXT
    
    return F_ext

def calculate_total_ext_force(Xe, Db, points=np.array([94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105])):
    """
    Calculate total force applied and the force per unit length.
    Must use the same points array as ext_force() for consistency!
    
    Returns:
        F_ext_total: Total force applied (K_EXT × number of points)
        total_length: Total length of boundary where force is applied
        force_per_length: Force per unit length (F_ext_total / total_length)
    """
    # Total force is K_EXT applied to each point
    F_ext_total = K_EXT * len(points)
    
    # Calculate total length of the boundary segments at the force points
    fp = Db.dot(Xe)  # forward diffs
    lp = np.linalg.norm(fp, axis=1)[points]  # segment lengths
    total_length = np.sum(lp)
    
    # Force per unit length
    force_per_length = F_ext_total / total_length if total_length > 0 else 0
    
    return F_ext_total, total_length, force_per_length
def tune_F_per_segment(Xe, Db, desired_force_per_length, points=np.array([94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105])):
    """
    Calculate what K_EXT should be to achieve a desired force per unit length
    for the current epithelium geometry.
    Must use the same points array as ext_force() for consistency!
    
    Args:
        Xe: Current epithelium boundary positions
        Db: Forward difference matrix
        desired_force_per_length: Target force per unit length
        points: Indices where force is applied (must match ext_force!)
    
    Returns:
        k_ext_new: The K_EXT value that will give the desired force per unit length
    """
    # Calculate total length at the force application points
    fp = Db.dot(Xe)
    lp = np.linalg.norm(fp, axis=1)[points]
    total_length = np.sum(lp)
    
    # Total force needed = desired_force_per_length * total_length
    total_force_needed = desired_force_per_length * total_length
    
    # K_EXT needed = total_force_needed / number_of_points
    k_ext_new = total_force_needed / len(points)
    
    return k_ext_new

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
def calculate_velocities(F_cc, F_on_cell, F_bone):#, F_intercal=None):

    n_cells = len(F_cc)
    v_active = np.zeros((n_cells, 2), dtype=float64)
    
    for i in prange(n_cells):
        total_force_x = F_cc[i, 0] + F_on_cell[i, 0] + F_bone[i, 0]
        total_force_y = F_cc[i, 1] + F_on_cell[i, 1] + F_bone[i, 1]

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

@njit(fastmath=True, nogil=True)
def motion_update(i, t, pos_i, v_i, cell_type, eta, eta_jammed,
                   mig_rand_x, mig_rand_y, intercal_rand,
                   migration_front):
    """Compute one cell's random motion update"""

    # base noise: jammed vs regular
    if cell_type == 3:
        noise_x, noise_y = eta_jammed[i]
    else:
        noise_x, noise_y = eta[i]

    dx = v_i[0]*DT + noise_x
    dy = v_i[1]*DT + noise_y

    should_migrate = False
    should_intercal = False

    if cell_type == 1:
        if REGULATION_FRONT_FLAG:
            should_migrate = pos_i[0] > migration_front and t >= MIGRATION_DELAY
        else:
            should_migrate = t >= MIGRATION_DELAY
    elif cell_type == 2:
        should_intercal = (np.abs(pos_i[1]) > 0.5 or pos_i[0] > -0.5) and t >= INTERCAL_DELAY # stop at bone (+/- 0.5 sim units)

    # migration
    if should_migrate:
        if MIGRATION_DIRECTION == 'x':
            dx += mig_rand_x[i]
        elif MIGRATION_DIRECTION == 'y':
            dy += mig_rand_y[i]

    # intercalation
    elif should_intercal:
        if pos_i[1] > 0:
            dy -= intercal_rand[i]
        elif pos_i[1] < 0:
            dy += intercal_rand[i]

    return dx, dy

@njit(parallel=True, fastmath=True, nogil=True)
def update_positions(t, pos_active, v_active, Xe, cell_types_active, migration_front=np.inf):
    n_cells = len(pos_active)

    # Precompute all randomness
    eta = np.random.normal(0.0, K_RM*np.sqrt(DT), size=(n_cells, 2))
    eta_jammed = np.random.normal(0.0, K_RM_JAMMING*np.sqrt(DT), size=(n_cells, 2))
    # For drift+diffusion: mean scaled by DT, std by sqrt(DT)
    mig_rand_x = np.random.normal(MIGRATION_STRENGTH*DT, MIGRATION_STD_SCALE*np.sqrt(DT), size=n_cells)
    mig_rand_y = np.random.normal(MIGRATION_STRENGTH*DT, MIGRATION_STD_SCALE*np.sqrt(DT), size=n_cells)
    intercal_rand = np.random.normal(INTERCAL_STRENGTH*DT, INTERCAL_STD_SCALE*np.sqrt(DT), size=n_cells)

    for i in prange(n_cells):
        dx, dy = motion_update(
            i, t, pos_active[i], v_active[i],
            cell_types_active[i], eta, eta_jammed,
            mig_rand_x, mig_rand_y, intercal_rand,
            migration_front
        )
        pos_active[i, 0] += dx
        pos_active[i, 1] += dy

    return pos_active

def cell_operations(t, pos_active, v_active, Xe, Xb, left_wall, reset=False, 
                   step=0, cell_types_active=None):
    """
    Cell operations function that calls individual components.
    
    Args:
        pos_active: (n_cells, 2) array of active cell positions
        v_active: (n_cells, 2) array of active cell velocities  
        Xe: (n_boundary, 2) array of epithelium boundary points
        Xb: (n_bone, 2) array of bone boundary points
        left_wall: x-location of left wall
        reset: Whether to reset cells
        migration_direction: 'x' or 'y' for migration direction
        intercal_pairs: List of (i,j) pairs for intercalation
        cell_types_active: local slice of cell_types aligned to pos_active
        
    Returns:
        F_cc: Cell-cell repulsion forces
        F_on_cell: Total cell-boundary forces
        F_on_epi: Boundary collision forces
        pos_active: Updated cell positions
        v_active: Updated cell velocities
    """
    global pos, v, bone_passage_mask
    # Compute forces for active cells
    F_cc = cc_repulsion(pos_active, cell_types_active)
    F_on_cell, F_on_epi = BC_connect(pos_active, Xe)
    if BONE_FORCES:
        F_bone, bone_passage_mask = bone_interactions(pos_active, Xb, bone_passage_mask, step)
    else:
        F_bone = np.zeros((len(pos_active), 2))

    # Update velocities and positions for active cells
    v_active = calculate_velocities(F_cc, F_on_cell, F_bone)#, F_intercal)
    xb_ymax = Xb[:, 1].max() if BONE_FORCES else 0.0
    xb_ymin = Xb[:, 1].min() if BONE_FORCES else 0.0
    xe_ymax = Xe[:, 1].max()
    xe_ymin = Xe[:, 1].min()
    v_active = extra_damping(pos_active, v_active, left_wall, xb_ymax, xb_ymin, xe_ymax, xe_ymin, BONE_FORCES)
    pos_active = update_positions(t, pos_active, v_active, Xe, cell_types_active, regulation_front)

    if reset:
        # Pass empty array instead of None for Xb when bone is disabled
        Xb_param = Xb if BONE_FORCES and Xb is not None else np.empty((0, 2))
        pos_active, v_active = hard_reset(pos_active, v_active, Xe, Xb_param, check_bone=False)

    return pos_active, v_active, F_cc, F_on_cell, F_on_epi#, F_intercal

def single_iteration(step, t):
    global pos, v, n_daughter, Xe, division_status, Ne, x_cut, T_DORMANT, regulation_front#, intercal_pairs #migrant_cells, jammed_cells, intercal_cells, 
    F_cc_full = np.zeros((CELLS_MAX, 2))
    F_on_cell_full = np.zeros((CELLS_MAX, 2))

    F_on_epi = np.zeros((Ne, 2))

    # Handle cell cycle events (division/death)
    if t >= T_DORMANT:
        # Use directed division if specified
        cell_cycle(div_allowed=True, gradient=GRADIENT, directed_angle=DIRECTED_DIVISION_ANGLE)
    else:
        cell_cycle(div_allowed=False, gradient=GRADIENT)
    
    # Compute active cells after cell_cycle so new daughters are included
    active = np.where(~np.isnan(pos[:, 0]))[0]
    pos_active = pos[active].copy()
    v_active = np.zeros((len(active), 2))

    if JAMMING_ENABLED:
        update_jammed_cells(Xe)
    
    

    cell_types_active = cell_types[active]

    # Initialize and update migration front only if SHIFTING_MIGRATION_FRONT is enabled
    if REGULATION_FRONT_FLAG or GRADIENT=='zone':
        front_delay_step = int(MIGRATION_DELAY / DT)
        front_update_interval = int(1 / DT)  ### Update every 24 hours (if needd, try to make it shift every 12 hours to see if it helps growth)
        
        if step == front_delay_step: 
            regulation_front = np.max(pos_active[:, 0]) - (50 / CONVERSION_FACTOR_UM)
        
        # Update migration front every migration_update_interval steps (after delay)
        if step > front_delay_step and (step - front_delay_step) % front_update_interval == 0:
            regulation_front -= 50 / CONVERSION_FACTOR_UM
    
    # Cell cell opertaions, only hard reset every 100 steps
    if step % 100 == 0: 
        pos_active, v_active, F_cc, F_on_cell, F_on_epi = cell_operations(
            t, pos_active, v_active, Xe, Xb, left_wall, reset=True,
            step=step, cell_types_active=cell_types_active
        )
    else:
        pos_active, v_active, F_cc, F_on_cell, F_on_epi = cell_operations(
            t, pos_active, v_active, Xe, Xb, left_wall, reset=False,
            step=step, cell_types_active=cell_types_active
        )

    # Copy forces to full arrays for data collection
    if step % FRAME_SKIP == 0:
        F_cc_full[active] = F_cc.copy()
        F_on_cell_full[active] = F_on_cell.copy()
        F = F_cc_full + F_on_cell_full
    else:
        F = None

    # Copy updated positions and velocities back to main arrays
    pos[active] = pos_active.copy()
    v = np.zeros((CELLS_MAX, 2))  
    v[active] = v_active

    # Update boundary elasticity and positions
    F_elast = epithelium_elasticity(Xe, Db, DbT, blp0, blm0, dsb)
    F_ext = np.zeros((Ne, 2))
    if EXT_STRESS_FORCE and t > EXT_FORCE_DELAY:
        F_ext = ext_force(Xe)
    Fb = F_elast + F_on_epi + F_ext

    Xe[1:-1] += (Fb[1:-1] / XI) * DT

    # fix coordinates of the first and last points
    Xe[0, 0] = Xe0[0, 0]
    Xe[-1, 0] = Xe0[-1, 0]
    Xe[0, 1] = Xe0[0, 1]
    Xe[-1, 1] = Xe0[-1, 1]

    # Recalculate active cells after cell cycle events (division/death)
    active_after_cycle = np.where(~np.isnan(pos[:, 0]))[0]
    N_active = len(active_after_cycle)
    return F, v, division_status, N_active

#---------------------------------------------------------------------------------------------------
## MAIN SIMULATION FUNCTION
#---------------------------------------------------------------------------------------------------

def run_simulation():
    """Main simulation loop with CSV data collection"""
    global pos, Xe, Ne, Db, blp0, blm0, dsb, regulation_front, cell_types
    N0 = np.where(~np.isnan(pos[:,0]))[0].size ### redundent, calculated at the start
    if EXT_STRESS_FORCE:
        print(f"Starting X dpa, running for {TMAX} more days")
    else:
        print(f"Running simulation until {TMAX} dpa")
    print(f"Initial cell count: {N0}")
    # print(f"Boundary points: {Ne}")
    # Save a snapshot of abm11.py (this file) at the very start
    try:
        import config, inspect
        abm_path = inspect.getsourcefile(run_simulation)
        if abm_path and os.path.exists(abm_path):
            dst_abm = os.path.join(OUTPUT_DIR, 'abm_snapshot.py')
            shutil.copyfile(abm_path, dst_abm)
            print(f"abm snapshot saved to {dst_abm}")
    except Exception as e:
        print(f"Warning: could not save abm snapshot: {e}")
    
    if ALLOW_SOFTENING:
        print('Local Softening Enabled.')
    else:
        print('Local Softening Disabled.')
    # Initialize cases - jamming first, then migration/intercalation
    if JAMMING_ENABLED:
        update_jammed_cells(Xe)
        print(f"Jammming enabled.")
    
    if MIGRATION_ENABLED:
        init_migration_cells(CELLS_MAX)
        print(f"Migration enabled: {int(N0 * MIGRATION_PERCENT / 100.0)} cells migrating ({MIGRATION_PERCENT}%) in {MIGRATION_DIRECTION} direction")
        
    if INTERCALATION_ENABLED:
        init_intercal_cells()
        print(f"Intercalation enabled: {int(N0 * INTERCAL_PERCENT / 100.0)} cells intercalating ({INTERCAL_PERCENT}%)")
    
    if DIRECTED_DIVISION_ANGLE is not None:
        print(f"Directed division enabled: angle = {DIRECTED_DIVISION_ANGLE:.2f} radians")
    else:
        print("Directed division disabled: using uniform random division")
    
    # Initialize animation manager
    animation_manager = AnimationManager(OUTPUT_DIR, VIDEO_PARAMS, VIDEO_FLAG)
    
    # Set up signal handling for clean shutdown
    setup_signal_handler(animation_manager) ### a little broken
    
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
    morphometrics_data = {'area_growth_region': [], 'perimeter': [], 'AR_whole_limb': [], 'AR_outgrowth': [], 'ellipticity': [], 'roundness': [], 'a': [], 'b': [], 'volume_fraction': []}
    
    # Additional cell status data
    phase_clocks_data = []
    cycle_phases_data = []
    migrant_cells_data = []
    intercal_cells_data = []
    jammed_cells_data = []
    regulation_front_history = []
    
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
    migrant_cells_data.append((cell_types == 1).copy())
    intercal_cells_data.append((cell_types == 2).copy())
    jammed_cells_data.append((cell_types == 3).copy())
    regulation_front_history.append(regulation_front)
    
    start = time.time()
    last_time = start  # For step time calculation
    ext_force_printed = False  # Flag to print external force only once
    ext_force_paused = False  # Flag to pause animation only once at force application
    
    # Main simulation loop
    print(f'Running simulation...')
    x0 = None
    for step, t in enumerate(np.arange(0, TMAX, DT)):
        # Use the refactored single_iteration function
        F, v, div_status, N_active = single_iteration(step, t)

        if PRINT_STEPS_FLAG and step % PRINT_STEPS_INTERVAL == 0:
            print(f"t = {t:.2f}, Step {step}/{STEPS_TOTAL}, cells: {N_active}")
        
        # Print total external force at EXT_FORCE_DELAY
        if EXT_STRESS_FORCE and not ext_force_printed and t >= EXT_FORCE_DELAY:
            total_force, total_length, force_per_length = calculate_total_ext_force(Xe, Db)
            print(f"\n=== External Force Applied at t={t:.4f} ===")
            print(f"Total external force: {total_force:.6f}")
            print(f"Total boundary length: {total_length:.6f}")
            print(f"Force per unit length: {force_per_length:.6f}\n")
            ext_force_printed = True
            x0 = Xe[100,0] # initial x before applied force
            
            # Pause animation at force application with updated title
            if VIDEO_FLAG and not ext_force_paused:
                animation_manager.animate_frame(step, t, pos, Xe, Xb, pos0, cycle_phases,
                                            kb_vals=kb_vals,
                                            migrant_cells=(cell_types == 1),
                                            intercal_cells=(cell_types == 2),
                                            jammed_cells=(cell_types == 3),
                                            regulation_front=regulation_front,
                                            title_suffix=' (force applied)')
                animation_manager.pause_animation(duration_seconds=1.0)
                ext_force_paused = True
        
        if step % FRAME_SKIP == 0 and step > 0:

            if VIDEO_FLAG:
                animation_manager.animate_frame(step, t, pos, Xe, Xb, pos0, cycle_phases,
                                            kb_vals=kb_vals,
                                            migrant_cells=(cell_types == 1),
                                            intercal_cells=(cell_types == 2),
                                            jammed_cells=(cell_types == 3),
                                            regulation_front=regulation_front)
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

            # save current morphometrics
            area_growth_region_t, perimeter_t, AR_whole_limb_t, AR_outgrowth_t, ellipticity_t, roundness_t, a_t , b_t, volume_fraction_t = morphometrics(Xe, pos=pos, x_cut=x_cut)
            morphometrics_data['area_growth_region'].append(area_growth_region_t)
            morphometrics_data['perimeter'].append(perimeter_t)
            morphometrics_data['AR_whole_limb'].append(AR_whole_limb_t)
            morphometrics_data['AR_outgrowth'].append(AR_outgrowth_t)
            morphometrics_data['ellipticity'].append(ellipticity_t) # outgrowth region
            morphometrics_data['roundness'].append(roundness_t) # outgrowth region
            morphometrics_data['a'].append(a_t) # outgrowth region
            morphometrics_data['b'].append(b_t) # outgrowth region
            morphometrics_data['volume_fraction'].append(volume_fraction_t)

            # Save cell status data
            phase_clocks_data.append(phase_clocks.copy())
            cycle_phases_data.append(cycle_phases.copy())
            migrant_cells_data.append((cell_types == 1).copy())
            intercal_cells_data.append((cell_types == 2).copy())
            jammed_cells_data.append((cell_types == 3).copy())
            regulation_front_history.append(regulation_front)

            current_time = time.time()
            avg_step_time = (current_time - last_time) / FRAME_SKIP
            step_times.append(avg_step_time)

            last_time = current_time
    elapsed = time.time() - start
    readable_time = time.strftime('%H:%M:%S', time.gmtime(elapsed))
    print(f'time elapsed: {readable_time}')
    print(f"Number of deaths: {n_deaths}")
    # if EXT_STRESS_FORCE:

        # print(f'Work Done:')
    
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
        cycle_phases_first = np.zeros_like(cycle_phases)  # All cells start in G1
        cycle_phases_last = cycle_phases.copy()
                
    if EXT_STRESS_FORCE:
        xf = Xe[100,0] # final x after applied force
        deformation = (xf - x0)
    else:
        deformation = None
    # else:
    #     deformation = np.abs(Xe[-1,0] - Xe0[-1,0])
        print(f"Deformation: {deformation}")

    Xe_growth = Xe[Xe[:, 0] > x_cut]
    if len(Xe_growth) > 0:
        coefficients_growth = coefficients(Xe_growth, n=5, type='data', rotate=True) ### # chebyshev coefficients
    else:
        coefficients_growth = None
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
        'regulation_front_history': np.array(regulation_front_history),
        'n_deaths': n_deaths,
        'n_daughter': n_daughter,
        'kb_vals': kb_vals,
        'elapsed': readable_time,
        'coefficients_growth': coefficients_growth,
        'deformation': deformation,
        'final_cell_count': N_active,
        'morphometrics_time_series': morphometrics_data,
        'morphometrics_final': morphometrics(Xe, pos=pos, x_cut=x_cut),
        'Gphase': Gphase,
        'Mphase': Mphase,
        'cell_count': cell_count,
        'step_times': step_times,
        'Xb': Xb,
        'Xe0': Xe0,
        'Xe_final': Xe,
        'Xe_growth': Xe_growth,
        'pos0': pos0,
        'T_DORMANT': T_DORMANT,
        'TMAX': TMAX,
        'x_cut': x_cut,
        'N0': N0,
        'FRAME_SKIP': FRAME_SKIP,
        'OUTPUT_DIR': OUTPUT_DIR,
    }

    
    # Collect actual runtime config parameters
    config_params = {
        'Time Parameters': {
            'DT': config.DT,
            'TMAX': config.TMAX,
            'STEPS_TOTAL': config.STEPS_TOTAL,
        },
        'Physical Parameters': {
            'CONVERSION_FACTOR_UM': config.CONVERSION_FACTOR_UM,
            'DL_CRIT': config.DL_CRIT,
            'XI': config.XI,
            'K_BC_REP': config.K_BC_REP,
            'K_BC_ADH': config.K_BC_ADH,
            'K_CC_REP': config.K_CC_REP,
            'K_CC_ADH': config.K_CC_ADH,
            'K_BONE': config.K_BONE,
            'K_RM': config.K_RM,
            'KBEND': config.KBEND,
            'KB_MAX': config.KB_MAX,
            'KB_MID': config.KB_MID,
            'KB_MIN': config.KB_MIN,
        },
        'Bone/Softening': {
            'BONE_ENABLED': config.BONE_ENABLED,
            'BONE_FORCES': config.BONE_FORCES,
            'ALLOW_SOFTENING': config.ALLOW_SOFTENING,
            'BONE_PHASE_PERCENT': config.BONE_PHASE_PERCENT,
        },
        'Cell Cycle': {
            'T_DORMANT': config.T_DORMANT,
            'KDEATH': config.KDEATH,
            'KDIV': config.KDIV,
            'M_LENGTH': config.M_LENGTH,
            'G_LENGTH': config.G_LENGTH,
            'G_LENGTH_MAX': config.G_LENGTH_MAX,
            'G_LENGTH_MIN': config.G_LENGTH_MIN,
            'GRADIENT': config.GRADIENT,
            'DIRECTED_DIVISION_ANGLE': config.DIRECTED_DIVISION_ANGLE,
        },
        'Migration': {
            'MIGRATION_ENABLED': config.MIGRATION_ENABLED,
            'MIGRATION_PERCENT': config.MIGRATION_PERCENT,
            'MIGRATION_DELAY': config.MIGRATION_DELAY,
            'WHICH_MIGRATION': config.WHICH_MIGRATION,
            'MIGRATION_DIRECTION': config.MIGRATION_DIRECTION,
            'MIGRATION_STRENGTH': config.MIGRATION_STRENGTH,
            'MIGRATION_STD_SCALE': config.MIGRATION_STD_SCALE,
            'REGULATION_FRONT_FLAG': config.REGULATION_FRONT_FLAG,
        },
        'Intercalation': {
            'INTERCALATION_ENABLED': config.INTERCALATION_ENABLED,
            'INTERCAL_PERCENT': config.INTERCAL_PERCENT,
            'INTERCAL_DELAY': config.INTERCAL_DELAY,
            'INTERCAL_STRENGTH': config.INTERCAL_STRENGTH,
            'INTERCAL_STD_SCALE': config.INTERCAL_STD_SCALE,
        },
        'Jamming': {
            'JAMMING_ENABLED': config.JAMMING_ENABLED,
            'JAMMING_ZONE_WIDTH': config.JAMMING_ZONE_WIDTH,
            'K_CC_REP_JAMMING': config.K_CC_REP_JAMMING,
            'K_CC_ADH_JAMMING': config.K_CC_ADH_JAMMING,
            'K_RM_JAMMING': config.K_RM_JAMMING,
        },
        'External Stress': {
            'EXT_STRESS_FORCE': config.EXT_STRESS_FORCE,
            'EXT_FORCE_DELAY': config.EXT_FORCE_DELAY,
            'K_EXT': config.K_EXT,
            'FORCE_PER_UNIT_LENGTH': config.FORCE_PER_UNIT_LENGTH,
        },
        'Output': {
            'VIDEO_FLAG': config.VIDEO_FLAG,
            'FRAME_SKIP': config.FRAME_SKIP,
            'OUTPUT_DIR': config.OUTPUT_DIR,
        }
    }
    
    # Add config_params to data_dict for complete reproducibility
    data_dict['config_params'] = config_params
    
    # Save simulation data
    save_files(data_dict, config_params, save_data_dict=SAVE_DATA_DICT, save_figures=SAVE_FIGURES, )
    print("Simulation finished and data saved.")
    
    return data_dict

if __name__ == "__main__":
    if PROFILING_FLAG:
        # Run with profiling
        profiling(run_simulation, OUTPUT_DIR)
    else:
        run_simulation()