"""
Limb Regeneration Simulation - Agents Based Model
Ansa Brews-Smith, May 2025
Copos Lab, Northeastern University

Toy Model: Circlular Distribution of Cells

    CASES (can each be repeated with and without epithelium)
    - 'RM_RD': Random Motion Term (eta) and Random Division Angle
    - 'RM_DD_0': Random Motion Term (eta) and Division towards 0 radians with std pi/6
    - 'RM_DD_pi_2': Random Motion Term (eta) and Division towards pi/2 radians with std pi/6
    - 'DMx_RD': Random Motion Term (eta), X% of cells have directed motion in the +x direction, and Random Division Angle
    - 'DMy_RD': Random Motion Term (eta), X% of cells have directed motion in the +y direction, and Random Division Angle
    - 'INTERCAL': Random Motion Term (eta), Random Division Angle, and Pairwise Intercalation Force for X% of cells

"""

import warnings
warnings.filterwarnings('ignore')

import matplotlib
matplotlib.use('Agg')  # Set backend before importing pyplot
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.io as sio 
import time
import os
# os.environ['NUMBA_THREADING_LAYER'] = 'tbb'  # Use TBB for better threading
from numba import njit, prange, set_num_threads
from numpy import int64, float64
from scipy.sparse import spdiags

# Import configuration and utilities
from config import *
from utils.animations import AnimationManager
from utils.data_io import save_files
from utils.profiler import profiling
from utils.signal_handling import setup_signal_handler
from utils.post_process import morphometrics, cycle_plot, runtime_plot, density_heatmap, trajectory_plot, metric_beeswarm, MSD_plot, multi_trial_cycle_plot, multi_trial_metrics_plot
time_start = time.time()
# KDIV = 10
CASES = [
    'RM_RD',         # Random Motion Term (eta) and Random Division Angle
    'RM_DD_0',       # Random Motion Term (eta) and Division towards 0 radians with std pi/6
    'RM_DD_pi_2',    # Random Motion Term (eta) and Division towards pi/2 radians with std pi/6
    'DMx_RD',        # Random Motion Term (eta), X% of cells have directed motion in the +x direction, and Random Division Angle
    'DMy_RD',        # Random Motion Term (eta), X% of cells have directed motion in the +y direction, and Random Division Angle
    'INTERCAL'       # Random Motion Term (eta), Random Division Angle, and Pairwise Intercalation Force for X% of cells
]

METRIC_NAMES = ['area', 'perimeter', 'aspect_ratio', 'ellipticity', 'roundness', 'a', 'b', 'volume_fraction']    

pairs_picked = False # have intercal pairs been picked?
intercal_pairs = None

# CASES = ['INTERCAL', 'DMx_RD', 'DMy_RD']
# TMAX= 1.0
# XMIN, XMAX = -2.0, 2.0
# YMIN, YMAX = -2.0, 2.0
# KDIV = 1.45
# KDEATH = 0.05
# — Load initial cell positions —

    # Find indices of boundary points to the right of x_cut_val for interaction
# Gillespie clock setup
# Initialize t_next to infinity. This prevents any cell cycle events
# from being scheduled or processed until explicitly allowed after T_CELL_CYCLE_START.

def intial_cell_positions(N0=50, r_max=0.1):
    rng = np.random.RandomState(26)# For reproducibility
    pos0 = np.zeros((N0, 2))
    for i in range(N0):
        theta = 2*np.pi*rng.random()
        r = r_max*np.random.random()
        x, y = r*np.cos(theta), r*np.sin(theta)
        pos0[i, 0], pos0[i, 1] = x, y

    return pos0, N0
def initial_cell_positions_optimimal_packing(N0=50, r_max=0.1):
    """
    Generate optimal hexagonal close packing of cells within a circular domain.
    
    Args:
        N0: Target number of cells
        r_max: Maximum radius of the domain
    
    Returns:
        pos0: (N, 2) array of cell positions
        N_actual: Actual number of cells placed
    """
    cell_radius = DL_CRIT / 2  # Radius of each cell
    
    # Hexagonal lattice spacing (center-to-center distance)
    lattice_spacing = DL_CRIT  # Cells just touching
    
    # Create hexagonal lattice
    positions = []
    
    # Calculate how many rows/columns we need
    max_extent = r_max - cell_radius  # Keep cell centers within domain
    n_rows = int(2 * max_extent / (lattice_spacing * np.sqrt(3)/2)) + 1
    n_cols = int(2 * max_extent / lattice_spacing) + 1
    
    # Generate hexagonal grid
    for row in range(-n_rows//2, n_rows//2 + 1):
        for col in range(-n_cols//2, n_cols//2 + 1):
            # Standard hexagonal lattice coordinates
            if row % 2 == 0:  # Even rows
                x = col * lattice_spacing
            else:  # Odd rows (offset by half spacing)
                x = (col + 0.5) * lattice_spacing
            
            y = row * lattice_spacing * np.sqrt(3) / 2
            
            # Check if cell center is within the domain
            distance_from_center = np.sqrt(x**2 + y**2)
            if distance_from_center <= max_extent:
                positions.append([x, y])
    
    positions = np.array(positions)
    
    # If we have more positions than requested, select the N0 closest to center
    if len(positions) > N0:
        distances = np.sqrt(positions[:, 0]**2 + positions[:, 1]**2)
        closest_indices = np.argsort(distances)[:N0]
        positions = positions[closest_indices]
    
    # If we have fewer positions than requested, add additional cells
    elif len(positions) < N0:
        # Fill remaining spots with optimized random placement
        n_missing = N0 - len(positions)
        for _ in range(n_missing * 10):  # Try up to 10x the missing count
            if len(positions) >= N0:
                break
                
            # Generate random candidate position
            theta = 2 * np.pi * np.random.random()
            r = max_extent * np.random.random()
            candidate = np.array([r * np.cos(theta), r * np.sin(theta)])
            
            # Check minimum distance to existing cells
            if len(positions) > 0:
                min_dist = np.min(np.sqrt(np.sum((positions - candidate)**2, axis=1)))
                if min_dist >= DL_CRIT:  # Ensure no overlap
                    positions = np.vstack([positions, candidate])
            else:
                positions = np.array([candidate])
    
    N_actual = len(positions)
    
    # Add small random perturbation to break perfect symmetry (more realistic)
    perturbation_strength = DL_CRIT * 0.05  # 5% of cell diameter
    perturbations = np.random.normal(0, perturbation_strength, positions.shape)
    positions += perturbations
    
    # Ensure no cells moved outside the domain
    distances = np.sqrt(positions[:, 0]**2 + positions[:, 1]**2)
    too_far = distances > max_extent
    if np.any(too_far):
        # Scale back positions that went too far
        scale_factors = max_extent / distances[too_far]
        positions[too_far] *= scale_factors[:, np.newaxis]
    
    return positions, N_actual
    

# Initialize intercalation pairs for all cells
# def init_intercal_pairs(pos_active, axis=0):
#     n_cells = len(pos_active)
#     if n_cells % 2 != 0:
#         # Remove one cell if odd number
#         pos_active = pos_active[:-1]
#         n_cells = len(pos_active)
    
#     # Sort cells by position along specified axis
#     sorted_indices = np.argsort(pos_active[:, axis])
#     intercal_pairs = []
    
#     # Pair consecutive cells
#     for i in range(0, n_cells, 2):
#         intercal_pairs.append((sorted_indices[i], sorted_indices[i+1]))
        
#     return intercal_pairs

def init_intercal_pairs(pos_active, axis=1):
    """
    Pair each cell with y>0 to one with y<0. 
    If there are more in one group, the extras are dropped.
    
    Args:
        pos_active: (n_cells, 2) array of positions.
    Returns:
        intercal_pairs: list of (i,j) index pairs into pos_active.
    """
    n = len(pos_active)
    indices = np.arange(n)
    
    # split into above and below
    above = indices[pos_active[:, axis] > 0]
    below = indices[pos_active[:, axis] < 0]
    
    # shuffle each group to randomize pairings
    np.random.shuffle(above)
    np.random.shuffle(below)
    
    # only pair as many as the smaller group
    m = min(len(above), len(below))
    above = above[:m]
    below = below[:m]
    
    # zip them into pairs
    intercal_pairs = [(int(above[i]), int(below[i])) for i in range(m)]
    return intercal_pairs

# np.random.seed(None) # set seed back to random
# def initial_cell_2():
#     pos0 = np.zeros((3, 2))
#     pos0[0, 0], pos0[0, 1] = 0.0, 0.0
#     pos0[1, 0], pos0[1, 1] = 0.0, 0.5
#     pos0[2, 0], pos0[2, 1] = 0.0, -0.5
#     return pos0

t_next = np.inf 
# pos0, N0 = intial_cell_positions(N0=100, r_max=0.5)
pos0, N0 = initial_cell_positions_optimimal_packing(N0=75, r_max=0.5)

# N0 = 3
# pos0 = initial_cell_2() 
CELLS_MAX = 35 * N0

v = np.zeros((CELLS_MAX, 2))
pos = np.full((CELLS_MAX, 2), np.nan)
pos[:N0, :] = pos0

division_status = np.zeros((CELLS_MAX,), dtype=bool)
death_indices = np.zeros((CELLS_MAX,), dtype=bool)

# — Alive mask & daughter count —
n_daughter = 0

def cell_cycle(active, t_next_current_val, div_angle=None):
    """Handle cell division and death events using Gillespie algorithm"""
    global pos, n_daughter, division_status, death_indices

    n_alive = len(active)
    if n_alive == 0:
        return pos, division_status, np.inf # If no active cells, next event is never
        
    # Choose event type
    if np.random.rand() < KDIV/(KDIV+KDEATH):
        # Division
        free_slots = np.where(np.isnan(pos[:,0]))[0]
        if len(free_slots) > 0:
            mom = active[np.random.randint(n_alive)]

            if mom is not None:
                dau = free_slots[0]
                if div_angle==None:
                    ang = 2*np.pi*np.random.rand()
                else:
                    ang = np.random.normal(loc=div_angle, scale=np.pi/6)
                pos[dau, 0] = pos[mom, 0] + OFFSET*np.cos(ang)
                pos[dau, 1] = pos[mom, 1] + OFFSET*np.sin(ang)
                division_status[dau] = True
                n_daughter += 1
    else:
        # Death
        victim = active[np.random.randint(n_alive)]
        death_indices[victim] = True
        pos[victim, 0] = np.nan
        pos[victim, 1] = np.nan

    # Schedule next event
    active = np.where(~np.isnan(pos[:,0]))[0] # Re-check active cells after event
    n_alive = len(active)
    if n_alive == 0:
        new_t_next = np.inf
    else:
        rate0 = (KDIV+KDEATH)*n_alive
        new_t_next = t_next_current_val + np.random.exponential(1.0/rate0)
        
    return pos, division_status, new_t_next

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
                                    fx = (K_CC_REP * max((DL_CRIT - r, 0)) - K_CC_ATTR * max((r - DL_CRIT, 0))) * (dx / r) ### 0.1 -> 0.01
                                    fy = (K_CC_REP * max((DL_CRIT - r, 0)) - K_CC_ATTR * max((r - DL_CRIT, 0))) * (dy / r) ### 0.1 -> 0.01
                                    F_cc[i, 0] -= fx
                                    F_cc[i, 1] -= fy
                                    F_cc[j, 0] += fx
                                    F_cc[j, 1] += fy
                            j = nxt[j]
                i = nxt[i]
    return F_cc

@njit(parallel=True, fastmath=True, nogil=True)
def intercal_force(pos_active, intercal_pairs):
    n_cells = len(pos_active)
    F_intercal = np.zeros((n_cells, 2), dtype=float64)
    for k in range(len(intercal_pairs)):
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
def calculate_velocities(F_cc, F_intercal):#, F_on_cell, F_on_epid, F_bone):
    """
    Calculate velocities from total forces.
    
    Args:
        F_cc: Cell-cell forces
        F_on_cell: Cell-boundary repulsion forces
        F_on_cell_spring: Cell-boundary spring forces
        F_bone: Bone interaction forces
        xi: Drag coefficient
        
    Returns:
        v_active: Updated velocities
    """
    n_cells = len(F_cc)
    v_active = np.zeros((n_cells, 2), dtype=float64)
    
    for i in prange(n_cells):
        total_force_x = F_cc[i, 0] + F_intercal[i, 0]# + F_on_cell[i, 0] + F_bone[i, 0] + F_on_epid[i, 0] 
        total_force_y = F_cc[i, 1] + F_intercal[i, 1]# + F_on_cell[i, 1] + F_bone[i, 1] + F_on_epid[i, 1] 

        v_active[i, 0] = total_force_x / XI
        v_active[i, 1] = total_force_y / XI
    
    return v_active

@njit(parallel=True, fastmath=True, nogil=True)
def update_positions(pos_active, v_active, migrant_cells=None, migration=None):
    """
    Update cell positions with velocity and random noise.
    
    Args:
        pos_active: Current cell positions
        v_active: Cell velocities
        dt: Time step
        
    Returns:
        pos_active: Updated positions
    """
    n_cells = len(pos_active)
    if migration == 'x':
        eta_migration = 1000 * np.random.normal(loc=1, scale=np.sqrt(DT), size=(n_cells, 2)) #* np.random.random((n_cells, 2)) 
        eta_migration[:, 1] = 0 # no random motion in y direction
    elif migration == 'y':
        eta_migration = 1000 * np.random.normal(loc=1, scale=np.sqrt(DT), size=(n_cells, 2)) #* np.random.random((n_cells, 2))
        eta_migration[:, 0] = 0 # no random motion in x direction
    else:
        eta_migration = np.zeros((n_cells, 2))
    
    eta = 1000 * np.random.normal(loc=0, scale=np.sqrt(DT), size=(n_cells, 2)) # brownian motion     #-6 + 12*np.random.random((n_cells, 2))  # Random noise

    for i in prange(n_cells):
        if migrant_cells is not None and migrant_cells[i] and not death_indices[i]:
            pos_active[i, 0] += v_active[i, 0]*DT + eta_migration[i, 0]*DT
            pos_active[i, 1] += v_active[i, 1]*DT + eta_migration[i, 1]*DT
        else:
            pos_active[i, 0] += v_active[i, 0]*DT + eta[i, 0]*DT #### random noise on/off
            pos_active[i, 1] += v_active[i, 1]*DT + eta[i, 1]*DT #### random noise on/off

    return pos_active

def cell_operations(pos_active, v_active, intercal_pairs=None, migration=None, migrant_cells=None, t=0):#, Xe, Xb, left_wall, reset=False):
    """
    Cell operations function that calls individual components.
    This version splits the operations for better profiling while maintaining performance.
    
    Args:
        pos_active: (n_cells, 2) array of active cell positions
        v_active: (n_cells, 2) array of active cell velocities  
        Xe: (n_boundary, 2) array of epithelium boundary points
        Xb: (n_bone, 2) array of bone boundary points
        left_wall: x-location of left wall
        
    Returns:
        F_cc: Cell-cell repulsion forces
        F_on_cell: Total cell-boundary forces
        F_on_epid: Boundary collision forces
        pos_active: Updated cell positions
        v_active: Updated cell velocities
    """
    
    n_active = len(pos_active)
    F_cc = cc_repulsion(pos_active)
    # F_cc = np.zeros((n_active, 2)) ##### DEBUG
    if intercal_pairs and len(intercal_pairs) > 0:
        F_intercal = intercal_force(pos_active, intercal_pairs)
    else:
        F_intercal = np.zeros((n_active, 2))
    # F_on_cell, F_on_epid = BC_connect(pos_active, Xe)
    F_on_cell, F_on_epid = None, None
    v_active = calculate_velocities(F_cc, F_intercal)#, F_on_cell, F_on_epid, F_bone)
    pos_active = update_positions(pos_active, v_active, migrant_cells=migrant_cells, migration=migration)
    
    return F_cc, F_intercal, F_on_cell, F_on_epid, pos_active, v_active

def single_iteration(step, intercal_pairs, migrant_cells, t, case='RM_RD'):
    """A single iteration of the simulation using consolidated operations"""
    global pos, n_daughter, Xe, division_status, Ne, t_next, x_cut, pairs_picked

    # Get active cells
    active = np.where(~np.isnan(pos[:, 0]))[0]
    pos_active = pos[active].copy()
    
    # Initialize migrant cells active mask
    migrant_cells_active = None
    if migrant_cells is not None:
        migrant_cells_active = migrant_cells[active]

    active0 = active.copy() # active at the start of the step
    # intialize control variables
    # div_angle = None
    # migration = None
    # intercal = None

    # handle cases!
    if case == 'RM_RD':
        div_angle = None
        migration = None
    elif case == 'RM_DD_0':
        div_angle = 0
        migration = None
    elif case == 'RM_DD_pi_2':
        div_angle = np.pi/2
        migration = None
    elif case == 'DMx_RD':
        div_angle = None
        migration = 'x'
    elif case == 'DMy_RD':
        div_angle = None
        migration = 'y'
    elif case == 'INTERCAL':
        div_angle = None
        migration = None

    # Pre-allocate force arrays for data collection
    F_cc_full = np.zeros((CELLS_MAX, 2))
    F_intercal_full = np.zeros((CELLS_MAX, 2))
    # F_on_cell_full = np.zeros((CELLS_MAX, 2))
    # F_on_epid = np.zeros((Ne, 2))

    # --- Handle cell cycle events (division/death) ---
    if t >= T_DORMANT:
        if t_next == np.inf:
            n_alive_at_start = len(active)
            if n_alive_at_start > 0:
                rate_at_start = (KDIV + KDEATH) * n_alive_at_start
                t_next = T_DORMANT + np.random.exponential(1.0 / (rate_at_start+1e-10))
            else:
                t_next = np.inf

        while t >= t_next:
            if len(active) == 0:
                t_next = np.inf
                break
            pos, division_status, t_next = cell_cycle(active, t_next, div_angle=div_angle)
            active = np.where(~np.isnan(pos[:, 0]))[0]

    # --- Update intercalation pairs for new cells ---
    if case == 'INTERCAL':
        # Find new cells added in this iteration
        active_new = np.where(~np.isnan(pos[:, 0]))[0]
        new_cells = np.setdiff1d(active_new, active0)
        
        if len(new_cells) > 0:
            # Get all active cells
            active_all = np.where(~np.isnan(pos[:, 0]))[0]
            pos_active_all = pos[active_all].copy()
            
            # Re-initialize all pairs including new cells
            intercal_pairs = init_intercal_pairs(pos_active_all, axis=1)
            
            # Map indices back to global indices
            intercal_pairs = [(active_all[i], active_all[j]) for i, j in intercal_pairs]

    # --- Extract active cells ---
    active = np.where(~np.isnan(pos[:, 0]))[0]
    pos_active = pos[active].copy()
    v_active = np.zeros((len(active), 2))

    # --- Call consolidated function for all parallel operations ---
    if step % 10 == 0:
        F_cc, F_intercal, F_on_cell, F_on_epid, pos_active, v_active = cell_operations(
            pos_active, v_active, intercal_pairs=intercal_pairs, migrant_cells=migrant_cells_active, migration=migration, t=t#, Xe, Xb, left_wall, reset=True
        )
    else:
         F_cc, F_intercal, F_on_cell, F_on_epid, pos_active, v_active = cell_operations(
            pos_active, v_active, intercal_pairs=intercal_pairs, migrant_cells=migrant_cells_active, migration=migration, t=t#, Xe, Xb, left_wall, reset=False
        )
    # --- Copy updated positions and velocities back to main arrays ---
    pos[active] = pos_active.copy()
    v = np.zeros((CELLS_MAX, 2))  
    v[active] = v_active

    
    # --- Copy forces to full arrays for data collection ---
    if step % FRAME_SKIP == 0:
        F_cc_full[active] = F_cc.copy()
        if intercal_pairs and len(intercal_pairs) > 0:
            F_intercal_full[active] = F_intercal.copy()
        # F_on_cell_full[active] = F_on_cell.copy()
        F = F_intercal_full + F_cc_full       #+ F_on_cell_full # Include scaled F_on_epid
    else:
        F = None


    # --- Update boundary elasticity and positions ---
    # kb_vals = KB * np.ones(Ne)
    # kb_vals[soft_idx] /= SOFT_FACTOR

    # F_elast = epithelium_elasticity(Xe, Db, DbT, blp0, blm0, dsb, kb_vals)
    # Fb = F_elast + F_on_epid

    # Only update right part of the boundary
    # update_mask = Xe[1:-1, 0] >= 0.6
    # Xe[1:-1][update_mask] += (Fb[1:-1][update_mask] / XI) * DT

    # --- Combine forces for return (for data collection) ---

    # --- Count active cells ---
    N_active = len(active)  # Use active array length for consistency

    return F, v, division_status, N_active, intercal_pairs

#---------------------------------------------------------------------------------------------------
## MAIN SIMULATION FUNCTION
#---------------------------------------------------------------------------------------------------

def run_simulation(intercal_pairs=None, migrant_cells=None, boundary=False, case='RM_RD', OUTPUT_DIR=OUTPUT_DIR):
    """Main simulation loop with CSV data collection"""
    global pos, Xe, Ne, Db, blp0, blm0, dsb, soft_idx, t_next, T_DORMANT, pairs_picked
    Xb = None
    Xe0 = None
    Xe = None
    soft_idx = None
    x_cut = None
    # Reset t_next for a fresh simulation run if run_simulation is called multiple times
    t_next = np.inf 
    print(f"Running simulation for {STEPS_TOTAL} steps")
    print(f"Initial cell count: {np.where(~np.isnan(pos[:,0]))[0].size}")
    # print(f"Boundary points: {Ne}")
    
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
    metrics_dict = {}
    metrics_time_series = {
        'time': [],
        'area': [],
        'perimeter': [],
        'aspect_ratio': [],
        'ellipticity': [],
        'roundness': [],
        'a': [],
        'b': [],
        'volume_fraction': []
    }
    times = []
    cell_count = []
    step_times = [] # for timing runtime
    
    # Save initial state
    positions.append(pos.copy())
    forces.append(np.zeros_like(pos))
    velocities.append(np.zeros_like(pos))
    # if boundary:
        # boundaries.append(Xe.copy())
    divisions.append(division_status.copy())
    deaths.append(death_indices.copy())
    cell_count.append(N0)
    times.append(0.0)
    metrics_time_series['time'].append(0.0)
    area, perimeter, aspect_ratio, ellipticity, roundness, a , b, volume_fraction = morphometrics(Xe, pos=pos0)
    metrics_time_series['area'].append(area)
    metrics_time_series['perimeter'].append(perimeter)
    metrics_time_series['aspect_ratio'].append(aspect_ratio)
    metrics_time_series['ellipticity'].append(ellipticity)
    metrics_time_series['roundness'].append(roundness)
    metrics_time_series['a'].append(a)
    metrics_time_series['b'].append(b)
    metrics_time_series['volume_fraction'].append(volume_fraction)
    
    start = time.time()
    last_time = start  # For step time calculation
    
    # Main simulation loop
    # Initialize intercalation pairs for INTERCAL case
    if case == 'INTERCAL':
        active = np.where(~np.isnan(pos[:,0]))[0]
        pos_active = pos[active].copy()
        intercal_pairs = init_intercal_pairs(pos_active, axis=1)
        # Map to global indices
        intercal_pairs = [(active[i], active[j]) for i, j in intercal_pairs]
    else:
        intercal_pairs = None
    
    for step, t in enumerate(np.arange(0, TMAX, DT)):
        # Use the refactored single_iteration function

        F, v, div_status, N_active, intercal_pairs = single_iteration(
            step, intercal_pairs, migrant_cells, t, case=case
        )
        
        if step % FRAME_SKIP == 0 and step > 0:
            if VIDEO_FLAG:
                intercal_cells = None
                if case == 'INTERCAL':
                    # Create mask for intercalation cells
                    intercal_cells = np.zeros(CELLS_MAX, dtype=bool)
                    for pair in intercal_pairs:
                        intercal_cells[pair[0]] = True
                        intercal_cells[pair[1]] = True
                
                animation_manager.animate_frame(step, t, pos, Xe, Xb, pos0, 
                                          division_status, soft_idx, FRAME_SKIP, boundary=boundary, x_bounds=(XMIN, XMAX), y_bounds=(YMIN, YMAX), forces=None, migrant_cells=migrant_cells, intercal_cells=intercal_cells)
                animation_manager.animate_density_heatmap(step, t, pos, Xe, Xb, SOFT_RANGE, x_cut,
                                                    bin_size=0.1, frame_skip=FRAME_SKIP, boundary=boundary, x_bounds=(XMIN, XMAX), y_bounds=(YMIN, YMAX))
            # Save this timestep's data
            positions.append(pos.copy())
            forces.append(F.copy() if F is not None else np.zeros_like(pos))
            velocities.append(v.copy())
            # boundaries.append(Xe.copy())
            divisions.append(division_status.copy())
            deaths.append(death_indices.copy())
            times.append(t)
            cell_count.append(N_active)

            current_time = time.time()
            avg_step_time = (current_time - last_time) / FRAME_SKIP
            step_times.append(avg_step_time)

            area, perimeter, aspect_ratio, ellipticity, roundness, a , b, volume_fraction = morphometrics(Xe, pos=pos)
            metrics_time_series['time'].append(t)
            metrics_time_series['area'].append(area)
            metrics_time_series['perimeter'].append(perimeter)
            metrics_time_series['aspect_ratio'].append(aspect_ratio)
            metrics_time_series['ellipticity'].append(ellipticity)
            metrics_time_series['roundness'].append(roundness)
            metrics_time_series['a'].append(a)
            metrics_time_series['b'].append(b)
            metrics_time_series['volume_fraction'].append(volume_fraction)

            active_cells = np.where(~np.isnan(pos[:,0]))[0].size
            # print(f"t = {t:.2f}, Step {step}/{STEPS_TOTAL}, cells: {active_cells}")
            last_time = current_time
    elapsed = time.time() - start
    readable_time = time.strftime('%H:%M:%S', time.gmtime(elapsed))
    print(f'Final Cell Count: {active_cells}')
    print(f'Time elapsed: {readable_time}')
    
    # Clean up animation resources
    animation_manager.close()
    
    # Prepare data dictionary
    data_dict = {
        'positions': np.array(positions),
        'forces': np.array(forces),
        'velocities': np.array(velocities),
        'boundaries': np.array(boundaries),
        'divisions': np.array(divisions),
        'deaths': np.array(deaths),
        'times': np.array(times),
        'cell_count': np.array(cell_count),
        'elapsed': readable_time,
        'metrics_time_series': metrics_time_series
    }
    
    # Plot cell count vs. time, runtime plot
    cycle_plot(times, cell_count, N0, OUTPUT_DIR=OUTPUT_DIR)
    if TMAX > T_DORMANT:
        runtime_plot(cell_count, step_times, OUTPUT_DIR=OUTPUT_DIR)
    np.random.seed(42)
    trajectory_plot(positions, Xe, x_cut, deaths, ids=np.random.choice(N0, size=10, replace=False), boundary=boundary, OUTPUT_DIR=OUTPUT_DIR)
    MSD_plot(positions, ids=np.random.choice(N0, size=10, replace=False), fit_frac=1.0, FRAME_SKIP=FRAME_SKIP, OUTPUT_DIR=OUTPUT_DIR)

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
            'KB': KB,
            'KBEND': KBEND,
            'XPROX': XPROX,
            'K_BC': K_BC,
            'SOFT_RANGE': SOFT_RANGE,
            'SOFT_FACTOR': SOFT_FACTOR
        },
        'Cell Division Parameters': {
            'KDEATH': KDEATH,
            'KDIV': KDIV,
            'OFFSET': OFFSET
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
    

    area, perimeter, aspect_ratio, ellipticity, roundness, a , b, volume_fraction = morphometrics(Xe, pos)
    metrics_dict = {
        'area': area,
        'perimeter': perimeter,
        'aspect_ratio': aspect_ratio,
        'ellipticity': ellipticity,
        'roundness': roundness,
        'a': a,
        'b': b,
        'volume_fraction': volume_fraction
    }
    data_dict.update(metrics_dict)

    # Save per-FrameSkip metrics to CSV for convenience
    try:
        df_metrics = pd.DataFrame(metrics_time_series)
        df_metrics.to_csv(os.path.join(OUTPUT_DIR, 'metrics_time_series.csv'), index=False)
    except Exception:
        pass

    # Save simulation data
    if boundary == False:
        Xe, Xe0, x_cut = None, None, None  # Reset boundary variables
    save_files(data_dict, config_params, pos0, pos, Xe0, Xe, x_cut, n_daughter, N0, OUTPUT_DIR=OUTPUT_DIR)

    print("Simulation finished and data saved.")
    
    # Return data for profiler (required format)
    return data_dict, metrics_time_series, metrics_dict, readable_time

def run_case_trials(case='RM_RD', boundary=False, n_runs=10): ### more runs
    """
    Run the simulation multiple times, saving each run in its own folder.
    """

    case_cell_counts = []
    case_final_counts = []
    case_times = []
    case_data_dicts = []
    metrics_dicts_all_trials = []

    parent_dir = case

    # Create parent directory if it doesn't exist
    os.makedirs(f'data/output/{parent_dir}', exist_ok=True)

    for trial in range(n_runs):
        trial_dir = os.path.join(f"data/output/{parent_dir}", f"trial_{trial+1:03d}")
        os.makedirs(trial_dir, exist_ok=True)
        print(f"\n=== Running trial {trial+1}/{n_runs} ===")
        # Reset global variables for each run
        global pos, v, division_status, n_daughter, t_next, VIDEO_FLAG
        pos = np.full((CELLS_MAX, 2), np.nan)
        pos[:N0, :] = pos0
        v = np.zeros((CELLS_MAX, 2))
        division_status = np.zeros((CELLS_MAX,), dtype=bool)
        n_daughter = 0
        t_next = np.inf

        # Set VIDEO_FLAG: True for first trial, False for others
        VIDEO_FLAG = (trial == 0)

        # Pick new migrant and intercal cells
        if case in ['DMx_RD', 'DMy_RD']:
            migrant_cells = np.zeros(CELLS_MAX, dtype=bool)
            migrant_indices = np.random.choice(N0, size=int(N0/5), replace=False)
            migrant_cells[migrant_indices] = True
        else:
            migrant_cells = None
            
        # Pass the trial_dir to run_simulation and save_files
        data_dict, metrics_dict_one_trial, elapsed, readable_time = run_simulation(
            intercal_pairs=None, 
            migrant_cells=migrant_cells, 
            boundary=boundary, 
            case=case, 
            OUTPUT_DIR=trial_dir
        )
#data_dict, metrics_time_series, metrics_dict, readable_time
        metrics_dicts_all_trials.append(metrics_dict_one_trial)

        case_data_dicts.append(data_dict)
        case_cell_counts.append(data_dict['cell_count'] if 'cell_count' in data_dict else [])
        case_final_counts.append(data_dict['cell_count'][-1] if 'cell_count' in data_dict and len(data_dict['cell_count']) > 0 else 0)
        case_times.append(elapsed)

    case_metrics_dict = {k: [d[k] for d in metrics_dicts_all_trials] for k in metrics_dicts_all_trials[0]}

    print("\n=== Summary of all trials ===")
    print(f"Mean final cell count: {np.mean(case_final_counts):.2f} ± {np.std(case_final_counts):.2f}")
    print(f"Mean elapsed time: {np.mean([float(t.split(':')[-1]) for t in case_times]):.2f} seconds")
    
    # Create multi-trial comparison plots
    if len(case_data_dicts) > 1:
        print(f"\nCreating multi-trial comparison plots...")
        multi_trial_dir = os.path.join(f'data/output/{parent_dir}', 'multi_trial_analysis')
        os.makedirs(multi_trial_dir, exist_ok=True)
        multi_trial_cycle_plot(case_data_dicts, OUTPUT_DIR=multi_trial_dir)
        multi_trial_metrics_plot(case_data_dicts, OUTPUT_DIR=multi_trial_dir)
    
    return case_data_dicts, case_metrics_dict, case_cell_counts, case_final_counts, case_times

def run_all_cases(CASES=CASES, n_runs=10):

    # dict_metrics_all = dict.fromkeys(METRIC_NAMES) # to be passed to beeswarm
    rows = []

    for case in CASES:
        print(f'CASE: {case}')
        case_data_dicts, case_metrics_dict, case_cell_counts, case_final_counts, case_times = run_case_trials(case=case, boundary=False, n_runs=n_runs)
        for metric in case_metrics_dict:
            for val in case_metrics_dict[metric]:
                row = {'metric': metric, 'case': case, 'value': val}
                rows.append(row)

    metrics_df = pd.DataFrame(rows)
    metrics_df.to_csv('data/metrics_all_cases.csv')

    metric_beeswarm(metrics_df, 'a', OUTPUT_DIR='data')
    metric_beeswarm(metrics_df, 'aspect_ratio', OUTPUT_DIR='data')

if __name__ == "__main__":
    run_case_trials(case='RM_RD', n_runs=10, boundary=False)
    # run_case_trials(case='RM_DD_0', n_runs=N_RUNS, boundary=False)
    # run_all_cases(CASES=CASES,n_runs=10)
    print(f'Total time taken: {time.time() - time_start:.2f} seconds')