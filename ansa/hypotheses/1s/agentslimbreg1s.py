"""
Limb Regeneration Simulation - Agents Based Model
Ansa Brews-Smith, May 2025
Copos Lab, Northeastern University

SIM 1: RANDOM MOTION AND RANDOM DIVISION ANGLE
    - cells have a random motion term eta
    - division angle is ranfom uniform distribution (0, 2 pi)
"""
import warnings
warnings.filterwarnings('ignore')

import matplotlib
matplotlib.use('Agg')  # Set backend before importing pyplot
import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio 
import time
import imageio
import signal
import sys
import cProfile
import pstats
from pstats import SortKey
from scipy.sparse import spdiags
from numba import njit, prange, set_num_threads
import os
from numpy import int64, float64
from config.post_process import morphometrics, density_heatmap, boundary_plot, cycle_plot

set_num_threads(8)  ## CHANGE FOR YOUR SYSTEM ##

from config.config_soft import *

OUTPUT_DIR = os.path.join('data', f'1s')
os.makedirs(OUTPUT_DIR, exist_ok=True)

# plot setup
fig, ax = plt.subplots(figsize=VIDEO_PARAMS['figsize'], dpi=VIDEO_PARAMS['dpi'])
fig.canvas.draw()
w, h = fig.canvas.get_width_height()

# Load initial cell positions
matfile = sio.loadmat(CELL_INIT_FILE)
writer = imageio.get_writer(
    os.path.join(OUTPUT_DIR, 'out.mp4'),
    fps=VIDEO_PARAMS['fps'],
    ffmpeg_params=['-s', f'{w}x{h}']
) if VIDEO_FLAG else None

# — Load initial cell positions —
N0 = int(matfile['Ncells'][0,0])
pos0 = matfile['pos0']    # shape: (N,2)
print(f"Successfully loaded {N0} cells from MATLAB file")

# Gillespie clock setup
rate0  = (KDIV+KDEATH)*N0
t_next = np.random.exponential(1.0/rate0)

# — Pre‑allocate cell arrays —
CELLS_MAX = 6 * N0
pos = np.full((CELLS_MAX, 2), np.nan)
v = np.zeros((CELLS_MAX, 2))
pos[:N0, :] = pos0
division_status = np.zeros((CELLS_MAX,), dtype=bool)

# — Alive mask & daughter count —
n_daughter = 0

################################# FUNCTIONS #################################

def build_boundary():
    """Construct the boundary shape for the limb regeneration simulation"""

    Xb0 = np.loadtxt('data/input/boundary0.csv', delimiter=',')
    Xb = Xb0.copy()
    xb = Xb0[:, 0]
    yb = Xb0[:, 1]
    dsb = np.hypot(*(Xb0[1] - Xb0[0]))  # first segment length

    # Sparse forward‑difference matrix Db
    Nb = Xb.shape[0]
    e = np.ones(Nb)
    Db = spdiags([-e, e], [0,1], Nb, Nb, format='csr')
    Db[Nb-1, 0] = 1

    # rest lengths of boundary "springs"
    blp0 = np.hypot(*(Db @ Xb0).T)
    blm0 = np.hypot(*(Db.T @ Xb0).T)

    # Calculate mid and rest indices
    #soft_idx = np.where((yb >= -10) & (yb <=10))[0]
    soft_idx = np.where((yb >= SOFT_RANGE[0]) & (yb <= SOFT_RANGE[1]))[0]

    return Xb0, Xb, Nb, Db, blp0, blm0, dsb, soft_idx

# Initialize boundary
Xb0, Xb, Nb, Db, blp0, blm0, dsb, soft_idx = build_boundary()
x_cut = Xb0[:, 0].max()
# Create DbT for the semi_circle_elasticity function
DbT = Db.T.tocsr()

@njit(parallel=True)
def hard_reset(pos_active, v_active, Xb):
    """
    Reset cells that leave the growth region boundary. Left boundary (amputation site)
    handling has been moved elsewhere in the code.
    
    Args:
        pos_active: (n_cells, 2) array of cell positions
        v_active: (n_cells, 2) array of cell velocities
        Xb: (n_boundary, 2) array of boundary points
        
    Returns:
        Updated pos_active and v_active
    """
    n_cells = len(pos_active)
    n_boundary = len(Xb)
    
    for i in prange(n_cells):
        x, y = pos_active[i]
        
        # Handle growth region boundary using ray casting
        inside = False
        j = n_boundary - 1
        
        # Ray casting algorithm
        for k in range(n_boundary):
            xi, yi = Xb[k]
            xj, yj = Xb[j]
            
            # Check if ray intersects with edge
            if ((yi > y) != (yj > y)) and (x < (xj - xi) * (y - yi) / (yj - yi) + xi):
                inside = not inside
            j = k
        
        # Close the polygon properly
        xi, yi = Xb[0]
        xj, yj = Xb[-1]
        if ((yi > y) != (yj > y)) and (x < (xj - xi) * (y - yi) / (yj - yi) + xi):
            inside = not inside
        
        # If outside the boundary, reset to nearest point
        if not inside:
            # Optimized nearest point search (coarse + fine)
            min_dist_sq = np.inf
            min_idx = 0
            
            # Coarse search (check every 50th point)
            step = max(1, n_boundary // 50)
            for j in range(0, n_boundary, step):
                dx = Xb[j, 0] - x
                dy = Xb[j, 1] - y
                dist_sq = dx*dx + dy*dy
                if dist_sq < min_dist_sq:
                    min_dist_sq = dist_sq
                    min_idx = j
            
            # Fine search (+-3 points around best candidate)
            start = max(0, min_idx - 3)
            end = min(n_boundary, min_idx + 4)
            for j in range(start, end):
                dx = Xb[j, 0] - x
                dy = Xb[j, 1] - y
                dist_sq = dx*dx + dy*dy
                if dist_sq < min_dist_sq:
                    min_dist_sq = dist_sq
                    min_idx = j
            
            # Calculate inward normal at nearest boundary point
            prev_idx = (min_idx - 1) % n_boundary
            next_idx = (min_idx + 1) % n_boundary
            
            # Tangent vector (direction of boundary)
            tx = Xb[next_idx, 0] - Xb[prev_idx, 0]
            ty = Xb[next_idx, 1] - Xb[prev_idx, 1]
            
            # Normal vector (rotated tangent 90 degrees CCW)
            nx = -ty
            ny = tx
            norm = np.sqrt(nx*nx + ny*ny)
            
            if norm > 0:
                nx /= norm
                ny /= norm
                
                # Verify normal direction points inward
                test_x = Xb[min_idx, 0] + 0.01 * nx
                test_y = Xb[min_idx, 1] + 0.01 * ny
                
                # Quick inside check for test point
                test_inside = False
                j = n_boundary - 1
                for k in range(n_boundary):
                    xi, yi = Xb[k]
                    xj, yj = Xb[j]
                    if ((yi > test_y) != (yj > test_y)) and (test_x < (xj - xi) * (test_y - yi) / (yj - yi) + xi):
                        test_inside = not test_inside
                    j = k
                
                # Flip normal if pointing outward
                if not test_inside:
                    nx = -nx
                    ny = -ny
            else:
                # Fallback: direction toward center
                center_x = np.mean(Xb[:, 0])
                center_y = np.mean(Xb[:, 1])
                nx = center_x - Xb[min_idx, 0]
                ny = center_y - Xb[min_idx, 1]
                norm = np.sqrt(nx*nx + ny*ny)
                if norm > 0:
                    nx /= norm
                    ny /= norm
                else:
                    nx = 1.0  # Default rightward
                    ny = 0.0
            
            # Reset position with:
            # - Standard offset (0.07)
            # - Small random jitter
            # - Along calculated normal
            jitter = 0.01 * (np.random.rand() - 0.5)
            pos_active[i, 0] = Xb[min_idx, 0] + 0.07 * nx + jitter
            pos_active[i, 1] = Xb[min_idx, 1] + 0.07 * ny + jitter
            
            # Apply velocity damping
            v_active[i] *= 0.5  # Reduce velocity by 50%
    
    return pos_active, v_active


def save_files(data_dict, output_file=None):
    """Save all simulation data to a single CSV file."""
    
    # Create output directory if it doesn't exist
    if output_file is None:
            output_file = os.path.join(OUTPUT_DIR, 'simulation_data.csv')
    
    # Extract data
    positions = data_dict['positions']
    forces = data_dict['forces']
    velocities = data_dict['velocities']
    boundaries = data_dict['boundaries']
    divisions = data_dict['divisions']
    times = data_dict['times']
    elapsed = data_dict['elapsed']
    
    # Create rows for the CSV file
    rows = []
    
    # For each timestep
    for t_idx, t in enumerate(times):
        # For each active cell
        active_mask = ~np.isnan(positions[t_idx, :, 0])
        active_indices = np.where(active_mask)[0]
        
        for cell_idx in active_indices:
            # Basic cell data
            row = [
                t,                                  # Time
                int(t_idx),                         # Timestep
                int(cell_idx),                      # Cell ID
                positions[t_idx, cell_idx, 0],      # X position
                positions[t_idx, cell_idx, 1],      # Y position
                forces[t_idx, cell_idx, 0],         # X force
                forces[t_idx, cell_idx, 1],         # Y force
                velocities[t_idx, cell_idx, 0],     # X velocity
                velocities[t_idx, cell_idx, 1],     # Y velocity
                int(divisions[t_idx, cell_idx])     # Division status
            ]
            rows.append(row)
    
    # Convert to numpy array and save
    data_array = np.array(rows)
    header = 'time,timestep,cell_id,x,y,force_x,force_y,velocity_x,velocity_y,division_status'
    np.savetxt(output_file, data_array, delimiter=',', header=header, comments='')
    
    # Save boundary data in a separate file
    boundary_file = os.path.join(OUTPUT_DIR, 'boundary_data.csv')
    boundary_rows = []
    
    for t_idx, t in enumerate(times):
        for b_idx in range(len(boundaries[t_idx])):
            boundary_row = [
                t,                                  # Time
                t_idx,                              # Timestep
                b_idx,                              # Boundary point ID
                boundaries[t_idx, b_idx, 0],        # X position
                boundaries[t_idx, b_idx, 1]         # Y position
            ]
            boundary_rows.append(boundary_row)
    
    boundary_array = np.array(boundary_rows)
    boundary_header = 'time,timestep,boundary_id,x,y'
    np.savetxt(boundary_file, boundary_array, delimiter=',', header=boundary_header, comments='')
    
    print(f"Cell data ")
    print(f"Boundary data saved to {boundary_file}\n")
    
    # Save a small metadata file with simulation parameters
    metadata_file = os.path.join(OUTPUT_DIR, 'metadata.txt')
    with open(metadata_file, 'w') as f:
        f.write(f'Time Elapsed: {elapsed}\n\n')

        f.write("Simulation Parameters\n")
        f.write("====================\n\n")
        
        f.write("Time Parameters\n")
        f.write("--------------\n")
        f.write(f"Total time (TMAX): {TMAX}\n")
        f.write(f"Time step (DT): {DT}\n")
        f.write(f"Total steps: {STEPS_TOTAL}\n")
        f.write(f"Timesteps saved: {len(times)}\n\n")
        
        f.write("Physical Parameters\n")
        f.write("------------------\n")
        f.write(f"Critical distance (DL_CRIT): {DL_CRIT}\n")
        f.write(f"Damping coefficient (XI): {XI}\n")
        f.write(f"Boundary stiffness (KB): {KB}\n")
        f.write(f"Bending stiffness (KBEND): {KBEND}\n")
        f.write(f"Collision strength (KCOLL): {KCOLL}\n")
        f.write(f"Proximal force (KPROX): {KPROX}\n")
        f.write(f"Proximal position (XPROX): {XPROX}\n")
        f.write(f"Soft boundary range: {SOFT_RANGE}\n\n")
        f.write(f"Spring Constant Between Boundary and Closest Cell (K_BCC): {K_BCC} ")
        
        f.write("Cell Division Parameters\n")
        f.write("----------------------\n")
        f.write(f"Death rate (KDEATH): {KDEATH}\n")
        f.write(f"Division rate (KDIV): {KDIV}\n")
        f.write(f"Daughter offset (OFFSET): {OFFSET}\n\n")
        
        f.write("Domain Bounds\n")
        f.write("-------------\n")
        f.write(f"X range: [{XMIN}, {XMAX}]\n")
        f.write(f"Y range: [{YMIN}, {YMAX}]\n\n")
        
        f.write("Cell Statistics\n")
        f.write("--------------\n")
        f.write(f"Initial cells: {N0}\n")
        active_cells = np.where(~np.isnan(pos[:,0]))[0].size
        f.write(f"Final total cells: {active_cells}\n")
        visible_cells = np.where((pos[:,0] > -0.1) & 
                            (pos[:,0] < 3.0) & 
                            (pos[:,1] > -1.5) & 
                            (pos[:,1] < 1.5))[0].size
        # f.write(f"Final visible cells: {visible_cells}\n")
        f.write(f"Total division events: {n_daughter}\n")
        f.write(f"Growth region cells: {len(np.where(pos[:,0] > x_cut)[0])}\n\n")
        
        f.write("Simulation Settings\n")
        f.write("------------------\n")
        f.write(f"Video enabled: {VIDEO_FLAG}\n")
        f.write(f"Frame skip: {FRAME_SKIP}\n")
        f.write(f"Profiling enabled: {PROFILING_FLAG} \n")
        
        
        density_heatmap(SOFT_RANGE, pos=pos, Xb=Xb, x_cut=x_cut, bin_size=0.1, OUTPUT_DIR=OUTPUT_DIR)
        # Calculate final morphometrics
        f.write("Growth Region Morphometrics\n")
        f.write("-----------------------\n")
        # Get growth region points
        Xb_growth = Xb_growth = Xb[Xb[:, 0] > x_cut]
        
        # Calculate metrics
        area, perimeter, aspect_ratio, ellipticity, roundness, a = morphometrics(Xb_growth)
        boundary_plot(Xb0, Xb, Xb_growth, x_cut, aspect_ratio=aspect_ratio, area=area, roundness=roundness,a=a,perimeter=perimeter, ellipticity=ellipticity, OUTPUT_DIR=OUTPUT_DIR)
        # Write metrics to file
        f.write(f"Semi-major axis (length): {a:.2f}\n")
        f.write(f"Area: {area:.2f}\n")
        f.write(f"Perimeter: {perimeter:.2f}\n")
        f.write(f"Aspect ratio: {aspect_ratio:.2f}\n")
        f.write(f"Ellipticity: {ellipticity:.2f}\n")
        f.write(f"Roundness: {roundness:.2f}\n")

def signal_handler(sig, frame):
    """Handle Ctrl+C interruption"""
    print("\nSimulation interrupted! Cleaning up...")
    if VIDEO_FLAG and writer is not None:
        writer.close()
    plt.close(fig)
    sys.exit(0)

signal.signal(signal.SIGINT, signal_handler)

### ANIMATION SETUP ###
def animate_frame(step, t, pos, Xb, pos0, soft_idx=soft_idx):
    """Draw current frame and record to video if enabled"""
    if not VIDEO_FLAG or (step % FRAME_SKIP != 0) or step == 0:
        return

    active = ~np.isnan(pos[:, 0])
    daughters = division_status

    ax.clear()
    # boundary

    for i in range(len(Xb)-1):
        if i in soft_idx:
            color = 'pink'  # soft boundary segments
        else:
            color = 'red'   # hard boundary segments
            
        ax.plot([Xb[i,0], Xb[i+1,0]], 
                [Xb[i,1], Xb[i+1,1]], 
                '-', lw=2, color=color)

    # cells
    ax.scatter(pos[active & ~daughters, 0], pos[active & ~daughters, 1], s=50,
               facecolor=(170/255, 157/255, 241/255))  # Original cells
    ax.scatter(pos[active & daughters, 0], pos[active & daughters, 1], s=50,
               facecolor=(128/255, 0/255, 128/255))  # Daughter cells (purple)
    # amputation plane
    x_cut = pos0[:, 0].max()
    ax.plot([x_cut, x_cut], [-1.25, 1.25], '--', lw=1,
            color='k')
    # scale bar
    sbx = np.linspace(0.7, 0.7 + x_cut/5, 100)
    ax.plot(sbx, -1.5 * np.ones_like(sbx), '-', lw=5, color='w')

    ax.set_xlim(-0.5, 3)
    ax.set_ylim(-2, 2)

    ax.set_aspect('equal', 'box')
    ax.set_title(f"T = {t:.4f}")

    # Draw the figure to update the renderer
    fig.canvas.draw()

    # Capture the buffer and write to video
    buf = np.asarray(fig.canvas.renderer.buffer_rgba())[..., :3]
    writer.append_data(buf)

### OPTIMIZED CORE SIMULATION FUNCTIONS ###

def cell_cycle(active, t_next):
    """Handle cell division and death events using Gillespie algorithm"""
    global pos, n_daughter, division_status

    n_alive = len(active)
    if n_alive == 0:
        return pos, division_status, np.inf
        
    # Choose event type
    if np.random.rand() < KDIV/(KDIV+KDEATH):
        # Division
        free_slots = np.where(np.isnan(pos[:,0]))[0]
        if len(free_slots) > 0:
            mom = np.random.randint(n_alive)

            if mom is not None:
                dau = free_slots[0]
                ang = 2*np.pi*np.random.rand()
                pos[dau, 0] = pos[mom, 0] + OFFSET*np.cos(ang)
                pos[dau, 1] = pos[mom, 1] + OFFSET*np.sin(ang)
                division_status[dau] = True
                n_daughter += 1
    else:
        # Death
        victim = active[np.random.randint(n_alive)]
        pos[victim, 0] = np.nan
        pos[victim, 1] = np.nan

    # Schedule next event
    active = np.where(~np.isnan(pos[:,0]))[0]
    n_alive = len(active)
    if n_alive == 0:
        t_next = np.inf
    else:
        rate0 = (KDIV+KDEATH)*n_alive
        t_next += np.random.exponential(1.0/rate0)
        
    return pos, division_status, t_next

def semi_circle_elasticity(Xb, Db, DbT, blp0, blm0, dsb, kb_vals):
    """Compute elastic forces (stretch + bend) for the boundary without Numba"""
    # Xb: (Nb,2), Db, DbT: sparse matrices
    # 1) Stretch (spring) forces via matrix multiply
    fp = Db.dot(Xb)       # forward diffs
    fm = DbT.dot(Xb)      # backward diffs

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
    Nb = Xb.shape[0]
    Xp2 = np.roll(Xb, -2, axis=0)
    Xp1 = np.roll(Xb, -1, axis=0)
    Xm1 = np.roll(Xb, 1, axis=0)
    Xm2 = np.roll(Xb, 2, axis=0)
    fourth_diff = Xp2 - 4*Xp1 + 6*Xb - 4*Xm1 + Xm2
    Fbend = - (KBEND * kb_vals/dsb)[:,None] * fourth_diff
    # zero ends if desired
    Fbend[:2,:] = 0
    Fbend[-2:,:] = 0

    # 6) total
    Fbs = F_stretch + Fbend
    return Fbs

@njit(parallel=True)
def cell_operations(pos_active, v_active, Xb, dl_crit, kcoll, dt,
                               XMIN=XMIN, XMAX=XMAX, YMIN=YMIN, YMAX=YMAX):
    """
    Consolidated function containing all parallel cell operations to reduce Numba overhead.
    Now includes boundary friction to prevent sliding along epithelium.
    
    Returns:
    --------
    F_cc: Cell-cell repulsion forces
    F_epid: Cell-boundary repulsion forces
    F_collision: Boundary collision forces
    pos_active: Updated cell positions
    v_active: Updated cell velocities
    """
    n_cells = len(pos_active)
    n_boundary = len(Xb)
    
    # Pre-allocate all force arrays
    F_cc = np.zeros((n_cells, 2), dtype=float64)
    F_epid = np.zeros((n_cells, 2), dtype=float64)
    F_collision = np.zeros((n_boundary, 2), dtype=float64)
    F_correction = np.zeros((n_cells, 2), dtype=float64) ########

    # Generate random noise for position updates
    eta = -6 + 12*np.random.random((n_cells, 2))
    
    # 1. CELL-CELL REPULSION
    if n_cells > 1:
        # Build grid for cell list
        cs = dl_crit
        nx = int((XMAX - XMIN) / cs) + 1
        ny = int((YMAX - YMIN) / cs) + 1
        nbin = nx*ny
        
        # Initialize head pointers and linked list
        head = -1 * np.ones(nbin, dtype=int64)
        nxt = -1 * np.ones(n_cells, dtype=int64)
        
        # Bin each particle
        for i in range(n_cells):
            ix = int((pos_active[i,0] - XMIN) // cs)
            iy = int((pos_active[i,1] - YMIN) // cs)
            if ix < 0: ix = 0
            elif ix >= nx: ix = nx-1
            if iy < 0: iy = 0
            elif iy >= ny: iy = ny-1
            b = ix + iy*nx
            nxt[i] = head[b]
            head[b] = i
        
        dl2 = dl_crit * dl_crit
        
        # Loop particles in parallel for cell-cell interactions
        for i in prange(n_cells):
            xi = pos_active[i,0]
            yi = pos_active[i,1]
            # Determine own bin
            ix0 = int((xi - XMIN) // cs)
            iy0 = int((yi - YMIN) // cs)
            # Check 3×3 neighboring bins
            for dix in (-1,0,1):
                ix = ix0 + dix
                if ix < 0 or ix >= nx: continue
                for diy in (-1,0,1):
                    iy = iy0 + diy
                    if iy < 0 or iy >= ny: continue
                    b = ix + iy*nx
                    j = head[b]
                    while j != -1:
                        if j > i:
                            dx = pos_active[j,0] - xi
                            dy = pos_active[j,1] - yi
                            d2 = dx*dx + dy*dy
                            if 0.0 < d2 < dl2:
                                d = np.sqrt(d2)
                                # Symmetric update
                                fx = (2.0 * dx)/d ## CHANGE BACK TO 10
                                fy = (2.0 * dy)/d ## CHANGE BACK TO 10
                                F_cc[i,0] -= fx
                                F_cc[i,1] -= fy
                                F_cc[j,0] += fx
                                F_cc[j,1] += fy
                        j = nxt[j]
    
    ### SPRING LIKE BOUNDARY - CELL FORCE
# 2. BOUNDARY-CELL SPRING CONNECTIONS
    dl_crit_sq = DL_CRIT * DL_CRIT
    spring_rest_length = DL_CRIT
    # K_BCC is spring constant between boundary and closest cell

    # Initialize forces
    for i in prange(n_cells):
        total_force_x = 0.0
        total_force_y = 0.0
        
        # Standard repulsion from nearby boundary points
        for j in range(n_boundary):
            dx = pos_active[i, 0] - Xb[j, 0]
            dy = pos_active[i, 1] - Xb[j, 1]
            dist_sq = dx*dx + dy*dy
            
            if dist_sq < dl_crit_sq and dist_sq > 0:
                dist = np.sqrt(dist_sq)
                fx = dx / dist  # normalized direction
                fy = dy / dist
                force_magnitude = 2.0 * (1.0 - dist/DL_CRIT)
                
                total_force_x += force_magnitude * fx
                total_force_y += force_magnitude * fy
        
        F_epid[i, 0] = total_force_x
        F_epid[i, 1] = total_force_y

    # 3. BOUNDARY-CELL SPRING SYSTEM
    # For each boundary point, find closest cell and create spring connection
    for j in prange(n_boundary):
        min_dist_sq = 1e20
        closest_cell = -1
        
        # Find closest cell to this boundary point
        for i in range(n_cells):
            dx = pos_active[i, 0] - Xb[j, 0]
            dy = pos_active[i, 1] - Xb[j, 1]
            dist_sq = dx*dx + dy*dy
            
            if dist_sq < min_dist_sq:
                min_dist_sq = dist_sq
                closest_cell = i
        
        if closest_cell >= 0 and min_dist_sq > 0:
            current_dist = np.sqrt(min_dist_sq)
            
            # Spring force calculation: F = k * (current_length - rest_length)
            spring_extension = current_dist - spring_rest_length
            spring_force_magnitude = K_BCC * spring_extension
            
            # Direction from boundary point to cell
            dx = pos_active[closest_cell, 0] - Xb[j, 0]
            dy = pos_active[closest_cell, 1] - Xb[j, 1]
            fx = dx / current_dist  # normalized direction
            fy = dy / current_dist
            
            if spring_extension > 0:
                # Cell is too far: pull boundary toward cell and cell toward boundary
                F_collision[j, 0] += spring_force_magnitude * fx   # Pull boundary toward cell
                F_collision[j, 1] += spring_force_magnitude * fy
                
                # Also pull cell toward boundary (add to existing forces)
                F_epid[closest_cell, 0] -= spring_force_magnitude * fx
                F_epid[closest_cell, 1] -= spring_force_magnitude * fy
                
            else:
                # Cell is too close: push boundary away from cell and cell away from boundary
                F_collision[j, 0] += spring_force_magnitude * fx   # Push boundary away (negative force)
                F_collision[j, 1] += spring_force_magnitude * fy
                
                # Also push cell away from boundary
                F_epid[closest_cell, 0] -= spring_force_magnitude * fx
                F_epid[closest_cell, 1] -= spring_force_magnitude * fy

    # 4. RIGHT-WARD CORRECTION FORCE
    left_wall = np.min(Xb[:, 0])
    for i in prange(n_cells):
        if pos_active[i, 0] < left_wall:
            F_correction[i, 0] = -8.0 * pos_active[i, 0]
        elif pos_active[i, 0] < (left_wall + 0.5):
            F_correction[i, 0] = 1.0 * (1- (pos_active[i, 0] - left_wall))
        
    # 5. CALCULATE VELOCITIES FROM FORCES
    for i in prange(n_cells):
        # Calculate the total force on this cell
        total_force_x = F_cc[i, 0] + 5*F_epid[i, 0] + F_correction[i, 0]
        total_force_y = F_cc[i, 1] + 5*F_epid[i, 1] + F_correction[i, 1]
        
        # Update velocity (F/XI)
        v_x = total_force_x / XI
        v_y = total_force_y / XI
        
        # Store velocity
        v_active[i, 0] = v_x
        v_active[i, 1] = v_y
    
    # 6. UPDATE POSITIONS WITH DAMPED VELOCITIES
    for i in prange(n_cells):
        # Update position with random noise
        pos_active[i, 0] += v_active[i, 0]*dt + eta[i, 0]*dt*6
        pos_active[i, 1] += v_active[i, 1]*dt + eta[i, 1]*dt*6
    return F_cc, F_epid, F_collision, pos_active, v_active

def single_iteration(step, t):
    """A single iteration of the simulation using consolidated operations"""
    global pos, n_daughter, Xb, division_status, Nb, t_next
    
    # Pre-allocate complete force arrays (needed for data collection)
    F_cc_full = np.zeros((CELLS_MAX, 2))
    F_epid_full = np.zeros((CELLS_MAX, 2))
    F_collision = np.zeros((Nb, 2))
    
    # Get active cells
    active = np.where(~np.isnan(pos[:,0]))[0]

    # Handle cell cycle events
    while t >= t_next:
        pos, division_status, t_next = cell_cycle(active, t_next)
        active = np.where(~np.isnan(pos[:,0]))[0]  # Update active cells after potential changes
    
    if len(active) == 0:
        # No active cells, return zeros
        return np.zeros((CELLS_MAX, 2)), np.zeros((CELLS_MAX, 2)), division_status, 0
    
    # Extract active cells for efficient processing
    pos_active = pos[active].copy()
    v_active = np.zeros((len(active), 2))
    
    # Call consolidated function for all parallel operations
    F_cc, F_epid, F_collision, pos_active, v_active = cell_operations(
        pos_active, v_active, Xb, DL_CRIT, KCOLL, DT
    )
    
    # Check and reset positions in one efficient pass
    pos_active, v_active = hard_reset(pos_active, v_active, Xb)
    
    # Copy updated positions and velocities back to main arrays
    pos[active] = pos_active
    v = np.zeros((CELLS_MAX, 2))
    v[active] = v_active
    
    # Copy forces to full arrays for data collection
    F_cc_full[active] = F_cc
    F_epid_full[active] = F_epid
    
    # Update boundary
    kb_vals = KB * np.ones(Nb)
    kb_vals[soft_idx] /= SOFT_FACTOR
    
    # Calculate boundary elasticity forces
    F_elast = semi_circle_elasticity(Xb, Db, DbT, blp0, blm0, dsb, kb_vals)
    
    # Update boundary positions
    Fb = F_elast + 5*F_collision
    Xb[1:-1] += (Fb[1:-1]/XI)*DT
    
    # Combine forces for return (for data collection)
    F = F_cc_full + 5*F_epid_full
    
    # Count active cells
    N = np.where(~np.isnan(pos[:,0]))[0].size
    
    return F, v, division_status, N


def run_simulation():
    """Main simulation loop with CSV data collection"""
    global Xb, Nb, Db, blp0, blm0, dsb, KB, soft_idx
    
    print(f"Running simulation for {STEPS_TOTAL} steps")
    print(f"Initial cell count: {np.where(~np.isnan(pos[:,0]))[0].size}")
    print(f"Boundary points: {Nb}")
    
    # Data collection - use lists for flexibility
    positions = []
    forces = []
    velocities = []
    boundaries = []
    divisions = []
    times = []
    cell_count = []
    
    # Save initial state
    positions.append(pos.copy())
    forces.append(np.zeros_like(pos))
    velocities.append(np.zeros_like(pos))
    boundaries.append(Xb.copy())
    divisions.append(division_status.copy())
    cell_count.append(N0)
    times.append(0.0)
    
    start = time.time()
    
    # Main simulation loop
    for step, t in enumerate(np.arange(0, TMAX, DT)):
        # Use the consolidated single_iteration function
        F, v, div_status, N = single_iteration(step, t)
        
        if VIDEO_FLAG:
            animate_frame(step, t, pos, Xb, pos0, soft_idx)
            
        if step % FRAME_SKIP == 0 and step > 0:
            # Save this timestep's data
            positions.append(pos.copy())
            forces.append(F.copy())
            velocities.append(v.copy())
            boundaries.append(Xb.copy())
            divisions.append(division_status.copy())
            times.append(t)
            cell_count.append(N)
            
            active_cells = np.where(~np.isnan(pos[:,0]))[0].size
            print(f"t = {t:.2f}, Step {step}/{STEPS_TOTAL}, cells: {active_cells}")
    
    elapsed = time.time() - start
    readable_time = time.strftime('%H:%M:%S', time.gmtime(elapsed))
    
    # Prepare data dictionary
    data_dict = {
        'positions': np.array(positions),
        'forces': np.array(forces),
        'velocities': np.array(velocities),
        'boundaries': np.array(boundaries),
        'divisions': np.array(divisions),
        'times': np.array(times),
        'elapsed':readable_time
    }
    
    # Plot cell count vs. time
    cycle_plot(times, cell_count, N0, OUTPUT_DIR=OUTPUT_DIR)
    
    # Save to CSV
    save_files(data_dict)
    
    # Also save numpy arrays for Python-specific analysis
    np.save(os.path.join(OUTPUT_DIR, 'cell_positions.npy'), np.array(positions))
    np.save(os.path.join(OUTPUT_DIR, 'cell_forces.npy'), np.array(forces))
    np.save(os.path.join(OUTPUT_DIR, 'cell_velocities.npy'), np.array(velocities))
    np.save(os.path.join(OUTPUT_DIR, 'boundary_positions.npy'), np.array(boundaries))
    
    # Report simulation results
    print(f"\nSimulation complete in {readable_time}")
    print(f"Average time per step: {elapsed/STEPS_TOTAL*1000:.2f}ms")
    
    active_cells = np.where(~np.isnan(pos[:,0]))[0].size
    print(f"Final cell count: {active_cells}")
    
    print("\n--- Cell Analysis ---")
    print(f"Initial cells: {N0}")
    print(f"Final cells: {active_cells}")
    # print(f"Visible cells: {np.where((pos[:,0] > -0.1) & (pos[:,0] < 3.0) & (pos[:,1] > -1.5) & (pos[:,1] < 1.5))[0].size}")
    print(f"Division events: {n_daughter}")
    
    if n_daughter > 0:
        nan_after_init = np.isnan(pos[N0:]).any(axis=1)
        valid_after_init = ~nan_after_init
        cells_after_init = np.sum(valid_after_init)
        
        # if cells_after_init != n_daughter:
        #     print(f"WARNING: {n_daughter} division events occurred, but only {cells_after_init} new cells remain")
    
    return data_dict, elapsed

def profiling(print_stats=True):
    """Run simulation with profiling enabled"""
    print("Starting profiling with cProfile...")
    
    # Run with cProfile
    profiler = cProfile.Profile()
    profiler.enable()
    data_dict, elapsed = run_simulation()
    profiler.disable()
    
    stats = pstats.Stats(profiler).sort_stats(SortKey.CUMULATIVE)
    
    # Print statistics
    if print_stats:
        print("\n--- cProfile Results (Top 20 by cumulative time) ---")
        stats.print_stats(20)
        
        print("\n--- cProfile Results (Top 20 by total time) ---")
        stats.sort_stats(SortKey.TIME)
        stats.print_stats(20)
    
    # Save detailed stats to a file
    stats.dump_stats(os.path.join(OUTPUT_DIR, 'cell_sim_profile.prof'))
    print(f"Detailed profile saved to {os.path.join(OUTPUT_DIR, 'cell_sim_profile.prof')}")
    
    return elapsed

# Main execution
if __name__ == "__main__":
    try:
        if PROFILING_FLAG:
            # Run with profiling
            profiler = cProfile.Profile()
            profiler.enable()
            data_dict, elapsed = run_simulation()
            profiler.disable()
            
            # Print statistics
            stats = pstats.Stats(profiler).sort_stats(SortKey.CUMULATIVE)
            print("\n--- cProfile Results (Top 20 by cumulative time) ---")
            stats.print_stats(20)
            
            print("\n--- cProfile Results (Top 20 by total time) ---")
            stats.sort_stats(SortKey.TIME)
            stats.print_stats(20)
            
            # Save profile
            stats.dump_stats(os.path.join(OUTPUT_DIR, 'cell_sim_profile.prof'))
        else:
            # Just run the simulation
            data_dict, elapsed = run_simulation()
    finally:
        # Clean up resources
        if VIDEO_FLAG and writer is not None:
            writer.close()
        plt.close(fig)
        print("Simulation completed and resources cleaned up.")