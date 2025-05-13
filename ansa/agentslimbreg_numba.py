"""
Limb Regeneration Simulation
Ansa Brews-Smith, May 2025
Copos Lab, Northeastern University
"""
import matplotlib
matplotlib.use('Agg')  # Set backend before importing pyplot
import matplotlib.pyplot as plt
import numpy as np
import scipy as sp
import scipy.io as sio 
import time
import imageio
import signal
import sys
import cProfile
import pstats
from pstats import SortKey
from scipy.spatial import cKDTree
from scipy.sparse import spdiags
from numba import njit, prange

# — Global flags and settings — #
profiling_flag = True      # toggle profiling on/off
video_flag = True          # toggle video on/off
frame_skip = 1000          # only draw every N steps

# plot setup
fig, ax = plt.subplots(figsize=(6,6), dpi=200)
fig.canvas.draw()
w, h = fig.canvas.get_width_height()
writer = imageio.get_writer(
    'out.mp4',
    fps=30,
    ffmpeg_params=['-s', f'{w}x{h}']
) if video_flag else None

# — Simulation parameters — #
dt = 1e-5

if profiling_flag:
    profiling_steps = 500000
    Tmax = profiling_steps * dt  # total sim time
    steps_total = profiling_steps
else:
    Tmax = 5.0                   # total sim time
    steps_total = int(Tmax/dt)

dl_crit = 0.1
xi = 1.0
kb = 10.0
kcoll = 0.08
kproximal = 1000   # for penalty force
xproximal = 0.0    # left wall
yproximal = 1.5    # top and bottom "walls"
kdiv = 0.8
offset = 0.1

# — Load initial cell positions —
matfile = sio.loadmat('cellinitialization_n500.mat')
N = int(matfile['Ncells'][0,0])
pos0 = matfile['pos0']    # shape: (N,2)
print(f"Successfully loaded {N} cells from MATLAB file")

# — Pre‑allocate cell arrays —
cells_max = 5 * N
pos = np.full((cells_max, 2), np.nan)
v = np.zeros((cells_max, 2))
pos[:N, :] = pos0
division_status = np.zeros((cells_max,), dtype=bool)

# — Alive mask & daughter count —
alive = np.full((cells_max,), np.nan)
alive[:N] = 1
n_daughter = 0

########### FUNCTIONS ###########

def build_boundary():
    """Construct the boundary shape for the limb regeneration simulation"""
    theta = np.linspace(1.5*np.pi, 2.5*np.pi, 200)
    xb_semi = 1.5 * np.cos(theta)
    yb_semi = 1.5 * np.sin(theta)

    # find the two points closest to x=1.2
    diffs = np.abs(xb_semi - 1.2)
    closest_indices = np.argsort(diffs)[:2]
    min_idx = np.min(closest_indices)
    max_idx = np.max(closest_indices)

    # mean spacing along that arc
    dists = np.hypot(np.diff(xb_semi[:min_idx]), np.diff(yb_semi[:min_idx]))
    avg_ds = np.mean(dists)

    # vertical segment
    y_v = np.arange(yb_semi[closest_indices].min(),
                   yb_semi[closest_indices].max(), avg_ds)
    x_v = np.full_like(y_v, 1.2)

    # small horizontal caps
    x_top = xb_semi[0] - np.array([2*avg_ds, avg_ds])
    y_top = np.full(2, yb_semi[0])
    x_bottom = xb_semi[-1] - np.array([2*avg_ds, avg_ds])
    y_bottom = np.full(2, yb_semi[-1])

    # concatenate to form closed loop
    xb = np.concatenate([
        x_top,
        xb_semi[:min_idx],
        x_v,
        xb_semi[max_idx:],
        x_bottom
    ])
    yb = np.concatenate([
        y_top,
        yb_semi[:min_idx],
        y_v,
        yb_semi[max_idx:],
        y_bottom
    ])

    Xb0 = np.vstack([xb, yb]).T   # (Nb,2)
    Xb = Xb0.copy()
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
    mid_idx = np.where((yb >= -0.5) & (yb <= 0.5))[0]
    rest_idx = np.setdiff1d(np.arange(Nb), mid_idx)

    return Xb, Nb, Db, blp0, blm0, dsb, kb, mid_idx, rest_idx

# Initialize boundary
Xb, Nb, Db, blp0, blm0, dsb, kb, mid_idx, rest_idx = build_boundary()

def signal_handler(sig, frame):
    """Handle Ctrl+C interruption"""
    print("\nSimulation interrupted! Cleaning up...")
    if video_flag and writer is not None:
        writer.close()
    plt.close(fig)
    sys.exit(0)

signal.signal(signal.SIGINT, signal_handler)

### ANIMATION SETUP ###
def animate_frame(step, t, pos, Xb, pos0):
    """Draw current frame and record to video if enabled"""
    if not video_flag or (step % frame_skip != 0) or step == 0:
        return

    active = ~np.isnan(pos[:, 0])
    daughters = division_status

    ax.clear()
    # boundary
    ax.plot(Xb[:, 0], Xb[:, 1], '.-', lw=2, color=(220/255, 104/255, 94/255))
    # cells
    ax.scatter(pos[active & ~daughters, 0], pos[active & ~daughters, 1], s=50,
               facecolor=(170/255, 157/255, 241/255))  # Original cells
    ax.scatter(pos[active & daughters, 0], pos[active & daughters, 1], s=50,
               facecolor=(128/255, 0/255, 128/255))  # Daughter cells (purple)
    # amputation plane
    x_cut = pos0[:, 0].max()
    ax.plot([x_cut, x_cut], [-1.25, 1.25], '--', lw=1,
            color=(233/255, 244/255, 205/255))
    # scale bar
    sbx = np.linspace(0.7, 0.7 + x_cut/5, 100)
    ax.plot(sbx, -1.5 * np.ones_like(sbx), '-', lw=5, color='w')

    ax.set_xlim(-0.1, 3)
    ax.set_ylim(-1.5, 1.5)
    ax.set_aspect('equal', 'box')
    ax.set_title(f"T = {t:.4f}")

    # Draw the figure to update the renderer
    fig.canvas.draw()

    # Capture the buffer and write to video
    buf = np.asarray(fig.canvas.renderer.buffer_rgba())[..., :3]
    writer.append_data(buf)

### CORE SIMULATION FUNCTIONS (NUMBA-ACCELERATED) ###

@njit(parallel=True)
def cell_cell_repulsion(pos_slice, dl_crit):
    """
    Optimized cell-cell repulsion using Numba for parallelization.
    Each cell receives repulsive forces from nearby cells within dl_crit.
    """
    n = len(pos_slice)
    F = np.zeros((n, 2))
    
    if n <= 1:
        return F
    
    dl_crit_sq = dl_crit**2
    
    # Parallelize the outer loop for significant speedup
    for i in prange(n):
        for j in range(n):
            if i != j:
                dx = pos_slice[j, 0] - pos_slice[i, 0]
                dy = pos_slice[j, 1] - pos_slice[i, 1]
                dl_sq = dx**2 + dy**2
                
                if 0.0 < dl_sq < dl_crit_sq:
                    dl = np.sqrt(dl_sq)
                    fx = (10.0 * dx) / dl
                    fy = (10.0 * dy) / dl
                    
                    F[i, 0] -= fx
                    F[i, 1] -= fy
                    
    return F

@njit
def semi_circle_repulsion(pos_slice, Xb, dl_crit):
    """Calculate repulsion from boundary points"""
    n = len(pos_slice)
    F = np.zeros((n, 2))
    dl_crit_sq = dl_crit * dl_crit
    
    # Count boundary points within dl_crit of each cell
    for i in range(n):
        count = 0
        for j in range(len(Xb)):
            dx = pos_slice[i, 0] - Xb[j, 0]
            dy = pos_slice[i, 1] - Xb[j, 1]
            dist_sq = dx*dx + dy*dy
            if dist_sq < dl_crit_sq:
                count += 1
                
        # Apply repulsive force in x-direction based on neighbor count
        F[i, 0] = -2.0 * count # are we supposed to use neighbor cou
    
    return F

@njit
def semi_circle_elasticity(Xb, Db_data, Db_indices, Db_indptr, Db_shape, 
                           DbT_data, DbT_indices, DbT_indptr, DbT_shape,
                           blp0, blm0, dsb, kb_vals):
    """Compute elastic forces for the boundary"""
    Nb = len(Xb)
    Fbs = np.zeros((Nb, 2))
    
    # Apply forward difference (Db @ Xb)
    fp = np.zeros((Nb, 2))
    for i in range(Nb):
        row_start, row_end = Db_indptr[i], Db_indptr[i+1]
        for j in range(row_start, row_end):
            col = Db_indices[j]
            fp[i, 0] += Db_data[j] * Xb[col, 0]
            fp[i, 1] += Db_data[j] * Xb[col, 1]
    
    # Apply backward difference (Db.T @ Xb)
    fm = np.zeros((Nb, 2))
    for i in range(Nb):
        row_start, row_end = DbT_indptr[i], DbT_indptr[i+1]
        for j in range(row_start, row_end):
            col = DbT_indices[j]
            fm[i, 0] += DbT_data[j] * Xb[col, 0]
            fm[i, 1] += DbT_data[j] * Xb[col, 1]
    
    # Calculate lengths
    lp = np.zeros(Nb)
    lm = np.zeros(Nb)
    for i in range(Nb):
        lp[i] = np.sqrt(fp[i, 0]**2 + fp[i, 1]**2)
        lm[i] = np.sqrt(fm[i, 0]**2 + fm[i, 1]**2)
    
    # Compute elastic forces
    for i in range(Nb):
        # avoid zero
        lp_safe = 1e-12 if lp[i] == 0 else lp[i]
        lm_safe = 1e-12 if lm[i] == 0 else lm[i]
        
        # Calculate forces
        fp_term_x = (kb_vals[i] * (lp[i]/blp0[i] - 1)) * (fp[i, 0] / lp_safe)
        fp_term_y = (kb_vals[i] * (lp[i]/blp0[i] - 1)) * (fp[i, 1] / lp_safe)
        
        fm_term_x = (kb_vals[i] * (lm[i]/blm0[i] - 1)) * (fm[i, 0] / lm_safe)
        fm_term_y = (kb_vals[i] * (lm[i]/blm0[i] - 1)) * (fm[i, 1] / lm_safe)
        
        Fbs[i, 0] = (fp_term_x + fm_term_x) / dsb
        Fbs[i, 1] = (fp_term_y + fm_term_y) / dsb
    
    return Fbs

@njit(parallel=True)
def boundary_collision_force(pos, Xb, kcoll):
    """Calculate collision forces between cells and boundary"""
    n_cells = len(pos)
    n_boundary = len(Xb)
    F_collision = np.zeros((n_boundary, 2))
    
    for i in prange(n_cells):
        # Find 3 nearest boundary points
        distances = np.empty(n_boundary)
        indices = np.empty(3, dtype=np.int64)
        
        # Calculate distances to all boundary points
        for j in range(n_boundary):
            dx = Xb[j, 0] - pos[i, 0]
            dy = Xb[j, 1] - pos[i, 1]
            distances[j] = np.sqrt(dx*dx + dy*dy)
        
        # Find indices of 3 closest points
        for k in range(3):
            min_idx = 0
            min_dist = 1e10
            for j in range(n_boundary):
                if distances[j] < min_dist:
                    min_dist = distances[j]
                    min_idx = j
            indices[k] = min_idx
            distances[min_idx] = 1e10  # Mark as processed
        
        # Calculate directions and distances
        dirs = np.zeros((3, 2))
        d2 = np.zeros(3)
        
        for k in range(3):
            idx = indices[k]
            dx = Xb[idx, 0] - pos[i, 0]
            dy = Xb[idx, 1] - pos[i, 1]
            d2[k] = dx*dx + dy*dy
            if d2[k] > 1e-12:
                d = np.sqrt(d2[k])
                dirs[k, 0] = dx / d
                dirs[k, 1] = dy / d
        
        # Calculate weights and forces
        d2_sum = np.sum(d2)
        if d2_sum > 0:
            for k in range(3):
                w = d2[k] / d2_sum
                idx = indices[k]
                F_collision[idx, 0] += kcoll * w * dirs[k, 0]
                F_collision[idx, 1] += kcoll * w * dirs[k, 1]
    
    return F_collision

@njit
def penalty_force(cells, xprox, yprox, kprox):
    """Calculate penalty forces for cells near simulation boundaries"""
    n = len(cells)
    F = np.zeros((n, 2), dtype=np.float64)

    for i in range(n):
        # 1) Left wall: x < xprox
        if cells[i, 0] < xprox:
            dx = xprox - cells[i, 0]
            F[i, 0] = kprox * dx  # push right (+x)

        # 2) Top wall: y > +yprox
        if cells[i, 1] > yprox:
            dy = cells[i, 1] - yprox
            F[i, 1] = -kprox * dy  # push down (-y)

        # 3) Bottom wall: y < -yprox
        if cells[i, 1] < -yprox:
            dy = -yprox - cells[i, 1]
            F[i, 1] = kprox * dy  # push up (+y)

    return F

@njit
def update_positions(active, pos, v, dt):
    """Update cell positions with forces and random noise"""
    eta = -6 + 12*np.random.random((len(active), 2))
    
    for i in range(len(active)):
        idx = active[i]
        pos[idx, 0] += v[idx, 0]*dt + eta[i, 0]*dt*6
        pos[idx, 1] += v[idx, 1]*dt + eta[i, 1]*dt*6
    
    return pos

def division(active):
    """Handle cell division events"""
    global pos, alive, n_daughter, division_status
    if np.random.rand() < (len(active)-n_daughter)*kdiv*dt:
        nan_slots = np.where(np.isnan(pos[:,0]))[0]
        if nan_slots.size > 0: 
            i = np.random.choice(active)
            new_idx = nan_slots[0]
            ang = -np.pi/6*np.random.randn()
            dx, dy = offset*np.cos(ang), offset*np.sin(ang)
            pos[new_idx] = pos[i] + [dx, dy]
            alive[new_idx] = 1
            division_status[new_idx] = True
            n_daughter += 1

### MAIN SIMULATION FUNCTIONS ###

def single_iteration(step, t):
    """A single iteration of the simulation"""
    global pos, alive, n_daughter, Xb, division_status, Nb
    
    # zero forces
    F_cc = np.zeros((cells_max, 2))
    F_epid = np.zeros((cells_max, 2))
    F_penalty = np.zeros((cells_max, 2))
    F_collision = np.zeros((Nb, 2))
    
    # Get active cells
    active = np.where(~np.isnan(pos[:,0]))[0]
    division(active)
    active = np.where(~np.isnan(pos[:,0]))[0]  # Recompute after division
    
    # Calculate forces on cells
    F_cc[active] = cell_cell_repulsion(pos[active], dl_crit)
    F_epid[active] = semi_circle_repulsion(pos[active], Xb, dl_crit)
    F_penalty[active] = penalty_force(pos[active], xproximal, yproximal, kproximal)
    
    # Only calculate collision forces for cells near boundary
    near_boundary_mask = np.any(F_epid[active] != 0, axis=1)
    near_boundary_cells = active[near_boundary_mask]
    if len(near_boundary_cells) > 0:
        F_collision = boundary_collision_force(pos[near_boundary_cells], Xb, kcoll)

    # Combine forces and update cell positions
    F = F_cc + 5*F_epid + F_penalty
    v = F/xi
    active_array = np.array(active, dtype=np.int64)
    pos = update_positions(active_array, pos, v, dt)

    # Update boundary
    # Prepare sparse matrix data for Numba
    kb_vals = kb * np.ones(Nb)
    kb_vals[mid_idx] /= 1000.0
    
    # Get CSR format components for Db and Db.T
    Db_data = Db.data
    Db_indices = Db.indices
    Db_indptr = Db.indptr
    Db_shape = Db.shape
    
    DbT = Db.T.tocsr()
    DbT_data = DbT.data
    DbT_indices = DbT.indices
    DbT_indptr = DbT.indptr
    DbT_shape = DbT.shape
    
    # Calculate boundary elasticity forces
    F_elast = semi_circle_elasticity(Xb, Db_data, Db_indices, Db_indptr, Db_shape,
                                    DbT_data, DbT_indices, DbT_indptr, DbT_shape,
                                    blp0, blm0, dsb, kb_vals)
    
    # Update boundary positions
    Fb = F_elast + 5*F_collision
    Xb[1:-1] += (Fb[1:-1]/xi)*dt

def run_simulation():
    """Main simulation loop"""
    global Xb, Nb, Db, blp0, blm0, dsb, kb, mid_idx, rest_idx
    
    # Ensure boundary is initialized correctly
    Xb, Nb, Db, blp0, blm0, dsb, kb, mid_idx, rest_idx = build_boundary()
    
    print(f"Running simulation for {steps_total} steps with Numba optimization")
    print(f"Initial cell count: {np.where(~np.isnan(pos[:,0]))[0].size}")
    print(f"Boundary points: {Nb}")
    
    start = time.time()
    
    # Main simulation loop
    for step, t in enumerate(np.arange(0, Tmax, dt)):
        single_iteration(step, t)
        
        if video_flag:
            animate_frame(step, t, pos, Xb, pos0)
            
        if step % frame_skip == 0:
            active_cells = np.where(~np.isnan(pos[:,0]))[0].size
            print(f"Step {step}/{steps_total}, cells: {active_cells}")
    
    elapsed = time.time() - start
    
    # Report simulation results
    active_cells = np.where(~np.isnan(pos[:,0]))[0].size
    print(f"\nSimulation complete in {elapsed:.2f}s")
    print(f"Average time per step: {elapsed/steps_total*1000:.2f}ms")
    print(f"Final cell count: {active_cells}")
    
    print("\n--- Cell Analysis ---")
    print(f"Initial cells: {N}")
    print(f"Final cells: {active_cells}")
    print(f"Visible cells: {np.where((pos[:,0] > -0.1) & (pos[:,0] < 3.0) & (pos[:,1] > -1.5) & (pos[:,1] < 1.5))[0].size}")
    print(f"Division events: {n_daughter}")
    
    if n_daughter > 0:
        nan_after_init = np.isnan(pos[N:]).any(axis=1)
        valid_after_init = ~nan_after_init
        cells_after_init = np.sum(valid_after_init)
        
        if cells_after_init != n_daughter:
            print(f"WARNING: {n_daughter} division events occurred, but only {cells_after_init} new cells remain")
    
    # Save results
    np.save('cell_positions_numba.npy', pos)
    np.save('boundary_positions_numba.npy', Xb)
    
    return elapsed

def profiling(print_stats=True):
    """Run simulation with profiling enabled"""
    print("Starting profiling with cProfile...")
    
    # Run with cProfile
    profiler = cProfile.Profile()
    profiler.enable()
    elapsed = run_simulation()
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
    stats.dump_stats('cell_sim_profile_numba.prof')
    print("Detailed profile saved to 'cell_sim_profile_numba.prof'")
    
    return elapsed

# Main execution
if __name__ == "__main__":
    try:
        if profiling_flag:
            profiling(print_stats=True)
        else:
            run_simulation()
    finally:
        # Clean up resources
        if video_flag and writer is not None:
            writer.close()
        plt.close(fig)
        print("Simulation completed and resources cleaned up.")