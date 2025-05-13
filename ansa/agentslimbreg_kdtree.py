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
from line_profiler import LineProfiler
from scipy.spatial import cKDTree
from scipy.sparse import csc_matrix, lil_matrix, spdiags
#from numba import njit, prange

# — Global flags and settings — #
profiling_flag = True  # toggle profiling on/off
video_flag = True     # toggle video on/off
pause_time = 1e-5     # pause between frames when animating
frame_skip = 1000       # only draw every N steps
# plot setup
fig, ax = plt.subplots(figsize=(6,6), dpi=200)
fig.canvas.draw()
w, h = fig.canvas.get_width_height()   # these are the exact pixel dims
writer = imageio.get_writer(
    'out.mp4',
    fps=30,
    ffmpeg_params=['-s', f'{w}x{h}']    # e.g. ['-s', '1200x1200']
) if video_flag else None

# — Simulation parameters — #
dt = 1e-5

if profiling_flag:
    profiling_steps = 500000
    Tmax = profiling_steps * dt # total sim time
    steps_total = profiling_steps
else:
    Tmax = 5.0 # total sim time
    steps_total = int(Tmax/dt)
dl_crit = 0.1
xi = 1.0
kb = 10.0
kcoll = 0.08
kproximal = 1000 # for penalty force
xproximal = 0.0 # left wall
yproximal = 1.5 # top and bottom "walls"
kdiv = 0.8
offset = 0.1

# — Load initial cell positions —
matfile = sio.loadmat('/Users/ansa/Desktop/copos-lab/Regen_ABS/python/cellinitialization_n500.mat')
N = int(matfile['Ncells'][0,0])
pos0 = matfile['pos0']    # shape: (N,2)
print(f"Successfully loaded {N} cells from MATLAB file")

# — Pre‑allocate cell arrays —
cells_max = 5 * N
pos = np.full((cells_max,2), np.nan)
v = np.zeros((cells_max,2))
tau = np.zeros((cells_max,1))
pos[:N,:] = pos0
division_status = np.zeros((cells_max,), dtype=bool)

# — Alive mask & daughter count —
alive = np.full((cells_max,), np.nan)
alive[:N] = 1
n_daughter = 0

########### FUCNTIONS ###########
# @profile
def build_boundary():

    theta = np.linspace(1.5*np.pi, 2.5*np.pi, 200)
    xb_semi = 1.5 * np.cos(theta)
    yb_semi = 1.5 * np.sin(theta)

    # find the two points closest to x=1.2
    diffs = np.abs(xb_semi - 1.2)
    closest_indices = np.argsort(diffs)[:2]
    min_idx = np.min(closest_indices)

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

    #concatenate to form closed loop
    xb = np.concatenate([
        x_top,
        xb_semi[:min_idx],
        x_v,
        xb_semi[closest_indices.max():],
        x_bottom
    ])
    yb = np.concatenate([
        y_top,
        yb_semi[:min_idx],
        y_v,
        yb_semi[closest_indices.max():],
        y_bottom
    ])

    Xb0 = np.vstack([xb, yb]).T   # (Nb,2)
    Xb = Xb0.copy()
    dsb = np.hypot(*(Xb0[1] - Xb0[0]))  # first segment length

    # — Sparse forward‑difference matrix Db —
    Nb = Xb.shape[0]
    e = np.ones(Nb)
    Db = spdiags([-e, e], [0,1], Nb, Nb, format='csr')
    #Db = csc_matrix(Db)
    Db[Nb-1,0] = 1

    # rest lengths of boundary "springs"
    blp0 = np.hypot(*(Db @ Xb0).T)
    blm0 = np.hypot(*(Db.T @ Xb0).T)

    # Calculate mid and rest indices now (fixing the error)
    mid_idx = np.where((yb >= -0.5) & (yb <= 0.5))[0]
    rest_idx = np.setdiff1d(np.arange(Nb), mid_idx)

    return Xb, Nb, Db, blp0, blm0, dsb, kb, mid_idx, rest_idx, #left_wall

Xb, Nb, Db, blp0, blm0, dsb, kb, mid_idx, rest_idx = build_boundary() #left_wall = build_boundary()

def signal_handler(sig, frame):
    print("\nSimulation interrupted! Cleaning up...")
    sys.exit(0)
signal.signal(signal.SIGINT, signal_handler) # press (Ctrl+C) to exit

### PROFILING AND ANIMATION SETUPS ###
def profiling(print_stats=False):
    print("Starting profiling with cProfile...")
    
    # Run with cProfile
    profiler = cProfile.Profile()
    profiler.enable()
    elapsed = run_simulation()
    profiler.disable()
    
    stats = pstats.Stats(profiler).sort_stats(SortKey.CUMULATIVE)

    # Print sorted stats
    if print_stats:
        print("\n--- cProfile Results (Top 20 by cumulative time) ---")
        stats.print_stats(20)
        
        print("\n--- cProfile Results (Top 20 by total time) ---")
        stats.sort_stats(SortKey.TIME)
        stats.print_stats(20)
    
    # Save detailed stats to a file for later analysis
    stats.dump_stats('cell_sim_profile.prof')
    print("Detailed profile saved to 'cell_sim_profile.prof'")
    print("You can visualize this with tools like SnakeViz: 'snakeviz [path]/cell_sim_profile.prof'")

    # Line profiler note
    # print("\nTo use line_profiler for more detailed analysis:")
    # print("1. Install line_profiler: pip install line_profiler")
    # print("2. Add # @profile decorators to functions you want to profile")
    # print("3. Run: kernprof -l agentslimbreg.py")
    # print("4. View results: python -m line_profiler agentslimbreg.py.lprof")


def animate_frame(step, t, pos, Xb, pos0):
    """
    Draw current frame, record to writer if desired.
    """
    if not video_flag or (step % frame_skip != 0) or step == 0:
        return

    active = ~np.isnan(pos[:, 0])
    daughters = division_status  # Use division_status to identify daughter cells

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
########################################

### SIMULATION FUNCTIONS ###
# @profile
def cell_cell_repulsion(pos_slice, dl_crit):
    """
    Optimized cell-cell repulsion using KDTree for neighbor search.
    Each cell receives repulsive forces from nearby cells within dl_crit.
    """
    n = pos_slice.shape[0]
    F = np.zeros((n, 2))
    if n <= 1:
        return F

    # Use cKDTree for efficient spatial queries
    tree = cKDTree(pos_slice)

    # Find all pairs within dl_crit using spatial tree
    pairs = tree.query_pairs(dl_crit, output_type='ndarray')

    if len(pairs) == 0:
        return F
    i, j = pairs[:, 0], pairs[:, 1]

    # Calculate displacement vectors between pairs
    dx = pos_slice[j, 0] - pos_slice[i, 0]
    dy = pos_slice[j, 1] - pos_slice[i, 1]
    dl_sq = dx**2 + dy**2
    dl_crit_sq = dl_crit**2
    # Filter pairs within strict distance and avoid self-interaction
    valid = (dl_sq < dl_crit_sq) & (dl_sq > 0.0)
    i, j = i[valid], j[valid]
    dx, dy = dx[valid], dy[valid]
    dl = np.sqrt(dl_sq[valid])

    # Calculate directional force components
    fx = (10.0 * dx) / dl
    fy = (10.0 * dy) / dl

    # Apply forces symmetrically using atomic operations
    np.add.at(F, (j, 0), fx)
    np.add.at(F, (j, 1), fy)
    np.add.at(F, (i, 0), -fx)
    np.add.at(F, (i, 1), -fy)

    return F
# @profile
def semi_circle_repulsion(pos_slice, dl_crit, Xb, Xb_tree):

    n = pos_slice.shape[0]
    F = np.zeros((n,2))
    cells = pos_slice
    #tree = cKDTree(Xb,compact_nodes=True, balanced_tree=True)

    neighbor_counts = Xb_tree.query_ball_point(pos_slice, dl_crit, return_length=True)
    F[:, 0] = -2.0 * np.array(neighbor_counts)
    #print(F.shape)
    
    return F
# @profile
def semi_circle_elasticity(Xb, Db, blp0, blm0, dsb, kb, mid_idx, rest_idx): # reread over this I don't quite understand
    """Compute elastic forces for the boundary"""
    kb_vals = kb * np.ones(Nb)
    kb_vals[mid_idx] /= 1000.0
    # forward/back differences
    fp = Db @ Xb
    fm = Db.T @ Xb
    lp = np.hypot(fp[:,0], fp[:,1])
    lm = np.hypot(fm[:,0], fm[:,1])
    # avoid zero
    lp_safe = np.where(lp==0, 1e-12, lp)
    lm_safe = np.where(lm==0, 1e-12, lm)
    Fbs = ((kb_vals*(lp/blp0-1))[:,None] * (fp/lp_safe[:,None]) +
           (kb_vals*(lm/blm0-1))[:,None] * (fm/lm_safe[:,None]))
    return Fbs / dsb

# @profile
def boundary_collision_force(pos, Xb, kcoll, Xb_tree):
    """Vectorized boundary collision force using KDTree for nearest neighbor search."""

    #tree = cKDTree(Xb)
    distances, indices = Xb_tree.query(pos, k=3, workers=-1) # Find 2 or 3????? nearest neighbors for all cells in one batch
    
    # Get vectors to nearest points
    pts3 = Xb[indices]  # Shape: (Nc, 3, 2)
    dxy3 = pts3 - pos[:, None, :]
    
    # Calculate weights and directions
    d2 = np.linalg.norm(dxy3, axis=2)
    valid = d2 > 1e-12  # Avoid division by zero
    d2_safe = np.where(valid, d2, 1.0)
    dirs = dxy3 / d2_safe[:, :, None]
    
    # Calculate force contributions
    w = d2 / d2.sum(axis=1, keepdims=True)
    F3 = kcoll * w[:, :, None] * dirs
    
    # Accumulate forces using vectorized operations
    F_collision = np.zeros((Xb.shape[0], 2))
    np.add.at(F_collision, indices.ravel(), F3.reshape(-1, 2))
    
    return F_collision
# @profile
def penalty_force(cells, xprox, yprox, kprox):
    n = len(cells)
    F = np.zeros((n,2), dtype=float)

    # 1) Left wall: x < xprox
    mx = cells[:,0] < xprox
    if mx.any():
        dx = xprox - cells[mx,0]
        # push to the right (+x)
        F[mx,0] =  kprox * dx

    # 2) Top wall: y > +yprox
    my_top = cells[:,1] > yprox
    if my_top.any():
        dy = cells[my_top,1] - yprox
        # push down (–y)
        F[my_top,1] = -kprox * dy

    # 3) Bottom wall: y < -yprox
    my_bottom = cells[:,1] < -yprox
    if my_bottom.any():
        dy = -yprox - cells[my_bottom,1]
        # push up (+y)
        F[my_bottom,1] =  kprox * dy

    return F

'''
function [Fpenalty] = compute_proximal_penalty(pos,xb,kproximal)
    N = length(pos);
    Fpenalty = zeros(N,2);
    distances = (xb-pos(:,1)).*(pos(:,1)<xb);
    Fpenalty(:,1) = -(kproximal*distances.*pos(:,1)./abs(pos(:,1)));
    Fpenalty(:,2) = Fpenalty(:,2);
end
'''



# division and apoptosis
# @profile
def division(active):
        global pos, alive, n_daughter, Xb, division_status
        if np.random.rand() < (len(active)-n_daughter)*kdiv*dt:
            nan_slots = np.where(np.isnan(pos[:,0]))[0]
            if nan_slots.size>0: 
                i = np.random.choice(active)
                new_idx = nan_slots[0]
                ang = -np.pi/6*np.random.randn()
                dx,dy = offset*np.cos(ang), offset*np.sin(ang)
                pos[new_idx] = pos[i] + [dx,dy]
                alive[new_idx] = 1
                division_status[new_idx] = True
                n_daughter += 1

# Create a single iteration function to profile
# @profile
def single_iteration(step, t):
    """A single iteration of the simulation. This wraps the main loop body for profiling."""
    global pos, alive, n_daughter, Xb, division_status
    
    # zero forces
    F_cc = np.zeros((cells_max,2))
    F_epid = np.zeros((cells_max,2))
    F_pull = np.zeros((cells_max,2))
    F_collision = np.zeros((Nb,2))
    F_penalty = np.zeros((cells_max,2))
    active = np.where(~np.isnan(pos[:,0]))[0]
    division(active)

    # recompute active
    active = np.where(~np.isnan(pos[:,0]))[0]
    
    # KDTree for Xb
    Xb_tree = cKDTree(Xb)

    # forces on cells
    F_cc[active] = cell_cell_repulsion(pos[active], dl_crit)
    F_epid[active] = semi_circle_repulsion(pos[active], dl_crit, Xb, Xb_tree)
    # F_pull[active, 0] = 1.0  # x-component
    # F_pull[active, 1] = 0.0  # y-component
    #F_penalty(active_cells,:) = compute_proximal_penalty(pos(active_cells,:),xproximal,kproximal);
    F_penalty[active] = penalty_force(pos[active], xproximal,yproximal, kproximal)
    
    #F_collision = boundary_collision_force(pos[active], Xb, kcoll)
    near_boundary_mask = np.any(F_epid[active] != 0, axis=1)
    near_boundary_cells = active[near_boundary_mask]

    if len(near_boundary_cells) > 0:
        F_collision += boundary_collision_force(pos[near_boundary_cells], Xb, kcoll, Xb_tree)

    # combine & update cells
    F = F_cc + 5*F_epid + F_penalty #+ 0.5*F_pull
    v = F/xi
    eta = -6 + 12*np.random.rand(len(active),2) ### double check
    pos[active] += v[active]*dt + eta*dt*6 # euler step ### double check

    # boundary elasticity + update
    F_elast = semi_circle_elasticity(Xb, Db, blp0, blm0, dsb, kb, mid_idx, rest_idx)
    Fb = F_elast + 5*F_collision
    Xb[1:-1] += (Fb[1:-1]/xi)*dt

# Function to run the profiled simulation

def run_simulation(video_flag=video_flag): 
    build_boundary()
    print(f"Running for {steps_total} steps")
    start = time.time()    

    # Track cell counts
    initial_cells = np.where(~np.isnan(pos[:,0]))[0].size
    print(f"Initial cell count: {initial_cells}")
    
    # Run simulation
    for step, t in enumerate(np.arange(0, Tmax, dt)):
        single_iteration(step, t) # advance one step
        if video_flag:
            animate_frame(step, t, pos, Xb, pos0)
        if step % frame_skip == 0:
            active_cells = np.where(~np.isnan(pos[:,0]))[0].size
            print(f"Step {step}/{steps_total}, current cells: {active_cells}")
    
    elapsed = time.time() - start
    
    # Print summary statistics
    active_cells = np.where(~np.isnan(pos[:,0]))[0].size
    print(f"Simulation complete. Time elapsed: {elapsed:.2f}s")
    print(f"Average time per step: {elapsed/steps_total*1000:.2f}ms")
    print(f"Final cell count: {active_cells}")


    # Add code to check for disappearing cells
    print("\n--- Cell Stability Analysis ---")
    print(f"Initial cells: {N}")
    print(f"Final cells: {np.where(~np.isnan(pos[:,0]))[0].size}")
    print(f"Final cells in view: {np.where((pos[:,0] > -0.1) & (pos[:,0] < 3.0) & (pos[:,1] > -1.5) & (pos[:,1] < 1.5))[0].size}")
    print(f"Cell division events: {n_daughter}")
    if n_daughter > 0:
        print(f"Expected final count: {N + n_daughter}")
    
    # Diagnose NaN values
    nan_check = np.isnan(pos).any(axis=1)
    nan_after_initialization = nan_check[N:]
    valid_after_initialization = ~nan_after_initialization
    count_valid_after_init = np.sum(valid_after_initialization)
    if count_valid_after_init != n_daughter:
        print(f"WARNING: {n_daughter} cells were created, but only {count_valid_after_init} remain valid")
    
    np.save('cell_positions.npy', pos)
    np.save('boundary_positions.npy', Xb)
    return elapsed

# Main
if __name__ == "__main__":
    if profiling_flag:
        profiling()
    else:
        run_simulation()