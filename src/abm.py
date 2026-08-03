"""
Limb Regeneration Simulation - Agents Based Model (2D)
Ansa Brew-Smith, February 2026
Copos Lab, Northeastern University

Cell Types:
0 = Normal/unjammed
1 = Distal-oriented migrant
2 = intercalating migrant
3 = Jammed
"""
import matplotlib
matplotlib.use('Agg')

import warnings
warnings.filterwarnings('ignore')

from utils.post_process import morphometrics, coefficients, distance_metric
from utils.signal_handling import setup_signal_handler
from utils.profiler import profiling
from utils.data_io import save_files, save_config_snapshot
from utils.animations import AnimationManager
from config import *

import shutil
from scipy.special import expit
from scipy.sparse import spdiags
from numpy import dtype, int64, float64
from numba import njit, prange
import os
import time
import numpy as np

pos0 = np.loadtxt(CELL_INIT_FILE, delimiter=',')
N0 = len(pos0)

# Pre-allocate cell arrays
CELLS_MAX = 20 * N0
pos = np.full((CELLS_MAX, 2), np.nan)
v = np.zeros((CELLS_MAX, 2))
pos[:N0, :] = pos0

# Set up cell cycle
division_status = np.zeros((CELLS_MAX,), dtype=bool) # 0 = G1 (at rest), 1 = S/G2/M (mitotic), start all cells at G1
cycle_phases = np.zeros((CELLS_MAX,), dtype=int)
empty_slots = np.isnan(pos[:, 0])
cycle_phases[empty_slots] = -1
death_mask = np.zeros((CELLS_MAX,), dtype=bool)
phase_clocks = np.zeros((CELLS_MAX,), dtype=float)

n_daughter = 0
n_deaths = 0

regulation_front = np.inf
cell_types = np.zeros(CELLS_MAX, dtype=int)

def init_migration_cells(n_cells_max):
    """
    Initialize which cells are migrants at the start by marking entries in cell_types.
    (0 normal, 1 migrant, 2 intercalating, 3 jammed)
    """
    global cell_types
    n_migrants = int(N0 * MIGRATION_FRACTION)
    if n_migrants > 0:
        if WHICH_MIGRATION == 'anterior_posterior':
            anterior_posterior = np.zeros(n_cells_max, dtype=bool)
            anterior_posterior[np.where(
                (pos[:, 1] < -1.5 + 2 * D0) | (pos[:, 1] > 1.5 - 2 * D0))[0]] = True
            migrant_indices = np.random.choice(
                np.where(anterior_posterior)[0], size=n_migrants, replace=False)
            cell_types[migrant_indices] = 1
        elif WHICH_MIGRATION == 'random':
            if JAMMING_ENABLED:
                non_jammed_count = np.sum(cell_types != 3)
                n_migrants = int(non_jammed_count * MIGRATION_FRACTION)
                migrant_indices = np.random.choice(
                    np.where(
                        cell_types != 3)[0],
                    size=n_migrants,
                    replace=False)
            else:
                migrant_indices = np.random.choice(
                    N0, size=n_migrants, replace=False)

            cell_types[migrant_indices] = 1
        else:
            raise ValueError(f"Invalid migration type: {WHICH_MIGRATION}")
    return cell_types

def update_jammed_cells(Xe):
    """Update jammed cells (type 3) based on position relative to epithelium."""
    global pos, cell_types
    x_max = Xe[:, 0].max()
    jamming_area = ~(((x_max - pos[:,0]) < FLUID_LIKE_ZONE_WIDTH) & (pos[:,1] > -0.9) & (pos[:,1] < 0.9)) | (np.abs(pos[:,1]) > 1.0) & ~np.isnan(pos[:,0])
    cell_types[jamming_area] = 3

def init_intercal_cells():
    global cell_types
    n_intercal = int(N0 * INTERCAL_FRACTION)
    intercal_indices = np.random.choice(N0, size=n_intercal, replace=False)
    cell_types[intercal_indices] = 2

    return cell_types

def build_boundaries(soft=True):
    """Construct the boundary shapes for the limb regeneration simulation (Xb is bone, Xe is epithlium)"""

    Xe0 = np.loadtxt(BOUNDARY_FILE, delimiter=',')  # initial epithelium
    Xe = Xe0.copy()
    xe = Xe0[:, 0]
    ye = Xe0[:, 1]

    # Sparse forward-difference matrix Db
    Ne = Xe.shape[0]
    e = np.ones(Ne)
    Db = spdiags([-e, e], [0, 1], Ne, Ne, format='csr')
    Db[Ne - 1, :] = 0

    # Load rest lengths from saved files if available, otherwise compute from current geometry
    dsb_file = os.path.join(INPUT_DIR, 'sim_data', 'dsb.npy')
    blp0_file = os.path.join(INPUT_DIR, 'sim_data', 'blp0.npy')
    blm0_file = os.path.join(INPUT_DIR, 'sim_data', 'blm0.npy')
    ne_file = os.path.join(INPUT_DIR,'sim_data', 'Ne.npy')

    if all(os.path.exists(f) for f in [dsb_file, blp0_file, blm0_file, ne_file]):
        saved_Ne = int(np.load(ne_file))
        if saved_Ne == Ne:
            dsb = float(np.load(dsb_file))
            blp0 = np.load(blp0_file)
            blm0 = np.load(blm0_file)
        else:
            print(f"Warning: Saved rest lengths have Ne={saved_Ne} but current boundary has Ne={Ne}. Computing from current geometry.")
            dsb = np.hypot(*(Xe0[1] - Xe0[0]))
            blp0 = np.hypot(*(Db @ Xe0).T)
            blm0 = np.hypot(*(Db.T @ Xe0).T)
    else:
        dsb = np.hypot(*(Xe0[1] - Xe0[0]))  # first segment length
        blp0 = np.hypot(*(Db @ Xe0).T)  # rest length of edge from i to i+1
        blm0 = np.hypot(*(Db.T @ Xe0).T)  # rest length of edge from i-1 to i

    # Calculate soft boundary indices and sigmoid window
    center = Xe0[:, 0].max() - 0.05
    width = 0.01
    y_lo, y_hi = -1.0, 1.0
    sig_lo = expit((ye - y_lo) / 0.1)
    sig_hi = expit(-(ye - y_hi) / 0.1)
    window = sig_lo * sig_hi

    def k_b(x): return KAPPA1 + (KAPPA2 - KAPPA1) * expit((x - center) / width)
    if soft:
        kb_vals = KAPPA1 * (1 - window) + k_b(xe) * window
    else:
        kb_vals = np.ones(Ne) * KAPPA1
    
    # Smooth gradient
    transition_center = -1.0
    transition_half_width = 0.4

    x_left = transition_center - transition_half_width
    x_right = transition_center + transition_half_width

    transition_factor = np.clip(
        (xe - x_left) / (2 * transition_half_width), 0, 1)

    kb_vals = np.where(xe < x_left, KAPPA0, np.where(xe > x_right, kb_vals, KAPPA0 * (1 - transition_factor) + kb_vals * transition_factor))

    # Random patches of soft epithelium
    if SPORATIC_SOFTENING:
        soft_patches_ranges = [range(10, 15), 
                                range(26, 32),
                                range(57, 62),
                                range(120, 124), 
                                range(140, 144), 
                                range(165, 170), 
                                range(184, 190)]
        soft_indices = np.concatenate([np.array(list(r)) for r in soft_patches_ranges])
        kb_vals[soft_indices] = KAPPA2

    if BONE_VISUALIZATION:
        Xb = np.loadtxt(BONE_FILE, delimiter=',')
    else:
        Xb = None

    return Xe0, Xe, Ne, Db, blp0, blm0, dsb, kb_vals, Xb

def stiffness_swap(kb_vals0, kb_vals_f, t_, swap_interval=SOFTENING_SWAP_INTERVAL): ###
    t_scaled = t_/swap_interval
    kb_vals = (1-t_scaled)*kb_vals0 + t_scaled*kb_vals_f

    return kb_vals

def update_L0(blp0, lp):
    """Update rest length of epithelium springs (viscoelastic)"""
    
    blp0_new = blp0 + ((1/LAM_VISCO) * (lp-blp0))*DT
    ###
    blp0_new[-1] = blp0[-1]
    ###
    blm0_new = np.roll(blp0_new, 1)
    return blp0_new, blm0_new

# Initialize boundary
Xe0, Xe, Ne, Db, blp0, blm0, dsb, kb_vals, Xb = build_boundaries(
    soft=ALLOW_SOFTENING)

kb_vals0 = kb_vals # only significant with stiffness swapping
_, _, _, _, _, _, _, kb_vals_f, _ = build_boundaries(
    soft=not ALLOW_SOFTENING)
# stiffness_swapped = False

left_wall = np.min(Xe0[:, 0])
x_cut = 0  # amputation plane
DbT = Db.T.tocsr()

@njit(parallel=True)
def hard_reset(pos_active, v_active, Xe):
    """Reset cells that leave the epithelial boundary using a ray-casting algorithm"""

    n_cells = len(pos_active)
    n_epithelium = len(Xe)

    for i in prange(n_cells):
        x, y = pos_active[i]
        inside_epithelium = False
        j = n_epithelium - 1

        # Ray casting algorithm for epithelium
        for k in range(n_epithelium):
            xk, yk = Xe[k]
            xj, yj = Xe[j]

            if ((yk > y) != (yj > y)) and (x < (xj - xk) * (y - yk) / (yj - yk) + xk):
                inside_epithelium = not inside_epithelium
            j = k

        # Close the epithelium polygon properly
        xk, yk = Xe[0]
        xj, yj = Xe[-1]
        if ((yk > y) != (yj > y)) and (x < (xj - xk) * (y - yk) / (yj - yk) + xk):
            inside_epithelium = not inside_epithelium

        if not inside_epithelium:
            min_dist_sq = np.inf
            min_idx = 0

            # Outside epithelium - find nearest epithelium point
            # Coarse search
            step = max(1, n_epithelium // 50)
            for j in range(0, n_epithelium, step):
                dx = Xe[j, 0] - x
                dy = Xe[j, 1] - y
                dist_sq = dx * dx + dy * dy
                if dist_sq < min_dist_sq:
                    min_dist_sq = dist_sq
                    min_idx = j

            # Fine search (+-3 points around best candidate)
            start = max(0, min_idx - 3)
            end = min(n_epithelium, min_idx + 4)
            for j in range(start, end):
                dx = Xe[j, 0] - x
                dy = Xe[j, 1] - y
                dist_sq = dx * dx + dy * dy
                if dist_sq < min_dist_sq:
                    min_dist_sq = dist_sq
                    min_idx = j

            # Calculate inward normal at nearest epithelium point
            prev_idx = (min_idx - 1) % n_epithelium
            next_idx = (min_idx + 1) % n_epithelium
            nx = -(Xe[next_idx, 1] - Xe[prev_idx, 1])
            ny = Xe[next_idx, 0] - Xe[prev_idx, 0]
            norm = np.sqrt(nx * nx + ny * ny)

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
                    xk, yk = Xe[k]
                    xj, yj = Xe[j]
                    if ((yk > test_y) != (yj > test_y)) and (
                            test_x < (xj - xk) * (test_y - yk) / (yj - yk) + xk):
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
                norm = np.sqrt(nx * nx + ny * ny)
                if norm > 0:
                    nx /= norm
                    ny /= norm
                else:
                    nx = 0.0
                    ny = 0.0

            # Reset position inside epithelium and damp velocity
            jitter = 0.01 * (np.random.rand() - 0.5)
            pos_active[i, 0] = Xe[min_idx, 0] + 0.07 * nx + jitter
            pos_active[i, 1] = Xe[min_idx, 1] + 0.07 * ny + jitter
            v_active[i] *= 0.5

    return pos_active, v_active

def cell_cycle(div_allowed=True, directed_angle=None, gradient=False):
    global pos, cycle_phases, phase_clocks, death_mask, n_daughter, cell_types, n_deaths, regulation_front

    alive = ~np.isnan(pos[:, 0])
    r_death = np.random.rand(cycle_phases.size)
    x_arr = pos[:, 0][alive]

    if gradient:
        def G_lengths_func(x): return np.where(
            x < regulation_front, G_LENGTH_MAX, G_LENGTH_MIN)
        G_lengths = np.zeros(cycle_phases.shape, dtype=float)
        G_lengths[alive] = G_lengths_func(x_arr)
    else:
        G_lengths = G_LENGTH 

    # Cell death
    pD = KDEATH * DT
    die_mask = alive & (r_death < pD)
    pos[die_mask, :] = np.nan
    cycle_phases[die_mask] = -1
    death_mask[die_mask] = True
    n_deaths += np.sum(die_mask)
    cell_types[die_mask] = 0

    alive = ~np.isnan(pos[:, 0])

    if div_allowed:
        r_trans = np.random.rand(cycle_phases.size)
        maskG = alive & (cycle_phases == 0)
        maskM = alive & (cycle_phases == 1)

        # Geometrically distributed transition times with mean G_length or M_length
        pG = 1.0 / (G_lengths / DT)
        pM = 1.0 / (M_LENGTH / DT)

        enterM = maskG & (r_trans < pG)
        divideM = maskM & (r_trans < pM)
        cycle_phases[enterM] = 1
        phase_clocks[enterM] = 0.0
        cycle_phases[divideM] = 0
        phase_clocks[divideM] = 0.0

        # place daughters for each dividing mother
        mothers = np.where(divideM)[0]
        free_spots = np.where(np.isnan(pos[:, 0]))[0]
        for mom, dau in zip(mothers, free_spots):
            if directed_angle is not None:
                ang = np.random.normal(loc=directed_angle, scale=np.pi / 6)
                r_div_angle = np.random.rand()
                # 50% chance to flip direction
                if r_div_angle < 0.5:
                    ang += np.pi
            else:
                ang = 2 * np.pi * np.random.rand()

            pos[dau, 0] = pos[mom, 0] + OFFSET * np.cos(ang)
            pos[dau, 1] = pos[mom, 1] + OFFSET * np.sin(ang)
            division_status[dau] = True
            cycle_phases[dau] = 0
            phase_clocks[dau] = 0.0
            n_daughter += 1

            # Assign daughter cell type with proper priority
            if cell_types[mom] == 3:
                cell_types[dau] = 3
            elif MIGRATION_ENABLED and np.random.rand() < MIGRATION_FRACTION:
                cell_types[dau] = 1
            elif INTERCALATION_ENABLED and np.random.rand() < INTERCAL_FRACTION:
                cell_types[dau] = 2
            else:
                cell_types[dau] = 0

        alive = ~np.isnan(pos[:, 0])
        phase_clocks[alive] += DT

    return pos, cycle_phases, phase_clocks


def epithelium_elasticity(Xe, Db, DbT, blp0, blm0, dsb, Xe0=Xe0):
    """Compute elastic forces (stretch + bend) for the boundary"""
    global kb_vals

    fp = Db.dot(Xe)
    fm = DbT.dot(Xe)

    lp = np.linalg.norm(fp, axis=1)
    lm = np.linalg.norm(fm, axis=1)
    lp_safe = np.where(lp > 0, lp, 1e-12)
    lm_safe = np.where(lm > 0, lm, 1e-12)

    kb_vals_forward = 0.5 * (kb_vals + np.roll(kb_vals, -1))  # (kb[i] + kb[i+1])/2
    kb_vals_backward = 0.5 * (np.roll(kb_vals, 1) + kb_vals)  # (kb[i-1] + kb[i])/2
    
    t1 = kb_vals_forward * (lp / blp0 - 1)
    t2 = kb_vals_backward * (lm / blm0 - 1)

    F_stretch = (fp * (t1 / dsb)[:, None] / lp_safe[:, None] +
                 fm * (t2 / dsb)[:, None] / lm_safe[:, None])

    # Anchor stiff region (collagen)
    F_anchor = np.zeros((Ne, 2))
    
    anchor_strength = K_LATERAL * kb_vals**2
    
    # Only y-component
    y_displacement = Xe[:, 1] - Xe0[:, 1]
    F_anchor[:, 1] = -anchor_strength * y_displacement

    return F_stretch + F_anchor

@njit(parallel=True, fastmath=True, nogil=True)
def MM_Forces(pos_active, cell_types_active):
    """Calculate cell-cell repulsion/adhesion in the mesenchyme forces using spatial binning."""
    n_cells = len(pos_active)
    F_cc = np.zeros((n_cells, 2), dtype=float64)

    # Grid setup for spatial binning
    interaction_range = 2.0 * D0
    cell_size = interaction_range
    domain_width = XMAX - XMIN
    domain_height = YMAX - YMIN
    nx = max(1, int(domain_width / cell_size)) + 1
    ny = max(1, int(domain_height / cell_size)) + 1
    n_bins = nx * ny
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
                                        fx = (K_CC_REP_JAMMED * max((D0 - r, 0)) -
                                              K_CC_ADH_JAMMED * max((r - D0, 0))) * (dx / r)
                                        fy = (K_CC_REP_JAMMED * max((D0 - r, 0)) -
                                              K_CC_ADH_JAMMED * max((r - D0, 0))) * (dy / r)
                                    else:
                                        fx = (K_CC_REP * max((D0 - r, 0)) -
                                              K_CC_ADH * max((r - D0, 0))) * (dx / r)
                                        fy = (K_CC_REP * max((D0 - r, 0)) -
                                              K_CC_ADH * max((r - D0, 0))) * (dy / r)
                                    F_cc[i, 0] -= fx
                                    F_cc[i, 1] -= fy
                                    F_cc[j, 0] += fx
                                    F_cc[j, 1] += fy
                            j = nxt[j]
                i = nxt[i]
    return F_cc

@njit(fastmath=True, nogil=True, parallel=True)
def ME_forces(pos_active, Xe):
    """Mesenchymal cell and Epithelium interaction using spatial binning"""
    n_cells = len(pos_active)
    n_boundary = len(Xe)
    F_on_cell = np.zeros((n_cells, 2), dtype=float64)
    F_on_epi = np.zeros((n_boundary, 2), dtype=float64)

    half_dl_crit_sq = (0.5 * D0) * (0.5 * D0)
    max_spring_distance = 2.0 * D0

    # Grid setup for spatial binning
    interaction_range = max(2.0 * D0, max_spring_distance)
    cell_size = interaction_range
    nx = max(1, int((XMAX - XMIN) / cell_size)) + 1
    ny = max(1, int((YMAX - YMIN) / cell_size)) + 1
    n_bins = nx * ny

    head_cells = np.full(n_bins, -1, dtype=int64)
    nxt_cells = np.full(n_cells, -1, dtype=int64) # for pos

    head_boundary = np.full(n_bins, -1, dtype=int64)
    nxt_boundary = np.full(n_boundary, -1, dtype=int64) # For Xe

    # Bin all cells and boundary points
    for i in range(n_cells):
        ix = int((pos_active[i, 0] - XMIN) / cell_size)
        iy = int((pos_active[i, 1] - YMIN) / cell_size)
        ix = max(0, min(ix, nx - 1))
        iy = max(0, min(iy, ny - 1))
        cell_idx = ix + iy * nx
        nxt_cells[i] = head_cells[cell_idx]
        head_cells[cell_idx] = i

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
            i = head_cells[cell_idx]
            while i != -1:
                xi, yi = pos_active[i, 0], pos_active[i, 1]
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
                            dist_sq = dx * dx + dy * dy

                            if dist_sq < half_dl_crit_sq and dist_sq > 0:
                                dist = np.sqrt(dist_sq)
                                r = dist
                                fmagx = (K_BC_REP * max((D0 - r, 0)) -
                                         K_BC_ADH * max((r - D0, 0))) * (dx / r)
                                fmagy = (K_BC_REP * max((D0 - r, 0)) -
                                         K_BC_ADH * max((r - D0, 0))) * (dy / r)

                                F_on_cell[i, 0] += fmagx
                                F_on_cell[i, 1] += fmagy
                                F_on_epi[j, 0] -= fmagx
                                F_on_epi[j, 1] -= fmagy

                            j = nxt_boundary[j]

                i = nxt_cells[i]

    return F_on_cell, F_on_epi


@njit(parallel=True, fastmath=True, nogil=True)
def ext_force(Xe, points=POKING_POINTS):
    """
    Apply external force to the epithelium.
    Force is applied in -x direction to the specified boundary points.
    """
    Ne = len(Xe)
    F_ext = np.zeros((Ne, 2), dtype=float64)
    for i in points:
        F_ext[i, 0] = -K_EXT

    return F_ext


def calculate_total_ext_force(Xe, Db, points=POKING_POINTS):
    """
    Calculate total force applied and the force per unit length.
    Must use the same points array as ext_force() for consistency.
    """
    F_ext_total = K_EXT * len(points)
    fp = Db.dot(Xe)
    lp = np.linalg.norm(fp, axis=1)[points]
    total_length = np.sum(lp)
    force_per_length = F_ext_total / total_length if total_length > 0 else 0

    return F_ext_total, total_length, force_per_length

def tune_F_per_segment(Xe, Db, desired_force_per_length, points=POKING_POINTS):
    """
    Calculate what K_EXT should be to achieve a desired force per unit length
    for the current epithelium geometry.
    Must use the same points array as ext_force() for consistency.
    """
    fp = Db.dot(Xe)
    lp = np.linalg.norm(fp, axis=1)[points]
    total_length = np.sum(lp)
    total_force_needed = desired_force_per_length * total_length
    k_ext_new = total_force_needed / len(points)

    return k_ext_new

@njit(parallel=True, fastmath=True, nogil=True)
def calculate_velocities(F_cc, F_on_cell):
    n_cells = len(F_cc)
    v_active = np.zeros((n_cells, 2), dtype=float64)
    for i in prange(n_cells):
        total_force_x = F_cc[i, 0] + F_on_cell[i, 0]
        total_force_y = F_cc[i, 1] + F_on_cell[i, 1]
        v_active[i, 0] = total_force_x / XI
        v_active[i, 1] = total_force_y / XI

    return v_active


@njit(parallel=True, fastmath=True, nogil=True)
def extra_damping(pos_active, v_active, left_wall):
    """Apply damping effects to velocities."""
    n_cells = len(pos_active)
    damping_zone_width = 0.5

    for i in prange(n_cells):
        x = pos_active[i, 0]
        y = pos_active[i, 1]

        if x < left_wall + damping_zone_width:
            distance_ratio = (x - left_wall) / damping_zone_width
            damping_factor = 0.1 + 0.9 * distance_ratio  # 10% to 100% speed
            v_active[i, 0] *= damping_factor

    return v_active

@njit(fastmath=True, nogil=True)
def motion_update(i, t, pos_i, v_i, cell_type, eta, eta_jammed,
                  mig_rand_x, mig_rand_y, intercal_rand,
                  migration_front):
    """Compute one cell's motion update"""

    # Brownian Motion
    if cell_type == 3:
        noise_x, noise_y = eta_jammed[i]
    else:
        noise_x, noise_y = eta[i]

    dx = v_i[0] * DT + noise_x
    dy = v_i[1] * DT + noise_y

    should_migrate = False
    should_intercal = False

    if cell_type == 1:
        if REGULATION_FRONT_FLAG:
            should_migrate = pos_i[0] > migration_front and t >= MIGRATION_DELAY
        else:
            should_migrate = t >= MIGRATION_DELAY
    elif cell_type == 2:
        should_intercal = (np.abs(pos_i[1]) > 0.5 or pos_i[0] > -0.5) and t >= INTERCAL_DELAY

    # Distal-Oriented Migration
    if should_migrate:
        if MIGRATION_DIRECTION == 'x':
            dx += mig_rand_x[i]
        elif MIGRATION_DIRECTION == 'y':
            dy += mig_rand_y[i]

    # Intercalation
    elif should_intercal:
        if pos_i[1] > 0:
            dy -= intercal_rand[i]
        elif pos_i[1] < 0:
            dy += intercal_rand[i]

    return dx, dy

@njit(parallel=True, fastmath=True, nogil=True)
def update_positions(t, pos_active, v_active, cell_types_active, migration_front=np.inf):
    """Computes random motion and updates positions"""
    n_cells = len(pos_active)

    # Precompute all randomness
    if JAMMING_ENABLED:
        eta = np.random.normal(0.0,SIGMA_UNJAMMED *np.sqrt(DT),size=(n_cells,2))
    else:
        eta = np.random.normal(0.0, SIGMA * np.sqrt(DT), size=(n_cells, 2))

    eta_jammed = np.random.normal(0.0,SIGMA_JAMMED *np.sqrt(DT),size=(n_cells,2))

    mig_rand_x = np.random.normal(
        MU_MIGRATION * DT,
        SIGMA_MIGRATION * np.sqrt(DT),
        size=n_cells)
    mig_rand_y = np.random.normal(
        MU_MIGRATION * DT,
        SIGMA_MIGRATION * np.sqrt(DT),
        size=n_cells)
    intercal_rand = np.random.normal(
        MU_INTERCAL * DT,
        SIGMA_INTERCAL * np.sqrt(DT),
        size=n_cells)

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

def cell_operations(t, pos_active, v_active, Xe, left_wall, reset=False,
                    step=0, cell_types_active=None):
    global pos, v
    
    F_cc = MM_Forces(pos_active, cell_types_active)
    F_on_cell, F_on_epi = ME_forces(pos_active, Xe)

    v_active = calculate_velocities(F_cc, F_on_cell)
    v_active = extra_damping(pos_active, v_active, left_wall)
    pos_active = update_positions(t, pos_active, v_active, cell_types_active, regulation_front)

    if reset:
        pos_active, v_active = hard_reset(pos_active, v_active, Xe)

    return pos_active, v_active, F_cc, F_on_cell, F_on_epi


def single_iteration(step, t):
    """Steps the simulation forward one timestep"""
    global pos, v, n_daughter, Xe, division_status, Ne, x_cut, T_DORMANT, regulation_front, blp0, blm0
    Xe_before_update = Xe.copy()
    F_cc_full = np.zeros((CELLS_MAX, 2))
    F_on_cell_full = np.zeros((CELLS_MAX, 2))

    F_on_epi = np.zeros((Ne, 2))
    lp = np.linalg.norm(Db.dot(Xe), axis=1)
    if EPI_TYPE == "viscoelastic":
        blp0, blm0 = update_L0(blp0, lp)
    if t >= T_DORMANT:
        cell_cycle(
            div_allowed=True,
            gradient=GRADIENT,
            directed_angle=ORIENTED_DIVISION_ANGLE)
    else:
        cell_cycle(div_allowed=False, gradient=GRADIENT)

    active = np.where(~np.isnan(pos[:, 0]))[0]
    pos_active = pos[active].copy()
    v_active = np.zeros((len(active), 2))

    if JAMMING_ENABLED:
        update_jammed_cells(Xe)

    cell_types_active = cell_types[active]
    if REGULATION_FRONT_FLAG or GRADIENT:
        front_delay_step = int(MIGRATION_DELAY / DT)
        front_update_interval = int(1 / DT)
        if step == front_delay_step:
            regulation_front = np.max(
                pos_active[:, 0]) - (50 / CONVERSION_FACTOR_UM)

        if step > front_delay_step and (
                step - front_delay_step) % front_update_interval == 0:
            regulation_front -= 50 / CONVERSION_FACTOR_UM

    if step % 100 == 0:
        pos_active, v_active, F_cc, F_on_cell, F_on_epi = cell_operations(
            t, pos_active, v_active, Xe, left_wall, reset=True,
            step=step, cell_types_active=cell_types_active
        )
    else:
        pos_active, v_active, F_cc, F_on_cell, F_on_epi = cell_operations(
            t, pos_active, v_active, Xe, left_wall, reset=False,
            step=step, cell_types_active=cell_types_active
        )

    if step % FRAME_SKIP == 0:
        F_cc_full[active] = F_cc.copy()
        F_on_cell_full[active] = F_on_cell.copy()
        F = F_cc_full + F_on_cell_full
    else:
        F = None

    pos[active] = pos_active.copy()
    v = np.zeros((CELLS_MAX, 2))
    v[active] = v_active

    F_elast = epithelium_elasticity(Xe, Db, DbT, blp0, blm0, dsb)
    F_ext = np.zeros((Ne, 2))
    if EXT_STRESS_FORCE and t > EXT_FORCE_DELAY:
        F_ext = ext_force(Xe)
    Fb = F_elast + F_on_epi + F_ext

    # Anchor endpoints
    Fb[0, :] = 0
    Fb[-1, :] = 0

    Xe[1:-1] += (Fb[1:-1] / XI) * DT

    Xe[0, 0] = Xe0[0, 0]
    Xe[-1, 0] = Xe0[-1, 0]
    Xe[0, 1] = Xe0[0, 1]
    Xe[-1, 1] = Xe0[-1, 1]

    active_after_cycle = np.where(~np.isnan(pos[:, 0]))[0]
    N_active = len(active_after_cycle)
    return F, v, division_status, N_active

def run_simulation():
    """Main simulation loop with data collection"""
    global pos, Xe, kb_vals, Ne, Db, blp0, blm0, dsb, regulation_front, cell_types

    # Collect config parameters
    import config
    config_params = {
        'Time Parameters': {
            'DT': config.DT,
            'TMAX': config.TMAX,
            'STEPS_TOTAL': config.STEPS_TOTAL,
        },
        'Physical Parameters': {
            'CONVERSION_FACTOR_UM': config.CONVERSION_FACTOR_UM,
            'D0': config.D0,
            'XI': config.XI,
            'K_BC_REP': config.K_BC_REP,
            'K_BC_ADH': config.K_BC_ADH,
            'K_CC_REP': config.K_CC_REP,
            'K_CC_ADH': config.K_CC_ADH,
            'SIGMA': config.SIGMA,
            'K_LATERAL':config.K_LATERAL,
            'KB_MAX': config.KAPPA0,
            'KB_MID': config.KAPPA1,
            'KB_MIN': config.KAPPA2,
            'LAM_VISCO': config.LAM_VISCO
        },
        'Bone/Softening': {
            'BONE_VISUALIZATION': config.BONE_VISUALIZATION,
            'ALLOW_SOFTENING': config.ALLOW_SOFTENING,
            'SPORATIC_SOFTENING':config.SPORATIC_SOFTENING
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
            'DIRECTED_DIVISION_ANGLE': config.ORIENTED_DIVISION_ANGLE,
        },
        'Migration': {
            'MIGRATION_ENABLED': config.MIGRATION_ENABLED,
            'MIGRATION_FRACTION': config.MIGRATION_FRACTION,
            'MIGRATION_DELAY': config.MIGRATION_DELAY,
            'WHICH_MIGRATION': config.WHICH_MIGRATION,
            'MIGRATION_DIRECTION': config.MIGRATION_DIRECTION,
            'MU_MIGRATION': config.MU_MIGRATION,
            'SIGMA_MIGRATION': config.SIGMA_MIGRATION,
            'REGULATION_FRONT_FLAG': config.REGULATION_FRONT_FLAG,
        },
        'Intercalation': {
            'INTERCALATION_ENABLED': config.INTERCALATION_ENABLED,
            'INTERCAL_FRACTION': config.INTERCAL_FRACTION,
            'INTERCAL_DELAY': config.INTERCAL_DELAY,
            'MU_INTERCAL': config.MU_INTERCAL,
            'SIGMA_INTERCAL': config.SIGMA_INTERCAL,
        },
        'Jamming': {
            'JAMMING_ENABLED': config.JAMMING_ENABLED,
            'FLUID_LIKE_ZONE_WIDTH': config.FLUID_LIKE_ZONE_WIDTH,
            'K_CC_REP_JAMMED': config.K_CC_REP_JAMMED,
            'K_CC_ADH_JAMMED': config.K_CC_ADH_JAMMED,
            'SIGMA_JAMMED': config.SIGMA_JAMMED,
            'SIGMA_UNJAMMED': config.SIGMA_UNJAMMED
        },
        'External Stress': {
            'EXT_STRESS_FORCE': config.EXT_STRESS_FORCE,
            'EXT_FORCE_DELAY': config.EXT_FORCE_DELAY,
            'K_EXT': config.K_EXT,
            'FORCE_PER_UNIT_LENGTH': config.FORCE_PER_UNIT_LENGTH,
            'POKING_POINTS': config.POKING_POINTS
        },
        'Output': {
            'VIDEO_FLAG': config.VIDEO_FLAG,
            'FRAME_SKIP': config.FRAME_SKIP,
            'OUTPUT_DIR': config.OUTPUT_DIR,
        }
    }

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    try:
        import inspect
        abm_path = inspect.getsourcefile(run_simulation)
        if abm_path and os.path.exists(abm_path):
            dst_abm = os.path.join(OUTPUT_DIR, 'abm_snapshot.py')
            shutil.copyfile(abm_path, dst_abm)
    except Exception as e:
        print(f"Warning: could not save abm_snapshot.py: {e}")
    try:
        save_config_snapshot(config_params,OUTPUT_DIR)
    except Exception as e:
        print(f"Warning: could not save config.py snapshot: {e}")

    if JAMMING_ENABLED:
        update_jammed_cells(Xe)

    if MIGRATION_ENABLED:
        init_migration_cells(CELLS_MAX)

    if INTERCALATION_ENABLED:
        init_intercal_cells()

    animation_manager = AnimationManager(OUTPUT_DIR, VIDEO_PARAMS, VIDEO_FLAG)
    setup_signal_handler(animation_manager)

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
    step_times = []
    ###
    rest_lengths = []
    ###
    morphometrics_data = {
        'area_growth_region': [],
        'perimeter': [],
        'AR_whole_limb': [],
        'AR_outgrowth': [],
        'ellipticity': [],
        'roundness': [],
        'a': [],
        'b': [],
        'volume_fraction': []}
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
    ###
    rest_lengths.append(blp0.copy())
    ###
    Gphase.append(cycle_phases[cycle_phases == 0].shape[0])
    Mphase.append(cycle_phases[cycle_phases == 1].shape[0])
    divisions.append(division_status.copy())
    deaths.append(death_mask.copy())
    cell_count.append(N0)
    times.append(0.0)
    phase_clocks_data.append(phase_clocks.copy())
    cycle_phases_data.append(cycle_phases.copy())
    migrant_cells_data.append((cell_types == 1).copy())
    intercal_cells_data.append((cell_types == 2).copy())
    jammed_cells_data.append((cell_types == 3).copy())
    regulation_front_history.append(regulation_front)

    start = time.time()
    last_time = start
    ext_force_printed = False
    ext_force_paused = False

    # Main simulation loop
    if EXT_STRESS_FORCE:
        print(f"Running simulation for {TMAX} more days...")
    else:
        print(f"Running simulation until {TMAX} dpa...")
    x0 = None
    poking_dict = None

    for step, t in enumerate(np.arange(0, TMAX, DT)):
        F, v, div_status, N_active = single_iteration(step, t)

        if PRINT_STEPS_FLAG and step % PRINT_STEPS_INTERVAL == 0:
            print(f"t = {t:.2f}, Step {step}/{STEPS_TOTAL}, cells: {N_active}")
        
        if (SOFTENING_SWAP_TIME is not None) and abs(t - SOFTENING_SWAP_TIME) < SOFTENING_SWAP_INTERVAL/2:
            t_ = t - (SOFTENING_SWAP_TIME - SOFTENING_SWAP_INTERVAL/2)
            kb_vals = stiffness_swap(kb_vals0, kb_vals_f, t_, swap_interval=SOFTENING_SWAP_INTERVAL) ###

        # Print total external force at EXT_FORCE_DELAY
        if EXT_STRESS_FORCE and not ext_force_printed and t >= EXT_FORCE_DELAY:
            total_force, total_length, force_per_length = calculate_total_ext_force(Xe, Db)
            print(f"\n=== External Force Applied at t={t:.4f} ===")
            print(f"Total external force: {total_force:.6f}")
            print(f"Total boundary length: {total_length:.6f}")
            print(f"Force per unit length: {force_per_length:.6f}\n")
            ext_force_printed = True
            x0 = Xe[100, 0]  # initial x before applied force
            poking_dict = {
                'total_force': total_force,
                'total_length': total_length,
                'force_per_lenth': force_per_length}

            # Pause animation at force application with updated title
            if VIDEO_FLAG and not ext_force_paused:
                animation_manager.animate_frame(step, t, pos, Xe, Xb, cycle_phases,
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
                animation_manager.animate_frame(step, t, pos, Xe, Xb, cycle_phases,
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
            ###
            rest_lengths.append(blp0.copy())
            ###
            Gphase.append(cycle_phases[cycle_phases == 0].shape[0])
            Mphase.append(cycle_phases[cycle_phases == 1].shape[0])
            divisions.append(division_status.copy())
            deaths.append(death_mask.copy())
            times.append(t)
            cell_count.append(N_active)

            area_growth_region_t, perimeter_t, AR_whole_limb_t, AR_outgrowth_t, ellipticity_t, roundness_t, a_t, b_t, volume_fraction_t = morphometrics(Xe, pos=pos, x_cut=x_cut)
            morphometrics_data['area_growth_region'].append(area_growth_region_t)
            morphometrics_data['perimeter'].append(perimeter_t)
            morphometrics_data['AR_whole_limb'].append(AR_whole_limb_t)
            morphometrics_data['AR_outgrowth'].append(AR_outgrowth_t)
            morphometrics_data['ellipticity'].append(ellipticity_t)
            morphometrics_data['roundness'].append(roundness_t)
            morphometrics_data['a'].append(a_t)
            morphometrics_data['b'].append(b_t)
            morphometrics_data['volume_fraction'].append(volume_fraction_t)

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
    print(f'Simulation finished. Time elapsed: {readable_time}')

    animation_manager.close()

    if EXT_STRESS_FORCE:
        xf = Xe[100, 0]  # final x after applied force
        deformation = (xf - x0)
        poking_dict['deformation'] = deformation
        print(f"Deformation: {deformation}")
    else:
        deformation = None
        poking_dict = None

    Xe_growth = Xe[Xe[:, 0] > x_cut]
    if len(Xe_growth) > 0:
        coefficients_growth = coefficients(Xe_growth, n=5, type='data', rotate=True)  # Chebyshev coefficients
    else:
        coefficients_growth = None

    ctrl_curve = np.loadtxt(CTRL_AVG_FILE, skiprows=1, delimiter=',')
    c59_curve = np.loadtxt(C59_AVG_FILE, skiprows=1, delimiter=',')

    data_dict = {
        'positions': np.array(positions),
        'forces': np.array(forces),
        'velocities': np.array(velocities),
        'boundaries': np.array(boundaries),
        ###
        'rest_lengths': np.array(rest_lengths),
        ###
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
        'Xe_final': Xe.copy(),
        'Xe_growth': Xe_growth,
        'pos0': pos0,
        'pos_final': pos.copy(),
        'x_cut': x_cut,
        'poking_dict': poking_dict,
        'N0': N0,
        'rmse_ctrl': distance_metric(curve1=Xe_growth*CONVERSION_FACTOR_UM, curve2=ctrl_curve, which='rmse') if len(Xe_growth) > 0 else None,
        'rmse_c59': distance_metric(curve1=Xe_growth*CONVERSION_FACTOR_UM, curve2=c59_curve, which='rmse') if len(Xe_growth) > 0 else None,
        'haus_ctrl': distance_metric(curve1=Xe_growth*CONVERSION_FACTOR_UM, curve2=ctrl_curve, which='hausdorff') if len(Xe_growth) > 0 else None,
        'haus_c59': distance_metric(curve1=Xe_growth*CONVERSION_FACTOR_UM, curve2=c59_curve, which='hausdorff') if len(Xe_growth) > 0 else None,
        'OUTPUT_DIR':OUTPUT_DIR
    }

    data_dict['config_params'] = config_params

    save_files(
        data_dict,
        config_params,
        save_data_dict=SAVE_DATA_DICT,
        save_figures=SAVE_FIGURES)
    return data_dict
if __name__=='__main__':
    run_simulation()