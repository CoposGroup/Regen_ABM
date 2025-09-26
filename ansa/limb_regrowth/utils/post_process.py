'''
Limb Regeneration Simulation - Post Processing
Ansa Brews-Smith, May 2025
Copos Lab, Northeastern University

'''
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import seaborn as sns
import pandas as pd
import numpy as np
# from freud.msd import MSD
from scipy.spatial import ConvexHull
from skimage.measure import EllipseModel, ransac
from scipy.optimize import curve_fit
import os

# Remove import of OUTPUT_DIR and FRAME_SKIP from config
# from config import DL_CRIT, KDEATH, K_MIGRATE, DT, TMAX, T_DORMANT, FRAME_SKIP, OUTPUT_DIR, XMIN, XMAX, YMIN, YMAX, M_LENGTH, G_LENGTH
from config import DL_CRIT, KDEATH, DT, TMAX, T_DORMANT, XMIN, XMAX, YMIN, YMAX, M_LENGTH, G_LENGTH_MAX, G_LENGTH_MIN, GRADIENT, KB_MAX, KB_MIN
G_LENGTH = (G_LENGTH_MIN + G_LENGTH_MAX) / 2
from matplotlib.colors import Normalize, ListedColormap

def truncate_colormap(cmap, minval=0.0, maxval=10.0, n=256):
    new_cmap = ListedColormap(cmap(np.linspace(minval, maxval, n)))
    return new_cmap

# projection method
# def ellipse_axes(pos):
#     """
#     Estimate semi‐major (a) and semi‐minor (b) axes by:
#       1. Finding the point furthest from the centroid → defines major axis direction.
#       2. Projecting all points onto the perpendicular direction → defines minor axis span.
#     """

#     pts = pos.copy()
#     # 2) Center at the centroid
#     centroid = pts.mean(axis=0)
#     centered = pts - centroid

#     # 3) Find the furthest point from centroid
#     dists = np.linalg.norm(centered, axis=1)
#     i_furthest = np.argmax(dists)
#     a = dists[i_furthest]                   # semi-major axis length

#     # 4) Direction of major axis
#     v_major = centered[i_furthest] / a      # unit vector along major axis

#     # 5) Perpendicular direction
#     v_perp = np.array([-v_major[1], v_major[0]])

#     # 6) Project all points onto v_perp to get minor‐axis span
#     proj = centered @ v_perp
#     b = (proj.max() - proj.min()) / 2       # semi-minor axis length

#     return a, b


# covariance method
# def ellipse_axes(pos):
#     """
#     Estimate semi-major (a) and semi-minor (b) axes using the covariance of the convex hull points.
#     """

#     # Use only the convex hull points
#     hull = ConvexHull(pos)
#     hull_pts = pos[hull.vertices]

#     # Center at the centroid
#     centroid = hull_pts.mean(axis=0)
#     centered = hull_pts - centroid

#     # Covariance matrix and eigenvalues
#     cov = np.cov(centered, rowvar=False)
#     eigvals, eigvecs = np.linalg.eigh(cov)
#     # Sort eigenvalues (largest first)
#     order = np.argsort(eigvals)[::-1]
#     eigvals = eigvals[order]

#     # Semi-axes lengths (2*sqrt(eigenvalue) for full width)
#     a = 2 * np.sqrt(eigvals[0])
#     b = 2 * np.sqrt(eigvals[1])

#     return a, b

# ransac method
# def ellipse_axes(pos, residual_threshold=0.26):
#     # EllipseModel expects (x,y)
#     model, inliers = ransac(
#         data=pos,
#         model_class=EllipseModel,
#         min_samples=5,
#         residual_threshold=residual_threshold,
#         max_trials=1000
#     )
#     yc, xc, a, b, theta = model.params  # center (yc,xc), axes a,b

#     model_all = EllipseModel()
#     ok = model_all.estimate(pos)
#     if not ok:
#         raise RuntimeError("Initial ellipse fit failed")

#     # 2) Compute algebraic residuals on each point
#     resids = model_all.residuals(pos)

#     # 3) Look at percentiles
#     import numpy as np
#     p90 = np.percentile(resids, 90)
#     p95 = np.percentile(resids, 95)
#     print(f"90th pct residual = {p90:.3g}, 95th pct = {p95:.3g}")


#     return a, b


def ellipse_axes(points, tol=1e-8, max_iter=1000):
    """
    Compute center, semi-axes (a, b), and rotation angle theta of the
    minimum-area enclosing ellipse for a set of 2D points.
    """
    # Khachiyan's algorithm
    N, d = points.shape
    Q = np.column_stack([points, np.ones(N)])
    u = np.full(N, 1/N)
    
    for _ in range(max_iter):
        X = (Q.T * u) @ Q
        M = np.einsum('ij,ji->i', Q @ np.linalg.inv(X), Q.T)
        j = np.argmax(M)
        step = (M[j] - d - 1) / ((d + 1) * (M[j] - 1))
        if step <= tol:
            break
        u = (1 - step)*u
        u[j] += step

    center = u @ points
    P = points - center
    A = np.linalg.inv((P.T * u) @ P - np.outer(center, center)) / d
    
    # Extract axes and angle
    eigvals, eigvecs = np.linalg.eigh(A)
    axes = 1.0 / np.sqrt(eigvals)
    order = np.argsort(axes)[::-1]
    a, b = axes[order]
    v = eigvecs[:, order[0]]
    theta = np.arctan2(v[1], v[0])
    
    return  center, a, b, theta





def morphometrics(Xe, pos=None, x_cut=1.0):
    """
    Calculate morphometrics of the blastema growth region.
    
    Parameters:
    -----------
    Xe_growth : np.ndarray
        Boundary points of the full final epithelium
    pos : np.ndarray, optional
        Cell positions (not used in current implementation)
    x_cut : float, default=1.0
        Amputation plane position - area calculated for region x > x_cut
        
    Returns:
    --------
    area, perimeter, aspect_ratio, ellipticity, roundness, a, b, volume_fraction
    """
    
    # Filter to get only the boundary points past the amputation plane
    growth_boundary = Xe[Xe[:, 0] > x_cut]
    
    if len(growth_boundary) < 3:
        # Not enough points for meaningful calculation
        return 0, 0, 0, 0, 0, 0, 2.5, 0
    
    # Create a closed polygon for the growth region
    # The polygon should be: amputation line -> growth boundary -> back to amputation line
    
    # Sort growth boundary points by y-coordinate to create proper polygon
    growth_boundary = growth_boundary[np.argsort(growth_boundary[:, 1])]
    
    y_min = growth_boundary[:, 1].min()
    y_max = growth_boundary[:, 1].max()
    
    # Create closed polygon vertices
    polygon_vertices = []
    
    # Start at bottom of amputation plane
    polygon_vertices.append([x_cut, y_min])
    
    # Add growth boundary points (already sorted by y)
    polygon_vertices.extend(growth_boundary.tolist())
    
    # End at top of amputation plane
    polygon_vertices.append([x_cut, y_max])
    
    # Close the polygon by returning to start
    polygon_vertices.append([x_cut, y_min])
    
    # Convert to arrays for calculation
    polygon_vertices = np.array(polygon_vertices)
    x_poly = polygon_vertices[:, 0]
    y_poly = polygon_vertices[:, 1]
    
    # Calculate area using shoelace formula on closed polygon
    area = 0.5 * np.abs(np.sum(x_poly[:-1] * y_poly[1:]) - np.sum(y_poly[:-1] * x_poly[1:]))
    
    # Calculate perimeter of the growth boundary only (not including amputation line)
    if len(growth_boundary) > 1:
        boundary_diffs = np.diff(growth_boundary, axis=0)
        perimeter = np.sum(np.sqrt(boundary_diffs[:, 0]**2 + boundary_diffs[:, 1]**2))
    else:
        perimeter = 0
    
    # Calculate morphometric parameters based on growth region extent
    x_growth = growth_boundary[:, 0]
    y_growth = growth_boundary[:, 1]
    
    try:
        # 'a' is the outgrowth length (how far past amputation plane)
        a = np.max(x_growth) - x_cut if len(x_growth) > 0 else 0
        # 'b' is half the width of the growth region
        b = (np.max(y_growth) - np.min(y_growth)) / 2 if len(y_growth) > 0 else 2.5
    except:
        a = 0
        b = 2.5

    if len(x_growth) > 2 and area > 0:
        aspect_ratio = a / b if b > 0 else 0
        ellipticity = (a - b) / a if a > 0 else 0
        # Roundness based on growth boundary perimeter and growth region area
        roundness = (perimeter**2) / (4 * np.pi * area) if area > 0 else 0
        volume_fraction = ((DL_CRIT/2)**2) / (a**2) if a > 0 else 0  # cell radius^2 / a^2
    else:
        area = perimeter = aspect_ratio = ellipticity = roundness = volume_fraction = 0

    return area, perimeter, aspect_ratio, ellipticity, roundness, a, b, volume_fraction

def density_heatmap(
    kb_vals, pos, Xe, x_cut,
    bin_size=0.1, OUTPUT_DIR='.', shading='auto', conversion_factor_um=200, show_real_units=True,
    fig_mode=False, show_title=True, show_colorbar=True, show_info_box=True
):
    """
    Create a heatmap of cell density distribution with boundary overlay,
    with no extra whitespace around the axes. Now uses the same kb_vals logic
    as the fixed regenerate_density_plots.py.
    
    Parameters:
    -----------
    kb_vals : array
        Boundary stiffness values
    pos : array
        Cell positions
    Xe : array  
        Epithelium boundary points
    x_cut : float
        Amputation plane position
    bin_size : float, default=0.1
        Histogram bin size
    OUTPUT_DIR : str, default='.'
        Output directory for saving plots
    shading : str, default='auto'
        Pcolormesh shading type ('auto' or 'gouraud')
    conversion_factor_um : float, default=200
        Conversion factor from simulation units to micrometers
    show_real_units : bool, default=True
        Whether to show axes in real units (micrometers)
    fig_mode : bool, default=False
        Legacy parameter - when True, disables title, colorbar, and info box
    show_title : bool, default=True
        Whether to show plot title
    show_colorbar : bool, default=True
        Whether to show colorbar
    show_info_box : bool, default=True
        Whether to show stats info box
    """
    
    # Convert data to display units if needed
    if show_real_units:
        pos_display = pos * conversion_factor_um
        Xe_display = Xe * conversion_factor_um if Xe is not None else None
        x_cut_display = x_cut * conversion_factor_um if x_cut is not None else None
        xmin_display = XMIN * conversion_factor_um
        xmax_display = XMAX * conversion_factor_um
        ymin_display = YMIN * conversion_factor_um
        ymax_display = YMAX * conversion_factor_um
        bin_size_display = bin_size * conversion_factor_um
    else:
        pos_display = pos
        Xe_display = Xe
        x_cut_display = x_cut
        xmin_display = XMIN
        xmax_display = XMAX
        ymin_display = YMIN
        ymax_display = YMAX
        bin_size_display = bin_size

    # 1) compute 2D histogram using display coordinates
    x_edges = np.arange(xmin_display, xmax_display + bin_size_display, bin_size_display)
    y_edges = np.arange(ymin_display, ymax_display + bin_size_display, bin_size_display)
    H, xedges, yedges = np.histogram2d(pos_display[:, 0], pos_display[:, 1],
                                       bins=[x_edges, y_edges])

    # 2) prepare the mesh for pcolormesh
    if shading == 'gouraud':
        # For gouraud shading, extend the mesh slightly beyond edges to fill entire plot area
        x_centers = 0.5*(xedges[:-1] + xedges[1:])
        y_centers = 0.5*(yedges[:-1] + yedges[1:])
        
        # Extend centers to include edge positions for full coverage
        dx = x_centers[1] - x_centers[0] if len(x_centers) > 1 else bin_size_display
        dy = y_centers[1] - y_centers[0] if len(y_centers) > 1 else bin_size_display
        
        x_extended = np.concatenate([[x_centers[0] - dx], x_centers, [x_centers[-1] + dx]])
        y_extended = np.concatenate([[y_centers[0] - dy], y_centers, [y_centers[-1] + dy]])
        
        Xc, Yc = np.meshgrid(x_extended, y_extended, indexing='xy')
        
        # Extend H array to match extended mesh
        H_extended = np.zeros((len(y_extended), len(x_extended)))
        H_extended[1:-1, 1:-1] = H.T
        # Fill edges with nearest neighbor values
        H_extended[0, :] = H_extended[1, :]     # Top edge
        H_extended[-1, :] = H_extended[-2, :]   # Bottom edge  
        H_extended[:, 0] = H_extended[:, 1]     # Left edge
        H_extended[:, -1] = H_extended[:, -2]   # Right edge
        
        H_plot = H_extended
    else:
        X, Y = np.meshgrid(xedges, yedges, indexing='xy')
        H_plot = H.T

    # 3) fig+ax with constrained_layout to pack things tightly
    fig, ax = plt.subplots(constrained_layout=True, figsize=(10, 8))
    ax.set_aspect('equal', adjustable='box')

    # 4) plot the heatmap with standardized color scale
    if shading == 'gouraud':
        pcm = ax.pcolormesh(Xc, Yc, H_plot, cmap='plasma', shading=shading, vmin=0, vmax=20)
    else:
        pcm = ax.pcolormesh(X, Y, H_plot, cmap='plasma', shading=shading, vmin=0, vmax=20)
    
    # Add colorbar based on parameters (slideshow_mode overrides show_colorbar)
    if show_colorbar and not fig_mode:
        cbar = fig.colorbar(pcm, ax=ax, label='Cell count per bin')

    # 5) amputation plane (no label for cleaner slideshow look)
    if x_cut_display is not None:
        ax.axvline(x=x_cut_display, color='black', linestyle='--', linewidth=2)

    # 6) overlay boundary segments using FIXED kb_vals logic (same as regenerate_density_plots.py)
    if Xe_display is not None:
        if kb_vals is not None:
            # Use standardized color limits that match animations exactly
            vmin_kb = KB_MIN  # 1.0 - always maps to lightest purple
            vmax_kb = KB_MAX  # 150.0 - always maps to darkest purple
            
            norm = mcolors.Normalize(vmin=vmin_kb, vmax=vmax_kb, clip=True)
            cmap = truncate_colormap(cm.get_cmap('PuRd'), 0.2, 1.0)
            for i in range(len(Xe_display)-1):
                color = cmap(norm(kb_vals[i]))
                ax.plot([Xe_display[i,0], Xe_display[i+1,0]], 
                        [Xe_display[i,1], Xe_display[i+1,1]], 
                        '-', lw=2, color=color)
        else:
            # Plot boundary without color coding
            ax.plot(Xe_display[:, 0], Xe_display[:, 1], 'r-', lw=2, label='Boundary')

    # 7) axes formatting - set exact limits to eliminate whitespace and use display units
    ax.set_xlim(xmin_display, xmax_display)
    ax.set_ylim(ymin_display, ymax_display)
    
    if show_real_units:
        ax.set_xlabel('x (μm)', fontsize=16, fontweight='bold')
        ax.set_ylabel('y (μm)', fontsize=16, fontweight='bold')
    else:
        ax.set_xlabel('x', fontsize=16, fontweight='bold')
        ax.set_ylabel('y', fontsize=16, fontweight='bold')
    
    ax.tick_params(axis='both', which='major', labelsize=14)
    
    # Add title based on parameters (slideshow_mode overrides show_title)
    if show_title and not fig_mode:
        ax.set_title('Cell Density Distribution')
    
    ax.grid(True, alpha=0.3)

    # No legend for cleaner slideshow look
    # if x_cut_display is not None:
    #     ax.legend(loc='upper right')

    # 8) stats textbox based on parameters (slideshow_mode overrides show_info_box)
    if show_info_box and not fig_mode:
        active = np.count_nonzero(~np.isnan(pos[:,0]))
        stats = (
            f'Total cells: {active}\n'
            f'Max density: {H.max():.0f} cells/bin\n'
            f'Mean density: {H[H>0].mean():.1f} cells/bin\n'
            f'Bin size: {bin_size_display:.0f}×{bin_size_display:.0f}'
        )
        ax.text(0.02, 0.98, stats,
                transform=ax.transAxes,
                va='top',
                bbox=dict(boxstyle='round', fc='white', alpha=0.8))

    # 9) save with tight layout to minimize whitespace
    fig.savefig(f'{OUTPUT_DIR}/cell_density_{shading}.png',
                dpi=300, bbox_inches='tight', pad_inches=0.1)
    plt.close(fig)
    if shading == 'gouraud':
        return Xc, Yc, H
    else:
        return X, Y, H


# def cycle_plot(times, cell_count, N0, migration=False, OUTPUT_DIR=OUTPUT_DIR):
#     """Create a plot of cell count vs time with analytical comparison"""
#     def analytical1(t):
#         return np.where(
#             t < T_DORMANT,
#             N0,
#             N0 * np.exp((KDIV - KDEATH) * (t - T_DORMANT))
#         )


#     times, cell_count = np.array(times), np.array(cell_count)
#     fig, ax = plt.subplots(figsize=(8,5))
#     # if migration:
#     #     ax.plot(times, analytical2(times), '--', label='Analytical (migration)', lw=2)
    
#     ax.plot(times, cell_count, label='Gillespie ABM', lw=2)
#     ax.plot(times, analytical1(times), '--', label='Analytical (no migration)', lw=2)

#     ax.set_xlabel('Time')
#     ax.set_ylabel('Number of Cells')
#     ax.set_title('Cell Count vs Time')
#     ax.legend()
#     ax.set_xlim(0, TMAX)
#     ax.grid(True)
#     out_path = os.path.join(OUTPUT_DIR, 'cell_count_vs_time.png')
#     fig.savefig(out_path, dpi=300)
#     plt.close(fig)

# def cycle_plot(times, cell_count, N0, migration=False, OUTPUT_DIR=None):


#     times = np.array(times)
#     cell_count = np.array(cell_count)

#     # For aging transition p(age) = 1/(phase_length - age + 1), expected time = (phase_length + 2)/2
#     # avg_G_time = (G_LENGTH + 2) / 2 * DT
#     # avg_M_time = (M_LENGTH + 2) / 2 * DT
#     # k1, k2 = 1.0 / avg_G_time, 1.0 / avg_M_time
#     k1 = 1.0 / (G_LENGTH * DT)
#     k2 = 1.0 / (M_LENGTH * DT)

#     # analytic G and M from our two‐compartment model
#     def G_sol(t, k1, k2, kdeath, G0, M0):
#         delta = np.sqrt(k1*k1 + 6*k1*k2 + k2*k2)
#         rm = (k1 + k2 + 2*kdeath - delta) / 2.0
#         rp = (k1 + k2 + 2*kdeath + delta) / 2.0
#         Cplus  = 2*G0*k1 + M0*k1 - M0*k2 + M0*delta
#         Cminus = 2*G0*k1 + M0*k1 - M0*k2 - M0*delta
#         Fminus = ((k2+kdeath)/k1 - ((k1+k2)/2.0 + kdeath - delta/2.0)/k1) * Cplus
#         Fplus  = ((k2+kdeath)/k1 - ((k1+k2)/2.0 + kdeath + delta/2.0)/k1) * Cminus
#         return (np.exp(-rm*t)*Fminus - np.exp(-rp*t)*Fplus) / (2.0*delta)

#     def M_sol(t, k1, k2, kdeath, G0, M0):
#         delta = np.sqrt(k1*k1 + 6*k1*k2 + k2*k2)
#         rm = (k1 + k2 + 2*kdeath - delta) / 2.0
#         rp = (k1 + k2 + 2*kdeath + delta) / 2.0
#         Cplus  = 2*G0*k1 + M0*k1 - M0*k2 + M0*delta
#         Cminus = 2*G0*k1 + M0*k1 - M0*k2 - M0*delta
#         return (np.exp(-rm*t)*Cplus - np.exp(-rp*t)*Cminus) / (2.0*delta)

#     # total analytic N(t) = G(t)+M(t), with dormancy
#     def N_analytic(t):
#         t = np.array(t)
#         N = np.empty_like(t, dtype=float)
#         pre = t < T_DORMANT
#         post = ~pre
#         N[pre] = N0 * np.exp(-KDEATH * t[pre])
#         tt = t[post] - T_DORMANT
        
#         # Handle case where there are no pre-dormancy time points
#         if np.any(pre):
#             G1 = N[pre][-1]
#         else:
#             # If no pre-dormancy points, use initial condition
#             G1 = N0 * np.exp(-KDEATH * T_DORMANT)
#         M1 = 0

#         N[post] = G_sol(tt, k1, k2, KDEATH, G1, M1) + M_sol(tt, k1, k2, KDEATH, G1, M1)
#         return N

#     # plot
#     fig, ax = plt.subplots(figsize=(8,5))
#     ax.plot(times, cell_count, label='ABM cell count', lw=2)
#     ax.plot(times, N_analytic(times), '--', label='Analytical G+M', lw=2)

#     ax.set_xlabel('Time')
#     ax.set_ylabel('Number of Cells')
#     ax.set_title('Cell Count vs Time')
#     ax.legend()
#     ax.set_xlim(0, TMAX)
#     ax.grid(True)

#     out_path = os.path.join(OUTPUT_DIR, 'cell_count_vs_time.png') if OUTPUT_DIR is not None else 'cell_count_vs_time.png'
#     fig.savefig(out_path, dpi=300)
#     plt.close(fig)

def cycle_plot2(times, cell_count, N0, OUTPUT_DIR=None):
    """
    Plot cell count vs time with a numerical solution of the two-compartment ODE system.

    ODEs (post dormancy):
        dG/dt = (-(k1+kdeath)) * G + 2*k2 * M
        dM/dt = k1 * G - (k2 + kdeath) * M

    Dormancy (t < T_DORMANT): no cycling/division, only death
        dG/dt = -kdeath * G,  dM/dt = -kdeath * M (with M(0)=0)
    """
    times = np.array(times)
    cell_count = np.array(cell_count)

    # Rates consistent with existing analytics
    k1 = 1.0 / (G_LENGTH * DT)
    k2 = 1.0 / (M_LENGTH * DT)
    kdeath = KDEATH

    # Storage
    G_arr = np.zeros_like(times, dtype=float)
    M_arr = np.zeros_like(times, dtype=float)

    # Initial conditions
    G = float(N0)
    M = 0.0
    G_arr[0] = G
    M_arr[0] = M

    def euler_step(G, M, dt, before_dormancy):
        if before_dormancy:
            dG = -kdeath * G
            dM = -kdeath * M
        else:
            dG = (-(k1 + kdeath)) * G + 2.0 * k2 * M
            dM = k1 * G - (k2 + kdeath) * M
        return G + dt * dG, M + dt * dM

    for i in range(1, len(times)):
        t_prev = times[i - 1]
        t_next = times[i]
        dt = float(t_next - t_prev)

        # If interval crosses the dormancy boundary, split once at T_DORMANT
        if (t_prev < T_DORMANT) and (t_next > T_DORMANT):
            dt1 = T_DORMANT - t_prev
            dt2 = t_next - T_DORMANT
            G, M = euler_step(G, M, dt1, before_dormancy=True)
            G, M = euler_step(G, M, dt2, before_dormancy=False)
        else:
            before = (t_prev < T_DORMANT)
            G, M = euler_step(G, M, dt, before_dormancy=before)

        G_arr[i] = G
        M_arr[i] = M

    N_numeric = G_arr + M_arr

    # Plot
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(times, cell_count, label='ABM cell count', lw=2)
    ax.plot(times, N_numeric, '--', label='ODE numeric (G+M)', lw=2)
    ax.set_xlabel('Time')
    ax.set_ylabel('Number of Cells')
    ax.set_title('Cell Count vs Time (Numeric ODE)')
    ax.legend()
    ax.set_xlim(0, TMAX)
    ax.grid(True)

    out_path = os.path.join(OUTPUT_DIR, 'cell_count_vs_time_numeric.png') if OUTPUT_DIR is not None else 'cell_count_vs_time_numeric.png'
    fig.savefig(out_path, dpi=300)
    plt.close(fig)

def phase_distribution_plot(times, Gphase, Mphase, fit=False, OUTPUT_DIR=None):
    # array‐ify
    times  = np.array(times)
    Gphase = np.array(Gphase)
    Mphase = np.array(Mphase)

    # initial counts
    G0 = Gphase[0]
    M0 = Mphase[0]

    # analytic building blocks
    def G_sol(t, k1, k2, kdeath, G0, M0):
        delta = np.sqrt(k1*k1 + 6*k1*k2 + k2*k2)
        rm = (k1 + k2 + 2*kdeath - delta) / 2.0
        rp = (k1 + k2 + 2*kdeath + delta) / 2.0
        Cplus  = 2*G0*k1 + M0*k1 - M0*k2 + M0*delta
        Cminus = 2*G0*k1 + M0*k1 - M0*k2 - M0*delta
        Fminus = ((k2+kdeath)/k1 - ((k1+k2)/2.0 + kdeath - delta/2.0)/k1) * Cplus
        Fplus  = ((k2+kdeath)/k1 - ((k1+k2)/2.0 + kdeath + delta/2.0)/k1) * Cminus
        return (np.exp(-rm*t)*Fminus - np.exp(-rp*t)*Fplus) / (2.0*delta)

    def M_sol(t, k1, k2, kdeath, G0, M0):
        delta = np.sqrt(k1*k1 + 6*k1*k2 + k2*k2)
        rm = (k1 + k2 + 2*kdeath - delta) / 2.0
        rp = (k1 + k2 + 2*kdeath + delta) / 2.0
        Cplus  = 2*G0*k1 + M0*k1 - M0*k2 + M0*delta
        Cminus = 2*G0*k1 + M0*k1 - M0*k2 - M0*delta
        return (np.exp(-rm*t)*Cplus - np.exp(-rp*t)*Cminus) / (2.0*delta)
    # choose or fit parameters
    if fit:
        # prepare only post‑dormancy data for fitting
        mask_fit = times >= T_DORMANT
        t_fit    = times[mask_fit] - T_DORMANT
        yG_fit   = Gphase[mask_fit]
        yM_fit   = Mphase[mask_fit]

        # combined model for curve_fit
        def combined_model(t_concat, k1, k2):
            N = len(t_fit)
            t_shifted = t_concat[:N]
            return np.concatenate([
                G_sol(t_shifted, k1, k2, KDEATH, G0, M0),
                M_sol(t_shifted, k1, k2, KDEATH, G0, M0)
            ])

        # stack data and times
        tdata = np.concatenate([t_fit, t_fit])
        ydata = np.concatenate([yG_fit, yM_fit])

        # fit starting guess
        p0 = [0.5, KDEATH]
        popt, _ = curve_fit(combined_model, tdata, ydata, p0=p0)
        k1_fit, k2_fit = popt
        title_suffix = f"(fitted k1={k1_fit:.3g}, k2={k2_fit:.3g})"
    else:
        # use config constants
        k1_fit, k2_fit = 1.0 / (G_LENGTH*DT) , 1.0 / (M_LENGTH*DT)
        title_suffix = f"(analytical k1={k1_fit:.3g}, k2={k2_fit:.3g})"

    # build full analytic curves including dormancy
    G_analytic = np.empty_like(times)
    M_analytic = np.empty_like(times)

    pre_mask  = times < T_DORMANT
    post_mask = ~pre_mask
    # dormancy flat
    G_analytic[pre_mask] = G0 * np.exp(-KDEATH * times[pre_mask])
    M_analytic[pre_mask] = 0
    
    # Handle case where there are no pre-dormancy time points
    if np.any(pre_mask):
        G1 = G_analytic[pre_mask][-1]
        M1 = M_analytic[pre_mask][-1]
    else:
        # If no pre-dormancy points, use initial condition
        G1 = G0 * np.exp(-KDEATH * T_DORMANT)
        M1 = 0
    # post‑dormancy
    t_post = times[post_mask] - T_DORMANT
    G_analytic[post_mask] = G_sol(t_post, k1_fit, k2_fit, KDEATH, G1, M1)
    M_analytic[post_mask] = M_sol(t_post, k1_fit, k2_fit, KDEATH, G1, M1)

    # plot simulation vs. analytic/fit
    fig, ax = plt.subplots(figsize=(8,5))
    ax.plot(times, Gphase,     '-',  label='G0/G1 (sim)', color=(223/255, 224/255, 95/255))
    ax.plot(times, Mphase,     '-',  label='S/G2/M (sim)', color=(120/255, 237/255, 240/255))
    ax.plot(times, G_analytic, '--', label='G0/G1 (model)', color=(223/255, 224/255, 95/255))
    ax.plot(times, M_analytic, '--', label='S/G2/M (model)', color=(120/255, 237/255, 240/255))

    ax.set_xlabel('Time')
    ax.set_ylabel('Cell Count')
    ax.set_title(f'Phase Distributions {title_suffix}')
    ax.legend()
    ax.set_xlim(0, TMAX)
    ax.grid(True)

    if fit:
        name = 'phase_distr_fit.png'
    else:
        name = 'phase_distr.png'

    out_path = os.path.join(OUTPUT_DIR, name) if OUTPUT_DIR is not None else name
    fig.savefig(out_path, dpi=300)
    plt.close(fig)





def plot_ellipse(ax, model, **line_kwargs):
    """
    Given a fitted EllipseModel, sample points around it and plot.
    """
    yc, xc, a, b, theta = model.params
    t = np.linspace(0, 2*np.pi, 200)
    cos_t, sin_t = np.cos(t), np.sin(t)
    # Parametric ellipse (rotated)
    x_ell = xc +  a*cos_t*np.cos(theta) - b*sin_t*np.sin(theta)
    y_ell = yc +  a*cos_t*np.sin(theta) + b*sin_t*np.cos(theta)
    ax.plot(x_ell, y_ell, **line_kwargs)

def boundary_plot(Xe0, Xe_final, Xe_growth, Xb, x_cut, aspect_ratio, area,
                  roundness, a, perimeter, ellipticity,
                  OUTPUT_DIR, pos0=None, pos_final=None, bone_enabled=False, 
                  conversion_factor_um=200, show_real_units=True):
    """
    Plots either the growth-region trace (if Xe_growth present) or
    the initial/final clusters with their minimum enclosing ellipses.
    Now supports real units conversion and outgrowth length annotation.
    Uses 'a' parameter from morphometrics as the outgrowth length.
    """
    # Use the 'a' parameter (already calculated outgrowth length) and convert to real units
    outgrowth_length_um = a * conversion_factor_um
    
    fig, ax = plt.subplots(figsize=(6,6))
    ax.set_aspect('equal', adjustable='box')
    ax.axis('equal')
    
    # compute scatter marker size
    cell_radius = DL_CRIT / 2
    
    # Set axis limits and labels based on units
    if show_real_units:
        # Convert simulation units to micrometers for display
        xmin_um = XMIN * conversion_factor_um
        xmax_um = XMAX * conversion_factor_um
        ymin_um = YMIN * conversion_factor_um
        ymax_um = YMAX * conversion_factor_um
        x_cut_um = x_cut * conversion_factor_um
        
        bbox = ax.get_window_extent().transformed(ax.figure.dpi_scale_trans.inverted())
        ppf = (bbox.width * 72) / (xmax_um - xmin_um)
        area_pts2 = np.pi * (cell_radius * conversion_factor_um * ppf / 2)**2
        
        ax.set_xlim(xmin_um, xmax_um)
        ax.set_ylim(ymin_um, ymax_um)
        ax.set_xlabel('x (μm)', fontsize=16, fontweight='bold')
        ax.set_ylabel('y (μm)', fontsize=16, fontweight='bold')
        ax.tick_params(axis='both', which='major', labelsize=14)
        
        # Convert boundary coordinates to micrometers
        Xe0_display = Xe0 * conversion_factor_um if Xe0 is not None else None
        Xe_final_display = Xe_final * conversion_factor_um if Xe_final is not None else None
        Xe_growth_display = Xe_growth * conversion_factor_um if Xe_growth is not None else None
        Xb_display = Xb * conversion_factor_um if Xb is not None else None
    else:
        # Use simulation units
        bbox = ax.get_window_extent().transformed(ax.figure.dpi_scale_trans.inverted())
        ppf = (bbox.width * 72) / (XMAX - XMIN)
        area_pts2 = np.pi * (cell_radius * ppf / 2)**2
        
        ax.set_xlim(XMIN, XMAX)
        ax.set_ylim(YMIN, YMAX)
        ax.set_xlabel('x', fontsize=16, fontweight='bold')
        ax.set_ylabel('y', fontsize=16, fontweight='bold')
        ax.tick_params(axis='both', which='major', labelsize=14)
        
        # Use original coordinates
        Xe0_display = Xe0
        Xe_final_display = Xe_final
        Xe_growth_display = Xe_growth
        Xb_display = Xb
        x_cut_um = x_cut

    if Xe_growth is not None and Xe_growth_display is not None:
        ax.plot(Xe0_display[:,0], Xe0_display[:,1], 'k-', lw=2, alpha=1.0, label='Initial boundary')
        ax.plot(Xe_final_display[:,0], Xe_final_display[:,1], 'b-', lw=2, label='Final boundary')
        ax.plot(Xe_growth_display[:,0], Xe_growth_display[:,1],'r-', lw=2, label='Growth region')
        
        # Add shaded outgrowth region inside the limb (x_cut < x < boundary)
        # Create a polygon that fills the area between x_cut and the final boundary
        if show_real_units:
            Xe_past_cut = Xe_final_display[Xe_final_display[:, 0] > x_cut_um]
        else:
            Xe_past_cut = Xe_final_display[Xe_final_display[:, 0] > x_cut_um]
            
        if len(Xe_past_cut) > 0:
            # Create vertices for the polygon: from amputation line to boundary and back
            y_min_cut = Xe_past_cut[:, 1].min()
            y_max_cut = Xe_past_cut[:, 1].max()
            
            # Create polygon vertices: start at amputation line, follow boundary, return to amputation line
            vertices = []
            vertices.append([x_cut_um, y_min_cut])  # Start at bottom of amputation line
            
            # Add boundary points past the cut (sorted by y to create proper polygon)
            boundary_sorted = Xe_past_cut[np.argsort(Xe_past_cut[:, 1])]
            vertices.extend(boundary_sorted.tolist())
            
            vertices.append([x_cut_um, y_max_cut])  # End at top of amputation line
            
            # Create and add the polygon
            from matplotlib.patches import Polygon
            poly = Polygon(vertices, closed=True, color='red', alpha=0.4, label='Outgrowth region')
            ax.add_patch(poly)

            # Compute polygon area in display units (μm^2 if show_real_units)
            verts = np.array(vertices)
            area_poly_display = 0.5 * np.abs(np.sum(verts[:,0]*np.roll(verts[:,1], -1)) - np.sum(verts[:,1]*np.roll(verts[:,0], -1)))
        else:
            area_poly_display = None
        
        ax.axvline(x=x_cut_um, color='gray', ls='--', label='Amputation plane')
        # ax.set_title('Boundary Evolution and Growth Region')

    if Xb_display is not None and bone_enabled:
        ax.plot(Xb_display[:,0], Xb_display[:,1], 'k-', lw=2, alpha=0.5, label='Bone boundary')
    # else:
    #     def scatter_and_ellipse(pts, color, cell_label, ellipse_label):
    #         # Convert cell positions if using real units
    #         if show_real_units:
    #             pts_display = pts * conversion_factor_um
    #         else:
    #             pts_display = pts
    #         ax.scatter(pts_display[:,0], pts_display[:,1], s=area_pts2, c=color, alpha=0.3, label=cell_label)
    #         if len(pts) >= 5:
    #             c, a_e, b_e, th = ellipse_axes(pts)
    #             if show_real_units:
    #                 c = c * conversion_factor_um
    #                 a_e = a_e * conversion_factor_um
    #                 b_e = b_e * conversion_factor_um
    #             t = np.linspace(0, 2*np.pi, 200)
    #             R = np.array([[np.cos(th), -np.sin(th)],
    #                           [np.sin(th),  np.cos(th)]])
    #             ellipse_pts = (np.vstack([a_e*np.cos(t), b_e*np.sin(t)]).T @ R.T) + c
    #             ax.plot(ellipse_pts[:,0], ellipse_pts[:,1], c=color, lw=2, alpha=0.6, label=ellipse_label)

        # if pos0 is not None:
        #     pts0 = pos0[~np.isnan(pos0).any(axis=1)]
        #     scatter_and_ellipse(pts0, 'blue', 'Initial cells', 'Initial ellipse')

        # if pos_final is not None:
        #     ptsf = pos_final[~np.isnan(pos_final).any(axis=1)]
        #     scatter_and_ellipse(ptsf, 'red', 'Final cells', 'Final ellipse')

        # ax.set_title('Initial and Final Cell Clusters (Min-Enclosing Ellipse)')

        # ax.legend(loc='upper right')
    ax.grid(True, alpha=0.2)
    
    # ADD OUTGROWTH LENGTH ANNOTATION IN TOP RIGHT
    if show_real_units:
        area_display = area * (conversion_factor_um ** 2)
        area_units = 'μm²'
    else:
        area_display = area
        area_units = 'units²'

    # Prefer polygon area if we computed it from the shaded region
    try:
        if area_poly_display is not None:
            area_display = area_poly_display
            area_units = 'μm²' if show_real_units else area_units
    except NameError:
        pass

    annotation_text = (
        f'Outgrowth: {outgrowth_length_um:.1f} μm\n'
        f'Area: {area_display:.2f} {area_units}'
    )

    ax.text(0.95, 0.95, annotation_text,
            transform=ax.transAxes,
            verticalalignment='top',
            horizontalalignment='right',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='yellow', alpha=0.9, edgecolor='black'),
            fontsize=14, fontweight='bold')

    # metrics_text = (
    #     f'Growth Metrics:\n'
    #     f'Area: {area:.2f}\n'
    #     f'Aspect Ratio: {aspect_ratio:.2f}\n'
    #     f'Roundness: {roundness:.2f}\n'
    #     f'Radius/Outgrowth Length: {a:.2f}\n'
    #     f'Perimeter: {perimeter:.2f}\n'
    #     f'Ellipticity: {ellipticity:.2f}'
    # )
    # ax.text(0.02, 0.98, metrics_text, transform=ax.transAxes, va='top',
    #         bbox=dict(boxstyle='round', facecolor='white', alpha=0.8), fontsize=6)

    fig.tight_layout()
    fig.savefig(f'{OUTPUT_DIR}/growth.png', dpi=400, bbox_inches='tight', pad_inches=0)
    plt.close(fig)



def runtime_plot(cell_count, steptimes, OUTPUT_DIR=None):
    """Create a plot of runtime vs time"""
    fig, ax = plt.subplots(figsize=(8,5))
    skip = int(T_DORMANT / (DT*1000)) # default FRAME_SKIP=1000 if not provided
    min_len = min(len(cell_count[skip:]), len(steptimes[skip:]))
    x = cell_count[skip:skip+min_len]
    y = steptimes[skip:skip+min_len]
    
    # Plot with matched array lengths
    ax.scatter(x, y, label='Runtime', lw=2)
    
    coeffs = np.polyfit(x, y, 1)
    trendline = np.poly1d(coeffs)
    ax.plot(x, trendline(x), 'r--', 
            label=f'Trend: {coeffs[0]:.2e}x + {coeffs[1]:.2e}')

    ax.set_xlabel('Number of Cells')
    ax.set_ylabel('Time per Step (s)')
    ax.set_title('Runtime Scaling with Cell Count')
    ax.legend()
    ax.grid(True)

    
    out_path = os.path.join(OUTPUT_DIR, 'runtime_vs_cell_count.png') if OUTPUT_DIR is not None else 'runtime_vs_cell_count.png'
    fig.savefig(out_path, dpi=300)
    plt.close(fig)

# def MSD_plot(positions, ids=range(10)):
#     # Select subset of particles
#     if ids is not None:
#         ids = np.array(ids)
#         positions = positions[:, ids, :]

    
#     n_frames = positions.shape[0]
#     msd_values = np.zeros(n_frames)
    
#     # Compute MSD using the "window" definition
#     for m in range(n_frames):
#         displacements = positions[m:] - positions[:n_frames - m]
#         sq_displacements = np.sum(displacements**2, axis=2)
#         msd_values[m] = np.mean(sq_displacements)
    
#     # Generate lag time axis
#     times = np.arange(n_frames)
    
#     # Plot MSD
#     plt.figure()
#     plt.plot(times, msd_values)
#     plt.xlabel('Lag time (frames)')
#     plt.ylabel('MSD')
#     plt.title('Mean Squared Displacement')
#     plt.show()
    
#     return times, msd_values

def trajectory_plot(
    positions, Xe, x_cut, death_indicies, kb_vals,
    ids=range(30), OUTPUT_DIR=None,
    boundary=True
):
    """Plot trajectories of selected cells, circle at start, arrow at end if alive, × if died."""
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.set_aspect('equal', adjustable='box')
    ax.axis('equal')
    colors = plt.cm.rainbow(np.linspace(0, 1, len(ids)))

    # Precompute each cell's first death step
    death_step = {cell_id: None for cell_id in ids}
    if death_indicies is not None:
        for step, die_arr in enumerate(death_indicies):
            for cell_id in ids:
                if cell_id < die_arr.shape[0] and die_arr[cell_id]:
                    if death_step[cell_id] is None:
                        death_step[cell_id] = step

    for idx, cell_id in enumerate(ids):
        x_coords, y_coords = [], []

        # determine how far to walk
        last_step = death_step[cell_id] if death_step[cell_id] is not None else len(positions)
        for step in range(last_step):
            pos_arr = positions[step]
            if cell_id < pos_arr.shape[0] and not np.isnan(pos_arr[cell_id, 0]):
                curr_x, curr_y = pos_arr[cell_id]
                x_coords.append(curr_x)
                y_coords.append(curr_y)
            else:
                break

        # plot trajectory
        if x_coords:
            ax.plot(
                x_coords, y_coords, '--',
                markersize=2, alpha=0.5,
                color=colors[idx],
                label=f'Cell {cell_id}'
            )
            # start circle
            ax.scatter(
                x_coords[0], y_coords[0],
                s=40, marker='o',
                color=colors[idx], alpha=0.7
            )

        # if the cell died, plot × at death location; else arrow at end
        ds = death_step[cell_id]
        if ds is not None:
            # plot × at death
            pos_at_death = positions[ds]
            if cell_id < pos_at_death.shape[0] and not np.isnan(pos_at_death[cell_id, 0]):
                dx, dy = pos_at_death[cell_id]
                ax.plot(dx, dy, 'x', markersize=6, color=colors[idx], alpha=0.7)
        else:
            # alive: arrow from penultimate to last
            if len(x_coords) >= 2:
                ax.annotate(
                    '',
                    xy=(x_coords[-1], y_coords[-1]),
                    xytext=(x_coords[-2], y_coords[-2]),
                    arrowprops=dict(arrowstyle='->', color=colors[idx], lw=1.5, alpha=0.7)
                )

    # boundary overlay
    if boundary:
        ax.axvline(x=x_cut, color='black', linestyle='--', label='Amputation plane')
        hard_done = soft_done = False
        if kb_vals is not None:
            vmin = np.percentile(kb_vals, 8)  # clip out the lowest 5%
            vmax = np.max(kb_vals)
            norm = mcolors.Normalize(vmin=vmin, vmax=vmax, clip=True)
            cmap = truncate_colormap(cm.get_cmap('PuRd'), 0.2, 1.0)
            for i in range(len(Xe)-1):
                color = cmap(norm(kb_vals[i]))
                ax.plot([Xe[i,0], Xe[i+1,0]], 
                            [Xe[i,1], Xe[i+1,1]], 
                            '-', lw=2, color=color)

    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title('Cell Trajectories Over Time')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.set_xlim(XMIN, XMAX)
    ax.set_ylim(YMIN, YMAX)
    ax.grid(True)

    out_path = os.path.join(OUTPUT_DIR, 'cell_trajectories.png') if OUTPUT_DIR is not None else 'cell_trajectories.png'
    fig.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close(fig)


def MSD_plot(positions, ids=None, fit_frac=1.0, FRAME_SKIP=1000, OUTPUT_DIR=None):
    """
    Compute and plot mean squared displacement (MSD) for selected particle IDs,
    fit a power law to the MSD vs time curve, and extract the scaling exponent.
    
    Fixed version that properly handles NaN values and time units.

    Returns
    -------
    times : np.ndarray
        Lag times (in actual time units).
    msd_values : np.ndarray
        Computed MSD for each lag time.
    alpha : float
        Slope of the best‐fit line to log-log MSD data (MSD ~ t^alpha).
    intercept : float
        Intercept of the fit in log-log space.
    """
    positions = np.array(positions)
    n_frames, n_particles, dim = positions.shape
    
    # Find particles that exist throughout the simulation (no NaN values)
    valid_particles = []
    for p in range(n_particles):
        if not np.any(np.isnan(positions[:, p, :])):
            valid_particles.append(p)
    
    if len(valid_particles) == 0:
        print("Warning: No particles exist throughout entire simulation.")
        return np.array([]), np.array([]), np.nan, np.nan
    
    # Select subset of valid particles
    if ids is not None:
        # Use only requested IDs that are also valid
        ids = np.array(list(ids))
        valid_ids = [p for p in ids if p in valid_particles]
        if len(valid_ids) == 0:
            print(f"Warning: None of the requested particle IDs {ids} are valid throughout simulation.")
            print(f"Valid particles: {valid_particles[:10]}... (showing first 10)")
            # Fall back to using first few valid particles
            valid_ids = valid_particles[:min(30, len(valid_particles))]
    else:
        # Use first 30 valid particles or all if fewer than 30
        valid_ids = valid_particles[:min(30, len(valid_particles))]
    
    print(f"Computing MSD for {len(valid_ids)} particles")
    
    # Extract positions for selected valid particles
    pos = positions[:, valid_ids, :]
    n_frames, n_selected, dim = pos.shape
    
    msd_values = np.zeros(n_frames)
    
    # Compute MSD for each lag time
    for m in range(n_frames):
        if n_frames - m <= 1:
            msd_values[m] = np.nan
            continue
            
        # Calculate displacements for lag time m
        displacements = pos[m:] - pos[:n_frames - m]
        # Square displacements and sum over dimensions
        sq_displacements = np.sum(displacements**2, axis=2)
        # Average over particles and time origins
        msd_values[m] = np.mean(sq_displacements)
    
    # Create proper time array (in actual time units)
    times = np.arange(n_frames) * FRAME_SKIP * DT
    
    # Remove NaN values for fitting
    valid_mask = ~np.isnan(msd_values) & (times > 0) & (msd_values > 0)
    if np.sum(valid_mask) < 3:
        print("Warning: Not enough valid data points for MSD fit.")
        alpha, intercept = np.nan, np.nan
        fit_t, fit_msd = np.array([]), np.array([])
    else:
        valid_times = times[valid_mask]
        valid_msd = msd_values[valid_mask]
        
        # Determine fit region: middle portion of the data for better fitting
        start_idx = max(1, int(len(valid_times) * 0.1))  # Skip first 10%
        end_idx = int(len(valid_times) * 0.9)  # Use up to 90%
        
        fit_t = valid_times[start_idx:end_idx]
        fit_msd = valid_msd[start_idx:end_idx]
        
        if len(fit_t) < 3:
            print("Warning: Not enough data points in fitting region.")
            alpha, intercept = np.nan, np.nan
        else:
            # Fit in log-log space
            log_t = np.log(fit_t)
            log_msd = np.log(fit_msd)
            alpha, intercept = np.polyfit(log_t, log_msd, 1)
    
    # Create plot
    plt.figure(figsize=(8, 6))
    
    # Plot all data points
    plt.loglog(times[valid_mask], msd_values[valid_mask], 'o-', 
               alpha=0.7, markersize=4, label='MSD data')
    
    # Plot fit if available
    if not np.isnan(alpha) and len(fit_t) > 0:
        plt.loglog(fit_t, np.exp(intercept) * fit_t**alpha, '--', 
                   color='red', linewidth=2, label=f'Fit: α = {alpha:.3f}')
    
    plt.xlabel('Time (simulation units)')
    plt.ylabel('MSD (position units²)')
    plt.title(f'Mean Squared Displacement (N = {len(valid_ids)} particles)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Add text box with results
    if not np.isnan(alpha):
        textstr = f'Scaling exponent α = {alpha:.3f}\nDiffusion regime: '
        if abs(alpha - 1.0) < 0.1:
            textstr += 'Normal diffusion'
        elif alpha < 0.9:
            textstr += 'Sub-diffusive'
        elif alpha > 1.1:
            textstr += 'Super-diffusive'
        else:
            textstr += 'Near-normal'
        
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
        plt.text(0.05, 0.95, textstr, transform=plt.gca().transAxes, 
                fontsize=10, verticalalignment='top', bbox=props)
    
    plt.tight_layout()
    
    # Save plot
    out_path = os.path.join(OUTPUT_DIR, 'msd.png') if OUTPUT_DIR is not None else 'msd.png'
    plt.savefig(out_path, dpi=300, bbox_inches='tight', pad_inches=0)
    plt.close()
    
    return times, msd_values, alpha, intercept

def metric_beeswarm(df, metric, OUTPUT_DIR=None):
    '''makes a beeswarm plot for a metric for all cases'''
    df = df.loc[df['metric'] == metric]
    sns.swarmplot(data=df, x='case', y='value')
    sns.pointplot(data=df, x='case', y='value', estimator=np.mean,
                  join=False, color='red', markers='_', markersize=15)
    if metric == 'aspect_ratio':
        plt.ylim([1.0,1.5])
    if metric == 'a':
        plt.ylim([0.0, 1.1])
    plt.title(metric)
    out_path = f"{OUTPUT_DIR}/cases_{metric}.png" if OUTPUT_DIR is not None else f"cases_{metric}.png"
    plt.savefig(out_path)
    plt.close()
    plt.show()

def save_first_last_frames(pos_first, pos_last, Xe_first, Xe_last, Xb, kb_vals, 
                          cycle_phases_first, cycle_phases_last, 
                          x_cut, OUTPUT_DIR=None, migrant_cells=None, intercal_cells=None):
    """
    Save the first and last frames of simulation as separate images.
    
    Args:
        pos_first: Cell positions at first frame
        pos_last: Cell positions at last frame 
        Xe_first: Epithelium boundary at first frame
        Xe_last: Epithelium boundary at last frame
        Xb: Bone boundary (static)
        kb_vals: Boundary stiffness values
        cycle_phases_first: Cell cycle phases at first frame
        cycle_phases_last: Cell cycle phases at last frame
        x_cut: Amputation plane position
        OUTPUT_DIR: Output directory for saving images
        migrant_cells: Boolean array indicating migrant cells
        intercal_cells: Boolean array indicating intercalation cells
    """

    def create_frame(pos, Xe, cycle_phases, title, filename, add_metrics=False):
        global KB_MAX, KB_MIN
        """Helper function to create a single frame"""
        # Use same figure parameters as animations.py
        from config import VIDEO_PARAMS
        fig, ax = plt.subplots(figsize=VIDEO_PARAMS['figsize'], dpi=VIDEO_PARAMS['dpi'])
        
        # Clear axis and set bounds - exactly like animations.py
        ax.clear()
        x_bounds = (XMIN, XMAX)
        y_bounds = (YMIN, YMAX)
        
        # Get active cells
        active_mask = ~np.isnan(pos[:, 0])
        phase0 = np.where((cycle_phases == 0) & active_mask)[0]
        phase1 = np.where((cycle_phases == 1) & active_mask)[0]
        
        # Plot boundary with stiffness coloring (exactly like animations.py)
        if kb_vals is not None:
            vmin = np.percentile(kb_vals, 8)
            vmax = np.max(kb_vals)
            
            # Handle case where all values are the same (no softening)
            if vmax - vmin < 1e-6:  # essentially no variation
                # If all values are KB_MAX (no softening), use dark purple
                if abs(vmax - KB_MAX) < 1e-6:
                    color = truncate_colormap(cm.get_cmap('PuRd'), 0.2, 1.0)(1.0)  # Dark purple
                else:
                    color = truncate_colormap(cm.get_cmap('PuRd'), 0.2, 1.0)(0.0)  # Light purple
                
                for i in range(len(Xe)-1):
                    ax.plot([Xe[i,0], Xe[i+1,0]], 
                           [Xe[i,1], Xe[i+1,1]], 
                           '-', lw=2, color=color)
            else:
                # Normal case with variation in stiffness
                norm = mcolors.Normalize(vmin=vmin, vmax=vmax, clip=True)
                cmap = truncate_colormap(cm.get_cmap('PuRd'), 0.2, 1.0)
                for i in range(len(Xe)-1):
                    color = cmap(norm(kb_vals[i]))
                    ax.plot([Xe[i,0], Xe[i+1,0]], 
                           [Xe[i,1], Xe[i+1,1]], 
                           '-', lw=2, color=color)
        
        # Plot bone if enabled (exactly like animations.py)
        from config import BONE_ENABLED
        if BONE_ENABLED and Xb is not None:
            for i in range(len(Xb)-1):
                ax.plot([Xb[i,0], Xb[i+1,0]], 
                       [Xb[i,1], Xb[i+1,1]], 
                       '-', lw=3, color='black', alpha=0.8)
            ax.plot([Xb[-1,0], Xb[0,0]], 
                   [Xb[-1,1], Xb[0,1]], 
                   '-', lw=3, color='black', alpha=0.8)
            ax.plot([x_cut, x_cut], [-1.25, 1.25], '--', lw=1, color='k')
        
        # Calculate cell size EXACTLY like animations.py
        cell_radius = DL_CRIT / 2  # in data units
        
        # Cell size calculation
        bbox = ax.get_window_extent().transformed(ax.figure.dpi_scale_trans.inverted())
        width_inch = bbox.width
        data_width = x_bounds[1] - x_bounds[0]
        points_per_data_unit = (width_inch * 72) / data_width  # 72 points per inch
        r_points = cell_radius * points_per_data_unit
        area_points2 = np.pi * r_points**2
        
        # Plot cells by phase
        ax.scatter(pos[phase0, 0], pos[phase0, 1], s=area_points2, 
                  facecolor=(223/255, 224/255, 95/255), edgecolors='black', 
                  label='G0/G1 phase')
        ax.scatter(pos[phase1, 0], pos[phase1, 1], s=area_points2, 
                  facecolor=(120/255, 237/255, 240/255), edgecolors='black',
                  label='S/G2/M phase')
        
        # Highlight special cell types
        if migrant_cells is not None:
            migration_and_active = active_mask & migrant_cells
            ax.scatter(pos[migration_and_active, 0], pos[migration_and_active, 1], 
                      s=area_points2, facecolor='none', edgecolors='orange', 
                      linewidths=1.5, label='Migration')
        
        if intercal_cells is not None:
            intercal_and_active = active_mask & intercal_cells
            ax.scatter(pos[intercal_and_active, 0], pos[intercal_and_active, 1], 
                      s=area_points2, facecolor='none', edgecolors='red', 
                      linewidths=1.5, label='Intercalation')
        
        # Plot amputation plane
        ax.axvline(x=x_cut, color='gray', linestyle='--', linewidth=1, alpha=0.7)
        

        ax.set_xlim(x_bounds)
        ax.set_ylim(y_bounds)
        ax.grid(True)
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_aspect('equal', 'box')
        ax.set_title(title)
        
        # Add legend for all cell types in top left
        ax.legend(loc='upper left')
        
        # Add metrics text box to last frame
        if add_metrics:
            # Calculate metrics for growth region only (past amputation plane)
            Xe_growth = Xe[Xe[:, 0] > x_cut] if Xe is not None and len(Xe) > 0 else None
            
            if Xe_growth is not None and len(Xe_growth) > 0:
                # Use growth region for area calculation
                area, perimeter, aspect_ratio, ellipticity, roundness, a, b, volume_fraction = morphometrics(Xe, pos=pos, x_cut=x_cut)
            # else:
            #     (Deprecated/unused) Fallback using pos has been disabled per user's request.
            
            # Count cells where x > 1
            active_pos = pos[active_mask]
            cells_x_gt_1 = np.sum(active_pos[:, 0] > 1.0)
            
            # Create metrics text
            metrics_text = (
                f'Area: {area:.2f}\n'
                f'Aspect Ratio: {aspect_ratio:.2f}\n'
                f'Cells (x>1): {cells_x_gt_1}\n'
                f'Max Stiffness: {KB_MAX:.1f}\n'
                f'Min Stiffness: {KB_MIN:.1f}'
            )
            
            # Position text box in top right, smaller size
            ax.text(0.98, 0.98, metrics_text,
                   transform=ax.transAxes,
                   verticalalignment='top',
                   horizontalalignment='right',
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.9, edgecolor='black'),
                   fontsize=8)
        
        # Save figure
        output_path = os.path.join(OUTPUT_DIR, filename) if OUTPUT_DIR else filename
        fig.savefig(output_path, dpi=300, bbox_inches='tight', pad_inches=0.1)
        plt.close(fig)
        print(f"Saved frame: {output_path}")
    
    # Create first frame
    create_frame(pos_first, Xe_first, cycle_phases_first, 
                'First Frame (t=0)', 'first_frame.png', add_metrics=False)
    
    # Create last frame with metrics
    create_frame(pos_last, Xe_last, cycle_phases_last, 
                'Last Frame', 'last_frame.png', add_metrics=True)
