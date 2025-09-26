'''
Limb Regeneration Simulation - Post Processing
Ansa Brews-Smith, May 2025
Copos Lab, Northeastern University

'''

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from config import DL_CRIT, KDIV, KDEATH, K_MIGRATE, DT, TMAX, T_DORMANT, FRAME_SKIP, OUTPUT_DIR, SOFT_RANGE, XMIN, XMAX, YMIN, YMAX
from scipy.spatial import ConvexHull
from skimage.measure import EllipseModel, ransac
import os

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





def morphometrics(Xe, pos=None):
    """
    Calculate morphometrics of the blastema
    """
    if Xe is None:
        pos_valid = pos[~np.isnan(pos).any(axis=1)]
        hull = ConvexHull(pos_valid)
        verts = hull.vertices
        verts = np.append(verts, verts[0])  # Close the hull
        boundary = pos_valid[verts]
        x = boundary[:, 0]
        y = boundary[:, 1]
        center, a, b, theta = ellipse_axes(pos_valid)
        perimeter = hull.area
    else:
        x = Xe[:, 0]
        y = Xe[:, 1]
        perimeter = np.sum(np.sqrt(np.diff(x)**2 + np.diff(y)**2))
        try:
            a = np.max(x) - np.min(x)
            b = (np.max(y) - np.min(y)) / 2
        except:
            a = 0
            b = 2.5

    if len(x) > 2 and len(y) > 2:
        area = .5 * np.abs(np.sum(x * np.roll(y, -1)) - np.sum(y * np.roll(x, -1)))
        
        aspect_ratio = a / b 
        ellipticity = (a - b) / a 
        roundness = (perimeter**2) / (4 * np.pi * area)
        volume_fraction = ((DL_CRIT/2)**2) / (a**2) # cell radius^2 / a^2
    else:
        area = perimeter = aspect_ratio = ellipticity = roundness = volume_fraction = 0

    return area, perimeter, aspect_ratio, ellipticity, roundness, a , b, volume_fraction

def density_heatmap(
    soft_range, pos, Xe, x_cut,
    bin_size=0.1, OUTPUT_DIR='.',
):
    """
    Create a heatmap of cell density distribution with boundary overlay,
    with no extra whitespace around the axes.
    """

    # 1) compute 2D histogram
    x_edges = np.arange(XMIN, XMAX + bin_size, bin_size)
    y_edges = np.arange(YMIN, YMAX + bin_size, bin_size)
    H, xedges, yedges = np.histogram2d(pos[:, 0], pos[:, 1],
                                       bins=[x_edges, y_edges])

    # 2) prepare the mesh for pcolormesh
    X, Y = np.meshgrid(xedges, yedges, indexing='xy')

    # 3) fig+ax with constrained_layout to pack things tightly
    fig, ax = plt.subplots(constrained_layout=True)
    ax.set_aspect('equal', adjustable='box')
    ax.margins(0)  # no padding around data limits

    # 4) plot the heatmap
    pcm = ax.pcolormesh(X, Y, H.T, cmap='plasma', shading='auto')
    cbar = fig.colorbar(pcm, ax=ax, label='Cell count per bin')

    # 5) amputation plane
    if x_cut is not None:
        ax.axvline(x=x_cut, color='black', linestyle='--',
                   label='Amputation plane')

    # 6) overlay boundary segments
    if Xe is not None:
        hard_done = soft_done = False
        for i in range(len(Xe) - 1):
            x0, y0 = Xe[i]
            x1, y1 = Xe[i+1]
            if soft_range[0] < y0 < soft_range[1]:
                lbl = 'Soft boundary' if not soft_done else None
                ax.plot([x0, x1], [y0, y1], '-', lw=2, color='pink', label=lbl)
                soft_done = True
            else:
                lbl = 'Hard boundary' if not hard_done else None
                ax.plot([x0, x1], [y0, y1], '-', lw=2, color='red', label=lbl)
                hard_done = True

    # 7) axes formatting
    ax.set_xlim(XMIN, XMAX)
    ax.set_ylim(YMIN, YMAX)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title('Cell Density Distribution')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper right')

    # 8) stats textbox
    active = np.count_nonzero(~np.isnan(pos[:,0]))
    stats = (
        f'Total cells: {active}\n'
        f'Max density: {H.max():.0f} cells/bin\n'
        f'Mean density: {H[H>0].mean():.1f} cells/bin\n'
        f'Bin size: {bin_size}×{bin_size}'
    )
    ax.text(0.02, 0.98, stats,
            transform=ax.transAxes,
            va='top',
            bbox=dict(boxstyle='round', fc='white', alpha=0.8))

    # 9) save with zero padding
    fig.savefig(f'{OUTPUT_DIR}/cell_density.png',
                dpi=300,
                bbox_inches='tight',
                pad_inches=0)
    plt.close(fig)

    return X, Y, H

def cycle_plot(times, cell_count, N0, OUTPUT_DIR=OUTPUT_DIR):
    """Create a plot of cell count vs time with analytical comparison"""
    def analytical_cell_count(t):
        return N0 * np.exp((KDIV-KDEATH)*(t-T_DORMANT))

    times, cell_count = np.array(times), np.array(cell_count)
    fig, ax = plt.subplots(figsize=(8,5))
    
    ax.plot(times, cell_count, label='Gillespie ABM', lw=2)
    ax.plot(times, analytical_cell_count(times), '--', label='Analytical', lw=2, color='red')

    ax.set_xlabel('Time')
    ax.set_ylabel('Number of Cells')
    ax.set_title('Cell Count vs Time')
    ax.legend()
    ax.set_xlim(0, max(times)) 
    
    # Set y-axis limits based on actual data range with some padding
    max_cells = max(np.max(cell_count), np.max(analytical_cell_count(times)))
    ax.set_ylim(0, max_cells * 1.1)  # 10% padding above max value
    
    ax.grid(True)
    out_path = os.path.join(OUTPUT_DIR, 'cell_count_vs_time.png')
    fig.savefig(out_path, dpi=300)
    plt.close(fig)

def multi_trial_cycle_plot(trial_data_dicts, OUTPUT_DIR=OUTPUT_DIR):
    """Create plots comparing multiple simulation trials against analytical predictions"""
    
    # Extract data from all trials
    all_times = []
    all_cell_counts = []
    all_radii = []
    initial_cell_count = None
    
    for trial_data in trial_data_dicts:
        if 'times' in trial_data and 'cell_count' in trial_data:
            all_times.append(trial_data['times'])
            all_cell_counts.append(trial_data['cell_count'])
            # Use first trial's initial count as N0
            if initial_cell_count is None:
                initial_cell_count = trial_data['cell_count'][0]
            
            # Extract radius data (semi-major axis 'a' from metrics_time_series)
            if 'metrics_time_series' in trial_data and 'a' in trial_data['metrics_time_series']:
                all_radii.append(np.array(trial_data['metrics_time_series']['a']))
            else:
                all_radii.append([])
    
    if not all_times:
        print("Warning: No valid trial data found for plotting")
        return
    
    # Use initial cell count from first trial
    N0 = initial_cell_count
    print(f"DEBUG: Using N0 = {N0} for analytical predictions")
    
    # Since all simulations have the same time length, use the first trial's time grid
    common_times = all_times[0]
    min_time = common_times[0]
    max_time = common_times[-1]
    
    # Convert to numpy arrays (assuming all trials have same length)
    all_cell_counts = np.array(all_cell_counts)
    mean_counts = np.mean(all_cell_counts, axis=0)
    std_counts = np.std(all_cell_counts, axis=0)
    min_counts = np.min(all_cell_counts, axis=0)
    max_counts = np.max(all_cell_counts, axis=0)
    
    # Analytical predictions
    def analytical_cell_count(t):
        return N0 * np.exp((KDIV-KDEATH)*(t-T_DORMANT))
    
    # Analytical radius prediction (consistent with cell count)
    A_cell = np.pi * (DL_CRIT/2)**2
    phi = np.pi / (2*np.sqrt(3))  # hexagon packing factor ≈ 0.906
    A_t = lambda t: phi * A_cell * N0 * np.exp((KDIV-KDEATH)*(t-T_DORMANT))
    r_t = lambda t: np.sqrt(A_t(t)/np.pi)
    
    # Process radius data if available
    radius_data_available = any(len(r) > 0 for r in all_radii)
    if radius_data_available:
        # Filter out empty radius arrays and convert to numpy array
        valid_radii = [r for r in all_radii if len(r) > 0]
        if valid_radii:
            all_radii = np.array(valid_radii)
            mean_radii = np.mean(all_radii, axis=0)
            std_radii = np.std(all_radii, axis=0)
        else:
            radius_data_available = False
    
    # Create the plot - multiple subplots on same figure
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # Colors for different trials
    colors = plt.cm.tab10(np.linspace(0, 1, len(all_cell_counts)))
    
    # Plot 1: Cell Count
    ax1 = axes[0]
    for i, counts in enumerate(all_cell_counts):
        ax1.plot(common_times, counts, color=colors[i], alpha=0.3, linewidth=0.8)
    
    # Cell count mean with confidence bands
    ax1.plot(common_times, mean_counts, 'k-', linewidth=2, label='Simulation Mean')
    ax1.fill_between(common_times, mean_counts - std_counts, mean_counts + std_counts, 
                     alpha=0.3, color='gray', label='±1σ')
    
    # Cell count analytical curve
    ax1.plot(common_times, analytical_cell_count(common_times), 'r--', linewidth=2, 
             label='Analytical')
    
    ax1.set_xlabel('Time')
    ax1.set_ylabel('Number of Cells')
    ax1.set_title(f'Cell Count Evolution: {len(all_cell_counts)} Trials')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(min_time, max_time)
    
    # Plot 2: Radius (if available)
    if radius_data_available:
        ax2 = axes[1]
        # Plot individual radius trials (faded)
        for i, radii in enumerate(all_radii):
            ax2.plot(common_times, radii, color=colors[i], alpha=0.3, linewidth=0.8)
        
        # Radius mean with confidence bands
        ax2.plot(common_times, mean_radii, 'k-', linewidth=2, label='Simulation Mean')
        ax2.fill_between(common_times, mean_radii - std_radii, mean_radii + std_radii, 
                         alpha=0.3, color='gray', label='±1σ')
        
        # Analytical radius curve
        ax2.plot(common_times, r_t(common_times), 'r--', linewidth=2, label='Analytical')
        
        ax2.set_xlabel('Time')
        ax2.set_ylabel('Radius')
        ax2.set_title(f'Radius Evolution: {len(all_radii)} Trials')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        ax2.set_xlim(min_time, max_time)
        
        # Plot 3: Area
        ax3 = axes[2]
        # Analytical area (consistent with cell count)
        A_t_plot = lambda t: phi * A_cell * N0 * np.exp((KDIV-KDEATH)*(t-T_DORMANT))
        analytical_area = A_t_plot(common_times)
        ax3.plot(common_times, analytical_area, 'r--', linewidth=2, label='Analytical')
        
        # Simulation area (from radius)
        sim_area = np.pi * mean_radii**2
        ax3.plot(common_times, sim_area, 'k-', linewidth=2, label='Simulation Mean')
        
        # Individual trial areas
        for i, radii in enumerate(all_radii):
            trial_area = np.pi * radii**2
            ax3.plot(common_times, trial_area, color=colors[i], alpha=0.3, linewidth=0.8)
        
        ax3.set_xlabel('Time')
        ax3.set_ylabel('Area')
        ax3.set_title(f'Area Evolution: {len(all_radii)} Trials')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        ax3.set_xlim(min_time, max_time)
    else:
        # Hide radius and area plots if no radius data
        axes[1].axis('off')
        axes[2].axis('off')
    
    plt.tight_layout()
    
    # Save the plot
    out_path = os.path.join(OUTPUT_DIR, 'multi_trial_comparison.png')
    fig.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    
    # Calculate analytical prediction and errors for summary
    analytical_pred = analytical_cell_count(common_times)
    errors = all_cell_counts - analytical_pred[np.newaxis, :]
    mean_error = np.mean(errors, axis=0)
    std_error = np.std(errors, axis=0)
    
    # Print summary statistics
    print(f"\n=== Multi-Trial Analysis Summary ===")
    print(f"Number of trials: {len(all_cell_counts)}")
    print(f"Final mean cell count: {mean_counts[-1]:.2f} ± {std_counts[-1]:.2f}")
    print(f"Final analytical prediction: {analytical_pred[-1]:.2f}")
    print(f"Final cell count error: {mean_error[-1]:.2f} ± {std_error[-1]:.2f}")
    print(f"Mean absolute cell count error: {np.mean(np.abs(mean_error)):.2f}")
    
    if radius_data_available:
        analytical_radius_final = r_t(common_times[-1])
        radius_error = mean_radii[-1] - analytical_radius_final
        print(f"Final mean radius: {mean_radii[-1]:.3f} ± {std_radii[-1]:.3f}")
        print(f"Final analytical radius: {analytical_radius_final:.3f}")
        print(f"Final radius error: {radius_error:.3f}")
        print(f"Mean absolute radius error: {np.mean(np.abs(mean_radii - r_t(common_times))):.3f}")
    
    return {
        'times': common_times,
        'mean_counts': mean_counts,
        'std_counts': std_counts,
        'analytical_pred': analytical_pred,
        'mean_error': mean_error,
        'std_error': std_error
    }

def multi_trial_metrics_plot(trial_data_dicts, OUTPUT_DIR=OUTPUT_DIR):
    """Create plots for morphometric metrics across multiple trials"""
    
    # Extract metrics from all trials
    metrics_to_plot = ['area', 'perimeter', 'aspect_ratio', 'ellipticity', 'roundness', 'a', 'b']
    all_metrics = {metric: [] for metric in metrics_to_plot}
    all_times = []
    
    for trial_data in trial_data_dicts:
        if 'metrics_time_series' in trial_data:
            metrics_ts = trial_data['metrics_time_series']
            if 'time' in metrics_ts:
                all_times.append(np.array(metrics_ts['time']))
                for metric in metrics_to_plot:
                    if metric in metrics_ts:
                        all_metrics[metric].append(np.array(metrics_ts[metric]))
    
    if not all_times:
        print("Warning: No metrics time series data found")
        return
    
    # Find common time range
    min_time = max([t[0] for t in all_times])
    max_time = min([t[-1] for t in all_times])
    n_points = min([len(t) for t in all_times])
    common_times = np.linspace(min_time, max_time, n_points)
    
    # Create subplots for key metrics
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    axes = axes.flatten()
    
    key_metrics = ['area', 'aspect_ratio', 'roundness', 'a']
    
    for i, metric in enumerate(key_metrics):
        if i >= len(axes):
            break
            
        ax = axes[i]
        
        if metric in all_metrics and all_metrics[metric]:
            # Interpolate all trials to common time grid
            interpolated_values = []
            for j, values in enumerate(all_metrics[metric]):
                if len(all_times[j]) == len(values):
                    interp_values = np.interp(common_times, all_times[j], values)
                    interpolated_values.append(interp_values)
            
            if interpolated_values:
                interpolated_values = np.array(interpolated_values)
                mean_values = np.mean(interpolated_values, axis=0)
                std_values = np.std(interpolated_values, axis=0)
                
                # Plot individual trials (faded)
                for values in interpolated_values:
                    ax.plot(common_times, values, alpha=0.3, linewidth=0.8)
                
                # Plot mean with error bars
                ax.plot(common_times, mean_values, 'k-', linewidth=2, label='Mean')
                ax.fill_between(common_times, mean_values - std_values, mean_values + std_values, 
                               alpha=0.3, color='gray', label='±1σ')
                
                ax.set_xlabel('Time')
                ax.set_ylabel(metric.replace('_', ' ').title())
                ax.set_title(f'{metric.replace("_", " ").title()} Evolution')
                ax.grid(True, alpha=0.3)
                ax.legend()
        else:
            ax.text(0.5, 0.5, f'No data for {metric}', transform=ax.transAxes, 
                   ha='center', va='center')
            ax.set_title(f'{metric.replace("_", " ").title()} Evolution')
    
    plt.tight_layout()
    
    # Save the plot
    out_path = os.path.join(OUTPUT_DIR, 'multi_trial_metrics.png')
    fig.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    
    print(f"\n=== Multi-Trial Metrics Summary ===")
    for metric in key_metrics:
        if metric in all_metrics and all_metrics[metric]:
            interpolated_values = []
            for j, values in enumerate(all_metrics[metric]):
                if len(all_times[j]) == len(values):
                    interp_values = np.interp(common_times, all_times[j], values)
                    interpolated_values.append(interp_values)
            
            if interpolated_values:
                final_values = [vals[-1] for vals in interpolated_values]
                print(f"{metric}: {np.mean(final_values):.3f} ± {np.std(final_values):.3f}")

# def boundary_plot(Xe0, Xe_final, Xe_growth, x_cut, aspect_ratio, area, roundness, a, perimeter, ellipticity, OUTPUT_DIR=OUTPUT_DIR, pos0=None, pos_final=None):
#     # create fig+axes and lock aspect ratio before layouts
#     fig, ax = plt.subplots()
#     ax.set_aspect('equal', adjustable='box')

#     # compute the scatter‐marker size in points^2
#     cell_radius = DL_CRIT / 2  # in data units

#     # Convert radius from data units to points
#     # Get the scaling factor from data coordinates to display coordinates
#     ax = plt.gca()
#     bbox = ax.get_window_extent().transformed(ax.figure.dpi_scale_trans.inverted())
#     width_inch = bbox.width
#     data_width = XMAX - XMIN
#     points_per_data_unit = (width_inch * 72) / data_width  # 72 points per inch

#     # Calculate radius in points, then area in points^2
#     r_points = cell_radius * points_per_data_unit / 2
#     area_points2 = np.pi * r_points**2

#     # set up axes limits and labels
#     ax.set_xlim(XMIN, XMAX)
#     ax.set_ylim(YMIN, YMAX)
#     ax.set_xlabel('x')
#     ax.set_ylabel('y')

#     # plot according to whether we have a growth‐region trace or raw cell positions
#     if Xe_growth is not None:
#         ax.plot(Xe0[:, 0], Xe0[:, 1], 'k-', label='Initial boundary',
#                 linewidth=2, alpha=0.5)
#         ax.plot(Xe_final[:, 0], Xe_final[:, 1], 'b-', label='Final boundary',
#                 linewidth=2)
#         ax.plot(Xe_growth[:, 0], Xe_growth[:, 1], 'r-', label='Growth region',
#                 linewidth=2)
#         ax.axvline(x=x_cut, color='gray', linestyle='--',
#                    label='Amputation plane')
#         ax.set_title('Boundary Evolution and Growth Region')
#     else:
#         # initial cluster
#         if pos0 is not None:
#             pos0_valid = pos0[~np.isnan(pos0).any(axis=1)]
#             if len(pos0_valid) >= 3:
#                 hull0 = ConvexHull(pos0_valid)
#                 verts = np.append(hull0.vertices, hull0.vertices[0])
#                 ax.plot(pos0_valid[verts, 0], pos0_valid[verts, 1],
#                         c='blue', label='Initial boundary',
#                         linewidth=2, alpha=0.5)
#             ax.scatter(pos0_valid[:, 0], pos0_valid[:, 1],
#                        s=area_points2, color='blue', alpha=0.3,
#                        label='Initial cells')

#         # final cluster
#         if pos_final is not None:
#             posf_valid = pos_final[~np.isnan(pos_final).any(axis=1)]
#             if len(posf_valid) >= 3:
#                 hullf = ConvexHull(posf_valid)
#                 verts = np.append(hullf.vertices, hullf.vertices[0])
#                 ax.plot(posf_valid[verts, 0], posf_valid[verts, 1],
#                         c='red', label='Final boundary',
#                         linewidth=2, alpha=0.5)
#             ax.scatter(posf_valid[:, 0], posf_valid[:, 1],
#                        s=area_points2, color='red', alpha=0.3,
#                        label='Final cells')

#         ax.set_title('Initial and Final Cell Clusters')

#     # legend, grid
#     ax.legend(loc='upper right', fontsize=8)
#     ax.grid(True, alpha=0.3)

#     # overlay metrics text
#     metrics_text = (
#         f'Growth Metrics:\n'
#         f'Area: {area:.2f}\n'
#         f'Aspect Ratio: {aspect_ratio:.2f}\n'
#         f'Roundness: {roundness:.2f}\n'
#         f'Radius/Outgrowth Length: {a:.2f}\n'
#         f'Perimeter: {perimeter:.2f}\n'
#         f'Ellipticity: {ellipticity:.2f}'
#     )
#     ax.text(
#         0.02, 0.98, metrics_text,
#         transform=ax.transAxes,
#         verticalalignment='top',
#         bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
#         fontsize=7
#     )

#     fig.tight_layout()
#     fig.savefig(f'{OUTPUT_DIR}/growth.png')
#     plt.close(fig)


import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull
from skimage.measure import EllipseModel, ransac

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

def boundary_plot(Xe0, Xe_final, Xe_growth, x_cut, aspect_ratio, area,
                  roundness, a, perimeter, ellipticity,
                  OUTPUT_DIR, pos0=None, pos_final=None):
    """
    Plots either the growth-region trace (if Xe_growth present) or
    the initial/final clusters with their minimum enclosing ellipses.
    """
    fig, ax = plt.subplots()
    ax.set_aspect('equal', adjustable='box')
    
    # compute scatter marker size
    cell_radius = DL_CRIT / 2
    bbox = ax.get_window_extent().transformed(ax.figure.dpi_scale_trans.inverted())
    ppf = (bbox.width * 72) / (XMAX - XMIN)
    area_pts2 = np.pi * (cell_radius * ppf)**2  # Fixed: removed extra /2

    ax.set_xlim(XMIN, XMAX)
    ax.set_ylim(YMIN, YMAX)
    ax.set_xlabel('x')
    ax.set_ylabel('y')

    if Xe_growth is not None:
        ax.plot(Xe0[:,0], Xe0[:,1], 'k-', lw=2, alpha=0.5, label='Initial boundary')
        ax.plot(Xe_final[:,0], Xe_final[:,1], 'b-', lw=2, label='Final boundary')
        ax.plot(Xe_growth[:,0], Xe_growth[:,1],'r-', lw=2, label='Growth region')
        ax.axvline(x=x_cut, color='gray', ls='--', label='Amputation plane')
        ax.set_title('Boundary Evolution and Growth Region')
    else:
        def scatter_and_ellipse(pts, color, cell_label, ellipse_label):
            ax.scatter(pts[:,0], pts[:,1], s=area_pts2, c=color, alpha=0.3, label=cell_label)
            if len(pts) >= 5:
                c, a_e, b_e, th = ellipse_axes(pts)
                t = np.linspace(0, 2*np.pi, 200)
                R = np.array([[np.cos(th), -np.sin(th)],
                              [np.sin(th),  np.cos(th)]])
                ellipse_pts = (np.vstack([a_e*np.cos(t), b_e*np.sin(t)]).T @ R.T) + c
                ax.plot(ellipse_pts[:,0], ellipse_pts[:,1], c=color, lw=2, alpha=0.6, label=ellipse_label)

        if pos0 is not None:
            pts0 = pos0[~np.isnan(pos0).any(axis=1)]
            scatter_and_ellipse(pts0, 'blue', 'Initial cells', 'Initial ellipse')

        if pos_final is not None:
            ptsf = pos_final[~np.isnan(pos_final).any(axis=1)]
            scatter_and_ellipse(ptsf, 'red', 'Final cells', 'Final ellipse')

        ax.set_title('Initial and Final Cell Clusters (Min-Enclosing Ellipse)')

    ax.legend(loc='upper right', fontsize=8)
    ax.grid(True, alpha=0.3)

    metrics_text = (
        f'Growth Metrics:\n'
        f'Area: {area:.2f}\n'
        f'Aspect Ratio: {aspect_ratio:.2f}\n'
        f'Roundness: {roundness:.2f}\n'
        f'Radius/Outgrowth Length: {a:.2f}\n'
        f'Perimeter: {perimeter:.2f}\n'
        f'Ellipticity: {ellipticity:.2f}'
    )
    ax.text(0.02, 0.98, metrics_text, transform=ax.transAxes, va='top',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8), fontsize=7)

    fig.tight_layout()
    fig.savefig(f'{OUTPUT_DIR}/growth.png')
    plt.close(fig)



def runtime_plot(cell_count, steptimes, OUTPUT_DIR=OUTPUT_DIR):
    """Create a plot of runtime vs time"""
    fig, ax = plt.subplots(figsize=(8,5))
    skip = int(T_DORMANT / (DT*FRAME_SKIP)) # skip dormancy period
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

    
    out_path = os.path.join(OUTPUT_DIR, 'runtime_vs_cell_count.png')
    fig.savefig(out_path, dpi=300)
    plt.close(fig)

def trajectory_plot(
    positions, Xe, x_cut, death_indicies,
    ids=range(10), OUTPUT_DIR=OUTPUT_DIR,
    boundary=True
):
    """Plot trajectories of selected cells, circle at start, arrow at end if alive, × if died."""
    plt.figure(figsize=(8, 6))
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
            plt.plot(
                x_coords, y_coords, '--',
                markersize=2, alpha=0.5,
                color=colors[idx],
                label=f'Cell {cell_id}'
            )
            # start circle
            plt.scatter(
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
                plt.plot(dx, dy, 'x', markersize=6, color=colors[idx], alpha=0.7)
        else:
            # alive: arrow from penultimate to last
            if len(x_coords) >= 2:
                plt.annotate(
                    '',
                    xy=(x_coords[-1], y_coords[-1]),
                    xytext=(x_coords[-2], y_coords[-2]),
                    arrowprops=dict(arrowstyle='->', color=colors[idx], lw=1.5, alpha=0.7)
                )

    # boundary overlay
    if boundary:
        plt.axvline(x=x_cut, color='black', linestyle='--', label='Amputation plane')
        hard_done = soft_done = False
        for i in range(len(Xe) - 1):
            x0, y0 = Xe[i]
            x1, y1 = Xe[i+1]
            if SOFT_RANGE[0] < y0 < SOFT_RANGE[1]:
                lbl = 'Soft boundary' if not soft_done else None
                plt.plot([x0, x1], [y0, y1], '-', lw=2, color='pink', label=lbl)
                soft_done = True
            else:
                lbl = 'Hard boundary' if not hard_done else None
                plt.plot([x0, x1], [y0, y1], '-', lw=2, color='red', label=lbl)
                hard_done = True

    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Cell Trajectories Over Time')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    # plt.axis('equal')

    plt.xlim(XMIN, XMAX)
    plt.ylim(YMIN, YMAX)
    plt.grid(True)

    out_path = os.path.join(OUTPUT_DIR, 'cell_trajectories.png')
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()

def metric_beeswarm(df, metric, OUTPUT_DIR=OUTPUT_DIR):
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
    plt.savefig(f"{OUTPUT_DIR}/cases_{metric}.png")
    plt.close()
    plt.show()

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












#### OPTIMIZATION PLOTS ####

def plot_profile_likelihood(profile_results, OUTPUT_DIR=OUTPUT_DIR):
    """Plot profile likelihood results for all parameters"""
    import pickle
    
    # Load results if path is provided
    if isinstance(profile_results, str):
        with open(profile_results, 'rb') as f:
            profile_results = pickle.load(f)
    
    n_params = len(profile_results)
    param_names = list(profile_results.keys())
    
    # Create subplots
    if n_params == 1:
        fig, ax = plt.subplots(1, 1, figsize=(8, 6))
        axes = [ax]
    elif n_params == 2:
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    else:
        # For 3+ parameters, use a grid layout
        n_cols = min(3, n_params)
        n_rows = (n_params + n_cols - 1) // n_cols
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(4*n_cols, 4*n_rows))
        if n_rows == 1:
            axes = axes.reshape(1, -1)
        axes = axes.flatten()
    
    for i, param_name in enumerate(param_names):
        data = profile_results[param_name]
        param_grid = data['param_grid']
        errors = data['errors']
        param_star = data['param_star']
        
        # Find minimum error for relative likelihood calculation
        min_error = np.nanmin(errors)
        sigma = 0.1 * np.mean(errors)
        # Calculate relative likelihood
        relative_likelihood = 1/np.sqrt(2*np.pi*sigma**2)*np.exp(-(0.5/sigma**2)*(errors))
        
        # Plot profile likelihood
        ax = axes[i] if n_params > 1 else axes[0]
        
        # Filter out infinite values for plotting
        valid_mask = np.isfinite(relative_likelihood)
        if np.any(valid_mask):
            ax.plot(param_grid[valid_mask], relative_likelihood[valid_mask], 'b-', linewidth=2)
            ax.scatter(param_grid[valid_mask], relative_likelihood[valid_mask], c='blue', s=30, alpha=0.7)
        
        # Mark the actual best parameter value from profile likelihood
        # Compute here to avoid using it before assignment
        min_error = np.nanmin(errors)
        min_error_idx = np.nanargmin(errors)
        actual_best_param = param_grid[min_error_idx]
        ax.axvline(x=actual_best_param, color='red', linestyle='--', linewidth=2, alpha=0.8, 
                   label=f'Profile Best: {actual_best_param:.3f}')
        
        # Also mark the original "best" value if it's different
        if abs(actual_best_param - param_star) > 1e-3:
            ax.axvline(x=param_star, color='orange', linestyle=':', linewidth=2, alpha=0.6, 
                       label=f'Optimization Best: {param_star:.3f}')
        
        # # Add confidence intervals (68%, 95%, 99.7% confidence levels)
        # confidence_levels = [0.68, 0.95, 0.997]
        # confidence_colors = ['green', 'orange', 'red']
        # confidence_labels = ['68% CI', '95% CI', '99.7% CI']
        
        # for level, color, label in zip(confidence_levels, confidence_colors, confidence_labels):
        #     threshold = np.exp(-0.5 * (1 - level))
        #     ax.axhline(y=threshold, color=color, linestyle=':', alpha=0.7, label=label)
        
        ax.set_xlabel(param_name)
        ax.set_ylabel('Relative Likelihood')
        ax.set_title(f'Profile Likelihood: {param_name}')
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)
        
        # Set y-axis limits to focus on relevant range
        if np.any(valid_mask):
            y_max = np.nanmax(relative_likelihood[valid_mask])
            ax.set_ylim(0, y_max * 1.1)
    
    # Turn off any unused subplots
    if n_params > 1:
        for i in range(n_params, len(axes)):
            axes[i].axis('off')
    
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/profile_likelihood.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Also create a summary table
    print("\n=== Profile Likelihood Summary ===")
    for param_name in param_names:
        data = profile_results[param_name]
        param_star = data['param_star']
        errors = data['errors']
        min_error = np.nanmin(errors)
        min_error_idx = np.nanargmin(errors)
        actual_best_param = data['param_grid'][min_error_idx]
        
        print(f"{param_name}:")
        print(f"  Optimization Best = {param_star:.4f}")
        print(f"  Profile Likelihood Best = {actual_best_param:.4f}")
        print(f"  Min Error = {min_error:.6f}")
        if abs(actual_best_param - param_star) > 1e-3:
            print(f"  MISMATCH: Profile best differs from optimization best!")

def print_optimization_step(iteration, params, error, best_params=None, best_error=None, param_names=None):
    """Print current optimization step information"""
    print(f"\n--- Optimization Step {iteration} ---")
    
    if param_names is not None and len(param_names) == len(params):
        param_str = ', '.join([f'{name}={val:.4f}' for name, val in zip(param_names, params)])
    else:
        # Fallback for backward compatibility
        param_str = ', '.join([f'param_{i}={val:.4f}' for i, val in enumerate(params)])
    
    print(f"Parameters: {param_str}")
    print(f"Objective Error: {error:.6f}")
    
    if best_params is not None and best_error is not None:
        if param_names is not None and len(param_names) == len(best_params):
            best_param_str = ', '.join([f'{name}={val:.4f}' for name, val in zip(param_names, best_params)])
        else:
            best_param_str = ', '.join([f'param_{i}={val:.4f}' for i, val in enumerate(best_params)])
        print(f"Best so far: {best_param_str}, Error={best_error:.6f}")

def plot_optimization_history(param_history, error_history, OUTPUT_DIR=OUTPUT_DIR, param_names=None):
    """Plot optimization convergence history for any number of parameters"""
    param_history = np.array(param_history)
    error_history = np.array(error_history)
    n_params = param_history.shape[1]
    # Compute running-best error to visualize true convergence of differential evolution
    running_best = np.minimum.accumulate(error_history)
    
    if param_names is None:
        param_names = [f'Param_{i}' for i in range(n_params)]
    
    # For 1 parameter: 2x2 layout with error + parameter evolution
    if n_params == 1:
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        
        # Error history
        axes[0, 0].plot(error_history, 'b-o', markersize=4)
        axes[0, 0].set_xlabel('Iteration')
        axes[0, 0].set_ylabel('Objective Error')
        axes[0, 0].set_title('Optimization Convergence')
        axes[0, 0].grid(True)
        
        # Parameter evolution
        axes[0, 1].plot(param_history[:, 0], 'r-o', markersize=4)
        axes[0, 1].set_xlabel('Iteration')
        axes[0, 1].set_ylabel(param_names[0])
        axes[0, 1].set_title(f'{param_names[0]} Evolution')
        axes[0, 1].grid(True)
        
        # Parameter vs error
        axes[1, 0].scatter(param_history[:, 0], error_history, c=range(len(error_history)), 
                          cmap='viridis', s=50)
        axes[1, 0].set_xlabel(param_names[0])
        axes[1, 0].set_ylabel('Error')
        axes[1, 0].set_title(f'{param_names[0]} vs Error')
        axes[1, 0].grid(True)
        
        # Summary statistics
        axes[1, 1].axis('off')
        best_idx = np.argmin(error_history)
        stats_text = f"""
        Optimization Summary
        ===================
        Total Iterations: {len(error_history)}
        Best {param_names[0]}: {param_history[best_idx, 0]:.4f}
        Best Error: {error_history[best_idx]:.6f}
        
        Parameter Range:
        {param_names[0]}: [{param_history[:, 0].min():.3f}, {param_history[:, 0].max():.3f}]
        
        Error Reduction:
        {((error_history[0] - error_history[best_idx]) / error_history[0] * 100):.1f}%
        """
        axes[1, 1].text(0.1, 0.9, stats_text, transform=axes[1, 1].transAxes, 
                        verticalalignment='top', fontfamily='monospace', fontsize=10)
    
    # For 2 parameters: original 2x2 layout
    elif n_params == 2:
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        
        # Error history
        axes[0, 0].plot(error_history, 'b-o', markersize=4)
        axes[0, 0].set_xlabel('Iteration')
        axes[0, 0].set_ylabel('Objective Error')
        axes[0, 0].set_title('Optimization Convergence')
        axes[0, 0].grid(True)
        
        # Parameter histories
        colors = ['r', 'g']
        for i in range(2):
            row, col = (0, 1) if i == 0 else (1, 0)
            axes[row, col].plot(param_history[:, i], f'{colors[i]}-o', markersize=4)
            axes[row, col].set_xlabel('Iteration')
            axes[row, col].set_ylabel(param_names[i])
            axes[row, col].set_title(f'{param_names[i]} Evolution')
            axes[row, col].grid(True)
        
        # Parameter space exploration
        axes[1, 1].scatter(param_history[:, 0], param_history[:, 1], 
                          c=error_history, cmap='viridis', s=50)
        axes[1, 1].set_xlabel(param_names[0])
        axes[1, 1].set_ylabel(param_names[1])
        axes[1, 1].set_title('Parameter Space Exploration')
        cbar = plt.colorbar(axes[1, 1].collections[0], ax=axes[1, 1])
        cbar.set_label('Objective Error')
    
    # For 3+ parameters: adaptive layout
    else:
        n_rows = min(3, (n_params + 2) // 2)  # Up to 3 rows
        n_cols = min(3, n_params + 1)  # Up to 3 columns
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(4*n_cols, 4*n_rows))
        if n_rows == 1:
            axes = axes.reshape(1, -1)
        elif n_cols == 1:
            axes = axes.reshape(-1, 1)
        axes = axes.flatten()
        
        # Error history (always first)
        axes[0].plot(error_history, 'b-o', markersize=4)
        axes[0].set_xlabel('Iteration')
        axes[0].set_ylabel('Objective Error')
        axes[0].set_title('Optimization Convergence')
        axes[0].grid(True)
        
        # Parameter evolution plots
        colors = ['r', 'g', 'orange', 'purple', 'brown']
        for i in range(min(n_params, len(axes)-1)):
            ax_idx = i + 1
            color = colors[i % len(colors)]
            axes[ax_idx].plot(param_history[:, i], f'{color}-o', markersize=4)
            axes[ax_idx].set_xlabel('Iteration')
            axes[ax_idx].set_ylabel(param_names[i])
            axes[ax_idx].set_title(f'{param_names[i]} Evolution')
            axes[ax_idx].grid(True)
        
        # Turn off any unused axes
        for i in range(n_params + 1, len(axes)):
            axes[i].axis('off')
    
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/optimization_history.png', dpi=300)
    plt.close()

def plot_shape_comparison(target_shape, sim_boundary, target_error, OUTPUT_DIR=OUTPUT_DIR):
    """Plot comparison between target and simulated shapes"""
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_aspect('equal')
    
    # Plot target shape
    ax.plot(target_shape[:, 0], target_shape[:, 1], 'b-', linewidth=3, 
            label=f'Target Shape (r={np.mean(np.linalg.norm(target_shape, axis=1)):.1f})')
    
    # Plot simulated boundary
    ax.plot(sim_boundary[:, 0], sim_boundary[:, 1], 'r--', linewidth=2,
            label='Simulated Boundary')
    
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title(f'Shape Comparison (Error: {target_error:.6f})')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Set equal limits
    all_x = np.concatenate([target_shape[:, 0], sim_boundary[:, 0]])
    all_y = np.concatenate([target_shape[:, 1], sim_boundary[:, 1]])
    margin = 0.1
    ax.set_xlim(all_x.min() - margin, all_x.max() + margin)
    ax.set_ylim(all_y.min() - margin, all_y.max() + margin)
    
    plt.savefig(f'{OUTPUT_DIR}/shape_comparison.png', dpi=300)
    plt.close()



def plot_sdf_evolution(target_shape, boundary_history, param_history, error_history, OUTPUT_DIR=OUTPUT_DIR, param_names=None):
    """Plot how signed distance function evolves with different parameter combinations"""
    from signed_distance import make_sdf_polygon
    
    param_history = np.array(param_history)
    n_params = param_history.shape[1]
    
    if param_names is None:
        param_names = [f'Param_{i}' for i in range(n_params)]
    
    # Select key iterations to show (start, middle, best, end)
    n_iters = len(boundary_history)
    if n_iters < 4:
        selected_iters = list(range(n_iters))
    else:
        best_iter = np.argmin(error_history)
        selected_iters = [0, n_iters//2, best_iter, n_iters-1]
        selected_iters = sorted(list(set(selected_iters)))  # Remove duplicates
    
    n_plots = len(selected_iters)
    fig, axes = plt.subplots(2, n_plots, figsize=(4*n_plots, 8))
    if n_plots == 1:
        axes = axes.reshape(2, 1)
    
    # Create grid for SDF evaluation
    x_range = np.linspace(-1.0, 1.0, 100)
    y_range = np.linspace(-1.0, 1.0, 100)
    X, Y = np.meshgrid(x_range, y_range)
    
    # Target SDF
    target_sdf = make_sdf_polygon(target_shape)
    target_values = target_sdf(X, Y)
    
    # First pass: collect SDF differences for ALL iterations to determine global scale
    all_global_diffs = []
    
    # Calculate SDF differences for ALL iterations (not just selected ones)
    for iter_num in range(len(boundary_history)):
        boundary = boundary_history[iter_num]
        sim_sdf = make_sdf_polygon(boundary)
        sim_values = sim_sdf(X, Y)
        sdf_diff = sim_values - target_values
        all_global_diffs.append(sdf_diff)
    
    # Calculate global min/max across ALL iterations for consistent color scaling
    all_global_diffs = np.array(all_global_diffs)
    global_min = all_global_diffs.min()
    global_max = all_global_diffs.max()
    
    # Use maximum magnitude for symmetric colorbar centered at zero (white = 0 error)
    max_magnitude = max(abs(global_min), abs(global_max))
    vmin, vmax = -max_magnitude, max_magnitude
    
    # Second pass: collect SDF differences for selected iterations (for plotting)
    all_sdf_diffs = []
    all_sim_values = []
    
    for idx, iter_num in enumerate(selected_iters):
        boundary = boundary_history[iter_num]
        sim_sdf = make_sdf_polygon(boundary)
        sim_values = sim_sdf(X, Y)
        all_sim_values.append(sim_values)
        sdf_diff = sim_values - target_values
        all_sdf_diffs.append(sdf_diff)
    
    # Third pass: plot with consistent scaling
    for idx, iter_num in enumerate(selected_iters):
        boundary = boundary_history[iter_num]
        params = param_history[iter_num]
        error = error_history[iter_num]
        sim_values = all_sim_values[idx]
        sdf_diff = all_sdf_diffs[idx]
        
        # Build parameter string for title
        param_str = ', '.join([f'{name}={params[i]:.3f}' for i, name in enumerate(param_names)])
        
        # Plot boundaries
        ax_bound = axes[0, idx]
        ax_bound.plot(target_shape[:, 0], target_shape[:, 1], 'b-', linewidth=2, label='Target')
        ax_bound.plot(boundary[:, 0], boundary[:, 1], 'r--', linewidth=2, label='Simulated')
        ax_bound.set_aspect('equal')
        ax_bound.set_xlim(-1.0, 1.0)
        ax_bound.set_ylim(-1.0, 1.0)
        ax_bound.set_title(f'Iter {iter_num+1}\n{param_str}\nError={error:.4f}')
        ax_bound.legend()
        ax_bound.grid(True, alpha=0.3)
        
        # Plot SDF difference with standardized color scale
        ax_sdf = axes[1, idx]
        im = ax_sdf.contourf(X, Y, sdf_diff, levels=20, cmap='RdBu_r', alpha=0.8,
                            vmin=vmin, vmax=vmax)
        ax_sdf.contour(X, Y, target_values, levels=[0], colors='blue', linewidths=2, alpha=0.7)
        ax_sdf.contour(X, Y, sim_values, levels=[0], colors='red', linewidths=2, alpha=0.7, linestyles='--')
        ax_sdf.set_aspect('equal')
        ax_sdf.set_xlim(-1.0, 1.0)
        ax_sdf.set_ylim(-1.0, 1.0)
        ax_sdf.set_title(f'SDF Difference\n(Red > Target, Blue < Target)')
        
        # Add colorbar only to the rightmost plot
        if idx == len(selected_iters) - 1:
            cbar = plt.colorbar(im, ax=ax_sdf, label='Sim - Target')
            cbar.set_label(f'Sim - Target\n±{max_magnitude:.3f}\n(White = 0, All Iterations)', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/sdf_evolution.png', dpi=300, bbox_inches='tight')
    plt.close()

def create_optimization_summary(param_history, error_history, boundary_history, target_shape, OUTPUT_DIR=OUTPUT_DIR, param_names=None):
    """Create a comprehensive summary of the optimization process"""
    param_history = np.array(param_history)
    error_history = np.array(error_history)
    n_params = param_history.shape[1]
    
    if param_names is None:
        param_names = [f'Param_{i}' for i in range(n_params)]
    
    # Find best parameters
    best_idx = np.argmin(error_history)
    best_params = param_history[best_idx]
    best_error = error_history[best_idx]
    
    # Create summary figure  
    fig, axes = plt.subplots(3, 3, figsize=(18, 15))
    
    # 1. Error convergence (always first)
    axes[0, 0].semilogy(range(1, len(error_history)+1), error_history, 'b-o', markersize=4)
    axes[0, 0].axhline(y=best_error, color='r', linestyle='--', alpha=0.7, label=f'Best: {best_error:.4f}')
    axes[0, 0].set_xlabel('Iteration')
    axes[0, 0].set_ylabel('Log Error')
    axes[0, 0].set_title('Error Convergence')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2-3. Parameter evolution (adapt to number of parameters)
    colors = ['r', 'g', 'orange', 'purple', 'brown']
    param_plots_positions = [(0, 1), (0, 2), (1, 0)]  # Available positions for parameter plots
    
    for i in range(min(n_params, len(param_plots_positions))):
        row, col = param_plots_positions[i]
        color = colors[i % len(colors)]
        axes[row, col].plot(range(1, len(param_history)+1), param_history[:, i], 
                           f'{color}-o', markersize=4)
        axes[row, col].axhline(y=best_params[i], color=color, linestyle='--', alpha=0.5)
        axes[row, col].set_xlabel('Iteration')
        axes[row, col].set_ylabel(param_names[i])
        axes[row, col].set_title(f'{param_names[i]} Evolution')
        axes[row, col].grid(True, alpha=0.3)
    
    # If we have more than 3 parameters, turn the remaining positions into parameter correlation
    if n_params > 3:
        # Show parameter correlation matrix in the last parameter plot position
        pos = param_plots_positions[2]  # (1, 0)
        corr_matrix = np.corrcoef(param_history.T)
        im = axes[pos[0], pos[1]].imshow(corr_matrix, cmap='coolwarm', vmin=-1, vmax=1)
        axes[pos[0], pos[1]].set_xticks(range(min(n_params, 5)))
        axes[pos[0], pos[1]].set_yticks(range(min(n_params, 5)))
        axes[pos[0], pos[1]].set_xticklabels(param_names[:5], rotation=45)
        axes[pos[0], pos[1]].set_yticklabels(param_names[:5])
        axes[pos[0], pos[1]].set_title('Parameter Correlations')
        plt.colorbar(im, ax=axes[pos[0], pos[1]], shrink=0.8)
    elif n_params == 2:
        # For 2D: parameter space exploration
        scatter = axes[1, 0].scatter(param_history[:, 0], param_history[:, 1], c=error_history, 
                                    cmap='viridis', s=60, alpha=0.8)
        axes[1, 0].scatter(best_params[0], best_params[1], c='red', s=200, marker='*', 
                          edgecolors='white', linewidth=2, label='Best')
        axes[1, 0].set_xlabel(param_names[0])
        axes[1, 0].set_ylabel(param_names[1])
        axes[1, 0].set_title('Parameter Space Exploration')
        axes[1, 0].legend()
        plt.colorbar(scatter, ax=axes[1, 0], label='Error')
    elif n_params == 1:
        # For 1D: parameter vs error scatter
        axes[1, 0].scatter(param_history[:, 0], error_history, c=range(len(error_history)), 
                          cmap='viridis', s=60, alpha=0.8)
        axes[1, 0].set_xlabel(param_names[0])
        axes[1, 0].set_ylabel('Error')
        axes[1, 0].set_title(f'{param_names[0]} vs Error')
        axes[1, 0].grid(True, alpha=0.3)
    
    # 4. Shape comparison at key points
    first_boundary = boundary_history[0]
    best_boundary = boundary_history[best_idx]
    last_boundary = boundary_history[-1]
    
    # First attempt
    axes[1, 1].plot(target_shape[:, 0], target_shape[:, 1], 'b-', linewidth=2, label='Target')
    axes[1, 1].plot(first_boundary[:, 0], first_boundary[:, 1], 'r--', linewidth=2, label='First')
    axes[1, 1].set_aspect('equal')
    axes[1, 1].set_title(f'First Attempt (Error: {error_history[0]:.4f})')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    # Best attempt
    axes[1, 2].plot(target_shape[:, 0], target_shape[:, 1], 'b-', linewidth=2, label='Target')
    axes[1, 2].plot(best_boundary[:, 0], best_boundary[:, 1], 'g--', linewidth=2, label='Best')
    axes[1, 2].set_aspect('equal')
    axes[1, 2].set_title(f'Best Attempt (Error: {best_error:.4f})')
    axes[1, 2].legend()
    axes[1, 2].grid(True, alpha=0.3)
    
    # 5. Statistics
    axes[2, 0].axis('off')
    
    # Build parameter summary text
    param_summary = []
    for i, name in enumerate(param_names):
        param_summary.append(f"Best {name}: {best_params[i]:.4f}")
        param_summary.append(f"{name} Range: [{param_history[:, i].min():.3f}, {param_history[:, i].max():.3f}]")
    
    stats_text = f"""
    Optimization Summary
    ==================
    Total Iterations: {len(error_history)}
    Best Error: {best_error:.6f}
    """ + "\n".join(param_summary) + f"""
    
    Error Reduction: {((error_history[0] - best_error) / error_history[0] * 100):.1f}%
    Final Error: {error_history[-1]:.6f}
    """
    
    axes[2, 0].text(0.05, 0.95, stats_text, transform=axes[2, 0].transAxes, 
                   verticalalignment='top', fontfamily='monospace', fontsize=10)
    
    # 6. Error histogram
    axes[2, 1].hist(error_history, bins=min(20, len(error_history)//2), alpha=0.7, color='skyblue', edgecolor='black')
    axes[2, 1].axvline(x=best_error, color='r', linestyle='--', linewidth=2, label=f'Best: {best_error:.4f}')
    axes[2, 1].set_xlabel('Error')
    axes[2, 1].set_ylabel('Frequency')
    axes[2, 1].set_title('Error Distribution')
    axes[2, 1].legend()
    axes[2, 1].grid(True, alpha=0.3)
    
    # 7. Final comparison
    axes[2, 2].plot(target_shape[:, 0], target_shape[:, 1], 'b-', linewidth=3, label='Target')
    axes[2, 2].plot(last_boundary[:, 0], last_boundary[:, 1], 'r--', linewidth=2, label='Final')
    axes[2, 2].set_aspect('equal')
    axes[2, 2].set_title(f'Final Result (Error: {error_history[-1]:.4f})')
    axes[2, 2].legend()
    axes[2, 2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/optimization_summary.png', dpi=300, bbox_inches='tight')
    plt.close()













#### ARCHIVE ####

def plot_parameter_space_detailed(param_history, error_history, OUTPUT_DIR=OUTPUT_DIR, param_names=None):
    """Create detailed parameter space exploration plots that adapt to any number of parameters"""
    param_history = np.array(param_history)
    error_history = np.array(error_history)
    n_params = param_history.shape[1]
    
    if param_names is None:
        param_names = [f'Param_{i}' for i in range(n_params)]
    
    # For 1D: simple parameter vs error plot
    if n_params == 1:
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        
        # Parameter vs error
        axes[0, 0].scatter(param_history[:, 0], error_history, c=range(len(error_history)), cmap='viridis')
        axes[0, 0].set_xlabel(param_names[0])
        axes[0, 0].set_ylabel('Error')
        axes[0, 0].set_title('Parameter vs Error')
        
        # Parameter evolution
        axes[0, 1].plot(range(1, len(param_history)+1), param_history[:, 0], 'o-')
        axes[0, 1].set_xlabel('Iteration')
        axes[0, 1].set_ylabel(param_names[0])
        axes[0, 1].set_title('Parameter Evolution')
        
        # Error evolution
        axes[1, 0].plot(range(1, len(error_history)+1), error_history, 'o-')
        axes[1, 0].set_xlabel('Iteration')
        axes[1, 0].set_ylabel('Error')
        axes[1, 0].set_title('Error Evolution')
        
        # Best parameter tracking
        axes[1, 1].axis('off')
        best_idx = np.argmin(error_history)
        stats_text = f"""
        Parameter Analysis
        ==================
        Best {param_names[0]}: {param_history[best_idx, 0]:.4f}
        Best Error: {error_history[best_idx]:.6f}
        Range: [{param_history[:, 0].min():.3f}, {param_history[:, 0].max():.3f}]
        """
        axes[1, 1].text(0.1, 0.8, stats_text, transform=axes[1, 1].transAxes, 
                        verticalalignment='top', fontfamily='monospace')
    
    # For 2D: original detailed plots
    elif n_params == 2:
        fig = plt.figure(figsize=(16, 12))
        
        # 3D surface plot of parameter space
        ax1 = fig.add_subplot(2, 3, 1, projection='3d')
        param1_vals = param_history[:, 0]
        param2_vals = param_history[:, 1]
        
        scatter = ax1.scatter(param1_vals, param2_vals, error_history, c=error_history, 
                             cmap='viridis', s=60, alpha=0.8)
        ax1.set_xlabel(param_names[0])
        ax1.set_ylabel(param_names[1])
        ax1.set_zlabel('Error')
        ax1.set_title('3D Parameter Space')
        plt.colorbar(scatter, ax=ax1, shrink=0.8)
        
        # Heatmap of explored region
        ax2 = fig.add_subplot(2, 3, 2)
        if len(param_history) > 4:
            from scipy.interpolate import griddata
            p1_grid = np.linspace(param1_vals.min(), param1_vals.max(), 50)
            p2_grid = np.linspace(param2_vals.min(), param2_vals.max(), 50)
            P1, P2 = np.meshgrid(p1_grid, p2_grid)
            error_grid = griddata((param1_vals, param2_vals), error_history, 
                                 (P1, P2), method='cubic', fill_value=np.nan)
            
            im = ax2.contourf(P1, P2, error_grid, levels=20, cmap='viridis', alpha=0.8)
            ax2.scatter(param1_vals, param2_vals, c='red', s=30, alpha=0.8, edgecolors='white')
            plt.colorbar(im, ax=ax2)
        else:
            ax2.scatter(param1_vals, param2_vals, c=error_history, cmap='viridis', s=60)
            plt.colorbar(ax2.collections[0], ax=ax2)
        
        ax2.set_xlabel(param_names[0])
        ax2.set_ylabel(param_names[1])
        ax2.set_title('Parameter Space Heatmap')
        
        # Evolution trajectory
        ax3 = fig.add_subplot(2, 3, 3)
        ax3.plot(param1_vals, param2_vals, 'o-', alpha=0.7, markersize=4)
        for i, (p1, p2) in enumerate(zip(param1_vals, param2_vals)):
            ax3.annotate(f'{i+1}', (p1, p2), xytext=(5, 5), textcoords='offset points',
                        fontsize=8, alpha=0.7)
        ax3.set_xlabel(param_names[0])
        ax3.set_ylabel(param_names[1])
        ax3.set_title('Parameter Evolution Path')
        ax3.grid(True, alpha=0.3)
        
        # Error vs iteration
        ax4 = fig.add_subplot(2, 3, 4)
        ax4.plot(range(1, len(error_history)+1), error_history, 'o-', markersize=6)
        ax4.set_xlabel('Iteration')
        ax4.set_ylabel('Error')
        ax4.set_title('Error Evolution')
        ax4.grid(True)
        
        # Parameter correlation
        ax5 = fig.add_subplot(2, 3, 5)
        correlation = np.corrcoef(param1_vals, param2_vals)[0, 1]
        ax5.scatter(param1_vals, param2_vals, c=range(len(param1_vals)), cmap='plasma', s=60)
        ax5.set_xlabel(param_names[0])
        ax5.set_ylabel(param_names[1])
        ax5.set_title(f'Parameter Correlation (r={correlation:.3f})')
        plt.colorbar(ax5.collections[0], ax=ax5, label='Iteration')
        
        # Best parameters tracking
        ax6 = fig.add_subplot(2, 3, 6)
        best_errors = []
        best_params_track = [[] for _ in range(n_params)]
        current_best = float('inf')
        
        for i, err in enumerate(error_history):
            if err < current_best:
                current_best = err
            best_errors.append(current_best)
            
            best_idx = np.argmin(error_history[:i+1])
            for j in range(n_params):
                best_params_track[j].append(param_history[best_idx, j])
        
        ax6_twin = ax6.twinx()
        ax6.plot(range(1, len(best_errors)+1), best_errors, 'g-', linewidth=2, label='Best Error')
        colors = ['r', 'b']
        for j in range(min(2, n_params)):  # Only plot first 2 params for clarity
            ax6_twin.plot(range(1, len(best_params_track[j])+1), best_params_track[j], 
                         f'{colors[j]}--', alpha=0.7, label=f'Best {param_names[j]}')
        
        ax6.set_xlabel('Iteration')
        ax6.set_ylabel('Best Error', color='g')
        ax6_twin.set_ylabel('Parameter Values', color='k')
        ax6.set_title('Best Parameters Evolution')
        ax6.legend(loc='upper left')
        ax6_twin.legend(loc='upper right')
    
    # For 3D+: pairwise plots and projections
    else:
        n_plots = min(6, n_params)  # Limit to 6 parameters for display
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        axes = axes.flatten()
        
        # Individual parameter evolution plots
        for i in range(n_plots):
            if i < n_params:
                axes[i].plot(range(1, len(param_history)+1), param_history[:, i], 'o-', markersize=4)
                axes[i].set_xlabel('Iteration')
                axes[i].set_ylabel(param_names[i])
                axes[i].set_title(f'{param_names[i]} Evolution')
                axes[i].grid(True, alpha=0.3)
            else:
                axes[i].axis('off')
        
        # If we have space, add error evolution and correlation matrix
        if n_params < 6:
            # Error evolution
            axes[n_params].plot(range(1, len(error_history)+1), error_history, 'o-', color='red')
            axes[n_params].set_xlabel('Iteration')
            axes[n_params].set_ylabel('Error')
            axes[n_params].set_title('Error Evolution')
            axes[n_params].grid(True, alpha=0.3)
            
            # Correlation matrix (if space allows)
            if n_params < 5:
                corr_matrix = np.corrcoef(param_history.T)
                im = axes[n_params+1].imshow(corr_matrix, cmap='coolwarm', vmin=-1, vmax=1)
                axes[n_params+1].set_xticks(range(n_params))
                axes[n_params+1].set_yticks(range(n_params))
                axes[n_params+1].set_xticklabels(param_names, rotation=45)
                axes[n_params+1].set_yticklabels(param_names)
                axes[n_params+1].set_title('Parameter Correlation Matrix')
                plt.colorbar(im, ax=axes[n_params+1], shrink=0.8)
    
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/parameter_space_detailed.png', dpi=300, bbox_inches='tight')
    plt.close()