"""Plotting and misc. calculations/utilities"""
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.path import Path
import matplotlib.cm as cm
import matplotlib.colors as mcolors
from labellines import labelLines
import numpy as np
import pandas as pd
import re
import pickle
from scipy.special import chebyt
from scipy.spatial.distance import directed_hausdorff
from numpy.polynomial.chebyshev import chebgauss
import os
from config import D0, KDEATH, DT, TMAX, T_DORMANT, XMIN, XMAX, YMIN, YMAX, M_LENGTH, G_LENGTH, KAPPA0, KAPPA2, CONVERSION_FACTOR_UM
from email.message import EmailMessage
import smtplib

def send_email_notification(subject, body, recipient='brewsmith.a@northeastern.edu'):
    """Send email notification using Gmail SMTP. Requires environment variables:
    OPTIM_SENDER_EMAIL and OPTIM_SENDER_PASSWORD"""
    try:
        sender_email = os.environ.get('OPTIM_SENDER_EMAIL')
        sender_password = os.environ.get('OPTIM_SENDER_PASSWORD')
        
        if not sender_email or not sender_password:
            print("Email credentials not found in environment variables. Skipping notification.")
            return False
        
        msg = EmailMessage()
        msg['From'] = sender_email
        msg['To'] = recipient
        msg['Subject'] = subject
        msg.set_content(body)
        
        with smtplib.SMTP_SSL('smtp.gmail.com', 465) as smtp:
            smtp.login(sender_email, sender_password)
            smtp.send_message(msg)
        print(f"Email notification sent to {recipient}")
        return True
    except Exception as e:
        print(f"Failed to send email notification: {e}")
        return False
def truncate_colormap(cmap, minval=0.0, maxval=10.0, n=256):
    new_cmap = ListedColormap(cmap(np.linspace(minval, maxval, n)))
    return new_cmap

def morphometrics(Xe, pos=None, x_cut=1.0):
    """Calculate morphometrics of the blastema growth region."""

    growth_boundary = Xe[Xe[:, 0] > x_cut]    
    if len(growth_boundary) < 3:
        return 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
    
    growth_boundary = growth_boundary[np.argsort(growth_boundary[:, 1])]
    
    y_min = growth_boundary[:, 1].min()
    y_max = growth_boundary[:, 1].max()
    y_min_whole_limb = Xe[:, 1].min()
    y_max_whole_limb = Xe[:, 1].max()

    polygon_vertices = np.array([
        [x_cut, y_min],
        *growth_boundary.tolist(),
        [x_cut, y_max],
        [x_cut, y_min]
    ])

    polygon_vertices_whole_limb = np.array([
        [x_cut, y_min_whole_limb],
        *Xe.tolist(),
        [x_cut, y_max_whole_limb],
        [x_cut, y_min_whole_limb]
    ])
    
    # Calculate area using shoelace formula
    x_poly, y_poly = polygon_vertices[:, 0], polygon_vertices[:, 1]
    area_growth_region = 0.5 * np.abs(np.sum(x_poly[:-1] * y_poly[1:]) - np.sum(y_poly[:-1] * x_poly[1:]))
    
    if len(growth_boundary) > 1:
        boundary_diffs = np.diff(growth_boundary, axis=0)
        perimeter = np.sum(np.sqrt(np.sum(boundary_diffs**2, axis=1)))
    else:
        perimeter = 0.0
    
    x_growth = growth_boundary[:, 0]
    y_growth = growth_boundary[:, 1]
    
    a = np.max(x_growth) - x_cut if len(x_growth) > 0 else 0.0 # Outgrowth length
    b = (np.max(y_growth) - np.min(y_growth)) / 2 if len(y_growth) > 0 else 2.5 # Half-width of the growth region

    # Calculate derived metrics
    if len(x_growth) > 2 and area_growth_region > 0 and a > 0 and b > 0:
        AR_outgrowth = a / b
        AR_whole_limb = (Xe[:,0].max() - Xe[:,0].min()) / (Xe[:,1].max() - Xe[:,1].min())
        ellipticity = (a - b) / a
        roundness = (perimeter**2) / (4 * np.pi * area_growth_region)
        
        # Monte Carlo volume fraction
        limb_path = Path(Xe)
        radius = D0 / 2
        x_min, x_max = Xe[:, 0].min(), Xe[:, 0].max()
        y_min, y_max = Xe[:, 1].min(), Xe[:, 1].max()
        n_samples = 50000
        
        points = np.column_stack([
            np.random.uniform(x_min, x_max, n_samples),
            np.random.uniform(y_min, y_max, n_samples)
        ])
        
        in_limb_mask = limb_path.contains_points(points)
        points_in_limb = points[in_limb_mask]
        occupied_mask = np.zeros(len(points_in_limb), dtype=bool)
        active_cells = pos[~np.isnan(pos[:, 0])]
        
        for cell_pos in active_cells:
            dist = np.linalg.norm(points_in_limb - cell_pos, axis=1)
            occupied_mask |= (dist <= radius)
        
        volume_fraction = np.sum(occupied_mask) / np.sum(in_limb_mask) if np.sum(in_limb_mask) > 0 else 0
    else:
        AR_outgrowth = AR_whole_limb = ellipticity = roundness = volume_fraction = 0.0
        if area_growth_region <= 0:
            area_growth_region = perimeter = 0.0

    return area_growth_region, perimeter, AR_whole_limb, AR_outgrowth, ellipticity, roundness, a, b, volume_fraction

def morphometrics_time_series_plot(morphometrics_data, times, OUTPUT_DIR='', metric='volume_fraction', title='Volume Fraction'):
    """Plot the time series of morphometrics."""
    fig, ax = plt.subplots(figsize=(10, 8))

    ax.plot(times[1:], morphometrics_data[metric], linewidth=5, label=title)
    ax.tick_params(axis='both', which='major', labelsize=16)
    ax.set_xlabel('Time (days)', fontsize=16)
    ax.set_ylabel(title, fontsize=16)
    ax.set_title(f'{title} over time', fontsize=20)
    plt.ylim(0, 1)
    plt.xlim(0.05,max(times)+0.05)
    fig.savefig(os.path.join(OUTPUT_DIR, f'{metric}_time_series.pdf'))
    plt.close(fig)

def density_heatmap(
    kb_vals, pos, Xe, x_cut,
    bin_size=0.1, OUTPUT_DIR='.', shading='auto', conversion_factor_um=CONVERSION_FACTOR_UM, show_real_units=True,
    fig_mode=False, show_title=True, show_colorbar=True, show_info_box=True, mitotic_only=False, cycle_phases=None
):
    if cycle_phases is not None and mitotic_only:
        # Only include cells in mitotic phase (cycle_phases == 1)
        mitotic_mask = (cycle_phases == 1) & ~np.isnan(pos[:, 0])
        pos_filtered = pos[mitotic_mask]
    else:
        active_mask = ~np.isnan(pos[:, 0])
        pos_filtered = pos[active_mask]
    
    pos_display = pos_filtered * conversion_factor_um
    Xe_display = Xe * conversion_factor_um if Xe is not None else None
    xmin_display = XMIN * conversion_factor_um
    xmax_display = XMAX * conversion_factor_um
    ymin_display = YMIN * conversion_factor_um
    ymax_display = YMAX * conversion_factor_um
    bin_size_display = bin_size * conversion_factor_um

    x_edges = np.arange(xmin_display, xmax_display + bin_size_display, bin_size_display)
    y_edges = np.arange(ymin_display, ymax_display + bin_size_display, bin_size_display)
    H, xedges, yedges = np.histogram2d(pos_display[:, 0], pos_display[:, 1],
                                       bins=[x_edges, y_edges])

    if shading == 'gouraud':
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

    fig, ax = plt.subplots(constrained_layout=True, figsize=(10, 8))
    ax.set_aspect('equal', adjustable='box')

    # Convert density to real units (cells per 1000 um^2)
    bin_area_um2 = bin_size_display * bin_size_display
    if shading == 'gouraud':
        H_plot_real_units = H_plot / bin_area_um2 * 1000
    else:
        H_plot_real_units = H_plot / bin_area_um2 * 1000
    
    # Set vmax to get 20 cells per 1000 um^2 on the colorbar
    vmax_sim_units = 8.0
    vmax_real_units = (vmax_sim_units / bin_area_um2) * 1000
    
    # 4) plot the heatmap with standardized color scale
    if shading == 'gouraud':
        pcm = ax.pcolormesh(Xc, Yc, H_plot_real_units, cmap='Greys', shading=shading, vmin=0, vmax=vmax_real_units)
    else:
        pcm = ax.pcolormesh(X, Y, H_plot_real_units, cmap='Greys', shading=shading, vmin=0, vmax=vmax_real_units)
    
    if show_colorbar:
        cbar = fig.colorbar(pcm, ax=ax, label='Cell density (cells/1000 um^2)')

    if Xe_display is not None:
        if kb_vals is not None:

            vmin_kb = 1.0
            vmax_kb = 150.0
            norm = mcolors.Normalize(vmin=vmin_kb, vmax=vmax_kb, clip=True)
            cmap_kb = truncate_colormap(cm.get_cmap('PuRd'), 0.2, 1.0)
            for i in range(len(Xe_display)-1):
                color = cmap_kb(norm(kb_vals[i]))
                ax.plot([Xe_display[i,0], Xe_display[i+1,0]], 
                        [Xe_display[i,1], Xe_display[i+1,1]], 
                        '-', lw=2, color=color)
        else:
            # Plot boundary without color coding
            ax.plot(Xe_display[:, 0], Xe_display[:, 1], 'r-', lw=2, label='Boundary')

    xmin_um = XMIN * conversion_factor_um
    xmax_um = XMAX * conversion_factor_um
    ymin_um = YMIN * conversion_factor_um
    ymax_um = YMAX * conversion_factor_um

    ax.set_xlim(xmin_um, xmax_um)
    ax.set_ylim(ymin_um, ymax_um)
    ax.set_xticks(np.arange(xmin_um, xmax_um + 1, 200))
    ax.set_yticks(np.arange(ymin_um, ymax_um + 1, 200))
    ax.set_xlim(xmin_display, xmax_display)
    ax.set_ylim(ymin_display, ymax_display)
    
    ax.set_xlabel(r'x ($\mu$m)', fontsize=28, fontweight='bold')
    ax.set_ylabel(r'y ($\mu$m)', fontsize=28, fontweight='bold')
    ax.tick_params(axis='both', which='major', labelsize=20)

def cycle_plot(times, cell_count, N0, OUTPUT_DIR=None):
    """
    Plot cell count vs time with a numerical solution of the two-compartment ODE system.

    ODEs (post dormancy):
        dG/dt = -(k1+kdeath) * G + 2*k2 * M
        dM/dt = k1 * G - (k2 + kdeath) * M

    Dormancy (t < T_DORMANT): no cycling/division, only death
        dG/dt = -kdeath * G,  dM/dt = -kdeath * M (with M(0)=0)
    """
    times = np.array(times)
    cell_count = np.array(cell_count)

    k1 = 1.0 / (G_LENGTH)
    k2 = 1.0 / (M_LENGTH)
    kdeath = KDEATH

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

    out_path = os.path.join(OUTPUT_DIR, 'cell_count_vs_time_numeric.pdf') if OUTPUT_DIR is not None else 'cell_count_vs_time_numeric.pdf'
    fig.savefig(out_path, dpi=300)
    plt.close(fig)

def phase_distribution_plot(times, Gphase, Mphase, fit=False, OUTPUT_DIR=None):
    """Plot proportion of cells in each phase (G1 or S/G2/M) with numerical ODE solution"""
    times = np.array(times)
    Gphase = np.array(Gphase)
    Mphase = np.array(Mphase)

    k1 = 1.0 / G_LENGTH
    k2 = 1.0 / M_LENGTH
    kdeath = KDEATH

    G_arr = np.zeros_like(times, dtype=float)
    M_arr = np.zeros_like(times, dtype=float)

    # Initial conditions
    G = float(Gphase[0])
    M = float(Mphase[0])
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

    # Plot
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(times, Gphase, '-', label='G0/G1 (sim)', color=(223/255, 224/255, 95/255))
    ax.plot(times, Mphase, '-', label='S/G2/M (sim)', color=(120/255, 237/255, 240/255))
    ax.plot(times, G_arr, '--', label='G0/G1 (model)', color=(223/255, 224/255, 95/255))
    ax.plot(times, M_arr, '--', label='S/G2/M (model)', color=(120/255, 237/255, 240/255))

    ax.set_xlabel('Time')
    ax.set_ylabel('Cell Count')
    ax.set_title(f'Phase Distributions (numeric k1={k1:.3g}, k2={k2:.3g})')
    ax.legend()
    ax.set_xlim(0, TMAX)
    ax.grid(True)

    out_path = os.path.join(OUTPUT_DIR, 'phase_distr.pdf') if OUTPUT_DIR is not None else 'phase_distr.pdf'
    fig.savefig(out_path, dpi=300)
    plt.close(fig)

def growth_plot(Xe0, Xe_final, title, Xb=None, conversion_factor_um=CONVERSION_FACTOR_UM, OUTPUT_DIR='osei'):
    """Plot the limb with highlighted outgrowth region"""

    Xe_growth = Xe_final[Xe_final[:,0]>0]
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_aspect('equal')
    
    xmin_um = XMIN * conversion_factor_um
    xmax_um = XMAX * conversion_factor_um
    ymin_um = YMIN * conversion_factor_um
    ymax_um = YMAX * conversion_factor_um

    ax.set_xlim(xmin_um, xmax_um)
    ax.set_ylim(ymin_um, ymax_um)
    ax.set_xticks(np.arange(xmin_um, xmax_um + 1, 200))
    ax.set_yticks(np.arange(ymin_um, ymax_um + 1, 200))
    ax.set_xlabel(r'x ($\mu$m)', fontsize=28, fontweight='bold')
    ax.set_ylabel(r'y ($\mu$m)', fontsize=28, fontweight='bold')
    ax.tick_params(axis='both', which='major', labelsize=20)
    
    # Convert boundary coordinates to micrometers
    Xe0_display = Xe0 * conversion_factor_um
    Xe_display = Xe_final * conversion_factor_um
    Xe_growth_display = Xe_growth * conversion_factor_um
    if Xb is not None:
        Xb_display = Xb * conversion_factor_um if Xb is not None else None
    else:
        Xb_display = None

    ax.plot(Xe0_display[:,0], Xe0_display[:,1], 'k--', lw=2, alpha=1.0, label='Initial boundary')
    ax.plot(Xe_display[:,0], Xe_display[:,1], color='blue', lw=2, label='Final boundary')
    try:
        ax.plot(Xe_growth_display[:,0], Xe_growth_display[:,1],color='lightskyblue', lw=2, label='Growth region')
        # Shade outgrowth region light blue
        y_min_cut = Xe_growth_display[:, 1].min()
        y_max_cut = Xe_growth_display[:, 1].max()
        
        vertices = []
        vertices.append([0, y_min_cut])
        boundary_sorted = Xe_growth_display[np.argsort(Xe_growth_display[:, 1])]
        vertices.extend(boundary_sorted.tolist())
        vertices.append([0, y_max_cut])
        
        from matplotlib.patches import Polygon
        poly = Polygon(vertices, closed=True, color='lightskyblue', alpha=0.4, label='Outgrowth region')
        ax.add_patch(poly)
    except Exception as e:
        print(f'Exception: {e} \n ignoring growth region...')

    if Xb_display is not None:
        ax.plot(Xb_display[:,0], Xb_display[:,1], 'k-', lw=2, alpha=0.5, label='Bone boundary')

    fig.tight_layout()    
    fig.savefig(f'{OUTPUT_DIR}/{re.sub(r'\s+', '', title)}_growth.pdf', dpi=400, bbox_inches='tight', pad_inches=0)
    plt.close(fig)



def runtime_plot(cell_count, steptimes, OUTPUT_DIR=None):
    """Create a plot of runtime vs time"""
    fig, ax = plt.subplots(figsize=(8,5))
    skip = int(T_DORMANT / (DT*1000))
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
    out_path = os.path.join(OUTPUT_DIR, 'runtime_vs_cell_count.pdf') if OUTPUT_DIR is not None else 'runtime_vs_cell_count.pdf'
    fig.savefig(out_path, dpi=300)
    plt.close(fig)




def MSD_plot(positions, ids=None, fit_frac=1.0, FRAME_SKIP=1000, OUTPUT_DIR=None):
    """
    Compute and plot mean squared displacement (MSD) for selected particle IDs,
    fit a power law to the MSD vs time curve, and extract the scaling exponent.
    """
    positions = np.array(positions)
    n_frames, n_particles, dim = positions.shape
    
    # Find particles that exist throughout the simulation (no NaN values)
    valid_particles = []
    for p in range(n_particles):
        if not np.any(np.isnan(positions[:, p, :])):
            valid_particles.append(p)
    
    if len(valid_particles) == 0:
        print("Warning: No cells exist throughout entire simulation.")
        return np.array([]), np.array([]), np.nan, np.nan
    
    # Select subset of valid particles
    if ids is not None:
        ids = np.array(list(ids))
        valid_ids = [p for p in ids if p in valid_particles]
        if len(valid_ids) == 0:
            print(f"Warning: None of the requested particle IDs {ids} are valid throughout simulation.")
            print(f"Valid particles: {valid_particles[:10]}... (showing first 10)")
            valid_ids = valid_particles[:min(30, len(valid_particles))]
    else:
        # Use all particles
        valid_ids = valid_particles
    
        pos = positions[:, valid_ids, :]
    n_frames, n_selected, dim = pos.shape
    
    msd_values = np.zeros(n_frames)
    
    # Compute MSD for each lag time
    for m in range(n_frames):
        if n_frames - m <= 1:
            msd_values[m] = np.nan
            continue
            
        displacements = pos[m:] - pos[:n_frames - m]
        sq_displacements = np.sum(displacements**2, axis=2)
        msd_values[m] = np.mean(sq_displacements)
    
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
    plt.loglog(times[valid_mask], msd_values[valid_mask], 'o-', 
               alpha=0.7, markersize=4, label='MSD data')
    
    if not np.isnan(alpha) and len(fit_t) > 0:
        plt.loglog(fit_t, np.exp(intercept) * fit_t**alpha, '--', 
                   color='red', linewidth=2, label=f'Fit: alpha = {alpha:.3f}')
    
    plt.xlabel('Time (simulation units)')
    plt.ylabel('MSD (position units^2)')
    plt.title(f'Mean Squared Displacement (N = {len(valid_ids)} particles)')
    plt.grid(True, alpha=0.3)
    
    if not np.isnan(alpha):
        textstr = f'Scaling exponent alpha = {alpha:.3f}\nDiffusion regime: '
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
    out_path = os.path.join(OUTPUT_DIR, 'msd.pdf') if OUTPUT_DIR is not None else 'msd.pdf'
    plt.savefig(out_path, dpi=300, bbox_inches='tight', pad_inches=0)
    plt.close()
    
    return times, msd_values, alpha, intercept

def cell_type_proportions_plot(times, positions_history, migrant_history=None, 
                               intercal_history=None, jammed_history=None, 
                               OUTPUT_DIR=None, x_cut=None):
    """Plot the proportion of different cell types over time."""
    times = np.array(times)
    n_frames = len(positions_history)
    
    # Initialize arrays to store counts
    total_counts = np.zeros(n_frames)
    migrant_counts = np.zeros(n_frames)
    intercal_counts = np.zeros(n_frames)
    jammed_counts = np.zeros(n_frames)
    normal_counts = np.zeros(n_frames)
    
    # Count cell types at each time point
    for i, pos in enumerate(positions_history):
        active_mask = ~np.isnan(pos[:, 0])
        if x_cut is not None:
            active_mask = active_mask & (pos[:, 0] > x_cut)
        
        n_active = np.sum(active_mask)
        total_counts[i] = n_active
        
        if n_active == 0:
            continue
        
        # Count migrant cells
        if migrant_history is not None and i < len(migrant_history):
            migrant_mask = np.zeros(len(active_mask), dtype=bool)
            if migrant_history[i] is not None:
                # Ensure mask is the right size
                migrant_data = np.asarray(migrant_history[i])
                if len(migrant_data) == len(active_mask):
                    migrant_mask = migrant_data.astype(bool)
            migrant_counts[i] = np.sum(active_mask & migrant_mask)
        
        # Count intercalation cells
        if intercal_history is not None and i < len(intercal_history):
            intercal_mask = np.zeros(len(active_mask), dtype=bool)
            if intercal_history[i] is not None:
                intercal_data = np.asarray(intercal_history[i])
                if len(intercal_data) == len(active_mask):
                    intercal_mask = intercal_data.astype(bool)
            intercal_counts[i] = np.sum(active_mask & intercal_mask)
        
        # Count jammed cells
        if jammed_history is not None and i < len(jammed_history):
            jammed_mask = np.zeros(len(active_mask), dtype=bool)
            if jammed_history[i] is not None:
                jammed_data = np.asarray(jammed_history[i])
                if len(jammed_data) == len(active_mask):
                    jammed_mask = jammed_data.astype(bool)
            jammed_counts[i] = np.sum(active_mask & jammed_mask)
        
        # Normal cells are those that aren't any special type
        special_cells = migrant_counts[i] + intercal_counts[i] + jammed_counts[i]
        normal_counts[i] = n_active - special_cells
    
    # Calculate proportions (avoid division by zero)
    migrant_prop = np.divide(migrant_counts, total_counts, 
                            out=np.zeros_like(migrant_counts), where=total_counts>0)
    intercal_prop = np.divide(intercal_counts, total_counts,
                             out=np.zeros_like(intercal_counts), where=total_counts>0)
    jammed_prop = np.divide(jammed_counts, total_counts,
                           out=np.zeros_like(jammed_counts), where=total_counts>0)
    normal_prop = np.divide(normal_counts, total_counts,
                           out=np.zeros_like(normal_counts), where=total_counts>0)
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10))
    
    # Plot 1: Absolute counts
    ax1.plot(times, total_counts, 'k-', linewidth=2, label='Total cells', alpha=0.7)
    ax1.plot(times, normal_counts, '-', linewidth=2, label='Normal cells', 
             color=(223/255, 224/255, 95/255))
    if migrant_history is not None:
        ax1.plot(times, migrant_counts, '-', linewidth=2, label='Migrant cells', color='orange')
    if intercal_history is not None:
        ax1.plot(times, intercal_counts, '-', linewidth=2, label='Intercalation cells', color='red')
    if jammed_history is not None:
        ax1.plot(times, jammed_counts, '-', linewidth=2, label='Jammed cells', color='blue')
    
    ax1.set_xlabel('Time', fontsize=12)
    ax1.set_ylabel('Number of Cells', fontsize=12)
    if x_cut is not None:
        x_cut_str = '0' if abs(x_cut) < 0.01 else f'{x_cut:.2f}'
        title_suffix = f' (x > {x_cut_str})'
    else:
        title_suffix = ''
    ax1.set_title(f'Cell Type Counts Over Time{title_suffix}', fontsize=14, fontweight='bold')
    ax1.legend(loc='best')
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(0, times[-1])
    
    # Plot 2: Proportions (stacked area chart)
    ax2.fill_between(times, 0, normal_prop, 
                     color=(223/255, 224/255, 95/255), alpha=0.7, label='Normal cells')
    
    cumsum = normal_prop.copy()
    
    if migrant_history is not None:
        ax2.fill_between(times, cumsum, cumsum + migrant_prop, 
                        color='orange', alpha=0.7, label='Migrant cells')
        cumsum += migrant_prop
    
    if intercal_history is not None:
        ax2.fill_between(times, cumsum, cumsum + intercal_prop, 
                        color='red', alpha=0.7, label='Intercalation cells')
        cumsum += intercal_prop
    
    if jammed_history is not None:
        ax2.fill_between(times, cumsum, cumsum + jammed_prop, 
                        color='blue', alpha=0.7, label='Jammed cells')
    
    ax2.set_xlabel('Time', fontsize=12)
    ax2.set_ylabel('Proportion of Cells', fontsize=12)
    if x_cut is not None:
        # Format x_cut nicely - if close to 0, just show 0
        x_cut_str = '0' if abs(x_cut) < 0.01 else f'{x_cut:.2f}'
        title_suffix2 = f' (x > {x_cut_str})'
    else:
        title_suffix2 = ''
    ax2.set_title(f'Cell Type Proportions Over Time{title_suffix2}', fontsize=14, fontweight='bold')
    ax2.legend(loc='best')
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(0, times[-1])
    ax2.set_ylim(0, 1.0)
    
    # Add text box with final statistics
    final_stats = (
        f'Final Counts:\n'
        f'Total: {int(total_counts[-1])}\n'
        f'Normal: {int(normal_counts[-1])} ({normal_prop[-1]*100:.1f}%)\n'
        f'Migrant: {int(migrant_counts[-1])} ({migrant_prop[-1]*100:.1f}%)\n'
        f'Intercal: {int(intercal_counts[-1])} ({intercal_prop[-1]*100:.1f}%)\n'
        f'Jammed: {int(jammed_counts[-1])} ({jammed_prop[-1]*100:.1f}%)'
    )
    
    ax2.text(0.02, 0.98, final_stats,
            transform=ax2.transAxes,
            verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.9, edgecolor='black'),
            fontsize=10)
    
    plt.tight_layout()
    
    # Save plot
    out_path = os.path.join(OUTPUT_DIR, 'cell_type_proportions.pdf') if OUTPUT_DIR is not None else 'cell_type_proportions.pdf'
    fig.savefig(out_path, dpi=300, bbox_inches='tight', pad_inches=0.1)
    plt.close(fig)
    
    return {
        'times': times,
        'total_counts': total_counts,
        'normal_counts': normal_counts,
        'migrant_counts': migrant_counts,
        'intercal_counts': intercal_counts,
        'jammed_counts': jammed_counts,
        'normal_prop': normal_prop,
        'migrant_prop': migrant_prop,
        'intercal_prop': intercal_prop,
        'jammed_prop': jammed_prop
    }

def simulation_snapshot(t, pos, Xe, Xb, cycle_phases, kb_vals=None, x_bounds=(XMIN, XMAX), y_bounds=(YMIN, YMAX), 
                        boundary=True, migrant_cells=None, intercal_cells=None, jammed_cells=None, x_cut=0, 
                        show_real_units=True, regulation_front=None, title_suffix='', OUTPUT_DIR=None, filename=None):
    """Save a single frame of simulation as a PDF image."""
    from config import VIDEO_PARAMS, BONE_VISUALIZATION, MIGRATION_ENABLED, MIGRATION_DELAY, REGULATION_FRONT_FLAG, GRADIENT
    
    if filename is None:
        filename = f'snapshot_t_{t:.4f}.pdf'
    
    fig, ax = plt.subplots(figsize=VIDEO_PARAMS['figsize'], dpi=VIDEO_PARAMS['dpi'])
    
    active_mask = ~np.isnan(pos[:, 0])
    phase0 = np.where((cycle_phases == 0) & active_mask)[0]
    phase1 = np.where((cycle_phases == 1) & active_mask)[0]
    ax.clear()
    
    if show_real_units:
        x_bounds_display = (x_bounds[0] * CONVERSION_FACTOR_UM, x_bounds[1] * CONVERSION_FACTOR_UM)
        y_bounds_display = (y_bounds[0] * CONVERSION_FACTOR_UM, y_bounds[1] * CONVERSION_FACTOR_UM)
        pos_display = pos * CONVERSION_FACTOR_UM
        Xe_display = Xe * CONVERSION_FACTOR_UM
        if BONE_VISUALIZATION:
            Xb_display = Xb * CONVERSION_FACTOR_UM
    else:
        x_bounds_display = x_bounds
        y_bounds_display = y_bounds
        pos_display = pos
        Xe_display = Xe
        if BONE_VISUALIZATION:
            Xb_display = Xb
    
    # Plot boundary
    if boundary:
        if kb_vals is not None:
            vmin = 1.0
            vmax = 150.0
            
            # Use normalized color mapping for all cases
            norm = mcolors.Normalize(vmin=vmin, vmax=vmax, clip=True)
            cmap = truncate_colormap(cm.get_cmap('PuRd'), 0.2, 1.0)
            for i in range(len(Xe)-1):
                color = cmap(norm(kb_vals[i]))
                ax.plot([Xe_display[i,0], Xe_display[i+1,0]], 
                        [Xe_display[i,1], Xe_display[i+1,1]], 
                        '-', lw=2, color=color)

        if BONE_VISUALIZATION:
            for i in range(len(Xb)-1):
                ax.plot([Xb_display[i,0], Xb_display[i+1,0]], 
                        [Xb_display[i,1], Xb_display[i+1,1]], 
                        '-', lw=3, color='black', alpha=0.8)
            ax.plot([Xb_display[-1,0], Xb_display[0,0]], 
                    [Xb_display[-1,1], Xb_display[0,1]], 
                    '-', lw=3, color='black', alpha=0.8)
        
        x_cut_display = x_cut * CONVERSION_FACTOR_UM if show_real_units else x_cut
        ax.axvline(x=x_cut_display, color='black', linestyle='--', alpha=0.8, linewidth=1, label='Amputation Plane')

    # Plot regulation front if provided (used for migration and/or proliferation gradient)
    if ((MIGRATION_ENABLED or GRADIENT == 'zone') and 
        REGULATION_FRONT_FLAG and
        regulation_front is not None and 
        not np.isinf(regulation_front) and 
        t >= MIGRATION_DELAY):
        migration_front_display = regulation_front * CONVERSION_FACTOR_UM if show_real_units else regulation_front
        ax.axvline(x=migration_front_display, color='purple', linestyle='--', alpha=0.8, linewidth=2, label='Regulation Front')
    
    cell_radius = D0 / 2
    
    # Calculate cell size
    bbox = ax.get_window_extent().transformed(ax.figure.dpi_scale_trans.inverted())
    width_inch = bbox.width
    data_width = x_bounds[1] - x_bounds[0]
    points_per_data_unit = (width_inch * 72) / data_width  # 72 points per inch
    r_points = cell_radius * points_per_data_unit
    area_points2 = np.pi * r_points**2

    # Plot cells
    ax.scatter(pos_display[phase0, 0], pos_display[phase0, 1], s=area_points2, facecolor=(223/255, 224/255, 95/255), edgecolors='black')
    ax.scatter(pos_display[phase1, 0], pos_display[phase1, 1], s=area_points2, facecolor=(120/255, 237/255, 240/255), edgecolors='black')

    if REGULATION_FRONT_FLAG:
        currently_migrating = active_mask & migrant_cells & (pos[:, 0] > regulation_front)
    elif not REGULATION_FRONT_FLAG:
        currently_migrating = active_mask & migrant_cells
    if migrant_cells is not None:
        ax.scatter(pos_display[currently_migrating, 0], pos_display[currently_migrating, 1], s=area_points2, facecolor='none', edgecolors='purple', linewidths=1.5, label='Migration')
    if intercal_cells is not None:
        intercal_and_active = active_mask & intercal_cells
        ax.scatter(pos_display[intercal_and_active, 0], pos_display[intercal_and_active, 1], s=area_points2, facecolor='none', edgecolors='red', linewidths=1.5, label='Intercalation')
    if jammed_cells is not None:
        jammed_and_active = active_mask & jammed_cells
        ax.scatter(pos_display[jammed_and_active, 0], pos_display[jammed_and_active, 1], s=area_points2, facecolor='none', edgecolors='blue', linewidths=1.5, label='Jammed')

    # Set final axis properties
    ax.set_xlim(x_bounds_display)
    ax.set_ylim(y_bounds_display)
    
    if show_real_units:
        ax.set_xlabel('x (um)', fontsize=12, fontweight='bold')
        ax.set_ylabel('y (um)', fontsize=12, fontweight='bold')
    else:
        ax.set_xlabel('x', fontsize=12)
        ax.set_ylabel('y', fontsize=12)
    
    ax.grid(False)
    ax.set_aspect('equal', 'box')
    ax.set_title(f"T = {t:.4f}{title_suffix}")
    
    # Save figure
    output_path = os.path.join(OUTPUT_DIR, filename) if OUTPUT_DIR else filename
    fig.savefig(output_path, dpi=300, bbox_inches='tight', pad_inches=0.1)
    plt.close(fig)

"""
Distance metrics, curve processing and Chebyshev coefficient calculations. 
Used to calculate errors for parameter inference.
"""

def parameterize_curve(curve):
    """take curve of (x,y) points, return (theta, r) ordered by theta [-pi/2, pi/2]"""
    x, y = curve[:,0], curve[:,1]
    theta = np.atan(y/x)
    r = np.sqrt(x**2 + y**2)
    pCurve = np.column_stack((theta, r))
    sort_indices = pCurve[:, 0].argsort()
    pCurve = pCurve[sort_indices]
    return pCurve

def compute_average(curves, in_coords='xy', out_coords='polar'):
    """
    input is assumed to by in xy coords
    returns an average curve in terms of theta and r or xy depending on coords param"""
    if in_coords == 'xy':
        pCurves = [parameterize_curve(i) for i in curves] # Parametrize to polar if not already
    else:
        pCurves = curves
    curve_lengths = [len(i) for i in pCurves]
    most_points, most_points_idx = max(curve_lengths), curve_lengths.index(max(curve_lengths))
    common_theta = pCurves[most_points_idx][:,0]

    # interpolate r over common theta for all curves
    pCurves_r = [np.interp(common_theta, i[:,0], i[:,1]) for i in pCurves]    
    # Average r values across all curves at each theta point
    r_arr = np.mean(pCurves_r, axis=0)
    pCurve_avg = np.column_stack((common_theta, r_arr))
    if out_coords == 'polar':
        return pCurve_avg
    elif out_coords == 'xy':
        theta_avg, r_avg = pCurve_avg[:,0], pCurve_avg[:,1]
        x_avg = r_avg*np.cos(theta_avg)
        y_avg = r_avg*np.sin(theta_avg)
        pCurve_avg = np.column_stack([x_avg, y_avg])
        return pCurve_avg

def omega(x):
    """Chebyshev weight function with singularity protection"""
    x = np.clip(x, -1 + 1e-15, 1 - 1e-15)
    return 1 / np.sqrt(1 - x**2)

def cheb_basis(n=0):
    return lambda x: chebyt(n)(x)

def chebyshev_inner_product(f, g, n_points=50):
    """Inner product using Gauss-Chebyshev quadrature."""
    nodes, weights = chebgauss(n_points)
    return np.sum(f(nodes) * g(nodes) * weights)

def inner_product(f, g, omega, type='data'):
    """Compute inner product with Chebyshev weight function."""
    if type == 'data':
        nodes, weights = chebgauss(len(f))
        f_interp = np.interp(nodes, f[:, 0], f[:, 1])
        return np.sum(f_interp * g(nodes) * weights)
    elif type == 'function':
        return chebyshev_inner_product(f, g)

def scale_shape(data, target_range=(-1, 1)):
    """Scale data to target range while preserving aspect ratio."""
    x_min, x_max = data[:, 0].min(), data[:, 0].max()
    y_min, y_max = data[:, 1].min(), data[:, 1].max()
    
    x_range = x_max - x_min
    y_range = y_max - y_min
    max_range = max(x_range, y_range)
    
    scale_factor = (target_range[1] - target_range[0]) / max_range
    
    x_center = (x_min + x_max) / 2
    y_center = (y_min + y_max) / 2
    
    scaled_data = data.copy()
    scaled_data[:, 0] = (data[:, 0] - x_center) * scale_factor
    scaled_data[:, 1] = (data[:, 1] - y_center) * scale_factor
    
    return scaled_data, scale_factor, x_center, y_center

def coefficients(f, n, type='data', return_scaling=False, normalize_coeffs=False, rotate=False):
    """Compute Chebyshev expansion coefficients."""
    if type == 'data':
        if rotate:
            f = f[:, [1, 0]] * [1, -1] # rotate 90 degrees clockwise
        f_sorted = f[np.argsort(f[:, 0])]
        f_scaled, scale_factor, x_offset, y_offset = scale_shape(f_sorted)
    else:
        f_scaled = f
        
    coefficients = []
    for i in range(n):
        Tn = cheb_basis(i)
        numerator = inner_product(f_scaled, Tn, omega, type=type)
        denominator = chebyshev_inner_product(Tn, Tn)
        coefficients.append(numerator / denominator)

    if normalize_coeffs:
        max_coeff = max(abs(c) for c in coefficients)
        coefficients = [c / max_coeff for c in coefficients]
    if type == 'data' and return_scaling:
        return coefficients, (scale_factor, x_offset, y_offset)
    else:
        return coefficients

def cheb_expansion(coefficients, n):
    """Compute Chebyshev expansion function."""
    return lambda x: sum(coefficients[i] * cheb_basis(i)(x) for i in range(n))

def distance_metric(curve1=None, curve2=None, coeffs1=None, coeffs2=None, which='rmse'):
    """
    Compute distance between two boundary curves. Take note of scaling to domain [-1,1]
    Also note that distances are computed using polar coordinates to match curve domains.
    """
    if curve1 is None:
        cheb_func1 = cheb_expansion(coeffs1, n=len(coeffs1))  # Take note of scaling
        x = np.linspace(-1, 1, num=200)
        y = cheb_func1(x)
        curve1 = np.column_stack([x, y])
    if curve2 is None:
        cheb_func2 = cheb_expansion(coeffs2, n=len(coeffs2))
        x = np.linspace(-1, 1, num=200)
        y = cheb_func2(x)
        curve2 = np.column_stack([x, y])

    if coeffs1 is None:
        coeffs1 = coefficients(curve1, n=5, type='data')
    if coeffs2 is None:
        coeffs2 = coefficients(curve2, n=5, type='data')

    if which in ['rmse', 'l2_sum']:
        
        pCurve1, pCurve2 = parameterize_curve(curve1), parameterize_curve(curve2)
        theta_min, theta_max = -np.pi/2, np.pi/2
        theta_common = np.linspace(theta_min, theta_max, 200)

        idx1 = np.argsort(pCurve1[:, 0])
        r1 = np.interp(theta_common, pCurve1[idx1,0], pCurve1[idx1, 1])

        idx2 = np.argsort(pCurve2[:, 0])
        r2 = np.interp(theta_common, pCurve2[idx2, 0], pCurve2[idx2, 1])

        squared_diff = (r1 - r2)**2
        if which == 'rmse':
            return np.sqrt(np.mean(squared_diff))
        else:
            return np.sum(np.sum(squared_diff))
    
    elif which == 'linf':
        pCurve1, pCurve2 = parameterize_curve(curve1), parameterize_curve(curve2)
        theta_min, theta_max = -np.pi/2, np.pi/2
        theta_common = np.linspace(theta_min, theta_max, 200)

        idx1 = np.argsort(pCurve1[:, 0])
        r1 = np.interp(theta_common, pCurve1[idx1,0], pCurve1[idx1, 1])
        idx2 = np.argsort(pCurve2[:, 0])
        r2 = np.interp(theta_common, pCurve2[idx2, 0], pCurve2[idx2, 1])

        return np.max(np.abs(r1 - r2))
    
    elif which == 'hausdorff':
        return max(directed_hausdorff(curve1, curve2)[0], directed_hausdorff(curve2, curve1)[0])
    else:
        raise ValueError(f"Unknown metric: {which}. Use 'rmse', 'l2_sum', 'linf', or 'hausdorff'")

"""Plots for displaying data over mulitple runs"""
def cases_bar_plot(dict_soft, dict_no_soft, metric_name, output_dir=None, base_dir='CasesFolder', order='auto'):
    """Create grouped bar plot for a single metric"""
    if output_dir is None:
        output_dir = base_dir
    
    all_conditions = set(dict_soft.keys()) | set(dict_no_soft.keys())
    all_conditions = sorted(list(all_conditions))
    
    # Convert to micrometers (or micrometers squared for area)
    if metric_name.lower() == 'area':
        soft_vals = [dict_soft.get(cond, 0) * CONVERSION_FACTOR_UM**2 for cond in all_conditions]
        no_soft_vals = [dict_no_soft.get(cond, 0) * CONVERSION_FACTOR_UM**2 for cond in all_conditions]
        ylabel = r'Area ($\mu$m$^2$)'
    elif metric_name.lower() == 'length':
        soft_vals = [dict_soft.get(cond, 0) * CONVERSION_FACTOR_UM for cond in all_conditions]
        no_soft_vals = [dict_no_soft.get(cond, 0) * CONVERSION_FACTOR_UM for cond in all_conditions]
        ylabel = rf'{metric_name} ($\mu$m)'
    elif metric_name.lower() == 'aspect ratio':
        soft_vals = [dict_soft.get(cond, 0) for cond in all_conditions]
        no_soft_vals = [dict_no_soft.get(cond, 0) for cond in all_conditions]
        ylabel = rf'{metric_name}'
    elif metric_name.lower() in ['rmse (ctrl)', 'rmse (c59)']:
        soft_vals = [dict_soft.get(cond, 0) for cond in all_conditions]
        no_soft_vals = [dict_no_soft.get(cond, 0) for cond in all_conditions]
        ylabel = rf'{metric_name} ($\mu$m)'
    
    # Determine ordering
    if order == 'auto':
        sorted_indices = np.argsort(soft_vals)
        all_conditions_sorted = [all_conditions[i] for i in sorted_indices]
        soft_vals_sorted = [soft_vals[i] for i in sorted_indices]
        no_soft_vals_sorted = [no_soft_vals[i] for i in sorted_indices]
    else:
        all_conditions_sorted = [cond for cond in order if cond in all_conditions]
        soft_vals_sorted = [dict_soft.get(cond, 0) * (CONVERSION_FACTOR_UM**2 if metric_name.lower() == 'area' else 
                                                       CONVERSION_FACTOR_UM if metric_name.lower() == 'length' else 1) 
                           for cond in all_conditions_sorted]
        no_soft_vals_sorted = [dict_no_soft.get(cond, 0) * (CONVERSION_FACTOR_UM**2 if metric_name.lower() == 'area' else 
                                                             CONVERSION_FACTOR_UM if metric_name.lower() == 'length' else 1) 
                              for cond in all_conditions_sorted]
    
    morphometrics = {
        'SOFT': soft_vals_sorted,
        'NO_SOFT': no_soft_vals_sorted,
    }
    
    # Save the data to CSV
    df = pd.DataFrame({
        'Condition': all_conditions_sorted,
        'SOFT': soft_vals_sorted,
        'NO_SOFT': no_soft_vals_sorted
    })
    csv_path = os.path.join(output_dir, f'{metric_name.lower()}_data.csv')
    df.to_csv(csv_path, index=False)
    print(f"{metric_name} data saved to: {csv_path}")
    
    x = np.arange(len(all_conditions_sorted))
    width = 0.35
    multiplier = 0
    
    fig, ax = plt.subplots(figsize=(14, 6))
    colors = ['pink', 'mediumvioletred']
    
    for attribute, measurements in morphometrics.items():
        if attribute == 'SOFT':
            label = 'Local Softening'
        if attribute == 'NO_SOFT':
            label = 'No Local Softening'
        offset = width * multiplier
        rects = ax.bar(x + offset, measurements, width, label=label, color=colors[multiplier], edgecolor='black', linewidth=1)
        multiplier += 1
    ax.set_ylabel(ylabel)
    ax.set_title(f'{metric_name}')
    ax.set_xticks(x + width/2, all_conditions_sorted)
    ax.legend(loc='upper left', ncols=2)

    lines = plt.gca().get_lines()
    horizontal_lines = [line for line in lines]
    label_x_pos = len(all_conditions_sorted) - 1.5
    xvals = [label_x_pos] * len(horizontal_lines)
    labelLines(horizontal_lines, align=False, xvals=xvals, color='black')

    # center all label texts
    ax = plt.gca()
    fig = ax.figure
    renderer = fig.canvas.get_renderer()

    for text in ax.texts:
        bbox = text.get_window_extent(renderer=renderer)
        # convert bbox width (pixels) to data units
        width_data = bbox.width / fig.dpi * (ax.get_xlim()[1] - ax.get_xlim()[0]) / fig.get_size_inches()[0]
        text.set_x(text.get_position()[0] - width_data / 2)
    
    plt.xticks(rotation=45, ha='right')
    if metric_name == 'Area':
        ax.set_ylim(bottom=0, top=150000)
    elif metric_name == 'Length':
        ax.set_ylim(bottom=0, top=400)
    elif metric_name == 'Aspect Ratio':
        ax.set_ylim(bottom=0, top=1.5)
    elif metric_name in ['RMSE (CTRL)', 'RMSE (C59)']:
        ax.set_ylim(bottom=0, top=250)
    
    # Save
    output_path = os.path.join(output_dir, f'{metric_name.lower()}_comparison.pdf')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"{metric_name} bar plot saved to: {output_path}")
    
    plt.close(fig)
    return fig, ax, all_conditions_sorted


def load_cell_count_data(case_dir):
    """Load cell count time series from a case directory."""
    try:
        data_dict_path = os.path.join(case_dir, 'data_dict.pkl')
        if os.path.exists(data_dict_path):
            with open(data_dict_path, 'rb') as f:
                data_dict = pickle.load(f)
            times = data_dict['times']
            positions = data_dict['positions']
            cell_counts = [np.sum(~np.isnan(pos[:, 0])) for pos in positions]
            return times, cell_counts
    except Exception as e:
        print(f"Error loading {case_dir}: {e}")
    return None, None

def compute_ode_solution(times, N0):
    """Compute ODE solution for cell cycle dynamics."""
    k1 = 1.0 / G_LENGTH
    k2 = 1.0 / M_LENGTH
    kdeath = KDEATH
    
    G_arr = np.zeros_like(times, dtype=float)
    M_arr = np.zeros_like(times, dtype=float)
    
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
    
    return G_arr + M_arr

def plot_all_cell_counts(base_dir='data/output/Cases2025-11-12'):
    """Plot cell counts over time for all cases."""
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    gradient_cases = {'LINEAR', 'ZONE'}
    plotted_gradient = False
    plotted_other = False
    ode_plotted = False
    max_time = 7.0
    
    for soft_type in ['SOFT', 'NO_SOFT']:
        soft_dir = os.path.join(base_dir, soft_type)
        if not os.path.exists(soft_dir):
            continue
            
        for case_name in os.listdir(soft_dir):
            case_path = os.path.join(soft_dir, case_name)
            if not os.path.isdir(case_path):
                continue
            
            times, cell_counts = load_cell_count_data(case_path)
            if times is None or cell_counts is None:
                continue
            
            max_time = max(max_time, times[-1])
            is_gradient = case_name in gradient_cases
            
            if is_gradient:
                color = 'red'
                alpha = 0.6
                lw = 1.5
                label = 'Gradient cases' if not plotted_gradient else None
                plotted_gradient = True
            else:
                color = 'lightblue'
                alpha = 0.5
                lw = 1
                label = 'Other cases' if not plotted_other else None
                plotted_other = True
            
            ax.plot(times, cell_counts, color=color, alpha=alpha, lw=lw, label=label)
            
            if not ode_plotted:
                N0 = cell_counts[0]
                N_ode = compute_ode_solution(times, N0)
                ax.plot(times, N_ode, '--', color='black', lw=2.5, 
                       label='ODE solution', zorder=100)
                ode_plotted = True
    
    ax.set_xlabel('Time (days)', fontsize=14)
    ax.set_ylabel('Number of Cells', fontsize=14)
    ax.set_title('Cell Count vs Time: All Cases', fontsize=16, fontweight='bold')
    ax.legend(fontsize=12, loc='upper left')
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, max_time)
    ax.set_ylim(0,5000)
    plt.tight_layout()
    
    output_path = os.path.join(base_dir, 'all_cell_counts_comparison.pdf')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Cell count plot saved to: {output_path}")
    
    plt.close(fig)
    return fig, ax
# def metric_beeswarm(df, metric, OUTPUT_DIR=None):
#     '''makes a beeswarm plot for a metric for all cases'''
#     df = df.loc[df['metric'] == metric]
#     sns.swarmplot(data=df, x='case', y='value')
#     sns.pointplot(data=df, x='case', y='value', estimator=np.mean,
#                   join=False, color='red', markers='_', markersize=15)
#     if metric == 'aspect_ratio':
#         plt.ylim([1.0,1.5])
#     if metric == 'a':
#         plt.ylim([0.0, 1.1])
#     plt.title(metric)
#     out_path = f"{OUTPUT_DIR}/cases_{metric}.pdf" if OUTPUT_DIR is not None else f"cases_{metric}.pdf"
#     plt.savefig(out_path)
#     plt.close()
#     plt.show()