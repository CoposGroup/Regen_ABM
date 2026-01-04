"""Data I/O utilities for simulation"""
import numpy as np
import pickle
import os
import shutil
import inspect
from config import CONVERSION_FACTOR_UM
from utils.post_process import (
    morphometrics, growth_plot, density_heatmap, 
    morphometrics_time_series_plot, cell_type_proportions_plot,
    cycle_plot, phase_distribution_plot, runtime_plot, MSD_plot, simulation_snapshot
)

def save_config_snapshot(output_dir):
    """Save a snapshot of the config.py file used for this simulation."""
    try:
        import config
        config_path = inspect.getsourcefile(config)
        if config_path and os.path.exists(config_path):
            dst_config = os.path.join(output_dir, 'config_snapshot.py')
            shutil.copyfile(config_path, dst_config)
        else:
            print(f"Warning: could not find config.py source file")
    except Exception as e:
        print(f"Warning: could not save config snapshot: {e}")

def save_files(data_dict, config_params, save_data_dict=True, save_figures=True):
    """Save all simulation data to pickle file and generate analysis plots."""

    OUTPUT_DIR = data_dict.get('OUTPUT_DIR')
    if OUTPUT_DIR is None:
        raise ValueError("OUTPUT_DIR must be provided in data_dict.")
    
    FRAME_SKIP = data_dict.get('FRAME_SKIP', 2000)
    pos0 = data_dict.get('pos0')
    Xe0 = data_dict.get('Xe0')
    x_cut = data_dict.get('x_cut', 0.0)
    n_daughter = data_dict.get('n_daughter', 0)
    N0 = data_dict.get('N0')
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Save data_dict pickle file if requested (often very large!)
    if save_data_dict:
        with open(os.path.join(OUTPUT_DIR, 'data_dict.pkl'), 'wb') as file:
            pickle.dump(data_dict, file)

    # Extract data from data_dict
    positions = data_dict['positions']
    boundaries = data_dict['boundaries']
    kb_vals = data_dict['kb_vals']
    n_deaths = data_dict['n_deaths']
    coefficients_growth = data_dict.get('coefficients_growth', None)
    times = data_dict['times']
    elapsed = data_dict['elapsed']
    migrant_cells = data_dict['migrant_cells']
    intercal_cells = data_dict['intercal_cells']

    rmse_ctrl = data_dict['rmse_ctrl']
    rmse_c59 = data_dict['rmse_c59']
    haus_ctrl = data_dict['haus_ctrl']
    haus_c59 = data_dict['haus_c59']
    
    pos = positions[-1]
    Xe = boundaries[-1]

    if config_params is not None:
        config_file = os.path.join(OUTPUT_DIR, 'config.py')
        with open(config_file, 'w') as f:
            f.write('"""\n')
            f.write('Configuration parameters used for this simulation run.\n')
            f.write('These are the actual runtime values that were used.\n')
            f.write('"""\n\n')
            
            for section, params in config_params.items():
                f.write(f"# {section}\n")
                for key, value in params.items():
                    if isinstance(value, str):
                        f.write(f"{key} = '{value}'\n")
                    else:
                        f.write(f"{key} = {value}\n")
                f.write("\n")
    
    # Save metadata file with simulation outputs/results
    metadata_file = os.path.join(OUTPUT_DIR, 'metadata.txt')
    with open(metadata_file, 'w') as f:
        f.write("=" * 60 + "\n")
        f.write("SIMULATION RESULTS SUMMARY\n")
        f.write("=" * 60 + "\n\n")
        
        f.write(f"Time Elapsed (wall clock): {elapsed}\n\n")
        
        f.write("=" * 60 + "\n")
        f.write("CELL STATISTICS\n")
        f.write("=" * 60 + "\n\n")
        
        # Count initial and final cells
        initial_active_mask = ~np.isnan(positions[0, :, 0])
        initial_active_cells = np.sum(initial_active_mask)
        final_active_mask = ~np.isnan(positions[-1, :, 0])
        final_active_cells = np.sum(final_active_mask)
        
        f.write(f"Initial cells: {N0}\n")
        f.write(f"Final cells: {final_active_cells}\n")
        f.write(f"Net change: {final_active_cells - N0:+d}\n\n")
        
        f.write(f"Total division events: {n_daughter}\n")
        f.write(f"Total deaths: {n_deaths}\n")
        f.write(f"Net from divisions/deaths: {n_daughter - n_deaths:+d}\n\n")
        
        # Count initial and final migrant and intercal cells
        initial_migrant_count = np.sum(migrant_cells[0, initial_active_mask])
        initial_intercal_count = np.sum(intercal_cells[0, initial_active_mask])
        final_migrant_count = np.sum(migrant_cells[-1, final_active_mask])
        final_intercal_count = np.sum(intercal_cells[-1, final_active_mask])
        
        f.write(f"Initial migrant cells: {int(initial_migrant_count)}\n")
        f.write(f"Final migrant cells: {int(final_migrant_count)}\n")
        f.write(f"Initial intercalation cells: {int(initial_intercal_count)}\n")
        f.write(f"Final intercalation cells: {int(final_intercal_count)}\n\n")
        
        f.write("=" * 60 + "\n")
        f.write("MORPHOMETRICS\n")
        f.write("=" * 60 + "\n\n")
        
        # Calculate final morphometrics
        if Xe is not None:
            area_growth_region, perimeter, AR_whole_limb, AR_outgrowth, ellipticity, roundness, a, b, volume_fraction = morphometrics(Xe, pos=pos, x_cut=x_cut)

        # Write metrics to file
        f.write(f"Semi-major axis (length): {a * CONVERSION_FACTOR_UM:.2f} um\n")
        f.write(f"Semi-minor axis (width): {b * CONVERSION_FACTOR_UM:.2f} um\n")
        f.write(f"Area: {area_growth_region*CONVERSION_FACTOR_UM**2:.2f} um^2\n")
        f.write(f"Perimeter: {perimeter*CONVERSION_FACTOR_UM:.2f} um\n")
        f.write(f"Aspect ratio (Growth Region): {AR_outgrowth:.2f}\n")
        f.write(f"Aspect ratio (Whole Limb): {AR_whole_limb:.2f}\n")
        f.write(f"Ellipticity: {ellipticity:.2f}\n")
        f.write(f"Roundness: {roundness:.2f}\n")
        f.write(f"Volume fraction: {volume_fraction:.5f}\n\n")

        f.write(f"RMSE (CTRL): {rmse_ctrl:.5f} um\n\n")
        f.write(f"RMSE (C59): {rmse_c59:.5f} um\n\n")
        f.write(f"Hausdorff (CTRL): {haus_ctrl:.5f} um\n\n")
        f.write(f"Hausdorff (C59): {haus_c59:.5f} um\n\n")

        # Add Chebyshev coefficients if available
        if coefficients_growth is not None:
            f.write("=" * 60 + "\n")
            f.write("CHEBYSHEV COEFFICIENTS\n")
            f.write("=" * 60 + "\n\n")
            f.write(f"Number of coefficients: {len(coefficients_growth)}\n")
            coeffs_formatted = [f"{c:.2f}" for c in coefficients_growth[:5]]
            f.write(f"First 5 coefficients: [{', '.join(coeffs_formatted)}]\n\n")
        
        f.write("=" * 60 + "\n")
        f.write(f"Data saved to: {OUTPUT_DIR}\n")
        f.write("=" * 60 + "\n")

    if save_figures:
        
        times = data_dict['times']
        Gphase = data_dict.get('Gphase', [])
        Mphase = data_dict.get('Mphase', [])
        cell_count = data_dict.get('cell_count', [])
        step_times = data_dict.get('step_times', [])
        Xb = data_dict.get('Xb', None)
        morphometrics_data = data_dict.get('morphometrics_time_series', {})
        T_DORMANT = data_dict.get('T_DORMANT', 0.0)
        TMAX = config_params.get('TMAX', 7.0)
        
        density_heatmap(kb_vals, pos=pos, Xe=Xe, x_cut=x_cut, OUTPUT_DIR=OUTPUT_DIR, 
                       bin_size=0.1, shading='gouraud', fig_mode=False, show_colorbar=True)
        density_heatmap(kb_vals, pos=pos, Xe=Xe, x_cut=x_cut, OUTPUT_DIR=OUTPUT_DIR, 
                       bin_size=0.1, shading='auto', fig_mode=False, show_colorbar=True)
        
        # Volume fraction time series
        if morphometrics_data:
            morphometrics_time_series_plot(morphometrics_data, times=times, OUTPUT_DIR=OUTPUT_DIR, metric='volume_fraction', title='Volume Fraction')
        
        cell_type_proportions_plot(
            times=times,
            positions_history=positions,
            migrant_history=migrant_cells,
            intercal_history=intercal_cells,
            jammed_history=data_dict.get('jammed_cells', np.zeros_like(positions)),
            OUTPUT_DIR=OUTPUT_DIR
        )
        
        if len(cell_count) > 0:
            cycle_plot(times, cell_count, N0, OUTPUT_DIR=OUTPUT_DIR)
        if len(Gphase) > 0 and len(Mphase) > 0:
            phase_distribution_plot(times, Gphase, Mphase, fit=False, OUTPUT_DIR=OUTPUT_DIR)
        if Xe is not None:
            growth_plot(Xe0, Xe, title=f'{TMAX} dpa', OUTPUT_DIR=OUTPUT_DIR)
        if TMAX > T_DORMANT and len(step_times) > 0:
            runtime_plot(cell_count, step_times, OUTPUT_DIR=OUTPUT_DIR)
        _, _, slope, D = MSD_plot(positions, OUTPUT_DIR=OUTPUT_DIR)