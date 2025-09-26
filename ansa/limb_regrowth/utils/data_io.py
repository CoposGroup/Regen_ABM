# File: utils/data_io.py
"""
Data I/O utilities for simulation
"""
import numpy as np
import os
from utils.post_process import morphometrics, boundary_plot
from config import FRAME_SKIP, BONE_ENABLED  # <-- Remove this import


def save_files(data_dict, config_params, pos0, pos, Xe0, Xe, x_cut, n_daughter, N0, n_cells_save=10, OUTPUT_DIR=None, FRAME_SKIP=1000):
    """Save all simulation data to CSV files and generate analysis plots."""
    if OUTPUT_DIR is None:
        raise ValueError("OUTPUT_DIR must be provided to save_files.")
    Xb = np.loadtxt('data/input/bone.csv', delimiter=',')
    # Create output directory if it doesn't exist
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    

    # Extract data
    positions = data_dict['positions']
    boundaries = data_dict['boundaries']
    forces = data_dict['forces']
    velocities = data_dict['velocities']
    divisions = data_dict['divisions']
    times = data_dict['times']
    elapsed = data_dict['elapsed']
    deaths = data_dict['deaths']
    
    # Extract additional cell status data
    phase_clocks = data_dict['phase_clocks']
    cycle_phases = data_dict['cycle_phases']
    migrant_cells = data_dict['migrant_cells']
    intercal_cells = data_dict['intercal_cells']
    jammed_cells = data_dict['jammed_cells']
    kb_vals = data_dict['kb_vals']

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
                int(t_idx*FRAME_SKIP),              # Timestep
                int(cell_idx),                      # Cell ID
                positions[t_idx, cell_idx, 0],      # X position
                positions[t_idx, cell_idx, 1],      # Y position
                forces[t_idx, cell_idx, 0],         # X force
                forces[t_idx, cell_idx, 1],         # Y force
                velocities[t_idx, cell_idx, 0],     # X velocity
                velocities[t_idx, cell_idx, 1],     # Y velocity
                int(divisions[t_idx, cell_idx]),     # Division status
                int(deaths[t_idx, cell_idx]),       # Death status
                phase_clocks[t_idx, cell_idx],      # Phase clock
                int(cycle_phases[t_idx, cell_idx]), # Cycle phase (0=G0/G1, 1=S/G2/M, -1=empty)
                int(migrant_cells[t_idx, cell_idx]), # Migrant status
                int(intercal_cells[t_idx, cell_idx]), # Intercalation status
                int(jammed_cells[t_idx, cell_idx])   # Jammed status
            ]
            rows.append(row)
    
    # Convert to numpy array and save
    data_array = np.array(rows)
    header = 'time,timestep,cell_id,x,y,force_x,force_y,velocity_x,velocity_y,division_status,death_status,phase_clock,cycle_phase,migrant_status,intercal_status,jammed_status'
    output_file = os.path.join(OUTPUT_DIR, 'cells.csv')
    np.savetxt(output_file, data_array, delimiter=',', header=header, comments='')

    # --- Save selected_cells.csv with only the first n_cells_save cell IDs ---
    # Convert cell_id column to int for comparison
    cell_ids = data_array[:, 2].astype(int)
    unique_ids = np.unique(cell_ids)
    selected_ids = unique_ids[:n_cells_save]
    selected_mask = np.isin(cell_ids, selected_ids)
    selected_array = data_array[selected_mask]
    selected_file = os.path.join(OUTPUT_DIR, 'selected_cells.csv')
    np.savetxt(selected_file, selected_array, delimiter=',', header=header, comments='')
    print(f"Selected cell data saved to {selected_file}")


    # Save boundary data in separate files (if boundary exists)
    if Xe is not None:
        # Save full time series boundary data
        boundary_time_series_file = os.path.join(OUTPUT_DIR, 'boundary_time_series.csv')
        boundary_rows = []
        
        for t_idx, t in enumerate(times):
            for b_idx in range(len(boundaries[t_idx])):
                boundary_row = [
                    t,                                  # Time
                    int(t_idx*FRAME_SKIP),              # Timestep
                    b_idx,                              # Boundary point ID
                    boundaries[t_idx, b_idx, 0],        # X position
                    boundaries[t_idx, b_idx, 1],        # Y position
                    kb_vals[b_idx]                      # Boundary stiffness
                ]
                boundary_rows.append(boundary_row)
        
        boundary_array = np.array(boundary_rows)
        boundary_header = 'time,timestep,boundary_id,x,y,kb_val'
        np.savetxt(boundary_time_series_file, boundary_array, delimiter=',', header=boundary_header, comments='')
        print(f"Boundary time series data saved to {boundary_time_series_file}")
        
        # Save initial boundary (t=0)
        boundary_initial_file = os.path.join(OUTPUT_DIR, 'boundary_initial.csv')
        initial_boundary_rows = []
        for b_idx in range(len(boundaries[0])):
            initial_boundary_row = [
                b_idx,                              # Boundary point ID
                boundaries[0, b_idx, 0],            # X position
                boundaries[0, b_idx, 1],            # Y position
                kb_vals[b_idx]                      # Boundary stiffness
            ]
            initial_boundary_rows.append(initial_boundary_row)
        
        initial_boundary_array = np.array(initial_boundary_rows)
        initial_boundary_header = 'boundary_id,x,y,kb_val'
        np.savetxt(boundary_initial_file, initial_boundary_array, delimiter=',', header=initial_boundary_header, comments='')
        print(f"Initial boundary data saved to {boundary_initial_file}")
        
        # Save final boundary (last timestep)
        boundary_final_file = os.path.join(OUTPUT_DIR, 'boundary_final.csv')
        final_boundary_rows = []
        for b_idx in range(len(boundaries[-1])):
            final_boundary_row = [
                b_idx,                              # Boundary point ID
                boundaries[-1, b_idx, 0],           # X position
                boundaries[-1, b_idx, 1],           # Y position
                kb_vals[b_idx]                      # Boundary stiffness
            ]
            final_boundary_rows.append(final_boundary_row)
        
        final_boundary_array = np.array(final_boundary_rows)
        final_boundary_header = 'boundary_id,x,y,kb_val'
        np.savetxt(boundary_final_file, final_boundary_array, delimiter=',', header=final_boundary_header, comments='')
        print(f"Final boundary data saved to {boundary_final_file}\n")

    
    print(f"Cell data saved to {output_file}")
    
    # Save a small metadata file with simulation parameters
    metadata_file = os.path.join(OUTPUT_DIR, 'metadata.txt')
    with open(metadata_file, 'w') as f:
        f.write(f'Time Elapsed: {elapsed}\n\n')

        f.write("Simulation Parameters\n")
        f.write("====================\n\n")
        
        # Write config parameters
        for section, params in config_params.items():
            f.write(f"{section}\n")
            f.write("-" * len(section) + "\n")
            for key, value in params.items():
                f.write(f"{key}: {value}\n")
            f.write("\n")
        
        f.write("Cell Statistics\n")
        f.write("--------------\n")
        f.write(f"Initial cells: {N0}\n")
        active_cells = np.where(~np.isnan(pos[:,0]))[0].size
        f.write(f"Final total cells: {active_cells}\n")
        f.write(f"Total division events: {n_daughter}\n")
        
        # Count initial and final migrant and intercal cells
        # Initial counts (first timestep)
        initial_active_mask = ~np.isnan(positions[0, :, 0])
        initial_migrant_count = np.sum(migrant_cells[0, initial_active_mask])
        initial_intercal_count = np.sum(intercal_cells[0, initial_active_mask])
        
        # Final counts (last timestep)
        final_active_mask = ~np.isnan(positions[-1, :, 0])
        final_migrant_count = np.sum(migrant_cells[-1, final_active_mask])
        final_intercal_count = np.sum(intercal_cells[-1, final_active_mask])
        
        f.write(f"Initial migrant cells: {initial_migrant_count}\n")
        f.write(f"Final migrant cells: {final_migrant_count}\n")
        f.write(f"Initial intercalation cells: {initial_intercal_count}\n")
        f.write(f"Final intercalation cells: {final_intercal_count}\n")
        # f.write(f"Growth region cells: {len(np.where(pos[:,0] > x_cut)[0])}\n\n")
        
        # Generate analysis plots and write morphometrics

        
        # Calculate final morphometrics
        f.write("Morphometrics\n")
        f.write("-----------------------\n")
        if Xe is not None:
            Xe_growth = Xe[Xe[:, 0] > x_cut]
            
            # Calculate metrics
            area, perimeter, aspect_ratio, ellipticity, roundness, a , b, volume_fraction = morphometrics(Xe, pos=pos, x_cut=x_cut)
            boundary_plot(Xe0, Xe, Xe_growth, Xb, x_cut, 
                        aspect_ratio=aspect_ratio, area=area, roundness=roundness,
                        a=a, perimeter=perimeter, ellipticity=ellipticity, 
                        OUTPUT_DIR=OUTPUT_DIR, bone_enabled=BONE_ENABLED,
                        conversion_factor_um=200, show_real_units=True)
        if Xe is None:
            area, perimeter, aspect_ratio, ellipticity, roundness, a , b, volume_fraction = morphometrics(Xe=None, pos=pos)
            Xe_growth = None
            boundary_plot(Xe0, Xe, Xe_growth, Xb, x_cut, 
                        aspect_ratio=aspect_ratio, area=area, roundness=roundness,
                        a=a, perimeter=perimeter, ellipticity=ellipticity, 
                        OUTPUT_DIR=OUTPUT_DIR, pos0=pos0, pos_final=pos, bone_enabled=BONE_ENABLED,
                        conversion_factor_um=200, show_real_units=True)
        # Write metrics to file
        f.write(f"Semi-major axis (length): {a:.2f}\n")
        f.write(f"Semi-minor axis (length): {b:.2f}\n")
        f.write(f"Area: {area:.2f}\n")
        f.write(f"Perimeter: {perimeter:.2f}\n")
        f.write(f"Aspect ratio: {aspect_ratio:.2f}\n")
        f.write(f"Ellipticity: {ellipticity:.2f}\n")
        f.write(f"Roundness: {roundness:.2f}\n")
        f.write(f"Volume Fraction: {volume_fraction:.5f}\n")