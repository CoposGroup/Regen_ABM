# File: utils/data_io.py
"""
Data I/O utilities for simulation
"""
import numpy as np
import os
from utils.post_process import morphometrics, density_heatmap, boundary_plot
from config import OUTPUT_DIR

def save_files(data_dict, config_params, pos0, pos, Xe0, Xe, x_cut, n_daughter, N0, n_cells_save=10, OUTPUT_DIR=OUTPUT_DIR):
    """Save all simulation data to CSV files and generate analysis plots."""
    
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
                int(divisions[t_idx, cell_idx]),     # Division status
                int(deaths[t_idx, cell_idx])        # Death status
            ]
            rows.append(row)
    
    # Convert to numpy array and save
    data_array = np.array(rows)
    header = 'time,timestep,cell_id,x,y,force_x,force_y,velocity_x,velocity_y,division_status'
    output_file = os.path.join(OUTPUT_DIR, 'cells.csv')
    np.savetxt(output_file, data_array, delimiter=',', header=header, comments='')

    # same thing, but for some selected cells
        # Convert to numpy array and save
    data_array = np.array(rows)
    header = 'time,timestep,cell_id,x,y,force_x,force_y,velocity_x,velocity_y,division_status'
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


    # Save boundary data in a separate file (if boundary exists)
    if Xe is not None:
        boundary_file = os.path.join(OUTPUT_DIR, 'boundary.csv')
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
        print(f"Boundary data saved to {boundary_file}\n")

    
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
        # f.write(f"Growth region cells: {len(np.where(pos[:,0] > x_cut)[0])}\n\n")
        
        # Generate analysis plots and write morphometrics
        density_heatmap(config_params['Physical Parameters']['SOFT_RANGE'], 
                       pos=pos, Xe=Xe, x_cut=x_cut, OUTPUT_DIR=OUTPUT_DIR, bin_size=0.1)
        
        # Calculate final morphometrics
        f.write("Morphometrics\n")
        f.write("-----------------------\n")
        if Xe is not None:
            Xe_growth = Xe[Xe[:, 0] > x_cut]
            
            # Calculate metrics
            area, perimeter, aspect_ratio, ellipticity, roundness, a , b, volume_fraction = morphometrics(Xe=Xe_growth, pos=pos)
            boundary_plot(Xe0, Xe, Xe_growth, x_cut, 
                        aspect_ratio=aspect_ratio, area=area, roundness=roundness,
                        a=a, perimeter=perimeter, ellipticity=ellipticity, 
                        OUTPUT_DIR=OUTPUT_DIR)
        if Xe is None:
            area, perimeter, aspect_ratio, ellipticity, roundness, a , b, volume_fraction = morphometrics(Xe=None, pos=pos)
            Xe_growth = None
            boundary_plot(Xe0, Xe, Xe_growth, x_cut, 
                        aspect_ratio=aspect_ratio, area=area, roundness=roundness,
                        a=a, perimeter=perimeter, ellipticity=ellipticity, 
                        OUTPUT_DIR=OUTPUT_DIR, pos0=pos0, pos_final=pos)
        # Write metrics to file
        f.write(f"Semi-major axis (length): {a:.2f}\n")
        f.write(f"Semi-minor axis (length): {b:.2f}\n")
        f.write(f"Area: {area:.2f}\n")
        f.write(f"Perimeter: {perimeter:.2f}\n")
        f.write(f"Aspect ratio: {aspect_ratio:.2f}\n")
        f.write(f"Ellipticity: {ellipticity:.2f}\n")
        f.write(f"Roundness: {roundness:.2f}\n")
        f.write(f"Volume Fraction: {volume_fraction:.5f}\n")