import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import glob
from config import DL_CRIT, KDIV, KDEATH, T_DORMANT

def load_multiple_runs(base_path='data/output/RM_RD'):
    """Load metrics from multiple simulation runs"""
    trial_dirs = glob.glob(os.path.join(base_path, 'trial_*'))
    trial_dirs.sort()  # Sort for consistent ordering
    
    all_data = {}
    successful_trials = []
    
    for trial_dir in trial_dirs:
        trial_name = os.path.basename(trial_dir)
        metrics_file = os.path.join(trial_dir, 'metrics_time_series.csv')
        
        if os.path.exists(metrics_file):
            try:
                df = pd.read_csv(metrics_file)
                if 'time' in df.columns and 'a' in df.columns and 'area' in df.columns:
                    all_data[trial_name] = df
                    successful_trials.append(trial_name)
                    print(f"Loaded {trial_name}: {len(df)} time points")
            except Exception as e:
                print(f"Error loading {trial_name}: {e}")
    
    print(f"Successfully loaded {len(successful_trials)} trials: {successful_trials}")
    return all_data

def plot_multiple_runs_comparison():
    """Plot multiple simulation runs against analytical solution"""
    # Load multiple runs
    all_data = load_multiple_runs()
    
    if not all_data:
        print("No trial data found. Using single trial fallback.")
        single_file = 'data/output/RM_RD/trial_001/metrics_time_series.csv'
        if os.path.exists(single_file):
            df = pd.read_csv(single_file)
            all_data = {'trial_001': df}
        else:
            print("No data available for plotting!")
            return
    
    # Define analytical solutions!
    A_cell = np.pi * (DL_CRIT/2)**2
    N0 = 73
    phi = 0.906  # packing factor
    A_t = lambda t: (1/phi) * A_cell * N0 * np.exp((KDIV-KDEATH)*(t-T_DORMANT))
    r_t = lambda t: np.sqrt(A_t(t)/np.pi)

    
    
    # Colors for different trials
    colors = plt.cm.Set1(np.linspace(0, 1, len(all_data)))
    
    # Track data for statistics
    all_radius_errors = []
    all_area_errors = []
    
    # Plot 1: Radius Evolution
    plt.figure(figsize=(12, 8))
    for i, (trial_name, df) in enumerate(all_data.items()):
        plt.plot(df['time'], df['a'], color=colors[i], alpha=0.7, linewidth=1.5, 
                label=f'{trial_name} (ABM)')
        
        # Calculate RMSE for radius
        analytical_r = r_t(df['time'])
        rmse_r = np.sqrt(np.mean((df['a'] - analytical_r)**2))
        all_radius_errors.append(rmse_r)
    
    # Plot analytical solution for radius
    if all_data:
        sample_df = next(iter(all_data.values()))
        time_points = sample_df['time']
        plt.plot(time_points, r_t(time_points), 'k--', linewidth=3, label='Analytical')
    
    plt.xlabel('Time')
    plt.ylabel('Radius')
    plt.title(f'Radius Evolution: {len(all_data)} Simulation Runs vs Analytical')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
    
    # Plot 2: Area Evolution
    plt.figure(figsize=(12, 8))
    for i, (trial_name, df) in enumerate(all_data.items()):
        plt.plot(df['time'], df['area'], color=colors[i], alpha=0.7, linewidth=1.5,
                label=f'{trial_name} (ABM)')
        
        # Calculate RMSE for area
        analytical_a = A_t(df['time'])
        rmse_a = np.sqrt(np.mean((df['area'] - analytical_a)**2))
        all_area_errors.append(rmse_a)
    
    # Plot analytical solution for area
    if all_data:
        plt.plot(time_points, A_t(time_points), 'k--', linewidth=3, label='Analytical')
    
    plt.xlabel('Time')
    plt.ylabel('Area')
    plt.title(f'Area Evolution: {len(all_data)} Simulation Runs vs Analytical')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
    
    # Print summary statistics
    print("\n" + "="*50)
    print("COMPARISON SUMMARY")
    print("="*50)
    print(f"Number of simulation runs: {len(all_data)}")
    print(f"Current KDIV value: {KDIV}")
    print(f"Current KDEATH value: {KDEATH}")
    print(f"T_DORMANT: {T_DORMANT}")
    
    if all_radius_errors:
        print(f"\nRadius RMSE Statistics:")
        print(f"  Mean: {np.mean(all_radius_errors):.6f}")
        print(f"  Std:  {np.std(all_radius_errors):.6f}")
        print(f"  Min:  {np.min(all_radius_errors):.6f}")
        print(f"  Max:  {np.max(all_radius_errors):.6f}")
    
    if all_area_errors:
        print(f"\nArea RMSE Statistics:")
        print(f"  Mean: {np.mean(all_area_errors):.6f}")
        print(f"  Std:  {np.std(all_area_errors):.6f}")
        print(f"  Min:  {np.min(all_area_errors):.6f}")
        print(f"  Max:  {np.max(all_area_errors):.6f}")

if __name__ == "__main__":
    plot_multiple_runs_comparison()