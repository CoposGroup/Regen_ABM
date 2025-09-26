import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import time
from datetime import datetime
from scipy.optimize import differential_evolution
import toy_model
import config
from config import *
from utils.post_process import plot_optimization_history, print_optimization_step

A_cell = np.pi * (DL_CRIT/2)**2
N0 = 73
phi = np.pi / (2*np.sqrt(3))  # hexagon packing factor
A_t = lambda t: phi * A_cell * N0 * np.exp((KDIV-KDEATH)*(t-T_DORMANT))
r_t = lambda t: np.sqrt(A_t(t)/np.pi)

# Global tracking variables
iteration_count = 0
param_history = []
error_history = []
best_error = float('inf')
best_params = None
optimization_output_dir = None

def objective_function(k_array):
    """Objective function that minimizes difference between simulation and analytical prediction"""
    global iteration_count, param_history, error_history, best_error, best_params
    
    # Extract KDIV value (differential_evolution passes arrays)
    kdiv_value = float(k_array[0]) if hasattr(k_array, '__len__') else float(k_array)
    
    # Set the parameter in config
    config.KDIV = kdiv_value
    
    # Create unique output directory for this simulation
    sim_output_dir = os.path.join(optimization_output_dir, f'sim_{iteration_count+1:03d}_KDIV_{kdiv_value:.4f}')
    os.makedirs(sim_output_dir, exist_ok=True)
    
    # Configure simulation
    config.VIDEO_FLAG = False
    config.PROFILING_FLAG = False
    config.OUTPUT_DIR = sim_output_dir
    
    print(f"Running simulation {iteration_count+1} with KDIV={kdiv_value:.4f}")
    
    # Run simulation
    data_dict, metrics_time_series, metrics_dict, readable_time = toy_model.run_simulation(case='RM_RD')
    
    if 'time' not in metrics_time_series or 'a' not in metrics_time_series:
        print("Warning: Missing required metrics data")
        return 1e6  # Large penalty for failed simulations
    
    # Get simulation data
    sim_times = np.array(metrics_time_series['time'])
    sim_radius = np.array(metrics_time_series['a'])
    
    # Calculate analytical prediction
    analytical_radius = r_t(sim_times)
    
    # Compute RMSE
    error = np.sqrt(np.mean((sim_radius - analytical_radius) ** 2))
    
    # Update tracking
    iteration_count += 1
    param_history.append([kdiv_value])
    error_history.append(error)
    
    # Update best parameters
    if error < best_error:
        best_error = error
        best_params = [kdiv_value]
    
    # Print progress
    print_optimization_step(iteration_count, [kdiv_value], error, best_params, best_error, ['KDIV'])
    
    return error

def run_kdiv_optimization():
    """Run KDIV optimization with visualization"""
    global param_history, error_history, best_params, best_error, optimization_output_dir
    
    # Create output directory
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    optimization_output_dir = os.path.join(OUTPUT_DIR, f'kdiv_optimization_{timestamp}')
    os.makedirs(optimization_output_dir, exist_ok=True)
    
    print("=== Starting KDIV Optimization ===")
    print(f"Target: Fit radius evolution to analytical prediction")
    print(f"Parameter: KDIV")
    print(f"Bounds: [0.0, 10.0]")
    print(f"Output directory: {optimization_output_dir}")
    
    # Run optimization
    t0 = time.time()
    result = differential_evolution(
        objective_function,
        bounds=[(0.0, 5.0)],
        maxiter=1,
        popsize=4,
        tol=1e-3,
        atol=1e-4,
        polish=False,
        # seed=42
    )
    
    elapsed = time.time() - t0
    
    # Print final results
    print("\n" + "="*50)
    print("OPTIMIZATION COMPLETE")
    print("="*50)
    print(f"Best KDIV: {result.x[0]:.6f}")
    print(f"Best error (RMSE): {result.fun:.6f}")
    print(f"Total evaluations: {result.nfev}")
    print(f"Success: {result.success}")
    print(f"Total time: {elapsed:.1f} seconds")
    
    # Generate plots
    print(f"\nGenerating optimization plots...")
    plot_optimization_history(param_history, error_history, optimization_output_dir, ['KDIV'])
    
    # Create comparison plot
    create_final_comparison_plot(result.x[0], optimization_output_dir)
    
    print(f"\nOptimization complete!")
    print(f"Results saved to: {optimization_output_dir}")
    
    return result

def create_final_comparison_plot(best_kdiv, output_dir):
    """Create a comparison plot showing simulation vs analytical prediction"""
    # Run final simulation with best parameters
    config.KDIV = best_kdiv
    config.OUTPUT_DIR = os.path.join(output_dir, 'final_comparison')
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)
    
    data_dict, metrics_time_series, metrics_dict, readable_time = toy_model.run_simulation(case='RM_RD')
    
    if 'time' in metrics_time_series and 'a' in metrics_time_series:
        sim_times = np.array(metrics_time_series['time'])
        sim_radius = np.array(metrics_time_series['a'])
        analytical_radius = r_t(sim_times)
        
        # Create comparison plot
        fig, ax = plt.subplots(figsize=(10, 6))
        
        ax.plot(sim_times, sim_radius, 'b-', linewidth=2, label='Simulation')
        ax.plot(sim_times, analytical_radius, 'r--', linewidth=2, label='Analytical')
        
        ax.set_xlabel('Time')
        ax.set_ylabel('Radius')
        ax.set_title(f'Radius Evolution Comparison (KDIV = {best_kdiv:.4f})')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Add error text
        rmse = np.sqrt(np.mean((sim_radius - analytical_radius) ** 2))
        ax.text(0.02, 0.98, f'RMSE: {rmse:.4f}', transform=ax.transAxes, 
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'final_comparison.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Final comparison plot saved")

if __name__ == "__main__":
    result = run_kdiv_optimization()