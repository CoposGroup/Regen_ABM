import numpy as np
import sys
import importlib
import os
import time
from scipy.optimize import differential_evolution, minimize
from signed_distance import make_sdf_polygon, comp_function
from config import XMIN, XMAX, YMIN, YMAX, OUTPUT_DIR
from utils.post_process import (print_optimization_step, plot_optimization_history, plot_shape_comparison,
                                plot_sdf_evolution, create_optimization_summary, plot_profile_likelihood)
# Global variables to track optimization progress
iteration_count = 0
param_history = []
error_history = []
boundary_history = []
best_error = float('inf')
best_params = None
target_shape_global = None
sim_boundary_global = None
optimization_output_dir = None
param_names = []  # Names of parameters being optimized



def parallel_objective_function(args):
    """Standalone objective function for parallel processing"""
    params, param_names_list, opt_output_dir = args
    
    # Create parameter dictionary and directory name
    param_dict = dict(zip(param_names_list, params))
    param_str = '_'.join([f'{name}_{val:.4f}' for name, val in param_dict.items()])
    
    # Create unique output directory for this simulation run
    import time
    timestamp = int(time.time() * 1000000)  # microsecond timestamp for uniqueness
    sim_output_dir = os.path.join(opt_output_dir, f'sim_{timestamp}_{param_str}')
    os.makedirs(sim_output_dir, exist_ok=True)
    
    # Need to reload modules to get updated config values
    import config
    for name, value in param_dict.items():
        setattr(config, name, float(value))
    config.VIDEO_FLAG = False#False
    config.PROFILING_FLAG = False
    config.OUTPUT_DIR = sim_output_dir  # Use unique directory for this simulation
    
    # Force reload of toy_model to pick up new config values
    import toy_model
    importlib.reload(toy_model)
    
    param_str_print = ', '.join([f'{name}={val:.4f}' for name, val in param_dict.items()])
    print(f"Running simulation with {param_str_print}")
    print(f"Output directory: {sim_output_dir}")
    data_dict, metrics_dict, readable_time = toy_model.run_simulation(case='RM_RD')

    r_target = 1.0
    theta = np.linspace(0, 2*np.pi, 100)
    target_shape = np.column_stack((r_target*np.cos(theta), r_target*np.sin(theta)))

    phi_target = make_sdf_polygon(target_shape)
    
    # Get final positions and create boundary from convex hull
    positions = data_dict['positions']
    final_pos = positions[-1]  # Last timestep
    pos_valid = final_pos[~np.isnan(final_pos).any(axis=1)]  # Remove NaN positions
    
    from scipy.spatial import ConvexHull
    hull = ConvexHull(pos_valid)
    verts = hull.vertices
    verts = np.append(verts, verts[0])  # Close the hull
    boundary_points = pos_valid[verts]
    
    phi_sim = make_sdf_polygon(boundary_points)
    error = comp_function(phi_target, phi_sim).integral

    return error

class ParallelObjectiveWrapper:
    """Wrapper class to make parallel objective function work with differential_evolution"""
    def __init__(self, param_names_list, opt_output_dir):
        self.param_names_list = param_names_list
        self.opt_output_dir = opt_output_dir
    
    def __call__(self, params):
        return parallel_objective_function((params, self.param_names_list, self.opt_output_dir))

def objective_function(params):
    """Legacy wrapper for plotting"""
    global param_names, optimization_output_dir, iteration_count, param_history, error_history, boundary_history
    global best_error, best_params, target_shape_global, sim_boundary_global
    
    # Create parameter dictionary and directory name
    param_dict = dict(zip(param_names, params))
    param_str = '_'.join([f'{name}_{val:.4f}' for name, val in param_dict.items()])
    
    # Create unique output directory for this simulation run
    sim_output_dir = os.path.join(optimization_output_dir, f'sim_{iteration_count+1:03d}_{param_str}')
    os.makedirs(sim_output_dir, exist_ok=True)
    
    # Need to reload modules to get updated config values
    import config
    for name, value in param_dict.items():
        setattr(config, name, float(value))
    config.VIDEO_FLAG = False
    config.PROFILING_FLAG = False
    config.OUTPUT_DIR = sim_output_dir  # Use unique directory for this simulation
    
    # Force reload of toy_model to pick up new config values
    import toy_model
    importlib.reload(toy_model)
    
    param_str_print = ', '.join([f'{name}={val:.4f}' for name, val in param_dict.items()])
    print(f"Running simulation with {param_str_print}")
    print(f"Output directory: {sim_output_dir}")
    data_dict, metrics_dict, readable_time = toy_model.run_simulation(case='RM_RD')

    r_target = 1.0
    theta = np.linspace(0, 2*np.pi, 100)
    target_shape = np.column_stack((r_target*np.cos(theta), r_target*np.sin(theta)))

    phi_target = make_sdf_polygon(target_shape)
    
    # Get final positions and create boundary from convex hull
    positions = data_dict['positions']
    final_pos = positions[-1]  # Last timestep
    pos_valid = final_pos[~np.isnan(final_pos).any(axis=1)]  # Remove NaN positions
    
    from scipy.spatial import ConvexHull
    hull = ConvexHull(pos_valid)
    verts = hull.vertices
    verts = np.append(verts, verts[0])  # Close the hull
    boundary_points = pos_valid[verts]
    
    phi_sim = make_sdf_polygon(boundary_points)
    error = comp_function(phi_target, phi_sim).integral

    # Track optimization progress
    iteration_count += 1
    param_history.append(list(params))
    error_history.append(error)
    boundary_history.append(boundary_points.copy())
    
    # Update best parameters
    if error < best_error:
        best_error = error
        best_params = list(params)
        target_shape_global = target_shape.copy()
        sim_boundary_global = boundary_points.copy()
    
    # Print progress
    print_optimization_step(iteration_count, params, error, best_params, best_error, param_names)

    return error

def evaluate_grid_point(args):
    """Evaluate a single grid point for profile likelihood analysis"""
    j, fixed_value, param_names, initial_guess_nuisance, param_bounds, i, optimization_output_dir = args
    
    nuisance_bounds = [param_bounds[k] for k in range(len(param_names)) if k != i]
    initial_guess = initial_guess_nuisance
    
    try:
        # Create self-contained objective for parallelization
        def profile_nuisance_objective(nuisance_params):
            # Reconstruct full parameter vector
            full_params = []
            nuisance_idx = 0
            for k in range(len(param_names)):
                if k == i:
                    full_params.append(fixed_value)  # Fixed parameter
                else:
                    full_params.append(nuisance_params[nuisance_idx])  # Nuisance parameter
                    nuisance_idx += 1
            
            # Use self-contained objective function
            return parallel_objective_function((full_params, param_names, optimization_output_dir))
        
        result = minimize(
            profile_nuisance_objective,
            initial_guess,
            bounds=nuisance_bounds,
            method='L-BFGS-B',
            options = {
                'maxiter': 100,  # More generous than 50
                'gtol': 1e-6,     # Matches MATLAB TolFun
                'ftol': 1e-6      # Matches MATLAB TolX
            }
        )
        
        error = result.fun
        optimized_nuisance = result.x
        
        # Reconstruct full optimized parameter vector
        full_optimized = []
        nuisance_idx = 0
        for k in range(len(param_names)):
            if k == i:
                full_optimized.append(fixed_value)
            else:
                full_optimized.append(optimized_nuisance[nuisance_idx])
                nuisance_idx += 1
        
        return j, error, full_optimized, None
        
    except Exception as e:
        return j, np.inf, [np.nan] * len(param_names), str(e)


def profile_likelihood(param_names, best_params, param_bounds, optimization_output_dir, nr=50, workers=1):
    """Profile likelihood analysis for each parameter with optional parallelization"""
    import multiprocessing
    
    print(f"\n=== Profile Likelihood Analysis ===")
    print(f"Computing likelihood profiles for {len(param_names)} parameters...")
    print(f"Grid points per parameter: {nr}")
    print(f"Workers: {workers}")
    
    profile_results = {}
    
    for i, param_name in enumerate(param_names):
        print(f"\nProfiling parameter: {param_name}")
        
        # Get the best value for this parameter
        param_star = best_params[i]
        
        param_min = param_star/10
        param_max = param_star + 9*param_star/10
        param_grid = np.linspace(param_min, param_max, nr)
        
        # Store results for this parameter
        errors = []
        optimized_params = []
        
        # Initial guess for nuisance parameters (will be updated as we go)
        current_initial_guess = [best_params[k] for k in range(len(param_names)) if k != i]
        
        if workers != 1:
            # Parallel processing
            # Convert workers=-1 to actual CPU count
            actual_workers = multiprocessing.cpu_count() if workers == -1 else workers
            print(f"  Using parallel processing with {actual_workers} workers...")
            
            # For parallel processing, we can't easily update initial guess between points
            # so we use the original best_params as initial guess for all points
            grid_args = [(j, fixed_value, param_names, current_initial_guess, param_bounds, i, optimization_output_dir) 
                        for j, fixed_value in enumerate(param_grid)]
            with multiprocessing.Pool(processes=actual_workers) as pool:
                results = pool.map(evaluate_grid_point, grid_args)
            
            # Process results in order
            for j, error, full_optimized, error_msg in results:
                errors.append(error)
                optimized_params.append(full_optimized)
                if error_msg:
                    print(f"    Warning: Optimization failed at {param_name}={param_grid[j]:.4f}: {error_msg}")
        else:
            # Serial processing with adaptive initial guess (following MATLAB approach)
            for j, fixed_value in enumerate(param_grid):
                print(f"  Grid point {j+1}/{nr}: {param_name}={fixed_value:.4f}")
                j, error, full_optimized, error_msg = evaluate_grid_point(
                    (j, fixed_value, param_names, current_initial_guess, param_bounds, i, optimization_output_dir))
                errors.append(error)
                optimized_params.append(full_optimized)
                if error_msg:
                    print(f"    Warning: Optimization failed at {param_name}={fixed_value:.4f}: {error_msg}")
                else:
                    # Update initial guess for next iteration using optimized nuisance parameters
                    # This follows MATLAB: initial_guess = b1b2_new (line 93)
                    current_initial_guess = [full_optimized[k] for k in range(len(param_names)) if k != i]
        
        # Store results for this parameter (plot_profile_likelihood will compute likelihoods)
        profile_results[param_name] = {
            'param_star': param_star,
            'param_grid': param_grid,
            'errors': np.array(errors),
            'optimized_params': np.array(optimized_params)
        }
        
        print(f"  Completed profile for {param_name}")
    
    # Save results
    import pickle
    profile_file = os.path.join(optimization_output_dir, 'profile_likelihood_results.pkl')
    with open(profile_file, 'wb') as f:
        pickle.dump(profile_results, f)
    
    print(f"\nProfile likelihood results saved to: {profile_file}")
    return profile_results


def run_optimization(parameter_config={'names': ['KDIV'], 'bounds': [(0.0, 5.0)], 'description': 'KDIV-only optimization'}, 
                     workers=-1):
    """
    Run optimization with flexible parameter configuration.
    
    Args:
        parameter_config: Dict with format:
            {
                'names': ['KDIV', 'KDEATH', ...],
                'bounds': [(min1, max1), (min2, max2), ...],
                'description': 'Custom parameters'
            }
            If None, defaults to KDIV and KDEATH optimization.
        workers: Number of worker processes (-1 for all cores, 1 for serial, default: -1)
    """
    global param_history, error_history, boundary_history, best_params, best_error
    global target_shape_global, sim_boundary_global, optimization_output_dir, param_names
    

    # Set global parameter names
    param_names = parameter_config['names']
    param_bounds = parameter_config['bounds']
    
    # Validation
    if len(param_names) != len(param_bounds):
        raise ValueError("Number of parameter names must match number of bounds")
    
    # Create main optimization output directory with timestamp
    from datetime import datetime
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    optimization_output_dir = os.path.join(OUTPUT_DIR, f'optimization_{timestamp}')
    os.makedirs(optimization_output_dir, exist_ok=True)
    
    # Reset global tracking variables and choose objective function based on workers
    global iteration_count
    iteration_count = 0
    param_history = []
    error_history = []
    boundary_history = []
    
    # Choose objective function based on parallelization
    if workers != 1:
        # Use wrapper class for parallel processing
        worker_func = ParallelObjectiveWrapper(param_names, optimization_output_dir)
        print("=== Starting Optimization (Parallel) ===")
    else:
        # Use global state tracking function for serial processing
        worker_func = objective_function
        print("=== Starting Optimization (Serial) ===")
    
    print("Target: Circular shape with radius 1.0")
    print(f"Parameters to optimize: {', '.join(param_names)}")
    print(f"Parameter bounds: {dict(zip(param_names, param_bounds))}")
    print(f"Description: {parameter_config['description']}")
    print(f"Workers: {workers}")
    print(f"Main output directory: {optimization_output_dir}")
    
    # Calculate and display expected number of simulations
    maxiter = 25
    popsize = 50
    n_params = len(param_names)
    
    initial_evals = popsize * n_params
    max_generation_evals = maxiter * popsize
    estimated_total = initial_evals + max_generation_evals
    
    print(f"Population size: {popsize}")
    print(f"Maximum iterations: {maxiter}")
    print(f"Parameters: {n_params}")
    
    t0 = time.time()
    result = differential_evolution(
        worker_func,
        param_bounds,
        maxiter=maxiter,
        popsize=popsize,
        tol=1e-1,
        atol=1e-2,
        polish=False,
        workers=workers
    )

    
    # Print final results
    # print("\n" + "="*50)
    print("OPTIMIZATION COMPLETE")
    # print("="*50)
    best_param_str = ', '.join([f'{name}={val:.4f}' for name, val in zip(param_names, result.x)])
    print(f"Best parameters: {best_param_str}")
    print(f"Best error: {result.fun:.6f}")
    print(f"Total evaluations: {result.nfev}")
    print(f"Success: {result.success}")
    print(f"Total time: {time.time() - t0:.1f} seconds")
    
    # For parallel processing, skip plotting (just save results)
    if workers != 1:
        print(f"\nParallel optimization complete. Skipping plots for performance.")
        print(f"Individual simulation data saved in subdirectories of: {optimization_output_dir}")
        print(f"To generate plots, run with workers=1 for serial execution.")
    else:
        # Generate visualizations for serial execution
        print(f"\nGenerating optimization analysis plots...")
        
        plot_optimization_history(param_history, error_history, optimization_output_dir, param_names)
        create_optimization_summary(param_history, error_history, boundary_history, 
                                  target_shape_global, optimization_output_dir, param_names)
        plot_shape_comparison(target_shape_global, sim_boundary_global, 
                            best_error, optimization_output_dir)
    
    # PROFILE LIKELIHOOD (with optional parallelization)
    # You can adjust sigma_method and sigma_factor based on your data characteristics
    # profile_results = profile_likelihood(param_names, result.x, param_bounds, optimization_output_dir, 
    #                                    nr=50, workers=workers)
    # profile_file = os.path.join(optimization_output_dir, 'profile_likelihood_results.pkl')
    # if os.path.exists(profile_file):
    #     print("Profile likelihood plots...")
    #     plot_profile_likelihood(profile_file, optimization_output_dir)
    
    elapsed = time.time() - t0
    print(f"\nAll plots saved to: {optimization_output_dir}")
    print(f"Individual simulation data in subdirectories of: {optimization_output_dir}")
    print(f"Time taken: {time.strftime('%H:%M:%S', time.gmtime(elapsed))}")
    
    return result

if __name__ == "__main__":
    result = run_optimization(
        parameter_config={'names': ['KDIV'], 'bounds': [(0.0, 10.0)], 'description': 'KDIV'}, # ADD KDEATH BACK  and KDEATH optimization  , (0.0, 0.5) ######
        workers=1  # Use all available cores
    )
    

    