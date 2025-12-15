""" Needs Cleanup! """
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize, differential_evolution
import time
import os
import importlib
from utils.curve_comp import distance_metric

from utils.optimization_utils import (print_optimization_step, plot_optimization_history, plot_shape_comparison,
                                animate_error_convergence, animate_simplex_convergence,
                                save_optimization_data, create_individual_optimization_plots)
from config import EXP_OUTGROWTH_LENGTH_C59, EXP_OUTGROWTH_LENGTH, EXP_AREA, DATA_DIR

def objective_function(params):
    """Objective function for optimization"""
    global param_names, optimization_output_dir, iteration_count, param_history, error_history, boundary_history
    global best_error, best_params, sim_boundary_global, coeffs_target, error_metric
    
    # Create parameter dictionary and convert MIGRATION_PERCENT from [0,1] to [0,100]
    param_dict = {}
    for i, name in enumerate(param_names):
        if name == 'MIGRATION_PERCENT':
            clipped_val = float(np.clip(params[i], 0.0, 1.0))
            param_dict[name] = clipped_val * 100.0
        else:
            param_dict[name] = float(params[i])
    
    # Clip other parameters to their bounds
    try:
        bounds_map = dict(zip(param_names, param_bounds))
        for name in param_names:
            if name != 'MIGRATION_PERCENT':
                low, high = bounds_map[name]
                param_dict[name] = float(np.clip(param_dict[name], low, high))
    except Exception:
        pass
    
    # Create unique output directory for this simulation
    param_str = '_'.join([f'{name}_{val:.4f}' for name, val in param_dict.items()])
    sim_output_dir = os.path.join(optimization_output_dir, f'sim_{iteration_count+1:03d}_{param_str}')
    os.makedirs(sim_output_dir, exist_ok=True)
    
    import config
    
    # Handle KDIV parameter: convert to G_LENGTH and M_LENGTH
    if 'KDIV' in param_dict:
        kdiv_val = param_dict['KDIV']
        if kdiv_val > 0:
            total_cycle_length = 1.0 / kdiv_val
            g_length = total_cycle_length / 2.0
            m_length = total_cycle_length / 2.0
            print(f"  KDIV={kdiv_val:.6f} -> G_LENGTH={g_length:.6f}, M_LENGTH={m_length:.6f}")
        else:
            g_length = float('inf')
            m_length = float('inf')
            print(f"  KDIV={kdiv_val:.6f} -> G_LENGTH=inf, M_LENGTH=inf")
        setattr(config, 'G_LENGTH', float(g_length))
        setattr(config, 'M_LENGTH', float(m_length))
    
    # Set all other parameters
    for name, value in param_dict.items():
        if name != 'KDIV':
            setattr(config, name, float(value))
    
    # Set migration flag and output settings
    config.MIGRATION_ENABLED = 'MIGRATION_PERCENT' in param_dict
    save_this_iteration = (iteration_count == 0) or ((iteration_count + 1) % 10 == 0)
    config.VIDEO_FLAG = save_this_iteration
    config.SAVE_DATA_DICT = save_this_iteration
    config.SAVE_FIGURES = save_this_iteration
    config.PROFILING_FLAG = save_this_iteration
    config.PRINT_STEPS_FLAG = False
    config.OUTPUT_DIR = sim_output_dir
    config.INTERCALATION_ENABLED = False
    config.JAMMING_ENABLED = False
    config.GRADIENT = None
    config.DIRECTED_DIVISION_ANGLE = None
    
    import abm11
    importlib.reload(abm11)
    abm11.OUTPUT_DIR = sim_output_dir
    
    param_str_print = ', '.join([f'{name}={val:.4f}' for name, val in param_dict.items()])
    print(f"Running simulation {iteration_count+1} with {param_str_print}")
    if config.VIDEO_FLAG:
        print(f"  📹 Video/data/figures enabled for this iteration")
    print(f"Output directory: {sim_output_dir}")
    data_dict = abm11.run_simulation()
    
    # Get final positions and create boundary from convex hull
    Xe_growth = data_dict['Xe_growth']
    outgrowth_length = Xe_growth[:,0].max()
    coeffs_sim = data_dict['coefficients_growth']
    positions = data_dict['positions']
    final_pos = positions[-1]
    pos_valid = final_pos[~np.isnan(final_pos).any(axis=1)]
    Xe_final = data_dict['Xe_final'].copy()
    
    area_growth_region, perimeter, aspect_ratio, ellipticity, roundness, outgrowth_length, b, volume_fraction = data_dict['morphometrics_final']
    
    # load experimental curves
    ctrl_exp_curve = np.loadtxt('data/input/exp_curves/averages/control_avg_xy.csv', delimiter=',' ,skiprows=1)
    c59_exp_curve = np.loadtxt('data/input/exp_curves/averages/c59_avg_xy.csv', delimiter=',' ,skiprows=1)
    
    # Calculate error based on selected metric
    if error_metric == 'l2_ctrl':
        error = distance_metric(curve1=Xe_growth, curve2=ctrl_exp_curve, which='l2_mean')
    elif error_metric == 'l2_c59':
        error = distance_metric(curve1=Xe_growth, curve2=c59_exp_curve, which='l2_mean')
    elif error_metric == 'hausdorff_ctrl':
        error = distance_metric(curve1=Xe_growth, curve2=ctrl_exp_curve, which='hausdorff')
    elif error_metric == 'hausdorff_c59':
        error = distance_metric(curve1=Xe_growth, curve2=c59_exp_curve, which='hausdorff')
    
    elif error_metric == 'length_ctrl':
        error = np.abs((EXP_OUTGROWTH_LENGTH / abm11.CONVERSION_FACTOR_UM) - outgrowth_length) * abm11.CONVERSION_FACTOR_UM
    elif error_metric == 'length_c59':
        error = np.abs((EXP_OUTGROWTH_LENGTH_C59 / abm11.CONVERSION_FACTOR_UM) - outgrowth_length) * abm11.CONVERSION_FACTOR_UM
    elif error_metric == 'area':
        error = np.abs((EXP_AREA / abm11.CONVERSION_FACTOR_UM**2) - area_growth_region)

        raise ValueError(f"Unknown error metric: {error_metric}")
    
    # Track optimization progress
    iteration_count += 1
    param_history.append(list(params))
    error_history.append(error)
    boundary_history.append(Xe_final)
    
    if error < best_error:
        best_error = error
        best_params = list(params)
        sim_boundary_global = Xe_final.copy()
    
    # Print progress with denormalized values
    params_for_print = [param_dict[name] for name in param_names]
    best_params_for_print = best_params.copy() if best_params else None
    if best_params_for_print and 'MIGRATION_PERCENT' in param_names:
        idx = param_names.index('MIGRATION_PERCENT')
        best_params_for_print[idx] = best_params_for_print[idx] * 100.0
    
    print_optimization_step(iteration_count, params_for_print, error, best_params_for_print, best_error, param_names)
    
    # Generate plots every 10 iterations
    if iteration_count % 10 == 0:
        print(f"\n--- Generating intermediate plots at iteration {iteration_count} ---")
        try:
            create_individual_optimization_plots(param_history, error_history, 
                                                optimization_output_dir, param_names)
            plot_optimization_history(param_history, error_history, optimization_output_dir, param_names)
            print(f"--- Intermediate plots saved to {optimization_output_dir} ---\n")
        except Exception as e:
            print(f"Warning: Could not generate intermediate plots: {e}\n")

    return error

def run_optimization(parameter_config={'names': ['MIGRATION_PERCENT'], 'bounds': [(0.0, 1.0)], 
                                      'description': 'MIGRATION_PERCENT-only optimization'}, 
                     algorithm='Nelder-Mead',
                     error_metric='length_ctrl',
                     random_seed=None,
                     maxfev=50,
                     parent_dir=None):
    """Run optimization with flexible parameter configuration."""
    global param_history, error_history, boundary_history, best_params, best_error
    global sim_boundary_global, optimization_output_dir, param_names, param_bounds
    global iteration_count
    
    globals()['error_metric'] = error_metric
    
    if random_seed is not None:
        np.random.seed(random_seed)
    
    param_names = parameter_config['names']
    param_bounds = parameter_config['bounds']
    
    if len(param_names) != len(param_bounds):
        raise ValueError("Number of parameter names must match number of bounds")
    
    from datetime import datetime
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    params_str = '_'.join(param_names)
    seed_str = f'_seed{random_seed}' if random_seed is not None else ''
    folder_name = f'opt_{error_metric}_{params_str}_{timestamp}{seed_str}'
    
    # Use parent_dir if provided, otherwise use default DATA_DIR/output
    if parent_dir is not None:
        optimization_output_dir = os.path.join(parent_dir, folder_name)
    else:
        optimization_output_dir = os.path.join(DATA_DIR, 'output', folder_name)
    os.makedirs(optimization_output_dir, exist_ok=True)
    
    iteration_count = 0
    param_history = []
    error_history = []
    boundary_history = []
    best_error = float('inf')
    best_params = None
    sim_boundary_global = None
    
    print(f"=== Starting {algorithm} Optimization ===")
    print(f"Error metric: {error_metric}")
    if error_metric == 'length_ctrl':
        print(f"Target: Outgrowth length of {EXP_OUTGROWTH_LENGTH} um (Control)")
    elif error_metric == 'length_c59':
        print(f"Target: Outgrowth length of {EXP_OUTGROWTH_LENGTH_C59} um (C59)")
    elif error_metric == 'area':
        print(f"Target: Area of {EXP_AREA} um^2")
    else:
        print(f"Target: Shape matching using {error_metric}")
    print(f"Parameters to optimize: {', '.join(param_names)}")
    print(f"Parameter bounds: {dict(zip(param_names, param_bounds))}")
    print(f"Description: {parameter_config['description']}")
    print(f"Main output directory: {optimization_output_dir}")
    
    t0 = time.time()
    
    if algorithm == 'Nelder-Mead':
        print(f"Starting Nelder-Mead optimization...\n")
        
        n_params = len(param_names)
        
        if random_seed is not None:
            x0 = np.array([np.random.uniform(low, high) for low, high in param_bounds])
        else:
            x0 = np.array([(low + high) / 2.0 for low, high in param_bounds])
        
        initial_simplex = np.zeros((n_params + 1, n_params))
        initial_simplex[0] = x0.copy()
        
        for i in range(n_params):
            initial_simplex[i + 1] = x0.copy()
            low, high = param_bounds[i]
            
            if random_seed is not None:
                perturbation = np.random.uniform(0.1, 0.3) * (high - low)
                direction = np.random.choice([-1, 1])
                initial_simplex[i + 1, i] = np.clip(x0[i] + direction * perturbation, low, high)
            else:
                perturbation = 0.2 * (high - low)
                initial_simplex[i + 1, i] = min(x0[i] + perturbation, high)
        
        print(f"Initial simplex:")
        for i, vertex in enumerate(initial_simplex):
            denorm_str = []
            for name, val in zip(param_names, vertex):
                if name == 'MIGRATION_PERCENT':
                    denorm_str.append(f'{name}={val:.3f} ({val*100:.1f}%)')
                else:
                    denorm_str.append(f'{name}={val:.3f}')
            print(f"  Vertex {i+1}: {', '.join(denorm_str)}")
        print()
        
        result = minimize(objective_function, x0, method='Nelder-Mead', 
                        options={'maxfev': maxfev, 'disp': True, 'initial_simplex': initial_simplex})
    elif algorithm == 'Differential Evolution':
        print(f"Starting Differential Evolution optimization...\n")
        de_seed = random_seed if random_seed is not None else None
        de_maxiter = max(1, maxfev // 20) if maxfev < 50 else 9
        result = differential_evolution(objective_function, param_bounds, maxiter=de_maxiter, 
                                       popsize=10, polish=False, workers=1, seed=de_seed)
    
    print("\nOPTIMIZATION COMPLETE")
    
    result_denorm = {}
    for i, name in enumerate(param_names):
        if name == 'MIGRATION_PERCENT':
            result_denorm[name] = result.x[i] * 100.0
        else:
            result_denorm[name] = result.x[i]
    
    best_param_str = ', '.join([f'{name}={val:.4f}' for name, val in result_denorm.items()])
    print(f"Best parameters: {best_param_str}")
    print(f"Best error: {result.fun:.6f}")
    print(f"Total evaluations: {result.nfev}")
    print(f"Total time: {time.time() - t0:.1f} seconds")
    
    print(f"\nSaving optimization data...")
    save_optimization_data(param_history, error_history, boundary_history, param_names, optimization_output_dir)
    
    print(f"\nGenerating optimization plots...")
    create_individual_optimization_plots(param_history, error_history, optimization_output_dir, param_names)
    plot_optimization_history(param_history, error_history, optimization_output_dir, param_names)
    
    if len(error_history) > 0:
        print(f"Generating error convergence animation...")
        animate_error_convergence(error_history, optimization_output_dir)
        
        ph_array = np.array(param_history)
        if algorithm == 'Nelder-Mead' and ph_array.ndim == 2 and ph_array.shape[0] > 0 and len(param_names) == 2:
            print(f"Generating simplex convergence animation...")
            animate_simplex_convergence(param_history, error_history, param_bounds, optimization_output_dir, param_names)
    
    elapsed = time.time() - t0
    print(f"\nAll outputs saved to: {optimization_output_dir}")
    print(f"Time taken: {time.strftime('%H:%M:%S', time.gmtime(elapsed))}")
    
    return result

if __name__ == "__main__":
    result = run_optimization(
        parameter_config={
            'names': ['MIGRATION_PERCENT', 'KDIV'], 
            'bounds': [(0.0, 1.0), (0.0, 0.6)],
            'description': 'MIGRATION_PERCENT and KDIV optimization'
        },
        algorithm='Nelder-Mead',
        error_metric='l2_ctrl'
    )