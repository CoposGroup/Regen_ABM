"""PSO-based parameter optimization for limb regeneration ABM with multi-swarm strategy."""
import _setup_path
import numpy as np
import os
import sys
import time
import importlib
import pickle
import traceback
from datetime import datetime
import pyswarms as ps
import matplotlib.pyplot as plt

import abm, config
from config import CTRL_AVG_FILE, C59_AVG_FILE
from utils.post_process import distance_metric, send_email_notification

SIM_BOUNDS = {
    'MIGRATION_FRACTION': (0.0, 1.0),
    'KDIV': (0.0, 0.5),
    'KB_MID': (0.0, 100.0),
    'KB_MIN': (0.0, 100.0),
}

def scale_to_sim(params_scaled, param_names):
    """Convert [0,1] scaled params to simulation bounds."""
    params_sim = {}
    for i, name in enumerate(param_names):
        lower, upper = SIM_BOUNDS[name]
        params_sim[name] = lower + params_scaled[i] * (upper - lower)
    return params_sim

def scale_from_sim(params_sim, param_names):
    """Convert simulation params to [0,1] scaled."""
    params_scaled = np.zeros(len(param_names))
    for i, name in enumerate(param_names):
        lower, upper = SIM_BOUNDS[name]
        params_scaled[i] = (params_sim[name] - lower) / (upper - lower)
    return params_scaled

def objective_function_single(params_scaled, param_names, error_metric, exp_curve, 
                             parent_dir, swarm_name, iteration_count=0, particle_idx=0):
    """Objective function for a single parameter set."""
    param_dict_sim = scale_to_sim(params_scaled, param_names)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    param_str = '_'.join([f"{name[:3].upper()}{int(param_dict_sim[name]):04d}" 
                          for name in param_names])
    run_name = f"iter{iteration_count:03d}_p{particle_idx:02d}_{swarm_name}_{param_str}"
    run_dir = os.path.join(parent_dir, run_name)
    os.makedirs(run_dir, exist_ok=True)
    
    try:
        for name, value in param_dict_sim.items():
            setattr(config, name, value)
        
        if 'KDIV' in param_dict_sim:
            kdiv_val = param_dict_sim['KDIV']
            if kdiv_val > 0:
                total_cycle_length = 1.0 / kdiv_val
                g_length = total_cycle_length / 2.0
                m_length = total_cycle_length / 2.0
            else:
                g_length = float('inf')
                m_length = float('inf')
            setattr(config, 'G_LENGTH', float(g_length))
            setattr(config, 'M_LENGTH', float(m_length))

        config.MIGRATION_ENABLED = 'MIGRATION_FRACTION' in param_dict_sim
        if 'KDIV' in param_dict_sim:
            config.KDIV = param_dict_sim['KDIV']
        
        is_best_particle = (particle_idx == _global_best_particle_idx)
        save_this_iteration = (iteration_count == 0) or ((iteration_count % 10 == 0) and is_best_particle)
        
        config.VIDEO_FLAG = save_this_iteration
        config.SAVE_DATA_DICT = save_this_iteration
        config.SAVE_FIGURES = save_this_iteration
        config.PROFILING_FLAG = save_this_iteration
        config.PRINT_STEPS_FLAG = False
        config.OUTPUT_DIR = run_dir
        config.INTERCALATION_ENABLED = False
        config.JAMMING_ENABLED = False
        config.GRADIENT = None
        config.DIRECTED_DIVISION_ANGLE = None
        config.TMAX = 7.0
        config.EXT_STRESS_FORCE = False
        config.SPORATIC_SOFTENING = False

        importlib.reload(abm)
        abm.OUTPUT_DIR = run_dir
        data_dict = abm.run_simulation()
        
        Xe_growth = data_dict['Xe_growth']
        Xe_final = data_dict['Xe_final']
        positions = data_dict['positions']
        final_pos = positions[-1] if len(positions) > 0 else np.array([])
        final_active_mask = ~np.isnan(final_pos).any(axis=1) if len(final_pos) > 0 else np.array([], dtype=bool)
        final_cell_count = int(np.sum(final_active_mask))
        
        sim_curve = Xe_growth * config.CONVERSION_FACTOR_UM
        error = distance_metric(curve1=sim_curve, curve2=exp_curve, which='rmse')
        
        light_results = {
            'Xe_final': Xe_final,
            'Xe_growth': Xe_growth,
            'final_cell_count': final_cell_count,
            'morphometrics_final': data_dict.get('morphometrics_final', None),
            'error': error,
            'params_sim': param_dict_sim.copy(),
            'params_scaled': dict(zip(param_names, params_scaled)),
            'iteration': iteration_count + 1,
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
        }
        
        with open(os.path.join(run_dir, 'results.pkl'), 'wb') as f:
            pickle.dump(light_results, f)
        
        return error
        
    except Exception as e:
        error_msg = f"ERROR in {run_name}:\n{traceback.format_exc()}"
        with open(os.path.join(run_dir, 'ERROR.txt'), 'w') as f:
            f.write(error_msg)
            f.write(f"\nParameters (scaled): {dict(zip(param_names, params_scaled))}\n")
            f.write(f"Parameters (sim): {param_dict_sim}\n")
        
        try:
            send_email_notification(f"ERROR: {run_name}", error_msg)
        except:
            pass
        
        return 500.0
def _eval_particle_wrapper(args):
    """Module-level wrapper for parallel particle evaluation."""
    i, params, pnames, err_metric, exp, pdir, sname, iter_cnt = args
    return objective_function_single(params, pnames, err_metric, exp, 
                                    pdir, sname, iter_cnt, i)

def objective_function_swarm(swarm_positions, param_names, error_metric, exp_curve, 
                            parent_dir, swarm_name, iteration_count=0, n_processes=1):
    """Objective function for entire swarm."""
    n_particles = swarm_positions.shape[0]
    errors = np.zeros(n_particles)
    
    if n_processes > 1:
        from multiprocessing import Pool, set_start_method
        try:
            set_start_method('spawn', force=True)
        except:
            pass
        
        args_list = [
            (i, swarm_positions[i, :], param_names, error_metric, exp_curve, 
             parent_dir, swarm_name, iteration_count)
            for i in range(n_particles)
        ]
        
        with Pool(processes=n_processes) as pool:
            results = pool.map(_eval_particle_wrapper, args_list)
        errors = np.array(results)
    else:
        for i in range(n_particles):
            errors[i] = objective_function_single(
                swarm_positions[i, :], param_names, error_metric, exp_curve,
                parent_dir, swarm_name, iteration_count, i
            )
    
    return errors

_global_cost_history = []
_global_param_names = None
_global_error_metric = None
_global_exp_data = None
_global_parent_dir = None
_global_swarm_name = None
_global_iteration_count = 0
_global_best_particle_idx = 0
_global_n_processes = 1

def objective_wrapper_global(positions):
    """Wrapper for PySwarms objective function."""
    return objective_function_swarm(
        positions, _global_param_names, _global_error_metric, _global_exp_data,
        _global_parent_dir, _global_swarm_name, _global_iteration_count, _global_n_processes
    )

def save_history(history, run_dir):
    """Save optimization history."""
    with open(os.path.join(run_dir, 'pso_history.pkl'), 'wb') as f:
        pickle.dump(history, f)
    
    cost_history = history['cost_history']
    iterations = range(1, len(cost_history) + 1)
    
    plt.figure(figsize=(10, 6))
    plt.plot(iterations, cost_history, 'b-', linewidth=2)
    plt.xlabel('Iteration')
    plt.ylabel('Best Error')
    plt.title('PSO Convergence')
    plt.grid(True, alpha=0.3)
    plt.ylim(0, 250)
    plt.tight_layout()
    plt.savefig(os.path.join(run_dir, 'pso_convergence.pdf'))
    plt.close()

def run_pso_optimization(
    swarm_center_sim,
    swarm_name,
    param_names,
    error_metric,
    exp_data_file,
    parent_dir,
    n_particles=10,
    n_iterations=50,
    n_processes=30,
    pso_options=None,
    bounds_custom=None,
    topology='lbest',
    velocity_clamp=None,
    init_vel=None,
    init_pos=None
):
    """Run PSO optimization for a single swarm."""
    
    print(f"\n{'='*70}")
    print(f"Starting: {swarm_name}")
    print(f"Center (sim): {swarm_center_sim}")
    print(f"Particles: {n_particles}, Iterations: {n_iterations}, Topology: {topology}")
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    run_dir = os.path.join(parent_dir, f"{swarm_name}_{timestamp}")
    os.makedirs(run_dir, exist_ok=True)
    
    exp_curve = np.loadtxt(exp_data_file, skiprows=1, delimiter=',')
    
    center_scaled = scale_from_sim(swarm_center_sim, param_names)
    
    if bounds_custom is None:
        bounds_lower = np.zeros(len(param_names))
        bounds_upper = np.ones(len(param_names))
    else:
        bounds_lower, bounds_upper = bounds_custom
    
    bounds = (bounds_lower, bounds_upper)
    
    if init_pos is not None:
        initial_positions = init_pos.copy()
    else:
        spread_abs = 0.05
        initial_positions = np.random.normal(
            loc=center_scaled,
            scale=spread_abs,
            size=(n_particles, len(param_names))
        )
        initial_positions = np.clip(initial_positions, bounds_lower, bounds_upper)
        
    if pso_options is None:
        pso_options = {'c1': 1.5, 'c2': 1.5, 'w': 0.6, 'k': 3, 'p': 2}
    
    if topology == 'lbest':
        optimizer = ps.single.LocalBestPSO(
            n_particles=n_particles,
            dimensions=len(param_names),
            options=pso_options,
            bounds=bounds,
            init_pos=initial_positions,
            velocity_clamp=velocity_clamp
        )
    else:
        optimizer = ps.single.GlobalBestPSO(
            n_particles=n_particles,
            dimensions=len(param_names),
            options=pso_options,
            bounds=bounds,
            init_pos=initial_positions,
            velocity_clamp=velocity_clamp
        )
    
    # Manually set initial velocities if provided
    if init_vel is not None:
        optimizer.swarm.velocity = init_vel.copy()
    
    global _global_cost_history, _global_param_names, _global_error_metric
    global _global_exp_data, _global_parent_dir, _global_swarm_name
    global _global_iteration_count, _global_best_particle_idx, _global_n_processes
    
    _global_cost_history = []
    _global_param_names = param_names
    _global_error_metric = error_metric
    _global_exp_data = exp_curve
    _global_parent_dir = run_dir
    _global_swarm_name = swarm_name
    _global_n_processes = n_processes
    
    pos_history = []
    vel_history = []
    
    start_time = time.time()
    
    for i in range(n_iterations):
        _global_iteration_count = i
        optimizer.optimize(objective_wrapper_global, iters=1, verbose=False)
        
        _global_cost_history.append(optimizer.swarm.best_cost)
        _global_best_particle_idx = np.argmin(optimizer.swarm.current_cost)
        
        pos_history.append(optimizer.swarm.position.copy())
        vel_history.append(optimizer.swarm.velocity.copy())
        
        print(f"Iter {i+1}/{n_iterations}: Best={optimizer.swarm.best_cost:.3f}")
        
        if i == 0 or (i + 1) % 10 == 0:
            elapsed = time.time() - start_time
            subject = f"PSO: {swarm_name} - Iter {i+1}/{n_iterations}"
            body = f"""Swarm: {swarm_name}
Iteration: {i+1}/{n_iterations} ({(i+1)/n_iterations*100:.1f}%)
Best Error: {optimizer.swarm.best_cost:.3f}
Time: {elapsed/3600:.2f}h elapsed, {(elapsed/(i+1))*(n_iterations-i-1)/3600:.2f}h remaining
"""
            try:
                send_email_notification(subject, body)
            except:
                pass
    
    elapsed_time = time.time() - start_time
    best_cost = optimizer.swarm.best_cost
    best_pos_scaled = optimizer.swarm.best_pos
    best_params_sim = scale_to_sim(best_pos_scaled, param_names)
    
    results = {
        'best_position_scaled': best_pos_scaled,
        'best_params_sim': best_params_sim,
        'best_cost': best_cost,
        'cost_history': _global_cost_history,
        'pos_history': pos_history,
        'vel_history': vel_history,
        'swarm_center_sim': swarm_center_sim,
        'param_names': param_names,
        'n_particles': n_particles,
        'n_iterations': n_iterations,
        'elapsed_time': elapsed_time,
        'pso_options': pso_options,
        'topology': topology,
    }
    
    with open(os.path.join(run_dir, 'pso_results.pkl'), 'wb') as f:
        pickle.dump(results, f)
    
    history = {'cost_history': _global_cost_history}
    save_history(history, run_dir)
    
    print(f"Complete: {swarm_name}, Best={best_cost:.3f}, Time={elapsed_time/3600:.2f}h")
    
    return results, run_dir

def define_swarms():
    """Define swarm configurations based on biological regions."""
    
    param_names = ['MIGRATION_FRACTION', 'KDIV', 'KB_MID', 'KB_MIN']
    
    swarms = {
        'CTRL': [
            {
                'name': 'CTRL_valley',
                'center': {'MIGRATION_FRACTION': 0.3, 'KDIV': 0.4, 'KB_MID': 75, 'KB_MIN': 1},
                'options': {'c1': 1.5, 'c2': 0.7, 'w': 0.5, 'k': 2, 'p': 2},
                'topology': 'lbest',
            },
            {
                'name': 'CTRL_low_stiff',
                'center': {'MIGRATION_FRACTION': 0.3, 'KDIV': 0.2, 'KB_MID': 20, 'KB_MIN': 20},
                'options': {'c1': 1.0, 'c2': 0.6, 'w': 0.5, 'k': 2, 'p': 2},
                'topology': 'lbest',
                'bounds_custom': (
                    np.array([0.0, 0.0, 0.0, 0.0]),
                    np.array([0.8, 0.4, 0.3, 0.3])
                ),
            },
            {
                'name': 'CTRL_high_stiff',
                'center': {'MIGRATION_FRACTION': 0.8, 'KDIV': 0.4, 'KB_MID': 90, 'KB_MIN': 90},
                'options': {'c1': 1.2, 'c2': 1.0, 'w': 0.4, 'k': 3, 'p': 2},
                'topology': 'lbest',
            },
        ],
        'C59': [
            {
                'name': 'C59_valley',
                'center': {'MIGRATION_FRACTION': 0.1, 'KDIV': 0.125, 'KB_MID': 75, 'KB_MIN': 1},
                'options': {'c1': 1.5, 'c2': 0.7, 'w': 0.5, 'k': 2, 'p': 2},
                'topology': 'lbest',
            },
            {
                'name': 'C59_uniform_stiff',
                'center': {'MIGRATION_FRACTION': 0.2, 'KDIV': 0.35, 'KB_MID': 75, 'KB_MIN': 75},
                'options': {'c1': 1.2, 'c2': 1.0, 'w': 0.4, 'k': 3, 'p': 2},
                'topology': 'lbest',
            },
            {
                'name': 'C59_tip_stiff',
                'center': {'MIGRATION_FRACTION': 0.1, 'KDIV': 0.2, 'KB_MID': 20, 'KB_MIN': 60},
                'options': {'c1': 1.3, 'c2': 0.8, 'w': 0.45, 'k': 2, 'p': 2},
                'topology': 'lbest',
            },
        ],
    }
    
    return swarms, param_names

if __name__ == '__main__':
    
    exp_data_files = {
        'l2_ctrl': CTRL_AVG_FILE,
        'l2_c59': C59_AVG_FILE
    }
    
    swarms, param_names = define_swarms()
    
    parent_output_dir = f"pso_multiswarm_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    os.makedirs(parent_output_dir, exist_ok=True)
    
    all_results = {}
    
    for condition in ['CTRL', 'C59']:
        print(f"\n{'#'*70}")
        print(f"# {condition} swarms")
        print(f"{'#'*70}")
        
        condition_dir = os.path.join(parent_output_dir, condition)
        os.makedirs(condition_dir, exist_ok=True)
        
        error_metric = 'l2_ctrl' if condition == 'CTRL' else 'l2_c59'
        exp_data_file = exp_data_files[error_metric]
        
        for swarm_config in swarms[condition]:
            results, run_dir = run_pso_optimization(
                swarm_center_sim=swarm_config['center'],
                swarm_name=swarm_config['name'],
                param_names=param_names,
                error_metric=error_metric,
                exp_data_file=exp_data_file,
                parent_dir=condition_dir,
                n_particles=10,
                n_iterations=150,
                n_processes=30,
                pso_options=swarm_config['options'],
                bounds_custom=swarm_config.get('bounds_custom', None),
                topology=swarm_config['topology']
            )
            
            all_results[swarm_config['name']] = {
                'results': results,
                'run_dir': run_dir,
            }
    
    with open(os.path.join(parent_output_dir, 'all_swarms_summary.pkl'), 'wb') as f:
        pickle.dump(all_results, f)
    
    print(f"\n{'#'*70}")
    print("# COMPLETE")
    print(f"{'#'*70}")
    print(f"Results: {parent_output_dir}")
    for name, data in all_results.items():
        print(f"  {name}: {data['results']['best_cost']:.3f}")