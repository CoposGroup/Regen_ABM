"""Simplified 2D PSO optimization cases: (m, kdiv) and (m, kdeath)."""
import _setup_path
import numpy as np
import os
from datetime import datetime
from config import CTRL_AVG_FILE, C59_AVG_FILE
import sys

from param_optimization_pso import (
    run_pso_optimization,
    SIM_BOUNDS,
    scale_from_sim
)

SIM_BOUNDS['KDEATH'] = (0.0, 0.1)

def get_valley_direction(condition, case):
    """
    Get unit vector along the known valley direction in scaled [0,1] space.
    Only for (m, kdiv) cases - returns None for (m, kdeath).
    
    CTRL: m = -1.75*kdiv + 1 (where m is fraction 0-1, kdiv is 0-0.5)
    C59: m = -0.4*kdiv + 0.15
    """
    if case != 'm_kdiv':
        return None
    
    if condition == 'CTRL':

        valley_dir = np.array([-0.875, 1.0])
    else:  # C59
        valley_dir = np.array([-0.2, 1.0])
    valley_dir = valley_dir / np.linalg.norm(valley_dir)
    return valley_dir

def initialize_positions_along_valley(condition, case, n_particles):
    """
    Initialize particle positions along the valley.
    
    CTRL: m = -1.75*kdiv + 1
    C59: m = -0.4*kdiv + 0.15
    
    Returns:
        init_pos: [n_particles, 2] positions in scaled [0,1] space, or None
    """
    if case != 'm_kdiv':
        return None
    
    if condition == 'CTRL':
        kdiv_scaled = np.linspace(0.1, 0.9, n_particles)
        m_scaled = -0.875 * kdiv_scaled + 1.0
    else:  # C59
        kdiv_scaled = np.linspace(0.0, 0.7, n_particles)
        m_scaled = -0.2 * kdiv_scaled + 0.15
    
    # Clip to valid [0, 1] range
    m_scaled = np.clip(m_scaled, 0.0, 1.0)
    kdiv_scaled = np.clip(kdiv_scaled, 0.0, 1.0)
    
    positions = np.column_stack([m_scaled, kdiv_scaled])
    
    valley_dir = get_valley_direction(condition, case)
    perpendicular = np.array([valley_dir[1], -valley_dir[0]])  # Noise
    
    for i in range(n_particles):
        noise = np.random.uniform(-0.02, 0.02)
        positions[i] += noise * perpendicular
    
    # Clip again to ensure bounds
    positions = np.clip(positions, 0.0, 1.0)
    
    return positions

def initialize_velocities_along_valley(n_particles, valley_dir, magnitude=0.02):
    """
    Initialize velocities.
    - If valley_dir provided: velocities along valley (for m-kdiv)
    - If valley_dir is None: small random velocities (for m-kdeath)
    """
    if valley_dir is None:
        return np.random.uniform(-magnitude, magnitude, size=(n_particles, 2))
    
    directions = np.random.choice([-1, 1], size=n_particles)
    magnitudes = np.random.uniform(0.5*magnitude, 1.5*magnitude, size=n_particles)
    init_vel = np.outer(directions * magnitudes, valley_dir)
    
    return init_vel

def run_2d_optimizations(
    n_particles=25,
    n_iterations=50,
    n_processes=30
):
    """Run two 2D optimization cases."""
    
    exp_data_files = {
        'l2_ctrl': CTRL_AVG_FILE,
        'l2_c59': C59_AVG_FILE
    }
    
    parent_output_dir = f"pso_2d_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    os.makedirs(parent_output_dir, exist_ok=True)
    
    all_results = {}
    
    pso_options_2d = {
        'c1': 1.6,
        'c2': 0.4,
        'w': 0.8, ### will try turning down to 0.65
        'k': 2,
        'p': 2
    }
    
    velocity_clamp_2d = (
        -0.25 * np.ones(2),
        0.25 * np.ones(2)
    )
    
    for condition in ['CTRL', 'C59']:
        print(f"\n{'#'*70}")
        print(f"# {condition} - 2D Optimizations")
        print(f"{'#'*70}")
        
        condition_dir = os.path.join(parent_output_dir, condition)
        os.makedirs(condition_dir, exist_ok=True)
        
        error_metric = 'l2_ctrl' if condition == 'CTRL' else 'l2_c59'
        exp_data_file = exp_data_files[error_metric]
        
        # Case 1: Optimize (m, kdiv) - initialize along valley
        print(f"\n{'='*70}")
        print(f"Case 1: Optimize MIGRATION_FRACTION and KDIV")
        print(f"Fixed: KB_MID=75, KB_MIN=1")
        print(f"{'='*70}")
        
        case1_param_names = ['MIGRATION_FRACTION', 'KDIV']
        
        case1_bounds = (
            np.array([0.0, 0.0]),
            np.array([1.0, 1.0])
        )
        
        # Initialize positions along valley
        init_pos_case1 = initialize_positions_along_valley(condition, 'm_kdiv', n_particles)
        
        # Initialize velocities along valley
        valley_dir_1 = get_valley_direction(condition, 'm_kdiv')
        init_vel_case1 = initialize_velocities_along_valley(
            n_particles, valley_dir_1, magnitude=0.08
        )
        
        print(f"Initializing along valley: {condition}")
        if condition == 'CTRL':
            print(f"  Equation: m = -1.75*kdiv + 1")
        else:
            print(f"  Equation: m = -0.4*kdiv + 0.15")
        print(f"  Valley direction (scaled): {valley_dir_1}")
        print(f"  Initial pos sample: m={init_pos_case1[0][0]:.3f}, kdiv={init_pos_case1[0][1]:.3f}")
        print(f"  Initial vel sample: {init_vel_case1[0]}")
        
        # Dummy center (will be overridden by init_pos)
        case1_center = {
            'MIGRATION_FRACTION': 0.5,
            'KDIV': 0.25,
            'KB_MID': 75.0,
            'KB_MIN': 1.0
        }
        
        results1, run_dir1 = run_pso_optimization(
            swarm_center_sim=case1_center,
            swarm_name=f'{condition}_m_kdiv',
            param_names=case1_param_names,
            error_metric=error_metric,
            exp_data_file=exp_data_file,
            parent_dir=condition_dir,
            n_particles=n_particles,
            n_iterations=n_iterations,
            n_processes=n_processes,
            pso_options=pso_options_2d,
            bounds_custom=case1_bounds,
            topology='lbest',
            velocity_clamp=velocity_clamp_2d,
            init_vel=init_vel_case1,
            init_pos=init_pos_case1
        )
        
        all_results[f'{condition}_m_kdiv'] = {
            'results': results1,
            'run_dir': run_dir1,
        }
        
        # Case 2: Optimize (m, kdeath) - random initialization
        print(f"\n{'='*70}")
        print(f"Case 2: Optimize MIGRATION_FRACTION and KDEATH")
        print(f"Fixed: KDIV=0.4, KB_MID=75, KB_MIN=1")
        print(f"{'='*70}")
        
        case2_center = {
            'MIGRATION_FRACTION': 0.5,
            'KDEATH': 0.05,
            'KDIV': 0.4,
            'KB_MID': 75.0,
            'KB_MIN': 1.0
        }
        
        case2_param_names = ['MIGRATION_FRACTION', 'KDEATH']
        
        case2_bounds = (
            np.array([0.0, 0.0]),
            np.array([1.0, 1.0])
        )
        
        # No valley for m-kdeath
        valley_dir_2 = get_valley_direction(condition, 'm_kdeath')
        init_vel_case2 = initialize_velocities_along_valley(
            n_particles, valley_dir_2, magnitude=0.02
        )
        
        print(f"No valley for m-kdeath, using random velocities")
        print(f"Initial vel sample: {init_vel_case2[0]}")
        
        results2, run_dir2 = run_pso_optimization(
            swarm_center_sim=case2_center,
            swarm_name=f'{condition}_m_kdeath',
            param_names=case2_param_names,
            error_metric=error_metric,
            exp_data_file=exp_data_file,
            parent_dir=condition_dir,
            n_particles=n_particles,
            n_iterations=n_iterations,
            n_processes=n_processes,
            pso_options=pso_options_2d,
            bounds_custom=case2_bounds,
            topology='lbest',
            velocity_clamp=velocity_clamp_2d,
            init_vel=init_vel_case2
        )
        
        all_results[f'{condition}_m_kdeath'] = {
            'results': results2,
            'run_dir': run_dir2,
        }
    
    import pickle
    with open(os.path.join(parent_output_dir, 'all_2d_summary.pkl'), 'wb') as f:
        pickle.dump(all_results, f)
    
    print(f"\n{'#'*70}")
    print("# ALL 2D OPTIMIZATIONS COMPLETE")
    print(f"{'#'*70}")
    print(f"Results: {parent_output_dir}")
    for name, data in all_results.items():
        print(f"  {name}: Best error = {data['results']['best_cost']:.3f}")
    
    return all_results, parent_output_dir

if __name__ == '__main__':
    results, output_dir = run_2d_optimizations(
        n_particles=25,
        n_iterations=50,
        n_processes=30
    )