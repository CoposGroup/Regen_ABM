"""
Script to run multiple optimization runs with different random initial conditions,
then generate comprehensive comparison plots.

Usage:
    python run_multiple_optimizations.py --n_runs 5 --algorithm Nelder-Mead
    python run_multiple_optimizations.py --n_runs 3 --algorithm "Differential Evolution"
    python run_multiple_optimizations.py --n_runs 3 --parallel --max_workers 8
"""
import numpy as np
import argparse
import os
import time
from datetime import datetime
import pickle
import multiprocessing as mp

from utils.optimization_utils import (
    load_optimization_data,
    create_individual_optimization_plots,
    create_multi_run_comparison_plots,
    create_ctrl_vs_c59_comparison_plots
)
from config import DATA_DIR


def run_single_optimization_with_seed(parameter_config, algorithm, error_metric, seed, maxfev=40, parent_dir=None):
    """
    Run a single optimization with a specific random seed.
    
    Args:
        parameter_config: Dict with 'names', 'bounds', 'description'
        algorithm: 'Nelder-Mead' or 'Differential Evolution'
        error_metric: 'length_ctrl' or 'length_c59'
        seed: Random seed for reproducibility
        maxfev: Maximum function evaluations (default: 40)
        parent_dir: Parent directory for this optimization (default: DATA_DIR/output)
    
    Returns:
        output_dir: Directory where optimization results were saved
    """
    print(f"\n{'='*80}")
    print(f"OPTIMIZATION RUN - Seed {seed}")
    print(f"{'='*80}\n")
    
    output_dir = run_optimization_with_seed(parameter_config, algorithm, error_metric, seed, maxfev, parent_dir)
    
    return output_dir


def run_optimization_with_seed(parameter_config, algorithm, error_metric, seed, maxfev=40, parent_dir=None):
    """Run optimization with a specific random seed."""
    import param_optimization
    
    # Call the updated run_optimization function with random_seed parameter
    result = param_optimization.run_optimization(
        parameter_config=parameter_config,
        algorithm=algorithm,
        error_metric=error_metric,
        random_seed=seed,
        maxfev=maxfev,
        parent_dir=parent_dir
    )
    
    # Extract output directory from the folder name that was created
    # The optimization_output_dir is set as a global in run_optimization
    output_dir = param_optimization.optimization_output_dir
    
    return output_dir


def run_optimization_wrapper(args):
    """
    Wrapper function for parallel execution.
    
    Args:
        args: Tuple of (run_number, seed, parameter_config, algorithm, error_metric, maxfev, parent_dir)
    
    Returns:
        Tuple of (success, output_dir or None, run_number)
    """
    run_num, seed, parameter_config, algorithm, error_metric, maxfev, parent_dir = args
    
    print(f"\n### RUN {run_num} (Seed {seed}) ###")
    
    try:
        output_dir = run_single_optimization_with_seed(
            parameter_config, algorithm, error_metric, seed, maxfev, parent_dir
        )
        print(f"✓ Run {run_num} complete: {output_dir}")
        return (True, output_dir, run_num)
    except Exception as e:
        print(f"✗ Run {run_num} failed with error: {e}")
        import traceback
        traceback.print_exc()
        return (False, None, run_num)


def run_multiple_optimizations(n_runs, parameter_config, algorithm, error_metric, 
                               parallel=True, max_workers=None, maxfev=40, parent_dir=None):
    """
    Run multiple optimization runs with different random seeds.
    
    Args:
        n_runs: Number of optimization runs
        parameter_config: Dict with 'names', 'bounds', 'description'
        algorithm: 'Nelder-Mead' or 'Differential Evolution'
        error_metric: 'length_ctrl' or 'length_c59'
        parallel: If True, run in parallel (default: True)
        max_workers: Max parallel workers (None = all CPU cores)
        maxfev: Maximum function evaluations (default: 40)
        parent_dir: Parent directory for optimization outputs (default: DATA_DIR/output)
    
    Returns:
        output_dirs: List of output directories for each run
    """
    base_seed = int(time.time()) % 10000
    
    print(f"\n{'='*80}")
    print(f"RUNNING {n_runs} OPTIMIZATION RUNS")
    print(f"Algorithm: {algorithm}")
    print(f"Error Metric: {error_metric}")
    print(f"Max function evaluations: {maxfev}")
    print(f"Base seed: {base_seed}")
    print(f"Mode: {'PARALLEL' if parallel else 'SERIAL'}")
    if parallel:
        if max_workers is None:
            print(f"Using all available CPU cores: {mp.cpu_count()}")
        else:
            print(f"Using {max_workers} parallel workers")
    if parent_dir:
        print(f"Output parent directory: {parent_dir}")
    print(f"{'='*80}\n")
    
    # Prepare job arguments
    jobs = []
    for i in range(n_runs):
        seed = base_seed + i
        jobs.append((i+1, seed, parameter_config, algorithm, error_metric, maxfev, parent_dir))
    
    # Run jobs
    if parallel:
        with mp.Pool(processes=max_workers) as pool:
            results = pool.map(run_optimization_wrapper, jobs)
    else:
        results = [run_optimization_wrapper(job) for job in jobs]
    
    # Extract output directories from successful runs
    output_dirs = []
    successful = 0
    for success, output_dir, run_num in results:
        if success:
            output_dirs.append(output_dir)
            successful += 1
    
    print(f"\n{'='*80}")
    print(f"Optimization runs complete!")
    print(f"Successful: {successful}/{n_runs}")
    print(f"Failed: {n_runs - successful}/{n_runs}")
    print(f"{'='*80}\n")
    
    return output_dirs


def load_all_runs_data(output_dirs):
    """Load optimization data from all runs."""
    all_data = []
    
    for output_dir in output_dirs:
        data_file = os.path.join(output_dir, 'optimization_data.pkl')
        if os.path.exists(data_file):
            data = load_optimization_data(data_file)
            all_data.append(data)
        else:
            print(f"Warning: Could not find {data_file}")
    
    return all_data


def main():
    parser = argparse.ArgumentParser(description='Run multiple optimizations with different random seeds')
    parser.add_argument('--n_runs', type=int, default=10, help='Number of optimization runs per metric')
    parser.add_argument('--algorithm', type=str, default='Nelder-Mead',
                       choices=['Nelder-Mead', 'Differential Evolution'],
                       help='Optimization algorithm')
    parser.add_argument('--ctrl_only', action='store_true', help='Only run Control optimizations')
    parser.add_argument('--c59_only', action='store_true', help='Only run C59 optimizations')
    parser.add_argument('--parallel', action='store_true', default=True, help='Run in parallel (default: True)')
    parser.add_argument('--serial', action='store_true', help='Run serially (overrides --parallel)')
    parser.add_argument('--max_workers', type=int, default=8, 
                       help='Maximum number of parallel workers (default: 8)')
    
    args = parser.parse_args()
    
    # Handle serial/parallel flags
    parallel = args.parallel and not args.serial
    
    # Define parameter configuration
    parameter_config = {
        'names': ['MIGRATION_PERCENT', 'KDIV'],
        'bounds': [(0.0, 1.0), (0.0, 0.6)],
        'description': 'MIGRATION_PERCENT and KDIV optimization'
    }
    
    # Create timestamp for this batch
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    batch_dir = os.path.join(DATA_DIR, 'output', f'multi_run_{timestamp}')
    os.makedirs(batch_dir, exist_ok=True)
    
    print(f"\n{'='*80}")
    print(f"MULTI-RUN OPTIMIZATION BATCH")
    print(f"Batch directory: {batch_dir}")
    print(f"{'='*80}\n")
    
    # Run Control optimizations
    ctrl_output_dirs = []
    if not args.c59_only:
        print("\n" + "="*80)
        print("RUNNING CONTROL OPTIMIZATIONS")
        print("="*80)
        ctrl_output_dirs = run_multiple_optimizations(
            args.n_runs, parameter_config, args.algorithm, 'length_ctrl',
            parallel=parallel, max_workers=args.max_workers, parent_dir=batch_dir
        )
    
    # Run C59 optimizations
    c59_output_dirs = []
    if not args.ctrl_only:
        print("\n" + "="*80)
        print("RUNNING C59 OPTIMIZATIONS")
        print("="*80)
        c59_output_dirs = run_multiple_optimizations(
            args.n_runs, parameter_config, args.algorithm, 'length_c59',
            parallel=parallel, max_workers=args.max_workers, parent_dir=batch_dir
        )
    
    # Generate comparison plots
    print("\n" + "="*80)
    print("GENERATING COMPARISON PLOTS")
    print("="*80)
    
    comparison_dir = os.path.join(batch_dir, 'comparison_plots')
    os.makedirs(comparison_dir, exist_ok=True)
    
    # Load all data
    if ctrl_output_dirs:
        ctrl_data = load_all_runs_data(ctrl_output_dirs)
        if len(ctrl_data) > 0:
            ctrl_plots_dir = os.path.join(comparison_dir, 'control')
            create_multi_run_comparison_plots(
                ctrl_data, ctrl_plots_dir, metric_name='Control',
                ylabel=r'Absolute Length Difference ($\mu$m)'
            )
    
    if c59_output_dirs:
        c59_data = load_all_runs_data(c59_output_dirs)
        if len(c59_data) > 0:
            c59_plots_dir = os.path.join(comparison_dir, 'c59')
            create_multi_run_comparison_plots(
                c59_data, c59_plots_dir, metric_name='C59',
                ylabel=r'Absolute Length Difference ($\mu$m)'
            )
    
    # Create Control vs C59 comparison if both were run
    if ctrl_output_dirs and c59_output_dirs:
        if len(ctrl_data) > 0 and len(c59_data) > 0:
            print("\n" + "="*80)
            print("GENERATING CONTROL VS C59 COMPARISON")
            print("="*80)
            ctrl_vs_c59_dir = os.path.join(comparison_dir, 'ctrl_vs_c59')
            create_ctrl_vs_c59_comparison_plots(ctrl_data, c59_data, ctrl_vs_c59_dir)
    
    # Save summary
    summary_file = os.path.join(batch_dir, 'batch_summary.txt')
    with open(summary_file, 'w') as f:
        f.write(f"Multi-Run Optimization Batch\n")
        f.write(f"{'='*80}\n\n")
        f.write(f"Timestamp: {timestamp}\n")
        f.write(f"Algorithm: {args.algorithm}\n")
        f.write(f"Runs per metric: {args.n_runs}\n\n")
        
        f.write(f"Control Runs ({len(ctrl_output_dirs)}):\n")
        for i, out_dir in enumerate(ctrl_output_dirs):
            f.write(f"  {i+1}. {out_dir}\n")
        
        f.write(f"\nC59 Runs ({len(c59_output_dirs)}):\n")
        for i, out_dir in enumerate(c59_output_dirs):
            f.write(f"  {i+1}. {out_dir}\n")
        
        f.write(f"\nComparison plots: {comparison_dir}\n")
    
    print("\n" + "="*80)
    print("BATCH COMPLETE")
    print(f"Summary saved to: {summary_file}")
    print(f"Comparison plots: {comparison_dir}")
    print("="*80)


if __name__ == "__main__":
    mp.set_start_method('spawn', force=True)
    main()

