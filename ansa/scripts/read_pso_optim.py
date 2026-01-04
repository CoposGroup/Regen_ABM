"""Analyze ongoing PSO optimization run (read-only)."""
import _setup_path
import numpy as np
import os
import pickle
import glob
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
import pandas as pd
from pathlib import Path
import re
from datetime import datetime
import pyswarms as ps
from pyswarms.utils.plotters import plot_cost_history
from pyswarms.utils.plotters.formatters import Designer

def parse_run_directory(run_dir):
    """Parse results.pkl files from PSO run directory."""
    print(f"Parsing run directory: {run_dir}")
    
    results_files = glob.glob(os.path.join(run_dir, '**/results.pkl'), recursive=True)
    print(f"Found {len(results_files)} results.pkl files")
    
    if len(results_files) == 0:
        print("No results files found")
        return None
    
    data = []
    for results_file in results_files:
        try:
            with open(results_file, 'rb') as f:
                result = pickle.load(f)
            
            dir_name = os.path.basename(os.path.dirname(results_file))
            match = re.match(r'iter(\d+)_p(\d+)_', dir_name)
            if match:
                iteration = int(match.group(1))
                particle_idx = int(match.group(2))
            else:
                continue
            
            entry = {
                'iteration': iteration,
                'particle_idx': particle_idx,
                'error': result['error'],
                'params_sim': result['params_sim'],
                'params_scaled': result.get('params_scaled', {}),
                'timestamp': result.get('timestamp', ''),
                'file_path': results_file
            }
            data.append(entry)
            
        except Exception as e:
            print(f"Error reading {results_file}: {e}")
            continue
    
    if len(data) == 0:
        print("No valid data parsed")
        return None
    
    df = pd.DataFrame(data)
    param_names = list(data[0]['params_sim'].keys())
    
    for param in param_names:
        df[param] = df['params_sim'].apply(lambda x: x.get(param, np.nan))
        if param in data[0].get('params_scaled', {}):
            df[f'{param}_scaled'] = df['params_scaled'].apply(lambda x: x.get(param, np.nan))
    
    print(f"Parsed {len(df)} evaluations")
    print(f"Iterations: {df['iteration'].min()} to {df['iteration'].max()}")
    print(f"Parameters: {param_names}")
    
    return df, param_names


def plot_convergence(df, pso_results, output_dir):
    """Plot best error over time using PySwarms if available."""
    print("\nGenerating convergence plot...")
    
    if pso_results and 'cost_history' in pso_results:
        cost_history = pso_results['cost_history']
        
        fig, ax = plt.subplots(figsize=(10, 6))
        plot_cost_history(cost_history, ax=ax, 
                         title='PSO Convergence',
                         designer=Designer(limits=(0, None, 0, 250),
                                         labels=('Iteration', 'Error')))
        
        current_best = cost_history[-1] if len(cost_history) > 0 else float('inf')
        ax.text(0.98, 0.98, f'Current Best: {current_best:.3f}',
                transform=ax.transAxes, ha='right', va='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'convergence_plot.pdf'), dpi=300)
        plt.savefig(os.path.join(output_dir, 'convergence_plot.png'), dpi=150)
        plt.close()
    else:
        convergence = df.groupby('iteration')['error'].min().sort_index()
        iterations = convergence.index.values
        best_errors = convergence.values
        mean_errors = df.groupby('iteration')['error'].mean().sort_index().values
        
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(iterations, best_errors, 'b-o', linewidth=2, markersize=4, label='Best')
        ax.plot(iterations, mean_errors, 'r--', linewidth=1, alpha=0.6, label='Mean')
        ax.set_xlabel('Iteration', fontsize=12)
        ax.set_ylabel('Error', fontsize=12)
        ax.set_title('PSO Convergence', fontsize=14)
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        current_best = best_errors[-1]
        ax.text(0.98, 0.98, f'Current Best: {current_best:.3f}',
                transform=ax.transAxes, ha='right', va='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'convergence_plot.pdf'), dpi=300)
        plt.savefig(os.path.join(output_dir, 'convergence_plot.png'), dpi=150)
        plt.close()
    

def animate_particle_trajectories_pyswarms(pso_results, df, param_names, output_dir):
    """Animate particle trajectories using pos_history from PSO results."""
    print("\nGenerating trajectory animation from PSO history...")
    
    if not pso_results or 'pos_history' not in pso_results:
        print("No pos_history found in PSO results")
        return False
    
    pos_history = pso_results['pos_history']
    cost_history = pso_results.get('cost_history', [])
    
    if 'KDIV' in param_names:
        x_param = 'KDIV'
        x_idx = param_names.index('KDIV')
    elif 'KDEATH' in param_names:
        x_param = 'KDEATH'
        x_idx = param_names.index('KDEATH')
    else:
        print("Neither KDIV nor KDEATH found")
        return False
    
    y_param = 'MIGRATION_FRACTION'
    y_idx = param_names.index('MIGRATION_FRACTION')
    
    from param_optimization_pso import SIM_BOUNDS
    
    x_min, x_max = SIM_BOUNDS[x_param]
    y_min, y_max = SIM_BOUNDS[y_param]
    
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_xlabel(x_param, fontsize=12)
    ax.set_ylabel(y_param, fontsize=12)
    
    n_particles = pos_history[0].shape[0] if len(pos_history) > 0 else 0
    
    errors_by_iter = []
    for i in range(len(pos_history)):
        iter_data = df[df['iteration'] == i]
        if len(iter_data) > 0:
            errors = iter_data.sort_values('particle_idx')['error'].values
            errors_by_iter.append(errors)
        else:
            errors_by_iter.append(np.full(n_particles, np.nan))
    
    # Standardized error colorbar: 0 to 150
    norm = Normalize(vmin=0, vmax=100)
    cmap = plt.cm.viridis
    
    sm = ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax)
    cbar.set_label('Error', fontsize=12)
    
    scatter = ax.scatter([], [], c=[], s=100, alpha=0.7, cmap=cmap, 
                        norm=norm, edgecolors='black', linewidth=0.5)
    trails = [ax.plot([], [], 'k-', alpha=0.2, linewidth=0.5)[0] for _ in range(n_particles)]
    title = ax.text(0.5, 1.05, '', transform=ax.transAxes, 
                   ha='center', fontsize=14, weight='bold')
    best_text = ax.text(0.02, 0.98, '', transform=ax.transAxes,
                       ha='left', va='top', fontsize=10,
                       bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))
    
    particle_histories = {i: {'x': [], 'y': []} for i in range(n_particles)}
    
    def init():
        scatter.set_offsets(np.empty((0, 2)))
        scatter.set_array(np.array([]))
        for trail in trails:
            trail.set_data([], [])
        title.set_text('')
        best_text.set_text('')
        return [scatter] + trails + [title, best_text]
    
    def update(frame):
        if frame >= len(pos_history):
            return [scatter] + trails + [title, best_text]
        
        positions_scaled = pos_history[frame]
        
        # Convert to simulation space
        x_scaled = positions_scaled[:, x_idx]
        y_scaled = positions_scaled[:, y_idx]
        x = x_min + x_scaled * (x_max - x_min)
        y = y_min + y_scaled * (y_max - y_min)
        
        errors = errors_by_iter[frame] if frame < len(errors_by_iter) else np.full(n_particles, np.nan)
        
        positions = np.column_stack([x, y])
        scatter.set_offsets(positions)
        scatter.set_array(errors)
        
        for i in range(n_particles):
            particle_histories[i]['x'].append(x[i])
            particle_histories[i]['y'].append(y[i])
            trails[i].set_data(particle_histories[i]['x'], 
                             particle_histories[i]['y'])
        
        best_error = np.nanmin(errors) if len(errors) > 0 else np.inf
        best_overall = cost_history[frame] if frame < len(cost_history) else np.inf
        title.set_text(f'Iteration {frame} / {len(pos_history)-1}')
        best_text.set_text(f'Current Best: {best_error:.3f}\nGlobal Best: {best_overall:.3f}')
        
        return [scatter] + trails + [title, best_text]
    
    anim = animation.FuncAnimation(fig, update, init_func=init,
                                  frames=len(pos_history), interval=200,
                                  blit=True, repeat=True)
    
    anim_path = os.path.join(output_dir, 'particle_trajectories.gif')
    anim.save(anim_path, writer='pillow', fps=5, dpi=100)
    plt.close()
    
    return True


def animate_particle_trajectories_fallback(df, param_names, output_dir):
    """Fallback animation from parsed results.pkl files."""
    print("\nGenerating trajectory animation from results files...")
    
    if 'KDIV' in param_names:
        x_param = 'KDIV'
    elif 'KDEATH' in param_names:
        x_param = 'KDEATH'
    else:
        print("Neither KDIV nor KDEATH found. Skipping animation.")
        return
    
    y_param = 'MIGRATION_FRACTION'
    
    from param_optimization_pso import SIM_BOUNDS
    
    x_min, x_max = SIM_BOUNDS[x_param]
    y_min, y_max = SIM_BOUNDS[y_param]
    
    iterations = sorted(df['iteration'].unique())
    n_particles = df['particle_idx'].nunique()
    
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_xlabel(x_param, fontsize=12)
    ax.set_ylabel(y_param, fontsize=12)
    
    # Standardized error colorbar: 0 to 150
    norm = Normalize(vmin=0, vmax=50) ### 150
    cmap = plt.cm.viridis
    
    sm = ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax)
    cbar.set_label('Error', fontsize=12)
    
    scatter = ax.scatter([], [], c=[], s=100, alpha=0.7, cmap=cmap, 
                        norm=norm, edgecolors='black', linewidth=0.5)
    trails = [ax.plot([], [], 'k-', alpha=0.2, linewidth=0.5)[0] for _ in range(n_particles)]
    title = ax.text(0.5, 1.05, '', transform=ax.transAxes, 
                   ha='center', fontsize=14, weight='bold')
    best_text = ax.text(0.02, 0.98, '', transform=ax.transAxes,
                       ha='left', va='top', fontsize=10,
                       bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))
    
    particle_histories = {i: {'x': [], 'y': []} for i in range(n_particles)}
    
    def init():
        scatter.set_offsets(np.empty((0, 2)))
        scatter.set_array(np.array([]))
        for trail in trails:
            trail.set_data([], [])
        return [scatter] + trails + [title, best_text]
    
    def update(frame):
        iter_num = iterations[frame]
        iter_data = df[df['iteration'] == iter_num].sort_values('particle_idx')
        
        if len(iter_data) == 0:
            return [scatter] + trails + [title, best_text]
        
        x = iter_data[x_param].values
        y = iter_data[y_param].values
        errors = iter_data['error'].values
        particle_indices = iter_data['particle_idx'].values
        
        positions = np.column_stack([x, y])
        scatter.set_offsets(positions)
        scatter.set_array(errors)
        
        for px, py, pidx in zip(x, y, particle_indices):
            particle_histories[pidx]['x'].append(px)
            particle_histories[pidx]['y'].append(py)
            trails[pidx].set_data(particle_histories[pidx]['x'], 
                                 particle_histories[pidx]['y'])
        
        best_error = errors.min()
        best_overall = df[df['iteration'] <= iter_num]['error'].min()
        title.set_text(f'Iteration {iter_num} / {iterations[-1]}')
        best_text.set_text(f'Current Best: {best_error:.3f}\nGlobal Best: {best_overall:.3f}')
        
        return [scatter] + trails + [title, best_text]
    
    anim = animation.FuncAnimation(fig, update, init_func=init,
                                  frames=len(iterations), interval=200,
                                  blit=True, repeat=True)
    
    anim_path = os.path.join(output_dir, 'particle_trajectories.gif')
    anim.save(anim_path, writer='pillow', fps=5, dpi=100)
    plt.close()
    


def create_summary(df, param_names, run_dir, output_dir):
    """Create comprehensive summary for warm start."""
    
    pso_results_file = os.path.join(run_dir, 'pso_results.pkl')
    pso_results = None
    if os.path.exists(pso_results_file):
        try:
            with open(pso_results_file, 'rb') as f:
                pso_results = pickle.load(f)
            print("Loaded pso_results.pkl")
        except:
            print("Could not load pso_results.pkl")
    
    iterations = sorted(df['iteration'].unique())
    
    summary = {
        'metadata': {
            'run_dir': run_dir,
            'analysis_timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'param_names': param_names,
            'n_iterations_completed': len(iterations),
            'n_particles': df['particle_idx'].nunique(),
            'iteration_range': (int(iterations[0]), int(iterations[-1])),
        },
        'convergence': {
            'iterations': iterations,
            'best_errors': df.groupby('iteration')['error'].min().to_dict(),
            'mean_errors': df.groupby('iteration')['error'].mean().to_dict(),
            'std_errors': df.groupby('iteration')['error'].std().to_dict(),
        },
        'best_solution': {
            'error': float(df['error'].min()),
            'iteration': int(df.loc[df['error'].idxmin(), 'iteration']),
            'particle_idx': int(df.loc[df['error'].idxmin(), 'particle_idx']),
            'params_sim': df.loc[df['error'].idxmin(), 'params_sim'],
            'params_scaled': df.loc[df['error'].idxmin()].get('params_scaled', {}),
        },
        'iteration_data': {},
        'pso_config': pso_results,
    }
    
    for iter_num in iterations:
        iter_data = df[df['iteration'] == iter_num].sort_values('particle_idx')
        
        summary['iteration_data'][int(iter_num)] = {
            'n_particles': len(iter_data),
            'best_error': float(iter_data['error'].min()),
            'mean_error': float(iter_data['error'].mean()),
            'particles': []
        }
        
        for _, row in iter_data.iterrows():
            particle_info = {
                'particle_idx': int(row['particle_idx']),
                'error': float(row['error']),
                'params_sim': row['params_sim'],
                'params_scaled': row.get('params_scaled', {}),
                'file_path': row['file_path'],
            }
            summary['iteration_data'][int(iter_num)]['particles'].append(particle_info)
    
    summary_file = os.path.join(output_dir, 'summary.pkl')
    with open(summary_file, 'wb') as f:
        pickle.dump(summary, f)
    
    import json
    json_summary = {
        'metadata': summary['metadata'],
        'convergence': {
            'iterations': [int(i) for i in summary['convergence']['iterations']],
            'best_errors': {int(k): float(v) for k, v in summary['convergence']['best_errors'].items()},
            'mean_errors': {int(k): float(v) for k, v in summary['convergence']['mean_errors'].items()},
        },
        'best_solution': summary['best_solution'],
    }
    
    json_file = os.path.join(output_dir, 'summary.json')
    with open(json_file, 'w') as f:
        json.dump(json_summary, f, indent=2)
    
    csv_file = os.path.join(output_dir, 'all_evaluations.csv')
    df_export = df[['iteration', 'particle_idx', 'error'] + param_names].copy()
    df_export.to_csv(csv_file, index=False)
    
    return summary


def analyze_pso_run(run_dir):
    """Main analysis function."""
    
    print("="*70)
    print("PSO Run Analysis")
    print("="*70)
    
    analysis_dir = os.path.join(run_dir, 'analysis')
    os.makedirs(analysis_dir, exist_ok=True)
    print(f"Analysis output: {analysis_dir}")
    
    result = parse_run_directory(run_dir)
    if result is None:
        print("Failed to parse run directory")
        return
    
    df, param_names = result
    
    pso_results_file = os.path.join(run_dir, 'pso_results.pkl')
    pso_results = None
    if os.path.exists(pso_results_file):
        try:
            with open(pso_results_file, 'rb') as f:
                pso_results = pickle.load(f)
        except:
            pass
    
    plot_convergence(df, pso_results, analysis_dir)
    
    if pso_results:
        success = animate_particle_trajectories_pyswarms(pso_results, df, param_names, analysis_dir)
        if not success:
            animate_particle_trajectories_fallback(df, param_names, analysis_dir)
    else:
        animate_particle_trajectories_fallback(df, param_names, analysis_dir)
    
    summary = create_summary(df, param_names, run_dir, analysis_dir)
    
    print("\n" + "="*70)
    print("Analysis Complete...")
    print("="*70)
    print(f"\nBest error found: {summary['best_solution']['error']:.3f}")
    print(f"At iteration: {summary['best_solution']['iteration']}")
    print(f"Best parameters (sim):")
    for param, value in summary['best_solution']['params_sim'].items():
        print(f"  {param}: {value:.4f}")
    print(f"\nAll results saved to: {analysis_dir}")


if __name__ == '__main__':
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python analyze_pso_run.py <path_to_run_directory>")
        sys.exit(1)
    
    run_dir = sys.argv[1]
    
    if not os.path.exists(run_dir):
        print(f"Error: Directory not found: {run_dir}")
        sys.exit(1)
    
    analyze_pso_run(run_dir)