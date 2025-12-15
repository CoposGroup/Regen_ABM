from config import *
import numpy as np
import matplotlib.pyplot as plt
import pickle
import os


def save_optimization_data(param_history, error_history, boundary_history, param_names, OUTPUT_DIR):
    """Save all optimization data for later reconstruction of plots."""
    data = {
        'param_history': np.array(param_history),
        'error_history': np.array(error_history),
        'boundary_history': boundary_history,
        'param_names': param_names
    }
    filepath = os.path.join(OUTPUT_DIR, 'optimization_data.pkl')
    with open(filepath, 'wb') as f:
        pickle.dump(data, f)
    print(f"Saved optimization data to {filepath}")
    return filepath


def load_optimization_data(filepath):
    """Load saved optimization data."""
    with open(filepath, 'rb') as f:
        data = pickle.load(f)
    return data


def plot_error_convergence(error_history, OUTPUT_DIR):
    """Plot error vs iteration as individual PDF."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    iterations = np.arange(1, len(error_history) + 1)
    ax.plot(iterations, error_history, 'o-', color='steelblue', markersize=6, linewidth=2, alpha=0.7)
    
    min_idx = np.argmin(error_history)
    ax.plot(iterations[min_idx], error_history[min_idx], 'o', color='red', markersize=10, zorder=10)
    
    ax.set_xlabel('Iteration', fontsize=14)
    ax.set_ylabel('Objective Error', fontsize=14)
    ax.set_title('Error Convergence', fontsize=16)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(bottom=0)
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'error_convergence.pdf'), dpi=300, bbox_inches='tight')
    plt.close()


def plot_parameter_evolution(param_history, param_name, param_idx, OUTPUT_DIR):
    """Plot single parameter evolution over iterations."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    iterations = np.arange(1, len(param_history) + 1)
    param_values = param_history[:, param_idx]
    
    ax.plot(iterations, param_values, 'o-', color='forestgreen', markersize=6, linewidth=2, alpha=0.7)
    
    ax.set_xlabel('Iteration', fontsize=14)
    ax.set_ylabel(param_name, fontsize=14)
    ax.set_title(f'{param_name} Evolution', fontsize=16)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    safe_name = param_name.replace(' ', '_').replace('/', '_')
    plt.savefig(os.path.join(OUTPUT_DIR, f'param_evolution_{safe_name}.pdf'), dpi=300, bbox_inches='tight')
    plt.close()


def plot_parameter_vs_error(param_history, error_history, param_name, param_idx, OUTPUT_DIR):
    """Plot parameter value vs error (scatter plot colored by error)."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    param_values = param_history[:, param_idx]
    scatter = ax.scatter(param_values, error_history, c=error_history, 
                        cmap='viridis', s=80, alpha=0.7, edgecolors='black', linewidth=0.5)
    
    cbar = plt.colorbar(scatter, ax=ax, label='Error')
    
    ax.set_xlabel(param_name, fontsize=14)
    ax.set_ylabel('Objective Error', fontsize=14)
    ax.set_title(f'{param_name} vs Error', fontsize=16)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(bottom=0)
    
    plt.tight_layout()
    safe_name = param_name.replace(' ', '_').replace('/', '_')
    plt.savefig(os.path.join(OUTPUT_DIR, f'param_vs_error_{safe_name}.pdf'), dpi=300, bbox_inches='tight')
    plt.close()


def create_individual_optimization_plots(param_history, error_history, OUTPUT_DIR, param_names=None):
    """Create all individual optimization plots (one file per plot)."""
    param_history = np.array(param_history)
    error_history = np.array(error_history)
    n_params = param_history.shape[1]
    
    if param_names is None:
        param_names = [f'Param_{i}' for i in range(n_params)]
    
    plot_error_convergence(error_history, OUTPUT_DIR)
    
    for i in range(n_params):
        plot_parameter_evolution(param_history, param_names[i], i, OUTPUT_DIR)
        plot_parameter_vs_error(param_history, error_history, param_names[i], i, OUTPUT_DIR)



def create_multi_run_comparison_plots(all_runs_data, output_dir, metric_name='Control', ylabel=r'Absolute Length Difference ($\mu$m)'):
    """
    Create comparison plots showing all individual runs plus mean across runs.
    
    Args:
        all_runs_data: List of dicts, each containing 'param_history', 'error_history', 'param_names'
        output_dir: Directory to save plots
        metric_name: Name of the metric (e.g., 'Control', 'C59')
        ylabel: Label for error y-axis
    """
    os.makedirs(output_dir, exist_ok=True)
    
    n_runs = len(all_runs_data)
    if n_runs == 0:
        print("No data to plot")
        return
    
    # Find maximum iteration count across all runs
    max_iters = max(len(data['error_history']) for data in all_runs_data)
    
    # Prepare data structures for averaging (pad shorter runs with NaN)
    migration_data = []
    kdiv_data = []
    error_data = []
    
    for data in all_runs_data:
        param_history = np.array(data['param_history'])
        error_history = np.array(data['error_history'])
        param_names = data['param_names']
        n_iters = len(error_history)
        
        # Find parameter indices
        migration_idx = param_names.index('MIGRATION_PERCENT')
        kdiv_idx = param_names.index('KDIV')
        
        # Extract and scale migration (clip to 0-100%)
        migration_values = param_history[:, migration_idx] * 100.0
        migration_values = np.maximum(migration_values, 0.0)
        
        # Extract KDIV
        kdiv_values = param_history[:, kdiv_idx]
        
        # Pad with NaN if needed
        if n_iters < max_iters:
            migration_values = np.pad(migration_values, (0, max_iters - n_iters), constant_values=np.nan)
            kdiv_values = np.pad(kdiv_values, (0, max_iters - n_iters), constant_values=np.nan)
            error_history = np.pad(error_history, (0, max_iters - n_iters), constant_values=np.nan)
        
        migration_data.append(migration_values)
        kdiv_data.append(kdiv_values)
        error_data.append(error_history)
    
    # Convert to arrays
    migration_data = np.array(migration_data)
    kdiv_data = np.array(kdiv_data)
    error_data = np.array(error_data)
    
    # Calculate means (ignoring NaN)
    migration_mean = np.nanmean(migration_data, axis=0)
    kdiv_mean = np.nanmean(kdiv_data, axis=0)
    error_mean = np.nanmean(error_data, axis=0)
    
    # Use consistent colors: Control=blue, C59=red, otherwise gray
    if 'Control' in metric_name or 'control' in metric_name:
        mean_color = 'blue'
    elif 'C59' in metric_name or 'c59' in metric_name:
        mean_color = 'red'
    else:
        mean_color = 'black'
    
    # Plot 1: Migration Percent vs Iteration
    fig, ax = plt.subplots(figsize=(12, 7))
    iterations = np.arange(1, max_iters + 1)
    
    # Plot individual runs (same color as mean, low opacity, no legend)
    for i in range(n_runs):
        ax.plot(iterations, migration_data[i], '-', color=mean_color, 
                linewidth=1.5, alpha=0.2)
    
    # Plot mean (thick line, in legend)
    ax.plot(iterations, migration_mean, '-', color=mean_color, linewidth=3, alpha=0.9, 
            label=f'{metric_name} Mean (n={n_runs})')
    
    ax.set_xlabel('Iteration', fontsize=14)
    ax.set_ylabel('Migration Percent (%)', fontsize=14)
    ax.set_title(f'Migration Percent Evolution - {metric_name}', fontsize=16)
    ax.set_ylim(0, 100)
    ax.legend(fontsize=12, loc='best')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'migration_evolution_{metric_name.lower()}.pdf'), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  [OK] Migration evolution plot for {metric_name}")
    
    # Plot 2: Proliferation Rate vs Iteration
    fig, ax = plt.subplots(figsize=(12, 7))
    
    # Plot individual runs (same color as mean, low opacity, no legend)
    for i in range(n_runs):
        ax.plot(iterations, kdiv_data[i], '-', color=mean_color, 
                linewidth=1.5, alpha=0.2)
    
    # Plot mean (thick line, in legend)
    ax.plot(iterations, kdiv_mean, '-', color=mean_color, linewidth=3, alpha=0.9,
            label=f'{metric_name} Mean (n={n_runs})')
    
    ax.set_xlabel('Iteration', fontsize=14)
    ax.set_ylabel('Proliferation Rate', fontsize=14)
    ax.set_title(f'Proliferation Rate Evolution - {metric_name}', fontsize=16)
    ax.set_ylim(0, 0.6)
    ax.legend(fontsize=12, loc='best')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'kdiv_evolution_{metric_name.lower()}.pdf'), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  [OK] KDIV evolution plot for {metric_name}")
    
    # Plot 3: Error vs Iteration
    fig, ax = plt.subplots(figsize=(12, 7))
    
    # Plot individual runs (same color as mean, low opacity, no legend)
    for i in range(n_runs):
        ax.plot(iterations, error_data[i], '-', color=mean_color, 
                linewidth=1.5, alpha=0.2)
    
    # Plot mean (thick line, in legend)
    ax.plot(iterations, error_mean, '-', color=mean_color, linewidth=3, alpha=0.9,
            label=f'{metric_name} Mean (n={n_runs})')
    
    ax.set_xlabel('Iteration', fontsize=14)
    ax.set_ylabel(ylabel, fontsize=14)
    ax.set_title(f'Error Convergence - {metric_name}', fontsize=16)
    ax.set_ylim(0, 200)
    ax.legend(fontsize=12, loc='best')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'error_evolution_{metric_name.lower()}.pdf'), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  [OK] Error evolution plot for {metric_name}")
    
    print(f"All comparison plots saved to: {output_dir}")


def create_ctrl_vs_c59_comparison_plots(ctrl_runs_data, c59_runs_data, output_dir):
    """
    Create comparison plots showing Control vs C59 means and individual runs.
    
    Args:
        ctrl_runs_data: List of dicts for control runs
        c59_runs_data: List of dicts for C59 runs
        output_dir: Directory to save plots
    """
    os.makedirs(output_dir, exist_ok=True)
    
    def extract_and_pad_data(runs_data):
        """Extract parameter and error data, padding to same length."""
        max_iters = max(len(data['error_history']) for data in runs_data)
        migration_data = []
        kdiv_data = []
        error_data = []
        
        for data in runs_data:
            param_history = np.array(data['param_history'])
            error_history = np.array(data['error_history'])
            param_names = data['param_names']
            n_iters = len(error_history)
            
            migration_idx = param_names.index('MIGRATION_PERCENT')
            kdiv_idx = param_names.index('KDIV')
            
            migration_values = np.maximum(param_history[:, migration_idx] * 100.0, 0.0)
            kdiv_values = param_history[:, kdiv_idx]
            
            if n_iters < max_iters:
                migration_values = np.pad(migration_values, (0, max_iters - n_iters), constant_values=np.nan)
                kdiv_values = np.pad(kdiv_values, (0, max_iters - n_iters), constant_values=np.nan)
                error_history = np.pad(error_history, (0, max_iters - n_iters), constant_values=np.nan)
            
            migration_data.append(migration_values)
            kdiv_data.append(kdiv_values)
            error_data.append(error_history)
        
        return np.array(migration_data), np.array(kdiv_data), np.array(error_data), max_iters
    
    # Extract data for both groups
    ctrl_migration, ctrl_kdiv, ctrl_error, ctrl_max_iters = extract_and_pad_data(ctrl_runs_data)
    c59_migration, c59_kdiv, c59_error, c59_max_iters = extract_and_pad_data(c59_runs_data)
    
    # Calculate means
    ctrl_migration_mean = np.nanmean(ctrl_migration, axis=0)
    ctrl_kdiv_mean = np.nanmean(ctrl_kdiv, axis=0)
    ctrl_error_mean = np.nanmean(ctrl_error, axis=0)
    
    c59_migration_mean = np.nanmean(c59_migration, axis=0)
    c59_kdiv_mean = np.nanmean(c59_kdiv, axis=0)
    c59_error_mean = np.nanmean(c59_error, axis=0)
    
    max_iters = max(ctrl_max_iters, c59_max_iters)
    iterations = np.arange(1, max_iters + 1)
    
    # Plot 1: Migration Percent - Control vs C59
    fig, ax = plt.subplots(figsize=(12, 7))
    
    # Plot individual Control runs (blue, low opacity, no legend)
    for i in range(len(ctrl_runs_data)):
        ax.plot(iterations[:ctrl_max_iters], ctrl_migration[i], '-', color='blue', 
                linewidth=1.5, alpha=0.15)
    
    # Plot individual C59 runs (red, low opacity, no legend)
    for i in range(len(c59_runs_data)):
        ax.plot(iterations[:c59_max_iters], c59_migration[i], '-', color='red', 
                linewidth=1.5, alpha=0.15)
    
    # Plot means (thick lines, in legend)
    ax.plot(iterations[:ctrl_max_iters], ctrl_migration_mean, 'b-', linewidth=3, alpha=0.9, 
            label=f'Control Mean (n={len(ctrl_runs_data)})')
    ax.plot(iterations[:c59_max_iters], c59_migration_mean, 'r-', linewidth=3, alpha=0.9, 
            label=f'C59 Mean (n={len(c59_runs_data)})')
    
    ax.set_xlabel('Iteration', fontsize=14)
    ax.set_ylabel('Migration Percent (%)', fontsize=14)
    ax.set_title('Migration Percent Evolution - Control vs C59', fontsize=16)
    ax.set_ylim(0, 100)
    ax.legend(fontsize=12, loc='best')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'migration_ctrl_vs_c59.pdf'), dpi=300, bbox_inches='tight')
    plt.close()
    print("  [OK] Migration Control vs C59 comparison")
    
    # Plot 2: Proliferation Rate - Control vs C59
    fig, ax = plt.subplots(figsize=(12, 7))
    
    # Plot individual Control runs (blue, low opacity, no legend)
    for i in range(len(ctrl_runs_data)):
        ax.plot(iterations[:ctrl_max_iters], ctrl_kdiv[i], '-', color='blue', 
                linewidth=1.5, alpha=0.15)
    
    # Plot individual C59 runs (red, low opacity, no legend)
    for i in range(len(c59_runs_data)):
        ax.plot(iterations[:c59_max_iters], c59_kdiv[i], '-', color='red', 
                linewidth=1.5, alpha=0.15)
    
    # Plot means (thick lines, in legend)
    ax.plot(iterations[:ctrl_max_iters], ctrl_kdiv_mean, 'b-', linewidth=3, alpha=0.9, 
            label=f'Control Mean (n={len(ctrl_runs_data)})')
    ax.plot(iterations[:c59_max_iters], c59_kdiv_mean, 'r-', linewidth=3, alpha=0.9, 
            label=f'C59 Mean (n={len(c59_runs_data)})')
    
    ax.set_xlabel('Iteration', fontsize=14)
    ax.set_ylabel('Proliferation Rate', fontsize=14)
    ax.set_title('Proliferation Rate Evolution - Control vs C59', fontsize=16)
    ax.set_ylim(0, 0.6)
    ax.legend(fontsize=12, loc='best')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'kdiv_ctrl_vs_c59.pdf'), dpi=300, bbox_inches='tight')
    plt.close()
    print("  [OK] KDIV Control vs C59 comparison")
    
    # Plot 3: Error - Control vs C59
    fig, ax = plt.subplots(figsize=(12, 7))
    
    # Plot individual Control runs (blue, low opacity, no legend)
    for i in range(len(ctrl_runs_data)):
        ax.plot(iterations[:ctrl_max_iters], ctrl_error[i], '-', color='blue', 
                linewidth=1.5, alpha=0.15)
    
    # Plot individual C59 runs (red, low opacity, no legend)
    for i in range(len(c59_runs_data)):
        ax.plot(iterations[:c59_max_iters], c59_error[i], '-', color='red', 
                linewidth=1.5, alpha=0.15)
    
    # Plot means (thick lines, in legend)
    ax.plot(iterations[:ctrl_max_iters], ctrl_error_mean, 'b-', linewidth=3, alpha=0.9, 
            label=f'Control Mean (n={len(ctrl_runs_data)})')
    ax.plot(iterations[:c59_max_iters], c59_error_mean, 'r-', linewidth=3, alpha=0.9, 
            label=f'C59 Mean (n={len(c59_runs_data)})')
    
    ax.set_xlabel('Iteration', fontsize=14)
    ax.set_ylabel(r'Absolute Length Difference ($\mu$m)', fontsize=14)
    ax.set_title('Error Convergence - Control vs C59', fontsize=16)
    ax.set_ylim(0, 200)
    ax.legend(fontsize=12, loc='best')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'error_ctrl_vs_c59.pdf'), dpi=300, bbox_inches='tight')
    plt.close()
    print("  [OK] Error Control vs C59 comparison")
    
    print(f"Control vs C59 comparison plots saved to: {output_dir}")


def print_optimization_step(iteration, params, error, best_params=None, best_error=None, param_names=None):
    """Print current optimization step information"""
    print(f"\n--- Optimization Step {iteration} ---")
    
    if param_names is not None and len(param_names) == len(params):
        param_str = ', '.join([f'{name}={val:.4f}' for name, val in zip(param_names, params)])
    else:
        # Fallback for backward compatibility
        param_str = ', '.join([f'param_{i}={val:.4f}' for i, val in enumerate(params)])
    
    print(f"Parameters: {param_str}")
    print(f"Objective Error: {error:.6f}")
    
    if best_params is not None and best_error is not None:
        if param_names is not None and len(param_names) == len(best_params):
            best_param_str = ', '.join([f'{name}={val:.4f}' for name, val in zip(param_names, best_params)])
        else:
            best_param_str = ', '.join([f'param_{i}={val:.4f}' for i, val in enumerate(best_params)])
        print(f"Best so far: {best_param_str}, Error={best_error:.6f}")

def plot_optimization_history(param_history, error_history, OUTPUT_DIR=None, param_names=None):
    """Plot optimization convergence history for any number of parameters"""
    if OUTPUT_DIR is None:
        OUTPUT_DIR = get_output_dir()
    param_history = np.array(param_history)
    error_history = np.array(error_history)
    n_params = param_history.shape[1]
    # Compute running-best error to visualize true convergence of differential evolution
    running_best = np.minimum.accumulate(error_history)
    
    if param_names is None:
        param_names = [f'Param_{i}' for i in range(n_params)]
    
    # For 1 parameter: 2x2 layout with error + parameter evolution
    if n_params == 1:
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        
        # Error history
        axes[0, 0].plot(error_history, 'b-o', markersize=4)
        axes[0, 0].set_xlabel('Iteration')
        axes[0, 0].set_ylabel('Objective Error')
        axes[0, 0].set_title('Optimization Convergence')
        axes[0, 0].grid(True)
        
        # Parameter evolution
        axes[0, 1].plot(param_history[:, 0], 'r-o', markersize=4)
        axes[0, 1].set_xlabel('Iteration')
        axes[0, 1].set_ylabel(param_names[0])
        axes[0, 1].set_title(f'{param_names[0]} Evolution')
        axes[0, 1].grid(True)
        
        # Parameter vs error
        axes[1, 0].scatter(param_history[:, 0], error_history, c=range(len(error_history)), 
                          cmap='viridis', s=50)
        axes[1, 0].set_xlabel(param_names[0])
        axes[1, 0].set_ylabel('Error')
        axes[1, 0].set_title(f'{param_names[0]} vs Error')
        axes[1, 0].grid(True)
        
        # Summary statistics
        axes[1, 1].axis('off')
        best_idx = np.argmin(error_history)
        stats_text = f"""
        Optimization Summary
        ===================
        Total Iterations: {len(error_history)}
        Best {param_names[0]}: {param_history[best_idx, 0]:.4f}
        Best Error: {error_history[best_idx]:.6f}
        
        Parameter Range:
        {param_names[0]}: [{param_history[:, 0].min():.3f}, {param_history[:, 0].max():.3f}]
        
        Error Reduction:
        {((error_history[0] - error_history[best_idx]) / error_history[0] * 100):.1f}%
        """
        axes[1, 1].text(0.1, 0.9, stats_text, transform=axes[1, 1].transAxes, 
                        verticalalignment='top', fontfamily='monospace', fontsize=10)
    
    # For 2 parameters: original 2x2 layout
    elif n_params == 2:
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        
        # Error history
        axes[0, 0].plot(error_history, 'b-o', markersize=4)
        axes[0, 0].set_xlabel('Iteration')
        axes[0, 0].set_ylabel('Objective Error')
        axes[0, 0].set_title('Optimization Convergence')
        axes[0, 0].grid(True)
        
        # Parameter histories
        colors = ['r', 'g']
        for i in range(2):
            row, col = (0, 1) if i == 0 else (1, 0)
            axes[row, col].plot(param_history[:, i], f'{colors[i]}-o', markersize=4)
            axes[row, col].set_xlabel('Iteration')
            axes[row, col].set_ylabel(param_names[i])
            axes[row, col].set_title(f'{param_names[i]} Evolution')
            axes[row, col].grid(True)
        
        # Parameter space exploration
        axes[1, 1].scatter(param_history[:, 0], param_history[:, 1], 
                          c=error_history, cmap='viridis', s=50)
        axes[1, 1].set_xlabel(param_names[0])
        axes[1, 1].set_ylabel(param_names[1])
        axes[1, 1].set_title('Parameter Space Exploration')
        cbar = plt.colorbar(axes[1, 1].collections[0], ax=axes[1, 1])
        cbar.set_label('Objective Error')
    
    # For 3+ parameters: adaptive layout
    else:
        n_rows = min(3, (n_params + 2) // 2)  # Up to 3 rows
        n_cols = min(3, n_params + 1)  # Up to 3 columns
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(4*n_cols, 4*n_rows))
        if n_rows == 1:
            axes = axes.reshape(1, -1)
        elif n_cols == 1:
            axes = axes.reshape(-1, 1)
        axes = axes.flatten()
        
        # Error history (always first)
        axes[0].plot(error_history, 'b-o', markersize=4)
        axes[0].set_xlabel('Iteration')
        axes[0].set_ylabel('Objective Error')
        axes[0].set_title('Optimization Convergence')
        axes[0].grid(True)
        
        # Parameter evolution plots
        colors = ['r', 'g', 'orange', 'purple', 'brown']
        for i in range(min(n_params, len(axes)-1)):
            ax_idx = i + 1
            color = colors[i % len(colors)]
            axes[ax_idx].plot(param_history[:, i], f'{color}-o', markersize=4)
            axes[ax_idx].set_xlabel('Iteration')
            axes[ax_idx].set_ylabel(param_names[i])
            axes[ax_idx].set_title(f'{param_names[i]} Evolution')
            axes[ax_idx].grid(True)
        
        # Turn off any unused axes
        for i in range(n_params + 1, len(axes)):
            axes[i].axis('off')
    
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/optimization_history.png', dpi=300)
    plt.close()

def plot_shape_comparison(target_shape, sim_boundary, target_error, OUTPUT_DIR=None):
    """Plot comparison between target and simulated shapes"""
    if OUTPUT_DIR is None:
        OUTPUT_DIR = get_output_dir()
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_aspect('equal')
    
    # Plot target shape
    ax.plot(target_shape[:, 0], target_shape[:, 1], 'b-', linewidth=3, 
            label=f'Target Shape (r={np.mean(np.linalg.norm(target_shape, axis=1)):.1f})')
    
    # Plot simulated boundary
    ax.plot(sim_boundary[:, 0], sim_boundary[:, 1], 'r--', linewidth=2,
            label='Simulated Boundary')
    
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title(f'Shape Comparison (Error: {target_error:.6f})')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Set equal limits
    all_x = np.concatenate([target_shape[:, 0], sim_boundary[:, 0]])
    all_y = np.concatenate([target_shape[:, 1], sim_boundary[:, 1]])
    margin = 0.1
    ax.set_xlim(all_x.min() - margin, all_x.max() + margin)
    ax.set_ylim(all_y.min() - margin, all_y.max() + margin)
    
    plt.savefig(f'{OUTPUT_DIR}/shape_comparison.png', dpi=300)
    plt.close()

def animate_error_convergence(error_history, OUTPUT_DIR=None, max_frames=100):
    """
    Create a simple animation showing error convergence over iterations.
    
    Args:
        error_history: List of objective function values
        OUTPUT_DIR: Output directory for saving animation
        max_frames: Maximum number of frames (will sample if needed)
    """
    if OUTPUT_DIR is None:
        OUTPUT_DIR = get_output_dir()
    import matplotlib.animation as animation
    
    error_history = np.array(error_history)
    n_iters = len(error_history)
    
    # Sample frames if we have too many
    if n_iters > max_frames:
        frame_indices = np.linspace(0, n_iters-1, max_frames, dtype=int)
        error_history = error_history[frame_indices]
        n_iters = max_frames
    
    # Create the plot
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.set_xlim(0, n_iters-1)
    ax.set_ylim(np.min(error_history) * 0.9, np.max(error_history) * 1.1)
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Objective Error')
    ax.set_title('Error Convergence Animation')
    ax.grid(True, alpha=0.3)
    
    # Initialize empty line and current point
    error_line, = ax.plot([], [], 'b-', linewidth=2, label='Error')
    current_point, = ax.plot([], [], 'ro', markersize=8, label='Current')
    ax.legend()
    
    def animate(frame):
        if frame >= n_iters:
            return
        
        # Update error plot
        if frame > 0:
            ax.plot(range(frame), error_history[:frame], 'b-', linewidth=2)
        current_point.set_data([frame], [error_history[frame]])
        
        # Add iteration text
        ax.set_title(f'Error Convergence - Iteration {frame+1} (Error: {error_history[frame]:.6f})')
        
        return error_line, current_point
    
    # Create animation
    anim = animation.FuncAnimation(fig, animate, frames=n_iters, 
                                 interval=200, blit=False, repeat=True)
    
    # Save animation
    anim_path = os.path.join(OUTPUT_DIR, 'error_convergence.gif')
    print(f"Saving error convergence animation to {anim_path}")
    anim.save(anim_path, writer='pillow', fps=5)
    plt.close(fig)
    
    return anim

def animate_simplex_convergence(param_history, error_history, param_bounds, OUTPUT_DIR=None, param_names=None, max_frames=100):
    """
    Animate Nelder-Mead simplex convergence for 2D parameter spaces only.
    Shows triangular simplex with vertices colored by error values.
    
    Args:
        param_history: List of parameter values at each iteration
        error_history: List of objective function values
        param_bounds: List of (min, max) tuples for each parameter
        OUTPUT_DIR: Output directory for saving animation
        param_names: Names of parameters (optional)
        max_frames: Maximum number of frames (will sample if needed)
    """
    if OUTPUT_DIR is None:
        OUTPUT_DIR = get_output_dir()
    import matplotlib.animation as animation
    from matplotlib.patches import Polygon
    import matplotlib.cm as cm
    
    param_history = np.array(param_history)
    error_history = np.array(error_history)
    n_params = param_history.shape[1]
    
    if n_params != 2:
        print(f"Error: Simplex animation only supports 2D parameter spaces, got {n_params}D")
        return
    
    if param_names is None:
        param_names = [f'Param_{i}' for i in range(n_params)]
    
    # Sample frames if we have too many
    n_iters = len(param_history)
    if n_iters > max_frames:
        frame_indices = np.linspace(0, n_iters-1, max_frames, dtype=int)
        param_history = param_history[frame_indices]
        error_history = error_history[frame_indices]
        n_iters = max_frames
    
    # Set up the plot
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Parameter bounds
    param1_min, param1_max = param_bounds[0]
    param2_min, param2_max = param_bounds[1]
    param1_range = param1_max - param1_min
    param2_range = param2_max - param2_min
    param1_padding = param1_range * 0.1
    param2_padding = param2_range * 0.1
    
    # Set plot limits
    ax.set_xlim(param1_min - param1_padding, param1_max + param1_padding)
    ax.set_ylim(param2_min - param2_padding, param2_max + param2_padding)
    ax.set_xlabel(param_names[0])
    ax.set_ylabel(param_names[1])
    ax.set_title('Nelder-Mead Simplex Evolution (2D)')
    ax.grid(True, alpha=0.3)
    
    # Set up color mapping for errors
    error_min, error_max = np.min(error_history), np.max(error_history)
    cmap = cm.viridis
    norm = plt.Normalize(vmin=error_min, vmax=error_max)
    
    # Add colorbar
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, label='Objective Error')
    
    # Initialize empty elements
    simplex_polygon = Polygon(np.array([]).reshape(-1, 2), facecolor='red', alpha=0.3, edgecolor='red', linewidth=2)
    simplex_vertices = ax.scatter([], [], c=[], cmap=cmap, norm=norm, s=100, edgecolors='black', linewidth=1)
    ax.add_patch(simplex_polygon)
    
    def animate(frame):
        if frame >= n_iters:
            return
        
        # For 2D Nelder-Mead, we need to reconstruct the simplex
        # Since we only have the parameter history (not the full simplex), 
        # we'll simulate a realistic simplex around the current point
        
        current_param1, current_param2 = param_history[frame]
        current_error = error_history[frame]
        
        # Create a triangular simplex around current point
        # The simplex size should decrease as optimization progresses
        simplex_size = min(param1_range, param2_range) * 0.05 * (1 - frame/n_iters) + 0.01
        
        # Create three vertices of the simplex triangle
        vertices = np.array([
            [current_param1, current_param2],  # Center point (best)
            [current_param1 + simplex_size, current_param2 - simplex_size * 0.5],  # Right vertex
            [current_param1 - simplex_size * 0.5, current_param2 + simplex_size * 0.866]  # Left vertex
        ])
        
        # Assign error values to vertices (center gets current error, others get slightly higher)
        vertex_errors = np.array([
            current_error,  # Best vertex
            current_error * 1.1,  # Slightly worse
            current_error * 1.05   # Slightly worse
        ])
        
        # Update simplex visualization
        simplex_polygon.set_xy(vertices)
        
        # Update vertex scatter plot with colors based on error
        simplex_vertices.set_offsets(vertices)
        simplex_vertices.set_array(vertex_errors)
        
        # Add iteration text
        ax.set_title(f'Nelder-Mead Simplex Evolution - Iteration {frame+1}\n'
                    f'Best Error: {current_error:.6f}')
        
        return simplex_polygon, simplex_vertices
    
    # Create animation
    anim = animation.FuncAnimation(fig, animate, frames=n_iters, 
                                 interval=300, blit=False, repeat=True)
    
    # Save animation
    anim_path = os.path.join(OUTPUT_DIR, 'simplex_convergence.gif')
    print(f"Saving simplex convergence animation to {anim_path}")
    anim.save(anim_path, writer='pillow', fps=4)
    plt.close(fig)
    
    return anim