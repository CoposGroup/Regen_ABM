import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from cheb_expansion import cheb_expansion, coefficients, inner_product, scale_shape, inner_product_compare
from sklearn.preprocessing import MinMaxScaler
import os

dir_exp = 'data/input/exp/day12s'
dir_sim = 'data/output/sim_batch'
dir_output = 'data/output/chebyshev'

CTRL_T12 = pd.read_csv(os.path.join(dir_exp, 'CTRL_T12.csv')).to_numpy()
BOX5_T12 = pd.read_csv(os.path.join(dir_exp, 'BOX5_T12.csv')).to_numpy()
C59_T12 = pd.read_csv(os.path.join(dir_exp, 'C59_T12.csv')).to_numpy()
FOXY5_T12 = pd.read_csv(os.path.join(dir_exp, 'FOXY5_T12.csv')).to_numpy()

SIM_T7_DEFAULT = pd.read_csv(os.path.join(dir_sim, 'DEFAULT_soft/boundary.csv')).loc[lambda df: (df['timestep'] == max(df['timestep'])) & (df['x'] > 1)][['x', 'y']].to_numpy()[:, [1, 0]] * [1, -1] # filter x>1, then rotate 90° clockwise (auto-sorted in coefficients)
SIM_T7_MIGRATION18 = pd.read_csv(os.path.join(dir_sim, 'MIGRATION_18_soft/boundary.csv')).loc[lambda df: (df['timestep'] == max(df['timestep'])) & (df['x'] > 1)][['x', 'y']].to_numpy()[:, [1, 0]] * [1, -1] 
SIM_T7_C59 = pd.read_csv(os.path.join(dir_sim, 'C59/boundary.csv')).loc[lambda df: (df['timestep'] == max(df['timestep'])) & (df['x'] > 1)][['x', 'y']].to_numpy()[:, [1, 0]] * [1, -1] 
n = 4

# Calculate coefficients with shape-preserving scaling and get scaling parameters
# coeffs_CTRL_T12, scaling_params_CTRL = coefficients(CTRL_T12, n, type='data', return_scaling=True)
# coeffs_BOX5_T12, scaling_params_BOX5 = coefficients(BOX5_T12, n, type='data', return_scaling=True)
# coeffs_C59_T12, scaling_params_C59 = coefficients(C59_T12, n, type='data', return_scaling=True)
# coeffs_FOXY5_T12, scaling_params_FOXY5 = coefficients(FOXY5_T12, n, type='data', return_scaling=True)

# print(f'coeffs_CTRL_T12 = {coeffs_CTRL_T12}')

# Use the same scaling that was used for coefficient calculation
# scale_factor_CTRL, x_offset_CTRL, y_offset_CTRL = scaling_params_CTRL
# CTRL_T12_scaled, _, _, _ = scale_shape(CTRL_T12)

def plot_scaling_comparison(data, data_scaled, title="Shape Scaling Comparison", save_path=None, show=True):
    """Plot original vs scaled data to verify shape preservation."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    ax1.plot(data[:,0], data[:,1], 'b-', label='Original')
    ax1.set_title('Original Shape')
    ax1.set_xlabel('x')
    ax1.set_ylabel('y')
    ax1.axis('equal')
    ax1.grid(True)

    ax2.plot(data_scaled[:,0], data_scaled[:,1], 'r-', label='Scaled (shape preserved)')
    ax2.set_title('Scaled Shape (Shape Preserved)')
    ax2.set_xlabel('x')
    ax2.set_ylabel('y')
    ax2.axis('equal')
    ax2.grid(True)
    
    plt.suptitle(title)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    if show:
        plt.show()
    if not show:
        plt.close()

def plot_chebyshev_fit(data_scaled, cheb_values, coeffs, n, title="Chebyshev Fit", save_path=None, show=True):
    """Plot original data vs Chebyshev expansion fit."""
    plt.figure(figsize=(10, 8))
    plt.plot(data_scaled[:,0], cheb_values, 'r--', label=f'Chebyshev Expansion (n={n})', alpha=0.7, linewidth=2)
    plt.plot(data_scaled[:,0], data_scaled[:,1], 'b-', label='Original Data', alpha=0.8)
    plt.xlim(-1, 1)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.axis('equal')  # Equal scaling on both axes
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    if show:
        plt.show()
    if not show:
        plt.close()

def plot_coefficients(coeffs, title="Chebyshev Coefficients", save_path=None, show=True, normalize_plot=False):
    """Plot Chebyshev coefficients as a bar chart."""
    n = len(coeffs)
    plt.figure(figsize=(10, 6))
    
    # Apply normalization if requested
    if normalize_plot:
        max_coeff = max(abs(c) for c in coeffs)
        plot_coeffs = [c / max_coeff for c in coeffs]
        title_suffix = " (Normalized to Max)"
        ylabel = "Normalized Coefficient Value"
        ylim_range = (-1.1, 1.1)
    else:
        plot_coeffs = coeffs
        title_suffix = ""
        ylabel = "Coefficient Value"
        ylim_range = (-0.6, 0.6)
    
    plt.bar(range(n), plot_coeffs, color='steelblue', alpha=0.7, edgecolor='black')
    plt.axhline(y=0, color='k', linestyle='--', alpha=0.5)
    plt.ylim(ylim_range)
    plt.xticks(range(n))
    plt.xlabel('Coefficient Index (n)')
    plt.ylabel(ylabel)
    plt.title(title + title_suffix)
    plt.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    if show:
        plt.show()
    if not show:
        plt.close()

def plot_multiple_shapes(datasets, labels, title="Shape Comparison", save_path=None, show=True):
    """Plot multiple shapes on the same axes for comparison."""
    plt.figure(figsize=(12, 10))  # Made it more square
    colors = ['blue', 'red', 'green', 'orange', 'purple', 'brown', 'pink', 'gray']
    
    for i, (data, label) in enumerate(zip(datasets, labels)):
        color = colors[i % len(colors)]
        plt.plot(data[:,0], data[:,1], label=label, color=color, alpha=0.7, linewidth=2)
    
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title(title)
    plt.legend()
    plt.axis('equal')  # Equal scaling on both axes
    plt.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    if show:
        plt.show()
    if not show:
        plt.close()

def analyze_and_plot_shape(data, name, n=10, output_dir=None, show_plots=True, normalize_plot=False):
    """Complete analysis pipeline: compute coefficients and generate all plots."""
    coeffs, scaling_params = coefficients(data, n, type='data', return_scaling=True)
    data_scaled, _, _, _ = scale_shape(data)
    cheb_values = cheb_expansion(coeffs, n)(data_scaled[:,0])
    
    base_path = f"{output_dir}/{name}" if output_dir else None
    
    # Only generate coefficient plots when normalize_plot=True (skip shape fits)
    if normalize_plot:
        plot_coefficients(coeffs, 
                         title=f'{name} Chebyshev Coefficients (n={n})',
                         save_path=f"{base_path}_coeffs.png" if base_path else None,
                         show=show_plots,
                         normalize_plot=True)
    else:
        # Generate both plots for non-normalized version
        plot_chebyshev_fit(data_scaled, cheb_values, coeffs, n, 
                          title=f'{name} Chebyshev Fit (n={n})',
                          save_path=f"{base_path}_fit.png" if base_path else None,
                          show=show_plots)
        
        plot_coefficients(coeffs, 
                         title=f'{name} Chebyshev Coefficients (n={n})',
                         save_path=f"{base_path}_coeffs.png" if base_path else None,
                         show=show_plots,
                         normalize_plot=False)
    
    # Save coefficients to CSV
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        pd.DataFrame(coeffs, columns=['coefficient']).to_csv(f"{base_path}_coeffs.csv", index=False)
        print(f'Saved coefficients to {base_path}_coeffs.csv')
    
    return coeffs, scaling_params, data_scaled, cheb_values


if __name__ == "__main__":
    # Calculate coefficients for the four key datasets
    print("=== CALCULATING COEFFICIENTS ===")
    
    target_datasets = {
        'C59_experiment': C59_T12,
        'CTRL_experiment': CTRL_T12, 
        'C59_simulation': SIM_T7_C59,
        'CTRL_simulation': SIM_T7_DEFAULT
    }
    
    coeffs_dict = {}
    for name, data in target_datasets.items():
        coeffs = coefficients(data, n, type='data')
        coeffs_dict[name] = coeffs
        print(f"Calculated coefficients for {name}")
    
    # Calculate inner products between all pairs
    print("\n=== INNER PRODUCT COMPARISONS ===")
    names = list(coeffs_dict.keys())
    
    print("Inner product matrix:")
    print(f"{'':20s}", end="")
    for name in names:
        print(f"{name:15s}", end="")
    print()
    
    for i, name1 in enumerate(names):
        print(f"{name1:20s}", end="")
        for j, name2 in enumerate(names):
            inner_prod = inner_product_compare(coeffs_dict[name1], coeffs_dict[name2])
            print(f"{inner_prod:15.6f}", end="")
        print()
    
    # Also save to CSV for easy analysis
    import pandas as pd
    inner_product_matrix = np.zeros((len(names), len(names)))
    for i, name1 in enumerate(names):
        for j, name2 in enumerate(names):
            inner_product_matrix[i, j] = inner_product_compare(coeffs_dict[name1], coeffs_dict[name2])
    
    df = pd.DataFrame(inner_product_matrix, index=names, columns=names)
    os.makedirs(dir_output, exist_ok=True)
    df.to_csv(f"{dir_output}/inner_product_matrix.csv")
    print(f"\nSaved inner product matrix to {dir_output}/inner_product_matrix.csv")
    
    print("\n=== ANALYZING ALL EXPERIMENTAL DATA ===")
    
    # Analyze all experimental datasets
    datasets = [CTRL_T12, BOX5_T12, C59_T12, FOXY5_T12, SIM_T7_DEFAULT, SIM_T7_MIGRATION18, SIM_T7_C59]
    names_all = ["CTRL_T12", "BOX5_T12", "C59_T12", "FOXY5_T12", "SIM_T7_DEFAULT", "SIM_T7_MIGRATION18", "SIM_T7_C59"]
    
    all_coeffs = {}
    all_scaled_data = []
    all_labels = []
    
    for data, name in zip(datasets, names_all):
        coeffs, _, data_scaled, _ = analyze_and_plot_shape(
            data, name, n=n, output_dir=dir_output, show_plots=False, normalize_plot=False
        )
        all_coeffs[name] = coeffs
        all_scaled_data.append(data_scaled)
        all_labels.append(name)
        
        # Also generate normalized plots
        analyze_and_plot_shape(
            data, f"{name}_normalized", n=n, output_dir=dir_output, show_plots=False, normalize_plot=True
        )
    
    # Plot all shapes together for comparison
    plot_multiple_shapes(all_scaled_data, all_labels, 
                        title="All Shapes Comparison", 
                        save_path=f"{dir_output}/all_shapes_comparison.png",
                        show=True)

# cheb_expansion(coeffs_BOX5_T12, 10, type='data')(BOX5_T12[:,0])
# cheb_expansion(coeffs_C59_T12, 10, type='data')(C59_T12[:,0])
# cheb_expansion(coeffs_FOXY5_T12, 10, type='data')(FOXY5_T12[:,0])

# plt.plot(CTRL_T12[:,0], CTRL_T12[:,1], label='CTRL')
# plt.plot(BOX5_T12[:,0], BOX5_T12[:,1], label='BOX5')
# plt.plot(C59_T12[:,0], C59_T12[:,1], label='C59')
# plt.plot(FOXY5_T12[:,0], FOXY5_T12[:,1], label='FOXY5')
# plt.legend()
# plt.show()
# plt.show()