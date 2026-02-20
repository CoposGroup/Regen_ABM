import _setup_path
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

def pso_scatter3d(evals, iteration, title, fit_plane=False, which_error=None):
    if iteration == None:
        eval = evals
    else:
        eval = evals.loc[evals['iteration']==iteration]
    
    eval = eval[
        (eval['KDIV'] >= 0) & (eval['KDIV'] <= 0.6) &
        (eval['MIGRATION_FRACTION'] >= 0) & (eval['MIGRATION_FRACTION'] <= 1.0) &
        (eval['MU_MIGRATION']*200 >= 50) & (eval['MU_MIGRATION']*200 <= 150)
    ]
    
    error_col = which_error if which_error is not None else 'error'
    eval = eval.sort_values(error_col, ascending=False).reset_index(drop=True)
    c = eval[error_col]

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    scatter = ax.scatter(eval['KDIV'], eval['MIGRATION_FRACTION'], eval['MU_MIGRATION']*200, c=c, cmap='viridis', vmin=0, vmax=100)
    ax.set_xlim(0, 0.6)
    ax.set_ylim(0, 1.0)
    ax.set_zlim(50, 150)

    ax.set_xlabel('KDIV')
    ax.set_ylabel('MIGRATION_FRACTION')
    ax.set_zlabel('MU_MIGRATION')
    cbar = fig.colorbar(scatter, ax=ax, pad=0.15)
    cbar.set_label(r'RMSE ($\mu$m)')
    
    if fit_plane:
        error_threshold = np.percentile(eval['error'], 20)
        low_error_mask = eval['error'] <= error_threshold
        x_low = eval.loc[low_error_mask, 'KDIV'].values
        y_low = eval.loc[low_error_mask, 'MIGRATION_FRACTION'].values
        z_low = eval.loc[low_error_mask, 'MU_MIGRATION'].values * 200
        
        A = np.vstack([x_low, y_low, np.ones(len(x_low))]).T
        coeffs = np.linalg.lstsq(A, z_low, rcond=None)[0]
        a, b, c = coeffs[0], coeffs[1], coeffs[2]
        
        x_range = np.linspace(eval['KDIV'].min(), eval['KDIV'].max(), 20)
        y_range = np.linspace(eval['MIGRATION_FRACTION'].min(), eval['MIGRATION_FRACTION'].max(), 20)
        X, Y = np.meshgrid(x_range, y_range)
        Z = a * X + b * Y + c
    
    plt.savefig(f"pso_param_space_{title}.pdf")
    plt.show()

def pso_scatter2d(evals, iteration=None, title='', fit_line=False, star_color='red'):
    if iteration == None:
        eval = evals
    else:
        eval = evals.loc[evals['iteration']==iteration]
    
    eval = eval[
        (eval['KDIV'] >= 0) & (eval['KDIV'] <= 0.5) &
        (eval['MIGRATION_FRACTION'] >= 0) & (eval['MIGRATION_FRACTION'] <= 1.0)
    ]
    
    fig = plt.figure()
    ax = fig.add_subplot()
    scatter = ax.scatter(eval['KDIV'], eval['MIGRATION_FRACTION'], c=eval['error'], cmap='viridis', vmin=0, vmax=100)
    ax.set_xlim(0, 0.5)
    ax.set_ylim(0, 1.0)
    
    ax.set_xlabel(r'1/Cell Cycle Duration ($k_{\mathrm{div}}$)', fontsize=22)
    ax.set_ylabel(r'Motility Fraction ($m$)', fontsize=22)
    plt.tick_params(axis='both', which='major', labelsize=18)
    
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: '' if abs(x) < 1e-10 else f'{x:.3f}'.rstrip('0').rstrip('.')))
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:.3f}'.rstrip('0').rstrip('.')))

    cbar = fig.colorbar(scatter, ax=ax)
    cbar.set_label(r'RMSE ($\mu$m)', fontsize=22)
    cbar.ax.tick_params(labelsize=18)
    
    min_error_idx = eval['error'].idxmin()
    ax.scatter(eval.loc[min_error_idx, 'KDIV'], 
              eval.loc[min_error_idx, 'MIGRATION_FRACTION'],
              marker='*', s=500, color='red', edgecolors='red', 
              linewidths=2, zorder=10, label='Lowest error')
    
    if title == "ctrl":
        ax.scatter([0.42], [0.3], marker='d', s=500, facecolor='white', 
                  edgecolors='black', linewidths=3, zorder=10)
    
    if fit_line:
        error_threshold = np.percentile(eval['error'], 20)
        low_error_mask = eval['error'] <= error_threshold
        x_low = eval.loc[low_error_mask, 'KDIV'].values
        y_low = eval.loc[low_error_mask, 'MIGRATION_FRACTION'].values
        
        A = np.vstack([x_low, np.ones(len(x_low))]).T
        coeffs = np.linalg.lstsq(A, y_low, rcond=None)[0]
        a, b = coeffs[0], coeffs[1]
        
        x_range = np.linspace(eval['KDIV'].min(), eval['KDIV'].max(), 100)
        y_fit = a * x_range + b
        ax.plot(x_range, y_fit, 'r--', linewidth=3, label='Fitted line')
    
    plt.tight_layout()
    plt.savefig(f"pso_param_space_{title}.pdf")
    plt.show()

if __name__ == '__main__':
    combined_evals = pd.read_csv('pso_data/mu_kdiv_m_ctrl_c59combined.csv')
    ctrl_kdiv_m = pd.read_csv('pso_data/ctrl_kdiv_m.csv')
    c59_kdiv_m = pd.read_csv('pso_data/c59_kdiv_m.csv')
    ctrl_softer = pd.read_csv('pso_data/ctrl_softer_evals.csv')
    c59_softer = pd.read_csv('pso_data/c59_softer_evals.csv')
    
    pso_scatter3d(combined_evals, None, 'ctrl_3d', which_error='error_ctrl')
    pso_scatter3d(combined_evals, None, 'c59_3d', which_error='error_c59')
    pso_scatter2d(ctrl_kdiv_m, None, 'ctrl_2d')
    pso_scatter2d(c59_kdiv_m, None, 'c59_2d')
    pso_scatter2d(ctrl_softer, None, 'ctrl_soft_2d')
    pso_scatter2d(c59_softer, None, 'c59_soft_2d')