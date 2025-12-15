"""
Plot cell counts over time for all simulation cases with analytical solution.
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import pickle
from config import G_LENGTH, M_LENGTH, KDEATH, T_DORMANT, DT

def load_cell_count_data(case_dir):
    """Load cell count time series from a case directory."""
    try:
        data_dict_path = os.path.join(case_dir, 'data_dict.pkl')
        if os.path.exists(data_dict_path):
            with open(data_dict_path, 'rb') as f:
                data_dict = pickle.load(f)
            times = data_dict['times']
            positions = data_dict['positions']
            cell_counts = [np.sum(~np.isnan(pos[:, 0])) for pos in positions]
            return times, cell_counts
    except Exception as e:
        print(f"Error loading {case_dir}: {e}")
    return None, None

def compute_ode_solution(times, N0):
    """Compute ODE solution for cell cycle dynamics."""
    k1 = 1.0 / G_LENGTH
    k2 = 1.0 / M_LENGTH
    kdeath = KDEATH
    
    G_arr = np.zeros_like(times, dtype=float)
    M_arr = np.zeros_like(times, dtype=float)
    
    G = float(N0)
    M = 0.0
    G_arr[0] = G
    M_arr[0] = M
    
    def euler_step(G, M, dt, before_dormancy):
        if before_dormancy:
            dG = -kdeath * G
            dM = -kdeath * M
        else:
            dG = (-(k1 + kdeath)) * G + 2.0 * k2 * M
            dM = k1 * G - (k2 + kdeath) * M
        return G + dt * dG, M + dt * dM
    
    for i in range(1, len(times)):
        t_prev = times[i - 1]
        t_next = times[i]
        dt = float(t_next - t_prev)
        
        if (t_prev < T_DORMANT) and (t_next > T_DORMANT):
            dt1 = T_DORMANT - t_prev
            dt2 = t_next - T_DORMANT
            G, M = euler_step(G, M, dt1, before_dormancy=True)
            G, M = euler_step(G, M, dt2, before_dormancy=False)
        else:
            before = (t_prev < T_DORMANT)
            G, M = euler_step(G, M, dt, before_dormancy=before)
        
        G_arr[i] = G
        M_arr[i] = M
    
    return G_arr + M_arr

def plot_all_cell_counts(base_dir='data/output/Cases2025-11-12'):
    """Plot cell counts over time for all cases."""
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    gradient_cases = {'LINEAR', 'ZONE'}
    plotted_gradient = False
    plotted_other = False
    ode_plotted = False
    max_time = 7.0
    
    for soft_type in ['SOFT', 'NO_SOFT']:
        soft_dir = os.path.join(base_dir, soft_type)
        if not os.path.exists(soft_dir):
            continue
            
        for case_name in os.listdir(soft_dir):
            case_path = os.path.join(soft_dir, case_name)
            if not os.path.isdir(case_path):
                continue
            
            times, cell_counts = load_cell_count_data(case_path)
            if times is None or cell_counts is None:
                continue
            
            max_time = max(max_time, times[-1])
            is_gradient = case_name in gradient_cases
            
            if is_gradient:
                color = 'red'
                alpha = 0.6
                lw = 1.5
                label = 'Gradient cases' if not plotted_gradient else None
                plotted_gradient = True
            else:
                color = 'lightblue'
                alpha = 0.5
                lw = 1
                label = 'Other cases' if not plotted_other else None
                plotted_other = True
            
            ax.plot(times, cell_counts, color=color, alpha=alpha, lw=lw, label=label)
            
            if not ode_plotted:
                N0 = cell_counts[0]
                N_ode = compute_ode_solution(times, N0)
                ax.plot(times, N_ode, '--', color='black', lw=2.5, 
                       label='ODE solution', zorder=100)
                ode_plotted = True
    
    ax.set_xlabel('Time (days)', fontsize=14)
    ax.set_ylabel('Number of Cells', fontsize=14)
    ax.set_title('Cell Count vs Time: All Cases', fontsize=16, fontweight='bold')
    ax.legend(fontsize=12, loc='upper left')
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, max_time)
    ax.set_ylim(0,5000)
    plt.tight_layout()
    
    output_path = os.path.join(base_dir, 'all_cell_counts_comparison.pdf')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Cell count plot saved to: {output_path}")
    
    plt.close(fig)
    return fig, ax

if __name__ == "__main__":
    base_dir = 'data/output/Cases2025-11-12'
    fig, ax = plot_all_cell_counts(base_dir)

