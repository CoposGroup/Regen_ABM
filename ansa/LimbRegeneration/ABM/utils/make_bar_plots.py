import matplotlib.pyplot as plt
from labellines import labelLines 
import numpy as np
import pandas as pd
import os
from config import CONVERSION_FACTOR_UM, EXP_OUTGROWTH_LENGTH, EXP_AREA, EXP_AREA_C59, EXP_OUTGROWTH_LENGTH_C59

length_dict_soft = {}
length_dict_no_soft = {}

area_dict_soft = {}
area_dict_no_soft = {}

base_dir = 'data/output/Cases2025-11-12'

def create_bar_plot(dict_soft, dict_no_soft, metric_name, output_dir=None):
    """Create grouped bar plot for a single metric, mimicking the penguin plot style"""
    if output_dir is None:
        output_dir = base_dir
    
    # get all unique condition names (union of both cases)
    all_conditions = set(dict_soft.keys()) | set(dict_no_soft.keys())
    all_conditions = sorted(list(all_conditions))
    
    # Convert to micrometers (or micrometers squared for area)
    if metric_name.lower() == 'area':
        # Convert area to micrometers squared
        soft_vals = [dict_soft.get(cond, 0) * CONVERSION_FACTOR_UM**2 for cond in all_conditions]
        no_soft_vals = [dict_no_soft.get(cond, 0) * CONVERSION_FACTOR_UM**2 for cond in all_conditions]
        ylabel = r'Area ($\mu$m$^2$)'
    else:
        # convert length to micrometers
        soft_vals = [dict_soft.get(cond, 0) * CONVERSION_FACTOR_UM for cond in all_conditions]
        no_soft_vals = [dict_no_soft.get(cond, 0) * CONVERSION_FACTOR_UM for cond in all_conditions]
        ylabel = rf'{metric_name} ($\mu$m)'
    
    # sort conditions by lowest softened
    sorted_indices = np.argsort(soft_vals)  # sort in ascending order
    
    all_conditions_sorted = [all_conditions[i] for i in sorted_indices]
    soft_vals_sorted = [soft_vals[i] for i in sorted_indices]
    no_soft_vals_sorted = [no_soft_vals[i] for i in sorted_indices]
    
    morphometrics = {
        'SOFT': soft_vals_sorted,
        'NO_SOFT': no_soft_vals_sorted,
    }
    
    # Save the data to CSV
    df = pd.DataFrame({
        'Condition': all_conditions_sorted,
        'SOFT': soft_vals_sorted,
        'NO_SOFT': no_soft_vals_sorted
    })
    csv_path = os.path.join(output_dir, f'{metric_name.lower()}_data.csv')
    df.to_csv(csv_path, index=False)
    print(f"{metric_name} data saved to: {csv_path}")
    
    x = np.arange(len(all_conditions_sorted))  # the label locations
    width = 0.35  # the width of the bars
    multiplier = 0

    max_stiff_val = max(no_soft_vals_sorted) if no_soft_vals_sorted else 0
    max_soft_val = max(soft_vals_sorted) if soft_vals_sorted else 0
    if metric_name == 'Area':
        real_val = 124973.7 #59425
    else:
        real_val = 238.5 #194.706
    
    fig, ax = plt.subplots(layout='constrained', figsize=(14, 6))
    colors = ['pink', 'mediumvioletred']
    
    for attribute, measurements in morphometrics.items():
        if attribute == 'SOFT':
            label = 'Local Softening'
        if attribute == 'NO_SOFT':
            label = 'No Local Softening'
        offset = width * multiplier
        rects = ax.bar(x + offset, measurements, width, label=label, color=colors[multiplier], edgecolor='black', linewidth=1)
        multiplier += 1
    ax.set_ylabel(ylabel)
    ax.set_title(f'{metric_name}')
    ax.set_xticks(x + width/2, all_conditions_sorted)
    ax.legend(loc='upper left', ncols=2)

    # horizontal lines
    if metric_name == 'Area':
        ax.axhline(y=EXP_AREA, color='green', linestyle='--', label=rf'CTRL Experimental Area: {EXP_AREA:.0f} $\mu$m$^2$')
        ax.axhline(y=EXP_AREA_C59, color='blue', linestyle='--', label=rf'C59 Experimental Area: {EXP_AREA_C59:.0f} $\mu$m$^2$')
        # ax.axhline(y=max_stiff_val, color='mediumvioletred', linestyle='--', label=f'Max Area Without Softening: {int(max_stiff_val)} $\mu$m$^2$')
        # ax.axhline(y=max_soft_val, color='pink', linestyle='--', label=f'Max Area with Softening: {int(max_soft_val)} $\mu$m$^2$')
    elif metric_name == 'Length':
        ax.axhline(y=EXP_OUTGROWTH_LENGTH, color='green', linestyle='--', label=rf'CTRL Experimental Length: {EXP_OUTGROWTH_LENGTH:.0f} $\mu$m')
        ax.axhline(y=EXP_OUTGROWTH_LENGTH_C59, color='blue', linestyle='--', label=rf'C59 Experimental Length: {EXP_OUTGROWTH_LENGTH_C59:.0f} $\mu$m')
        # ax.axhline(y=max_stiff_val, color='mediumvioletred', linestyle='--', label=f'Max Length Without Softening: {int(max_stiff_val)} $\mu$m')
        # ax.axhline(y=max_soft_val, color='pink', linestyle='--', label=f'Max Length With Softening: {int(max_soft_val)} $\mu$m')

    xvals = [7, 7, 7, 7]
    lines = plt.gca().get_lines()
    horizontal_lines = [line for line in lines]#if line.get_ydata()[0] == line.get_ydata()[-1]]
    labelLines(horizontal_lines, align=False, xvals=xvals, color='black')

    # center all label texts
    ax = plt.gca()
    fig = ax.figure
    renderer = fig.canvas.get_renderer()

    # convert text width from pixels to data coords, then shift left by half width
    for text in ax.texts:
        bbox = text.get_window_extent(renderer=renderer)
        # convert bbox width (pixels) to data units
        width_data = bbox.width / fig.dpi * (ax.get_xlim()[1] - ax.get_xlim()[0]) / fig.get_size_inches()[0]
        text.set_x(text.get_position()[0] - width_data / 2)
    
    plt.xticks(rotation=45, ha='right')
    if metric_name == 'Area':
        ax.set_ylim(bottom=0, top=150000)
    elif metric_name == 'Length':
        ax.set_ylim(bottom=0, top=400)
    
    plt.tight_layout()
    
    # Save the plot
    output_path = os.path.join(output_dir, f'{metric_name.lower()}_comparison.pdf')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"{metric_name} bar plot saved to: {output_path}")
    
    plt.close(fig)
    return fig, ax

def create_bar_plot_no_soft_only(dict_soft, dict_no_soft, metric_name, output_dir=None):
    """Create bar plot for NO_SOFT case only, matching exact positioning of comparison plot"""
    if output_dir is None:
        output_dir = base_dir
    
    # Use same logic as comparison plot to get conditions and sorting
    all_conditions = set(dict_soft.keys()) | set(dict_no_soft.keys())
    all_conditions = sorted(list(all_conditions))
    
    # Convert to micrometers (or micrometers squared for area)
    if metric_name.lower() == 'area':
        soft_vals = [dict_soft.get(cond, 0) * CONVERSION_FACTOR_UM**2 for cond in all_conditions]
        no_soft_vals = [dict_no_soft.get(cond, 0) * CONVERSION_FACTOR_UM**2 for cond in all_conditions]
        ylabel = r'Area ($\mu$m$^2$)'
    else:
        soft_vals = [dict_soft.get(cond, 0) * CONVERSION_FACTOR_UM for cond in all_conditions]
        no_soft_vals = [dict_no_soft.get(cond, 0) * CONVERSION_FACTOR_UM for cond in all_conditions]
        ylabel = rf'{metric_name} ($\mu$m)'
    
    # Sort by SOFT values (same as comparison plot)
    sorted_indices = np.argsort(soft_vals)
    all_conditions_sorted = [all_conditions[i] for i in sorted_indices]
    no_soft_vals_sorted = [no_soft_vals[i] for i in sorted_indices]
    
    x = np.arange(len(all_conditions_sorted))
    width = 0.35  # SAME width as comparison plot
    
    fig, ax = plt.subplots(layout='constrained', figsize=(14, 6))
    
    # Plot only NO_SOFT bars at the SAME offset as in comparison plot (where it was the second bar)
    offset = width * 1  # Same offset as multiplier=1 in comparison plot
    ax.bar(x + offset, no_soft_vals_sorted, width, label='No Local Softening', 
           color='mediumvioletred', edgecolor='black', linewidth=1)
    
    ax.set_ylabel(ylabel)
    ax.set_title(f'{metric_name} (No Softening)')
    ax.set_xticks(x + width/2, all_conditions_sorted)  # Same centering as comparison plot
    ax.legend(loc='upper left', ncols=2)
    
    # Add horizontal reference lines
    if metric_name == 'Area':
        ax.axhline(y=EXP_AREA, color='green', linestyle='--', 
                   label=rf'CTRL Experimental Area: {EXP_AREA:.0f} $\mu$m$^2$')
        ax.axhline(y=EXP_AREA_C59, color='blue', linestyle='--', 
                   label=rf'C59 Experimental Area: {EXP_AREA_C59:.0f} $\mu$m$^2$')
    elif metric_name == 'Length':
        ax.axhline(y=EXP_OUTGROWTH_LENGTH, color='green', linestyle='--', 
                   label=rf'CTRL Experimental Length: {EXP_OUTGROWTH_LENGTH:.0f} $\mu$m')
        ax.axhline(y=EXP_OUTGROWTH_LENGTH_C59, color='blue', linestyle='--', 
                   label=rf'C59 Experimental Length: {EXP_OUTGROWTH_LENGTH_C59:.0f} $\mu$m')
    
    xvals = [7, 7]
    lines = plt.gca().get_lines()
    horizontal_lines = [line for line in lines]
    labelLines(horizontal_lines, align=False, xvals=xvals, color='black')
    
    # Center label texts
    ax = plt.gca()
    fig = ax.figure
    renderer = fig.canvas.get_renderer()
    
    for text in ax.texts:
        bbox = text.get_window_extent(renderer=renderer)
        width_data = bbox.width / fig.dpi * (ax.get_xlim()[1] - ax.get_xlim()[0]) / fig.get_size_inches()[0]
        text.set_x(text.get_position()[0] - width_data / 2)
    
    plt.xticks(rotation=45, ha='right')
    if metric_name == 'Area':
        ax.set_ylim(bottom=0, top=150000)
    elif metric_name == 'Length':
        ax.set_ylim(bottom=0, top=400)
    
    plt.tight_layout()
    
    # Save the plot
    output_path = os.path.join(output_dir, f'{metric_name.lower()}_no_soft_only.pdf')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"{metric_name} NO_SOFT only plot saved to: {output_path}")
    
    plt.close(fig)
    return fig, ax

# Create comparison plots (both SOFT and NO_SOFT)
fig_length, ax_length = create_bar_plot(length_dict_soft, length_dict_no_soft, 'Length')
fig_area, ax_area = create_bar_plot(area_dict_soft, area_dict_no_soft, 'Area')

# Create NO_SOFT only plots (with same positioning as comparison)
fig_length_no_soft, ax_length_no_soft = create_bar_plot_no_soft_only(length_dict_soft, length_dict_no_soft, 'Length')
fig_area_no_soft, ax_area_no_soft = create_bar_plot_no_soft_only(area_dict_soft, area_dict_no_soft, 'Area')