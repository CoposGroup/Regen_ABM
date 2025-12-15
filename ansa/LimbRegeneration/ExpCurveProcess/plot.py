""" just call plot functions from average_shape.py """
from average_shape import plot, plot_averages, compute_average
import numpy as np

parent_dir = 'curves/csv_aligned'
control_curves = [np.loadtxt(f'{parent_dir}/control{i}.csv', delimiter=',', skiprows=1) for i in range(1,6)]
c59_curves = [np.loadtxt(f'{parent_dir}/c59{i}.csv', delimiter=',', skiprows=1) for i in range(1,6)]

# plot(control_curves, 'control', save=True)
# plot(c59_curves, 'c59', save=True)


c59_avg = compute_average(curves=c59_curves, in_coords='xy', out_coords='xy')
control_avg = compute_average(curves=control_curves, in_coords='xy', out_coords='xy')

plot_averages(c59_avg, control_avg, save=True)