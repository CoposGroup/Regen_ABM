""" Fix the shifting after removing the amputation plane (MATLAB) """
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from average_shape import compute_average, plot

parent_dir = 'curves/csv_aligned'
control_curves = [np.loadtxt(f'{parent_dir}/control{i}.csv', delimiter=',', skiprows=1) for i in range(1,6)]
c59_curves = [np.loadtxt(f'{parent_dir}/c59{i}.csv', delimiter=',', skiprows=1) for i in range(1,6)]

magnifications_c59 = [(0.1, 0.1), (0.11, 0.10), (0.13, 0.13), (0.12, 0.12), (0.15, 0.15)] # pixels are for example, 0.1 x 0.11 um
magnifications_control = [(0.21, 0.21), (0.21, 0.21), (0.21, 0.21), (0.21, 0.21), (0.21, 0.21)] # all are 0.21 x 0.21 um


shift_control = []
shift_c59 = []

def shift(curves, magnifications, save=False):
    shifted = []
    # x_max_lst_old = []
    # x_max_lst_new = []

    for i, curve_old in enumerate(curves):
        x_shift_um = magnifications[i][0] * 4.0 # x magnification for curve i converted to um. original error shifted back by ~4 px
        curve_new = curve_old.copy()
        curve_new[:,0] += x_shift_um

        shifted.append(curve_new)
        # x_max_lst_new.append(curve_old[:,0].max())

    average_old = compute_average(curves, in_coords='xy', out_coords='xy')
    average_new = compute_average(shifted, in_coords='xy', out_coords='xy')

    print(f'avg curve length (old)={average_old[:,0].max()}')
    print(f'avg curve length (new)={average_new[:,0].max()}')

    plot(shifted, title='', save=save)

shift(c59_curves, magnifications_c59, save=False)
shift(control_curves, magnifications_control, save=False)

