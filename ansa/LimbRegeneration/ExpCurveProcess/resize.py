""" rescale old c59 curves (orginally thought pixels were 0.21 x 0.21 um, but they are not.) 
DID NOT USE THIS! NOT NEEDED!!
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from average_shape import plot, compute_average
parent_dir = 'curves_backup1/csv_aligned'
c59_curves = [np.loadtxt(f'{parent_dir}/c59{i}.csv', delimiter=',', skiprows=1) for i in range(1,6)]
c59_curves_rescaled = []
magnifications = [(0.1, 0.1), (0.11, 0.10), (0.13, 0.13), (0.12, 0.12), (0.15, 0.15)] # pixels are for example, 0.1 x 0.11

# (0.11, 0.11) for c591 gives really close to sam's number

for i, curve in enumerate(c59_curves):
    c, d = magnifications[i]
    transformation_matrix = np.array([
        [c, 0], # scale x by c
        [0, d] # scale y by d
    ])
    curve_rescaled = curve @ transformation_matrix * (1/0.21)
    c59_curves_rescaled.append(curve_rescaled)
for i, curve in enumerate(c59_curves_rescaled):
    xy = pd.DataFrame(data=curve, columns=['X', 'Y'])
    xy.to_csv(f'c59{i+1}.csv', index=False)

plot(c59_curves_rescaled, title='c59')


