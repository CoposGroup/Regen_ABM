"""
Takes curves which are already oriented and aligned, creates an average curve over arc length..
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def parameterize_curve(curve):
    """take curve of (x,y) points, return (theta, r) ordered by theta [-pi/2, pi/2]"""
    x, y = curve[:,0], curve[:,1]
    theta = np.atan(y/x)
    r = np.sqrt(x**2 + y**2)
    pCurve = np.column_stack((theta, r))
    sort_indices = pCurve[:, 0].argsort()
    pCurve = pCurve[sort_indices]
    return pCurve

def compute_average(curves, in_coords='xy', out_coords='polar'):
    """
    input is assumed to by in xy coords
    returns an average curve in terms of theta and r or xy depending on coords param"""
    if in_coords == 'xy':
        pCurves = [parameterize_curve(i) for i in curves] # parametrize to polar if not already
    else:
        pCurves = curves
    curve_lengths = [len(i) for i in pCurves]
    most_points, most_points_idx = max(curve_lengths), curve_lengths.index(max(curve_lengths))
    common_theta = pCurves[most_points_idx][:,0]

    # interpolate r over common theta for all curves
    pCurves_r = [np.interp(common_theta, i[:,0], i[:,1]) for i in pCurves]    
    # Average r values across all curves at each theta point
    r_arr = np.mean(pCurves_r, axis=0)
    pCurve_avg = np.column_stack((common_theta, r_arr))
    if out_coords == 'polar':
        return pCurve_avg
    elif out_coords == 'xy':
        theta_avg, r_avg = pCurve_avg[:,0], pCurve_avg[:,1]
        x_avg = r_avg*np.cos(theta_avg)
        y_avg = r_avg*np.sin(theta_avg)
        pCurve_avg = np.column_stack([x_avg, y_avg])
        return pCurve_avg

def plot(curves, title='', save=False):
    fig = plt.figure()
    for i, curve in enumerate(curves):
        pCurve = parameterize_curve(curve)
        theta, r = pCurve[:,0], pCurve[:,1]
        x = r*np.cos(theta)
        y = r*np.sin(theta)
        plt.scatter(x, y, s=1, label=i+1)
    
    avg = compute_average(curves)
    theta_avg, r_avg = avg[:,0], avg[:,1]
    x_avg = r_avg*np.cos(theta_avg)
    y_avg = r_avg*np.sin(theta_avg)

    plt.scatter(x_avg, y_avg, s=7,c='black', label='average')
    plt.xlim([0, 800])
    plt.ylim([-400, 400])
    plt.title(title)
    ax = plt.gca()
    ax.set_aspect('equal')
    plt.xlabel('x (um)')
    plt.ylabel('y (um)')
    plt.legend()

    # save
    avg_df_polar = pd.DataFrame(data=np.column_stack([theta_avg, r_avg]), columns=['theta', 'r'])
    avg_df_xy = pd.DataFrame(data=np.column_stack([x_avg, y_avg]), columns=['X', 'Y'])
    if save:
        avg_df_polar.to_csv(f'{title}_avg_polar.csv', index=False)
        avg_df_xy.to_csv(f'{title}_avg_xy.csv', index=False)
        plt.savefig(f'{title}_curves.pdf')
    
    print(f"{title}: outgrowth length (x) = {max(x_avg)}")
    print(f"{title}: width (y) = {max(y_avg)-min(y_avg)}")
    plt.show()

    for i, curve in enumerate(curves):
        print(f'{title}{i+1}: outgrowth length = {max(curve[:,0])}')


    #plotting x, y normally (should be same, this is for double check)
    # fig = plt.figure()
    # for i, curve in enumerate(curves):
    #     plt.scatter(curve[:,0], curve[:,1], s=1, label=i+1)
    # plt.axis('equal')
    # plt.legend()
    # plt.show()

def plot_averages(c59, control, save=False):
    plt.figure()
    plt.scatter(c59[:,0], c59[:,1], s=7,c='red', label='C59')
    plt.scatter(control[:,0], control[:,1], s=7,c='navy', label='Control')

    plt.xlim([0, 800])
    plt.ylim([-400, 400])
    plt.title('Average Control vs C59')
    ax = plt.gca()
    ax.set_aspect('equal')
    plt.xlabel('x (um)')
    plt.ylabel('y (um)')
    plt.legend()
    if save:
        plt.savefig('average_curves.pdf')
    plt.show()



if __name__ == "__main__":
    parent_dir = 'curves/csv_aligned'

    control_curves = [np.loadtxt(f'{parent_dir}/control{i}.csv', delimiter=',', skiprows=1) for i in range(1,6)]
    c59_curves = [np.loadtxt(f'{parent_dir}/c59{i}.csv', delimiter=',', skiprows=1) for i in range(1,6)]

    plot(control_curves, 'control')
    plot(c59_curves, 'c59')


    c59_avg = compute_average(curves=c59_curves, in_coords='xy', out_coords='xy')
    control_avg = compute_average(curves=control_curves, in_coords='xy', out_coords='xy')

    plot_averages(c59_avg, control_avg, save=True)

    print(f'\n \n')
    print('AVG OF INDIVIDUAL CURVE OUTGROWTH LENGTHS')
    print('C59:')
    x_max_c59 = []
    for curve in c59_curves:
        x_max_c59.append(curve[:,0].max())
    print(np.mean(x_max_c59))
    print(f'\n')
    print('CONTROL:')
    x_max_control = []
    for curve in control_curves:
        x_max_control.append(curve[:,0].max())
    print(np.mean(x_max_control))

