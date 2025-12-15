"""Create Smooth Interpolated Bone Boundary"""

import numpy as np
from scipy.sparse import spdiags
from scipy.interpolate import CubicSpline, interpn, interp1d
import matplotlib.pyplot as plt

theta = np.linspace(1.5*np.pi, 2.5*np.pi, 200)
xb_semi = 0.5 * np.cos(theta)
yb_semi = 0.5 * np.sin(theta)
# find the two points closest to x=1.0
diffs = np.abs(xb_semi - 0.5)
closest_indices = np.argsort(diffs)[:2]
min_idx = np.min(closest_indices)
max_idx = np.max(closest_indices)

# mean spacing along that arc
dists = np.hypot(np.diff(xb_semi[:min_idx]), np.diff(yb_semi[:min_idx]))
avg_ds = np.mean(dists)

# vertical segment
y_v = np.arange(yb_semi[closest_indices].min(),
                yb_semi[closest_indices].max(), avg_ds)
print(y_v.max()- y_v.min())
x_v = np.full_like(y_v, 0.5)

# small horizontal caps
    # small horizontal caps
# x_top = xb_semi[0] - np.array([2*avg_ds, avg_ds])
# y_top = np.full(2, yb_semi[0])
# x_bottom = xb_semi[-1] - np.array([2*avg_ds, avg_ds])
# y_bottom = np.full(2, yb_semi[-1])

print(yb_semi[0])
print(yb_semi[-1])

# concatenate to form closed loop
xb = np.concatenate([
    # x_top,
    xb_semi[:min_idx],
    x_v,
    xb_semi[max_idx:],
    # x_bottom,

])
yb = np.concatenate([
    # y_top,
    yb_semi[:min_idx],
    y_v,
    yb_semi[max_idx:],
    # y_bottom,
])
# small horizontal caps with consistent spacing
# Calculate number of points needed for horizontal caps (from 0.25 distance)


Xb0 = np.vstack([xb, yb]).T   # (Nb,2)
Xb = Xb0.copy()
dsb = np.hypot(*(Xb0[1] - Xb0[0]))  # first segment length

# 3) Combine segments

# Plot to verify
plt.figure(figsize=(10,6))







t_orig = np.arange(len(Xb))

# 2) Create finer sampling (4 points between each original point)
t_fine = np.linspace(0, len(Xb)-1, len(Xb)*2)

# 3) Create separate splines for x and y coordinates
spline_x = CubicSpline(t_orig, Xb[:,0], bc_type='not-a-knot')
spline_y = CubicSpline(t_orig, Xb[:,1], bc_type='not-a-knot')

# 4) Interpolate to get new points
Xb_smooth = np.zeros((len(t_fine), 2))

###################
# Add horizontal extensions at y = ±0.5
n_points = int(0.25*2/avg_ds)  # Number of points for 0.25 length segment

# Create top horizontal line
x_extend_top = np.linspace(-1.00, xb_semi[0], n_points)[:-1] # change -1.00 back to -0.25 if issues
y_extend_top = np.full_like(x_extend_top, 0.5)

# Create bottom horizontal line
x_extend_bottom = np.linspace(-1.00, xb_semi[-1], n_points)[:-1] # change -1.00 back to -0.25 if issues
y_extend_bottom = np.full_like(x_extend_bottom, -0.5)

Xb_smooth[:,0] = spline_x(t_fine)
Xb_smooth[:,1] = spline_y(t_fine)
# Add extensions to smoothed boundary
Xb_smooth2 = np.vstack([
    np.column_stack([x_extend_top, y_extend_top]),
    Xb_smooth[::-1],
    np.column_stack([x_extend_bottom[::-1], y_extend_bottom])
])
################
# After creating Xb_smooth2 and before saving, add:
def check_point_spacing(points, min_dist=1e-6):
    """Check for points that are too close together"""
    # Calculate distances between consecutive points
    diff = np.diff(points, axis=0)
    distances = np.sqrt(np.sum(diff**2, axis=1))
    
    # Find where points are too close
    close_idx = np.where(distances < min_dist)[0]
    
    if len(close_idx) > 0:
        print(f"\nWarning: Found {len(close_idx)} points that are too close:")
        for idx in close_idx:
            print(f"Points {idx} and {idx+1}:")
            print(f"  P1: {points[idx]}")
            print(f"  P2: {points[idx+1]}")
            print(f"  Distance: {distances[idx]:.2e}")
            
    # Print statistics
    print(f"\nPoint spacing statistics:")
    print(f"  Mean distance: {np.mean(distances):.6f}")
    print(f"  Min distance: {np.min(distances):.6f}")
    print(f"  Max distance: {np.max(distances):.6f}")
    print(f"  Std deviation: {np.std(distances):.6f}")
    
    return len(close_idx) == 0

# Check the point spacing
Xb = Xb_smooth2.copy()
if not check_point_spacing(Xb, min_dist=avg_ds/10):
    print("\nWarning: Boundary may have overlapping or very close points!")
########
# Continue with existing code...
# 5) Plot and Save

np.savetxt("bone.csv", Xb, delimiter=",", comments='')
print('saved bone.csv')

plt.figure(figsize=(10,6))
# plt.plot(Xb[:,0], Xb[:,1], 'o', label='Original points', markersize=4)
plt.plot(Xb[:,0], Xb[:,1], '-', label='Interpolated curve')
plt.scatter(Xb[:,0], Xb[:,1], c='r', s=2, label='Interpolated points')
plt.title(f'Boundary: {len(Xb)} original points → {len(Xb)} interpolated points')
plt.xlim(0, 0.3)
plt.ylim(1.25, 1.75)
plt.legend()
plt.axis('equal')
plt.grid(True)
plt.show()
# Update Xb to use the smoother boundary
Xb = Xb_smooth.copy()
Nb = len(Xb)






# # Replace the existing plotting code with:
# plt.figure(figsize=(10,6))

# # Create color array based on point order
# colors = np.arange(len(Xb[350:,0]))

# # Plot points colored by order
# scatter = plt.scatter(Xb[350:,0], Xb[350:,1], c=colors, cmap='viridis', 
#                      s=2, label='Boundary points')
# plt.colorbar(scatter, label='Point order')

# plt.title(f'Boundary: {len(Xb)} points, colored by order')
# plt.xlim(-0.2, 0.1)
# plt.ylim(1.25, 1.75)
# plt.legend()
# plt.axis('equal')
# plt.grid(True)

# # Add arrows to show direction every 50 points
# # step = 50
# # for i in range(0, len(Xb)-1, step):
# #     plt.arrow(Xb[i,0], Xb[i,1], 
# #              (Xb[i+1,0] - Xb[i,0])*0.5, 
# #              (Xb[i+1,1] - Xb[i,1])*0.5,
# #              head_width=0.02, head_length=0.03, 
# #              fc='red', ec='red', alpha=0.5)

# plt.show()




