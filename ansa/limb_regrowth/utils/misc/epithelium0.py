"""Create Smooth Interpolated Epithelium Boundary"""

import numpy as np
from scipy.sparse import spdiags
from scipy.interpolate import CubicSpline, interpn, interp1d
import matplotlib.pyplot as plt

# --- USER PARAMETERS ---
do_interpolate = False      # Set to False for no interpolation
interp_factor = 1         # 1 = no interpolation, 2 = double, 4 = quadruple, etc.
n_points_main = 100       # Number of points for the final boundary (semicircle + verticals + extensions)

# --- SEMICIRCLE + VERTICALS ---
theta = np.linspace(1.5*np.pi, 2.5*np.pi, n_points_main)
xb_semi = 1.5 * np.cos(theta)
yb_semi = 1.5 * np.sin(theta)
diffs = np.abs(xb_semi - 1.0)
closest_indices = np.argsort(diffs)[:2]
min_idx = np.min(closest_indices)
max_idx = np.max(closest_indices)
dists = np.hypot(np.diff(xb_semi[:min_idx]), np.diff(yb_semi[:min_idx]))
avg_ds = np.mean(dists)
y_v = np.arange(yb_semi[closest_indices].min(), yb_semi[closest_indices].max(), avg_ds)
x_v = np.full_like(y_v, 1.0)

xb = np.concatenate([
    xb_semi[:min_idx],
    x_v,
    xb_semi[max_idx:],
])
yb = np.concatenate([
    yb_semi[:min_idx],
    y_v,
    yb_semi[max_idx:],
])
Xb0 = np.vstack([xb, yb]).T

# --- EXTENSIONS (before interpolation) ---
def make_extension(x0, y0, x1, y1, n_points):
    return np.column_stack([
        np.linspace(x0, x1, n_points),
        np.linspace(y0, y1, n_points)
    ])

# Top extension (from -1.0, 1.5 to xb_semi[0], yb_semi[0])
L_top = np.sqrt((xb_semi[0] + 1.0)**2 + (yb_semi[0] - 1.5)**2)
n_top = max(2, int(np.round(L_top / avg_ds)) + 1)
extend_top = make_extension(-1.0, 1.5, xb_semi[0], yb_semi[0], n_top)
# Bottom extension (from -1.0, -1.5 to xb_semi[-1], yb_semi[-1])
L_bot = np.sqrt((xb_semi[-1] + 1.0)**2 + (yb_semi[-1] + 1.5)**2)
n_bot = max(2, int(np.round(L_bot / avg_ds)) + 1)
extend_bot = make_extension(-1.0, -1.5, xb_semi[-1], yb_semi[-1], n_bot)

# --- INTERPOLATION (including extensions) ---
if do_interpolate and interp_factor > 1:
    # Interpolate main boundary
    t_main = np.arange(len(Xb0))
    t_main_fine = np.linspace(0, len(Xb0)-1, len(Xb0)*interp_factor)
    spline_x_main = CubicSpline(t_main, Xb0[:,0], bc_type='not-a-knot')
    spline_y_main = CubicSpline(t_main, Xb0[:,1], bc_type='not-a-knot')
    Xb_main = np.zeros((len(t_main_fine), 2))
    Xb_main[:,0] = spline_x_main(t_main_fine)
    Xb_main[:,1] = spline_y_main(t_main_fine)
    # Interpolate extensions
    def interp_ext(ext):
        t_ext = np.arange(len(ext))
        t_ext_fine = np.linspace(0, len(ext)-1, len(ext)*interp_factor)
        spline_x_ext = CubicSpline(t_ext, ext[:,0], bc_type='not-a-knot')
        spline_y_ext = CubicSpline(t_ext, ext[:,1], bc_type='not-a-knot')
        ext_fine = np.zeros((len(t_ext_fine), 2))
        ext_fine[:,0] = spline_x_ext(t_ext_fine)
        ext_fine[:,1] = spline_y_ext(t_ext_fine)
        return ext_fine
    extend_top_fine = interp_ext(extend_top)
    extend_bot_fine = interp_ext(extend_bot)
    # Assemble full boundary (reverse main for correct order)
    Xb_full = np.vstack([
        extend_top_fine,
        Xb_main[::-1],
        extend_bot_fine[::-1]
    ])
else:
    Xb_full = np.vstack([
        extend_top,
        Xb0[::-1],
        extend_bot[::-1]
    ])

# After interpolation (or not), get the main boundary in the desired order
main = Xb0[::-1] if not (do_interpolate and interp_factor > 1) else Xb_main[::-1]

# Top extension: from (-1, 1.5) to main[0]
extend_top = make_extension(-1.0, 1.5, main[0,0], main[0,1], n_top)
# Bottom extension: from main[-1] to (-1, -1.5)
extend_bot = make_extension(main[-1,0], main[-1,1], -1.0, -1.5, n_bot)

# Interpolate extensions if needed
if do_interpolate and interp_factor > 1:
    extend_top = interp_ext(extend_top)
    extend_bot = interp_ext(extend_bot)

# Concatenate all segments (no duplicates at joins)
full_boundary = np.vstack([
    extend_top[:-1],
    main[1:-1],
    extend_bot
])

# Compute cumulative arc length
diffs = np.diff(full_boundary, axis=0)
dists = np.sqrt((diffs**2).sum(axis=1))
arc_length = np.concatenate([[0], np.cumsum(dists)])

# Choose number of points for the final boundary
n_final = n_points_main  # Always use the user-specified value

# Uniformly spaced arc-length values
arc_uniform = np.linspace(0, arc_length[-1], n_final)

# Interpolate x and y as a function of arc length
fx = interp1d(arc_length, full_boundary[:,0], kind='cubic')
fy = interp1d(arc_length, full_boundary[:,1], kind='cubic')
Xb_full = np.column_stack([fx(arc_uniform), fy(arc_uniform)])

# (Optional) Remove any remaining duplicates
def remove_duplicate_joins(points, tol=1e-10):
    diffs = np.diff(points, axis=0)
    dists = np.linalg.norm(diffs, axis=1)
    keep = np.ones(len(points), dtype=bool)
    keep[1:] = dists > tol
    return points[keep]
Xb_full = remove_duplicate_joins(Xb_full)

# --- CHECK SPACING ---
def check_point_spacing(points, min_dist=1e-6):
    diff = np.diff(points, axis=0)
    distances = np.sqrt(np.sum(diff**2, axis=1))
    close_idx = np.where(distances < min_dist)[0]
    if len(close_idx) > 0:
        print(f"\nWarning: Found {len(close_idx)} points that are too close:")
        for idx in close_idx:
            print(f"Points {idx} and {idx+1}:")
            print(f"  P1: {points[idx]}")
            print(f"  P2: {points[idx+1]}")
            print(f"  Distance: {distances[idx]:.2e}")
    print(f"\nPoint spacing statistics:")
    print(f"  Mean distance: {np.mean(distances):.6f}")
    print(f"  Min distance: {np.min(distances):.6f}")
    print(f"  Max distance: {np.max(distances):.6f}")
    print(f"  Std deviation: {np.std(distances):.6f}")
    return len(close_idx) == 0

if not check_point_spacing(Xb_full, min_dist=avg_ds/10):
    print("\nWarning: Boundary may have overlapping or very close points!")

# --- SAVE AND PLOT ---
np.savetxt("epi100.csv", Xb_full, delimiter=",", comments='')
print('saved epi100.csv')

plt.figure(figsize=(10,6))
plt.plot(Xb_full[:,0], Xb_full[:,1], '-', label='Boundary')
plt.scatter(Xb_full[:,0], Xb_full[:,1], c='r', s=2, label='Points')
plt.title(f'Boundary: {len(Xb_full)} points (n_points_main={n_points_main}, uniform arc-length spacing)')
plt.axis('equal')
plt.grid(True)
plt.legend()
plt.show()

# --- Plot points colored by index to visualize order ---
plt.figure(figsize=(10,6))
colors = np.arange(len(Xb_full))
scatter = plt.scatter(Xb_full[:,0], Xb_full[:,1], c=colors, cmap='viridis', s=10)
plt.plot(Xb_full[:,0], Xb_full[:,1], '-', color='gray', alpha=0.5)
cbar = plt.colorbar(scatter, label='Point index')
plt.title('Boundary point order (color = index)')
plt.axis('equal')
plt.grid(True)
plt.show()




