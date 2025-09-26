import numpy as np
import matplotlib.pyplot as plt

# Load your epithelium file (update the path as needed)
epithelium_file = 'data/input/epi200.csv'
Xe = np.loadtxt(epithelium_file, delimiter=',')

# 1. Plot the points in order
plt.figure(figsize=(8, 8))
plt.plot(Xe[:, 0], Xe[:, 1], '-o', label='Epithelium (in order)')
plt.scatter(Xe[:, 0], Xe[:, 1], c='r', s=10)

# Plot and label the first 30 points
n_label = min(30, Xe.shape[0])
for i in range(n_label):
    plt.text(Xe[i, 0], Xe[i, 1], str(i), fontsize=9, color='blue', ha='center', va='center')

plt.title('Epithelium Boundary Point Order (First 30 Points Labeled)')
plt.xlabel('x')
plt.ylabel('y')
plt.xlim(-1, 0.01)
plt.ylim(1.25, 1.6)
# plt.axis('equal')
plt.legend()
plt.show()

# 2. Check for large jumps between consecutive points
dists = np.sqrt(np.sum(np.diff(Xe, axis=0)**2, axis=1))
print("Max consecutive distance:", np.max(dists))
print("Mean consecutive distance:", np.mean(dists))
print("Indices of large jumps (>2x mean):", np.where(dists > 2 * np.mean(dists))[0])

# 3. Optional: Check for self-intersections using shapely
try:
    from shapely.geometry import Polygon
    poly = Polygon(Xe)
    print("Is polygon simple (no self-intersections)?", poly.is_simple)
    print("Is polygon valid (no self-intersections, no bowties)?", poly.is_valid)
except ImportError:
    print("Install shapely for self-intersection check: pip install shapely")