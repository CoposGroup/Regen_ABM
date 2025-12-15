"""
Script to save the initial rest lengths from t=0 epithelium configuration.
Run this once to generate the rest length file, then these values will be 
used consistently even when loading different epithelium geometries.
"""
import numpy as np
import os
from scipy.sparse import spdiags
from config import BOUNDARY_FILE, INPUT_DIR

# Load initial boundary
Xe0 = np.loadtxt(BOUNDARY_FILE, delimiter=',')

# Compute rest lengths (same as in build_boundaries)
Ne = Xe0.shape[0]
e = np.ones(Ne)
Db = spdiags([-e, e], [0,1], Ne, Ne, format='csr')
Db[Ne-1, 0] = 1

dsb = np.hypot(*(Xe0[1] - Xe0[0]))  # first segment length
blp0 = np.hypot(*(Db @ Xe0).T)      # rest length of edge from i to i+1
blm0 = np.hypot(*(Db.T @ Xe0).T)    # rest length of edge from i-1 to i

# Save to separate .npy files
dsb_file = os.path.join(INPUT_DIR, 'dsb.npy')
blp0_file = os.path.join(INPUT_DIR, 'blp0.npy')
blm0_file = os.path.join(INPUT_DIR, 'blm0.npy')
ne_file = os.path.join(INPUT_DIR, 'Ne.npy')

np.save(dsb_file, dsb)
np.save(blp0_file, blp0)
np.save(blm0_file, blm0)
np.save(ne_file, Ne)

print(f"Saved rest lengths to {INPUT_DIR}:")
print(f"  {dsb_file} (dsb={dsb:.6f})")
print(f"  {blp0_file} (shape {blp0.shape}, mean={blp0.mean():.6f}, std={blp0.std():.6f})")
print(f"  {blm0_file} (shape {blm0.shape}, mean={blm0.mean():.6f}, std={blm0.std():.6f})")
print(f"  {ne_file} (Ne={Ne})")

