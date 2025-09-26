import numpy as np
import os

# Import config parameters
import importlib.util
config_path = os.path.join(os.path.dirname(__file__), 'config.py')
spec = importlib.util.spec_from_file_location('config', config_path)
config = importlib.util.module_from_spec(spec)
spec.loader.exec_module(config)

# Load epithelium boundary file
epi_file = os.path.join(os.path.dirname(__file__), 'data', 'input', 'epi200.csv')
Xe = np.loadtxt(epi_file, delimiter=',')

# Compute segment lengths (ds) for boundary
# ds is the spacing between boundary points, used for boundary CFL
# DL_CRIT is used for cell-cell and cell-boundary repulsion CFL

diffs = np.diff(Xe, axis=0)
dists = np.linalg.norm(diffs, axis=1)
ds_min = np.min(dists)
ds_mean = np.mean(dists)

print(f"Boundary segment length (ds):")
print(f"  Minimum ds: {ds_min:.6f}")
print(f"  Mean ds:    {ds_mean:.6f}")

# Get parameters from config
XI = config.XI
KB_MAX = config.KB_MAX
KBEND = config.KBEND
DL_CRIT = config.DL_CRIT
K_CC = config.K_CC
K_BC = config.K_BC
K_BONE = config.K_BONE
DT = config.DT

# CFL for boundary stretch (use ds_min)
cfl_no_bend = XI * ds_min / KB_MAX
print(f"CFL condition for boundary stretch (no bending): DT < {cfl_no_bend:.2e} (uses ds_min)")

# CFL for boundary bending (use ds_min)
cfl_bend = XI * ds_min**4 / (8 * KBEND)
print(f"CFL condition for boundary bending: DT < {cfl_bend:.2e} (uses ds_min)")

# CFL for cell-cell, cell-boundary, and bone repulsion (use DL_CRIT)
cfl_cc = XI * DL_CRIT / K_CC
cfl_bc = XI * DL_CRIT / K_BC
cfl_bone = XI * DL_CRIT / K_BONE

print(f"CFL condition for cell-cell repulsion:     DT < {cfl_cc:.2e} (uses DL_CRIT)")
print(f"CFL condition for cell-boundary spring:    DT < {cfl_bc:.2e} (uses DL_CRIT)")
print(f"CFL condition for bone repulsion:          DT < {cfl_bone:.2e} (uses DL_CRIT)")

# Check your DT against all CFL limits
print(f"\nYour config.DT = {DT:.2e}")

if DT >= cfl_no_bend:
    print("WARNING: DT exceeds CFL for boundary stretch (no bending)!")
if DT >= cfl_bend:
    print("WARNING: DT exceeds CFL for boundary bending!")
if DT >= cfl_cc:
    print("WARNING: DT exceeds CFL for cell-cell repulsion!")
if DT >= cfl_bc:
    print("WARNING: DT exceeds CFL for cell-boundary spring!")
if DT >= cfl_bone:
    print("WARNING: DT exceeds CFL for bone repulsion!") 