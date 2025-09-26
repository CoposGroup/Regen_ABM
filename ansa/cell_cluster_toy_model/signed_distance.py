# Experimental vs Simulation Shape Comparison
# Loads experimental data, simulation data, and compares shapes using signed distance functions

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.path import Path
from scipy.integrate import dblquad, qmc_quad
import re

#---------------------------------------------------------------------------------------------------
## SIGNED DISTANCE FUNCTION
#---------------------------------------------------------------------------------------------------

def make_sdf_polygon(boundary_pts):
    """
    Returns a function that computes the signed distance from (x, y) to a closed polygon boundary.
    Uses all segments, not just points.
    """
    from matplotlib.path import Path

    # Ensure closed
    if not np.allclose(boundary_pts[0], boundary_pts[-1]):
        boundary_pts = np.vstack([boundary_pts, boundary_pts[0]])

    path = Path(boundary_pts)

    def sdf(x, y):
        x = np.asarray(x)
        y = np.asarray(y)
        pts = np.column_stack([x.ravel(), y.ravel()])
        min_dist = np.full(pts.shape[0], np.inf)

        # Loop over segments
        for i in range(len(boundary_pts) - 1):
            p1 = boundary_pts[i]
            p2 = boundary_pts[i + 1]
            # Vector from p1 to p2
            d = p2 - p1
            # Vector from p1 to pts
            v = pts - p1
            # Project v onto d, clamp to [0,1]
            t = np.clip(np.dot(v, d) / np.dot(d, d), 0, 1)
            proj = p1 + t[:, None] * d
            dist = np.linalg.norm(pts - proj, axis=1)
            min_dist = np.minimum(min_dist, dist)

        # Inside/outside
        inside = path.contains_points(pts)
        signed_dist = np.where(inside, -min_dist, min_dist)
        return signed_dist.reshape(x.shape)

    return sdf

def comp_function(phi_exp, phi_sim, xmin=-1, xmax=3, ymin=-1.5, ymax=1.5):
    # qmc_quad expects a function that takes a single array argument [x, y]
    integrand = lambda point: np.abs(phi_exp(point[0], point[1]) - phi_sim(point[0], point[1]))**2
    return qmc_quad(integrand, a=np.array([xmin, ymin]), b=np.array([xmax, ymax]))