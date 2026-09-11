"""Divergence-only masks using the reference dataset.py heuristics."""

from pathlib import Path

import numpy as np
import torch
from scipy.io import loadmat
from scipy.spatial import cKDTree


def divergence_interior_mask(points, problem, data_root):
    points = np.asarray(points, dtype=np.float64)
    problem = problem.lower().removesuffix("_ood")
    if problem == "species_transport":
        path = Path(data_root) / problem / "full_domain_boundary_points.mat"
        boundary = np.asarray(loadmat(path)["boundary_points"]).reshape(-1, 3)
        distances, _ = cKDTree(boundary).query(points, p=np.inf)
        keep = distances > 1e-12
    elif problem == "forced_turb":
        # Move the reference loader's periodic-copy omission into this mask.
        keep = ~np.any(np.isclose(points, 2 * np.pi), axis=1)
    else:
        if problem.startswith(("taylor_green", "merge_vortices")):
            lower, upper = (0., 0.), (2 * np.pi, 2 * np.pi)
        elif problem in {"lid_cavity_flow", "buoyancy_cavity_flow"}:
            lower, upper = (0., 0.), (1., 1.)
        elif problem.startswith("flow_cylinder"):
            lower, upper = (0., 0.), (20., 14.)
        elif problem.startswith("backward_facing_step"):
            lower, upper = (0., -.5), (15., .5)
        else:
            raise ValueError(f"No divergence boundary heuristic for {problem}")
        x, y = points[:, 0], points[:, 1]
        x0, y0 = lower
        x1, y1 = upper
        tol = 1e-12
        boundary = (
            (np.isclose(x, x0, atol=tol) | np.isclose(x, x1, atol=tol))
            & (y >= y0 - tol) & (y <= y1 + tol)
        ) | (
            (np.isclose(y, y0, atol=tol) | np.isclose(y, y1, atol=tol))
            & (x >= x0 - tol) & (x <= x1 + tol)
        )
        keep = ~boundary
    if not np.any(keep):
        raise ValueError(f"No divergence-interior points selected for {problem}")
    return torch.tensor(keep, dtype=torch.bool)
