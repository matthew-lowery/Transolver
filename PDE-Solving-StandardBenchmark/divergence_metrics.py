from itertools import product

import numpy as np
import torch
from scipy.linalg import lstsq
from scipy.spatial import cKDTree


def build_rbf_fd_gradient(points, order=5):
    points = np.asarray(points, dtype=np.float64)
    dimension = points.shape[1]
    powers = np.asarray([p for p in product(range(order + 1), repeat=dimension) if sum(p) <= order])
    polynomial_count = len(powers)
    stencil_size = 2 * polynomial_count + 1
    rbf_power = min(max(order - (order % 2 == 0), 5), 11)
    tree = cKDTree(points)
    rows = np.repeat(np.arange(len(points)), stencil_size)
    columns = np.empty_like(rows)
    weights = np.empty((dimension, len(rows)), dtype=np.float64)
    eps = np.finfo(np.float64).eps
    for center_index, center in enumerate(points):
        distances, stencil = tree.query(center, k=stencil_size)
        scale = distances[-1]
        if not np.isfinite(scale) or scale <= eps:
            raise ValueError("RBF-FD stencil contains coincident points")
        local = (points[stencil] - center) / scale
        pairwise = np.linalg.norm(local[:, None] - local[None, :], axis=-1)
        polynomial = np.prod(local[:, None] ** powers[None, :], axis=-1)
        system = np.block([[pairwise ** rbf_power, polynomial],
                           [polynomial.T, np.zeros((polynomial_count, polynomial_count))]])
        derivative = np.zeros((stencil_size + polynomial_count, dimension))
        derivative[:stencil_size] = -local * rbf_power * (pairwise[0, :, None] + eps) ** (rbf_power - 2) / scale
        for axis in range(dimension):
            unit = np.zeros(dimension, dtype=int)
            unit[axis] = 1
            polynomial_index = np.flatnonzero(np.all(powers == unit, axis=1))[0]
            derivative[stencil_size + polynomial_index, axis] = 1 / scale
        solution = lstsq(system, derivative, lapack_driver="gelsy", check_finite=False)[0]
        block = slice(center_index * stencil_size, (center_index + 1) * stencil_size)
        columns[block] = stencil
        weights[:, block] = solution[:stencil_size].T
    indices = torch.tensor(np.stack((rows, columns)), dtype=torch.long)
    return tuple(torch.sparse_coo_tensor(indices, torch.tensor(axis_weights, dtype=torch.float32),
                                         (len(points), len(points))).coalesce()
                 for axis_weights in weights)


def interior_mask(points):
    points = np.asarray(points)
    minimum = points.min(axis=0)
    maximum = points.max(axis=0)
    return torch.tensor(~((points == minimum) | (points == maximum)).any(axis=1), dtype=torch.bool)


def summarize_divergence(predictions, operators, interior_mask, time_steps=1):
    """Return scalar absolute-divergence summaries across test functions and points."""
    batch, points, components = predictions.shape
    if points % time_steps:
        raise ValueError("Prediction point count is not divisible by time_steps")
    fields = predictions.reshape(batch, -1, time_steps, components)
    fields = fields.permute(0, 2, 1, 3).reshape(batch * time_steps, -1, components)
    divergence = sum(
        torch.sparse.mm(operator, fields[..., axis].T).T
        for axis, operator in enumerate(operators)
    )
    absolute = divergence.abs()
    interior = absolute[:, interior_mask]
    return {
        "test_div/max_abs_all": absolute.max().item(),
        "test_div/median_abs_all": absolute.reshape(-1).median().item(),
        "test_div/max_abs_interior": interior.max().item(),
        "test_div/median_abs_interior": interior.reshape(-1).median().item(),
    }
