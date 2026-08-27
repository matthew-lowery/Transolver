"""Load the MATLAB operator-learning datasets in ``ram_dataset/dataset.py`` format."""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.io import loadmat


N_TEST = 200
N_TEST_OOD = 100
TIME_LEVELS = (0.7, 0.8, 0.9, 1.0)


@dataclass
class OperatorDataset:
    input_points: np.ndarray
    output_points: np.ndarray
    train_input: np.ndarray
    train_output: np.ndarray
    test_input: np.ndarray
    test_output: np.ndarray


def _base_problem(problem):
    problem = problem.lower()
    is_ood = problem.endswith("_ood")
    if is_ood:
        problem = problem[:-4]
    aliases = {
        "taylor_green": "taylor_green_exact",
        "taylor_green_spacetime": "taylor_green_time",
        "taylor_green_spacetime_coeffs": "taylor_green_time_coeffs",
    }
    return aliases.get(problem, problem)


def _data_path(root, problem, ood=False):
    root = Path(root)
    suffix = "_ood" if ood else ""
    if problem == "flow_cylinder_laminar":
        return root / "flow_cylinder" / f"data_laminar{suffix}.mat"
    if problem == "flow_cylinder_shedding":
        return root / "flow_cylinder" / f"data_shedding{suffix}.mat"
    if problem in {"taylor_green", "taylor_green_exact", "taylor_green_coeffs", "taylor_green_time", "taylor_green_time_coeffs"}:
        if problem in {"taylor_green", "taylor_green_exact"}:
            return root / "taylor_green" / "data_exact_matt.mat"
        return root / "taylor_green" / f"data_{'time' if 'time' in problem else 'coeffs' if 'coeffs' in problem else 'exact'}.mat"
    return root / problem / f"data{suffix}.mat"


def _load_mat(root, problem, ood=False):
    data = loadmat(_data_path(root, problem, ood))
    if problem == "taylor_green_time_coeffs" and not ood:
        coeffs = loadmat(Path(root) / "taylor_green" / "data_coeffs.mat")
        return data, coeffs
    return data, None


def _fekete_indices(root, problem, size):
    if problem == "flow_cylinder_laminar":
        path = Path(root) / "flow_cylinder" / "flow_cylinder_laminar_fekete.mat"
    elif problem == "flow_cylinder_shedding":
        path = Path(root) / "flow_cylinder" / "flow_cylinder_shedding_fekete.mat"
    elif problem in {"taylor_green_exact", "taylor_green_coeffs", "taylor_green_time", "taylor_green_time_coeffs"}:
        path = Path(root) / "taylor_green" / "taylor_green_fekete.mat"
    else:
        path = Path(root) / problem / f"{problem}_fekete.mat"
    if not path.exists():
        return np.arange(size)
    indices = np.asarray(loadmat(path)["E"]).squeeze().astype(int)
    if indices.size and indices.min() >= 1 and indices.max() <= size:
        indices = indices - 1
    return indices


def _point_filter(problem, points, point_count, root):
    points = np.asarray(points, dtype=np.float64)
    keep = np.arange(len(points))
    if problem in {
        "taylor_green_exact", "taylor_green_coeffs", "taylor_green_time", "taylor_green_time_coeffs",
        "merge_vortices_easier",
    }:
        keep = np.flatnonzero(~((points[:, 0] == 0) | (points[:, 1] == 0)))
    elif problem == "forced_turb":
        keep = np.flatnonzero(~np.any(np.isclose(points, 2 * np.pi), axis=1))

    points = points[keep]
    if point_count is None or point_count >= len(points):
        return points, keep, np.arange(len(points))

    fekete = _fekete_indices(root, problem, len(points))
    selected = fekete[:point_count]
    return points[selected], keep, selected


def _point_values(values, keep, selected, full_point_count):
    values = np.asarray(values)
    if values.ndim >= 2 and values.shape[1] == full_point_count:
        values = values[:, keep]
        values = values[:, selected]
    return values


def _with_channel(values):
    values = np.asarray(values)
    return values[..., None] if values.ndim == 2 else values


def _split_indices(count, ntrain, test_count):
    if ntrain > count - test_count:
        raise ValueError(f"ntrain={ntrain} leaves fewer than {test_count} test functions")
    perm = np.random.default_rng(seed=0).permutation(count)
    return perm[:ntrain], perm[-test_count:]


def _build_values(problem, data, coeffs, indices, points, keep, selected, full_point_count):
    velocity = _point_values(data["velocity"], keep, selected, full_point_count)[indices]

    if problem in {"flow_cylinder_laminar", "flow_cylinder_shedding", "lid_cavity_flow"}:
        key = "init_velocity"
        inputs = _point_values(data[key], keep, selected, full_point_count)[indices]
        return _with_channel(inputs), _with_channel(velocity)

    if problem == "buoyancy_cavity_flow":
        inputs = _point_values(data["init_temperature"], keep, selected, full_point_count)[indices]
        return _with_channel(inputs), _with_channel(velocity)

    if problem == "backward_facing_step":
        inlet = np.asarray(data["init_velocity"])[indices].copy()
        inlet[:, [0, 1]] = inlet[:, [1, 0]]
        inputs = np.zeros((len(indices), len(points), 2), dtype=velocity.dtype)
        left = np.flatnonzero(np.isclose(points[:, 0], points[:, 0].min()))
        inputs[:, left, 0] = inlet
        return inputs, _with_channel(velocity)

    if problem == "merge_vortices_easier":
        inputs = _point_values(data["init_vorticity"], keep, selected, full_point_count)[indices]
        return _with_channel(inputs), _with_channel(velocity)

    if problem == "species_transport":
        inputs = np.asarray(data["init_velocity"])[indices]
        inputs = _with_channel(inputs)
        if inputs.shape[1] < len(points):
            pad = np.zeros((len(inputs), len(points) - inputs.shape[1], inputs.shape[-1]), dtype=inputs.dtype)
            inputs = np.concatenate((inputs, pad), axis=1)
        elif inputs.shape[1] > len(points):
            inputs = inputs[:, :len(points)]
        return inputs, _with_channel(velocity)

    if problem == "forced_turb":
        inputs = _point_values(data["forcing"], keep, selected, full_point_count)[indices]
        return _with_channel(inputs), _with_channel(velocity)

    if problem == "taylor_green_coeffs":
        inputs = np.asarray(data["init_coeffs"])[indices]
        return inputs, _with_channel(velocity)

    if problem == "taylor_green_exact":
        inputs = _point_values(data["init_velocity"], keep, selected, full_point_count)[indices]
        return _with_channel(inputs), _with_channel(velocity)

    if problem in {"taylor_green_time", "taylor_green_time_coeffs"}:
        if problem.endswith("coeffs"):
            inputs = np.asarray(coeffs["init_coeffs"])[indices]
        else:
            inputs = _point_values(data["init_velocity"], keep, selected, full_point_count)[indices]
        outputs = np.stack([
            _point_values(data[f"vel_{level.replace('.', '')}"], keep, selected, full_point_count)[indices]
            for level in TIME_LEVELS
        ], axis=2)
        return inputs, outputs

    raise ValueError(f"Unsupported dataset: {problem}")


def _format_geometry(problem, points, train_input, test_input, train_output, test_output):
    if problem in {"taylor_green_time", "taylor_green_time_coeffs"}:
        input_points = np.column_stack((points, np.zeros(len(points))))
        output_points = np.repeat(input_points, len(TIME_LEVELS), axis=0)
        output_points[:, 2] = np.tile(TIME_LEVELS, len(points))
        if problem.endswith("coeffs"):
            train_input = np.repeat(train_input[:, None, :], len(input_points), axis=1)
            test_input = np.repeat(test_input[:, None, :], len(input_points), axis=1)
        else:
            train_input = np.asarray(train_input)
            test_input = np.asarray(test_input)
        train_output = train_output.reshape(len(train_output), len(output_points), -1)
        test_output = test_output.reshape(len(test_output), len(output_points), -1)
        return input_points, output_points, train_input, train_output, test_input, test_output

    if problem == "taylor_green_coeffs":
        train_input = np.repeat(train_input[:, None, :], len(points), axis=1)
        test_input = np.repeat(test_input[:, None, :], len(points), axis=1)

    if problem == "species_transport":
        input_points = output_points = points
    else:
        input_points = output_points = points
    return input_points, output_points, train_input, train_output, test_input, test_output


def load_dataset(problem, ntrain, point_count, data_root, test_count=N_TEST):
    """Load a training split and the fixed held-out split."""
    problem = _base_problem(problem)
    data, coeffs = _load_mat(data_root, problem)
    raw_points = np.asarray(data["points"])
    if raw_points.ndim > 2:
        raw_points = raw_points.reshape(-1, raw_points.shape[-1])
    points, keep, selected = _point_filter(problem, raw_points, point_count, data_root)
    train_idx, test_idx = _split_indices(np.asarray(data["velocity"]).shape[0], ntrain, test_count)
    train_input, train_output = _build_values(problem, data, coeffs, train_idx, points, keep, selected, len(raw_points))
    test_input, test_output = _build_values(problem, data, coeffs, test_idx, points, keep, selected, len(raw_points))
    geometry = _format_geometry(problem, points, train_input, test_input, train_output, test_output)
    return OperatorDataset(*geometry)


def load_ood_dataset(problem, point_count, data_root, test_count=N_TEST_OOD):
    """Load the deterministic OOD evaluation split for ``problem``."""
    problem = _base_problem(problem)
    data, coeffs = _load_mat(data_root, problem, ood=True)
    raw_points = np.asarray(data["points"])
    if raw_points.ndim > 2:
        raw_points = raw_points.reshape(-1, raw_points.shape[-1])
    points, keep, selected = _point_filter(problem, raw_points, point_count, data_root)
    count = np.asarray(data["velocity"]).shape[0]
    indices = np.random.default_rng(seed=0).permutation(count)[:test_count]
    test_input, test_output = _build_values(problem, data, coeffs, indices, points, keep, selected, len(raw_points))
    geometry = _format_geometry(problem, points, test_input[:0], test_input, test_output[:0], test_output)
    input_points, output_points, _, _, test_input, test_output = geometry
    return input_points, output_points, test_input, test_output
