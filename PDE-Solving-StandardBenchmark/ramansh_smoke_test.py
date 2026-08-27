import argparse
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from scipy.spatial import cKDTree

from model_dict import get_model
from ram_dataset_loader import load_dataset
from utils.normalizer import UnitTransformer


DATASETS = (
    "flow_cylinder_laminar",
    "flow_cylinder_shedding",
    "backward_facing_step",
    "buoyancy_cavity_flow",
    "lid_cavity_flow",
    "merge_vortices_easier",
    "species_transport",
    "forced_turb",
    "taylor_green_exact",
    "taylor_green_coeffs",
    "taylor_green_spacetime",
    "taylor_green_spacetime_coeffs",
)


def union_grid(input_points, output_points):
    grid = np.unique(np.concatenate((input_points, output_points)), axis=0)
    input_indices = cKDTree(grid).query(input_points, k=1)[1]
    output_indices = cKDTree(grid).query(output_points, k=1)[1]
    return grid, input_indices, output_indices


def smoke_test(dataset_name, data_root, point_count):
    dataset = load_dataset(
        dataset_name,
        ntrain=2,
        point_count=point_count,
        data_root=data_root,
        test_count=2,
    )
    grid, input_indices, output_indices = union_grid(
        dataset.input_points, dataset.output_points
    )
    inputs = np.zeros(
        (2, len(grid), dataset.train_input.shape[-1]), dtype=np.float32
    )
    inputs[:, input_indices] = dataset.train_input
    inputs = torch.tensor(inputs)
    targets = torch.tensor(dataset.train_output, dtype=torch.float32)
    positions = torch.tensor(grid, dtype=torch.float32).repeat(2, 1, 1)
    input_normalizer = UnitTransformer(inputs)
    output_normalizer = UnitTransformer(targets)
    model = get_model(SimpleNamespace(model="Transolver_Irregular_Mesh")).Model(
        space_dim=grid.shape[-1],
        n_layers=1,
        n_hidden=8,
        n_head=2,
        fun_dim=inputs.shape[-1],
        out_dim=targets.shape[-1],
        slice_num=4,
        ref=2,
    )
    predictions = output_normalizer.decode(
        model(positions, fx=input_normalizer.encode(inputs)).squeeze(-1)
    )[:, output_indices]
    if predictions.shape != targets.shape:
        raise AssertionError(
            f"{dataset_name}: {predictions.shape=} != {targets.shape=}"
        )
    loss = torch.mean((predictions - targets) ** 2)
    loss.backward()
    if not torch.isfinite(loss):
        raise AssertionError(f"{dataset_name}: non-finite loss")
    if not all(
        parameter.grad is None or torch.isfinite(parameter.grad).all()
        for parameter in model.parameters()
    ):
        raise AssertionError(f"{dataset_name}: non-finite gradient")
    print(f"{dataset_name}: output={tuple(predictions.shape)}, loss={loss.item():.3e}")


def main():
    default_root = Path(__file__).resolve().parents[2] / "ram_dataset"
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=default_root)
    parser.add_argument("--npoints", type=int, default=8)
    parser.add_argument("datasets", nargs="*", default=DATASETS)
    args = parser.parse_args()
    for dataset_name in args.datasets:
        smoke_test(dataset_name, args.data_root, args.npoints)


if __name__ == "__main__":
    main()
