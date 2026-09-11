"""MATLAB dataset adapter for the specialized training entry points."""

import numpy as np

from ram_dataset_loader import DEFAULT_DATA_ROOT, load_dataset


def add_dataset_arguments(parser, default_points="all"):
    parser.add_argument(
        "--data-root", "--dir", dest="data_root",
        default=str(DEFAULT_DATA_ROOT),
        help="Directory containing the RAM MATLAB dataset folders",
    )
    parser.add_argument("--npoints", default=default_points)


def load_training_arrays(args):
    count = None if str(args.npoints) in {"all", "0"} else int(args.npoints)
    dataset = load_dataset(args.dataset, args.ntrain, count, args.data_root)
    input_points = dataset.input_points
    train_input, test_input = dataset.train_input, dataset.test_input
    if args.dataset.endswith("coeffs"):
        train_input, test_input = train_input[:, 0, :], test_input[:, 0, :]
        input_points = np.zeros((2, dataset.output_points.shape[1]))
        input_points[0, 1] = 1
    return {
        "x_grid": input_points,
        "y_grid": dataset.output_points,
        "x_train": train_input,
        "x_test": test_input,
        "y_train": dataset.train_output,
        "y_test": dataset.test_output,
    }
