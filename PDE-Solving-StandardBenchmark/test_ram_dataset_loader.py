import tempfile
import unittest
from contextlib import ExitStack
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
from scipy.io import savemat

import ram_dataset_loader as loader
from ram_training_data import load_training_arrays


class DatasetSelectionTest(unittest.TestCase):
    def setUp(self):
        self.context = ExitStack()
        self.addCleanup(self.context.close)
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.directory = self.root / "taylor_green"
        self.directory.mkdir()
        self.points = np.array([
            [0., 0.], [1., 0.], [.2, .3], [.4, .7],
            [.8, 1.1], [1.2, .5], [1.5, 1.8], [2., 1.4],
        ])
        self.order = np.array([1, 8, 2, 6, 3, 7, 4, 5])
        self.values = np.arange(6 * 8 * 2).reshape(6, 8, 2)
        self.coefficients = np.arange(12).reshape(6, 2)
        savemat(self.directory / "taylor_green_fekete.mat", {"E": self.order})
        savemat(self.directory / "data_exact_matt.mat", {
            "points": self.points,
            "init_velocity": self.values,
            "velocity": 2 * self.values,
        })
        savemat(self.directory / "data_coeffs_matt.mat", {
            "points": self.points,
            "init_coeffs": self.coefficients,
            "velocity": 2 * self.values,
        })
        time_data = {"points": self.points, "init_velocity": self.values}
        time_data.update({f"vel_{t}": t * self.values for t in [7, 8, 9, 10]})
        savemat(self.directory / "data_time.mat", time_data)
        savemat(self.directory / "data_time_ood.mat", time_data)
        self.context.enter_context(patch.dict(loader.TRAIN_SAMPLE_COUNTS, {
            key: 4 for key in loader.TRAIN_SAMPLE_COUNTS if key.startswith("taylor_green")
        }))
        self.context.enter_context(patch.object(loader, "N_TEST", 2))

    def test_boundary_points_stay_in_functions(self):
        data = loader.load_dataset("taylor_green", 2, 3, self.root, test_count=2)
        selected = [0, 7, 1]
        samples = np.random.default_rng(0).permutation(6)[:2]
        np.testing.assert_array_equal(data.output_points, self.points[selected])
        np.testing.assert_array_equal(data.train_input, self.values[samples][:, selected])
        np.testing.assert_array_equal(data.train_output, 2 * self.values[samples][:, selected])

    def test_coefficients_use_same_output_mesh(self):
        velocity = loader.load_dataset("taylor_green", 2, 3, self.root, test_count=2)
        coefficients = loader.load_dataset("taylor_green_coeffs", 2, 3, self.root, test_count=2)
        np.testing.assert_array_equal(coefficients.output_points, velocity.output_points)
        np.testing.assert_array_equal(coefficients.train_output, velocity.train_output)

    def test_spacetime_order_and_ood(self):
        data = loader.load_dataset("taylor_green_spacetime", 2, 3, self.root, test_count=2)
        self.assertEqual(data.train_output.shape, (2, 12, 2))
        np.testing.assert_array_equal(data.output_points[:4, 2], loader.TIME_LEVELS)
        for step, multiplier in enumerate([7, 8, 9, 10]):
            np.testing.assert_array_equal(
                data.train_output.reshape(2, 3, 4, 2)[:, :, step],
                multiplier * data.train_input,
            )
        _, _, inputs, outputs = loader.load_ood_dataset(
            "taylor_green_spacetime", 3, self.root, test_count=2,
        )
        self.assertEqual(inputs.shape, (2, 3, 2))
        self.assertEqual(outputs.shape, (2, 12, 2))

    def test_zero_based_and_one_based_permutations_agree(self):
        expected = loader._point_filter("taylor_green_exact", self.points, 3, self.root)[0]
        savemat(self.directory / "taylor_green_fekete.mat", {"E": self.order - 1})
        actual = loader._point_filter("taylor_green_exact", self.points, 3, self.root)[0]
        np.testing.assert_array_equal(actual, expected)

    def test_invalid_permutation_is_rejected(self):
        for values in [[1, 1, 3], [1, 2, 99], [1, 2.5, 3]]:
            with self.subTest(values=values):
                savemat(self.directory / "taylor_green_fekete.mat", {"E": values})
                with self.assertRaises(ValueError):
                    loader._point_filter("taylor_green_exact", self.points, 3, self.root)

    def test_training_cannot_overlap_test_split(self):
        with self.assertRaises(ValueError):
            loader.load_dataset("taylor_green", 5, 3, self.root, test_count=2)

    def test_specialized_adapter_uses_requested_point_count(self):
        for name in ["taylor_green", "taylor_green_coeffs", "taylor_green_spacetime",
                     "taylor_green_spacetime_coeffs"]:
            with self.subTest(name=name):
                args = SimpleNamespace(dataset=name, ntrain=2, npoints="3", data_root=self.root)
                with patch.object(loader, "_split_indices", return_value=(np.array([0, 1]), np.array([4, 5]))):
                    data = load_training_arrays(args)
                count = 12 if "spacetime" in name else 3
                self.assertEqual(data["y_train"].shape, (2, count, 2))
                self.assertEqual(data["y_grid"].shape[0], count)
                if name.endswith("coeffs"):
                    np.testing.assert_array_equal(data["x_train"], self.coefficients[:2])
                    self.assertEqual(data["x_grid"].shape, (2, data["y_grid"].shape[1]))


if __name__ == "__main__":
    unittest.main()
