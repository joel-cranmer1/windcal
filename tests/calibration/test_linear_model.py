import unittest
from typing import Any

import numpy as np

from windcal.core.io import CalibrationDataSet, STANDARD_CHANNELS
from windcal.calibration.models import LinearModel


class TestLinearModel(unittest.TestCase):

    def setUp(self):
        self.model = LinearModel()

    def test_fit_returns_expected_shapes(self):
        data, _, _ = self._create_simple_dataset()

        C, a = self.model.fit(data)

        self.assertEqual(C.shape, (6, 6))
        self.assertEqual(a.shape, (6,))

    def test_fit_raises_error_on_invalid_shape(self):
        """Tests that fit() raises a ValueError for non N x 6 inputs."""
        # Create data with 5 columns instead of 6
        loads_invalid = np.random.rand(10, 5)
        voltages_invalid = np.random.rand(10, 5)

        # The error should be caught by CalibrationDataSet first
        with self.assertRaisesRegex(ValueError, "must have exactly 6 channels"):
            CalibrationDataSet(loads_invalid, voltages_invalid, [])

        # Test for shape mismatch between loads and voltages
        loads_valid = np.random.rand(10, 6)
        voltages_mismatch = np.random.rand(11, 6)  # 11 rows instead of 10
        with self.assertRaisesRegex(ValueError, "have the same shape"):
            CalibrationDataSet(loads_valid, voltages_mismatch, [])

    def test_fit_known_solution(self):
        data, expected_C, expected_a = self._create_simple_dataset()

        C, a = self.model.fit(data)

        # Check against known expected values
        np.testing.assert_array_almost_equal(C, expected_C)
        np.testing.assert_array_almost_equal(a, expected_a)

    def test_fit_random_solution(self):
        """Tests if the model can solve for a known solution from random data."""
        data, expected_C, expected_a = self._create_random_dataset(seed=42)
        C, a = self.model.fit(data)
        np.testing.assert_array_almost_equal(C, expected_C)
        np.testing.assert_array_almost_equal(a, expected_a)

    def test_reduce_inverts_fit(self):
        """Tests if reduce(fit(data)) returns the original loads."""
        # 1. Create a known dataset
        data, _, _ = self._create_random_dataset(seed=123)

        # 2. Fit the model to the data
        C_fit, a_fit = self.model.fit(data)

        # 3. Reduce the voltages using the fitted coefficients
        loads_reduced = self.model.reduce(C_fit, a_fit, data.voltages)

        # 4. Check if the result matches the original loads
        np.testing.assert_array_almost_equal(data.loads, loads_reduced)

    def test_reduce_handles_1d_and_2d_voltages(self):
        """Tests that 'reduce' works correctly for single and multiple voltage vectors."""
        C = np.random.rand(6, 6)
        a = np.random.rand(6)

        # Test with a single voltage vector (1D)
        voltages_1d = np.random.rand(6)
        loads_1d = self.model.reduce(C, a, voltages_1d)
        self.assertEqual(loads_1d.shape, (6,))

        # Test with multiple voltage vectors (2D)
        voltages_2d = np.random.rand(10, 6)
        loads_2d = self.model.reduce(C, a, voltages_2d)
        self.assertEqual(loads_2d.shape, (10, 6))

    def _create_simple_dataset(self):
        """
        Creates a simple dataset of loads and calculates the corresponding
        voltages based on a known calibration matrix and bias vector.
        """
        # Known coefficients
        C_expected = np.eye(6)
        a_expected = np.ones(6) * 2

        # loads
        loads = np.eye(6)
        loads = np.append(loads, np.eye(6) * 2, axis=0)
        voltages = loads @ C_expected.T + a_expected

        data = CalibrationDataSet(loads, voltages, STANDARD_CHANNELS)
        return data, C_expected, a_expected

    def _create_random_dataset(self, n_points: int = 100, seed: Any = None):
        """
        Creates a dataset with random loads and calculates the corresponding
        voltages based on a known calibration matrix and bias vector.
        """
        if seed is not None:
            np.random.seed(seed)

        # 1. Create known, random coefficients for the expected solution, but keep it mostly on the diagonal
        A = np.random.rand(6, 6)
        B = np.eye(6) * 100
        C = A + B
        col_norms = np.linalg.norm(C, axis=0, keepdims=True)
        C_expected = C / col_norms
        a_expected = np.random.rand(6) * 5

        # loads → identity matrix
        # 2. Create random but realistic loads
        # Using a larger number of points (n_points > 7) ensures the system
        # is well-defined for the least-squares solver.
        loads = (np.random.rand(n_points, 6) - 0.5) * 1000

        # 3. Calculate the resulting voltages using the model's equation
        # The fit method solves R = G @ C.T + a (where G=loads, R=voltages)
        # So we generate the voltages using this forward equation.
        voltages = loads @ C_expected.T + a_expected

        # 4. Create the dataset object
        data = CalibrationDataSet(loads, voltages, STANDARD_CHANNELS)

        return data, C_expected, a_expected


if __name__ == '__main__':
    unittest.main()
