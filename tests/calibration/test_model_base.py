import unittest
from typing import Any, Tuple

import numpy as np

from windcal.core.io import STANDARD_CHANNELS, CalibrationDataSet, ZeroLoadOutput


class CalibrationModelContractTest(unittest.TestCase):
    """
    Contract test suite for all CalibrationMathModel subclasses.

    Subclasses MUST define:
        MODEL_CLASS
    """
    __test__ = False  # pytest ignores it

    MODEL_CLASS = None
    FIT_ATOL = 1e-7
    FIT_RTOL = 1e-7
    REDUCE_ATOL = 0.01

    feature_list = []
    """List of lambda functions to generate non-linear features"""

    def setUp(self):
        if self.MODEL_CLASS is None:
            self.skipTest(f"Base contract class: {self.__class__.__name__}")
        self.build_zero_load()

    # -------------------------
    # Required factory methods
    # -------------------------

    def create_model(self):
        if self.MODEL_CLASS is None:
            self.fail(f"{self.__class__.__name__} must define MODEL_CLASS")
        return self.MODEL_CLASS()

    def create_simple_dataset(self) -> Tuple[CalibrationDataSet, np.ndarray]:
        """Default simple dataset (override only if needed)"""
        C_expected = np.eye(6)
        a_expected = np.ones(6) * 2

        loads = np.eye(6)
        loads = np.append(loads, np.eye(6) * 2, axis=0)
        voltages = loads @ C_expected.T + a_expected

        data = CalibrationDataSet(loads, voltages, STANDARD_CHANNELS)
        return data, C_expected

    def create_random_dataset(
            self,
            n: int = 6,
            p: int = 100,
            voltage_noise_std: float = 0.001,
            seed: Any = None
    ) -> Tuple[CalibrationDataSet, np.ndarray]:
        """
        Creates a random single-axis loading dataset with customizable feature mapping and noise.

        Args:
            n (int): Number of balance elements (physical channels).
            p (int): Number of load series / data points.
            voltage_noise_std (float): Standard deviation of Gaussian noise added to the voltage signal.
            seed (Any): Random seed for reproducibility.

        Returns:
            Tuple[CalibrationDataSet, np.ndarray]: A tuple containing the generated dataset
                                                   and the expected calibration matrix C.
        """
        if seed is not None:
            np.random.seed(seed)

        feature_count = len(self.feature_list)

        # 1. Generate the Calibration Matrix C (n x m)
        m = n * (1 + feature_count)
        # Start with small random values for baseline cross-talk.
        C = np.random.rand(n, m) * 0.01

        # Add a dominant diagonal to EACH n x n block (linear and non-linear).
        for i in range(1 + feature_count):
            start_col = i * n
            end_col = start_col + n
            C[:, start_col:end_col] += np.eye(n) * 1e5

        # Normalize columns to create the final expected matrix.
        col_norms = np.linalg.norm(C, axis=0, keepdims=True)
        col_norms[col_norms == 0] = 1  # Avoid division by zero.
        C_expected = C / col_norms

        # Damp the non-linear (e.g. absolute-value) blocks relative to the
        # linear block.
        NONLINEAR_SCALE = 0.05
        C_expected[:, n:] *= NONLINEAR_SCALE

        # Generate bias (a)
        a_expected = np.random.rand(n) * 0.01

        # 2. Generate Raw Single-Axis Loads L (p rows, n columns)
        p_per_dim = p // n
        remainder = p % n
        loads_list = []
        for dim in range(n):
            count = p_per_dim + (1 if dim < remainder else 0)

            # Create jittered load points for more realistic spacing.
            mags = np.linspace(-500, 500, count)
            mags += np.random.randn(count) * 20

            block = np.zeros((count, n))
            block[:, dim] = mags
            loads_list.append(block)

        loads = np.vstack(loads_list)
        np.random.shuffle(loads)

        # 3. Expand into the Feature Space G (p rows, m columns)
        G_blocks = [loads]
        for func in self.feature_list:
            G_blocks.append(func(loads))
        G = np.hstack(G_blocks)

        # 4. Calculate Voltages
        # Using NumPy row-major format: V(p,n) = G(p,m) @ C(n,m).T + a
        voltages = G @ C_expected.T + a_expected

        # Note: Ensure STANDARD_CHANNELS matches the variable `n` if it's currently hardcoded to 6
        data = CalibrationDataSet(loads, voltages, STANDARD_CHANNELS[:n])

        return data, C_expected

    def build_zero_load(self, n: int = 6):
        self.zlo = ZeroLoadOutput()
        self.zlo.add(np.zeros(n), 0)
        self.zlo.add(np.zeros(n), 90)
        self.zlo.add(np.zeros(n), 180)
        self.zlo.add(np.zeros(n), -90)

    # -------------------------
    # Contract tests
    # -------------------------

    def test_fit_returns_expected_shapes(self):
        model = self.create_model()
        data, _ = self.create_simple_dataset()

        C = model.fit(data)
        n = len(data.channels)
        m = n * (1 + len(self.feature_list or []))
        self.assertEqual(C.shape, (n, m))

    def test_fit_known_solution(self):
        model = self.create_model()
        data, expected_C = self.create_simple_dataset()

        C = model.fit(data)
        np.testing.assert_allclose(C, expected_C, atol=self.FIT_ATOL)

    def test_fit_random_solution(self):
        model = self.create_model()
        data, expected_C = self.create_random_dataset(n=6, p=100, seed=42)

        C = model.fit(data)
        np.testing.assert_allclose(C, expected_C, rtol=self.FIT_RTOL, atol=self.FIT_ATOL)

    def test_reduce_inverts_fit(self):
        model = self.create_model()
        model.zlo = self.zlo
        data, _ = self.create_random_dataset(n=6, p=100, seed=123)

        C_fit = model.fit(data)
        loads_reduced = model.reduce(C_fit, data.voltages)

        np.testing.assert_allclose(loads_reduced, data.loads, atol=self.REDUCE_ATOL)

    def test_reduce_handles_1d_and_2d(self):
        model = self.create_model()
        model.zlo = self.zlo
        n = 6
        m = n * (1 + len(self.feature_list or []))
        C = np.random.rand(n, m)

        v1 = np.random.rand(n)
        l1 = model.reduce(C, v1)
        self.assertEqual(l1.shape, (n,))

        v2 = np.random.rand(10, n)
        l2 = model.reduce(C, v2)
        self.assertEqual(l2.shape, (10, n))

    # -------------------------
    # Validation tests (shared)
    # -------------------------

    def test_invalid_dataset_shapes(self):
        loads_invalid = np.random.rand(10, 5)
        voltages_invalid = np.random.rand(10, 5)

        with self.assertRaisesRegex(ValueError, "must have exactly 6 channels"):
            CalibrationDataSet(loads_invalid, voltages_invalid,
                               ['NF', 'SF', 'AF', 'PM', 'RM'])

        loads_valid = np.random.rand(10, 6)
        voltages_mismatch = np.random.rand(11, 6)

        with self.assertRaisesRegex(ValueError, "have the same shape"):
            CalibrationDataSet(loads_valid, voltages_mismatch,
                               STANDARD_CHANNELS)

    def test_no_valid_channels(self):
        loads = np.random.rand(6, 6)
        voltages = np.random.rand(6, 6)

        with self.assertRaisesRegex(ValueError, "No valid channels found"):
            CalibrationDataSet(loads, voltages, [])
