import unittest
from typing import Any, Tuple
import numpy as np

from windcal.core.io import CalibrationDataSet, STANDARD_CHANNELS


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

    def setUp(self):
        if self.MODEL_CLASS is None:
            self.skipTest(f"Base contract class: {self.__class__.__name__}")

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
        self, n_points: int = 100, seed: Any = None
    ) -> Tuple[CalibrationDataSet, np.ndarray]:

        if seed is not None:
            np.random.seed(seed)

        A = np.random.rand(6, 6)
        B = np.eye(6) * 1e5
        C = A + B
        col_norms = np.linalg.norm(C, axis=0, keepdims=True)
        C_expected = C / col_norms
        a_expected = np.random.rand(6) * 0.01

        n_per_dim = n_points // 6
        remainder = n_points % 6

        loads_list = []
        for dim in range(6):
            count = n_per_dim + (1 if dim < remainder else 0)

            mags = np.linspace(-500, 500, count)
            mags += np.random.randn(count) * 20

            block = np.zeros((count, 6))
            block[:, dim] = mags
            loads_list.append(block)

        loads = np.vstack(loads_list)
        np.random.shuffle(loads)

        assert np.all(np.count_nonzero(loads, axis=1) == 1)

        voltages = loads @ C_expected.T + a_expected
        data = CalibrationDataSet(loads, voltages, STANDARD_CHANNELS)

        return data, C_expected

    # -------------------------
    # Contract tests
    # -------------------------

    def test_fit_returns_expected_shapes(self):
        model = self.create_model()
        data, _ = self.create_simple_dataset()

        C = model.fit(data)
        self.assertEqual(C.shape, (6, 6))

    def test_fit_known_solution(self):
        model = self.create_model()
        data, expected_C = self.create_simple_dataset()

        C = model.fit(data)
        np.testing.assert_allclose(C, expected_C, atol=self.FIT_ATOL)

    def test_fit_random_solution(self):
        model = self.create_model()
        data, expected_C = self.create_random_dataset(seed=42)

        C = model.fit(data)
        np.testing.assert_allclose(C, expected_C, rtol=self.FIT_RTOL)

    def test_reduce_inverts_fit(self):
        model = self.create_model()
        data, _ = self.create_random_dataset(seed=123)

        C_fit = model.fit(data)
        loads_reduced = model.reduce(C_fit, data.voltages)

        np.testing.assert_allclose(
            loads_reduced,
            data.loads,
            atol=self.REDUCE_ATOL
        )

    def test_reduce_handles_1d_and_2d(self):
        model = self.create_model()
        C = np.random.rand(6, 6)

        v1 = np.random.rand(6)
        l1 = model.reduce(C, v1)
        self.assertEqual(l1.shape, (6,))

        v2 = np.random.rand(10, 6)
        l2 = model.reduce(C, v2)
        self.assertEqual(l2.shape, (10, 6))

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
