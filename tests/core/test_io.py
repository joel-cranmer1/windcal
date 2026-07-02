import logging
import os
import unittest

import numpy as np

from windcal.core.io import CalibrationDataSet, ZeroLoadOutput

logger = logging.getLogger(__name__)


class TestCalibrationDataSet(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.base_dir = os.path.dirname(os.path.abspath(__file__))
        # Define the exact path to static fixture directory
        cls.fixture_dir = os.path.join(cls.base_dir, "fixtures")

    def test_load_csv(self):
        data_set = CalibrationDataSet.from_csv(os.path.join(self.fixture_dir, "Test_cal_data_SymLin.csv"))
        self.assertTrue(data_set.loads.size > 0)
        self.assertTrue(data_set.voltages.size > 0)

    def test_load_csv_extra_ch(self):
        with self.assertRaises(ValueError):
            data_set = CalibrationDataSet.from_csv(os.path.join(self.fixture_dir, "Test_cal_data_extra_ch.csv"))
            logger.info(f"Loads: {data_set.loads.size}")
            logger.info(f"Volts: {data_set.voltages.size}")


class TestCalibrationDataSetOrdering(unittest.TestCase):
    def setUp(self):
        # Canonical subset we'll test against
        self.channels = ["PF", "PA", "YF", "YA", "AF", "RM"]

        # Create deterministic test data
        # Each column will be easily identifiable by its value
        self.loads = np.array([[10, 20, 30, 40, 50, 60]])

        self.voltages = np.array([[1, 2, 3, 4, 5, 6]])

    def test_order_preserved_when_already_correct(self):
        ds = CalibrationDataSet(self.loads.copy(), self.voltages.copy(), self.channels.copy())

        self.assertEqual(ds.channels, ["PF", "PA", "YF", "YA", "AF", "RM"])
        np.testing.assert_allclose(ds.loads, self.loads)
        np.testing.assert_allclose(ds.voltages, self.voltages)

    def test_order_corrected_when_shuffled(self):
        reorder = [2, 0, 1, 3, 5, 4]  # Shuffled IDs
        # shuffle columns accordingly
        shuffled_channels = [self.channels[i] for i in reorder]
        loads = self.loads[:, reorder]
        voltages = self.voltages[:, reorder]

        ds = CalibrationDataSet(loads, voltages, shuffled_channels)

        # Expected canonical order based on BALANCE_CHANNELS
        expected_channels = ["PF", "PA", "YF", "YA", "AF", "RM"]

        self.assertEqual(ds.channels, expected_channels)

        # Verify loads reordered correctly
        expected_loads = np.array([[10, 20, 30, 40, 50, 60]])
        expected_voltages = np.array([[1, 2, 3, 4, 5, 6]])

        np.testing.assert_allclose(ds.loads, expected_loads)
        np.testing.assert_allclose(ds.voltages, expected_voltages)

    def test_multiple_random_permutations(self):
        import itertools

        canonical = ["PF", "PA", "YF", "YA", "AF", "RM"]

        for perm in itertools.permutations(range(6)):
            permuted_channels = [canonical[i] for i in perm]

            loads = self.loads[:, list(perm)]
            voltages = self.voltages[:, list(perm)]

            ds = CalibrationDataSet(loads, voltages, permuted_channels)

            # Always expect canonical order
            self.assertEqual(ds.channels, canonical)

            np.testing.assert_allclose(ds.loads, self.loads)
            np.testing.assert_allclose(ds.voltages, self.voltages)

    def test_ignores_extra_channels(self):
        channels = ["XX", "PF", "PA", "YY", "YF", "YA", "ZZ", "AF", "RM"]

        loads = np.array([[999, 10, 20, 888, 30, 40, 777, 50, 60]])
        voltages = np.array([[999, 1, 2, 888, 3, 4, 777, 5, 6]])

        ds = CalibrationDataSet(loads, voltages, channels)

        self.assertEqual(ds.channels, ["PF", "PA", "YF", "YA", "AF", "RM"])

        np.testing.assert_allclose(ds.loads, [[10, 20, 30, 40, 50, 60]])
        np.testing.assert_allclose(ds.voltages, [[1, 2, 3, 4, 5, 6]])


class TestZeroLoadOutput(unittest.TestCase):
    def setUp(self) -> None:
        self.ZLO = ZeroLoadOutput()

    def test_ave_fail_empty(self):
        with self.assertRaises(ValueError) as cm:
            self.ZLO.average()

        self.assertIn("Cannot calculate average", str(cm.exception))

    def test_ave_fail_partial(self):
        # add some points but not all
        self.ZLO.add(np.zeros(6), 0)
        self.ZLO.add(np.zeros(6), 90)

        with self.assertRaises(ValueError) as cm:
            self.ZLO.average()

        self.assertIn("Cannot calculate average", str(cm.exception))

    def test_ave_full(self):
        # Full points should cancel
        self.ZLO = ZeroLoadOutput(num_bridges=3)
        self.ZLO.add([1, 2, 3], 0)
        self.ZLO.add([4, 5, 6], 90)
        self.ZLO.add([-1, -2, -3], 180)
        self.ZLO.add([-4, -5, -6], -90)

        expected = np.array([[0.0, 0.0, 0.0]])

        np.testing.assert_allclose(self.ZLO.average(), expected, strict=True)
        # Check that the value is also stored
        np.testing.assert_allclose(self.ZLO.final_average, expected, strict=True)

    def test_add_point(self):
        prev = len(self.ZLO.Z[0])
        self.ZLO.add([1, 2, 3, 4, 5, 6], 0)
        post = len(self.ZLO.Z[0])
        # Difference should be 1
        self.assertEqual(1, post - prev)

    def test_add_too_short(self):
        with self.assertRaises(ValueError) as cm:
            self.ZLO.add([1, 2, 3], 0)
        self.assertIn("Data must be a 1D array of length", str(cm.exception))

    def test_add_2D_array(self):
        with self.assertRaises(ValueError) as cm:
            self.ZLO.add([[1, 2, 3, 4, 5, 6], [1, 2, 3, 4, 5, 6]], 0)
        self.assertIn("Data must be a 1D array of length", str(cm.exception))

    def test_add_invalid_orientation(self):
        with self.assertRaises(ValueError) as cm:
            self.ZLO.add([[1, 2, 3, 4, 5, 6]], 42)
        self.assertIn("Orientation must be one", str(cm.exception))

    def test_add_point_clears_ave(self):
        # Full points should cancel
        self.ZLO = ZeroLoadOutput(num_bridges=3)
        self.ZLO.add([1, 2, 3], 0)
        self.ZLO.add([4, 5, 6], 90)
        self.ZLO.add([-1, -2, -3], 180)
        self.ZLO.add([-4, -5, -6], -90)

        self.ZLO.average()
        self.assertIsNotNone(self.ZLO.final_average)
        self.assertIsInstance(self.ZLO.final_average, np.ndarray)

        # Adding a point should set average to None
        self.ZLO.add([4.1, 5.1, 6.1], 0)
        self.assertIsNone(self.ZLO.final_average)

    def test_delta_r_precalculated(self):
        self.ZLO.add([1, 2, 3, 4, 5, 6], 0)
        self.ZLO.add([-2, -3, -4, -5, -6, -7], 180)
        self.ZLO.add([50, 40, 30, 22, 12, 2], 90)
        self.ZLO.add([-50, -40, -30, -20, -10, 0], 270)
        self.ZLO.average()

        v = np.zeros((3, 6))
        expected = v - np.array([-0.25, -0.25, -0.25, 0.25, 0.25, 0.25])
        res = self.ZLO.delta_r(v)

        np.testing.assert_allclose(res, expected)

    def test_delta_r_calls_ave(self):
        self.ZLO.add([1, 2, 3, 4, 5, 6], 0)
        self.ZLO.add([-2, -3, -4, -5, -6, -7], 180)
        self.ZLO.add([50, 40, 30, 22, 12, 2], 90)
        self.ZLO.add([-50, -40, -30, -20, -10, 0], 270)
        # Don't call average() explicitly, it should be automatically called
        v = np.zeros((3, 6))
        expected = v - np.array([-0.25, -0.25, -0.25, 0.25, 0.25, 0.25])
        res = self.ZLO.delta_r(v)

        np.testing.assert_allclose(res, expected)

    def test_delta_r_with_1d_array(self):
        """delta_r should accept a 1D array and return a 1D result."""
        self.ZLO.final_average = np.zeros((1, 6))
        v = np.zeros(6)

        result = self.ZLO.delta_r(v)

        self.assertEqual(result.shape, (6,))
        np.testing.assert_array_equal(result, np.zeros(6))

    def test_delta_r_with_2d_array(self):
        """delta_r should accept a 2D array and return a 2D result."""
        self.ZLO.final_average = np.zeros((1, 6))
        v = np.zeros(6).reshape(1, -1)

        result = self.ZLO.delta_r(v)

        self.assertEqual(result.shape, (1, 6))
        np.testing.assert_array_equal(result, np.zeros((1, 6)))

    def test_initial_data_added(self):
        self.ZLO = ZeroLoadOutput([1, 0, 1], 0, 3)
        np.testing.assert_allclose(self.ZLO.Z[0], np.array([[1, 0, 1]]))
        self.assertEqual(self.ZLO.num_bridges, 3)

    def test_initial_data_incomplete(self):
        with self.assertRaises(ValueError) as cm:
            self.ZLO = ZeroLoadOutput([1, 0, 1])

        self.assertIn("Initial 'orientation' must be provided", str(cm.exception))


if __name__ == "__main__":
    unittest.main()
