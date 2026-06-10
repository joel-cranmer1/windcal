import os
import unittest
import numpy as np

from windcal.core.io import CalibrationDataSet, BALANCE_CHANNELS


class TestCalibrationDataSet(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.base_dir = os.path.dirname(os.path.abspath(__file__))
        # Define the exact path to static fixture directory
        cls.fixture_dir = os.path.join(cls.base_dir, 'fixtures')

    def test_load_csv(self):
        data_set = CalibrationDataSet.from_csv(os.path.join(self.fixture_dir, 'Test_cal_data_SymLin.csv'))
        self.assertTrue(data_set.loads.size > 0)
        self.assertTrue(data_set.voltages.size > 0)

    # Never raises error due to Pandas filling in the blank csv data
    # def test_load_csv_omit_voltage(self):
    #     with self.assertRaises(ValueError):
    #         data_set = CalibrationDataSet.from_csv(os.path.join(self.fixture_dir, 'Test_cal_data_omit_volt.csv'))
    #         print(f"Loads: {data_set.loads.size}")
    #         print(f"Volts: {data_set.voltages.size}")

    def test_load_csv_extra_ch(self):
        with self.assertRaises(ValueError):
            data_set = CalibrationDataSet.from_csv(os.path.join(self.fixture_dir, 'Test_cal_data_extra_ch.csv'))
            print(f"Loads: {data_set.loads.size}")
            print(f"Volts: {data_set.voltages.size}")


class TestCalibrationDataSetOrdering(unittest.TestCase):

    def setUp(self):
        # Canonical subset we’ll test against
        self.channels = ['PF', 'PA', 'YF', 'YA', 'AF', 'RM']

        # Create deterministic test data
        # Each column will be easily identifiable by its value
        self.loads = np.array([
            [10, 20, 30, 40, 50, 60]
        ])

        self.voltages = np.array([
            [1, 2, 3, 4, 5, 6]
        ])

    def test_order_preserved_when_already_correct(self):
        ds = CalibrationDataSet(
            self.loads.copy(),
            self.voltages.copy(),
            self.channels.copy()
        )

        self.assertEqual(ds.channels, ['PF', 'PA', 'YF', 'YA', 'AF', 'RM'])
        np.testing.assert_array_equal(ds.loads, self.loads)
        np.testing.assert_array_equal(ds.voltages, self.voltages)

    def test_order_corrected_when_shuffled(self):
        reorder = [2, 0, 1, 3, 5, 4]  # Shuffled IDs
        # shuffle columns accordingly
        shuffled_channels = [self.channels[i] for i in reorder]
        loads = self.loads[:, reorder]
        voltages = self.voltages[:, reorder]

        ds = CalibrationDataSet(loads, voltages, shuffled_channels)

        # Expected canonical order based on BALANCE_CHANNELS
        expected_channels = ['PF', 'PA', 'YF', 'YA', 'AF', 'RM']

        self.assertEqual(ds.channels, expected_channels)

        # Verify loads reordered correctly
        expected_loads = np.array([[10, 20, 30, 40, 50, 60]])
        expected_voltages = np.array([[1, 2, 3, 4, 5, 6]])

        np.testing.assert_array_equal(ds.loads, expected_loads)
        np.testing.assert_array_equal(ds.voltages, expected_voltages)

    def test_multiple_random_permutations(self):
        import itertools

        canonical = ['PF', 'PA', 'YF', 'YA', 'AF', 'RM']

        for perm in itertools.permutations(range(6)):
            permuted_channels = [canonical[i] for i in perm]

            loads = self.loads[:, list(perm)]
            voltages = self.voltages[:, list(perm)]

            ds = CalibrationDataSet(loads, voltages, permuted_channels)

            # Always expect canonical order
            self.assertEqual(ds.channels, canonical)

            np.testing.assert_array_equal(ds.loads, self.loads)
            np.testing.assert_array_equal(ds.voltages, self.voltages)

    def test_ignores_extra_channels(self):
        channels = ['XX', 'PF', 'PA', 'YY', 'YF', 'YA', 'ZZ', 'AF', 'RM']

        loads = np.array([[999, 10, 20, 888, 30, 40, 777, 50, 60]])
        voltages = np.array([[999, 1, 2, 888, 3, 4, 777, 5, 6]])

        ds = CalibrationDataSet(loads, voltages, channels)

        self.assertEqual(ds.channels, ['PF', 'PA', 'YF', 'YA', 'AF', 'RM'])

        np.testing.assert_array_equal(ds.loads, [[10, 20, 30, 40, 50, 60]])
        np.testing.assert_array_equal(ds.voltages, [[1, 2, 3, 4, 5, 6]])


if __name__ == '__main__':
    unittest.main()
