import unittest
from unittest.mock import Mock

import numpy as np

from windcal.core.artifact import BalanceCalibration
from windcal.core.io import STANDARD_CHANNELS, ZeroLoadOutput
from windcal.reduction.reducer import DataReducer


class TestDataReducer(unittest.TestCase):
    def setUp(self):
        self.mock_artifact = Mock(spec=BalanceCalibration)
        self.mock_artifact.component_names = ["N1", "N2", "Y1", "Y2", "AF", "RM"]
        self.mock_artifact.math_model_type = "LinearModel"
        # Lower-triangular ones: not symmetric, so a transposed multiply can't pass by luck.
        # Each sample becomes its running total: [1, 2, 3, ...] -> [1, 3, 6, ...]
        self.mock_artifact.transformation_matrix = np.tril(np.ones((6, 6), dtype=int))
        self.mock_artifact.coefficient_matrix = np.eye(6)
        z = ZeroLoadOutput()
        self.reducer = DataReducer(self.mock_artifact, zero_loads=z)
        # Pass voltages straight through so these tests cover ordering/labelling, not the math model
        self.reducer.math_model = Mock()
        self.reducer.math_model.reduce.side_effect = lambda coefficients, voltages: voltages

    def test_order_voltages_no_change(self):
        v_head = ["rN1", "rN2", "rY1", "rY2", "rAF", "rRM"]
        v_data = [[1, 7, 13],
                  [2, 8, 14],
                  [3, 9, 15],
                  [4, 10, 16],
                  [5, 11, 17],
                  [6, 12, 18]]
        voltages = dict(zip(v_head, v_data))

        v_out = self.reducer._order_voltages(voltages)

        expected_voltages = np.array([[1, 2, 3, 4, 5, 6],
                                      [7, 8, 9, 10, 11, 12],
                                      [13, 14, 15, 16, 17, 18]])

        np.testing.assert_allclose(v_out, expected_voltages)

    def test_order_voltages_reorder_voltages(self):
        v_head = ["rN2", "rY2", "rY1", "rAF", "rRM", "rN1"]
        v_data = [[2, 8, 14],
                  [4, 10, 16],
                  [3, 9, 15],
                  [5, 11, 17],
                  [6, 12, 18],
                  [1, 7, 13]]
        voltages = dict(zip(v_head, v_data))

        v_out = self.reducer._order_voltages(voltages)

        expected_voltages = np.array([[1, 2, 3, 4, 5, 6],
                                      [7, 8, 9, 10, 11, 12],
                                      [13, 14, 15, 16, 17, 18]])

        np.testing.assert_allclose(v_out, expected_voltages)

    def test_order_voltages_jagged_values(self):
        v_head = ["rN2", "rY2", "rY1", "rAF", "rRM", "rN1"]
        v_data = [[2, 8, 14],
                  [4, 10, 16],
                  [3, 9, 15],
                  [5, 11, 17],
                  [6, 12, 18],
                  [1, 7, 13, 999]]
        voltages = dict(zip(v_head, v_data))

        with self.assertRaises(ValueError) as cm:
            self.reducer._order_voltages(voltages)

        self.assertIn("Inconsistent component lengths", str(cm.exception))

    def test_order_voltages_missing_key(self):
        v_head = ["rN2", "rY2", "rY1", "rRM", "rN1"]  # <-- Removed "rAF"
        v_data = [[2, 8, 14],
                  [4, 10, 16],
                  [3, 9, 15],
                  [6, 12, 18],
                  [1, 7, 13]]
        voltages = dict(zip(v_head, v_data))

        with self.assertRaises(ValueError) as cm:
            self.reducer._order_voltages(voltages)

        self.assertIn("Missing component", str(cm.exception))

    def test_order_voltages_single_sample(self):
        v_head = ["rN1", "rN2", "rY1", "rY2", "rAF", "rRM"]
        v_data = [[1], [2], [3], [4], [5], [6]]
        voltages = dict(zip(v_head, v_data))

        v_out = self.reducer._order_voltages(voltages)

        expected_voltages = np.array([[1, 2, 3, 4, 5, 6]])

        self.assertEqual(v_out.shape, (1, 6))
        np.testing.assert_allclose(v_out, expected_voltages)

    def test_order_voltages_scalar_values(self):
        v_head = ["rN2", "rY2", "rY1", "rAF", "rRM", "rN1"]
        v_data = [2, 4, 3, 5, 6, 1]
        voltages = dict(zip(v_head, v_data))

        v_out = self.reducer._order_voltages(voltages)

        expected_voltages = np.array([1, 2, 3, 4, 5, 6])

        self.assertEqual(v_out.shape, (6,))
        np.testing.assert_allclose(v_out, expected_voltages)

    def test_label_loads_no_keys(self):
        loads = np.array([[1, 2, 3, 4, 5, 6],
                         [7, 8, 9, 10, 11, 12],
                         [13, 14, 15, 16, 17, 18]])

        l_out = self.reducer._label_loads(loads)

        expected_loads = {
            "N1": [1, 7, 13],
            "N2": [2, 8, 14],
            "Y1": [3, 9, 15],
            "Y2": [4, 10, 16],
            "AF": [5, 11, 17],
            "RM": [6, 12, 18],
        }
        self.assertEqual(l_out, expected_loads)

    def test_label_loads_with_valid_keys(self):
        loads = np.array([[1, 2, 3, 4, 5, 6],
                         [7, 8, 9, 10, 11, 12],
                         [13, 14, 15, 16, 17, 18]])

        l_out = self.reducer._label_loads(loads, keys=STANDARD_CHANNELS)

        expected_loads = {
            "NF": [1, 7, 13],
            "SF": [2, 8, 14],
            "AF": [3, 9, 15],
            "PM": [4, 10, 16],
            "RM": [5, 11, 17],
            "YM": [6, 12, 18],
        }
        self.assertEqual(l_out, expected_loads)

    def test_label_loads_with_missing_keys(self):
        loads = np.array([[1, 2, 3, 4, 5, 6],
                          [7, 8, 9, 10, 11, 12],
                          [13, 14, 15, 16, 17, 18]])

        with self.assertRaises(ValueError) as cm:
            self.reducer._label_loads(loads, keys=["NF", "SF"])

        self.assertIn("Provided Keys are not the same size", str(cm.exception))

    def test_label_loads_single_sample(self):
        loads = np.array([-6.40646588, -51.74580914, -2.64635733, -1.70494488, -1.8366333, 6.60970448])

        l_out = self.reducer._label_loads(loads)

        expected_loads = {
            "N1": [-6.40646588],
            "N2": [-51.74580914],
            "Y1": [-2.64635733],
            "Y2": [-1.70494488],
            "AF": [-1.8366333],
            "RM": [6.60970448],
        }
        self.assertEqual(l_out, expected_loads)

    def test_label_loads_returns_python_types(self):
        loads = np.array([1.5, 2.5, 3.5, 4.5, 5.5, 6.5])

        l_out = self.reducer._label_loads(loads)

        self.assertIs(type(l_out["N1"]), list)
        self.assertIs(type(l_out["N1"][0]), float)  # not np.float64

    def test_label_loads_with_valid_keys_single_sample(self):
        loads = np.array([1, 2, 3, 4, 5, 6])

        l_out = self.reducer._label_loads(loads, keys=STANDARD_CHANNELS)

        expected_loads = {
            "NF": [1],
            "SF": [2],
            "AF": [3],
            "PM": [4],
            "RM": [5],
            "YM": [6],
        }
        self.assertEqual(l_out, expected_loads)

    def test_label_loads_with_missing_keys_single_sample(self):
        loads = np.array([1, 2, 3, 4, 5, 6])

        with self.assertRaises(ValueError) as cm:
            self.reducer._label_loads(loads, keys=["NF", "SF"])

        self.assertIn("Provided Keys are not the same size", str(cm.exception))

    def test_label_loads_component_count_mismatch(self):
        loads = np.array([1, 2, 3, 4, 5])  # 5 values, but the calibration has 6 components

        with self.assertRaises(ValueError) as cm:
            self.reducer._label_loads(loads)

        self.assertIn("Provided Keys are not the same size", str(cm.exception))

    def test_to_eng_units_single_sample(self):
        v_head = ["rN1", "rN2", "rY1", "rY2", "rAF", "rRM"]
        v_data = [[1], [2], [3], [4], [5], [6]]
        voltages = dict(zip(v_head, v_data))

        l_out = self.reducer.to_eng_units(voltages)

        expected_loads = {
            "N1": [1],
            "N2": [2],
            "Y1": [3],
            "Y2": [4],
            "AF": [5],
            "RM": [6],
        }
        self.assertEqual(l_out, expected_loads)

    def test_to_eng_units_batch(self):
        v_head = ["rN1", "rN2", "rY1", "rY2", "rAF", "rRM"]
        v_data = [[1, 7, 13],
                  [2, 8, 14],
                  [3, 9, 15],
                  [4, 10, 16],
                  [5, 11, 17],
                  [6, 12, 18]]
        voltages = dict(zip(v_head, v_data))

        l_out = self.reducer.to_eng_units(voltages)

        expected_loads = {
            "N1": [1, 7, 13],
            "N2": [2, 8, 14],
            "Y1": [3, 9, 15],
            "Y2": [4, 10, 16],
            "AF": [5, 11, 17],
            "RM": [6, 12, 18],
        }
        self.assertEqual(l_out, expected_loads)

    def test_to_force_moment_single_sample(self):
        l_head = ["N1", "N2", "Y1", "Y2", "AF", "RM"]
        l_data = [[1], [2], [3], [4], [5], [6]]
        loads = dict(zip(l_head, l_data))

        fm_out = self.reducer.to_force_moment(loads)

        expected_fm = {
            "NF": [1],
            "SF": [3],
            "AF": [6],
            "PM": [10],
            "RM": [15],
            "YM": [21],
        }
        self.assertEqual(fm_out, expected_fm)

    def test_to_force_moment_batch(self):
        l_head = ["N1", "N2", "Y1", "Y2", "AF", "RM"]
        l_data = [[1, 7, 13],
                  [2, 8, 14],
                  [3, 9, 15],
                  [4, 10, 16],
                  [5, 11, 17],
                  [6, 12, 18]]
        loads = dict(zip(l_head, l_data))

        fm_out = self.reducer.to_force_moment(loads)

        expected_fm = {
            "NF": [1, 7, 13],
            "SF": [3, 15, 27],
            "AF": [6, 24, 42],
            "PM": [10, 34, 58],
            "RM": [15, 45, 75],
            "YM": [21, 57, 93],
        }
        self.assertEqual(fm_out, expected_fm)

    def test_to_force_moment_scalar_values(self):
        l_head = ["N1", "N2", "Y1", "Y2", "AF", "RM"]
        l_data = [1, 2, 3, 4, 5, 6]  # plain numbers instead of single-item lists
        loads = dict(zip(l_head, l_data))

        fm_out = self.reducer.to_force_moment(loads)

        expected_fm = {
            "NF": [1],
            "SF": [3],
            "AF": [6],
            "PM": [10],
            "RM": [15],
            "YM": [21],
        }
        self.assertEqual(fm_out, expected_fm)

    def test_to_force_moment_no_transformation_matrix(self):
        self.mock_artifact.transformation_matrix = None
        l_head = ["N1", "N2", "Y1", "Y2", "AF", "RM"]
        l_data = [[1, 7, 13],
                  [2, 8, 14],
                  [3, 9, 15],
                  [4, 10, 16],
                  [5, 11, 17],
                  [6, 12, 18]]
        loads = dict(zip(l_head, l_data))

        fm_out = self.reducer.to_force_moment(loads)

        self.assertIs(fm_out, loads)

    def test_voltages_to_force_moment_single_sample(self):
        v_head = ["rN1", "rN2", "rY1", "rY2", "rAF", "rRM"]
        v_data = [[1], [2], [3], [4], [5], [6]]
        voltages = dict(zip(v_head, v_data))

        fm_out = self.reducer.to_force_moment(self.reducer.to_eng_units(voltages))

        expected_fm = {
            "NF": [1],
            "SF": [3],
            "AF": [6],
            "PM": [10],
            "RM": [15],
            "YM": [21],
        }
        self.assertEqual(fm_out, expected_fm)

    def test_voltages_to_force_moment_batch(self):
        v_head = ["rN1", "rN2", "rY1", "rY2", "rAF", "rRM"]
        v_data = [[1, 7, 13],
                  [2, 8, 14],
                  [3, 9, 15],
                  [4, 10, 16],
                  [5, 11, 17],
                  [6, 12, 18]]
        voltages = dict(zip(v_head, v_data))

        fm_out = self.reducer.to_force_moment(self.reducer.to_eng_units(voltages))

        expected_fm = {
            "NF": [1, 7, 13],
            "SF": [3, 15, 27],
            "AF": [6, 24, 42],
            "PM": [10, 34, 58],
            "RM": [15, 45, 75],
            "YM": [21, 57, 93],
        }
        self.assertEqual(fm_out, expected_fm)


if __name__ == "__main__":
    unittest.main()