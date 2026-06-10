import unittest
from unittest.mock import Mock, patch
import numpy as np

from windcal.core.artifact import BalanceCalibration
from windcal.reduction.reducer import DataReducer


class TestDataReducer(unittest.TestCase):
    def setUp(self):
        self.mock_artifact = Mock(spec=BalanceCalibration)
        self.mock_artifact.component_names = ["N1", "N2", "Y1", "Y2", "AF", "RM"]
        self.mock_artifact.math_model_type = "LinearModel"
        self.reducer = DataReducer(self.mock_artifact)

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

    def test_label_loads(self):
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
            "RM": [6, 12, 18]
        }

        self.assertEqual(l_out, expected_loads)


if __name__ == '__main__':
    unittest.main()
