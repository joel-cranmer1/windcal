import os
import unittest
import numpy as np
import tempfile

from windcal.core.artifact import BalanceCalibration
from windcal.core.metadata import CalibrationMetadata


class TestBalanceCalibration(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.base_dir = os.path.dirname(os.path.abspath(__file__))
        # Define the exact path to static fixture directory
        cls.fixture_dir = os.path.join(cls.base_dir, 'fixtures')
        cls.metadata = CalibrationMetadata(
            author="System",
            balance_info={"test_balance": True}
        )

    def test_load_valid_file(self):
        file_path = os.path.join(self.fixture_dir, "Test_balance_cal.json")
        Cal = BalanceCalibration.load(file_path)
        self.assertEqual(Cal.__class__.__name__, "BalanceCalibration")
        self.assertEqual(Cal.coefficient_matrix[1, 3], 0.024591532002882124)

    def test_load_invalid_file(self):
        file_path = os.path.join(self.fixture_dir, "Test_bad_cal.json")
        Cal = BalanceCalibration.load(file_path)
        self.assertEqual(Cal.__class__.__name__, "NoneType")

    def test_save_cal_file(self):
        # dummy 6x6 matrix
        C = np.array([
            [0.21341, 0.65599, 0.16580, 0.69299, 0.08340, 0.27650],
            [0.56696, 0.26050, 0.39282, 0.48158, 0.28395, 0.54540],
            [0.51717, 0.70352, 0.75610, 0.00266, 0.15926, 0.31440],
            [0.88432, 0.71919, 0.77881, 0.40355, 0.84471, 0.63410],
            [0.81589, 0.62132, 0.77874, 0.27097, 0.45457, 0.47010],
            [0.58883, 0.41102, 0.39994, 0.16626, 0.14407, 0.95258]
        ])
        bias = np.array([0, 0, 0, 0, 0, 0])
        x = BalanceCalibration.create_5f1m_transformation_matrix(1.5, 1.5, 1.25, 1.25)
        cal = BalanceCalibration(C, "LinearModel", ["N1", "N2", "Y1", "Y2", "AF", "RM"], bias, x, self.metadata)

        with tempfile.TemporaryDirectory() as temp_dir:
            demo_file = os.path.join(temp_dir, "tmp_file.json")
            cal.save(demo_file)

            with open(demo_file) as f:
                lines = f.readlines()

            self.assertTrue(len(lines) >= 126)


class Test5F1MTransformationMatrix(unittest.TestCase):

    def test_identity_passthrough(self):
        """AF and RM should pass through unchanged."""
        T = BalanceCalibration.create_5f1m_transformation_matrix(1.0, 1.0, 1.0, 1.0)

        # [N1, N2, Y1, Y2, AF, RM]
        raw = np.array([0, 0, 0, 0, 123.4, -56.7])
        out = T @ raw

        self.assertAlmostEqual(out[2], 123.4)  # AF
        self.assertAlmostEqual(out[4], -56.7)  # RM

    def test_zero_loads(self):
        """Zero input should produce zero output."""
        T = BalanceCalibration.create_5f1m_transformation_matrix(1.0, 1.0, 1.0, 1.0)

        raw = np.zeros(6)
        out = T @ raw

        np.testing.assert_array_almost_equal(out, np.zeros(6))

    def test_general_case(self):
        x_n1, x_n2 = 1.5, 1.5
        x_y1, x_y2 = 1.25, 1.25

        # Load components
        G = np.array([81.4, -70.0, -9.7, 9.0, 5.3, 11.9]).T

        xform = BalanceCalibration.create_5f1m_transformation_matrix(x_n1, x_n2, x_y1, x_y2)
        out = xform @ G

        expected = np.array([11.4, -0.7, 5.3, 227.1, 11.9, -23.375]).T

        np.testing.assert_allclose(out, expected, rtol=1e-10, atol=0, strict=True)


class Test1F5MTransformationMatrix(unittest.TestCase):

    def test_identity_passthrough(self):
        """AF and RM should pass through unchanged."""
        T = BalanceCalibration.create_1f5m_transformation_matrix(1.0, 1.0, 1.0, 1.0)

        # [PM_fore, PM_aft, YM_fore, YM_aft, AF, RM]
        raw = np.array([0, 0, 0, 0, 123.4, -56.7])
        out = T @ raw

        self.assertAlmostEqual(out[2], 123.4)  # AF
        self.assertAlmostEqual(out[4], -56.7)  # RM

    def test_zero_loads(self):
        """Zero input should produce zero output."""
        T = BalanceCalibration.create_1f5m_transformation_matrix(1.0, 1.0, 1.0, 1.0)

        raw = np.zeros(6)
        out = T @ raw

        np.testing.assert_array_almost_equal(out, np.zeros(6))

    def test_general_case(self):
        """Validate full transformation against manually computed solution."""
        x_pf, x_pa = 1.278, 1.278
        x_yf, x_ya = 1.274, 1.274

        # Load components
        G = np.array([-5.71, 5.31, -0.09, -0.10, 0.129, -0.03]).T

        xform = BalanceCalibration.create_1f5m_transformation_matrix(x_pf, x_pa, x_yf, x_ya)
        out = xform @ G

        expected = np.array([4.3114, -3.9246e-3, 0.129, -0.19999, -0.03, -0.095]).T

        np.testing.assert_allclose(out, expected, rtol=1e-4, atol=0, strict=True)

    def test_invalid_pitch_spacing(self):
        """Should raise if pitch gauges are colocated."""
        with self.assertRaises(ValueError):
            BalanceCalibration.create_1f5m_transformation_matrix(1.0, 1.0, -1.0, 1.0)

    def test_invalid_yaw_spacing(self):
        """Should raise if yaw gauges are colocated."""
        with self.assertRaises(ValueError):
            BalanceCalibration.create_1f5m_transformation_matrix(-1.0, 1.0, 2.0, 2.0)


if __name__ == "__main__":
    unittest.main()
