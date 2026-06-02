import os
import unittest
import numpy as np

from windcal.core.artifact import BalanceCalibration


class TestBalanceCalibration(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.base_dir = os.path.dirname(os.path.abspath(__file__))
        # Define the exact path to static fixture directory
        cls.fixture_dir = os.path.join(cls.base_dir, 'fixtures')

    def test_load_valid_file(self):
        self.assertEqual(True, False)  # add assertion here


class Test1F5MTransformationMatrix(unittest.TestCase):

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
        xform = BalanceCalibration.create_5f1m_transformation_matrix(
            x_nf_fore=0.1,
            x_nf_aft=0.2,
            x_sf_fore=0.15,
            x_sf_aft=0.25
        )

        expected_xform = np.array([[1.0, 1.0, 0.0, 0.0, 0.0, 0.0],
                                   [0.0, 0.0, 1.0, 1.0, 0.0, 0.0],
                                   [0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
                                   [0.1, -0.2, 0.0, 0.0, 0.0, 0.0],
                                   [0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
                                   [0.0, 0.0, -0.15, 0.25, 0.0, 0.0]])

        np.testing.assert_allclose(xform, expected_xform, rtol=1e-10, atol=0, strict=True)


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
        x_pf, x_pa = -0.5, 1.5
        x_yf, x_ya = -1.0, 2.0

        xform = BalanceCalibration.create_1f5m_transformation_matrix(x_pf, x_pa, x_yf, x_ya)

        expected_xform = np.array([[1.0, -1.0, 0.0, 0.0, 0.0, 0.0],
                                   [0.0, 0.0, 1.0, -1.0, 0.0, 0.0],
                                   [0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
                                   [0.5, 0.5, 0.0, 0.0, 0.0, 0.0],
                                   [0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
                                   [0.0, 0.0, 0.0, 1.0, 0.0, 0.0]])

        np.testing.assert_allclose(xform, expected_xform, rtol=1e-10, atol=0, strict=True)

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
