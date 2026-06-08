import unittest
from unittest.mock import Mock, patch

from windcal.core.artifact import BalanceCalibration
from windcal.core.io import STANDARD_CHANNELS
from windcal.calibration.calibrator import Calibrator
from windcal.calibration.models import *


class TestCalibrator(unittest.TestCase):
    def test_model_initialization(self):

        calibrator = Calibrator(math_model=LinearModel())
        self.assertTrue(isinstance(calibrator.math_model, CalibrationMathModel))
        # add each model type here

    def test_invalid_model(self):
        with self.assertRaises(TypeError):
            Calibrator(math_model=CalibrationMathModel())

        with self.assertRaises(TypeError):
            Calibrator(math_model="not a model")

    def test_generate_calls_fit(self):
        """Validate Calibrator calls the fit() method from dependency when generate_calibration() is called"""
        # Create fake model
        mock_model = Mock(spec=CalibrationMathModel)

        # Define what fit() should return
        mock_model.fit.return_value = ("C", "a")

        # Inject into system under test
        calibrator = Calibrator(math_model=mock_model)

        # Fake data object
        data = Mock()
        data.channels = []

        # Run method
        calibrator.generate_calibration(data)

        # Verify behavior
        mock_model.fit.assert_called_once_with(data)

    def test_artifact_fields(self):
        mock_model = Mock(spec=CalibrationMathModel)
        mock_model.fit.return_value = ("C", "a")

        calibrator = Calibrator(math_model=mock_model)

        data = Mock()
        data.channels = []

        artifact = calibrator.generate_calibration(data)

        self.assertEqual(artifact.coefficient_matrix, "C")
        self.assertEqual(artifact.bias_vector, "a")
        self.assertEqual(artifact.math_model_type, mock_model.__class__.__name__)

    def test_no_transform_for_standard_channels(self):
        mock_model = Mock(spec=CalibrationMathModel)
        mock_model.fit.return_value = ("C", "a")

        calibrator = Calibrator(mock_model)

        data = Mock()
        data.channels = STANDARD_CHANNELS

        artifact = calibrator.generate_calibration(data)

        self.assertIsNone(artifact.transformation_matrix)

    def test_force_balance_transform(self):
        mock_model = Mock(spec=CalibrationMathModel)
        mock_model.fit.return_value = ("C", "a")

        calibrator = Calibrator(mock_model)

        data = Mock()
        data.channels = ["N1"]  # triggers branch

        metadata = Mock()
        metadata.distances = "dist1"

        with patch.object(
                BalanceCalibration,
                "create_5f1m_transformation_matrix",
                return_value="XFORM"
        ) as mock_xform:
            artifact = calibrator.generate_calibration(data, metadata)

            mock_xform.assert_called_once()
            mock_xform.assert_called_once_with("dist1")
            self.assertEqual(artifact.transformation_matrix, "XFORM")

    def test_moment_balance_transform(self):
        mock_model = Mock(spec=CalibrationMathModel)
        mock_model.fit.return_value = ("C", "a")

        calibrator = Calibrator(mock_model)

        data = Mock()
        data.channels = ["PF"]  # triggers branch

        metadata = Mock()
        metadata.distances = "dist2"

        with patch.object(
                BalanceCalibration,
                "create_1f5m_transformation_matrix",
                return_value="XFORM2"
        ) as mock_xform:
            artifact = calibrator.generate_calibration(data, metadata)

            mock_xform.assert_called_once()
            mock_xform.assert_called_once_with("dist2")
            self.assertEqual(artifact.transformation_matrix, "XFORM2")


if __name__ == '__main__':
    unittest.main()
