import unittest
from unittest.mock import Mock, patch

from windcal.core.io import STANDARD_CHANNELS
from windcal.calibration.calibrator import Calibrator
from windcal.calibration.models import *


class TestCalibrator(unittest.TestCase):
    def setUp(self):
        """Runs before EACH test to ensure isolated, fresh mocks."""
        # 1. Setup the mock mathematical model
        self.mock_C = np.eye(6)
        self.mock_model = Mock(spec=CalibrationMathModel)
        self.mock_model.fit.return_value = self.mock_C
        self.mock_model.__class__.__name__ = "MockModel"

        # 2. Instantiate the system under test
        self.calibrator = Calibrator(math_model=self.mock_model)

        # 3. Setup common fake data
        self.mock_data = Mock()
        self.mock_data.channels = STANDARD_CHANNELS

        self.mock_metadata = Mock()
        self.mock_metadata.distances = "mock_distances"

        # 4. Patch BalanceCalibration manually
        # Create the patcher
        patcher = patch('windcal.calibration.calibrator.BalanceCalibration', autospec=True)
        # Start the patcher and save the mock to a class instance variable
        self.MockBalanceCalibration = patcher.start()
        # Ensure the patcher stops after the test finishes, even if the test fails
        self.addCleanup(patcher.stop)

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
        """Validate Calibrator calls the fit() method from dependency"""
        self.calibrator.generate_calibration(self.mock_data)
        self.mock_model.fit.assert_called_once_with(self.mock_data)

    def test_artifact_fields(self):
        self.calibrator.generate_calibration(self.mock_data)

        # Verify the artifact was instantiated with the data from the math model
        self.MockBalanceCalibration.assert_called_once_with(
            coefficient_matrix=self.mock_C,
            math_model_type='MockModel',
            component_names=STANDARD_CHANNELS,
            transformation_matrix=None,
            metadata=None
        )

    def test_no_transform_for_standard_channels(self):
        self.calibrator.generate_calibration(self.mock_data)

        called_kwargs = self.MockBalanceCalibration.call_args[1]
        self.assertIsNone(called_kwargs.get('transformation_matrix'))

    def test_force_balance_transform(self):
        # Override the default mock data for this specific test
        self.mock_data.channels = ["N1"]

        with patch.object(
                self.MockBalanceCalibration,
                "create_5f1m_transformation_matrix",
                return_value="XFORM"
        ) as mock_xform:
            self.calibrator.generate_calibration(self.mock_data, self.mock_metadata)

            mock_xform.assert_called_once_with("mock_distances")

            # Verify the output of the transform was passed to the artifact
            called_kwargs = self.MockBalanceCalibration.call_args[1]
            self.assertEqual(called_kwargs.get('transformation_matrix'), "XFORM")

    def test_moment_balance_transform(self):
        self.mock_data.channels = ["PF"]

        with patch.object(
                self.MockBalanceCalibration,
                "create_1f5m_transformation_matrix",
                return_value="XFORM2"
        ) as mock_xform:

            self.calibrator.generate_calibration(self.mock_data, self.mock_metadata)

            mock_xform.assert_called_once_with("mock_distances")

            called_kwargs = self.MockBalanceCalibration.call_args[1]
            self.assertEqual(called_kwargs.get('transformation_matrix'), "XFORM2")


if __name__ == '__main__':
    unittest.main()
