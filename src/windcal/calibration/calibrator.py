from windcal.core.io import CalibrationDataSet, STANDARD_CHANNELS
from windcal.core.artifact import BalanceCalibration
from windcal.core.metadata import CalibrationMetadata
from windcal.calibration.models import CalibrationMathModel


class Calibrator:
    """Orchestrates the calibration process to produce a validated artifact."""

    def __init__(self, math_model: CalibrationMathModel):
        self.math_model = math_model
        self.validate_model()

    def validate_model(self):
        if not issubclass(self.math_model.__class__, CalibrationMathModel):
            raise TypeError("math_model must ve a CalibrationMathModel")

    def generate_calibration(
            self, data: CalibrationDataSet,
            metadata: CalibrationMetadata = None,
    ) -> BalanceCalibration:
        """Uses the selected math model to fit the data; Returns the calibration artifact."""
        # Unpack the tuple returned by the updated fit() method
        C, a = self.math_model.fit(data)
        xform = None

        # Check if the channel list matches the STANDARD_CHANNELS
        if not data.channels == STANDARD_CHANNELS:
            if "N1" in data.channels:  # Force Balance
                xform = BalanceCalibration.create_5f1m_transformation_matrix(metadata.distances)
            elif "PF" in data.channels:  # Moment Balance
                xform = BalanceCalibration.create_1f5m_transformation_matrix(metadata.distances)

        # Create and return the auditable artifact
        artifact = BalanceCalibration(
            coefficient_matrix=C,
            math_model_type=self.math_model.__class__.__name__,
            component_names=data.channels,
            bias_vector=a,
            transformation_matrix=xform,
            metadata=metadata
        )
        return artifact
