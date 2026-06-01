from windcal.core.io import CalibrationDataSet, STANDARD_CHANNELS
from windcal.core.artifact import BalanceCalibration
from windcal.calibration.models import CalibrationMathModel


class Calibrator:
    """Orchestrates the calibration process to produce a validated artifact."""

    def __init__(self, math_model: CalibrationMathModel):
        self.math_model = math_model

    def generate_calibration(self, data: CalibrationDataSet, author: str = "System") -> BalanceCalibration:
        """Fits the data using the injected math model and returns the artifact."""
        # Unpack the tuple returned by the updated fit() method
        coeff_matrix, bias_vector = self.math_model.fit(data)

        # Create and return the auditable artifact
        artifact = BalanceCalibration(
            coefficient_matrix=coeff_matrix,
            bias_vector=bias_vector,
            math_model_type=self.math_model.__class__.__name__,
            component_names=STANDARD_CHANNELS,
            author=author
        )
        return artifact
