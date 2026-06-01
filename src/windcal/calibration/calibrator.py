from windcal.core.io import CalibrationDataSet
from windcal.core.artifact import BalanceCalibration
from windcal.calibration.models import CalibrationMathModel


class Calibrator:
    """Orchestrates the calibration process to produce a validated artifact."""

    def __init__(self, math_model: CalibrationMathModel):
        self.math_model = math_model

    def generate_calibration(self, data: CalibrationDataSet, author: str = "System") -> BalanceCalibration:
        """Fits the data using the injected math model and returns the artifact."""
        coeff_matrix = self.math_model.fit(data)

        # Create and return the auditable artifact
        artifact = BalanceCalibration(
            coefficient_matrix=coeff_matrix,
            math_model_type=self.math_model.__class__.__name__,
            author=author
        )
        return artifact
