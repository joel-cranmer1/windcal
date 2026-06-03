from windcal.core.io import CalibrationDataSet, STANDARD_CHANNELS
from windcal.core.artifact import BalanceCalibration
from windcal.calibration.models import CalibrationMathModel


class Calibrator:
    """Orchestrates the calibration process to produce a validated artifact."""

    def __init__(self, math_model: CalibrationMathModel):
        self.math_model = math_model

    def generate_calibration(
            self, data: CalibrationDataSet,
            author: str = "System",
            channels: list = STANDARD_CHANNELS,
            *,
            balance_info: dict = None,
            max_loads: dict = None,
            distances: tuple = None,
    ) -> BalanceCalibration:
        """Fits the data using the injected math model and returns the artifact."""
        # Unpack the tuple returned by the updated fit() method
        C, a = self.math_model.fit(data)
        xform = None

        # Check if the channel list matches the STANDARD_CHANNELS
        if not channels == STANDARD_CHANNELS:
            if "N1" in channels:
                xform = BalanceCalibration.create_5f1m_transformation_matrix(distances)
            elif "PF" in channels:
                xform = BalanceCalibration.create_1f5m_transformation_matrix(distances)

        # Create and return the auditable artifact
        artifact = BalanceCalibration(
            coefficient_matrix=C,
            math_model_type=self.math_model.__class__.__name__,
            component_names=channels,
            bias_vector=a,
            transformation_matrix=xform,
            author=author,
            balance_info=balance_info,
            max_loads=max_loads,
            distances=distances
        )
        return artifact
