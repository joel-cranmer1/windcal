import numpy as np
import pandas as pd

from windcal.calibration.models import CalibrationMathModel
from windcal.core.artifact import BalanceCalibration
from windcal.core.io import STANDARD_CHANNELS


class DataReducer:
    """Operational engine for reducing voltages to physical loads.

    Conditions the data into the proper format for the calibration since the column
    order maters very much. The data will be returned in a dictionary of items to
    denote the values' axis
    """

    def __init__(self, calibration: BalanceCalibration):
        self.calibration = calibration
        self.math_model = CalibrationMathModel.create(calibration.math_model_type)

    def process_point(self, voltages: dict) -> dict:
        """Fast processing for a single reading (DAQ Loop)."""
        # Step 1: Get raw balance loads (could be 5F/1M or 3F/3M depending on calibration)
        raw_loads = self.math_model.reduce(
            self.calibration.coefficient_matrix,
            self.calibration.bias_vector,
            self._order_voltages(voltages)
        )

        # Step 2: Auto-resolve to 3F/3M if it's a 5F/1M balance
        if self.calibration.transformation_matrix is not None:
            return self.calibration.transformation_matrix @ raw_loads

        return raw_loads

    def process_batch(self, voltages_df: pd.DataFrame) -> pd.DataFrame:
        """Vectorized processing for an entire legacy file/dataset."""
        voltages = voltages_df.to_numpy()
        raw_loads = self.math_model.reduce(
            self.calibration.coefficient_matrix,
            self.calibration.bias_vector,
            self._order_voltages(voltages)
        )

        if self.calibration.transformation_matrix is not None:
            # Apply geometric 5F/1M -> 3F/3M matrix transformation to all rows
            resolved_loads = (self.calibration.transformation_matrix @ raw_loads.T).T
            columns = STANDARD_CHANNELS  # Standardized 3F/3M names
        else:
            resolved_loads = raw_loads
            columns = self.calibration.component_names  # Keep original names if already 3F/3M

        return pd.DataFrame(resolved_loads, columns=columns, index=voltages_df.index)

    def _order_voltages(self, voltages: dict) -> np.ndarray:
        try:
            return np.stack([
                voltages[f"r{c}"]
                for c in self.calibration.component_names
            ], axis=-1)
        except KeyError as e:
            raise ValueError(f"Missing voltage for component: {e.args[0]}")
        except ValueError as e:
            raise ValueError(f"Inconsistent voltage lengths: {e}")

    def _label_loads(self, loads: np.ndarray) -> dict:
        header = self.calibration.component_names
        result = {h: col.tolist() for h, col in zip(header, loads.T)}
        return result
