import numpy as np
import pandas as pd
from windcal.core.artifact import BalanceCalibration
from windcal.core.io import STANDARD_CHANNELS
from windcal.calibration.models import \
    Linear6x6Model  # (Would dynamically load based on artifact string in full version)


class DataReducer:
    """Operational engine for reducing voltages to physical loads."""

    def __init__(self, calibration: BalanceCalibration):
        self.calibration = calibration
        # Math model instantiation here...

    def process_point(self, voltages: np.ndarray) -> np.ndarray:
        """Fast processing for a single reading (DAQ Loop)."""
        # Step 1: Get raw balance loads (could be 5F/1M or 3F/3M depending on calibration)
        raw_loads = self.math_model.reduce(self.calibration.coefficient_matrix, voltages)

        # Step 2: Auto-resolve to 3F/3M if it's a 5F/1M balance
        if self.calibration.transformation_matrix is not None:
            return self.calibration.transformation_matrix @ raw_loads

        return raw_loads

    def process_batch(self, voltages_df: pd.DataFrame) -> pd.DataFrame:
        """Vectorized processing for an entire legacy file/dataset."""
        voltages = voltages_df.to_numpy()
        raw_loads = self.math_model.reduce(self.calibration.coefficient_matrix, voltages)

        if self.calibration.transformation_matrix is not None:
            # Apply geometric 5F/1M -> 3F/3M matrix transformation to all rows
            resolved_loads = (self.calibration.transformation_matrix @ raw_loads.T).T
            columns = STANDARD_CHANNELS  # Standardized 3F/3M names
        else:
            resolved_loads = raw_loads
            columns = self.calibration.component_names  # Keep original names if already 3F/3M

        return pd.DataFrame(resolved_loads, columns=columns, index=voltages_df.index)

