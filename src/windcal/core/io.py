import numpy as np
import pandas as pd

# The global standard channel order for WindCal
STANDARD_CHANNELS = ['NF', 'SF', 'AF', 'PM', 'RM', 'YM']
VOLTAGE_CHANNELS = ['V1', 'V2', 'V3', 'V4', 'V5', 'V6']


class CalibrationDataSet:
    """Holds known applied loads and resultant voltage vectors for calibration."""

    def __init__(self, loads: np.ndarray, voltages: np.ndarray):
        self.loads = loads
        self.voltages = voltages
        self.validate()

    def validate(self):
        if self.loads.shape != self.voltages.shape:
            raise ValueError("Loads and voltages must have the same shape (N x 6).")
        if self.loads.shape[1] != 6:
            raise ValueError("Data must have exactly 6 channels/components.")

    @classmethod
    def from_csv(cls, filepath: str):
        """Loads calibration data and automatically re-orders columns to match WindCal standard."""
        df = pd.read_csv(filepath)

        # Explicitly indexing by standard lists guarantees the NumPy arrays
        # are perfectly aligned, regardless of the CSV's column order.
        try:
            loads = df[STANDARD_CHANNELS].to_numpy()
            voltages = df[VOLTAGE_CHANNELS].to_numpy()
        except KeyError as e:
            raise KeyError(f"Missing required wind tunnel channel in CSV: {e}")

        return cls(loads, voltages)
