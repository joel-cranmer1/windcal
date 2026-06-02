import numpy as np
import pandas as pd

# The global standard channel order for WindCal
STANDARD_CHANNELS = ['NF', 'SF', 'AF', 'PM', 'RM', 'YM']
BALANCE_CHANNELS = ['NF', 'N1', 'N2', 'SF', 'Y1', 'Y2', 'AX', 'AF', 'PM', 'PF', 'PA', 'YF', 'YA', 'RM', 'YM']
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

        # Find intersection of expected vs actual columns
        load_cols = [c for c in BALANCE_CHANNELS if c in df.columns]
        volt_cols = [c for c in VOLTAGE_CHANNELS if c in df.columns]

        if not load_cols:
            raise ValueError("No valid load channels found in CSV")

        if not volt_cols:
            raise ValueError("No valid voltage channels found in CSV")

        loads = df[load_cols].to_numpy()
        voltages = df[volt_cols].to_numpy()

        return cls(loads, voltages)
