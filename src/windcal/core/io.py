import numpy as np
import pandas as pd

# The global standard channel order for WindCal
STANDARD_CHANNELS = ['NF', 'SF', 'AF', 'PM', 'RM', 'YM']
BALANCE_CHANNELS = ['NF', 'N1', 'N2', 'SF', 'Y1', 'Y2', 'PF', 'PA', 'YF', 'YA', 'AX', "AF", 'PM', 'RM', 'YM']


class CalibrationDataSet:
    """Holds known applied loads and resultant voltage vectors for calibration."""

    def __init__(self, loads: np.ndarray, voltages: np.ndarray, channels: list):
        self.loads = np.asarray(loads)
        self.voltages = np.asarray(voltages)
        self.channels = list(channels)

        # enforce canonical ordering HERE
        self._enforce_channel_order()

        self.validate()

    def validate(self):
        if self.loads.shape != self.voltages.shape:
            raise ValueError("Loads and voltages must have the same shape (N x 6).")
        if self.loads.shape[1] != 6:
            raise ValueError("Data must have exactly 6 channels/components.")

    def _enforce_channel_order(self):
        """Reorders loads/voltages to match BALANCE_CHANNELS canonical order."""

        # Build mapping: channel -> column index
        index_map = {c: i for i, c in enumerate(self.channels)}

        new_indices = []
        new_channels = []

        # Single pass, O(n)
        for c in BALANCE_CHANNELS:
            if c in index_map:
                new_indices.append(index_map[c])
                new_channels.append(c)

        if not new_indices:
            raise ValueError("No valid channels found for ordering")

        # Reorder arrays efficiently using numpy slicing
        self.loads = self.loads[:, new_indices]
        self.voltages = self.voltages[:, new_indices]
        self.channels = new_channels

    @classmethod
    def from_csv(cls, filepath: str):
        """Loads calibration data and automatically re-orders columns to match WindCal standard."""
        df = pd.read_csv(filepath)

        # Find intersection of expected vs actual columns
        cols = set(df.columns)

        load_cols = []
        volt_cols = []

        # enforce pairing at load time
        for c in BALANCE_CHANNELS:
            rc = f"r{c}"
            if c in cols and rc in cols:
                load_cols.append(c)
                volt_cols.append(rc)

        if not load_cols:
            raise ValueError("No valid load/voltage channel pairs found")

        loads = df[load_cols].to_numpy()
        voltages = df[volt_cols].to_numpy()

        return cls(loads, voltages, load_cols)
