import numpy as np
import pandas as pd
import logging

logger = logging.getLogger(__name__)

# The global standard channel order for WindCal
STANDARD_CHANNELS = ['NF', 'SF', 'AF', 'PM', 'RM', 'YM']
BALANCE_CHANNELS = ['NF', 'N1', 'N2', 'SF', 'Y1', 'Y2', 'PF', 'PA', 'YF', 'YA', 'AX', "AF", 'PM', 'RM', 'YM']


class CalibrationDataSet:
    """Holds known applied loads and resultant voltage vectors for calibration."""

    def __init__(self, loads: np.ndarray, voltages: np.ndarray, channels: list):
        self.loads = np.asarray(loads)
        self.voltages = np.asarray(voltages)
        self.channels = list(channels)

        logger.debug(f"New CalibrationDataSet created!")

        # enforce canonical ordering HERE
        self._enforce_channel_order()

        self.validate()

    def validate(self):
        logger.debug(f"Validating load and voltage shapes...")
        if self.loads.shape != self.voltages.shape:
            raise ValueError("Loads and voltages must have the same shape (N x 6).")
        if self.loads.shape[1] != 6:
            raise ValueError("Data must have exactly 6 channels/components.")
        logger.debug(f"Validation complete.")

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

        logger.debug(f"Channels being reordered: {self.channels} -> {new_channels}")

        # Reorder arrays efficiently using numpy slicing
        self.loads = self.loads[:, new_indices]
        self.voltages = self.voltages[:, new_indices]
        self.channels = new_channels

    @classmethod
    def from_csv(cls, filepath: str):
        """Loads calibration data and automatically re-orders columns to match WindCal standard."""
        logger.debug(f"Loading CSV file: {filepath}")
        df = pd.read_csv(filepath, skipinitialspace=True)
        df.columns = df.columns.str.strip()  # Clean Whitespace
        df.columns = df.columns.str.upper()

        # Find intersection of expected vs actual columns
        cols = set(df.columns)

        load_cols = []
        volt_cols = []

        logger.debug(f"Loading data...")
        # enforce pairing at load time
        for c in BALANCE_CHANNELS:
            rc = f"R{c}"
            if c in cols and rc in cols:
                load_cols.append(c)
                volt_cols.append(rc)

        if not load_cols:
            raise ValueError("No valid load/voltage channel pairs found")

        loads = df[load_cols].to_numpy()
        voltages = df[volt_cols].to_numpy()

        logger.debug(f"Creating new object...")

        return cls(loads, voltages, load_cols)


class ZeroLoadOutput:
    """
    A class to implement the Zero Load output portion of a calibration process
    for a strain-gage balance.

    This class collects voltage readings from bridge elements at different roll
    orientations and calculates the average zero-load output. The calculation
    method follows the recommended practice of averaging measurements from four
    orientations (0, 90, 180, -90/270 degrees).
    """

    def __init__(self, data=None, orientation=None, num_bridges=6):
        """
        Initializes the ZeroLoadOutput object.

        Args:
            data (list or np.ndarray, optional): An array of voltage readings
                for each bridge element. Defaults to None.
            orientation (int or float, optional): The roll orientation for the
                initial data point. Required if 'data' is provided. Must be one
                of [0, 90, 180, -90, 270]. Defaults to None.
            num_bridges (int, optional): The number of bridge elements.
                Defaults to 6.
        """
        self.num_bridges = num_bridges
        self.orientations = {0, 90, 180, -90, 270}
        self.Z = {angle: [] for angle in self.orientations}
        self.final_average: np.ndarray = None

        if data is not None:
            if orientation is None:
                raise ValueError("Initial 'orientation' must be provided if 'data' is given.")
            self.add(data, orientation)

    def add(self, data, orientation) -> None:
        """
        Adds a new data point (voltage readings) for a given orientation.

        Args:
            data (list or np.ndarray): A 1D array of voltage readings of
                length `num_bridges`.
            orientation (int or float): The roll orientation of the data point.
                Must be one of [0, 90, 180, -90, 270].
        """
        if orientation not in self.orientations:
            raise ValueError(f"Orientation must be one of {self.orientations}, but got {orientation}.")

        data_array = np.array(data)
        if data_array.shape != (self.num_bridges,):
            raise ValueError(f"Data must be a 1D array of length {self.num_bridges}, but got shape {data_array.shape}.")

        # Store -90 as 270 for consistent grouping.
        if orientation == -90:
            orientation = 270

        self.Z[orientation].append(data_array)
        logger.debug(f"Added data point for orientation {orientation}°. "
                     f"Total points for this orientation: {len(self.Z[orientation])}")
        self.final_average = None  # ensure that average is recalculated

    def average(self) -> np.ndarray:
        """
        Calculates the average zero-load output across all required orientations.

        The process is as follows:
        1.  Average the readings for each orientation independently.
        2.  Average the resulting four orientation averages together.

        Returns:
            np.ndarray: A 1xN numpy array representing the average zero-load
            output, where N is the number of bridges.

        Raises:
            ValueError: If there is not at least one data point for each of the
            four required orientation groups (0, 90, 180, 270/-90).
        """
        required_orientations = {0, 90, 180, 270}
        missing_orientations = []

        # Check if we have data for 270 or -90
        has_270_or_neg90 = self.Z[270] or self.Z[-90]

        if not self.Z[0]:
            missing_orientations.append(0)
        if not self.Z[90]:
            missing_orientations.append(90)
        if not self.Z[180]:
            missing_orientations.append(180)
        if not has_270_or_neg90:
            missing_orientations.append("-90/270")

        if missing_orientations:
            raise ValueError(
                f"Cannot calculate average. Missing data for the following orientations: {missing_orientations}")

        orientation_averages = []

        # Calculate average for 0 degrees
        if self.Z[0]:
            orientation_averages.append(np.mean(self.Z[0], axis=0))

        # Calculate average for 90 degrees
        if self.Z[90]:
            orientation_averages.append(np.mean(self.Z[90], axis=0))

        # Calculate average for 180 degrees
        if self.Z[180]:
            orientation_averages.append(np.mean(self.Z[180], axis=0))

        # Combine and calculate average for -90/270 degrees
        combined_neg90_270 = self.Z[-90] + self.Z[270]
        if combined_neg90_270:
            orientation_averages.append(np.mean(combined_neg90_270, axis=0))

        # Average the averages of the four orientations
        self.final_average = np.mean(orientation_averages, axis=0).reshape(1, -1)

        return self.final_average

    def delta_r(self, bridge: np.ndarray) -> np.ndarray:
        """Calculates the delta in output"""

        if bridge.shape[1] != self.num_bridges:
            raise ValueError(f"Bridge must be an array of length {self.num_bridges}, "
                             f"but got shape {bridge.shape}.")

        if self.final_average is None:
            self.average()  # calculate average

        z = self.final_average

        return bridge - z
