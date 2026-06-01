from abc import ABC, abstractmethod
import numpy as np
from windcal.core.io import CalibrationDataSet


class CalibrationMathModel(ABC):
    """Abstract base class for all balance calibration math strategies."""

    @abstractmethod
    def fit(self, data: CalibrationDataSet) -> np.ndarray:
        """Calculates and returns the calibration coefficient matrix."""
        pass

    @abstractmethod
    def reduce(self, coefficient_matrix: np.ndarray, voltages: np.ndarray) -> np.ndarray:
        """Inverse function: calculates loads from voltages and the matrix."""
        pass


class Linear6x6Model(CalibrationMathModel):
    """A simple 6x6 linear regression model."""

    def fit(self, data: CalibrationDataSet) -> np.ndarray:
        # Simple least squares: [F] = [C][V] -> [C] = [F] * pinv([V])
        # For N points, F is Nx6, V is Nx6.
        # C is 6x6. F^T = C * V^T => C = F^T * pinv(V^T)
        F_T = data.loads.T
        V_T = data.voltages.T
        C = F_T @ np.linalg.pinv(V_T)
        return C

    def reduce(self, coefficient_matrix: np.ndarray, voltages: np.ndarray) -> np.ndarray:
        # Calculates F = C * V
        # Handles both 1D array (fast DAQ) and 2D array (batch)
        if voltages.ndim == 1:
            return coefficient_matrix @ voltages
        else:
            return (coefficient_matrix @ voltages.T).T
