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


class LinearModel(CalibrationMathModel):
    """A simple 6x6 linear regression model."""
    """A simple 6x6 linear regression model with a bias (intercept) term."""

    def fit(self, data: CalibrationDataSet) -> tuple[np.ndarray, np.ndarray]:
        """Calculates the calibration matrix and bias vector.

        Uses least-squares to solve the augmented system [V | 1] * [C | B]^T = F.

        Args:
            data: A CalibrationDataSet instance containing the N x 6 loads
                and N x 6 voltages.

        Returns:
            A tuple containing:
                - A 6x6 numpy array representing the calibration matrix C.
                - A 1D numpy array of length 6 representing the bias vector B.
        """
        N = data.voltages.shape[0]

        # Pad voltages with a column of 1s to solve for the intercept (bias)
        # V_aug shape: (N, 7)
        V_aug = np.hstack([data.voltages, np.ones((N, 1))])

        # Solve V_aug * C_aug^T = F
        C_aug_T, _, _, _ = np.linalg.lstsq(V_aug, data.loads, rcond=None)

        # Transpose back. C_aug is now a 6x7 matrix.
        C_aug = C_aug_T.T

        # The first 6 columns are the sensitivities (C), the 7th is the bias (B)
        C = C_aug[:, :6]
        bias = C_aug[:, 6]

        return C, bias

    def reduce(
            self,
            coefficient_matrix: np.ndarray,
            bias_vector: np.ndarray,
            voltages: np.ndarray
    ) -> np.ndarray:
        """Calculates physical loads from voltages using the matrix and bias.

        Args:
            coefficient_matrix: A 6x6 numpy array representing the matrix C.
            bias_vector: A 1D numpy array of length 6 representing the bias B.
            voltages: A numpy array of measured voltages (1D or 2D).

        Returns:
            A numpy array of physical loads (F = C * V + B).
        """
        if voltages.ndim == 1:
            return (coefficient_matrix @ voltages) + bias_vector

        return (coefficient_matrix @ voltages.T).T + bias_vector
