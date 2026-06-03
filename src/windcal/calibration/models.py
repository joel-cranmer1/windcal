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
    def reduce(self, coefficient_matrix: np.ndarray, bias_vector: np.ndarray, voltages: np.ndarray) -> np.ndarray:
        """Inverse function: calculates loads from voltages and the matrix."""
        pass


class LinearModel(CalibrationMathModel):
    """A simple 6x6 linear regression model with a bias (intercept) term."""

    def fit(self, data: CalibrationDataSet) -> tuple[np.ndarray, np.ndarray]:
        """Calculates the calibration matrix and bias vector.

        Uses least-squares to solve the system [R] = [a] + [C][G]. (Eq-3.1.8)
        Where:
        - R is the bridge output,
        - a is the intercepts (bias),
        - C is calibration matrix, and
        - G is the component load matrix

        Args:
            data: A CalibrationDataSet instance containing the N x 6 loads
                and N x 6 voltages.

        Returns:
            A tuple containing:
                - A 6x6 numpy array representing the calibration matrix C.
                - A 1D numpy array of length 6 representing the bias vector B.
        """

        # Extract inputs
        G = data.loads  # shape (N, 6)
        R = data.voltages  # shape (N, 6)

        # Validate shapes (optional but useful)
        if G.shape != R.shape or G.shape[1] != 6:
            raise ValueError("loads and voltages must both be N x 6 arrays")

        # Add bias column (intercept term)
        ones = np.ones((G.shape[0], 1))
        G_aug = np.hstack((ones, G))   # shape (N, 7)

        # Solve least squares: G_aug * X = R
        # X will be shape (7, 6)
        X, residuals, rank, s = np.linalg.lstsq(G_aug, R, rcond=None)

        # Extract bias and calibration matrix
        bias = X[0, :]  # shape (6,)
        C = X[1:, :].T  # shape (6, 6) and transpose

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
