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
    def reduce(self, calibration_matrix: np.ndarray, bias_vector: np.ndarray, voltages: np.ndarray) -> np.ndarray:
        """Inverse function: calculates loads from voltages and the matrix."""
        pass

    # ---------------- ^   Abstract    ^ ----------------
    # ---------------- v Class Methods v ----------------

    _registry: dict[str, type] = {}

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        # Register the subclass using its name or a custom identifier
        cls._registry[cls.__name__] = cls

    @classmethod
    def create(cls, name: str, *args, **kwargs) -> "CalibrationMathModel":
        """Create a CalibrationMathModel instance from the class type's string name"""
        if name not in cls._registry:
            raise ValueError(f"Unknown model type: {name}")
        return cls._registry[name](*args, **kwargs)


class LinearModel(CalibrationMathModel):
    """A simple 6x6 linear regression model with a bias (intercept) term."""

    def fit(self, data: CalibrationDataSet) -> tuple[np.ndarray, np.ndarray]:
        """Calculates the calibration matrix and bias vector.

        Uses least-squares to solve the system [R] = [a] + [C][G]. (Eq-3.1.8)
        Where:
        - R is the bridge output (voltage),
        - a is the intercepts (bias),
        - C is calibration matrix, and
        - G is the component load matrix

        Args:
            data: A CalibrationDataSet instance containing the 6 x N loads
                and 6 x N voltages.

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

        # Solve least-squares: G_aug * X = R
        # X will be shape (7, 6)
        X, residuals, rank, s = np.linalg.lstsq(G_aug, R, rcond=None)

        # Extract bias and calibration matrix
        bias = X[0, :]  # shape (6,)
        C = X[1:, :].T  # shape (6, 6) and transpose

        # --- R^2 Calculation ---
        # 1. Calculate predicted R (R_hat)
        R_predicted = G_aug @ X  # shape (N, 6)

        # 2. Calculate Residual Sum of Squares (SS_res) per component
        # axis=0 computes the sum down the columns (for each of the 6 components)
        ss_res = np.sum((R - R_predicted) ** 2, axis=0)

        # 3. Calculate Total Sum of Squares (SS_tot) per component
        r_mean = np.mean(R, axis=0)
        ss_tot = np.sum((R - r_mean) ** 2, axis=0)

        # 4. Calculate R^2 per component
        # Avoid division by zero if a component has zero variance
        r2 = np.zeros(6)
        valid_indices = ss_tot > 0
        r2[valid_indices] = 1.0 - (ss_res[valid_indices] / ss_tot[valid_indices])

        print("Coefficient of determination (R^2):")
        print(r2)

        return C, bias

    def reduce(
            self,
            calibration_matrix: np.ndarray,
            bias_vector: np.ndarray,
            voltages: np.ndarray
    ) -> np.ndarray:
        """Calculates physical loads from voltages using the matrix and bias.
        Assumes the data is already aligned with the calibration matrix.

        Args:
            calibration_matrix: A 6x6 numpy array representing the matrix C.
            bias_vector: A 1D numpy array of length 6 representing the bias B.
            voltages: A numpy array of measured voltages (1D or 2D).

        Returns:
            A numpy array of physical loads (F = C^-1 * (V - B)).
        """
        c_inv = np.linalg.inv(calibration_matrix)

        return (c_inv @ (voltages - bias_vector).T).T
