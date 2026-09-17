import logging
from abc import ABC, abstractmethod

import numpy as np

from windcal.core.io import CalibrationDataSet, ZeroLoadOutput

logger = logging.getLogger(__name__)


class CalibrationMathModel(ABC):
    """Abstract base class for all balance calibration math strategies."""
    zlo = ZeroLoadOutput()

    @abstractmethod
    def fit(self, data: CalibrationDataSet) -> np.ndarray:
        """Calculates and returns the calibration coefficient matrix."""
        pass

    @abstractmethod
    def reduce(self, calibration_matrix: np.ndarray, voltages: np.ndarray) -> np.ndarray:
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

    def fit(self, data: CalibrationDataSet) -> np.ndarray:
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
        logger.debug(f"Starting {self.__class__.__name__} data fit routine.")
        # Extract inputs
        G = data.loads  # shape (N, 6)
        R = data.voltages  # shape (N, 6)

        # Validate shapes (optional but useful)
        if G.shape != R.shape or G.shape[1] != 6:
            raise ValueError("loads and voltages must both be N x 6 arrays")

        # Add bias column (intercept term)
        ones = np.ones((G.shape[0], 1))
        G_aug = np.hstack((ones, G))  # shape (N, 7)

        # Solve least-squares: G_aug * X = R
        # X will be shape (7, 6)
        X, residuals, rank, s = np.linalg.lstsq(G_aug, R, rcond=None)

        # Extract bias and calibration matrix
        # bias = X[0, :]  # shape (6,)
        C = X[1:, :].T  # shape (6, 6) and transpose

        # # --- R^2 Calculation ---
        # # 1. Calculate predicted R (R_hat)
        # R_predicted = G_aug @ X  # shape (N, 6)
        #
        # # 2. Calculate Residual Sum of Squares (SS_res) per component
        # # axis=0 computes the sum down the columns (for each of the 6 components)
        # ss_res = np.sum((R - R_predicted) ** 2, axis=0)
        #
        # # 3. Calculate Total Sum of Squares (SS_tot) per component
        # r_mean = np.mean(R, axis=0)
        # ss_tot = np.sum((R - r_mean) ** 2, axis=0)
        #
        # # 4. Calculate R^2 per component
        # # Avoid division by zero if a component has zero variance
        # r2 = np.zeros(6)
        # valid_indices = ss_tot > 0
        # r2[valid_indices] = 1.0 - (ss_res[valid_indices] / ss_tot[valid_indices])
        #
        # print("Coefficient of determination (R^2):")
        # print(r2)

        return C

    def reduce(self, calibration_matrix: np.ndarray, voltages: np.ndarray) -> np.ndarray:
        """Calculates physical loads from voltages using the matrix and bias.
        Assumes the data is already aligned with the calibration matrix.

        Args:
            calibration_matrix: A 6x6 numpy array representing the matrix C.
            voltages: A numpy array of measured voltages (1D or 2D).

        Returns:
            A numpy array of physical loads (F = C^-1 * (delR)).
        """
        c_inv = np.linalg.inv(calibration_matrix)

        return (c_inv @ voltages.T).T


class LinearAbsoluteModel(CalibrationMathModel):
    """A simple 6x6 linear regression model with a bias (intercept) term."""

    def fit(self, data: CalibrationDataSet) -> np.ndarray:
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
                - A numpy array representing the calibration matrix C.
        """
        logger.debug(f"Starting {self.__class__.__name__} data fit routine.")
        # Extract inputs
        G = data.loads  # shape (N, 6)
        R = data.voltages  # shape (N, 6)

        # Validate shapes (optional but useful)
        if G.shape != R.shape or G.shape[1] != 6:
            raise ValueError("loads and voltages must both be N x 6 arrays")

        F_ab = np.abs(G)

        # Combine columns [1, F, ... |F|, ...]
        ones = np.ones((G.shape[0], 1))
        G_aug = np.hstack((ones, G, F_ab))

        # Solve least-squares: R = C' * G'
        X, residuals, rank, s = np.linalg.lstsq(G_aug, R, rcond=None)

        # Extract bias and calibration matrix
        # bias = X[0, :]  # shape (6,)
        C = X[1:, :].T  # shape (12, 6) and transpose

        return C

    def reduce(self, coefficient_matrix: np.ndarray, voltages: np.ndarray, max_iter: int = 10,
               tol: float = 1e-3) -> np.ndarray:
        """
        Calculates physical loads from voltages using the iterative method
        from AIAA R-091A-2020, Section 3.5.3.

        Args:
            coefficient_matrix: The (13, 6) numpy array from the fit() method.
            voltages: A (p, 6) numpy array of measured voltages.
            max_iter: The maximum number of iterations to perform.
            tol: The convergence tolerance for the load vector.

        Returns:
            A (p, 6) numpy array of the calculated physical loads.
        """
        res_flat = False  # Flag to flatten the result
        # 1. Decompose the coefficient matrix as per the AIAA standard
        C = coefficient_matrix.T  # This is the (6, 12) calibration matrix
        C1 = C[:6, :]  # The (6, 6) linear part
        C2 = C[6:, :]  # The (6, 6) non-linear (absolute value) part

        # Pre-calculate the inverse of the linear matrix, as it's constant
        try:
            C1_inv = np.linalg.inv(C1)
        except np.linalg.LinAlgError as e:
            logger.error("The linear part of the calibration matrix (C1) is singular and cannot be inverted.")
            raise e

        # Pre-calculate the combined matrix
        C1invC2 = C1_inv @ C2

        # Ensure input is 2D for consistent processing
        if voltages.ndim == 1:
            res_flat = True
            voltages = voltages.reshape(1, -1)

        # 2. Perform the iterative calculation for each row of voltage data
        all_reduced_loads = []
        for v_row in voltages:
            # Subtract the ZLO to get delta_R for this data point
            delta_R = self.zlo.delta_r(v_row)

            # Calculate the constant linear part of the solution
            F_linear_part = C1_inv @ delta_R

            # Initial guess is the linear-only solution
            loads_t_minus_1 = F_linear_part

            for i in range(max_iter):
                H_t_minus_1 = np.abs(loads_t_minus_1)

                # Apply the optimized iterative equation (Eq. 3.3.7)
                loads_t = F_linear_part - (C1invC2 @ H_t_minus_1)

                if np.linalg.norm(loads_t - loads_t_minus_1) < tol:
                    break
                loads_t_minus_1 = loads_t
                logger.debug(f"try new loads: {loads_t}")
            else:
                # This 'else' belongs to the 'for' loop. It runs if the loop completes without a 'break'.
                logger.warning(f"Reduction did not converge within {max_iter} iterations for voltage row {v_row}.")

            all_reduced_loads.append(loads_t)
            loads_out = np.array(all_reduced_loads)

            # revert to 1-D array
            if res_flat:
                loads_out = loads_out.reshape(-1)

        return loads_out
