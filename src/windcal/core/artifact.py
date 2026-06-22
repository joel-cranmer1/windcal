import json
import uuid
from datetime import datetime
from typing import Union, Tuple
import numpy as np
import logging

from windcal.core.io import STANDARD_CHANNELS
from windcal.core.metadata import CalibrationMetadata

logger = logging.getLogger(__name__)


class BalanceCalibration:
    """The immutable artifact containing the calibration matrix and metadata."""

    def __init__(self,
                 coefficient_matrix: np.ndarray,
                 math_model_type: str,
                 component_names: list,  # e.g. ["N1", "N2", "Y1", "Y2", "AF", "RM"]
                 transformation_matrix: np.ndarray = None,  # Optional 5F/1M -> 3F/3M matrix
                 metadata: CalibrationMetadata = None,
                 ):
        self.uuid = str(uuid.uuid4())
        self.timestamp = datetime.now().isoformat()
        self.math_model_type = math_model_type
        self.coefficient_matrix = coefficient_matrix

        self.component_names = component_names
        self.transformation_matrix = transformation_matrix

        # Unpack CalibrationMetadata
        if metadata is not None:
            self.author = metadata.author
            self.balance_info = metadata.balance_info
            self.max_loads = metadata.max_loads
            self.distances = metadata.distances
        else:
            self.author = "Unknown"
            self.balance_info = {}
            self.max_loads = {}
            self.distances = ()
        self.validate()

    def validate(self):
        """Ensure all provided values make sense and are compatible"""
        logger.debug(f"Validating {self.__class__.__name__}...")
        # Not the correct type
        if not isinstance(self.coefficient_matrix, np.ndarray):
            raise TypeError("Coefficient matrix should be an np.array")
        if not isinstance(self.component_names, list):
            raise TypeError("Component names should be a List")

        # Empty values
        if len(self.coefficient_matrix.shape) < 1:
            raise ValueError("Coefficient matrix cannot be empty.")
        if len(self.component_names) < 1:
            raise TypeError("Component names cannot be empty")

        cx, cy = self.coefficient_matrix.shape
        tx, ty = self.transformation_matrix.shape if self.transformation_matrix is not None else (0, 0)
        n_size = len(self.component_names)
        if cy < cx:
            raise ValueError(f"Coefficient matrix shape M x N - M must be less than or equal to N: ({cx}, {cy})")
        if tx != ty:
            raise ValueError(f"Transformation matrix shape must be square: ({tx}, {ty})")
        if cx != n_size:
            raise ValueError(f"Coefficient matrix shape M x N does not match component names length: ({cx}, {cy}), ({n_size})")
        if self.component_names != STANDARD_CHANNELS and self.transformation_matrix is None:
            raise ValueError(f"Transformation matrix is required if balance is Force or Moment type.")
        if self.transformation_matrix is not None and len(self.distances) != 4:
            raise ValueError(f"Four balance distances are required with a Transformation matrix: {self.distances}")

        logger.debug(f"Validation complete.")

    def save(self, filepath: str):
        logger.debug(f"Saving calibration to file: {filepath}")
        data = {
            "balance_calibration": True,
            "uuid": self.uuid,
            "timestamp": self.timestamp,
            "author": self.author,
            "balance": self.balance_info,
            "max_loads": self.max_loads,
            "distances": self.distances,
            "math_model_type": self.math_model_type,
            "component_names": self.component_names,
            "coefficient_matrix": self.coefficient_matrix.tolist(),
            "transformation_matrix": self.transformation_matrix.tolist() if self.transformation_matrix is not None else None,
        }
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=4)

    @classmethod
    def load(cls, filepath: str):
        """Deserializes the artifact from a JSON file."""

        logger.debug(f"Loading calibration from file: {filepath}")
        with open(filepath, 'r') as f:
            data = json.load(f)

        if not data.get("balance_calibration"):
            return None

        instance = cls(
            coefficient_matrix=np.array(data.get("coefficient_matrix")),
            math_model_type=data.get("math_model_type"),
            component_names=data.get("component_names"),
            transformation_matrix=np.array(data.get("transformation_matrix")),
            metadata=CalibrationMetadata(
                author=data.get("author", "Unknown"),
                balance_info=data.get("balance"),
                max_loads=data.get("max_loads"),
                distances=tuple(data.get("distances"))
            )
        )
        instance.uuid = data["uuid"]  # Restore exact UUID
        instance.timestamp = data["timestamp"]
        return instance

    @staticmethod
    def create_5f1m_transformation_matrix(
            x_nf_fore: Union[float, Tuple[float, float, float, float]],
            x_nf_aft: float = None,
            x_sf_fore: float = None,
            x_sf_aft: float = None
    ) -> np.ndarray:
        """Generates a 6x6 coordinate transformation matrix for 5F/1M balances.

        This matrix resolves raw 5F/1M load components (NF_fore, NF_aft, SF_fore,
        SF_aft, AF, RM) into standard 3F/3M load components (NF, SF, AF, PM, RM, YM).
        See Annex C

        Args:
            x_nf_fore: Distance from the balance electrical center to the forward
                Normal Force gauge. Positive values indicate a forward direction.
            x_nf_aft: Distance from the balance electrical center to the aft
                Normal Force gauge. Positive values indicate an aft direction.
            x_sf_fore: Distance from the balance electrical center to the forward
                Side Force gauge. Positive values indicate a forward direction.
            x_sf_aft: Distance from the balance electrical center to the aft
                Side Force gauge. Positive values indicate an aft direction.

        Returns:
            A 6x6 numpy array representing the geometric transformation matrix.
        """
        if isinstance(x_nf_fore, (tuple, list)):
            x_nf_fore, x_nf_aft, x_sf_fore, x_sf_aft = x_nf_fore

        T = np.zeros((6, 6))

        # 1. Normal Force: NF = NF_fore + NF_aft
        T[0, 0] = 1.0
        T[0, 1] = 1.0

        # 2. Side Force: SF = SF_fore + SF_aft
        T[1, 2] = 1.0
        T[1, 3] = 1.0

        # 3. Axial Force: AF = AF
        T[2, 4] = 1.0

        # 4. Pitching Moment: PM = (NF_fore * x_nf_fore) + (NF_aft * x_nf_aft)
        T[3, 0] = x_nf_fore
        T[3, 1] = -x_nf_aft

        # 5. Rolling Moment: RM = RM
        T[4, 5] = 1.0

        # 6. Yawing Moment: YM = (SF_fore * x_sf_fore) + (SF_aft * x_sf_aft)
        T[5, 2] = x_sf_fore
        T[5, 3] = -x_sf_aft

        return T

    @staticmethod
    def create_1f5m_transformation_matrix(
            x_pm_fore: Union[float, Tuple[float, float, float, float]],
            x_pm_aft: float = None,
            x_ym_fore: float = None,
            x_ym_aft: float = None
    ) -> np.ndarray:
        """Generates a 6x6 coordinate transformation matrix for 1F/5M balances.

        This matrix resolves raw 1F/5M load components (PM_fore, PM_aft, YM_fore,
        YM_aft, AF, RM) into standard 3F/3M load components (NF, SF, AF, PM, RM, YM).
        See Annex C

        Args:
            x_pm_fore: Distance from the balance electrical center to the forward
                Pitch Moment gauge. Positive values indicate a forward direction.
            x_pm_aft: Distance from the balance electrical center to the aft
                Pitch Moment gauge. Positive values indicate an aft direction.
            x_ym_fore: Distance from the balance electrical center to the forward
                Yaw Moment gauge. Positive values indicate a forward direction.
            x_ym_aft: Distance from the balance electrical center to the aft
                Yaw Moment gauge. Positive values indicate an aft direction.

        Returns:
            A 6x6 numpy array representing the geometric transformation matrix.
        """
        if isinstance(x_pm_fore, (tuple, list)):
            x_pm_fore, x_pm_aft, x_ym_fore, x_ym_aft = x_pm_fore

        T = np.zeros((6, 6))

        # 1. Pitch pair (PM_fore, PM_aft) -> NF, PM
        dx_pm = x_pm_aft + x_pm_fore
        if abs(dx_pm) < 1e-12:
            raise ValueError("x_pm_fore and x_pm_aft must be distinct.")

        # NF = (PM_aft - PM_fore) / dx_pm
        T[0, 0] = -1.0 / dx_pm  # PM_fore
        T[0, 1] = 1.0 / dx_pm  # PM_aft

        # PM = (x_pm_aft * (PM_fore) + x_pm_fore * (PM_aft)) / dx_pm
        T[3, 0] = x_pm_aft / dx_pm  # PM_fore
        T[3, 1] = x_pm_fore / dx_pm  # PM_aft

        # 2. Yaw pair (YM_fore, YM_aft) -> SF, YM
        dx_ym = x_ym_aft + x_ym_fore
        if abs(dx_ym) < 1e-12:
            raise ValueError("x_ym_fore and x_ym_aft must be distinct.")

        # SF = (YM_aft - YM_fore) / dx_ym
        T[1, 2] = -1.0 / dx_ym  # YM_fore
        T[1, 3] = 1.0 / dx_ym  # YM_aft

        # YM = (x_ym_fore * (YM_aft) + x_ym_aft * (YM_fore)) / dx_ym
        T[5, 2] = x_ym_aft / dx_ym  # YM_fore
        T[5, 3] = x_ym_fore / dx_ym  # YM_aft

        # 3. Axial Force: AF = AF
        T[2, 4] = 1.0

        # 4. Rolling Moment: RM = RM
        T[4, 5] = 1.0

        return T
