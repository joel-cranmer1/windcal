import json
import uuid
from datetime import datetime
import numpy as np


class BalanceCalibration:
    """The immutable artifact containing the calibration matrix and metadata."""

    def __init__(self,
                 coefficient_matrix: np.ndarray,
                 math_model_type: str,
                 component_names: list,  # e.g. ["N1", "N2", "Y1", "Y2", "AF", "RM"]
                 bias_vector: np.ndarray = None,
                 transformation_matrix: np.ndarray = None,  # Optional 5F/1M -> 3F/3M matrix
                 author: str = "Unknown"):
        self.uuid = str(uuid.uuid4())
        self.timestamp = datetime.now().isoformat()
        self.author = author
        self.math_model_type = math_model_type
        self.coefficient_matrix = coefficient_matrix

        # Default to a zero vector if no bias is provided
        if bias_vector is None:
            self.bias_vector = np.zeros(6)
        else:
            self.bias_vector = bias_vector

        self.component_names = component_names
        self.transformation_matrix = transformation_matrix

    def save(self, filepath: str):
        data = {
            "uuid": self.uuid,
            "timestamp": self.timestamp,
            "author": self.author,
            "math_model_type": self.math_model_type,
            "bias_vector": self.bias_vector,
            "component_names": self.component_names,
            "coefficient_matrix": self.coefficient_matrix.tolist(),
            "transformation_matrix": self.transformation_matrix.tolist() if self.transformation_matrix is not None else None
        }
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=4)

    @classmethod
    def load(cls, filepath: str):
        """Deserializes the artifact from a JSON file."""
        with open(filepath, 'r') as f:
            data = json.load(f)

        instance = cls(
            coefficient_matrix=np.array(data["coefficient_matrix"]),
            math_model_type=data["math_model_type"],
            bias_vector=data["bias_vector"],
            component_names=data["component_names"],
            transformation_matrix=np.array(data["transformation_matrix"]),
            author=data.get("author", "Unknown")
        )
        instance.uuid = data["uuid"]  # Restore exact UUID
        instance.timestamp = data["timestamp"]
        return instance

    @staticmethod
    def create_5f1m_transformation_matrix(
            x_nf_fore: float,
            x_nf_aft: float,
            x_sf_fore: float,
            x_sf_aft: float
    ) -> np.ndarray:
        """Generates a 6x6 coordinate transformation matrix for 5F/1M balances.

        This matrix resolves raw 5F/1M load components (NF_fore, NF_aft, SF_fore,
        SF_aft, AF, RM) into standard 3F/3M load components (NF, SF, AF, PM, RM, YM).

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

        # 6. Yawing Moment: YM = -(SF_fore * x_sf_fore) - (SF_aft * x_sf_aft)
        # (Standard RHR convention: positive force forward of CG yields negative yaw moment)
        T[5, 2] = -x_sf_fore
        T[5, 3] = x_sf_aft

        return T

    @staticmethod
    def create_1f5m_transformation_matrix(
        x_pm_fore: float,
        x_pm_aft: float,
        x_ym_fore: float,
        x_ym_aft: float
    ) -> np.ndarray:
        """Generates a 6x6 coordinate transformation matrix for 1F/5M balances.

        This matrix resolves raw 1F/5M load components (PM_fore, PM_aft, YM_fore,
        YM_aft, AF, RM) into standard 3F/3M load components (NF, SF, AF, PM, RM, YM).

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
        T = np.zeros((6, 6))

        # 1. Pitch pair (PM_fore, PM_aft) -> NF, PM
        dx_pm = x_pm_aft + x_pm_fore
        if abs(dx_pm) < 1e-12:
            raise ValueError("x_pm_fore and x_pm_aft must be distinct.")

        # NF = (PM_fore - PM_aft) / dx_pm
        T[0, 0] = 1.0 / dx_pm  # PM_fore
        T[0, 1] = -1.0 / dx_pm  # PM_aft

        # PM = PM_fore + NF * x_pm_fore
        T[3, 0] = 1.0 + x_pm_fore / dx_pm
        T[3, 1] = -x_pm_fore / dx_pm

        # 2. Yaw pair (YM_fore, YM_aft) -> SF, YM
        dx_ym = x_ym_aft + x_ym_fore
        if abs(dx_ym) < 1e-12:
            raise ValueError("x_ym_fore and x_ym_aft must be distinct.")

        # SF = (YM_fore - YM_aft) / dx_ym
        T[1, 2] = 1.0 / dx_ym  # YM_fore
        T[1, 3] = -1.0 / dx_ym  # YM_aft

        # YM = YM_fore + SF * x_ym_fore
        T[5, 2] = 1.0 + x_ym_fore / dx_ym
        T[5, 3] = -x_ym_fore / dx_ym

        # 3. Axial Force: AF = AF
        T[2, 4] = 1.0

        # 4. Rolling Moment: RM = RM
        T[4, 5] = 1.0

        return T
