import numpy as np

from windcal.calibration.models import CalibrationMathModel
from windcal.core.artifact import BalanceCalibration
from windcal.core.geometry import ModelGeometry
from windcal.core.io import STANDARD_CHANNELS


class DataReducer:
    """Operational engine for reducing voltages to physical loads.

    Conditions the data into the proper format for the calibration since the column
    order maters very much. The data will be returned in a dictionary of items to
    denote the values' axis
    """

    def __init__(self, calibration: BalanceCalibration, model: ModelGeometry = ModelGeometry()):
        self.calibration = calibration
        self.math_model = CalibrationMathModel.create(calibration.math_model_type)
        self.model_geom = model

    def to_eng_units(self, voltages: dict) -> dict:
        """Process voltages to component units"""
        # Step 1: Get raw balance loads
        raw_loads = self.math_model.reduce(
            self.calibration.coefficient_matrix, self.calibration.bias_vector, self._order_voltages(voltages)
        )
        return self._label_loads(raw_loads)

    def to_force_moment(self, loads: dict) -> dict:
        """Process loads to 3F/3M components"""
        xform = self.calibration.transformation_matrix
        if xform is None:
            return loads

        transformed_loads = xform @ self._order_components(loads)
        return self._label_loads(transformed_loads, keys=STANDARD_CHANNELS)

    def _order_voltages(self, voltages: dict) -> np.ndarray:
        return self._order_components(voltages, pre="r")

    def _order_components(self, component: dict, pre: str = "") -> np.ndarray:
        """Arranges the components in the order defined in the calibration artifact

        Args:
            component: Dictionary of components where the Keys should align with
            the calibration component names
            pre: A prefix string to be added to the component names

        Returns:
            Matrix of components in the correct order as a Numpy array.

        Raises:
            ValueError
        """
        try:
            return np.stack(
                [
                    component[f"{pre}{c}"]  # Use "r" for voltages
                    for c in self.calibration.component_names
                ],
                axis=-1,
            )
        except KeyError as e:
            raise ValueError(f"Missing component: {e.args[0]}")
        except ValueError as e:
            raise ValueError(f"Inconsistent component lengths: {e}")

    def _label_loads(self, loads: np.ndarray, *, keys: list = None) -> dict:
        """Assigns the loads array to a dictionary with unique Keys for later consumption or manipulation

        Transforms the loads array into a dictionary of lists where the Key is defined by the component names in
        the calibration artifact

        Args:
            loads: Numpy array of the components to turn into a dictionary
            keys: List of keys to use (overrides the calibration artifact)

        Returns:
            A dictionary of the components where the Keys align with component names

        Raises:
            ValueError:
        """
        if keys is None:
            header = self.calibration.component_names
        else:
            _k = len(keys)
            _l = loads.shape[1]
            if _k != _l:
                raise ValueError(f"Provided Keys are not the same size as the loads: K={_k}; L={_l};")
            header = keys
        result = {h: col.tolist() for h, col in zip(header, loads.T)}
        return result
