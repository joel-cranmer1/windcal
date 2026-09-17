from typing import Tuple

import numpy as np

from windcal.calibration.models import LinearAbsoluteModel
from windcal.core.io import STANDARD_CHANNELS, CalibrationDataSet

from .test_model_base import CalibrationModelContractTest


class TestLinearAbsoluteModel(CalibrationModelContractTest):
    MODEL_CLASS = LinearAbsoluteModel
    feature_list = [lambda x: np.abs(x)]

    def create_simple_dataset(self) -> Tuple[CalibrationDataSet, np.ndarray]:
        """Default simple dataset (override only if needed)"""
        mod_C = np.eye(6) * 0.5
        C_expected = np.hstack([mod_C, mod_C])
        a_expected = np.ones(6) * 2

        loads = np.append(np.eye(6), np.eye(6) * 2, axis=0)
        loads = np.hstack([loads, abs(loads)])
        voltages = loads @ C_expected.T + a_expected

        data = CalibrationDataSet(loads, voltages, STANDARD_CHANNELS)
        return data, C_expected
