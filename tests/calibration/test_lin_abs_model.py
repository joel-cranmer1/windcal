from windcal.calibration.models import LinearAbsoluteModel

from .test_model_base import CalibrationModelContractTest


class TestLinearAbsoluteModel(CalibrationModelContractTest):
    MODEL_CLASS = LinearAbsoluteModel
