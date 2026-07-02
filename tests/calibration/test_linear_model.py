from windcal.calibration.models import LinearModel

from .test_model_base import CalibrationModelContractTest


class TestLinearModel(CalibrationModelContractTest):
    MODEL_CLASS = LinearModel
