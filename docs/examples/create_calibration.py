import logging

from windcal.calibration.calibrator import Calibrator
from windcal.calibration.models import LinearAbsoluteModel
from windcal.core.io import CalibrationDataSet, ZeroLoadOutput
from windcal.core.metadata import CalibrationMetadata

logging.basicConfig(level=logging.DEBUG, format="%(asctime)s|%(levelname)s|%(name)s|%(message)s")


def main():
    # Load known applied loads and voltage responses
    dataset = CalibrationDataSet.from_csv("32780HalfInch_2025-09-26.csv")

    # Select a math strategy and fit the data
    calibrator = Calibrator(math_model=LinearAbsoluteModel())

    meta = CalibrationMetadata(
        author="Hayden",
        balance_info={
            "type": "5F/1M",
            "manufacturer": "Unknown",
            "serial": "32780",
            "diameter": "0.5 in",
        },
        max_loads={"N1": 112.5, "N2": 112.5, "Y1": 105, "Y2": 105, "AX": 55, "RM": 45, "units": "lb & in-lb"},
        distances=(1.05000, 1.05000, 0.85000, 0.85000),
    )

    z = ZeroLoadOutput()
    z.add([-2.33046e-05, -4.48277e-05, -0.000101021, -8.90045e-06, -0.000266337, -4.12518e-05], 0)
    z.add([-3.48997e-05, -3.17755e-05, -0.000100863, -8.22905e-06, -0.000263974, -4.13878e-05], 0)
    z.add([-5.91319e-05, -4.26745e-05, -0.000106913, -1.3589e-05, -0.000267572, -4.04032e-05], 180)
    z.add([-5.76239e-05, -4.12042e-05, -0.000106011, -1.34061e-05, -0.000268388, -3.92152e-05], 180)
    z.add([-4.45605e-05, -5.32963e-05, -0.000106221, -1.39871e-05, -0.000266819, -3.86485e-05], 180)
    z.add([-4.08495e-05, -4.33127e-05, -8.90984e-05, -1.3735e-05, -0.0002645, -4.05922e-05], 90)
    z.add([-4.05972e-05, -4.30048e-05, -9.07837e-05, -1.46178e-05, -0.000265958, -4.07656e-05], 90)
    z.add([-3.37175e-05, -3.65706e-05, -0.000129428, -1.20966e-05, -0.000267333, -3.91669e-05], 270)
    z.add([-3.42185e-05, -3.67775e-05, -0.000118525, -2.9216e-05, -0.000266199, -4.0736e-05], 270)
    z.average()

    calibration_artifact = calibrator.generate_calibration(dataset, metadata=meta)

    # Save as a versioned JSON artifact
    calibration_artifact.save("balance_sn32780_v1.json")


if __name__ == "__main__":
    main()
