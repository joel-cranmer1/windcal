from windcal.core.io import CalibrationDataSet
from windcal.core.metadata import CalibrationMetadata
from windcal.calibration.models import LinearModel
from windcal.calibration.calibrator import Calibrator


def main():
    # Load known applied loads and voltage responses
    dataset = CalibrationDataSet.from_csv("../../tests/core/fixtures/NK_Bio_10lb_200009.csv")

    # Select a math strategy and fit the data
    calibrator = Calibrator(math_model=LinearModel())

    meta = CalibrationMetadata(
        author="J. Cranmer",
        balance_info={
            "type": "1F/5M",
            "manufacturer": "NK Biotech",
            "serial": "200009",
            "diameter": "0.75 in",
        },
        max_loads={
            "aft_pitch": 15,
            "aft_yaw": 7,
            "fwd_pitch": 15,
            "fwd_yaw": 7,
            "axial": 10,
            "roll": 5,
            "units": "lb & in-lb"
        },
        distances=(1.278, 1.278, 1.274, 1.274)
    )

    calibration_artifact = calibrator.generate_calibration(
        dataset,
        metadata=meta
    )

    # Save as a versioned JSON artifact
    calibration_artifact.save("balance_sn123_v1.json")


if __name__ == '__main__':
    main()
