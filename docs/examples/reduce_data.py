import numpy as np
from windcal.core.artifact import BalanceCalibration
from windcal.reduction.reducer import DataReducer


def main():
    cal_file = "balance_sn123_v1.json"
    Cal = BalanceCalibration.load(cal_file)

    reducer = DataReducer(Cal)

    data = np.array([0.00352692, -0.030615058, -0.002608072, 0.114516169, -0.073077982, 0.963958881])

    res = reducer.process_point(data)

    for i in range(len(Cal.component_names)):
        print(f"{Cal.component_names[i]}: {res[i]}")


if __name__ == '__main__':
    main()
