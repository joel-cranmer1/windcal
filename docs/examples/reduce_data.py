import numpy as np
from windcal.core.artifact import BalanceCalibration
from windcal.reduction.reducer import DataReducer


def main():
    cal_file = "balance_sn123_v1.json"
    Cal = BalanceCalibration.load(cal_file)

    reducer = DataReducer(Cal)

    v_head = ["rPA", "rYA", "rPF", "rYF", "rAF", "rRM"]
    v_data = [0.00352692, -0.030615058, -0.002608072, 0.114516169, -0.073077982, 0.963958881]
    data = dict(zip(v_head, v_data))

    res = reducer.to_eng_units(data)

    print(res)


if __name__ == '__main__':
    main()
