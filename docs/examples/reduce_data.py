import logging

from windcal.core.artifact import BalanceCalibration
from windcal.core.io import ZeroLoadOutput
from windcal.reduction.reducer import DataReducer


class ColorFormatter(logging.Formatter):
    COLORS = {
        logging.DEBUG: "\033[96m",  # light-cyan
        logging.INFO: "\033[34m",  # blue
        logging.WARNING: "\033[33m",  # yellow
        logging.ERROR: "\033[31m",  # red
        logging.CRITICAL: "\033[1;41;30m",  # bold red
    }
    RESET = "\033[0m"

    def format(self, record):
        color = self.COLORS.get(record.levelno, "")
        message = super().format(record)
        return f"{color}{message}{self.RESET}"


handler = logging.StreamHandler()
handler.setFormatter(ColorFormatter("%(asctime)s|%(levelname)s|%(name)s|%(message)s"))

logger = logging.getLogger()
logger.addHandler(handler)
logger.setLevel(logging.DEBUG)


def main():
    cal_file = "balance_sn32780_v1.json"
    Cal = BalanceCalibration.load(cal_file)

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

    reducer = DataReducer(Cal, z)

    v_head = ["rN1", "rN2", "rY1", "rY2", "rAX", "rRM"]
    # v_data = [0.000931816, -0.000158178, -0.000104949, -1.12334e-05, -0.000245912, -4.91633e-05]
    v_data = [-1.44671E-05, -0.00098276, -0.000109654, -3.94336E-05, -0.00026562, -4.75877E-05]
    data = dict(zip(v_head, v_data))

    res = reducer.to_eng_units(data)

    logging.info(res)

    fm_data = reducer.to_force_moment(res)

    logging.info(fm_data)


if __name__ == "__main__":
    main()
