import logging
from windcal.core.artifact import BalanceCalibration
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


# logging.debug("Debug message")
# logging.info("Info message")
# logging.warning("Warning message")
# logging.error("Error message")
# logging.critical("Critical message")


def main():
    cal_file = "balance_sn32780_v1.json"
    Cal = BalanceCalibration.load(cal_file)

    reducer = DataReducer(Cal)

    v_head = ["rN1", "rN2", "rY1", "rY2", "rAX", "rRM"]
    v_data = [0.000931816, -0.000158178, -0.000104949, -1.12334e-05, -0.000245912, -4.91633e-05]
    data = dict(zip(v_head, v_data))

    res = reducer.to_eng_units(data)

    logging.info(res)


if __name__ == "__main__":
    main()
