import os
import unittest

from windcal.core.io import CalibrationDataSet


class TestCalibrationDataSet(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.base_dir = os.path.dirname(os.path.abspath(__file__))
        # Define the exact path to static fixture directory
        cls.fixture_dir = os.path.join(cls.base_dir, 'fixtures')

    def test_load_csv(self):
        data_set = CalibrationDataSet.from_csv(os.path.join(self.fixture_dir, 'Test_cal_data_SymLin.csv'))
        self.assertTrue(data_set.loads.size > 0)
        self.assertTrue(data_set.voltages.size > 0)

    # Never raises error due to Pandas filling in the blank csv data
    # def test_load_csv_omit_voltage(self):
    #     with self.assertRaises(ValueError):
    #         data_set = CalibrationDataSet.from_csv(os.path.join(self.fixture_dir, 'Test_cal_data_omit_volt.csv'))
    #         print(f"Loads: {data_set.loads.size}")
    #         print(f"Volts: {data_set.voltages.size}")

    def test_load_csv_extra_ch(self):
        with self.assertRaises(ValueError):
            data_set = CalibrationDataSet.from_csv(os.path.join(self.fixture_dir, 'Test_cal_data_extra_ch.csv'))
            print(f"Loads: {data_set.loads.size}")
            print(f"Volts: {data_set.voltages.size}")


if __name__ == '__main__':
    unittest.main()
