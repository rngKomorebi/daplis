import glob
import os
import shutil
import unittest

import numpy as np
import pandas as pd

from daplis.functions.cross_talk import (
    calculate_dark_count_rate,
    collect_cross_talk,
    plot_cross_talk,
)


class TestCTFull(unittest.TestCase):
    def setUp(self):
        # Set up test variables
        self.path = "tests/test_data"
        self.pixels = [x for x in range(0, 20)]
        self.daughterboard_number = "NL11"
        self.motherboard_number = "#33"
        self.firmware_version = "2212s"
        self.timestamps = 300
        self.delta_window = 10e3
        self.step = 1
        self.pix1 = 0
        self.scale = "linear"
        self.include_offset = False

    def test_a_collect_ct_positive(self):
        work_dir = r"{}".format(os.path.dirname(os.path.realpath(__file__)) + "/..")
        os.chdir(work_dir)
        # Test positive case of collect_ct function
        collect_cross_talk(
            self.path,
            self.pixels,
            self.daughterboard_number,
            self.motherboard_number,
            self.firmware_version,
            self.timestamps,
            self.delta_window,
            self.step,
            self.include_offset,
        )
        # Check if the output file is created and has the correct number of rows
        file = glob.glob("*CT_data_*.csv*")[0]
        data = pd.read_csv(file, header=None)
        self.assertEqual(len(data), 20)

    def test_b_collect_ct_negative(self):
        work_dir = r"{}".format(os.path.dirname(os.path.realpath(__file__)) + "/..")
        os.chdir(work_dir)
        # Test negative case of collect_ct function
        with self.assertRaises(TypeError):
            collect_cross_talk(
                self.path,
                self.pixels,
                123,
                22,
                self.timestamps,
                self.delta_window,
            )

    def test_c_plot_ct_positive(self):
        # Test positive case of plot_ct function
        work_dir = r"{}".format(os.path.dirname(os.path.realpath(__file__)) + "/..")
        os.chdir(work_dir)
        plot_cross_talk(self.path, self.pix1, self.scale)
        # Check if the plot file is created
        plot_name = "test_data_2212b.dat_test_data_2212b.dat"
        plot_file = "{plot}_{pix}.png".format(plot=plot_name, pix=self.pix1)
        self.assertTrue(os.path.exists(plot_file))

    def test_d_plot_ct_negative(self):
        # Test negative case of plot_ct function
        with self.assertRaises(FileNotFoundError):
            plot_cross_talk("nonexistent_folder", self.pix1, self.scale)

    def test_calculate_dark_count_rate_positive(self):
        work_dir = r"{}".format(os.path.dirname(os.path.realpath(__file__)) + "/..")
        os.chdir(work_dir)
        result = calculate_dark_count_rate(
            self.path,
            self.daughterboard_number,
            self.motherboard_number,
            self.firmware_version,
            self.timestamps,
        )
        # Add your assertions here based on expected results
        assert isinstance(result, float)
        assert result >= 0  # Assuming the result should be non-negative

    def tearDownClass():
        # Clean up after tests
        os.chdir(r"{}".format(os.path.dirname(os.path.realpath(__file__))))
        shutil.rmtree("test_data/cross_talk_data")
        shutil.rmtree("test_data/results")


if __name__ == "__main__":
    unittest.main()
