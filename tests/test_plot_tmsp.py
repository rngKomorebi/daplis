import os
import shutil
import unittest

from daplis.functions.sensor_plot import (
    plot_sensor_population,
    plot_single_pix_hist,
    unpickle_plot,
)


class TestPlotScripts(unittest.TestCase):
    def setUp(self):
        self.path = "tests/test_data"
        self.pix = 15
        self.daughterboard_number = "NL11"
        self.motherboard_number = "#33"
        self.firmware_version = "2212b"
        self.timestamps = 300

    def test_a_plot_pixel_hist(self):
        # Positive test case
        os.chdir(
            r"{}".format(os.path.dirname(os.path.realpath(__file__)) + "/..")
        )

        plot_single_pix_hist(
            self.path,
            self.pix,
            self.daughterboard_number,
            self.motherboard_number,
            self.firmware_version,
            self.timestamps,
        )
        self.assertTrue(
            os.path.exists(
                "results/single pixel histograms/test_data_2212b.dat, pixel 15.png"
            )
        )

    def test_b_plot_sen_pop(self):
        # Positive test case
        os.chdir(
            r"{}".format(os.path.dirname(os.path.realpath(__file__)) + "/..")
        )
        plot_sensor_population(
            self.path,
            self.daughterboard_number,
            self.motherboard_number,
            self.firmware_version,
            self.timestamps,
        )
        self.assertTrue(
            os.path.isfile(
                "results/sensor_population/test_data_2212b-test_data_2212b_rates.png"
            )
        )

    def test_c_plot_sen_pop_pickle(self):
        # Test that plot_sensor_population creates pickle files when requested
        os.chdir(
            r"{}".format(os.path.dirname(os.path.realpath(__file__)) + "/..")
        )
        plot_sensor_population(
            self.path,
            self.daughterboard_number,
            self.motherboard_number,
            self.firmware_version,
            self.timestamps,
            pickle_fig=True,
        )
        self.assertTrue(
            os.path.isfile(
                "results/sensor_population/test_data_2212b-test_data_2212b_rates.pickle"
            )
        )

    def test_d_unpickle_sensor_plot(self):
        # Test that unpickle_plot returns valid figure and data
        work_dir = os.path.dirname(os.path.realpath(__file__)) + "/.."
        pkl_file = os.path.join(
            work_dir,
            self.path,
            "results/sensor_population/test_data_2212b-test_data_2212b_rates.pickle",
        )
        result = unpickle_plot(pkl_file)
        self.assertIsNotNone(result)

    def tearDownClass():
        # Clean up after tests
        os.chdir(r"{}".format(os.path.dirname(os.path.realpath(__file__))))
        shutil.rmtree("test_data/results")


if __name__ == "__main__":
    unittest.main()
