import os
import shutil
import unittest

from daplis.functions.sensor_plot import plot_sensor_population, plot_single_pix_hist


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
        os.chdir(r"{}".format(os.path.dirname(os.path.realpath(__file__)) + "/.."))

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
        os.chdir(r"{}".format(os.path.dirname(os.path.realpath(__file__)) + "/.."))
        plot_sensor_population(
            self.path,
            self.daughterboard_number,
            self.motherboard_number,
            self.firmware_version,
            self.timestamps,
            scale="linear",
            style="-o",
            show_fig=True,
            app_mask=True,
        )
        self.assertTrue(
            os.path.isfile(
                "results/sensor_population/test_data_2212b-test_data_2212b.png"
            )
        )

    # TODO: data for SPDC with background needed
    # def test_c_plot_spdc(self):
    #     # Positive test case
    #     os.chdir(r"{}".format(os.path.realpath(__file__) + "/../.."))
    #     plot_spdc(self.path, self.board_number, self.timestamps, show_fig=True)
    #     self.assertTrue(
    #         os.path.exists(
    #             "results/test_data_2212b-test_data_2212b_SPDC_counts.png"
    #         )
    #     )

    #     # Negative test case
    #     with self.assertRaises(TypeError):
    #         plot_spdc(
    #             self.path, self.board_number, self.timestamps, show_fig="True"
    #         )

    def tearDownClass():
        # Clean up after tests
        os.chdir(r"{}".format(os.path.dirname(os.path.realpath(__file__))))
        shutil.rmtree("test_data/results")


if __name__ == "__main__":
    unittest.main()
