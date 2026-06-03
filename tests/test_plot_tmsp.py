import os
import shutil
import unittest

import numpy as np

from daplis.functions.sensor_plot import (
    collect_data_and_apply_mask,
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

    def test_a_neg_firmware_type_raises(self):
        work_dir = os.path.dirname(os.path.realpath(__file__)) + "/.."
        path = os.path.join(work_dir, self.path)
        with self.assertRaises(TypeError):
            plot_single_pix_hist(
                path, self.pix, self.daughterboard_number,
                self.motherboard_number, firmware_version=12345,
            )

    def test_a_neg_daughterboard_type_raises(self):
        work_dir = os.path.dirname(os.path.realpath(__file__)) + "/.."
        path = os.path.join(work_dir, self.path)
        with self.assertRaises(TypeError):
            plot_single_pix_hist(
                path, self.pix, daughterboard_number=99,
                motherboard_number=self.motherboard_number,
                firmware_version=self.firmware_version,
            )

    def test_a_neg_motherboard_type_raises(self):
        work_dir = os.path.dirname(os.path.realpath(__file__)) + "/.."
        path = os.path.join(work_dir, self.path)
        with self.assertRaises(TypeError):
            plot_single_pix_hist(
                path, self.pix, self.daughterboard_number,
                motherboard_number=99,
                firmware_version=self.firmware_version,
            )

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

    def test_b_neg_firmware_type_raises(self):
        work_dir = os.path.dirname(os.path.realpath(__file__)) + "/.."
        path = os.path.join(work_dir, self.path)
        with self.assertRaises(TypeError):
            plot_sensor_population(
                path, self.daughterboard_number,
                self.motherboard_number, firmware_version=12345,
            )

    def test_b_neg_daughterboard_type_raises(self):
        work_dir = os.path.dirname(os.path.realpath(__file__)) + "/.."
        path = os.path.join(work_dir, self.path)
        with self.assertRaises(TypeError):
            plot_sensor_population(
                path, daughterboard_number=99,
                motherboard_number=self.motherboard_number,
                firmware_version=self.firmware_version,
            )

    def test_b_neg_motherboard_type_raises(self):
        work_dir = os.path.dirname(os.path.realpath(__file__)) + "/.."
        path = os.path.join(work_dir, self.path)
        with self.assertRaises(TypeError):
            plot_sensor_population(
                path, self.daughterboard_number,
                motherboard_number=99,
                firmware_version=self.firmware_version,
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


class TestCollectData(unittest.TestCase):
    def setUp(self):
        work_dir = os.path.dirname(os.path.realpath(__file__)) + "/.."
        self.files = [
            os.path.join(work_dir, "tests/test_data/test_data_2212b.dat")
        ]
        self.daughterboard_number = "NL11"
        self.motherboard_number = "#33"
        self.firmware_version = "2212b"
        self.timestamps = 300

    def test_a_returns_array_of_256(self):
        result = collect_data_and_apply_mask(
            self.files,
            self.daughterboard_number,
            self.motherboard_number,
            self.firmware_version,
            self.timestamps,
            apply_hot_pixel_mask=False,
        )
        self.assertIsInstance(result, np.ndarray)
        self.assertEqual(result.shape, (256,))

    def test_b_calculate_rates_returns_tuple(self):
        result = collect_data_and_apply_mask(
            self.files,
            self.daughterboard_number,
            self.motherboard_number,
            self.firmware_version,
            self.timestamps,
            apply_hot_pixel_mask=False,
            calculate_rates=True,
        )
        self.assertIsInstance(result, tuple)
        self.assertEqual(len(result), 2)
        timestamps_per_pixel, rates = result
        self.assertEqual(timestamps_per_pixel.shape, (256,))
        self.assertEqual(rates.shape, (256,))

    def test_c_single_file_as_string(self):
        # Function should accept a bare string path, not just a list
        result = collect_data_and_apply_mask(
            self.files[0],
            self.daughterboard_number,
            self.motherboard_number,
            self.firmware_version,
            self.timestamps,
            apply_hot_pixel_mask=False,
        )
        self.assertEqual(result.shape, (256,))


if __name__ == "__main__":
    unittest.main()
