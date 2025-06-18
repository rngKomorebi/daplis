import os
import shutil
import unittest

import numpy as np

from daplis.functions.delta_t import (
    calculate_and_save_timestamp_differences,
    collect_and_plot_timestamp_differences,
)
from daplis.functions.fits import (
    fit_with_gaussian,
    fit_with_gaussian_all,
    fit_with_gaussian_fancy,
)


class TestDeltasFull(unittest.TestCase):
    def setUp(self):
        # Set up test variables
        self.partial_path = "tests/test_data"
        self.pixels = [
            [x for x in range(66, 70)],
            [x for x in range(170, 178)],
        ]
        self.daughterboard_number = "NL11"
        self.motherboard_number = "#33"
        self.firmware_version = "2212b"
        self.timestamps = 300
        self.delta_window = 20e3
        self.rewrite = True
        self.range_left = -20e3
        self.range_right = 20e3
        self.same_y = False
        self.app_mask = True
        self.include_offset = False

    def test_a_deltas_save_positive(self):
        # Test positive case for deltas_save function
        work_dir = os.path.dirname(os.path.realpath(__file__)) + "/.."

        # os.chdir(work_dir)

        path = os.path.join(work_dir, self.partial_path)
        calculate_and_save_timestamp_differences(
            path,
            self.pixels,
            self.rewrite,
            self.daughterboard_number,
            self.motherboard_number,
            self.firmware_version,
            self.timestamps,
            self.delta_window,
            self.app_mask,
            self.include_offset,
        )

        os.chdir(
            os.path.join(
                os.path.dirname(os.path.realpath(__file__)), "test_data"
            )
        )

        # Check if the csv file is created
        self.assertTrue(
            os.path.isfile(
                "delta_ts_data/test_data_2212b-test_data_2212b.feather"
            )
        )

    def test_a_deltas_save_fast_positive(self):
        # Test positive case for deltas_save function
        work_dir = os.path.dirname(os.path.realpath(__file__)) + "/.."

        # os.chdir(work_dir)

        path = os.path.join(work_dir, self.partial_path)
        calculate_and_save_timestamp_differences(
            path,
            self.pixels,
            self.rewrite,
            self.daughterboard_number,
            self.motherboard_number,
            self.firmware_version,
            self.timestamps,
            self.delta_window,
            app_mask=self.app_mask,
            include_offset=self.include_offset,
        )

        os.chdir(
            os.path.join(
                os.path.dirname(os.path.realpath(__file__)), "test_data"
            )
        )

        # Check if the csv file is created
        self.assertTrue(
            os.path.isfile(
                "delta_ts_data/test_data_2212b-test_data_2212b.feather"
            )
        )

    # Negative test case
    # Invalid firmware version
    def test_b_deltas_save_negative(self):
        work_dir = os.path.dirname(os.path.realpath(__file__)) + "/.."

        # os.chdir(work_dir)
        path = os.path.join(work_dir, self.partial_path)

        # Test negative case for deltas_save function
        with self.assertRaises(TypeError):
            calculate_and_save_timestamp_differences(
                path,
                self.pixels,
                "2212",
                self.daughterboard_number,
                self.motherboard_number,
                self.firmware_version,
                self.timestamps,
                self.delta_window,
            )

    def test_c_delta_cp(self):
        # Test case for delta_cp function
        # Positive test case
        work_dir = os.path.dirname(os.path.realpath(__file__)) + "/.."

        # os.chdir(work_dir)
        path = os.path.join(work_dir, self.partial_path)

        collect_and_plot_timestamp_differences(
            path,
            pixels=[x for x in range(67, 69)] + [x for x in range(173, 175)],
            rewrite=self.rewrite,
            range_left=self.range_left,
            range_right=self.range_right,
            same_y=self.same_y,
        )

        # Check if the plot file is created
        self.assertTrue(
            os.path.isfile(
                os.path.join(
                    path,
                    "results/delta_t/test_data_2212b-test_data_2212b_delta_t_grid.png",
                )
            )
        )

    def test_d_fit_with_gaussian_positive(self):
        # Test with valid input
        work_dir = os.path.dirname(os.path.realpath(__file__)) + "/.."

        path = os.path.join(work_dir, self.partial_path)
        pixels = [67, 174]
        window = 20e3
        multiplier = 5

        # Call the function
        fit_with_gaussian(
            path,
            pixels,
            ft_file=None,
            window=window,
            multiplier=multiplier,
        )

        # Assert that the function runs without raising any exceptions
        self.assertTrue(
            os.path.isfile(
                os.path.join(
                    path,
                    "results/fits/test_data_2212b-test_data_2212b_"
                    f"pixels_{pixels[0]},{pixels[1]}_fit.png",
                )
            )
        )

    def test_d_fit_with_gaussian_all_positive(self):
        # Test with valid input
        pixels = [82, 116]

        work_dir = os.path.dirname(os.path.realpath(__file__)) + "/.."

        path = os.path.join(work_dir, self.partial_path)

        ft_file = r"test.feather"

        # Call the function
        fit_with_gaussian_all(
            path,
            pixels=pixels,
            ft_file=ft_file,
            window=10e3,
            multiplier=5,
        )

        # Assert that the function runs without raising any exceptions
        self.assertTrue(
            os.path.isfile(
                f"results/fits/test_pixels_{pixels[0]},{pixels[1]}_all_fit.png"
            )
        )

    def test_d_fit_with_gaussian_fancy_positive(self):
        # Test with valid input
        pixels = [82, 116]

        work_dir = os.path.dirname(os.path.realpath(__file__)) + "/.."

        path = os.path.join(work_dir, self.partial_path)

        ft_file = r"test.feather"

        # Call the function
        fit_with_gaussian_fancy(
            path,
            pixels=pixels,
            ft_file=ft_file,
            range_left=-5e3,
            range_right=5e3,
            multiplier=5,
        )

        # Assert that the function runs without raising any exceptions
        self.assertTrue(
            os.path.isfile(
                f"results/fits/test_pixels_{pixels[0]},{pixels[1]}_fancy_fit.png"
            )
        )

    def test_d_fit_with_gaussian_pickle_positive(self):
        # Test with valid input
        work_dir = os.path.dirname(os.path.realpath(__file__)) + "/.."

        path = os.path.join(work_dir, self.partial_path)
        pixels = [67, 174]
        window = 20e3
        multiplier = 5

        # Call the function
        fit_with_gaussian(
            path,
            pixels,
            ft_file=None,
            window=window,
            multiplier=multiplier,
            pickle_figure=True,
        )

        # Assert that the function runs without raising any exceptions
        self.assertTrue(
            os.path.isfile(
                "results/fits/test_data_2212b-test_data_2212b_"
                f"pixels_{pixels[0]},{pixels[1]}_fit.pkl"
            )
        )

    def test_d_fit_with_gaussian_all_pickle_positive(self):
        # Test with valid input
        pixels = [82, 116]

        work_dir = os.path.dirname(os.path.realpath(__file__)) + "/.."

        path = os.path.join(work_dir, self.partial_path)

        ft_file = r"test.feather"

        # Call the function
        fit_with_gaussian_all(
            path,
            pixels=pixels,
            ft_file=ft_file,
            window=10e3,
            multiplier=5,
            pickle_figure=True,
        )

        # Assert that the function runs without raising any exceptions
        self.assertTrue(
            os.path.isfile(
                f"results/fits/test_pixels_{pixels[0]},{pixels[1]}_all_fit.pkl"
            )
        )

    def test_d_fit_with_gaussian_fancy_pickle_positive(self):
        # Test with valid input
        pixels = [82, 116]

        work_dir = os.path.dirname(os.path.realpath(__file__)) + "/.."

        path = os.path.join(work_dir, self.partial_path)

        ft_file = r"test.feather"

        # Call the function
        fit_with_gaussian_fancy(
            path,
            pixels=pixels,
            ft_file=ft_file,
            range_left=-5e3,
            range_right=5e3,
            multiplier=5,
            pickle_figure=True,
        )

        # Assert that the function runs without raising any exceptions
        self.assertTrue(
            os.path.isfile(
                f"results/fits/test_pixels_{pixels[0]},{pixels[1]}_fancy_fit.pkl"
            )
        )

    def tearDownClass():
        # Clean up after tests
        os.chdir(r"{}".format(os.path.dirname(os.path.realpath(__file__))))
        shutil.rmtree("test_data/delta_ts_data")
        shutil.rmtree("test_data/results")


if __name__ == "__main__":
    unittest.main()
