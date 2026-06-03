import os
import shutil
import unittest

import numpy as np
from pyarrow import feather as ft

from daplis.functions.delta_t import (
    calculate_and_save_timestamp_differences,
    calculate_and_save_timestamp_differences_1v1,
    collect_and_plot_timestamp_differences,
    unpickle_plot,
)
from daplis.functions.fits import (
    fit_with_gaussian,
    fit_with_gaussian_all,
    fit_with_gaussian_combine,
    fit_with_gaussian_lmfit,
    unpickle_fit,
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
        self.cycle_length = 4e9
        self.apply_mask = True
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
            self.cycle_length,
            self.apply_mask,
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

    def test_c_delta_cp_pickle(self):
        # Test that collect_and_plot produces a pickle file when requested
        work_dir = os.path.dirname(os.path.realpath(__file__)) + "/.."

        path = os.path.join(work_dir, self.partial_path)

        collect_and_plot_timestamp_differences(
            path,
            pixels=[x for x in range(67, 69)] + [x for x in range(173, 175)],
            rewrite=self.rewrite,
            range_left=self.range_left,
            range_right=self.range_right,
            same_y=self.same_y,
            pickle_figure=True,
        )

        self.assertTrue(
            os.path.isfile(
                os.path.join(
                    path,
                    "results/delta_t/test_data_2212b-test_data_2212b_delta_t_grid.pkl",
                )
            )
        )

    def test_d_fit_with_gaussian_positive(self):
        # Test with valid input
        work_dir = os.path.dirname(os.path.realpath(__file__)) + "/.."

        path = os.path.join(work_dir, self.partial_path)
        pixels = [67, 174]
        range_left = -5e3
        range_right = 5e3
        multiplier = 3

        # Call the function
        fit_with_gaussian(
            path,
            pixels,
            ft_file=None,
            range_left=range_left,
            range_right=range_right,
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

    def test_d_fit_with_gaussian_combine_positive(self):
        # Test with valid input
        pixels = [82, 116]

        work_dir = os.path.dirname(os.path.realpath(__file__)) + "/.."

        path = os.path.join(work_dir, self.partial_path)

        ft_file = r"test.feather"

        fit_with_gaussian_combine(
            path,
            pixels=pixels,
            ft_file=ft_file,
            range_left=-10e3,
            range_right=10e3,
            multiplier=5,
        )

        self.assertTrue(
            os.path.isfile("results/fits/test_pixels_82-82,116-116_fit.png")
        )

    def test_d_fit_with_gaussian_combine_pickle_positive(self):
        # Test with valid input
        pixels = [82, 116]

        work_dir = os.path.dirname(os.path.realpath(__file__)) + "/.."

        path = os.path.join(work_dir, self.partial_path)

        ft_file = r"test.feather"

        fit_with_gaussian_combine(
            path,
            pixels=pixels,
            ft_file=ft_file,
            range_left=-10e3,
            range_right=10e3,
            multiplier=5,
            pickle_figure=True,
        )

        self.assertTrue(
            os.path.isfile("results/fits/test_pixels_82-82,116-116_fit.pkl")
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
            range_left=-10e3,
            range_right=10e3,
            multiplier=5,
        )

        # Assert that the function runs without raising any exceptions
        self.assertTrue(
            os.path.isfile(
                f"results/fits/test_pixels_{pixels[0]},{pixels[1]}_all_fit.png"
            )
        )

    def test_d_fit_with_gaussian_lmfit_positive(self):
        # Test with valid input
        pixels = [82, 116]

        work_dir = os.path.dirname(os.path.realpath(__file__)) + "/.."

        path = os.path.join(work_dir, self.partial_path)

        ft_file = r"test.feather"

        # Call the function
        fit_with_gaussian_lmfit(
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
        range_left = -15e3
        range_right = 15e3
        multiplier = 5

        # Call the function
        fit_with_gaussian(
            path,
            pixels,
            ft_file=None,
            range_left=range_left,
            range_right=range_right,
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
            range_left=-10e3,
            range_right=10e3,
            multiplier=5,
            pickle_figure=True,
        )

        # Assert that the function runs without raising any exceptions
        self.assertTrue(
            os.path.isfile(
                f"results/fits/test_pixels_{pixels[0]},{pixels[1]}_all_fit.pkl"
            )
        )

    def test_d_fit_with_gaussian_lmfit_pickle_positive(self):
        # Test with valid input
        pixels = [82, 116]

        work_dir = os.path.dirname(os.path.realpath(__file__)) + "/.."

        path = os.path.join(work_dir, self.partial_path)

        ft_file = r"test.feather"

        # Call the function
        fit_with_gaussian_lmfit(
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

    def test_e_unpickle_delta_t_plot(self):
        # Test that unpickle_plot returns valid figure and data
        work_dir = os.path.dirname(os.path.realpath(__file__)) + "/.."
        path = os.path.join(work_dir, self.partial_path)
        pkl_file = os.path.join(
            path,
            "results/delta_t/test_data_2212b-test_data_2212b_delta_t_grid.pkl",
        )
        result = unpickle_plot(pkl_file)
        self.assertIsNotNone(result)

    def test_e_unpickle_fit(self):
        # Test that unpickle_fit returns valid figure, data and params
        work_dir = os.path.dirname(os.path.realpath(__file__)) + "/.."
        path = os.path.join(work_dir, self.partial_path)
        pkl_file = os.path.join(
            path,
            "results/fits/test_data_2212b-test_data_2212b_pixels_67,174_fit.pkl",
        )
        fig, plot_data, params_df = unpickle_fit(pkl_file)
        self.assertIsNotNone(fig)
        self.assertIsInstance(plot_data, dict)

    def tearDownClass():
        # Clean up after tests
        os.chdir(r"{}".format(os.path.dirname(os.path.realpath(__file__))))
        shutil.rmtree("test_data/delta_ts_data")
        shutil.rmtree("test_data/results")


class TestDeltasOneVOne(unittest.TestCase):
    def setUp(self):
        self.partial_path = "tests/test_data"
        self.pixels = [
            [66, 67],
            [170, 171],
        ]
        self.daughterboard_number = "NL11"
        self.motherboard_number = "#33"
        self.firmware_version = "2212b"
        self.timestamps = 300
        self.delta_window = 20e3
        self.rewrite = True
        self.cycle_length = 4e9
        self.include_offset = False

    def _path(self):
        work_dir = os.path.dirname(os.path.realpath(__file__)) + "/.."
        return os.path.join(work_dir, self.partial_path)

    def test_a_1v1_creates_feather(self):
        calculate_and_save_timestamp_differences_1v1(
            self._path(),
            self.pixels,
            self.rewrite,
            self.daughterboard_number,
            self.motherboard_number,
            self.firmware_version,
            self.timestamps,
            self.delta_window,
            self.cycle_length,
            include_offset=self.include_offset,
        )
        feather_file = os.path.join(
            self._path(),
            "delta_ts_data",
            "test_data_2212b-test_data_2212b.feather",
        )
        self.assertTrue(os.path.isfile(feather_file))

    def test_b_1v1_feather_columns_match_pairs(self):
        # Columns must be exactly the requested diagonal pairs, not a cross product
        calculate_and_save_timestamp_differences_1v1(
            self._path(),
            self.pixels,
            self.rewrite,
            self.daughterboard_number,
            self.motherboard_number,
            self.firmware_version,
            self.timestamps,
            self.delta_window,
            self.cycle_length,
            include_offset=self.include_offset,
        )
        feather_file = os.path.join(
            self._path(),
            "delta_ts_data",
            "test_data_2212b-test_data_2212b.feather",
        )
        data = ft.read_feather(feather_file)
        self.assertEqual(set(data.columns), {"66,170", "67,171"})

    def test_c_1v1_wrong_rewrite_type_raises(self):
        with self.assertRaises(TypeError):
            calculate_and_save_timestamp_differences_1v1(
                self._path(),
                self.pixels,
                "not_a_bool",
                self.daughterboard_number,
                self.motherboard_number,
                self.firmware_version,
                self.timestamps,
                self.delta_window,
            )

    def test_c_1v1_wrong_pixels_type_raises(self):
        with self.assertRaises(TypeError):
            calculate_and_save_timestamp_differences_1v1(
                self._path(),
                "not_a_list",
                self.rewrite,
                self.daughterboard_number,
                self.motherboard_number,
                self.firmware_version,
                self.timestamps,
                self.delta_window,
            )

    def test_d_1v1_unequal_pixel_lists_raise(self):
        # This is the exact bug that went undetected: unequal-length lists
        mismatched = [[66, 67, 68], [170, 171]]
        with self.assertRaises(ValueError):
            calculate_and_save_timestamp_differences_1v1(
                self._path(),
                mismatched,
                self.rewrite,
                self.daughterboard_number,
                self.motherboard_number,
                self.firmware_version,
                self.timestamps,
                self.delta_window,
                self.cycle_length,
                include_offset=self.include_offset,
            )

    @classmethod
    def tearDownClass(cls):
        os.chdir(os.path.dirname(os.path.realpath(__file__)))
        if os.path.isdir("test_data/delta_ts_data"):
            shutil.rmtree("test_data/delta_ts_data")


if __name__ == "__main__":
    unittest.main()
