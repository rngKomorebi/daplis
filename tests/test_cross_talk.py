import os
import shutil
import unittest

import numpy as np

from daplis.functions.cross_talk import (
    collect_dcr_by_file,
    plot_dcr_histogram_and_stability,
    zero_to_cross_talk_collect,
    zero_to_cross_talk_plot,
)


class TestCrossTalkFunctions(unittest.TestCase):
    def setUp(self):
        # Set up test variables
        self.partial_path = "tests/test_data"
        self.pixels = (70,)
        self.daughterboard_number = "NL11"
        self.motherboard_number = "#33"
        self.firmware_version = "2212s"
        self.timestamps = 300
        self.delta_window = 20e3
        self.rewrite = True
        self.range_left = -20e3
        self.range_right = 20e3
        self.same_y = False
        self.app_mask = True
        self.include_offset = False

    def test_collect_dcr_by_file_positive(self):
        # Test positive case for deltas_save function
        work_dir = os.path.dirname(os.path.realpath(__file__)) + "/.."

        path = os.path.join(work_dir, self.partial_path)
        collect_dcr_by_file(
            path,
            self.daughterboard_number,
            self.motherboard_number,
            self.firmware_version,
            self.timestamps,
        )

        # Check if the csv file is created
        self.assertTrue(
            os.path.isfile(
                os.path.join(
                    path,
                    "dcr_data/test_data_2212b-test_data_2212b_dcr_data.pkl",
                )
            )
        )

    def test_plot_dcr_histogram_and_stability(self):

        # Test positive case for deltas_save function
        work_dir = os.path.dirname(os.path.realpath(__file__)) + "/.."

        path = os.path.join(work_dir, self.partial_path)

        plot_dcr_histogram_and_stability(
            path,
        )

        self.assertTrue(
            os.path.isfile(
                os.path.join(path, "results/dcr/DCR_stability_graph.png")
            )
            and os.path.isfile(
                os.path.join(path, "results/dcr/DCR_histogram_w_integral.png")
            )
        )

    def test_zero_to_cross_talk_collect(self):

        # Test positive case for deltas_save function
        work_dir = os.path.dirname(os.path.realpath(__file__)) + "/.."

        path = os.path.join(work_dir, self.partial_path)

        zero_to_cross_talk_collect(
            path,
            self.pixels,
            self.rewrite,
            self.daughterboard_number,
            self.motherboard_number,
            self.firmware_version,
            self.timestamps,
        )

        # Check if the csv file is created
        self.assertTrue(
            os.path.isfile(
                os.path.join(
                    path,
                    "cross_talk_data/test_data_2212b-test_"
                    "data_2212b_pixels_70-50.feather",
                )
            )
            and os.path.isfile(
                os.path.join(
                    path,
                    "cross_talk_data/test_data_2212b-test_"
                    "data_2212b_pixels_70-90.feather",
                )
            )
        )

    def test_zero_to_cross_talk_plot(self):

        # Test positive case for deltas_save function
        work_dir = os.path.dirname(os.path.realpath(__file__)) + "/.."

        path = os.path.join(work_dir, self.partial_path)

        zero_to_cross_talk_plot(
            path,
            self.pixels,
        )

        # Check if the csv file is created
        self.assertTrue(
            os.path.isfile(
                os.path.join(
                    path,
                    "ct_vs_distance/Average_cross-talk.png",
                )
            )
        )

    def tearDownClass():
        # Clean up after tests
        os.chdir(r"{}".format(os.path.dirname(os.path.realpath(__file__))))
        shutil.rmtree("test_data/dcr_data")
        shutil.rmtree("test_data/results")
        shutil.rmtree("test_data/cross_talk_data")
        shutil.rmtree("test_data/ct_vs_distance")
        shutil.rmtree("test_data/senpop_data")
