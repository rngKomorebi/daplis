"""Module for calculating and plotting the timestamp differences.

This script utilizes an unpacking module used specifically for the
LinoSPAD2 binary data output.

This file can also be imported as a module and contains the following
functions:

    * _flatten - flatten the input list into a single list of numbers.
    The input list could contain any number of lists of numbers.

    * _combine_intermediate_feather_files - combine '.feather' files
    into one. Used for combining the intermediate steps of saving
    the timestamp differences into a single file.

    * calculate_and_save_timestamp_differences - unpacks the binary
    data, calculates timestamp differences, and saves into a '.feather'
    file.

    * calculate_and_save_timestamp_differences_1v1 - unpacks the
    binary data, calculates timestamp differences, and saves into a
    '.feather' file. Uses a faster algorithm than the standard version.
    Calculated only for the diagonal pixels, i.e., 1-1, 2-2, etc., and
    not for all pairs.

    * calculate_and_save_timestamp_differences_full_sensor - unpacks the
    binary data, calculates timestamp differences and saves into a
    '.feather' file. Works with firmware versions '2208', '2212s' and
    '2212b'. Analyzes data from both sensor halves/both FPGAs. Useful
    for data where the signal is at 0 across the whole sensor from the
    start of data collecting.

    * calculate_and_save_timestamp_differences_full_sensor_alt - unpacks
    the binary data, finds the constant board-to-board offset
    automatically from the absolute timestamps, calculates timestamp
    differences and saves into a '.feather' file. Works with firmware
    versions '2212s' and '2212b'. Analyzes data from both sensor
    halves/both FPGAs that share a common external clock but no common
    trigger. Useful for data where signal is above zero right from the
    start.

    * collect_and_plot_timestamp_differences - collect timestamps from a
    '.feather' file and plot them in a grid.

    * collect_and_plot_timestamp_differences_full_sensor - collect
    timestamps from a '.feather' file and plot histograms of them
    in a grid. This function should be used for the full sensor
    setup.

    * unpickle_plot - unpickle the '.pkl' file with a delta_t histogram.
    Can be used to presaved '.pkl' files for fine control over the plot.
"""

from __future__ import annotations

import glob
import os
import pickle
import sys
from math import ceil

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from pyarrow import feather as ft
from tqdm import tqdm

from daplis.functions import calc_diff as cd
from daplis.functions import calibrate as cb
from daplis.functions import unpack as f_up
from daplis.functions import utils


def _flatten(input_list: list):
    """Flatten the input list.

    Flatten the input list, which can be a list of numbers, lists,
    or a combination of the two above, and return a list of
    numbers only, unpacking the lists inside.

    Parameters
    ----------
    input_list : List
        Input list that can contain numbers, lists of numbers, or a
        combination of both.

    Returns
    -------
    list
        Flattened list of numbers only.
    """
    flattened = []
    for item in input_list:
        if isinstance(item, list):
            flattened.extend(item)
        else:
            flattened.append(item)
    return flattened


def _combine_intermediate_feather_files(path: str, skip_data: bool = False):
    """Combine intermediate '.feather' files into one.

    Find all numbered '.feather' files for the data files found in the
    path and combine them all into one.

    Parameters
    ----------
    path : str
        Path to the folder with the '.dat' data files.
    skip_data : bool
        Switch for skipping the data and working directly in the
        'delta_ts_data' folder. Can be used when the raw '.dat' files
        are not available or the data set is incomplete. The default is
        False.

    Raises
    ------
    FileNotFoundError
        Raised when the folder "delta_ts_data", where timestamp
        differences are saved, cannot be found in the path.
    """
    os.chdir(path)

    if not skip_data:
        files_all = sorted(glob.glob("*.dat*"))
        if files_all != []:
            feather_file_name = files_all[0][:-4] + "-" + files_all[-1][:-4]
            combined_feather_file_name = feather_file_name
        else:
            feather_file_name = ""
            combined_feather_file_name = "combined"
    else:
        feather_file_name = ""
        combined_feather_file_name = "combined"

    try:
        os.chdir("delta_ts_data")
    except FileNotFoundError as exc:
        raise FileNotFoundError(
            "Folder with saved timestamp differences was not found"
        ) from exc

    file_pattern = f"{feather_file_name}*_*.feather"

    feather_files = glob.glob(file_pattern)

    data_combined = []
    data_combined = pd.DataFrame(data_combined)

    for ft_file in feather_files:
        data = ft.read_feather(ft_file)

        data_combined = pd.concat([data_combined, data], ignore_index=True)

        data_combined.to_feather(f"{combined_feather_file_name}.feather")

    for ft_file in feather_files:
        os.remove(ft_file)


def calculate_and_save_timestamp_differences(
    path: str,
    pixels: list[int] | list[list[int]],
    rewrite: bool,
    daughterboard_number: str,
    motherboard_number: str,
    firmware_version: str,
    timestamps: int = 512,
    delta_window: float = 50e3,
    cycle_length: float = None,
    apply_mask: bool = True,
    include_offset: bool = False,
    apply_calibration: bool = True,
    absolute_timestamps: bool = False,
    correct_pix_address: bool = False,
):
    """Calculate and save timestamp differences into a '.feather' file.

    Unpacks data, computes timestamp differences for the requested pixels,
    and saves them into a '.feather' table. Compatible with firmware
    versions 2212 and uses a faster algorithm.

    Parameters
    ----------
    path : str
        Path to the folder containing '.dat' data files.
    pixels : List[int] | List[List[int]]
        List of pixel numbers for which timestamp differences should
        be computed, or a list of two lists for peak-vs-peak calculations.
    rewrite : bool
        Switch for rewriting the '.feather' file if it already exists.
        Used as a safeguard to avoid unwanted overwriting.
    daughterboard_number : str
        LinoSPAD2 daughterboard number.
    motherboard_number : str
        LinoSPAD2 motherboard (FPGA) number, including the '#'.
    firmware_version : str
        LinoSPAD2 firmware version. Versions '2212s' (skip) and '2212b'
        (block) are recognized.
    timestamps : int, optional
        Number of timestamps per acquisition cycle per pixel.
        The default is 512.
    delta_window : float, optional
        Width of the window in which timestamp differences are counted.
        The default is 50e3 (50 ns).
    cycle_length : float, optional
        Length of one acquisition cycle. The default is None.
    apply_mask : bool, optional
        Apply mask for hot pixels. The default is True.
    include_offset : bool, optional
        Apply offset calibration. The default is False.
    apply_calibration : bool, optional
        Apply TDC and offset calibration. If True while
        ``include_offset`` is False, only TDC calibration is applied.
        The default is True.
    absolute_timestamps : bool, optional
        Indicates whether the data contains absolute timestamps.
        The default is False.
    correct_pix_address : bool, optional
        Correct pixel addresses for the sensor half on side 23 of the
        daughterboard. The default is False.

    Raises
    ------
    TypeError
        If ``pixels`` is not a list.
    TypeError
        If ``firmware_version`` is not a string.
    TypeError
        If ``rewrite`` is not a boolean.
    TypeError
        If ``daughterboard_number`` is not a string.
    """
    # Parameter type check
    if isinstance(pixels, list) is False:
        raise TypeError(
            "'pixels' should be a list of integers or a list of two lists"
        )
    if isinstance(firmware_version, str) is False:
        raise TypeError(
            "'firmware_version' should be string, '2212s', '2212b' or '2208'"
        )
    if isinstance(rewrite, bool) is False:
        raise TypeError("'rewrite' should be boolean")
    if isinstance(daughterboard_number, str) is False:
        raise TypeError("'daughterboard_number' should be string")

    os.chdir(path)

    # Handle the input list
    pixels = utils.pixel_list_transform(pixels)
    files_all = glob.glob("*.dat")

    files_all = sorted(files_all)

    out_file_name = files_all[0][:-4] + "-" + files_all[-1][:-4]

    # Feather file counter for saving delta ts into separate files
    # of up to 100 MB
    ft_file_number = 0

    # Check if the feather file exists and if it should be rewrited
    feather_file = os.path.join(
        path, "delta_ts_data", f"{out_file_name}.feather"
    )

    # Remove the old '.feather' files with the pattern
    # for ft_file in feather_files:
    utils.file_rewrite_handling(feather_file, rewrite)

    # Go back to the folder with '.dat' files
    os.chdir(path)

    # Collect the data for the required pixels
    print(
        "\n> > > Collecting data for delta t plot for the requested "
        "pixels and saving it to .feather in a cycle < < <\n"
    )
    # Define matrix of pixel coordinates, where rows are numbers of TDCs
    # and columns are the pixels that connected to these TDCs

    if firmware_version == "2212s":
        pixel_coordinates = np.arange(256).reshape(4, 64).T
    elif firmware_version == "2212b":
        pixel_coordinates = np.arange(256).reshape(64, 4)
    else:
        print("\nFirmware version is not recognized.")
        sys.exit()

    # Correct pixel addressing for motherboard on side '23'
    if correct_pix_address:
        pixels = utils.correct_pixels_address(pixels)

    # Mask the hot/warm pixels
    if apply_mask is True:
        mask = utils.apply_mask(daughterboard_number, motherboard_number)
        if isinstance(pixels[0], int) and isinstance(pixels[1], int):
            pixels = [pix for pix in pixels if pix not in mask]
        else:
            pixels[0] = [pix for pix in pixels[0] if pix not in mask]
            pixels[1] = [pix for pix in pixels[1] if pix not in mask]

    for i in tqdm(range(ceil(len(files_all))), desc="Collecting data"):
        file = files_all[i]

        # Unpack data for the requested pixels into dictionary
        if not absolute_timestamps:
            data_pixels, data_timestamps = f_up.unpack_binary_data(
                file,
                daughterboard_number,
                motherboard_number,
                firmware_version,
                timestamps,
            )
        else:
            data_pixels, data_timestamps, _ = (
                f_up.unpack_binary_data_with_absolute_timestamps(
                    file,
                    daughterboard_number,
                    motherboard_number,
                    firmware_version,
                    timestamps,
                )
            )

        # If cycle_length is not given manually, estimate from the data
        if cycle_length is None:
            cycle_length = np.max(data_timestamps * 2500 / 140).astype(int)

        # Offset timestamps by cycles (e.g. +4 ms to each next cycle)
        number_of_cycles = len(data_timestamps.flatten()) / 64 / timestamps
        offsets = np.repeat(
            np.arange(number_of_cycles, dtype=np.int64) * int(cycle_length),
            timestamps,
        )

        data_selected_pixels = {}

        if apply_calibration is False:

            for i in [x for sublist in pixels for x in sublist]:

                tdc, pix = np.argwhere(pixel_coordinates == i)[0]
                mask = (data_pixels[tdc] == pix) & (data_timestamps[tdc] >= 0)
                ind = np.nonzero(mask)[0]
                data_cut = data_timestamps[tdc][ind]
                data_cut = data_cut * 2500 / 140
                data_cut += offsets[mask]

                data_selected_pixels[f"{i}"] = data_cut
        else:
            # Path to the calibration data
            path_calibration_data = os.path.join(
                os.path.dirname(os.path.realpath(__file__)),
                "..",
                "params",
                "calibration_data",
            )

            # Include the offset calibration or not
            try:
                if include_offset:
                    calibration_matrix, offset_array = (
                        cb.load_calibration_data(
                            path_calibration_data,
                            daughterboard_number,
                            motherboard_number,
                            firmware_version,
                            include_offset,
                        )
                    )
                else:
                    calibration_matrix = cb.load_calibration_data(
                        path_calibration_data,
                        daughterboard_number,
                        motherboard_number,
                        firmware_version,
                        include_offset,
                    )
            except FileNotFoundError as exc:
                raise FileNotFoundError(
                    "No .csv file with the calibration data was found. "
                    "Check the path or run the calibration."
                ) from exc

            for i in [x for sublist in pixels for x in sublist]:
                # Transform pixel number to TDC number and pixel coordinates in
                # that TDC (from 0 to 3)
                tdc, pix = np.argwhere(pixel_coordinates == i)[0]
                mask = (data_pixels[tdc] == pix) & (data_timestamps[tdc] >= 0)
                ind = np.nonzero(mask)[0]
                data_cut = data_timestamps[tdc][ind]

                if include_offset:
                    data_cut = (
                        (data_cut - data_cut % 140) * 2500 / 140
                        + calibration_matrix[i, (data_cut % 140)]
                        + offset_array[i]
                    )
                else:
                    data_cut = (
                        data_cut - data_cut % 140
                    ) * 2500 / 140 + calibration_matrix[i, (data_cut % 140)]

                data_cut += offsets[mask]

                data_selected_pixels[f"{i}"] = data_cut

        delta_ts = cd.calculate_differences(
            data_selected_pixels, pixels, delta_window, cycle_length
        )

        # Save data as a .feather file in a cycle so data is not lost
        # in the case of failure close to the end
        delta_ts = pd.DataFrame.from_dict(delta_ts, orient="index")
        delta_ts = delta_ts.T

        try:
            os.chdir("delta_ts_data")
        except FileNotFoundError:
            os.mkdir("delta_ts_data")
            os.chdir("delta_ts_data")

        # Check if feather file exists
        feather_file = f"{out_file_name}_{ft_file_number}.feather"
        if os.path.isfile(feather_file):
            # Check the size of the existing '.feather', if larger
            # than 100 MB, create new one
            if os.path.getsize(feather_file) / 1024 / 1024 < 100:
                # Load existing feather file
                existing_data = ft.read_feather(feather_file)

                # Append new data to the existing feather file
                combined_data = pd.concat([existing_data, delta_ts], axis=0)
                ft.write_feather(combined_data, feather_file)
            else:
                ft_file_number += 1
                feather_file = f"{out_file_name}_{ft_file_number}.feather"
                ft.write_feather(delta_ts, feather_file)

        else:
            # Save as a new feather file
            ft.write_feather(delta_ts, feather_file)
        os.chdir("..")

    # Combine the numbered feather files into a single one
    _combine_intermediate_feather_files(path)

    # Check, if the file was created
    if (
        os.path.isfile(path + f"/delta_ts_data/{out_file_name}.feather")
        is True
    ):
        print(
            "\n> > > Timestamp differences are saved as"
            f"{out_file_name}.feather in "
            f"{os.path.join(path, 'delta_ts_data')} < < <"
        )

    else:
        print("File wasn't generated. Check input parameters.")


def calculate_and_save_timestamp_differences_1v1(
    path: str,
    pixels: list[int] | list[list[int]],
    rewrite: bool,
    daughterboard_number: str,
    motherboard_number: str,
    firmware_version: str,
    timestamps: int = 512,
    delta_window: float = 50e3,
    cycle_length: float = None,
    apply_mask: bool = True,
    include_offset: bool = False,
    apply_calibration: bool = True,
    absolute_timestamps: bool = False,
    correct_pix_address: bool = False,
):
    """Calculate and save timestamp differences into '.feather' file.

    Unpacks data into a dictionary, calculates timestamp differences for
    the requested pixels, and saves them into a '.feather' table. Works with
    firmware version 2212. Uses a faster algorithm. Calculated only for
    the diagonal pixels, i.e., 1-1, 2-2, etc., and not for all pairs.

    Parameters
    ----------
    path : str
        Path to the folder with '.dat' data files.
    pixels : List[int] | List[List[int]]
        List of pixel numbers for which the timestamp differences should
        be calculated and saved or list of two lists with pixel numbers
        for peak vs. peak calculations.
    rewrite : bool
        switch for rewriting the plot if it already exists. used as a
        safeguard to avoid unwanted overwriting of the previous results.
        Switch for rewriting the '.feather' file if it already exists.
    daughterboard_number : str
        LinoSPAD2 daughterboard number.
    motherboard_number : str
        LinoSPAD2 motherboard (FPGA) number, including the '#'.
    firmware_version: str
        LinoSPAD2 firmware version. Versions "2212s" (skip) and "2212b"
        (block) are recognized.
    timestamps : int, optional
        Number of timestamps per acquisition cycle per pixel. The default
        is 512.
    delta_window : float, optional
        Size of a window to which timestamp differences are compared.
        Differences in that window are saved. The default is 50e3 (50 ns).
    cycle_length: float, optional
        Length of the acquisition cycle. The default is None.
    apply_mask : bool, optional
        Switch for applying the mask for hot pixels. The default is True.
    include_offset : bool, optional
        Switch for applying offset calibration. The default is True.
    apply_calibration : bool, optional
        Switch for applying TDC and offset calibration. If set to 'True'
        while apply_offset_calibration is set to 'False', only the TDC
        calibration is applied. The default is True.
    absolute_timestamps: bool, optional
        Indicator for data with absolute timestamps. The default is
        False.
    correct_pix_address : bool, optional
        Correct pixel address for the sensor half on side 23 of the
        daughterboard. The default is False.

    Raises
    ------
    TypeError
        Raised if "pixels" is not a list.
    TypeError
        Raised if "firmware_version" is not a string.
    TypeError
        Raised if "rewrite" is not a boolean.
    TypeError
        Raised if "daughterboard_number" is not a string.
    """
    # Parameter type check
    if isinstance(pixels, list) is False:
        raise TypeError(
            "'pixels' should be a list of integers or a list of two lists"
        )
    if isinstance(firmware_version, str) is False:
        raise TypeError(
            "'firmware_version' should be string, '2212s', '2212b' or '2208'"
        )
    if isinstance(rewrite, bool) is False:
        raise TypeError("'rewrite' should be boolean")
    if isinstance(daughterboard_number, str) is False:
        raise TypeError("'daughterboard_number' should be string")

    os.chdir(path)

    # Handle the input list
    # A flat list of pixel numbers means "all combinations" and cannot
    # be split into explicit 1-to-1 pairs
    if all(
        isinstance(pixel, (int, np.integer)) and not isinstance(pixel, bool)
        for pixel in pixels
    ):
        raise TypeError(
            "1v1 requires two lists of equal length, "
            "e.g. [[q0, q1], [w0, w1]]"
        )

    pixels = utils.pixel_list_transform(pixels)
    files_all = glob.glob("*.dat")

    files_all = sorted(files_all)

    out_file_name = files_all[0][:-4] + "-" + files_all[-1][:-4]

    # Feather file counter for saving delta ts into separate files
    # of up to 100 MB
    ft_file_number = 0

    # Check if the feather file exists and if it should be rewrited
    feather_file = os.path.join(
        path, "delta_ts_data", f"{out_file_name}.feather"
    )

    # Remove the old '.feather' files with the pattern
    # for ft_file in feather_files:
    utils.file_rewrite_handling(feather_file, rewrite)

    # Go back to the folder with '.dat' files
    os.chdir(path)

    # Collect the data for the required pixels
    print(
        "\n> > > Collecting data for delta t plot for the requested "
        "pixels and saving it to .feather in a cycle < < <\n"
    )
    # Define matrix of pixel coordinates, where rows are numbers of TDCs
    # and columns are the pixels that connected to these TDCs
    if firmware_version == "2212s":
        pixel_coordinates = np.arange(256).reshape(4, 64).T
    elif firmware_version == "2212b":
        pixel_coordinates = np.arange(256).reshape(64, 4)
    else:
        print("\nFirmware version is not recognized.")
        sys.exit()

    # Correct pixel addressing for motherboard on side '23'
    if correct_pix_address:
        pixels = utils.correct_pixels_address(pixels)

    for i in tqdm(range(ceil(len(files_all))), desc="Collecting data"):
        file = files_all[i]

        # Unpack data for the requested pixels into dictionary
        if not absolute_timestamps:
            data_pixels, data_timestamps = f_up.unpack_binary_data(
                file,
                daughterboard_number,
                motherboard_number,
                firmware_version,
                timestamps,
            )
        else:
            data_pixels, data_timestamps, _ = (
                f_up.unpack_binary_data_with_absolute_timestamps(
                    file,
                    daughterboard_number,
                    motherboard_number,
                    firmware_version,
                    timestamps,
                )
            )

        # If cycle_length is not given manually, estimate from the data
        if cycle_length is None:
            cycle_length = np.max(data_timestamps * 2500 / 140).astype(int)

        # Offset timestamps by cycles (e.g. +4 ms to each next cycle)
        number_of_cycles = len(data_timestamps.flatten()) / 64 / timestamps
        offsets = np.repeat(
            np.arange(number_of_cycles, dtype=np.int64) * int(cycle_length),
            timestamps,
        )

        data_selected_pixels = {}

        if apply_calibration is False:

            for pix in [x for sublist in pixels for x in sublist]:

                tdc, pix_c = np.argwhere(pixel_coordinates == pix)[0]
                pix_mask = (data_pixels[tdc] == pix_c) & (
                    data_timestamps[tdc] >= 0
                )
                ind = np.nonzero(pix_mask)[0]
                data_cut = data_timestamps[tdc][ind]
                data_cut = data_cut * 2500 / 140
                data_cut += offsets[pix_mask]

                data_selected_pixels[f"{pix}"] = data_cut
        else:
            # Path to the calibration data
            path_calibration_data = os.path.join(
                os.path.dirname(os.path.realpath(__file__)),
                "..",
                "params",
                "calibration_data",
            )

            try:
                if include_offset:
                    calibration_matrix, offset_array = (
                        cb.load_calibration_data(
                            path_calibration_data,
                            daughterboard_number,
                            motherboard_number,
                            firmware_version,
                            include_offset,
                        )
                    )
                else:
                    calibration_matrix = cb.load_calibration_data(
                        path_calibration_data,
                        daughterboard_number,
                        motherboard_number,
                        firmware_version,
                        include_offset,
                    )
            except FileNotFoundError as exc:
                raise FileNotFoundError(
                    "No .csv file with the calibration data was found. "
                    "Check the path or run the calibration."
                ) from exc

            for pix in [x for sublist in pixels for x in sublist]:
                tdc, pix_c = np.argwhere(pixel_coordinates == pix)[0]
                pix_mask = (data_pixels[tdc] == pix_c) & (
                    data_timestamps[tdc] >= 0
                )
                ind = np.nonzero(pix_mask)[0]
                data_cut = data_timestamps[tdc][ind]

                if include_offset:
                    data_cut = (
                        (data_cut - data_cut % 140) * 2500 / 140
                        + calibration_matrix[pix, (data_cut % 140)]
                        + offset_array[pix]
                    )
                else:
                    data_cut = (
                        data_cut - data_cut % 140
                    ) * 2500 / 140 + calibration_matrix[pix, (data_cut % 140)]

                data_cut += offsets[pix_mask]

                data_selected_pixels[f"{pix}"] = data_cut

        delta_ts = cd.calculate_differences_1v1(
            data_selected_pixels, pixels, delta_window, cycle_length
        )

        # Save data as a .feather file in a cycle so data is not lost
        # in the case of failure close to the end
        delta_ts = pd.DataFrame.from_dict(delta_ts, orient="index")
        delta_ts = delta_ts.T

        try:
            os.chdir("delta_ts_data")
        except FileNotFoundError:
            os.mkdir("delta_ts_data")
            os.chdir("delta_ts_data")

        # Check if feather file exists
        feather_file = f"{out_file_name}_{ft_file_number}.feather"
        if os.path.isfile(feather_file):
            # Check the size of the existing '.feather', if larger
            # than 100 MB, create new one
            if os.path.getsize(feather_file) / 1024 / 1024 < 100:
                # Load existing feather file
                existing_data = ft.read_feather(feather_file)

                # Append new data to the existing feather file
                combined_data = pd.concat([existing_data, delta_ts], axis=0)
                ft.write_feather(combined_data, feather_file)
            else:
                ft_file_number += 1
                feather_file = f"{out_file_name}_{ft_file_number}.feather"
                ft.write_feather(delta_ts, feather_file)

        else:
            # Save as a new feather file
            ft.write_feather(delta_ts, feather_file)
        os.chdir("..")

    # Combine the numbered feather files into a single one
    _combine_intermediate_feather_files(path)

    # Check, if the file was created

    if (
        os.path.isfile(path + f"/delta_ts_data/{out_file_name}.feather")
        is True
    ):
        print(
            "\n> > > Timestamp differences are saved as"
            f"{out_file_name}.feather in "
            f"{os.path.join(path, 'delta_ts_data')} < < <"
        )

    else:
        print("File wasn't generated. Check input parameters.")


def calculate_and_save_timestamp_differences_full_sensor(
    path,
    pixels: list,
    rewrite: bool,
    daughterboard_number: str,
    motherboard_number1: str,
    motherboard_number2: str,
    firmware_version: str,
    timestamps: int = 512,
    delta_window: float = 50e3,
    apply_mask: bool = True,
    include_offset: bool = False,
    apply_calibration: bool = True,
    absolute_timestamps: bool = False,
):  # TODO add option for collecting from more than just two pixels
    # TODO use pixel handling function for modularity (if possible)
    """Calculate and save timestamp differences into '.feather' file.

    Unpacks data into a dictionary, calculates timestamp differences for
    the requested pixels and saves them into a '.feather' table. Works with
    firmware version 2212. Analyzes data from both sensor halves/both
    FPGAs, hence the two input parameters for LinoSPAD2 motherboards.

    Parameters
    ----------
    path : str
        Path to where two folders with data from both motherboards
        are. The folders should be named after the motherboards.
    pixels : list
        List of two pixels, one from each sensor half.
    rewrite : bool
        switch for rewriting the plot if it already exists. used as a
        safeguard to avoid unwanted overwriting of the previous results.
        Switch for rewriting the '.feather' file if it already exists.
    daughterboard_number : str
        LinoSPAD2 daughterboard number.
    motherboard_number1 : str
        LinoSPAD2 motherboard (FPGA) number, including the '#'.
    motherboard_number2 : str
        Second LinoSPAD2 motherboard (FPGA) number.
    firmware_version: str
        LinoSPAD2 firmware version. Versions "2212s" (skip) and "2212b"
        (block) are recognized.
    timestamps : int, optional
        Number of timestamps per acquisition cycle per pixel. The default
        is 512.
    delta_window : float, optional
        Size of a window to which timestamp differences are compared.
        Differences in that window are saved. The default is 50e3 (50 ns).
    apply_mask : bool, optional
        Switch for applying the mask for hot pixels. The default is True.
    include_offset : bool, optional
        Switch for applying offset calibration. The default is True.
    apply_calibration : bool, optional
        Switch for applying TDC and offset calibration. If set to 'True'
        while include_offset is set to 'False', only the TDC calibration is
        applied. The default is True.
    absolute_timestamps : bool, optional
        Switch for unpacking data that were collected together with
        absolute timestamps. The default is False.

    Raises
    ------
    TypeError
        Only boolean values of 'rewrite' and string values of
        'daughterboard_number', 'motherboard_number', and 'firmware_version'
        are accepted. The first error is raised so that the plot does not
        accidentally get rewritten in the case no clear input was given.
    FileNotFoundError
        Raised if data from the first LinoSPAD2 motherboard were not
        found.
    FileNotFoundError
        Raised if data from the second LinoSPAD2 motherboard were not
        found.
    """
    # parameter type check
    if isinstance(pixels, list) is False:
        raise TypeError(
            "'pixels' should be a list of integers or a list of two lists"
        )
    if isinstance(firmware_version, str) is False:
        raise TypeError(
            "'firmware_version' should be string, '2212s', '2212b' or" "'2208'"
        )
    if isinstance(rewrite, bool) is False:
        raise TypeError("'rewrite' should be boolean")
    if isinstance(daughterboard_number, str) is False:
        raise TypeError("'daughterboard_number' should be string")

    os.chdir(path)

    # Check the data from the first FPGA board
    try:
        os.chdir(f"{motherboard_number1}")
    except FileNotFoundError as exc:
        raise FileNotFoundError(
            f"Data from {motherboard_number1} not found"
        ) from exc
    # files_all1 = sorted(glob.glob("*.dat*"))
    files_all1 = glob.glob("*.dat*")
    files_all1.sort(key=os.path.getmtime)
    out_file_name = files_all1[0][:-4]
    os.chdir("..")

    # Check the data from the second FPGA board
    try:
        os.chdir(f"{motherboard_number2}")
    except FileNotFoundError as exc:
        raise FileNotFoundError(
            f"Data from {motherboard_number2} not found"
        ) from exc
    # files_all2 = sorted(glob.glob("*.dat*"))
    files_all2 = glob.glob("*.dat*")
    files_all2.sort(key=os.path.getmtime)
    out_file_name = out_file_name + "-" + files_all2[-1][:-4]
    os.chdir("..")

    # Define matrix of pixel coordinates, where rows are numbers of TDCs
    # and columns are the pixels that connected to these TDCs
    if firmware_version == "2212s":
        pixel_coordinates = np.arange(256).reshape(4, 64).T
    elif firmware_version == "2212b":
        pixel_coordinates = np.arange(256).reshape(64, 4)
    else:
        print("\nFirmware version is not recognized.")
        sys.exit()

    # Check if '.feather' file with timestamps differences already
    # exists

    feather_file = os.path.join(
        path, "delta_ts_data", f"{out_file_name}.feather"
    )
    feather_file = os.path.join(
        path, "delta_ts_data", f"{out_file_name}.feather"
    )

    utils.file_rewrite_handling(feather_file, rewrite)

    # TODO add check for masked/noisy pixels
    # if apply_mask is True:
    #     path_to_back = os.getcwd()
    #     path_to_mask = os.path.realpath(__file__) + "/../.." + "/params/masks"
    #     os.chdir(path_to_mask)
    #     file_mask1 = glob.glob("*{}_{}*".format(daughterboard_number, motherboard_number1))[0]
    #     mask1 = np.genfromtxt(file_mask1).astype(int)
    #     file_mask2 = glob.glob("*{}_{}*".format(daughterboard_number, motherboard_number2))[0]
    #     mask2 = np.genfromtxt(file_mask2).astype(int)
    #     os.chdir(path_to_back)

    abs_tmsp_list1 = []
    abs_tmsp_list2 = []
    for i in tqdm(range(ceil(len(files_all1))), desc="Collecting data"):
        deltas_all = {}

        # First board, unpack data
        os.chdir(f"{motherboard_number1}")
        file = files_all1[i]
        if not absolute_timestamps:
            data_all1 = f_up.unpack_binary_data(
                file,
                daughterboard_number,
                motherboard_number1,
                firmware_version,
                timestamps,
                include_offset,
                apply_calibration,
            )
        else:
            (
                data_all1,
                abs_tmsp1,
            ) = f_up.unpack_binary_data_with_absolute_timestamps(
                file,
                daughterboard_number,
                motherboard_number1,
                firmware_version,
                timestamps,
                include_offset,
                apply_calibration,
            )
        # abs_tmsp_list1.append(abs_tmsp1)
        # Collect indices of cycle ends (the '-2's)
        cycle_ends1 = np.where(data_all1[0].T[1] == -2)[0]
        cyc1 = np.argmin(
            np.abs(cycle_ends1 - np.where(data_all1[:].T[1] > 0)[0].min())
        )
        cyc1 = np.argmin(
            np.abs(cycle_ends1 - np.where(data_all1[:].T[1] > 0)[0].min())
        )
        if cycle_ends1[cyc1] > np.where(data_all1[:].T[1] > 0)[0].min():
            cyc1 = cyc1 - 1
            cycle_start1 = cycle_ends1[cyc1]
        else:
            cycle_start1 = cycle_ends1[cyc1]

        os.chdir("..")

        # Second board, unpack data
        os.chdir(f"{motherboard_number2}")
        file = files_all2[i]
        if not absolute_timestamps:
            data_all2 = f_up.unpack_binary_data(
                file,
                daughterboard_number,
                motherboard_number2,
                firmware_version,
                timestamps,
                include_offset,
                apply_calibration,
            )
        else:
            (
                data_all2,
                abs_tmsp2,
            ) = f_up.unpack_binary_data_with_absolute_timestamps(
                file,
                daughterboard_number,
                motherboard_number2,
                firmware_version,
                timestamps,
                include_offset,
                apply_calibration,
            )
        # abs_tmsp_list2.append(abs_tmsp2)
        # Collect indices of cycle ends (the '-2's)
        cycle_ends2 = np.where(data_all2[0].T[1] == -2)[0]
        cyc2 = np.argmin(
            np.abs(cycle_ends2 - np.where(data_all2[:].T[1] > 0)[0].min())
        )
        if cycle_ends2[cyc2] > np.where(data_all2[:].T[1] > 0)[0].min():
            cyc2 = cyc2 - 1
            cycle_start2 = cycle_ends2[cyc2]
        else:
            cycle_start2 = cycle_ends2[cyc2]

        if cyc1 > cyc2:
            abs_tmsp_list1.append(abs_tmsp1[cyc1:])
            abs_tmsp_list2.append(abs_tmsp2[cyc2 : -cyc1 + cyc2])
        else:
            abs_tmsp_list1.append(abs_tmsp1[cyc1 : -cyc2 + cyc1])
            abs_tmsp_list2.append(abs_tmsp2[cyc2:])

        os.chdir("..")

        pix_left_peak = pixels[0]
        # The following piece of code take any value for the pixel from
        # the second motherboard, either in terms of full sensor (so
        # a value >=256) or in terms of single sensor half

        if pixels[1] >= 256:
            if pixels[1] > 256 + 127:
                pix_right_peak = 255 - (pixels[1] - 256)
            else:
                pix_right_peak = pixels[1] - 256 + 128
        elif pixels[1] > 127:
            pix_right_peak = 255 - pixels[1]
        else:
            pix_right_peak = pixels[1] + 128

        # Get the data from the requested pixel only
        deltas_all[f"{pix_left_peak},{pix_right_peak}"] = []
        tdc1, pix_c1 = np.argwhere(pixel_coordinates == pix_left_peak)[0]
        pix1 = np.where(data_all1[tdc1].T[0] == pix_c1)[0]
        tdc2, pix_c2 = np.argwhere(pixel_coordinates == pix_right_peak)[0]
        pix2 = np.where(data_all2[tdc2].T[0] == pix_c2)[0]

        # Data from one of the board should be shifted as data collection
        # on one of the board is started later
        if cycle_start1 > cycle_start2:
            cyc = len(data_all1[0].T[1]) - cycle_start1 + cycle_start2
            cycle_ends1 = cycle_ends1[cycle_ends1 >= cycle_start1]
            cycle_ends2 = np.intersect1d(
                cycle_ends2[cycle_ends2 >= cycle_start2],
                cycle_ends2[cycle_ends2 <= cyc],
            )

        else:
            cyc = len(data_all1[0].T[1]) - cycle_start2 + cycle_start1
            cycle_ends2 = cycle_ends2[cycle_ends2 >= cycle_start2]
            cycle_ends1 = np.intersect1d(
                cycle_ends1[cycle_ends1 >= cycle_start1],
                cycle_ends1[cycle_ends1 <= cyc],
            )

        # Get timestamps for both pixels in the given cycle
        for cyc in range(len(cycle_ends1) - 1):
            pix1_ = pix1[
                np.logical_and(
                    pix1 >= cycle_ends1[cyc], pix1 < cycle_ends1[cyc + 1]
                )
            ]
            if not np.any(pix1_):
                continue
            pix2_ = pix2[
                np.logical_and(
                    pix2 >= cycle_ends2[cyc], pix2 < cycle_ends2[cyc + 1]
                )
            ]

            if not np.any(pix2_):
                continue
            # Calculate delta t
            tmsp1 = data_all1[tdc1].T[1][
                pix1_[np.where(data_all1[tdc1].T[1][pix1_] > 0)[0]]
            ]
            tmsp2 = data_all2[tdc2].T[1][
                pix2_[np.where(data_all2[tdc2].T[1][pix2_] > 0)[0]]
            ]
            for t1 in tmsp1:
                deltas = tmsp2 - t1
                ind = np.where(np.abs(deltas) < delta_window)[0]
                deltas_all[f"{pix_left_peak},{pix_right_peak}"].extend(
                    deltas[ind]
                )

        # Save data to a feather file in a cycle so data is not lost
        # in the case of failure close to the end
        data_for_plot_df = pd.DataFrame.from_dict(deltas_all, orient="index")
        del deltas_all
        data_for_plot_df = data_for_plot_df.T

        feather_file = f"{out_file_name}.feather"

        try:
            os.chdir("delta_ts_data")
        except FileNotFoundError:
            os.mkdir("delta_ts_data")
            os.chdir("delta_ts_data")

        if os.path.isfile(feather_file):
            # Load existing Feather file
            existing_data = ft.read_feather(feather_file)

            # Append new data to the existing Feather file
            combined_data = pd.concat(
                [existing_data, data_for_plot_df], axis=0
            )
            ft.write_feather(combined_data, feather_file)

        else:
            # Save as a new Feather file
            ft.write_feather(data_for_plot_df, feather_file)

        os.chdir("..")

    # Check if the file with the results was created
    if (
        os.path.isfile(os.path.join(path, f"/delta_ts_data/{feather_file}"))
        is True
    ):
        print(
            "\n> > > Timestamp differences are saved as"
            f"{feather_file} in {os.path.join(path, 'delta_ts_data/')} < < <"
        )
    else:
        print("File wasn't generated. Check input parameters.")

    return abs_tmsp_list1, abs_tmsp_list2


# Absolute-timestamp tick (133.333 MHz clock), in ps
_ABS_TICK_PS = 7500
# Average LinoSPAD2 TDC bin width, in ps (raw fine code -> ps)
_TDC_BIN_PS = 2500 / 140


def _remap_full_sensor_pixel(pixel: int) -> int:
    """Map a full-sensor pixel index to its raw pixel on the second board.

    Full-sensor indices 256..511 address the right sensor half; this
    returns the corresponding raw pixel (0..255) on the second
    motherboard, matching the convention used throughout the full-sensor
    functions.

    Parameters
    ----------
    pixel : int
        Full-sensor pixel index.

    Returns
    -------
    int
        Raw pixel index on the second motherboard.
    """
    if pixel >= 256:
        if pixel > 256 + 127:
            return 255 - (pixel - 256)
        return pixel - 256 + 128
    if pixel > 127:
        return 255 - pixel
    return pixel + 128


def _calibrated_fine(
    raw, calib_pixel, calib, offset, apply_calibration, include_offset
):
    """Convert raw fine codes to picoseconds within a cycle.

    Applies TDC calibration (and, optionally, offset calibration) exactly
    as it is applied elsewhere in this module.

    Parameters
    ----------
    raw : numpy.ndarray
        Raw fine codes for a single pixel.
    calib_pixel : int
        Pixel index into the calibration matrices.
    calib : numpy.ndarray or None
        TDC calibration matrix, or None if calibration is disabled.
    offset : numpy.ndarray or None
        Offset calibration array, or None.
    apply_calibration : bool
        Switch for applying TDC calibration.
    include_offset : bool
        Switch for also applying offset calibration.

    Returns
    -------
    numpy.ndarray
        Calibrated fine part of the timestamp, in ps.
    """
    if not apply_calibration or calib is None:
        return raw * _TDC_BIN_PS
    fine = (raw - raw % 140) * _TDC_BIN_PS + calib[calib_pixel, raw % 140]
    if include_offset and offset is not None:
        fine = fine + offset[calib_pixel]
    return fine


def _collect_board_timeline(
    files,
    daughterboard_number,
    motherboard_number,
    firmware_version,
    timestamps,
    raw_pixels,
    pixel_coordinates,
    calib,
    offset,
    apply_calibration,
    include_offset,
):
    """Pool every requested pixel's photons onto one continuous timeline.

    Each file is unpacked once. For every requested (raw) pixel the photon
    times are placed on the board's free-running absolute-timestamp
    timeline (absolute tick * 7500 ps + calibrated fine part) and pooled
    across the whole run.

    Parameters
    ----------
    files : list
        Data files for this board, in acquisition order.
    daughterboard_number : str
        LinoSPAD2 daughterboard number.
    motherboard_number : str
        LinoSPAD2 motherboard (FPGA) number, including the '#'.
    firmware_version : str
        LinoSPAD2 firmware version ("2212s" or "2212b").
    timestamps : int
        Number of timestamps per acquisition cycle per pixel.
    raw_pixels : list
        Raw pixel indices (0..255) to collect on this board.
    pixel_coordinates : numpy.ndarray
        Matrix mapping pixels to (TDC, column) for the firmware version.
    calib : numpy.ndarray or None
        TDC calibration matrix, or None.
    offset : numpy.ndarray or None
        Offset calibration array, or None.
    apply_calibration : bool
        Switch for applying TDC calibration.
    include_offset : bool
        Switch for also applying offset calibration.

    Returns
    -------
    per_pixel : dict
        Maps each raw pixel to its sorted array of absolute times (ps).
    file_first_ticks : numpy.ndarray
        First absolute tick of each file, in acquisition order. Used to
        verify the counters are monotonic across the run.
    increment : int
        Modal per-cycle absolute-tick increment, taken from the first
        file (the full-cycle counter step).
    """
    raw_pixels = list(dict.fromkeys(raw_pixels))
    tdc_column = {
        p: tuple(np.argwhere(pixel_coordinates == p)[0]) for p in raw_pixels
    }
    parts = {p: [] for p in raw_pixels}
    file_first_ticks = []
    increment = None

    for file_number, file in enumerate(
        tqdm(files, desc=f"Unpacking {motherboard_number}")
    ):
        if os.path.getsize(file) // 4 % (timestamps * 65 + 2) != 0:
            raise ValueError(
                f"File '{os.path.basename(file)}' does not contain "
                "absolute timestamps (expected 65 data rows + a 2-word "
                "header per cycle). Enable 'acqTimestamps' in the "
                "LinoSPAD2 GUI before acquiring."
            )
        data_pixels, data_timestamps, abs_tmsp = (
            f_up.unpack_binary_data_with_absolute_timestamps(
                file,
                daughterboard_number,
                motherboard_number,
                firmware_version,
                timestamps,
            )
        )
        abs_tmsp = abs_tmsp.astype(np.int64)
        file_first_ticks.append(int(abs_tmsp[0]))
        if file_number == 0:
            steps, step_counts = np.unique(
                np.diff(abs_tmsp), return_counts=True
            )
            increment = int(steps[step_counts.argmax()])
        cycle_of_column = np.arange(data_timestamps.shape[1]) // timestamps
        for pixel in raw_pixels:
            tdc, pix_c = tdc_column[pixel]
            mask = (data_pixels[tdc] == pix_c) & (data_timestamps[tdc] >= 0)
            raw = data_timestamps[tdc][mask].astype(np.int64)
            fine = _calibrated_fine(
                raw, pixel, calib, offset, apply_calibration, include_offset
            )
            parts[pixel].append(
                abs_tmsp[cycle_of_column[mask]].astype(np.float64)
                * _ABS_TICK_PS
                + fine
            )

    per_pixel = {
        p: (
            np.sort(np.concatenate(parts[p]))
            if parts[p]
            else np.array([], dtype=np.float64)
        )
        for p in raw_pixels
    }
    return per_pixel, np.array(file_first_ticks, dtype=np.int64), increment


def _count_within_window(times_left, times_right_sorted, offset, window):
    """Count coincidences within +-window of each left time plus offset.

    Parameters
    ----------
    times_left : numpy.ndarray
        Left-board times (ps), any order.
    times_right_sorted : numpy.ndarray
        Right-board times (ps), sorted ascending.
    offset : float
        Board-to-board offset (ps) added to the left times.
    window : float
        Half-width of the coincidence window (ps).

    Returns
    -------
    int
        Number of right-board times inside the window.
    """
    lo = np.searchsorted(times_right_sorted, times_left + offset - window)
    hi = np.searchsorted(times_right_sorted, times_left + offset + window)
    return int((hi - lo).sum())


def _gather_within_window(times_left, times_right_sorted, offset, window):
    """Return every (t_right - t_left - offset) with |.| < window.

    Parameters
    ----------
    times_left : numpy.ndarray
        Left-board times (ps), any order.
    times_right_sorted : numpy.ndarray
        Right-board times (ps), sorted ascending.
    offset : float
        Board-to-board offset (ps) subtracted from each difference.
    window : float
        Half-width of the window (ps) around the offset that is kept.

    Returns
    -------
    numpy.ndarray
        Centered timestamp differences (ps) inside the window.
    """
    collected = []
    block = 20000
    for start in range(0, times_left.size, block):
        chunk = times_left[start : start + block]
        lo = np.searchsorted(times_right_sorted, chunk + offset - window)
        hi = np.searchsorted(times_right_sorted, chunk + offset + window)
        counts = hi - lo
        total = int(counts.sum())
        if total == 0:
            continue
        idx = np.repeat(lo, counts) + (
            np.arange(total) - np.repeat(np.cumsum(counts) - counts, counts)
        )
        collected.append(
            times_right_sorted[idx] - np.repeat(chunk, counts) - offset
        )
    return (
        np.concatenate(collected) if collected else np.array([], dtype=float)
    )


def calculate_and_save_timestamp_differences_full_sensor_alt(
    path,
    pixels: list,
    rewrite: bool,
    daughterboard_number: str,
    motherboard_number1: str,
    motherboard_number2: str,
    firmware_version: str,
    timestamps: int = 512,
    delta_window: float = 50e3,
    include_offset: bool = False,
    apply_calibration: bool = True,
    scan_window: float = 1e6,
):
    """Calculate and save cross-board timestamp differences to '.feather'.

    Unpacks data from both sensor halves/both FPGAs, finds the constant
    board-to-board offset automatically from the absolute timestamps,
    calculates timestamp differences for the requested pixels around that
    offset, and saves them into a '.feather' table. Works with firmware
    versions "2212s" and "2212b". Useful for data where the signal is
    above zero right from the start of data collecting.

    The two boards are assumed to share a common external clock but NOT a
    common trigger (CLK_IN/J11 only). Each FPGA then has its own
    free-running 133.333 MHz absolute-timestamp counter (7.5 ns per tick)
    that starts at an arbitrary power-up moment, so the boards are related
    by a single large constant offset (Delta_epoch). For this acquisition
    mode the counters are monotonic and continuous across the whole run
    (each file's first tick exceeds the previous file's last), so one
    global offset aligns every photon on both boards:

        T_right(event) - T_left(event) = Delta_epoch   (constant)

    Every requested pixel's photons are pooled onto each board's
    continuous timeline; the integer-cycle lag that maximizes the number
    of coincidences within 'scan_window' fixes Delta_epoch; then, per
    pixel pair, all differences within 'delta_window' of that offset are
    kept. There is no 'epoch_offset_ps' parameter - the offset is always
    found from the data.

    Absolute timestamps are required: each data file must have been
    collected with 'acqTimestamps' enabled in the LinoSPAD2 GUI (adds a
    2-word header per acquisition cycle).

    Absolute timestamps are always required. Each data file must have been
    collected with 'acqTimestamps' enabled in the LinoSPAD2 GUI (adds a
    2-word header per acquisition cycle). The per-cycle clock offset between
    the two boards is computed from these timestamps and subtracted from
    every photon timestamp difference before saving.

    Parameters
    ----------
    path : str
        Path to where two folders with data from both motherboards are.
        The folders should be named after the motherboards.
    pixels : list
        Either a list of two pixels (one from each sensor half) or a list
        of two lists of pixels, [[left, ...], [right, ...]]. Pixels for
        the second (right) half are given as full-sensor indices
        (256..511).
    rewrite : bool
        Switch for rewriting the '.feather' file if it already exists.
        Used as a safeguard to avoid unwanted overwriting.
    daughterboard_number : str
        LinoSPAD2 daughterboard number.
    motherboard_number1 : str
        First LinoSPAD2 motherboard (FPGA) number, including the '#'.
        Corresponds to the left sensor half.
    motherboard_number2 : str
        Second LinoSPAD2 motherboard (FPGA) number, including the '#'.
        Corresponds to the right sensor half.
    firmware_version : str
        LinoSPAD2 firmware version. Versions "2212s" (skip) and "2212b"
        (block) are recognized.
    timestamps : int, optional
        Number of timestamps per acquisition cycle per pixel. The default
        is 512.
    delta_window : float, optional
        Half-width of the window (in ps) around the located offset within
        which timestamp differences are kept. Must be wider than the
        board-to-board skew (tens of ns). The default is 50e3 (50 ns).
    include_offset : bool, optional
        Switch for applying offset calibration (requires an offset
        calibration file for both boards). The default is False.
    apply_calibration : bool, optional
        Switch for applying TDC calibration. If set to 'True' while
        'include_offset' is 'False', only the TDC calibration is applied.
        The default is True.
    scan_window : float, optional
        Half-width of the window (in ps) used while locating the coarse
        board-to-board offset. The default is 1e6 (1 us).

    Returns
    -------
    dict
        Dictionary describing the located alignment, with the keys:
        'feather' (path to the saved file), 'cycle_lag_N' (the integer
        cycle lag that aligns the boards), 'delta_epoch_ps' and
        'delta_epoch_s' (the board-to-board offset), and 'pair_counts'
        (number of differences saved per pixel pair).

    Raises
    ------
    TypeError
        If 'pixels' is not a list, 'firmware_version' is not a string,
        'rewrite' is not a boolean, or 'daughterboard_number' is not a
        string.
    FileNotFoundError
        If data from either motherboard, or the calibration data, were
        not found.
    ValueError
        If a data file does not contain absolute timestamps.
    RuntimeError
        If the absolute-timestamp counters are not monotonic across the
        run, if the requested pixels contain no photons, or if no
        significant coincidence offset can be located.
    """
    # parameter type check
    if isinstance(pixels, list) is False:
        raise TypeError(
            "'pixels' should be a list of integers or a list of two lists"
        )
    if isinstance(firmware_version, str) is False:
        raise TypeError(
            "'firmware_version' should be string, '2212s' or '2212b'"
        )
    if isinstance(rewrite, bool) is False:
        raise TypeError("'rewrite' should be boolean")
    if isinstance(daughterboard_number, str) is False:
        raise TypeError("'daughterboard_number' should be string")

    # Define matrix of pixel coordinates, where rows are numbers of TDCs
    # and columns are the pixels that connected to these TDCs
    if firmware_version == "2212s":
        pixel_coordinates = np.arange(256).reshape(4, 64).T
    elif firmware_version == "2212b":
        pixel_coordinates = np.arange(256).reshape(64, 4)
    else:
        print("\nFirmware version is not recognized.")
        sys.exit()

    # Collect data files from both boards in acquisition order. File names
    # are timestamped, so sorting by name matches acquisition order and is
    # consistent with how the '.feather' name is reconstructed by the fit
    # functions.
    files_all1 = sorted(
        glob.glob(os.path.join(path, motherboard_number1, "*.dat*")),
        key=os.path.basename,
    )
    files_all2 = sorted(
        glob.glob(os.path.join(path, motherboard_number2, "*.dat*")),
        key=os.path.basename,
    )
    if not files_all1:
        raise FileNotFoundError(f"Data from {motherboard_number1} not found")
    if not files_all2:
        raise FileNotFoundError(f"Data from {motherboard_number2} not found")

    out_file_name = (
        os.path.basename(files_all1[0])[:-4]
        + "-"
        + os.path.basename(files_all2[-1])[:-4]
    )

    # Check if '.feather' file with timestamps differences already exists
    feather_file = os.path.join(
        path, "delta_ts_data", f"{out_file_name}.feather"
    )
    utils.file_rewrite_handling(feather_file, rewrite)

    # Normalize pixel input: accept [p1, p2] or [[p1a, ...], [p2a, ...]]
    if isinstance(pixels[0], list):
        pixels_left = pixels[0]
        pixels_right = pixels[1]
    else:
        pixels_left = [pixels[0]]
        pixels_right = [pixels[1]]

    # Second-board raw pixel for each requested full-sensor right pixel
    pix_right_remapped = {
        p: _remap_full_sensor_pixel(p) for p in pixels_right
    }

    # Load calibration data once before unpacking
    calib1 = calib2 = offset1 = offset2 = None
    if apply_calibration:
        path_calibration_data = os.path.join(
            os.path.dirname(os.path.realpath(__file__)),
            "..",
            "params",
            "calibration_data",
        )
        try:
            if include_offset:
                calib1, offset1 = cb.load_calibration_data(
                    path_calibration_data,
                    daughterboard_number,
                    motherboard_number1,
                    firmware_version,
                    include_offset,
                )
                calib2, offset2 = cb.load_calibration_data(
                    path_calibration_data,
                    daughterboard_number,
                    motherboard_number2,
                    firmware_version,
                    include_offset,
                )
            else:
                calib1 = cb.load_calibration_data(
                    path_calibration_data,
                    daughterboard_number,
                    motherboard_number1,
                    firmware_version,
                    include_offset,
                )
                calib2 = cb.load_calibration_data(
                    path_calibration_data,
                    daughterboard_number,
                    motherboard_number2,
                    firmware_version,
                    include_offset,
                )
        except FileNotFoundError as exc:
            raise FileNotFoundError(
                "No .csv file with the calibration data was found. "
                "Check the path or run the calibration."
            ) from exc

    # Pool every requested pixel's photons onto each board's continuous
    # absolute-timestamp timeline (each file is unpacked once).
    left_timeline, first_ticks1, increment = _collect_board_timeline(
        files_all1,
        daughterboard_number,
        motherboard_number1,
        firmware_version,
        timestamps,
        pixels_left,
        pixel_coordinates,
        calib1,
        offset1,
        apply_calibration,
        include_offset,
    )
    right_timeline_raw, first_ticks2, _ = _collect_board_timeline(
        files_all2,
        daughterboard_number,
        motherboard_number2,
        firmware_version,
        timestamps,
        list(pix_right_remapped.values()),
        pixel_coordinates,
        calib2,
        offset2,
        apply_calibration,
        include_offset,
    )
    times_left = {p: left_timeline[p] for p in pixels_left}
    times_right = {
        p: right_timeline_raw[pix_right_remapped[p]] for p in pixels_right
    }

    # The global-alignment assumption requires monotonic counters
    if not (
        np.all(np.diff(first_ticks1) > 0)
        and np.all(np.diff(first_ticks2) > 0)
    ):
        raise RuntimeError(
            "Absolute-timestamp counters are not monotonically increasing "
            "across files; the global-alignment assumption fails. Check "
            "that the boards were not power-cycled mid-run."
        )

    # The counters step by one full cycle each acquisition cycle; the
    # boards latch acquisition starts on service half-cycles, so search
    # candidate offsets on the half-cycle grid.
    step = increment // 2
    common = min(first_ticks1.size, first_ticks2.size)
    epoch = first_ticks2[:common] - first_ticks1[:common]
    phase = int(np.median(epoch % step))
    if (epoch % step).std() > 2:
        print(
            "\nWarning: the sub-cycle phase between the boards is not "
            f"constant (std {(epoch % step).std():.1f} ticks); the offset "
            "search may be less reliable."
        )
    lag_low = int((epoch - phase).min() // step) - 40
    lag_high = int((epoch - phase).max() // step) + 40

    # Locate the global coarse offset from the pooled beams
    left_pool = np.sort(np.concatenate([times_left[p] for p in pixels_left]))
    right_pool = np.sort(
        np.concatenate([times_right[p] for p in pixels_right])
    )
    if left_pool.size == 0 or right_pool.size == 0:
        raise RuntimeError("No photons in the requested pixels.")

    lags = np.arange(lag_low, lag_high + 1)
    counts = np.array(
        [
            _count_within_window(
                left_pool,
                right_pool,
                (phase + lag * step) * _ABS_TICK_PS,
                scan_window,
            )
            for lag in lags
        ]
    )
    best = int(counts.argmax())
    cycle_lag_N = int(lags[best])
    delta_epoch_ps = (phase + cycle_lag_N * step) * _ABS_TICK_PS
    baseline = np.median(np.delete(counts, slice(max(best - 3, 0), best + 4)))
    excess = counts[best] - baseline
    significance = excess / np.sqrt(max(baseline, 1.0))
    print(
        f"\nBest cycle lag N={cycle_lag_N}: {counts[best]} coincidences "
        f"within +-{scan_window / 1e3:.0f} ns "
        f"(baseline ~{baseline:.0f}, {significance:.0f} sigma above it)"
    )
    if excess < 5 * np.sqrt(max(baseline, 1.0)):
        raise RuntimeError(
            "No significant coincidence offset was found. Either the two "
            "beams are not conjugate, one arm is blocked, or the boards "
            "were not both locked to the shared external clock (check that "
            "both read EXT LOCKED at the same frequency)."
        )
    print(
        f"Board-to-board offset Delta_epoch = {delta_epoch_ps / 1e12:.6f} s"
    )

    # Per pixel pair: keep all differences within delta_window of offset
    deltas_all = {}
    pair_counts = {}
    for pix_left in pixels_left:
        left = times_left[pix_left]
        for pix_right in pixels_right:
            diffs = _gather_within_window(
                left, times_right[pix_right], delta_epoch_ps, delta_window
            )
            key = f"{pix_left},{pix_right}"
            deltas_all[key] = diffs.tolist()
            pair_counts[key] = int(diffs.size)

    # Save data to a single '.feather' file
    os.makedirs(os.path.join(path, "delta_ts_data"), exist_ok=True)
    data_for_plot_df = pd.DataFrame.from_dict(deltas_all, orient="index").T
    ft.write_feather(data_for_plot_df, feather_file)

    if os.path.isfile(feather_file):
        print(
            "\n> > > Timestamp differences are saved as "
            f"{out_file_name}.feather in "
            f"{os.path.join(path, 'delta_ts_data')} < < <"
        )
    else:
        print("File wasn't generated. Check input parameters.")

    return {
        "feather": feather_file,
        "cycle_lag_N": cycle_lag_N,
        "delta_epoch_ps": float(delta_epoch_ps),
        "delta_epoch_s": float(delta_epoch_ps / 1e12),
        "pair_counts": pair_counts,
    }


def collect_and_plot_timestamp_differences(
    path,
    pixels,
    rewrite: bool,
    ft_file: str = None,
    range_left: int = -10e3,
    range_right: int = 10e3,
    multiplier: int = 1,
    same_y: bool = False,
    color: str | None = None,
    correct_pix_address: bool = False,
    pickle_figure: bool = False,
    file_offset_abs: str = None,
):
    """Collect and plot timestamp differences from a '.feather' file.

    Plots timestamp differences from a '.feather' file as a grid of histograms
    and as a single plot. The plot is saved in the 'results/delta_t' folder,
    which is created (if it does not already exist) in the same folder
    where data are.

    Parameters
    ----------
    path : str
        Path to the folder with '.dat' data files.
    pixels : list
        List of pixel numbers for which the timestamp differences
        should be plotted.
    rewrite : bool
        switch for rewriting the plot if it already exists. used as a
        safeguard to avoid unwanted overwriting of the previous results.
        Switch for rewriting the plot if it already exists. Used as a
        safeguard to avoid unwanted overwriting of the previous results.
    ft_file : str, optional
        Path to the feather file with timestamp differences. If used,
        the data files in the path are ignored. The default is None.
    range_left : int, optional
        Lower limit for timestamp differences, lower values are not used.
        The default is -10e3.
    range_right : int, optional
        Upper limit for timestamp differences, higher values are not used.
        The default is 10e3.
    multiplier : int, optional
        Histogram binning multiplier. Can be used for coarser binning.
        The default is 1.
    same_y : bool, optional
        Switch for plotting the histograms with the same y-axis.
        The default is False.
    color : str, optional
        Color for the plot. The default is 'rebeccapurple'.
    correct_pix_address : bool, optional
        Correct pixel address for the sensor half on side 23 of the
        daughterboard. The default is False.
    pickle_figure : bool, optional
        Switch for pickling the plot. Can be used to extract the plot
        data. The default is False.
    file_offset_abs : str, optional
        Absolute path to the '.npy' file with the offset calibration
        for the particular board. The default is "".

    Raises
    ------
    TypeError
        Only boolean values of 'rewrite' are accepted. The error is
        raised so that the plot does not accidentally gets rewritten.

    Returns
    -------
    None.
    """
    # parameter type check
    if isinstance(rewrite, bool) is not True:
        raise TypeError("'rewrite' should be boolean")
    os.chdir(path)

    if ft_file is not None:
        feather_file_name = ft_file.split(".")[0]
    else:
        files_all = glob.glob("*.dat*")
        files_all.sort(key=os.path.getmtime)
        feather_file_name = files_all[0][:-4] + "-" + files_all[-1][:-4]

        # Check if plot exists and if it should be rewritten
        try:
            os.chdir("results/delta_t")
            if os.path.isfile(f"{feather_file_name}_delta_t_grid.png"):
                if rewrite is True:
                    print(
                        "\n! ! ! Plot of timestamp differences already"
                        "exists and will be rewritten ! ! !\n"
                    )
                else:
                    sys.exit(
                        "\nPlot already exists, 'rewrite' set to 'False', exiting."
                    )
            os.chdir("../..")
        except FileNotFoundError:
            pass

    print(
        "\n> > > Plotting timestamps differences as a grid of histograms < < <"
    )
    print(
        "\n> > > Plotting timestamps differences as a grid of histograms < < <"
    )

    # In the case two lists given - the left and right peaks - _flatten
    # into a single list

    # Save to use in the title
    # pixels_title = np.copy(pixels)

    if correct_pix_address:
        for i, pixel in enumerate(pixels):
            if pixel > 127:
                pixels[i] = 255 - pixels[i]
            else:
                pixels[i] = pixels[i] + 128

    pixels = _flatten(pixels)
    # TODO test
    # pixels_left, pixels_right = utils.pixel_list_transform(pixels)

    if len(pixels) > 2:
        fig, axs = plt.subplots(
            len(pixels) - 1,
            len(pixels) - 1,
            figsize=(5.5 * len(pixels), 5.5 * len(pixels)),
        )
        for ax in axs:
            for x in ax:
                x.axes.set_axis_off()
    else:
        fig = plt.figure()

    # Check if the y limits of all plots should be the same
    if same_y is True:
        y_max_all = 0

    for q, _ in tqdm(enumerate(pixels), desc="Row in plot"):
        for w, _ in enumerate(pixels):
            if w <= q:
                continue
            if len(pixels) > 2:
                axs[q][w - 1].axes.set_axis_on()

            # Read data from Feather file
            if ft_file is not None:
                try:
                    data_to_plot = ft.read_feather(
                        ft_file, columns=[f"{pixels[q]},{pixels[w]}"]
                    ).dropna()
                except ValueError:
                    continue
            else:
                try:
                    data_to_plot = ft.read_feather(
                        f"delta_ts_data/{feather_file_name}.feather",
                        columns=[f"{pixels[q]},{pixels[w]}"],
                    ).dropna()
                except ValueError:
                    continue

            # Prepare the data for the plot
            data_to_plot = np.array(data_to_plot)
            data_to_plot = np.delete(
                data_to_plot, np.argwhere(data_to_plot < range_left)
            )
            data_to_plot = np.delete(
                data_to_plot, np.argwhere(data_to_plot > range_right)
            )

            if file_offset_abs is not None:
                try:
                    data_offset = np.load(file_offset_abs)
                    delay_pix1 = data_offset[pixels[q]]
                    delay_pix2 = data_offset[pixels[w]]
                    data_to_plot = data_to_plot - delay_pix1 + delay_pix2
                except FileNotFoundError:
                    print(
                        "No absolute path to the '.npy' file with "
                        "the offset calibration data was provided. Offset "
                        "calibration is not applied."
                    )
                    pass
            # else:
            #     print(
            #         "No '.npy' file with offset calibration was provided. "
            #         "Offset calibration is not included."
            #     )

            # Bins should be in units of 17.857 ps - average bin width
            # of the LinoSPAD2 TDCs
            try:
                bins = np.arange(
                    np.min(data_to_plot),
                    np.max(data_to_plot),
                    2500 / 140 * multiplier,
                )
            except ValueError:
                print(
                    f"\nCouldn't calculate bins for {q}-{w} pair: "
                    "probably not enough delta ts."
                )
                continue

            if len(pixels) > 2:
                axs[q][w - 1].set_xlabel("\u0394t (ps)")
                axs[q][w - 1].set_ylabel("# of coincidences (-)")
                n, b, p = axs[q][w - 1].hist(
                    data_to_plot,
                    bins=bins,
                    color=color,
                )
            else:
                plt.xlabel("\u0394t (ps)")
                plt.ylabel("# of coincidences (-)")
                n, b, p = plt.hist(
                    data_to_plot,
                    bins=bins,
                    color=color,
                )

            # Find number of timestamps differences in a 2 ns window
            # around the peak (HBT or cross-talk)
            try:
                peak_max_pos = np.argmax(n).astype(np.intc)
                # 2 ns window around peak
                win = int(1000 / ((range_right - range_left) / 100))
                # Integrated peak, computed for inspection only here.
                peak_max = np.sum(  # noqa: F841
                    n[peak_max_pos - win : peak_max_pos + win]
                )
            except ValueError:
                peak_max = 0  # noqa: F841

            if same_y is True:
                try:
                    y_max = np.max(n)
                except ValueError:
                    y_max = 0
                    print("\nCould not find maximum y value\n")
                if y_max_all < y_max:
                    y_max_all = y_max
                if len(pixels) > 2:
                    axs[q][w - 1].set_ylim(0, y_max + 4)
                else:
                    plt.ylim(0, y_max + 4)

            if len(pixels) > 2:
                axs[q][w - 1].set_xlim(range_left - 100, range_right + 100)
                axs[q][w - 1].set_title(
                    f"Pixels {pixels[q]},{pixels[w]}"
                    # f"Pixels {pixels[q]},{pixels[w]}\nPeak in 2 ns "
                    # f"window: {int(peak_max)}"
                )
            else:
                plt.xlim(range_left - 100, range_right + 100)
                # Cut the first x tick label to avoid overlapping with
                # y-axis ticks
                ax = plt.gca()
                ticks = ax.get_xticks()
                tick_labels = ax.get_xticklabels()
                ax.set_xticks(ticks[2:-1], tick_labels[2:-1])

                plt.title(f"Pixels {pixels[q]},{pixels[w]}")

            # Save the figure
            try:
                os.chdir("results/delta_t")
            except FileNotFoundError:
                os.makedirs("results/delta_t")
                os.chdir("results/delta_t")
            fig.tight_layout()  # for perfect spacing between the plots
            plt.savefig(f"{feather_file_name}_delta_t_grid.png")

            os.chdir(path)

    # Pickle the figure if requested
    try:
        os.chdir("results/delta_t")
    except FileNotFoundError:
        os.makedirs("results/delta_t")
        os.chdir("results/delta_t")

    if pickle_figure:
        with open(f"{feather_file_name}_delta_t_grid.pkl", "wb") as f:
            pickle.dump(fig, f)

    print(
        "\n> > > Plot is saved as {file} in {path}< < <".format(
            file=feather_file_name + "_delta_t_grid.png",
            path=path + "/results/delta_t",
        )
    )


def collect_and_plot_timestamp_differences_full_sensor(
    path,
    pixels,
    rewrite: bool,
    range_left: int = -10e3,
    range_right: int = 10e3,
    multiplier: int = 1,
    same_y: bool = False,
    color: str | None = None,
):
    """Collect and plot timestamp differences from a '.feather' file.

    Plots timestamp differences from a '.feather' file as a grid of
    histograms and as a single plot. The plot is saved in the
    'results/delta_t' folder, which is created (if it does not already
    exist) in the same folder where data are.

    Parameters
    ----------
    path : str
        Path to the folder with '.dat' data files.
    pixels : list
        List of pixel numbers for which the timestamp differences
        should be plotted.
    rewrite : bool
        switch for rewriting the plot if it already exists. used as a
        safeguard to avoid unwanted overwriting of the previous results.
        Switch for rewriting the plot if it already exists. Used as a
        safeguard to avoid unwanted overwriting of the previous results.
    range_left : int, optional
        Lower limit for timestamp differences, lower values are not used.
        The default is -10e3.
    range_right : int, optional
        Upper limit for timestamp differences, higher values are not used.
        The default is 10e3.
    multiplier : int, optional
        Histogram binning multiplier. Can be used for coarser binning.
        The default is 1.
    same_y : bool, optional
        Switch for plotting the histograms with the same y-axis.
        The default is False.
    color : str, optional
        Color for the plot. The default is 'rebeccapurple'.

    Raises
    ------
    TypeError
        Only boolean values of 'rewrite' are accepted. The error is
        raised so that the plot does not accidentally gets rewritten.

    Returns
    -------
    None.
    """
    # parameter type check
    if isinstance(rewrite, bool) is not True:
        raise TypeError("'rewrite' should be boolean")
    plt.ioff()
    os.chdir(path)

    # Flatten the input list of pixels: if there are lists of pixels'
    # numbers inside the list, unpack them so that the output is a list
    # of numbers only
    pixels = _flatten(pixels)

    # Get the data files names for finding the appropriate '.feather'
    # file with timestamps differences, checking both options, depending
    # on which board was analyzed first
    folders = glob.glob("*#*")
    os.chdir(folders[0])
    # files_all = sorted(glob.glob("*.dat*"))
    files_all = glob.glob("*.dat*")
    # files_all.sort(key=lambda x: os.path.getmtime(x))
    files_all.sort(key=os.path.getmtime)

    feather_file_name1 = files_all[0][:-4] + "-"
    feather_file_name2 = "-" + files_all[-1][:-4]

    os.chdir(f"../{folders[1]}")
    # files_all = sorted(glob.glob("*.dat*"))
    files_all = glob.glob("*.dat*")
    # files_all.sort(key=lambda x: os.path.getmtime(x))
    files_all.sort(key=os.path.getmtime)

    feather_file_name1 += files_all[-1][:-4]
    feather_file_name2 = files_all[0][:-4] + feather_file_name2
    os.chdir("..")

    # Check if plot exists and if it should be rewritten
    try:
        os.chdir(os.path.join(path, "results/delta_t"))
        if os.path.isfile(f"{feather_file_name1}_delta_t_grid.png"):
            if rewrite is True:
                print(
                    "\n! ! ! Plot of timestamp differences already"
                    "exists and will be rewritten ! ! !\n"
                )
            else:
                sys.exit(
                    "\nPlot already exists, 'rewrite' set to 'False', exiting."
                )

        elif os.path.isfile(f"{feather_file_name2}_delta_t_grid.png"):
            if rewrite is True:
                print(
                    "\n! ! ! Plot of timestamp differences already"
                    "exists and will be rewritten ! ! !\n"
                )
            else:
                sys.exit(
                    "\nPlot already exists, 'rewrite' set to 'False', exiting."
                )

        # os.chdir("../..")
    except FileNotFoundError:
        pass

    print(
        "\n> > > Plotting timestamps differences as a grid of histograms < < <"
    )
    # Prepare the grid for the plots based on the number of pixels
    # given
    if len(pixels) > 2:
        fig, axs = plt.subplots(
            len(pixels) - 1,
            len(pixels) - 1,
            figsize=(5.5 * len(pixels), 5.5 * len(pixels)),
        )
        for ax in axs:
            for x in ax:
                x.axes.set_axis_off()
    else:
        fig = plt.figure(figsize=(16, 16))

    # Check if the y limits of all plots should be the same
    if same_y is True:
        y_max_all = 0

    for q, _ in tqdm(enumerate(pixels), desc="Row in plot"):
        for w, _ in enumerate(pixels):
            if w <= q:
                continue
            if len(pixels) > 2:
                axs[q][w - 1].axes.set_axis_on()

            # Check if the file with timestamps differences is there
            feather_file_path1 = f"delta_ts_data/{feather_file_name1}.feather"
            feather_file_path2 = f"delta_ts_data/{feather_file_name2}.feather"
            feather_file, feather_file_name = (
                (feather_file_path1, feather_file_name1)
                if os.path.isfile(os.path.join(path, feather_file_path1))
                else (feather_file_path2, feather_file_name2)
            )
            if not os.path.isfile(os.path.join(path, feather_file)):
                raise FileNotFoundError(
                    "'.feather' file with timestamps differences was not found"
                )
            os.chdir(path)
            # Read data from Feather file
            try:
                data_to_plot = ft.read_feather(
                    os.path.join(
                        path, f"delta_ts_data/{feather_file_name}.feather"
                    ),
                    columns=[f"{pixels[q]},{pixels[w]}"],
                ).dropna()
            except ValueError:
                continue

            # Prepare the data for the plot
            data_to_plot = np.array(data_to_plot)
            data_to_plot = np.delete(
                data_to_plot, np.argwhere(data_to_plot < range_left)
            )
            data_to_plot = np.delete(
                data_to_plot, np.argwhere(data_to_plot > range_right)
            )

            # Bins should be in units of 17.857 ps - average bin width
            # of the LinoSPAD2 TDCs
            try:
                bins = np.arange(
                    np.min(data_to_plot),
                    np.max(data_to_plot),
                    2500 / 140 * multiplier,
                )
            except ValueError:
                print(
                    f"\nCouldn't calculate bins for {q}-{w} pair: probably "
                    "not enough delta ts."
                )
                continue

            if len(pixels) > 2:
                axs[q][w - 1].set_xlabel("\u0394t (ps)")
                axs[q][w - 1].set_ylabel("# of coincidences (-)")
                n, b, p = axs[q][w - 1].hist(
                    data_to_plot,
                    bins=bins,
                    color=color,
                )
            else:
                plt.xlabel("\u0394t (ps)")
                plt.ylabel("# of coincidences (-)")
                n, b, p = plt.hist(
                    data_to_plot,
                    bins=bins,
                    color=color,
                )

            # Find number of timestamps differences in a 2 ns window
            # around the peak (HBT or cross-talk)
            try:
                peak_max_pos = np.argmax(n).astype(np.intc)
                # 2 ns window around peak
                win = int(1000 / ((range_right - range_left) / 100))
                peak_max = np.sum(n[peak_max_pos - win : peak_max_pos + win])
            except ValueError:
                peak_max = 0

            if same_y is True:
                try:
                    y_max = np.max(n)
                except ValueError:
                    y_max = 0
                    print("\nCould not find maximum y value\n")
                if y_max_all < y_max:
                    y_max_all = y_max
                if len(pixels) > 2:
                    axs[q][w - 1].set_ylim(0, y_max + 4)
                else:
                    plt.ylim(0, y_max + 4)

            if len(pixels) > 2:
                axs[q][w - 1].set_xlim(range_left - 100, range_right + 100)
                axs[q][w - 1].set_title(
                    f"Pixels {pixels[q]},{pixels[w]}\nPeak in 2 ns "
                    f"window: {int(peak_max)}"
                )
            else:
                plt.xlim(range_left - 100, range_right + 100)

                plt.title(f"Pixels {pixels[0]},{256 + 255 - pixels[1]}")

            # Save the figure
            try:
                os.chdir("results/delta_t")
            except FileNotFoundError:
                os.makedirs("results/delta_t")
                os.chdir("results/delta_t")
            fig.tight_layout()  # for perfect spacing between the plots
            plt.savefig(
                f"{feather_file_name}_delta_t_grid.png"
            )
            os.chdir("../..")

    print(
        "\n> > > Plot is saved as {file} in {path}< < <".format(
            file=feather_file_name + "_delta_t_grid.png",
            path=path + "/results/delta_t",
        )
    )


def unpickle_plot(delta_t_pickle_file: str) -> dict:
    """Unpickle a saved figure and return plot data.

    Load a pickled figure, extract and return the data for each subplot
    found.

    Parameters
    ----------
    delta_t_pickle_file : str
        The absolute path to the pickle file with the histogram plot.

    Returns
    -------
    tuple
        A tuple containing:
        - fig (matplotlib.figure.Figure): The unpickled figure object.
        - plot_data (dict): A dictionary with keys "Histogram_i,j" where
        i,j are the pixels that were used for the particular histogram
        where each value is a tuple of x and y data for the
        corresponding bar plot.

    Raises
    ------
    FileNotFoundError
        If the specified pickle file does not exist, a FileNotFoundError
        is raised and an error message is printed.
    """
    # Unpickle the plot from '.pkl' file
    try:
        with open(delta_t_pickle_file, "rb") as f:
            fig = pickle.load(f)
    except FileNotFoundError as e:
        print(f"{e}")

    # Get the axes from the plot
    axes = fig.axes

    # Go over axes if there are any
    if len(axes) > 0:
        plot_data = {}
        for _, ax in enumerate(axes):
            if ax.containers:
                histogram_label = ax.get_title().split(" ")[1]
                bar_container = ax.containers[0]
                x = [
                    bar.get_x() + bar.get_width() / 2 for bar in bar_container
                ]
                # x = [bar.get_x() for bar in bar_container]
                y = [bar.get_height() for bar in bar_container]
                plot_data[f"Histogram_{histogram_label}"] = (x, y)

    if plot_data != {}:
        return fig, plot_data
    else:
        return fig
