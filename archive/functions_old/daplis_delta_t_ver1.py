import glob
import os
import sys
from math import ceil
from typing import List
from warnings import warn

import numpy as np
import pandas as pd
from pyarrow import feather as ft
from tqdm import tqdm

from daplis.functions import calc_diff as cd
from daplis.functions import unpack as f_up
from daplis.functions import utils


def calculate_and_save_timestamp_differences(
    path: str,
    pixels: List[int] | List[List[int]],
    rewrite: bool,
    daughterboard_number: str,
    motherboard_number: str,
    firmware_version: str,
    timestamps: int = 512,
    delta_window: float = 50e3,
    app_mask: bool = True,
    include_offset: bool = False,
    apply_calibration: bool = True,
    absolute_timestamps: bool = False,
    correct_pix_address: bool = False,
):
    """Calculate and save timestamp differences into '.feather' file.

    Unpacks data into a dictionary, calculates timestamp differences for
    the requested pixels, and saves them into a '.feather' table. Works with
    firmware version 2212.

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
    app_mask : bool, optional
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
        Only boolean values of 'rewrite' and string values of
        'daughterboard_number', 'motherboard_number', and 'firmware_version'
        are accepted. The first error is raised so that the plot does not
        accidentally get rewritten in the case no clear input was given.

    Returns
    -------
    None.

    Examples
    -------
    For the sensor half on the '23' side of the daughterboard, the
    pixel addressing should be correct. Let's assume the offset
    calibration was not done for this sensor and, therefore, the
    calibration matrix is not available - it should be passed as False.
    Let's collect timestamp differences for pairs of pixels 15-25,
    15-26, and 15-27.

    First, get the absolute path to where the '.dat' files are.
    >>> path = r'C:/Path/To/Data'

    Now to the function itself.
    >>> calculate_and_save_timestamp_differences(
    >>> path,
    >>> pixels = [15, [25,26,27]],
    >>> rewrite = True,
    >>> daughterboard_number="NL11",
    >>> motherboard_number="#21",
    >>> firmware_version="2212s",
    >>> timestamps = 1000,
    >>> include_offset = False,
    >>> correct_pixel_addressing = True,
    >>> )
    """

    # TODO: remove
    warn(
        "This function is deprecated. Use"
        "'calculate_and_save_timestamp_differences_fast'",
        DeprecationWarning,
        stacklevel=2,
    )

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

    # Handle the input pixel list
    pixels = utils.pixel_list_transform(pixels)

    files_all = glob.glob("*.dat*")
    files_all.sort(key=os.path.getmtime)

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

    # # Go back to the folder with '.dat' files
    # os.chdir(path)

    # Collect the data for the required pixels
    print(
        "\n> > > Collecting data for delta t plot for the requested "
        "pixels and saving it to .feather in a cycle < < <\n"
    )
    # Define matrix of pixel coordinates, where rows are numbers of TDCs
    # and columns are the pixels that connected to these TDCs
    if firmware_version == "2212s":
        pix_coor = np.arange(256).reshape(4, 64).T
    elif firmware_version == "2212b":
        pix_coor = np.arange(256).reshape(64, 4)
    else:
        print("\nFirmware version is not recognized.")
        sys.exit()

    # Correct pixel addressing for motherboard on side '23'
    if correct_pix_address:
        pixels = utils.correct_pixels_address(pixels)

    # Mask the hot/warm pixels
    if app_mask is True:
        mask = utils.apply_mask(daughterboard_number, motherboard_number)
        if isinstance(pixels[0], int) and isinstance(pixels[1], int):
            pixels = [pix for pix in pixels if pix not in mask]
        else:
            pixels[0] = [pix for pix in pixels[0] if pix not in mask]
            pixels[1] = [pix for pix in pixels[1] if pix not in mask]

    for i in tqdm(range(ceil(len(files_all))), desc="Collecting data"):
        file = files_all[i]

        # Prepare a dictionary for output
        deltas_all = {}

        # Unpack data for the requested pixels into dictionary
        if not absolute_timestamps:
            data_all = f_up.unpack_binary_data(
                file,
                daughterboard_number,
                motherboard_number,
                firmware_version,
                timestamps,
                include_offset,
                apply_calibration,
            )
        else:
            data_all, _ = f_up.unpack_binary_data_with_absolute_timestamps(
                file,
                daughterboard_number,
                motherboard_number,
                firmware_version,
                timestamps,
                include_offset,
                apply_calibration,
            )

        # Calculate the timestamp differences for the given pixels
        deltas_all = cd.calculate_differences_2212(
            data_all, pixels, pix_coor, delta_window
        )

        # Save data as a .feather file in a cycle so data is not lost
        # in the case of failure close to the end
        data_for_plot_df = pd.DataFrame.from_dict(deltas_all, orient="index")
        del deltas_all
        data_for_plot_df = data_for_plot_df.T
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
                combined_data = pd.concat(
                    [existing_data, data_for_plot_df], axis=0
                )
                combined_data = pd.concat(
                    [existing_data, data_for_plot_df], axis=0
                )
                ft.write_feather(combined_data, feather_file)
            else:
                ft_file_number += 1
                feather_file = f"{out_file_name}_{ft_file_number}.feather"
                ft.write_feather(data_for_plot_df, feather_file)

        else:
            # Save as a new feather file
            ft.write_feather(data_for_plot_df, feather_file)
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
