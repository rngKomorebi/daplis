"""Module for analyzing cross-talk of LinoSPAD2.

A set of functions to calculate and save, collect and plot the
cross-talk data for the given data sets.

This file can also be imported as a module and contains the following
functions:

    * collect_dcr_by_file - calculate the DCR for each file in the
    given folder and save the result into a .pkl file. Calculate
    average and median DCR with and without masking the hot pixels.

    * plot_dcr_histogram_and_stability - plot the DCR vs file to check
    setup stability, and plot histogram of DCR averaged over files
    together with integral over pixels

    * zero_to_cross_talk_collect - collect timestamp differences from
    the '.dat' with the cross-talk data (preferably, noise only).
    Timestamp differences are collected for the given hot pixels plus
    20 pixels to each side of the hot pixel where it is possible. The
    result is saved into a '.feather' file.

    * zero_to_cross_talk_plot - collect the timestamp differences from
    the '.feather' file and plot the cross-talk peaks applying a 
    Gaussian fit where possible, plot the average cross-talk probability
    vs distance from the hot pixel for each pair of aggressor-victim
    pixels, and finally plot the average cross-talk probability
    for the whole sensor half.

"""

import glob
import os
import pickle
import sys
from typing import List

import matplotlib
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from pyarrow import feather as ft
from scipy.optimize import curve_fit
from tqdm import tqdm

from daplis.functions import calc_diff as cd
from daplis.functions import sensor_plot
from daplis.functions import unpack as f_up
from daplis.functions import utils


def _collect_cross_talk(
    path: str,
    pixels: List[int],
    rewrite: bool,
    daughterboard_number: str,
    motherboard_number: str,
    firmware_version: str,
    timestamps: int = 1000,
    delta_window: float = 50e3,
    include_offset: bool = False,
    apply_calibration: bool = True,
    absolute_timestamps: bool = False,
    correct_pix_address: bool = False,
):
    """Collect timestamp differences from cross-talk data.

    Collect timestamp differences from cross-talk data for the given
    list of pixels.

    Parameters
    ----------
    path : str
        Path to the folder with '.dat' data files.
    pixels : List[int]
        List of pixels to collect timestamp differences for, where the
        first pixel is the aggressor.
    rewrite : bool
        Switch for rewriting the '.feather' file with timestamp
        differences. Used for conscious start of an hours-long data
        processing and avoiding rewriting/deleting already existing
        files.
    daughterboard_number : str
        LinoSPAD2 daughterboard number.
    motherboard_number : str
        LinoSPAD2 motherboard (FPGA) number, including the '#'.
    firmware_version : str
        LinoSPAD2 firmware version.
    timestamps : int, optional
        Number of timestamps per TDC per cycle used during data
        collection. The default is 1000.
    include_offset : bool
        Switch for including the offset calibration.
    delta_window : float, optional
        Window size for collecting timestamp differences. The default
        is 50e3.
    apply_calibration : bool, optional
        If True, apply calibration to the collected data. The default
        is True.
    absolute_timestamps : bool, optional
        Indicator of data collected with absolute timestamps. The
        default is False.
    correct_pix_address : bool, optional
        Correct pixel address for the FPGA board on side 23. Here
        used to reverse the correction. The default is False.

    Raises
    ------
    TypeError
        Raised if any of the following conditions are met:
        - `pixels` is not a list of integers,
        - `firmware_version` is not recognized,
        - `rewrite` is not boolean,
        - `daughterboard_number` is not a string.
    """

    # parameter type check
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

    # If requested, get the correct pixel address - must be used
    # for the motherboard on side '23'
    if correct_pix_address:
        for i, pixel in enumerate(pixels):
            if pixel > 127:
                pixels[i] = 255 - pixels[i]
            else:
                pixels[i] = pixels[i] + 128

    # files_all = sorted(glob.glob("*.dat*"))
    files_all = glob.glob("*.dat*")
    files_all.sort(key=os.path.getmtime)

    out_file_name = files_all[0][:-4] + "-" + files_all[-1][:-4]

    feather_file = os.path.join(
        path,
        "cross_talk_data",
        f"{out_file_name}_pixels_{pixels[0]}-{pixels[-1]}.feather",
    )

    utils.file_rewrite_handling(feather_file, rewrite)

    # Define matrix of pixel coordinates, where rows are numbers of TDCs
    # and columns are the pixels that connected to these TDCs
    if firmware_version == "2212s":
        pix_coor = np.arange(256).reshape(4, 64).T
    elif firmware_version == "2212b":
        print(
            "\nFor firmware version '2212b' cross-talk numbers "
            "would be incorrect, try data collected with '2212s'"
        )
        sys.exit()
    else:
        print("\nFirmware version is not recognized.")
        sys.exit()

    pixels_formatted = [[pixels[0]], pixels[1:]]

    for i in tqdm(range(len(files_all)), desc="Collecting data"):
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

        # Collect timestamp differences for the given pixels
        # deltas_all = cd.calculate_differences_2212(
        #     data_all, pixels_formatted, pix_coor, delta_window
        # )
        deltas_all = cd.calculate_differences_2212_fast(
            data_all, pixels_formatted, pix_coor, delta_window
        )

        # Save data as a .feather file in a cycle so data is not lost
        # in the case of failure close to the end
        data_for_plot_df = pd.DataFrame.from_dict(deltas_all, orient="index")
        del deltas_all
        data_for_plot_df = data_for_plot_df.T
        try:
            os.chdir("cross_talk_data")
        except FileNotFoundError:
            os.mkdir("cross_talk_data")
            os.chdir("cross_talk_data")

        # Check if feather file exists
        feather_file = (
            f"{out_file_name}_pixels_{pixels[0]}-{pixels[-1]}.feather"
        )
        if os.path.isfile(feather_file):
            # Load existing feather file
            existing_data = ft.read_feather(feather_file)

            # Append new data to the existing feather file
            combined_data = pd.concat(
                [existing_data, data_for_plot_df], axis=0
            )
            ft.write_feather(combined_data, feather_file)

        else:
            # Save as a new feather file
            ft.write_feather(data_for_plot_df, feather_file)
        os.chdir("..")

    # Check, if the file was created
    if (
        os.path.isfile(
            path + f"/cross_talk_data/{out_file_name}_pixels_"
            f"{pixels[0]}-{pixels[-1]}.feather"
        )
        is True
    ):
        print(
            "\n> > > Timestamp differences are saved as"
            f"{out_file_name}_pixels_{pixels[0]}-{pixels[-1]}.feather in "
            f"{os.path.join(path, 'cross_talk_data')} < < <"
        )

    else:
        print("File wasn't generated. Check input parameters.")


def _plot_cross_talk_peaks(
    path: str,
    pixels: List,
    multiplier: int,
    window: float = 50e3,
    senpop: list = None,
    pix_on_left: bool = False,
    feather_file_name: str = "",
    show_plots: bool = False,
):
    """Plot cross-talk peaks and calculate peak population.

    Plots a histogram of timestamp differences collected from the
    cross-talk data, fits using Gaussian to extract sigma and peak
    position. These two numbers are then used to calculate the peak
    population as numbers of data points in a +/- 2*sigma window
    around the peak position minus background in the same window.

    Parameters
    ----------
    path : str
        Path to the folder with the '.dat' data files.
    pixels : list
        List of pixel number to plot cross-talk peaks for. The first
        one is the aggressor pixel.
    multiplier : int
        Multiplier of the LinoSPAD2 average timing bin of 17.857 ps for
        histograms of the timestamp differences.
    window : float
        Window length for the histogram range. The default is 50e3.
    senpop : list
        List with numbers of timestamps per pixel.
    pix_on_left : bool, optional
        Indicator of pixels to the left of the aggressor. The default is False.
    feather_file_name : str, optional
        Naming pattern of the feather files that can be used when
        actual data files are not available. The pattern should
        include the number of the first and the last data files, e.g.,
        "0000010000-0000012000". The default is "".
    show_plots : bool, optional
        Switch for showing the plots at the end. The default is False.

    Returns
    -------
    dict, dict
        Two dictionaries with distances in pixels from the aggressor
        pixel, where the first dictionary contains cross-talk
        probabilities and the second - the corresponding errors.
    """

    os.chdir(path)
    if feather_file_name == "":
        files_all1 = glob.glob("*.dat")
        files_all1.sort(key=os.path.getmtime)
        ft_file_name = files_all1[0][:-4] + "-" + files_all1[-1][:-4]
    else:
        ft_file_name = feather_file_name

    os.chdir(os.path.join(path, "cross_talk_data"))

    ft_file = glob.glob(
        f"{ft_file_name}_*{pixels[0]}" f"-{pixels[-1]}*.feather"
    )[0]

    ct_output = {}
    ct_err_output = {}

    for _, pix in enumerate(pixels[1:]):
        if pix_on_left:

            data_pix = (
                ft.read_feather(ft_file, columns=[f"{pix},{pixels[0]}"])
                .dropna()
                .values
            )
        else:
            data_pix = (
                ft.read_feather(ft_file, columns=[f"{pixels[0]},{pix}"])
                .dropna()
                .values
            )

        data_cut = data_pix[(data_pix > -window / 2) & (data_pix < window / 2)]

        if not data_cut.any().any():
            continue
        counts, bin_edges = np.histogram(
            data_cut,
            bins=(
                np.arange(
                    np.min(data_cut), np.max(data_cut), multiplier * 2500 / 140
                )
            ),
        )
        if len(counts) < 1:
            continue
        bin_centers = (bin_edges - multiplier * 2500 / 140 / 2)[1:]

        try:
            params, _ = curve_fit(
                utils.gaussian,
                bin_centers,
                counts,
                p0=[
                    np.max(counts),
                    bin_centers[np.argmax(counts)],
                    100,
                    np.median(counts),
                ],
            )
        except (RuntimeError, TypeError):
            continue

        # Cross-talk probability estimation
        if senpop is not None:
            aggressor_pix_tmsps = senpop[pixels[0]]
            victim_pix_tmsps = senpop[pix]

            peak_population = data_cut[
                (data_cut > params[1] - params[2] * 2)
                & (data_cut < params[1] + params[2] * 2)
            ]
            bckg = data_cut[
                (data_cut > params[1] + 1e3 - params[2] * 2)
                & (data_cut < params[1] + 1e3 + params[2] * 2)
            ]
            peak_population_err = np.sqrt(len(peak_population))
            bckg_err = np.sqrt(len(bckg))

            CT = (
                (len(peak_population) - len(bckg))
                * 100
                / (aggressor_pix_tmsps + victim_pix_tmsps)
            )

            CT_err = (
                np.sqrt(peak_population_err**2 + bckg_err**2)
                * 100
                / (aggressor_pix_tmsps + victim_pix_tmsps)
            )

        plt.rcParams.update({"font.size": 27})
        fig = plt.figure(figsize=(16, 10))
        plt.step(bin_centers, counts, color="rebeccapurple", label="Data")
        plt.plot(
            bin_centers,
            utils.gaussian(bin_centers, *params),
            color="darkorange",
            label="Fit",
        )
        if senpop is not None:
            plt.title(
                f"Cross-talk peak, pixels {pixels[0]},{pix}\n"
                f"Cross-talk is {CT:.1e}" + "\u00B1" + f"{CT_err:.1e}" + "%"
            )
        else:
            plt.title(f"Cross-talk peak, pixels {pixels[0]},{pix}")

        plt.xlabel("\u0394t (ps)")
        plt.ylabel("# of coincidences (-)")
        plt.legend()
        try:
            os.chdir(os.path.join(path, "results/ct_fit"))
        except FileNotFoundError as _:
            os.makedirs(os.path.join(path, "results/ct_fit"))
            os.chdir(os.path.join(path, "results/ct_fit"))
        plt.savefig(f"CT_fit_pixels_{pixels[0]},{pix}.png")
        os.chdir(os.path.join(path, "cross_talk_data"))

        if show_plots:
            plt.show()
        else:
            plt.close(fig)

        ct_output[f"{pixels[0], pix}"] = CT
        ct_err_output[f"{pixels[0], pix}"] = CT_err

    return ct_output, ct_err_output


def _plot_cross_talk_grid(
    path,
    pixels,
    multiplier,
    window: int = 50e3,
    pix_on_left: bool = False,
    feather_file_name: str = "",
    show_plots: bool = False,
):
    """Plot grid of 4x5 of cross-talk peaks.

    Parameters
    ----------
    path : str
        Path to data files.
    pixels : list
        List of pixels to plot the grid for. The first is the aggressor
        pixel.
    multiplier : int
        Multiplier of the LinoSPAD2 average timing bin of 17.857 ps for
        the histograms.
    window : int, optional
        Window size for the histogram range. The default is 50e3.
    pix_on_left : bool, optional
        Indicator that pixels provided are to the left of the aggressor.
        The default is False.
    feather_file_name : str, optional
        Naming pattern of the feather files that can be used when
        actual data files are not available. The pattern should
        include the number of the first and the last data files, e.g.,
        "0000010000-0000012000". The default is "".
    show_plots : bool, optional
        Switch for showing the plots at the end. The default is False.
    """

    os.chdir(path)
    if feather_file_name == "":
        files_all1 = glob.glob("*.dat")
        files_all1.sort(key=os.path.getmtime)
        ft_file_name = files_all1[0][:-4] + "-" + files_all1[-1][:-4]
    else:
        ft_file_name = feather_file_name

    os.chdir(os.path.join(path, "cross_talk_data"))

    ft_file = glob.glob(f"{ft_file_name}_*{pixels[0]}-{pixels[-1]}*.feather")[
        0
    ]

    fig, axes = plt.subplots(4, 5, figsize=(16, 10))
    plt.rcParams.update({"font.size": 27})

    for i, pix in enumerate(pixels[1:]):
        if pix_on_left:
            data_pix = (
                ft.read_feather(ft_file, columns=[f"{pix},{pixels[0]}"])
                .dropna()
                .values
            )
        else:
            data_pix = (
                ft.read_feather(ft_file, columns=[f"{pixels[0]},{pix}"])
                .dropna()
                .values
            )

        data_cut = data_pix[(data_pix > -window / 2) & (data_pix < window / 2)]

        if not data_cut.any().any():
            continue
        counts, bin_edges = np.histogram(
            data_cut,
            bins=(
                np.arange(
                    np.min(data_cut), np.max(data_cut), multiplier * 2500 / 140
                )
            ),
        )
        if len(counts) < 1:
            continue
        bin_centers = (bin_edges - multiplier * 2500 / 140 / 2)[1:]

        try:
            params, _ = curve_fit(
                utils.gaussian,
                bin_centers,
                counts,
                p0=[
                    np.max(counts),
                    bin_centers[np.argmax(counts)],
                    100,
                    np.median(counts),
                ],
            )
        except (RuntimeError, TypeError):
            continue

        if i < 5:
            axes[0, i].plot(bin_centers, counts, ".", color="rebeccapurple")
            axes[0, i].plot(
                bin_centers,
                utils.gaussian(bin_centers, *params),
                "--",
                color="darkorange",
            )
            axes[0, i].set_xticks([])
            axes[0, i].set_yticks([])
            axes[0, i].set_title(f"{pixels[0]},{pix}")
        elif i >= 5 and i < 10:
            axes[1, i % 5].plot(
                bin_centers, counts, ".", color="rebeccapurple"
            )
            axes[1, i % 5].plot(
                bin_centers,
                utils.gaussian(bin_centers, *params),
                "--",
                color="darkorange",
            )
            axes[1, i % 5].set_xticks([])
            axes[1, i % 5].set_yticks([])
            axes[1, i % 5].set_title(f"{pixels[0]},{pix}")
        elif i >= 10 and i < 15:
            axes[2, i % 5].plot(
                bin_centers, counts, ".", color="rebeccapurple"
            )
            axes[2, i % 5].plot(
                bin_centers,
                utils.gaussian(bin_centers, *params),
                "--",
                color="darkorange",
            )
            axes[2, i % 5].set_xticks([])
            axes[2, i % 5].set_yticks([])
            axes[2, i % 5].set_title(f"{pixels[0]},{pix}")
        else:
            axes[3, i % 5].plot(
                bin_centers, counts, ".", color="rebeccapurple"
            )
            axes[3, i % 5].plot(
                bin_centers,
                utils.gaussian(bin_centers, *params),
                "--",
                color="darkorange",
            )
            axes[3, i % 5].set_xticks([])
            axes[3, i % 5].set_yticks([])
            axes[3, i % 5].set_title(f"{pixels[0]},{pix}")

    axes[1, 2].set_xlabel("\u0394t (ps)", fontsize=26)
    fig.text(0, 0.25, "# of coincidences (-)", fontsize=26, rotation=90)

    # Make plots tight
    plt.tight_layout()

    try:
        os.chdir(os.path.join(path, "results/ct_fit"))
    except FileNotFoundError:
        os.makedirs(os.path.join(path, "results/ct_fit"))
        os.chdir(os.path.join(path, "results/ct_fit"))
    plt.savefig(f"CT_fit_pixels_{pixels[0]},{pix}_grid.png")
    os.chdir(os.path.join(path, "cross_talk_data"))

    if show_plots:
        plt.show()
    else:
        plt.close(fig)


def collect_dcr_by_file(
    path: str,
    daughterboard_number: str,
    motherboard_number: str,
    firmware_version: str,
    timestamps: int = 1000,
):
    """Calculate dark count rate in counts per second per pixel.

    Calculate dark count rate for the given daughterboard and
    motherboard in units of counts per second per pixel for each file
    in the given folder and save the resulting list as a .pkl file.

    Parameters
    ----------
    path : str
        Path to the folder with the '.dat' data files.
    daughterboard_number : str
        LinoSPAD2 daughterboard number.
    motherboard_number : str
        LinoSPAD2 motherboard (FPGA) number, including the '#'.
    firmware_version : str
        LinoSPAD2 firmware version.
    timestamps : int, optional
        Number of timestamps per acquisition cycle per TDC. The default
        is 1000.

    Raises
    ------
    TypeError
        Raised if the firmware version given is not recognized.
    TypeError
        Raised if the daughterboard number given is not recognized.
    TypeError
        Raised if the motherboard number given is not recognized.
    """

    # Parameter type check
    if not isinstance(firmware_version, str):
        raise TypeError(
            "'firmware_version' should be a string, '2212b' or '2212s'"
        )
    if not isinstance(daughterboard_number, str):
        raise TypeError(
            "'daughterboard_number' should be a string, 'NL11' or 'A5'"
        )
    if not isinstance(motherboard_number, str):
        raise TypeError("'motherboard_number' should be a string")

    os.chdir(path)

    files = glob.glob("*.dat*")
    files = sorted(files)

    output_file_name = files[0][:-4] + "-" + files[-1][:-4]

    valid_per_pixel = np.zeros(256)

    dcr = []

    for i in tqdm(range(len(files)), desc="Going through files"):
        # Define matrix of pixel coordinates, where rows are numbers of TDCs
        # and columns are the pixels that connected to these TDCs
        if firmware_version == "2212s":
            pix_coor = np.arange(256).reshape(4, 64).T
        elif firmware_version == "2212b":
            pix_coor = np.arange(256).reshape(64, 4)
        else:
            print("\nFirmware version is not recognized.")
            sys.exit()

        # Unpack the data; offset calibration is not necessary
        data = f_up.unpack_binary_data(
            files[i],
            daughterboard_number,
            motherboard_number,
            firmware_version,
            timestamps,
            include_offset=False,
        )

        # Collect number of timestamps in each pixel - DCR
        for i in range(256):
            tdc, pix = np.argwhere(pix_coor == i)[0]
            ind = np.where(data[tdc].T[0] == pix)[0]
            ind1 = np.where(data[tdc].T[1][ind] > 0)[0]
            valid_per_pixel[i] = len(data[tdc].T[1][ind[ind1]])

        acq_window_length = np.max(data[:].T[1]) * 1e-12
        number_of_cycles = len(np.where(data[0].T[0] == -2)[0])

        dcr.append(valid_per_pixel / acq_window_length / number_of_cycles)

    dcr = np.array(dcr)

    # Save the results into a .pkl file
    try:
        os.chdir("dcr_data")
    except FileNotFoundError:
        os.mkdir("dcr_data")
        os.chdir("dcr_data")

    file_with_dcr = f"{output_file_name}_dcr_data.pkl"

    with open(file_with_dcr, "wb") as f:
        pickle.dump(dcr, f)

    # Check, if the file was created
    absolute_address = os.path.join(path, "dcr_data", file_with_dcr)
    if os.path.isfile(absolute_address) is True:
        print(f"\n> > > DCR data are saved in {absolute_address} < < <")
    else:
        print("File wasn't generated. Check input parameters.")

    # Average and median DCR, including the hot pixels
    dcr_average = np.average(dcr)
    dcr_median = np.median(dcr)

    # Average and median DCR, with the hot pixels masked
    mask = utils.apply_mask(daughterboard_number, motherboard_number)
    dcr[:, mask] = 0
    dcr_average_masked = np.average(dcr)
    dcr_median_masked = np.median(dcr)

    print(
        "\n".join(
            [
                "DCR including hot pixels:",
                f"DCR Average: {dcr_average:.0f} cps/pixel",
                f"DCR Median: {dcr_median:.0f} cps/pixel",
                "                                                 ",
                "DCR without hot pixels:",
                f"DCR Average: {dcr_average_masked:.0f} cps/pixel",
                f"DCR Median: {dcr_median_masked:.0f} cps/pixel",
            ]
        )
    )


def plot_dcr_histogram_and_stability(
    path: str,
    hist_number_of_bins: int = 200,
):
    """Plot median DCR vs file and histogram of DCR with integral.

    Plot median DCR vs file for checking the setup stability. Also, plot
    a histogram of DCR together with intergral over pixels in units of
    %.

    Parameters
    ----------
    path : str
        Path to the folder with '.dat' data files.
    hist_number_of_bins : int, optional
        Number of bins for the DCR histogram. The default is 200.

    Raises
    ------
    FileNotFoundError
        Raised if the folder with the DCR data was not found.
    FileNotFoundError
        Raised if the .pkl file with the DCR data was not found.
    """

    os.chdir(path)

    # Collect all files in the given folder
    files = glob.glob("*.dat")
    files = sorted(files)

    # Find the file with the DCR data for the found files
    dcr_file_name = files[0][:-4] + "-" + files[-1][:-4]
    file_with_dcr = f"{dcr_file_name}_dcr_data.pkl"

    try:
        os.chdir("dcr_data")
    except FileNotFoundError:
        raise FileNotFoundError("The folder with DCR data was not found.")

    try:
        with open(file_with_dcr, "rb") as f:
            data = pickle.load(f)
    except FileNotFoundError:
        raise FileNotFoundError(f"{file_with_dcr} was not found")

    # Median DCR over the whole sensor
    dcr_median = np.median(data)

    # Plot the DCR stability graph: median DCR vs file
    plt.rcParams.update({"font.size": 27})
    plt.figure(figsize=(16, 10))
    plt.plot(
        [x + 1 for x in range(len(data))],
        np.median(data, axis=1),
        color="darkslateblue",
        label=f"Median DCR: {dcr_median:.0f} cps/pixel",
    )
    plt.title("DCR stability")
    plt.xlabel("File (-)")
    plt.ylabel("Median DCR (cps)")
    plt.legend(loc="best")

    # Save the plot to "results/dcr"
    try:
        os.chdir("../results/dcr")
    except FileNotFoundError:
        os.makedirs("../results/dcr")
        os.chdir("../results/dcr")
    plt.savefig("DCR_stability_graph.png")

    # DCR histogram with integral
    # Compute the histogram
    bins = np.logspace(
        np.log10(0.1), np.log10(np.max(data[0])), hist_number_of_bins
    )
    hist, bin_edges = np.histogram(np.average(data, axis=0), bins=bins)
    bin_centers = (bin_edges - (bin_edges[1] - bin_edges[0]) / 2)[1:]

    # Plot the histogram
    fig, ax = plt.subplots(figsize=(16, 10))
    ax.bar(
        bin_centers,
        hist,
        width=np.diff(bin_edges),
        color="rebeccapurple",
    )

    # Calculate and plot the integral
    cumul = np.cumsum(hist)
    ax1 = ax.twinx()
    ax1.plot(bin_centers, cumul / 256 * 100, color="darkorange", linewidth=3)
    ax.set_xlim(10)
    ax1.set_xlim(10)
    ax.set_ylim(0)
    ax1.set_ylim(0)
    ax.set_xscale("log")
    ax.set_xlabel("DCR (cps/pixel)")
    ax.set_ylabel("Count (-)")
    ax1.set_ylabel("Integral (%)")
    ax.set_title(f"Median DCR: {dcr_median:.0f} cps/pixel")
    # plt.show()

    # Save the plot to "results/dcr"
    fig.savefig("DCR_histogram_w_integral.png")


def _calculate_and_plot_cross_talk(
    path: str,
    pixels: List,
    multiplier: int,
    delta_window: float,
    senpop,
    pix_on_left: bool = False,
    feather_file_name: str = "",
    show_plots: bool = False,
):
    """Plot cross-talk peaks and calculate peak population.

    Plots a histogram of timestamp differences collected from the
    cross-talk data, fits using Gaussian to extract sigma and peak
    position. These two numbers are then used to calculate the peak
    population as numbers of data points in a +/- 2*sigma window
    around the peak position minus background in the same window.

    Parameters
    ----------
    path : str
        Path to the folder with the '.dat' data files.
    pixels : List[int]
        List of lists with pixels numbers. Each list starts with the
        aggressor pixel plus 20 pixels either on the left or right.
    multiplier : int
        Multiplier of the LinoSPAD2 average timing bin of 17.857 ps for
        histograms of the timestamp differences.
    delta_window : float
        Length of the window for the histogram range.
    senpop : list
        List with numbers of timestamps per pixel.
    pix_on_left : bool, optional
        Indicator of pixels to the left of the aggressor. The default is False.
    feather_file_name : str, optional
        Naming pattern of the feather files that can be used when
        actual data files are not available. The pattern should
        include the number of the first and the last data files, e.g.,
        "0000010000-0000012000". The default is "".
    show_plots : bool, optional
        Switch for showing the plots at the end. The default is False.

    Returns
    -------
    dict, dict
        Two dictionaries with distances in pixels from the aggressor
        pixel, where the first dictionary contains cross-talk
        probabilities and the second - the corresponding errors.
    """

    ct = []
    ct_err = []

    for pix in pixels:
        CT, CT_err = _plot_cross_talk_peaks(
            path,
            pixels=pix,
            multiplier=multiplier,
            window=delta_window,
            senpop=senpop,
            pix_on_left=pix_on_left,
            feather_file_name=feather_file_name,
            show_plots=show_plots,
        )
        _plot_cross_talk_grid(
            path,
            pixels=pix,
            multiplier=multiplier,
            window=delta_window,
            pix_on_left=pix_on_left,
            feather_file_name=feather_file_name,
            show_plots=show_plots,
        )

        ct.append(CT)
        ct_err.append(CT_err)

    return ct, ct_err


def _plot_cross_talk_vs_distance(
    path, ct, ct_err, pix_on_left: bool = False, show_plots: bool = False
):
    """Plot cross-talk vs distance from the aggressor.

    Plot cross-talk probability as a function of distance from the
    aggressor pixel.

    Parameters
    ----------
    path : str
        Path to the folder with the '.dat' data files.
    ct : dict
        Dictionary of cross-talk probabilities with distance from the
        aggressor as keys.
    ct_err : dict
        Dictionary of cross-talk error values with distance from the
        aggressor as keys.
    pix_on_left : bool, optional
        Indicator of pixels to the left of the aggressor. The default is False.
    show_plots : bool, optional
        Switch for showing the plots at the end. The default is False.
    """

    try:
        os.chdir(os.path.join(path, "ct_vs_distance"))
    except FileNotFoundError:
        os.makedirs(os.path.join(path, "ct_vs_distance"))
        os.chdir(os.path.join(path, "ct_vs_distance"))

    for _, (CT, CT_err) in enumerate(zip(ct, ct_err)):
        if CT == {} or CT_err == {}:
            continue

        differences = []
        keys = CT.keys()

        for key in keys:
            # Convert the key to a tuple
            key_tuple = eval(key)
            # Extract the difference and append it to the list
            differences.append(key_tuple[1] - key_tuple[0])

        fig = plt.figure(figsize=(16, 10))
        plt.rcParams.update({"font.size": 27})
        plt.errorbar(
            differences,
            list(CT.values()),
            list(CT_err.values()),
            fmt=".",
            color="indianred",
        )
        aggressor_pix = int(list(CT.keys())[0].split(",")[0].split("(")[1])
        plt.title(
            f"Cross-talk probability for aggressor pixel {aggressor_pix}"
        )
        plt.xlabel("Distance in pixels (-)")
        plt.ylabel("Cross-talk probability (%)")
        if pix_on_left:
            plt.savefig(
                f"Cross-talk_aggressor_pixel_{aggressor_pix}_onleft.png"
            )
            with open(
                f"Cross-talk_aggressor_pixel_{aggressor_pix}_onleft.pkl", "wb"
            ) as f:
                pickle.dump(fig, f)
        else:
            plt.savefig(
                f"Cross-talk_aggressor_pixel_{aggressor_pix}_onright.png"
            )
            with open(
                f"Cross-talk_aggressor_pixel_{aggressor_pix}_onright.pkl", "wb"
            ) as f:
                pickle.dump(fig, f)

        if show_plots:
            plt.show()
        else:
            plt.close(fig)


def _plot_average_cross_talk_vs_distance(
    path, ct, ct_err, pix_on_left: bool = False
):
    """Plot average cross-talk vs distance from the aggressor pixel.

    Plot average cross-talk probability vs distance from the aggressor
    averaging accross the whole sensor and all hot pixels byt dividing
    between either to the left or to the right of the aggressor the
    cross-talk is calculated.

    Parameters
    ----------
    path : str
        Path to the folder with the '.dat' data files.
    ct : dict
        Dictionary of cross-talk probabilities for each distance from 1 to 20.
    ct_err : dict
        Dictionary of cross-talk error values for each distance from 1 to 20.
    pix_on_left : bool, optional
        Indicator of pixels to the left of the aggressor. The default is False.

    Returns
    -------
    dict
        Dictionary of average cross-talk numbers and error with distance
        from the aggressor as keys.
    """

    try:
        os.chdir(os.path.join(path, "ct_vs_distance"))
    except FileNotFoundError:
        os.makedirs(os.path.join(path, "ct_vs_distance"))
        os.chdir(os.path.join(path, "ct_vs_distance"))

    if pix_on_left:
        final_result = {key: [] for key in range(-20, 0)}
        final_result_averages = {key: [] for key in range(-20, 0)}
    else:
        final_result = {key: [] for key in range(1, 21)}
        final_result_averages = {key: [] for key in range(1, 21)}

    for _, (ct_pick, ct_err_pick) in enumerate(zip(ct, ct_err)):
        if ct_pick == {} or ct_err_pick == {}:
            continue

        for key in ct_pick.keys():
            key_tuple = eval(key)

            key_difference = key_tuple[1] - key_tuple[0]
            final_result[key_difference].append(
                (ct_pick[key], ct_err_pick[key])
            )

    for key in final_result.keys():
        value = np.average([x[0] for x in final_result[key]])
        error = np.sqrt(np.sum([x[1] ** 2 for x in final_result[key]])) / len(
            final_result[key]
        )
        final_result_averages[key].append((value, error))

    fig = plt.figure(figsize=(16, 10))
    plt.rcParams.update({"font.size": 27})
    plt.title("Average cross-talk probability")
    plt.xlabel("Distance in pixels (-)")
    plt.ylabel("Cross-talk probability (%)")
    plt.errorbar(
        final_result_averages.keys(),
        [x[0][0] for x in final_result_averages.values()],
        yerr=[x[0][1] for x in final_result_averages.values()],
        fmt=".",
        color="darkred",
    )
    if pix_on_left:
        plt.savefig("Average_cross-talk_onleft.png")
        with open("Average_cross-talk_onleft.pkl", "wb") as f:
            pickle.dump(fig, f)
    else:
        plt.savefig("Average_cross-talk_onright.png")
        with open("Average_cross-talk_onright.pkl", "wb") as f:
            pickle.dump(fig, f)

    return final_result_averages


def zero_to_cross_talk_collect(
    path: str,
    pixels: List[int],
    rewrite: bool,
    daughterboard_number: str,
    motherboard_number: str,
    firmware_version: str,
    timestamps: int,
    include_offset: bool = False,
    delta_window: float = 50e3,
    apply_calibration: bool = True,
    absolute_timestamps: bool = False,
    correct_pix_address: bool = False,
):
    """Collect timestamp differences from cross-talk data.

    For the given list of hot pixels, collect all timestamp differences
    for pairs of pixels as aggressor-victim, where the victim pixel is
    taken from each side of the aggressor (where applicable) and up to
    the distance of 20 pixels (e.g., 100,120).

    Parameters
    ----------
    path : str
        Path to the folder with the '.dat' data files.
    pixels : list
        List of pixels to use as aggressors. These can be hot pixels
        or pixels where the signal is focused.
    rewrite : bool
        Switch for rewriting the feather file with timestamp differences.
        Used for conscious start of an hours-long data processing and
        avoiding rewriting/deleting already existing files.
    daughterboard_number : str
        LinoSPAD2 daughterboard number.
    motherboard_number : str
        LinoSPAD2 motherboard (FPGA) number, including the '#'.
    firmware_version : str
        LinoSPAD2 firmware version.
    timestamps : int
        Number of timestamps per TDC per cycle used during data
        collection.
    include_offset : bool
        Switch for including the offset calibration. The default is
        False.
    delta_window : float, optional
        Window size for collecting timestamp differences. The default
        is 50e3.
    apply_calibration : bool, optional
        Switch for applying callibration. The default is True.
    absolute_timestamps : bool, optional
        Indicator of data collected with absolute timestamps. The
        default is False.
    correct_pix_address : bool, optional
        Correct pixel address for the FPGA board on side 23. Here
        used to reverse the correction. The default is False.
    """

    # Define matrix of pixel coordinates, where rows are numbers of TDCs
    # and columns are the pixels that connected to these TDCs
    if firmware_version == "2212s":
        pass
    elif firmware_version == "2212b":
        print(
            "\nFor firmware version '2212b' cross-talk numbers "
            "would be incorrect, try data collected with '2212s'"
        )
        sys.exit()
    else:
        print(
            "\nFirmware version is not recognized. For cross-talk "
            "calculations, firmware version '2212s' should be used."
        )
        sys.exit()

    print("\n> > > Collecting cross-talk data < < <\n")

    # Reverse correction if the motherboard connected to side "23" of the
    # daughterboard
    if correct_pix_address:
        for i, pixel in enumerate(pixels):
            if pixel > 127:
                pixels[i] = 255 - pixel
            else:
                pixels[i] = pixel + 128

    pixels_plus_20 = [
        [x + i for i in range(0, 21)] for x in pixels if x <= 235
    ]
    pixels_minus_20 = [
        [x - i for i in range(0, 21)] for x in pixels if x >= 20
    ]

    # Collecting sensor population
    os.chdir(path)
    files = glob.glob("*.dat")
    sensor_plot.collect_data_and_apply_mask(
        files,
        daughterboard_number,
        motherboard_number,
        firmware_version,
        timestamps,
        app_mask=False,
        save_to_file=True,
        absolute_timestamps=absolute_timestamps,
        correct_pix_address=correct_pix_address,
    )

    for pixels in pixels_plus_20:
        print(
            "Calculating timestamp differences between aggressor pixel "
            f"{pixels[0]} and pixels to the right"
        )
        _collect_cross_talk(
            path,
            pixels,
            rewrite,
            daughterboard_number,
            motherboard_number,
            firmware_version,
            timestamps,
            delta_window,
            include_offset,
            apply_calibration,
            absolute_timestamps,
            correct_pix_address,
        )

    for pixels in pixels_minus_20:
        print(
            "Calculating timestamp differences between aggressor pixel "
            f"{pixels[0]} and pixels to the left"
        )
        _collect_cross_talk(
            path,
            pixels,
            rewrite,
            daughterboard_number,
            motherboard_number,
            firmware_version,
            timestamps,
            delta_window,
            include_offset,
            apply_calibration,
            absolute_timestamps,
            correct_pix_address,
        )


def zero_to_cross_talk_plot(
    path: str,
    pixels: List[int],
    delta_window: float = 50e3,
    multiplier: int = 10,
    show_plots: bool = False,
    feather_file_name: str = "",
):
    """Plot cross-talk peaks and as average vs distance from aggressor.

    For the given pixels, look for the timestamp differences in
    feather files, plot histograms and fits with Gaussian of cross-talk
    peaks for each pair of pixels (aggressor-victim), collects
    cross-talk peak population in a 2*sigma window, plots as average
    vs distance from the aggressor for each aggressor pixel and to each
    side (to the left and to the right). Finally, averages all numbers
    and produces a single plot of average cross-talk vs distance from
    the aggressor across all hot pixels.

    Parameters
    ----------
    path : str
        Path to the folder with the '.dat' data files.
    pixels : List[int]
        List of pixels to use as aggressors. These can be hot pixels
        or pixels where the signal is focused.
    delta_window : float, optional
        Window size for histograms of timestamp differences. The default
        is 50e3.
    multiplier : int, optional
        Multiplier of average LinoSPAD2 time bin, which is 17.857 ps,
        for histograms. The default is 10.
    show_plots : bool, optional
        Switch for showing the plots at the end. The default is False.
    feather_file_name : str, optional
        Naming pattern of the feather files that can be used when
        actual data files are not available. The pattern should
        include the number of the first and the last data files, e.g.,
        "0000010000-0000012000". The default is "".

    Raises
    ------
    FileNotFoundError
        Raised when no txt file with the sensor population data is
        found.
    """

    print("\n> > > Plotting cross-talk peaks and averages < < <\n")
    try:
        os.chdir(os.path.join(path, "ct_vs_distance"))
    except FileNotFoundError:
        os.makedirs(os.path.join(path, "ct_vs_distance"))
        os.chdir(os.path.join(path, "ct_vs_distance"))

    pixels_plus_20 = [
        [x + i for i in range(0, 21)] for x in pixels if x <= 235
    ]
    pixels_minus_20 = [
        [x - i for i in range(0, 21)] for x in pixels if x >= 20
    ]
    try:
        os.chdir(os.path.join(path, "senpop_data"))
        senpop_data_txt = glob.glob("*.txt")[0]
        senpop = np.genfromtxt(senpop_data_txt)
    except Exception as _:
        raise FileNotFoundError(
            "Txt file with sensor population data is not found. Collect "
            "sensor population first."
        )

    ct_right, ct_err_right = _calculate_and_plot_cross_talk(
        path,
        pixels_plus_20,
        multiplier,
        delta_window,
        senpop,
        pix_on_left=False,
        feather_file_name=feather_file_name,
        show_plots=show_plots,
    )
    ct_left, ct_err_left = _calculate_and_plot_cross_talk(
        path,
        pixels_minus_20,
        multiplier,
        delta_window,
        senpop,
        pix_on_left=True,
        feather_file_name=feather_file_name,
        show_plots=show_plots,
    )

    _plot_cross_talk_vs_distance(
        path,
        ct_right,
        ct_err_right,
        pix_on_left=False,
        show_plots=show_plots,
    )
    _plot_cross_talk_vs_distance(
        path,
        ct_left,
        ct_err_left,
        pix_on_left=True,
        show_plots=show_plots,
    )

    averages_right = _plot_average_cross_talk_vs_distance(
        path,
        ct_right,
        ct_err_right,
        pix_on_left=False,
    )
    averages_left = _plot_average_cross_talk_vs_distance(
        path,
        ct_left,
        ct_err_left,
        pix_on_left=True,
    )

    on_both_average = {key: [] for key in range(1, 21)}
    for key in averages_left.keys():
        ct_value_average = (
            averages_left[key][0][0] + averages_right[np.abs(key)][0][0]
        ) / 2
        ct_error_average = (
            np.sqrt(
                averages_left[key][0][1] ** 2
                + averages_right[np.abs(key)][0][1] ** 2
            )
            / 2
        )
        on_both_average[np.abs(key)] = (ct_value_average, ct_error_average)

    fig = plt.figure(figsize=(16, 10))
    plt.rcParams.update({"font.size": 27})
    plt.title("Average cross-talk probability")
    plt.xlabel("Distance in pixels (-)")
    plt.ylabel("Cross-talk probability (%)")
    plt.yscale("log")
    plt.errorbar(
        on_both_average.keys(),
        [x[0] for x in on_both_average.values()],
        yerr=[x[1] for x in on_both_average.values()],
        fmt=".",
        color="darkred",
        label=f"Immediate neighbor: {on_both_average[1][0]:.2f}%",
    )
    plt.tight_layout()
    os.chdir(os.path.join(path, "ct_vs_distance"))
    plt.legend(loc="best")
    plt.xticks(list(on_both_average.keys()), list(on_both_average.keys()))
    plt.savefig("Average_cross-talk.png")
    with open("Average_cross-talk.pkl", "wb") as f:
        pickle.dump(fig, f)

    return on_both_average, ct_right, ct_left


def unpickle_cross_talk(
    pkl_file: str,
):
    """Unpickle a saved cross-talk plot and return data.

    Load the pickled '.pkl' file with the figure, return the figure,
    extract the data points and errorbars for y and return them.

    Parameters
    ----------
    pkl_file : str
        Absolute path to the '.pkl' file with the cross-talk plot.

    Returns
    -------
    tuple of numpy.ndarray
        A tuple containing three numpy arrays:
        - x : ndarray
            The x-coordinates of the data points.
        - y : ndarray
            The y-coordinates of the data points.
        - yerr : ndarray
            The y-error values associated with each data point.
    """

    # Read the '.pkl' file
    with open(pkl_file, "rb") as f:
        fig = pickle.load(f)

    # Get the axes
    ax = fig.gca()

    # Collect all objects from the axes
    children = ax.get_children()

    # Loop through the children to find the LineCollection (error bars)
    # and Line2D (data points)
    for child in children:
        if isinstance(child, matplotlib.collections.LineCollection):
            error_lines = child
        elif isinstance(child, matplotlib.lines.Line2D):
            x = child.get_xdata()
            y = child.get_ydata()

    # Check if error_lines is found and extract data
    if error_lines:
        # Extract error bar line segments
        segments = error_lines.get_segments()
    else:
        print("No error bars found.")

    # Recalculate the error for y from the pickled data
    # yerr = [(x[1][1] - y) for (x, y) in zip(segments, y)]

    yerr = np.zeros(len(x))

    for i, (seg, y_slice) in enumerate(zip(segments, y)):
        if seg.size > 0:
            yerr[i] = seg[1][1] - y_slice

    return x, y, yerr
