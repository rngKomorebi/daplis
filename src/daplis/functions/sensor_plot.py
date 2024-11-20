"""
Module with scripts for plotting the LinoSPAD2 sensor population.

This script utilizes an unpacking module used specifically for the
LinoSPAD2 data output.

This file can also be imported as a module and contains the following
functions:

    * collect_data_and_apply_mask - Collect data from files and apply
    mask to the valid pixel count.

    * plot_single_pix_hist - Plot a histogram for each pixel in the given
    range.

    * plot_sensor_population - Plot number of timestamps in each pixel
    for all data files.

    * plot_sensor_population_spdc - Plot sensor population for SPDC data.

    * plot_sensor_population_full_sensor - Plot the number of timestamps
    in each pixel for all data files from two different FPGAs/sensor
    halves.
"""

import glob
import os
import pickle
import sys
from typing import List

import numpy as np
from matplotlib import pyplot as plt
from scipy.optimize import curve_fit
from scipy.signal import find_peaks
from tqdm import tqdm

from daplis.functions import unpack as f_up
from daplis.functions import utils


def collect_data_and_apply_mask(
    files: List[str],
    daughterboard_number: str,
    motherboard_number: str,
    firmware_version: str,
    timestamps: int,
    app_mask: bool = True,
    absolute_timestamps: bool = False,
    save_to_file: bool = False,
    correct_pix_address: bool = False,
    calculate_rates: bool = False,
) -> np.ndarray:
    """Collect data from files and apply mask to the valid pixel count.

    Unpacks data and returns the number of timestamps in each pixel.
    This function introduces modularity to the whole module and is
    called multiple times here.

    Parameters
    ----------
    files : List[str]
        List of data file paths.
    daughterboard_number : str
        The LinoSPAD2 daughterboard number.
    motherboard_number : str
        The LinoSPAD2 motherboard number, including the "#".
    firmware_version : str
        LinoSPAD2 firmware version.
    timestamps : int
        Number of timestamps per pixel per acquisition cycle.
    app_mask : bool, optional
        Switch for applying the mask on warm/hot pixels. Default is True.
    absolute_timestamps : bool, optional
        Indicator for data files with absolute timestamps. Default is
        False.
    correct_pix_address : bool, optional
        Check for correcting the pixel addresing. THe default is False.
    calculate_rates : bool, optional
        Switch for calculating the photon rate for each pixel. The
        default is 'False'.
    Returns
    -------
    np.ndarray
        Array with the number of timestamps per pixel.
    """
    # Define matrix of pixel coordinates, where rows are numbers of TDCs
    # and columns are the pixels that connected to these TDCs
    if firmware_version == "2212s":
        pix_coor = np.arange(256).reshape(4, 64).T
    elif firmware_version == "2212b":
        pix_coor = np.arange(256).reshape(64, 4)
    else:
        print("\nFirmware version is not recognized.")
        sys.exit()

    timestamps_per_pixel = np.zeros(256)

    # In the case a single file is passed, make a list out of it
    if isinstance(files, str):
        files = [files]

    for i in tqdm(range(len(files)), desc="Collecting data"):
        if not absolute_timestamps:
            data = f_up.unpack_binary_data(
                files[i],
                daughterboard_number,
                motherboard_number,
                firmware_version,
                timestamps,
                include_offset=False,
                apply_calibration=False,
            )
        else:
            data, _ = f_up.unpack_binary_data_with_absolute_timestamps(
                files[i],
                daughterboard_number,
                motherboard_number,
                firmware_version,
                timestamps,
                include_offset=False,
                apply_calibration=False,
            )
        for i in range(256):
            tdc, pix = np.argwhere(pix_coor == i)[0]
            ind = np.where(data[tdc].T[0] == pix)[0]
            ind1 = np.where(data[tdc].T[1][ind] > 0)[0]
            timestamps_per_pixel[i] += len(data[tdc].T[1][ind[ind1]])

    if correct_pix_address:
        fix = np.zeros(len(timestamps_per_pixel))
        fix[:128] = timestamps_per_pixel[128:]
        fix[128:] = np.flip(timestamps_per_pixel[:128])
        timestamps_per_pixel = fix
        del fix

    # Apply mask if requested
    if app_mask:
        mask = utils.apply_mask(daughterboard_number, motherboard_number)
        timestamps_per_pixel[mask] = 0

    acq_window_length = np.max(data[:].T[1]) * 1e-12
    number_of_cycles = len(np.where(data[0].T[0] == -2)[0])

    rates = (
        timestamps_per_pixel
        / acq_window_length
        / number_of_cycles
        / len(files)
    )

    if save_to_file:
        files.sort(key=os.path.getmtime)
        file_name = files[0][:-4] + "-" + files[-1][:-4]
        try:
            os.chdir("senpop_data")
        except FileNotFoundError as _:
            os.mkdir("senpop_data")
            os.chdir("senpop_data")

        np.savetxt(f"{file_name}_senpop_numbers.txt", timestamps_per_pixel)
        os.chdir("..")

    if calculate_rates:
        return timestamps_per_pixel, rates
    else:
        return timestamps_per_pixel


def plot_single_pix_hist(
    path,
    pixels,
    daughterboard_number: str,
    motherboard_number: str,
    firmware_version: str,
    timestamps: int = 512,
    cycle_length: float = 4e9,
    multiplier: int = 1e6,
    show_fig: bool = False,
    fit_average: bool = False,
    color: str = "teal",
):
    """Plot a histogram for each pixel in the given range.

    Used mainly for checking the homogeneity of the LinoSPAD2 output
    (mainly clock and acquisition window size settings).

    Parameters
    ----------
    path : str
        Path to data file.
    pixels : array-like, list
        Array of pixels indices.
    daughterboard_number : str
        LinoSPAD2 daughterboard number.
    motherboard_number : str
        LinoSPAD2 motherboard (FPGA) number, including the '#'.
    firmware_version : str
        LinoSPAD2 firmware version.
    timestamps : int, optional
        Number of timestamps per pixel per acquisition cycle. The
        default is 512.
    cycle_length : float, optional
        Length of the data acquisition cycle. The default is 4e9, or 4 ms.
    multiplier : int, optional
        Multiplier of 17.857 for the bin size of the histogram. The
        default is 1e6.
    show_fig : bool, optional
        Switch for showing the output figure. The default is False.
    fit_average : int, optional
        Switch for fitting averages of histogram counts in windows of
        +/-10. The default is False.
    color : str, optional
        Color of the histogram. The default is 'teal'.

    Returns
    -------
    None.

    """
    # parameter type check
    if isinstance(firmware_version, str) is not True:
        raise TypeError("'firmware_version' should be a string")
    if isinstance(daughterboard_number, str) is not True:
        raise TypeError("'daughterboard_number' should be a string")
    if isinstance(motherboard_number, str) is not True:
        raise TypeError("'motherboard_number' should be a string")

    def lin_fit(x, a, b):
        return a * x + b

    if type(pixels) is int:
        pixels = [pixels]

    os.chdir(path)

    # data_files = glob.glob("*.dat*")
    data_files = glob.glob("*.dat*")
    data_files.sort(key=os.path.getmtime)

    if show_fig is True:
        plt.ion()
    else:
        plt.ioff()
    for i, num in enumerate(data_files):
        print(f"> > > Plotting pixel histograms, Working on {num} < < <\n")

        data = f_up.unpack_binary_data(
            num,
            daughterboard_number,
            motherboard_number,
            firmware_version,
            timestamps,
            include_offset=False,
            apply_calibration=False,
        )

        bins = np.arange(
            0, cycle_length, 2500 / 140 * multiplier
        )  # bin size of 17.867 us

        if pixels is None:
            pixels = np.arange(145, 165, 1)

        for i, _ in enumerate(pixels):
            plt.figure(figsize=(16, 10))
            plt.rcParams.update({"font.size": 27})
            # Define matrix of pixel coordinates, where rows are numbers
            # of TDCs and columns are the pixels that connected to
            # these TDCs
            if firmware_version == "2212s":
                pix_coor = np.arange(256).reshape(4, 64).T
            elif firmware_version == "2212b":
                pix_coor = np.arange(256).reshape(64, 4)
            else:
                print("\nFirmware version is not recognized.")
                sys.exit()
            tdc, pix = np.argwhere(pix_coor == pixels[i])[0]
            ind = np.where(data[tdc].T[0] == pix)[0]
            ind1 = np.where(data[tdc].T[1][ind] > 0)[0]
            data_to_plot = data[tdc].T[1][ind[ind1]]

            n, b, p = plt.hist(data_to_plot, bins=bins, color=color)
            if fit_average is True:
                av_win = np.zeros(int(len(n) / 10) + 1)
                av_win_in = np.zeros(int(len(n) / 10) + 1)
                for j, _ in enumerate(av_win):
                    av_win[j] = n[j * 10 : j * 10 + 1]
                    av_win_in[j] = b[j * 10 : j * 10 + 1]

                a = 1
                b = np.average(n)

                par, _ = curve_fit(lin_fit, av_win_in, av_win, p0=[a, b])

                av_win_fit = lin_fit(av_win_in, par[0], par[1])

            plt.xlabel("Time (ps)")
            plt.ylabel("Counts (-)")
            # plt.plot(av_win_in, av_win, color="black", linewidth=8)
            if fit_average is True:
                plt.gcf()
                plt.plot(av_win_in, av_win_fit, color="black", linewidth=8)
            plt.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
            plt.title(f"Pixel {pixels[i]}")
            try:
                os.chdir("results/single pixel histograms")
            except FileNotFoundError as _:
                os.makedirs("results/single pixel histograms")
                os.chdir("results/single pixel histograms")
            plt.savefig(f"{num}, pixel {pixels[i]}.png")
            os.chdir("../..")


def plot_sensor_population(
    path: str,
    daughterboard_number: str,
    motherboard_number: str,
    firmware_version: str,
    timestamps: int = 512,
    scale: str = "linear",
    style: str = "-o",
    show_fig: bool = False,
    app_mask: bool = True,
    color: str = "rebeccapurple",
    correct_pix_address: bool = False,
    fit_peaks: bool = False,
    threshold_multiplier: int = 10,
    pickle_fig: bool = False,
    absolute_timestamps: bool = False,
    single_file: bool = False,
) -> None:
    """Plot number of timestamps in each pixel for all datafiles.

    Plot sensor population as number of timestamps vs. pixel number.
    Analyzes all data files in the given folder. The output figure is saved
    in the "results" folder, which is created if it does not exist, in
    the same folder where datafiles are. Works with the firmware version
    '2212'.

    Parameters
    ----------
    path : str
        Path to the datafiles.
    daughterboard_number : str
        The LinoSPAD2 daughterboard number. Required for choosing the
        correct calibration data.
    motherboard_number : str
        The LinoSPAD2 motherboard number, including the "#".
    firmware_version : str
        LinoSPAD2 firmware version. Versions '2212b' (block) or '2212s'
        (skip) are recognized.
    timestamps : int, optional
        Number of timestamps per pixel per acquisition cycle. Default is
        "512".
    scale : str, optional
        Scale for the y-axis of the plot. Use "log" for logarithmic.
        The default is "linear".
    style : str, optional
        Style of the plot. The default is "-o".
    show_fig : bool, optional
        Switch for showing the plot. The default is False.
    app_mask : bool, optional
        Switch for applying the mask on warm/hot pixels. The default is
        True.
    color : str, optional
        Color for the plot. The default is 'rebeccapurple'.
    correct_pix_address : bool, optional
        Switch for correcting pixel addressing for the faulty firmware
        version for the 23 side of the daughterboard. The default is
        False.
    fit_peaks : bool, optional
        Switch for finding the highest peaks and fitting them with a
        Gaussian to provide their position. The default is False.
    threshold_multiplier : int, optional
        Threshold multiplier that is applied to median across the whole
        sensor for finding peaks. The default is 10.
    pickle_fig : bool, optional
        Switch for pickling the figure. Can be used when plotting takes
        a lot of time. The default is False.
    absolute_timestamps : bool, optional
        Indicator for data files with absolute timestamps. Default is
        False.
    single_file : optional
        Switch for using only the first file in the folder. Can be
        utilized for a quick plot. The default is False.

    Returns
    -------
    None.

    Examples
    -------
    An example how the function can be used to get the sensor
    occupation from a single file while looking for peaks - the most
    quick and straightforward approach to find where the beams were
    focused and get the peak position for further use in, e.g., delta_t
    functions. Here, the data were collected using the '23'-side
    sensor half which required correction of the pixel addressing.
    Offset calibration for the sensor is not available therefore
    it should be skipped.

    First, get the absolute path to where the '.dat' files are.
    >>> path = r'C:/Path/To/Data'

    Now to the function itself.
    >>> plot_sensor_popuation(
    >>> path,
    >>> daughterboard_number="NL11",
    >>> motherboard_number="#21",
    >>> firmware_version="2212s",
    >>> timestamps = 1000,
    >>> show_fig = True,
    >>> correct_pix_address = True,
    >>> fit_peaks = True,
    >>> single_file = True,
    >>> )
    """
    # parameter type check
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
    if show_fig:
        plt.ion()
    else:
        plt.ioff()

    os.chdir(path)

    # files = glob.glob("*.dat*")
    files = glob.glob("*.dat*")
    files.sort(key=os.path.getmtime)

    if single_file:
        files = files[0]
        plot_name = files[:-4]
    else:
        plot_name = files[0][:-4] + "-" + files[-1][:-4]

    # valid_per_pixel = np.zeros(256)

    print(
        "\n> > > Collecting data for sensor population plot,"
        f"Working in {path} < < <\n"
    )

    # If fitting the peaks, calculate the photon rates in each peak as
    # well
    if fit_peaks:
        timestamps_per_pixel, rates = collect_data_and_apply_mask(
            files,
            daughterboard_number,
            motherboard_number,
            firmware_version,
            timestamps,
            app_mask,
            absolute_timestamps,
            save_to_file=False,
            correct_pix_address=correct_pix_address,
            calculate_rates=True,
        )
    else:
        timestamps_per_pixel = collect_data_and_apply_mask(
            files,
            daughterboard_number,
            motherboard_number,
            firmware_version,
            timestamps,
            app_mask,
            absolute_timestamps,
            save_to_file=False,
            correct_pix_address=correct_pix_address,
        )

    # Plotting
    print("\n> > > Plotting < < <\n")
    plt.rcParams.update({"font.size": 27})
    fig = plt.figure(figsize=(16, 10))
    if scale == "log":
        plt.yscale("log")
    plt.plot(timestamps_per_pixel, style, color=color)
    plt.xlabel("Pixel number (-)")
    plt.ylabel("Photons (-)")

    # Find and fit peaks if fit_peaks is True
    if fit_peaks:
        threshold = np.median(timestamps_per_pixel) * threshold_multiplier
        fit_width = 20
        peaks, _ = find_peaks(timestamps_per_pixel, height=threshold)
        peaks = np.unique(peaks)

        print("Fitting the peaks with gaussian")
        for peak_index in peaks:
            x_fit = np.arange(
                peak_index - fit_width, peak_index + fit_width + 1
            )
            cut_above_256 = np.where(x_fit >= 256)[0]
            x_fit = np.delete(x_fit, cut_above_256)
            y_fit = timestamps_per_pixel[x_fit]
            try:
                params, _ = utils.fit_gaussian(x_fit, y_fit)
            except Exception as _:
                continue

            plt.plot(
                x_fit,
                utils.gaussian(x_fit, *params),
                "--",
                label=f"Peak at {peak_index}, "
                f"Rate: {rates[peak_index]/1000:.0f} kHz",
            )

        plt.legend()

    # Save the figure
    try:
        os.chdir("results/sensor_population")
    except FileNotFoundError:
        os.makedirs("results/sensor_population")
        os.chdir("results/sensor_population")
    # fig.tight_layout()
    if single_file:
        plt.savefig(f"{plot_name}_single_file.png")
        print(
            "> > > The plot is saved as '{plot_name}_single_file.png'"
            "in {os.getcwd()} < < <"
        )
        if pickle_fig:
            pickle.dump(fig, open(f"{plot_name}_single_file.pickle", "wb"))
    else:
        plt.savefig(f"{plot_name}.png")
        print(
            f"> > > The plot is saved as '{plot_name}.png' "
            f"in {os.getcwd()} < < <"
        )
        if pickle_fig:
            pickle.dump(fig, open(f"{plot_name}.pickle", "wb"))

    os.chdir("../..")


# TODO don't have data to test the function
# def plot_sensor_population_spdc(
#     path,
#     daughterboard_number: str,
#     motherboard_number: str,
#     firmware_version: str,
#     timestamps: int = 512,
#     show_fig: bool = False,
# ):
#     """Plot sensor population for SPDC data.

#     Plots SPDC data subtracting the background (data with the SPDC
#     output turned off). Due to the low sensitivity of the LinoSPAD2
#     sensor to light at 810 nm of Thorlabs SPDC output, subtracting
#     background is required to show any meaningful signal.

#     Parameters
#     ----------
#     path : str
#         Path to data files.
#     daughterboard_number : str
#         LinoSPAD2 daughterboard number. Either "A5" or "NL11" is
#         accepted.
#     motherboard_number : str
#         LinoSPAD2 motherboard number.
#     timestamps : int, optional
#         Number of timestamps per pixel per acquisition cycle. The
#         default is 512.
#     show_fig : bool, optional
#         Switch for showing the plot. The default is False.

#     Raises
#     ------
#     TypeError
#         Raised when 'daughterboard_number' is not a string.
#     ValueError
#         Raised when the number of data files of SPDC data is different
#         from the number of data files of background data.

#     Returns
#     -------
#     None.

#     """
#     # parameter type check
#     if isinstance(daughterboard_number, str) is not True:
#         raise TypeError(
#             "'daughterboard_number' should be string, either 'NL11' or 'A5'"
#         )
#     if isinstance(firmware_version, str) is not True:
#         raise TypeError("'firmware_version' should be string")
#     if isinstance(motherboard_number, str) is not True:
#         raise TypeError(
#             "'daughterboard_number' should be string, either 'NL11' or 'A5'"
#         )

#     if show_fig is True:
#         plt.ion()
#     else:
#         plt.ioff()

#     os.chdir(path)

#     # files = glob.glob("*.dat*")

#     files = glob.glob("*.dat*")
#     files.sort(key=os.path.getmtime)

#     # background data for subtracting
#     path_bckg = path + "/bckg"
#     os.chdir(path_bckg)
#     files_bckg = glob.glob("*.dat*")
#     os.chdir("..")

#     if len(files) != len(files_bckg):
#         raise ValueError(
#             "Number of files with background data is different from the"
#             "number of actual data, exiting."
#         )

#     plot_name = files[0][:-4] + "-" + files[-1][:-4]

#     valid_per_pixel = np.zeros(256)
#     valid_per_pixel_bckg = np.zeros(256)

#     pix_coor = np.arange(256).reshape(64, 4)

#     # Collect SPDC data
#     for i in tqdm(range(len(files)), desc="Going through datafiles"):
#         data_all = f_up.unpack_binary_data(
#             files[i],
#             daughterboard_number="A5",
#             firmware_version="2212b",
#             timestamps=timestamps,
#         )

#         for i in np.arange(0, 256):
#             tdc, pix = np.argwhere(pix_coor == i)[0]
#             ind = np.where(data_all[tdc].T[0] == pix)[0]
#             ind1 = ind[np.where(data_all[tdc].T[1][ind] > 0)[0]]
#             valid_per_pixel[i] += len(data_all[tdc].T[1][ind1])

#     # Collect background data for subtracting
#     os.chdir(path_bckg)

#     for i in tqdm(
#         range(len(files_bckg)), desc="Going through background datafiles"
#     ):
#         data_all_bckg = f_up.unpack_binary_data(
#             files_bckg[i],
#             daughterboard_number="A5",
#             motherboard_number="34",
#             firmware_version="2212b",
#             timestamps=timestamps,
#         )

#         # Fot plotting counts
#         for i in np.arange(0, 256):
#             tdc, pix = np.argwhere(pix_coor == i)[0]
#             ind = np.where(data_all_bckg[tdc].T[0] == pix)[0]
#             ind1 = ind[np.where(data_all_bckg[tdc].T[1][ind] > 0)[0]]
#             valid_per_pixel_bckg[i] += len(data_all_bckg[tdc].T[1][ind1])

#     os.chdir("..")

#     # Mask the hot/warm pixels
#     path_to_back = os.getcwd()
#     path_to_mask = os.path.realpath(__file__) + "/../.." + "/params/masks"
#     os.chdir(path_to_mask)
#     file_mask = glob.glob(
#         "*{}_{}*".format(daughterboard_number, motherboard_number)
#     )[0]
#     mask = np.genfromtxt(file_mask).astype(int)
#     os.chdir(path_to_back)

#     for i in mask:
#         valid_per_pixel[i] = 0
#         valid_per_pixel_bckg[i] = 0

#     plt.rcParams.update({"font.size": 22})
#     plt.figure(figsize=(16, 10))
#     plt.xlabel("Pixel [-]")
#     plt.ylabel("Counts [-]")
#     plt.title("SPDC data, background subtracted")
#     plt.plot(valid_per_pixel - valid_per_pixel_bckg, "o-", color="teal")
#     plt.tight_layout()

#     try:
#         os.chdir("results/sensor_population")
#     except Exception:
#         os.makedirs("results/sensor_population")
#         os.chdir("results/sensor_population")
#     plt.savefig("{}_SPDC_counts.png".format(plot_name))
#     plt.pause(0.1)
#     os.chdir("../..")


def plot_sensor_population_full_sensor(
    path,
    daughterboard_number: str,
    motherboard_number1: str,
    motherboard_number2: str,
    firmware_version: str,
    timestamps: int = 512,
    scale: str = "linear",
    style: str = "-o",
    show_fig: bool = False,
    app_mask: bool = True,
    color: str = "salmon",
    fit_peaks: bool = False,
    threshold_multiplier: int = 10,
    pickle_fig: bool = False,
    single_file: bool = False,
    absolute_timestamps: bool = False,
):
    """Plot the number of timestamps in each pixel for all datafiles.

    Plot sensor population as the number of timestamps vs. pixel number.
    Analyzes all data files in the given folder. The output figure is saved
    in the "results" folder, which is created if it does not exist, in
    the same folder where datafiles are. Works with the firmware version
    '2212'.

    Parameters
    ----------
    path : str
        Path to the datafiles.
    daughterboard_number : str
        The LinoSPAD2 daughterboard number. Required for choosing the
        correct calibration data.
    motherboard_number1 : str
        The LinoSPAD2 motherboard number for the first board.
    motherboard_number2 : str
        The LinoSPAD2 motherboard number for the second board.
    firmware_version : str
        LinoSPAD2 firmware version. Versions '2212b' (block) or '2212s'
        (skip) are recognized.
    timestamps : int, optional
        Number of timestamps per pixel per acquisition cycle. Default is
        "512".
    scale : str, optional
        Scale for the y-axis of the plot. Use "log" for logarithmic.
        The default is "linear".
    style : str, optional
        Style of the plot. The default is "-o".
    show_fig : bool, optional
        Switch for showing the plot. The default is False.
    app_mask : bool, optional
        Switch for applying the mask on warm/hot pixels. The default is
        True.
    color : str, optional
        Color for the plot. The default is 'salmon'.
    fit_peaks : bool, optional
        Switch for finding the highest peaks and fitting them with a
        Gaussian to provide their position. The default is False.
    threshold_multiplier : int, optional
        Threshold multiplier for setting the threshold for finding peaks.
        The default is 10.
    pickle_fig : bool, optional
        Switch for pickling the figure. Can be used when plotting takes
        a lot of time. The default is False.
    single_file : bool, optional
        Switch for unpacking only the first file for a quick plot.
        The default is False.
    absolute_timestamps : bool, optional
        Indicator for data files with absolute timestamps. Default is
        False.

    Returns
    -------
    None.

    Notes
    -----

    As the pixel addressing is incorrect for one of the sensor halves
    (depends on the daughterboard-motherboards combinatios; in NL11, for
    motherboard #21 pixel addressing should be applied), it is important
    to apply pixel addressing to the correct board/sensor half. For this
    function, the order in which the motherboards are input is key, as
    pixel addressing is corrected for the second board.

    TLDR: motherboard_number2 should be for the one where correction of
    the pixel addressing is required.

    """
    # parameter type check
    if not isinstance(firmware_version, str):
        raise TypeError(
            "'firmware_version' should be a string, '2212b' or '2212s'"
        )
    if not isinstance(daughterboard_number, str):
        raise TypeError(
            "'daughterboard_number' should be a string, 'NL11' or 'A5'"
        )
    if not isinstance(motherboard_number1, str):
        raise TypeError("'motherboard_number1' should be a string")
    if not isinstance(motherboard_number2, str):
        raise TypeError("'motherboard_number2' should be a string")
    if show_fig:
        plt.ion()
    else:
        plt.ioff()

    valid_per_pixel1 = np.zeros(256)
    valid_per_pixel2 = np.zeros(256)

    # Get the two folders with data from both FPGAs/sensor halves
    os.chdir(path)
    path1 = glob.glob(f"*{motherboard_number1}*")[0]
    path2 = glob.glob(f"*{motherboard_number2}*")[0]

    # First motherboard / half of the sensor
    os.chdir(path1)
    # files1 = sorted(glob.glob("*.dat*"))

    files1 = glob.glob("*.dat*")
    files1.sort(key=os.path.getmtime)

    if single_file:
        files1 = files1[0]
    plot_name1 = files1[0][:-4] + "-"

    print(
        "\n> > > Collecting data for sensor population plot,"
        f"Working in {path1} < < <\n"
    )

    valid_per_pixel1 = collect_data_and_apply_mask(
        files1,
        daughterboard_number,
        motherboard_number1,
        firmware_version,
        timestamps,
        app_mask,
        absolute_timestamps,
    )

    os.chdir("..")

    # Second motherboard / half of the sensor
    os.chdir(path2)
    # files2 = sorted(glob.glob("*.dat*"))
    files2 = glob.glob("*.dat*")
    files2.sort(key=os.path.getmtime)
    if single_file:
        files2 = files2[0]
    plot_name2 = files2[-1][:-4]

    print(
        "\n> > > Collecting data for sensor population plot,"
        f"Working in {path2} < < <\n"
    )
    valid_per_pixel2 = collect_data_and_apply_mask(
        files2,
        daughterboard_number,
        motherboard_number2,
        firmware_version,
        timestamps,
        app_mask,
        absolute_timestamps,
    )

    # Fix pixel addressing for the second board
    fix = np.zeros(len(valid_per_pixel2))
    fix[:128] = valid_per_pixel2[128:]
    fix[128:] = np.flip(valid_per_pixel2[:128])
    valid_per_pixel2 = fix
    del fix

    # Concatenate and plot
    valid_per_pixel = np.concatenate([valid_per_pixel1, valid_per_pixel2])
    plot_name = plot_name1 + plot_name2

    print("\n> > > Plotting < < <\n")

    plt.rcParams.update({"font.size": 27})
    fig = plt.figure(figsize=(16, 10))
    if scale == "log":
        plt.yscale("log")
    plt.plot(valid_per_pixel, style, color=color)
    plt.xlabel("Pixel number (-)")
    plt.ylabel("Photons (-)")

    # Find and fit peaks if fit_peaks is True
    if fit_peaks:
        threshold = np.median(valid_per_pixel) * threshold_multiplier
        fit_width = 10
        peaks, _ = find_peaks(valid_per_pixel, height=threshold)
        peaks = np.unique(peaks)

        for peak_index in tqdm(peaks, desc="Fitting Gaussians"):
            x_fit = np.arange(
                peak_index - fit_width, peak_index + fit_width + 1
            )
            y_fit = valid_per_pixel[x_fit]
            try:
                params, _ = utils.fit_gaussian(x_fit, y_fit)
            except Exception:
                continue

            # amplitude, position, width = params
            # position = np.clip(int(position), 0, 255)

            plt.plot(
                x_fit,
                utils.gaussian(x_fit, *params),
                "--",
                label=f"Peak at {peak_index}",
            )

        plt.legend()

    os.chdir("..")

    try:
        os.chdir("results/sensor_population")
    except FileNotFoundError:
        os.makedirs("results/sensor_population")
        os.chdir("results/sensor_population")
    fig.tight_layout()
    plt.savefig("{}.png".format(plot_name))
    print(
        f"> > > The plot is saved as '{plot_name}.png' "
        f"in {os.getcwd()} < < <"
    )
    if pickle_fig:
        pickle.dump(fig, open(f"{plot_name}.pickle", "wb"))


def unpickle_plot(plot_pickle_file: str) -> dict:
    """Unpickle a saved figure and return plot data.

    Load a pickled figure with the sensor population plot, extract and
    return the plot data.

    Parameters
    ----------
    plot_pickle_file : str
        The absolute path to the pickle file with the sensor population
        plot.

    Returns
    -------
    tuple
        A tuple containing:
        - fig (matplotlib.figure.Figure): The unpickled figure object.
        - plot_data (dict): A dictionary with keys "Histogram" and
        "Fit_{i}" (for i >= 1), where each value is a tuple of x and y
        data for the corresponding line plot.

    Raises
    ------
    FileNotFoundError
        If the specified pickle file does not exist, a FileNotFoundError
        is raised and an error message is printed.
    """
    try:
        with open(plot_pickle_file, "rb") as f:
            fig = pickle.load(f)
    except FileNotFoundError as e:
        print(f" {e}")

    # Pack the data into a dictionary, first is the histogram, others
    # are the fits
    plot_data = {}

    # Extract the data; go over lines, stack them into dictionary
    ax = fig.axes[0]
    for i, line in enumerate(ax.lines):
        x, y = line.get_data()
        if i == 0:
            plot_data["Plot"] = (x, y)
        else:
            plot_data[f"Fit_{i}"] = (x, y)

    # Extract the legend
    legend = ax.get_legend()

    if legend is not None:
        legend_labels = [text.get_text() for text in legend.get_texts()]

        legend_labels = "\n".join(legend_labels)

        return fig, plot_data, legend_labels
    else:
        return fig, plot_data
