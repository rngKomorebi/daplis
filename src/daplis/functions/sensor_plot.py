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

    * plot_sensor_population_rates - Plot photon rate in each pixel
    for all data files.

    * plot_sensor_population_spdc - Plot sensor population for SPDC data.

    * plot_sensor_population_full_sensor - Plot the number of timestamps
    in each pixel for all data files from two different FPGAs/sensor
    halves.
"""

from __future__ import annotations

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
    files: List[str] | str,
    daughterboard_number: str,
    motherboard_number: str,
    firmware_version: str,
    timestamps: int,
    apply_hot_pixel_mask: bool = True,
    absolute_timestamps: bool = False,
    save_to_file: bool = False,
    correct_pix_address: bool = False,
    calculate_rates: bool = False,
    acq_window_length: float | None = None,
    number_of_cycles: float | None = None,
) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
    """Collect data from files and return number of timestamps in pixels.

    Unpacks data and returns the number of timestamps (and photon rate)
    in each pixel. Optionally, aplly mask or hot pixels, save the data
    into a file.


    Parameters
    ----------
    files : List[str], str
        List of data file paths.
    daughterboard_number : str
        The LinoSPAD2 daughterboard number.
    motherboard_number : str
        The LinoSPAD2 motherboard number, including the "#".
    firmware_version : str
        LinoSPAD2 firmware version.
    timestamps : int
        Number of timestamps per pixel per acquisition cycle.
    apply_hot_pixel_mask : bool, optional
        Switch for applying the mask on warm/hot pixels. Default is True.
    absolute_timestamps : bool, optional
        Indicator for data files with absolute timestamps. Default is
        False.
    correct_pix_address : bool, optional
        Check for correcting the pixel addressing. The default is False.
    calculate_rates : bool, optional
        Switch for calculating the photon rate for each pixel. The
        default is False.
    acq_window_length : float, optional
        Length of each acquisition cycle; maximum is 4 ms. If None,
        estimated from the data. The default is None.
    number_of_cycles : float, optional
        Number of acquisition cycles per data file. If None,
        estimated from the data. The default is None.

    Returns
    -------
    timestamps_per_pixel : ndarray of shape (256,)
        Number of valid timestamps accumulated in each pixel.
    rates : ndarray of shape (256,), optional
        Returned only when 'calculate_rates=True'. Photon detection rates
        per pixel, in events per second.
    """
    # Define matrix of pixel coordinates, where rows are numbers of TDCs
    # and columns are the pixels that connected to these TDCs
    if firmware_version == "2212s":
        pixel_coordinates = np.arange(256).reshape(4, 64).T
    elif firmware_version == "2212b":
        pixel_coordinates = np.arange(256).reshape(64, 4)
    else:
        print("\nFirmware version is not recognized.")
        sys.exit()

    timestamps_per_pixel = np.zeros(256)

    # In the case a single file is passed, make a list out of it
    if isinstance(files, str):
        files = [files]

    for i in tqdm(range(len(files)), desc="Collecting data"):
        if not absolute_timestamps:
            data_pixels, data_timestamps = f_up.unpack_binary_data(
                files[i],
                daughterboard_number,
                motherboard_number,
                firmware_version,
                timestamps,
            )
        else:
            data_pixels, data_timestamps, _ = (
                f_up.unpack_binary_data_with_absolute_timestamps(
                    files[i],
                    daughterboard_number,
                    motherboard_number,
                    firmware_version,
                    timestamps,
                    include_offset=False,
                    apply_calibration=False,
                )
            )
        for i in range(256):
            # tdc, pix = np.argwhere(pix_coor == i)[0]
            # ind = np.where(data[tdc].T[0] == pix)[0]
            # ind1 = np.where(data[tdc].T[1][ind] > 0)[0]
            # timestamps_per_pixel[i] += len(data[tdc].T[1][ind[ind1]])
            tdc, pix = np.argwhere(pixel_coordinates == i)[0]
            mask = (data_pixels[tdc] == pix) & (data_timestamps[tdc] >= 0)
            ind = np.nonzero(mask)[0]
            timestamps_per_pixel[i] += len(data_timestamps[tdc][ind])

    if correct_pix_address:
        fix = np.zeros(len(timestamps_per_pixel))
        fix[:128] = timestamps_per_pixel[128:]
        fix[128:] = np.flip(timestamps_per_pixel[:128])
        timestamps_per_pixel = fix
        del fix

    # Apply mask if requested
    if apply_hot_pixel_mask:
        mask = utils.apply_mask(daughterboard_number, motherboard_number)
        timestamps_per_pixel[mask] = 0

    # if calculate_rates:
    #     if acq_window_length is None:
    #         acq_window_length = np.max(data[:].T[1]) * 1e-12
    #     if number_of_cycles is None:
    #         number_of_cycles = len(np.where(data[0].T[0] == -2)[0])
    if calculate_rates:
        if acq_window_length is None:
            acq_window_length = np.max(data_timestamps * 2500 / 140).astype(
                int
            )
        if number_of_cycles is None:
            number_of_cycles = len(data_timestamps.flatten()) / 64 / timestamps

        rates = (
            timestamps_per_pixel
            / acq_window_length
            * 1e12  # transform to seconds
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

    def _lin_fit(x, a, b):
        return a * x + b

    if type(pixels) is int:
        pixels = [pixels]

    os.chdir(path)

    # data_files = glob.glob("*.dat*")
    data_files = glob.glob("*.dat*")
    data_files.sort(key=os.path.getmtime)

    for i, num in enumerate(data_files):
        print(f"> > > Plotting pixel histograms, Working on {num} < < <\n")

        data_pixels, data_timestamps = f_up.unpack_binary_data(
            num,
            daughterboard_number,
            motherboard_number,
            firmware_version,
            timestamps,
        )

        bins = np.arange(
            0, cycle_length, 2500 / 140 * multiplier
        )  # bin size of 17.867 us

        if pixels is None:
            pixels = np.arange(145, 165, 1)

        for i, _ in enumerate(pixels):
            plt.figure(figsize=(16, 10))
            plt.rcParams.update({"font.size": 30})
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
            mask = (data_pixels[tdc] == pix) & (data_timestamps[tdc] >= 0)
            ind = np.nonzero(mask)[0]
            data_to_plot = data_timestamps[tdc][ind]

            n, b, p = plt.hist(data_to_plot, bins=bins, color=color)
            if fit_average is True:
                av_win = np.zeros(int(len(n) / 10) + 1)
                av_win_in = np.zeros(int(len(n) / 10) + 1)
                for j, _ in enumerate(av_win):
                    av_win[j] = n[j * 10 : j * 10 + 1]
                    av_win_in[j] = b[j * 10 : j * 10 + 1]

                a = 1
                b = np.average(n)

                par, _ = curve_fit(_lin_fit, av_win_in, av_win, p0=[a, b])

                av_win_fit = _lin_fit(av_win_in, par[0], par[1])

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
    y_scale: str = "linear",
    apply_hot_pixel_mask: bool = True,
    look_for_peaks: bool = True,
    peak_threshold: int = 10,
    single_file: bool = False,
    plot_rates: bool = True,
    pickle_fig: bool = False,
    correct_pix_address: bool = False,
    absolute_timestamps: bool = False,
) -> None:
    """Plot number of timestamps in each pixel for all datafiles.

    Plot sensor population as number of timestamps vs. pixel number.
    Analyzes all data files in the given folder. The output figure is
    saved in the "results" folder, which is created if it does not
    exist, in the same folder where datafiles are. Works with the
    firmware version '2212'.

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
    y_scale : str, optional
        Scale for the y-axis of the plot. Use "log" for logarithmic.
        The default is "linear".
    style : str, optional
        Style of the plot. The default is "-o".
    apply_hot_pixel_mask : bool, optional
        Switch for applying the mask on warm/hot pixels. The default is
        True.
    look_for_peaks : bool, optional
        Switch for finding the highest peaks. The default is False.
    peak_threshold : int, optional
        Threshold multiplier that is applied to median across the whole
        sensor for finding peaks. The default is 10.
    single_file : bool, optional
        Switch for using only the first file in the folder. Can be
        utilized for a quick plot. The default is False.
    plot_rates : bool, optional
        Switch for using the rates for the plot. The default is True.
    pickle_fig : bool, optional
        Switch for pickling the figure. Can be used when plotting takes
        a lot of time. The default is False.
    correct_pix_address : bool, optional
        Switch for correcting pixel addressing for the faulty firmware
        version for the 23 side of the daughterboard. The default is
        False.
    absolute_timestamps : bool, optional
        Indicator for data files with absolute timestamps. Default is
        False.

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

    First, get the absolute path to where the '.dat' files are.
    >>> path = r'C:/Path/To/Data'

    Now to the function itself.
    >>> plot_sensor_popuation(
    >>> path,
    >>> daughterboard_number="NL11",
    >>> motherboard_number="#21",
    >>> firmware_version="2212s",
    >>> timestamps = 1000,
    >>> correct_pix_address = True,
    >>> look_for_peaks = True,
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
    if look_for_peaks or plot_rates:
        timestamps_per_pixel, rates = collect_data_and_apply_mask(
            files,
            daughterboard_number,
            motherboard_number,
            firmware_version,
            timestamps,
            apply_hot_pixel_mask,
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
            apply_hot_pixel_mask,
            absolute_timestamps,
            save_to_file=False,
            correct_pix_address=correct_pix_address,
        )

    print("Rates", rates)

    # Plotting
    print("\n> > > Plotting < < <\n")
    plt.rcParams.update({"font.size": 30})
    fig = plt.figure(figsize=(16, 10))
    fig.subplots_adjust(top=0.94, right=0.93)
    if y_scale == "log":
        plt.yy_scale("log")

    if not plot_rates:
        plt.plot(timestamps_per_pixel, "o-", color="rebeccapurple")
        plt.xlabel("Pixel number (-)")
        plt.ylabel("Photons (-)")

    # Plotting
    else:
        if np.max(rates) > 1e3:
            plt.plot(rates / 1e3, "o-", color="rebeccapurple")
            plt.ylabel("Photon rate (kHz)")
        elif np.max(rates) > 1e6:
            plt.plot(rates / 1e6, "o-", color="rebeccapurple")
            plt.ylabel("Photon rate (MHz)")
        else:
            plt.plot(rates, "o-", color="rebeccapurple")
            plt.ylabel("Photon rate (Hz)")
        plt.xlabel("Pixel number (-)")

    # Find and fit peaks if look_for_peaks is True
    if look_for_peaks:
        threshold = np.median(timestamps_per_pixel) * peak_threshold
        peak_search_width = 5
        peaks, _ = find_peaks(timestamps_per_pixel, height=threshold)
        peaks = np.unique(peaks)

        for peak_index in peaks:
            x_peak = np.arange(
                peak_index - peak_search_width,
                peak_index + peak_search_width + 1,
            )
            cut_above_256 = np.where(x_peak >= 256)[0]
            x_peak = np.delete(x_peak, cut_above_256)
            y_peak = timestamps_per_pixel[x_peak]
            # try:
            #     params, _ = utils.fit_gaussian(x_peak, y_peak)
            # except Exception as _:
            #     continue

            plt.plot(
                0,
                0,
                "o--",
                # markersize=8,
                color="rebeccapurple",
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


# def plot_sensor_population_rates(
#     path: str,
#     daughterboard_number: str,
#     motherboard_number: str,
#     firmware_version: str,
#     timestamps: int = 512,
#     y_scale: str = "linear",
#     style: str = "-o",
#     apply_hot_pixel_mask: bool = True,
#     color: str = "rebeccapurple",
#     correct_pix_address: bool = False,
#     pickle_fig: bool = False,
#     absolute_timestamps: bool = False,
#     single_file: bool = False,
# ) -> None:
#     """Plot number of timestamps in each pixel for all datafiles.

#     Plot sensor population as number of timestamps vs. pixel number.
#     Analyzes all data files in the given folder. The output figure is saved
#     in the "results" folder, which is created if it does not exist, in
#     the same folder where datafiles are. Works with the firmware version
#     '2212'.

#     Parameters
#     ----------
#     path : str
#         Path to the datafiles.
#     daughterboard_number : str
#         The LinoSPAD2 daughterboard number. Required for choosing the
#         correct calibration data.
#     motherboard_number : str
#         The LinoSPAD2 motherboard number, including the "#".
#     firmware_version : str
#         LinoSPAD2 firmware version. Versions '2212b' (block) or '2212s'
#         (skip) are recognized.
#     timestamps : int, optional
#         Number of timestamps per pixel per acquisition cycle. Default is
#         "512".
#     y_scale : str, optional
#         Scale for the y-axis of the plot. Use "log" for logarithmic.
#         The default is "linear".
#     style : str, optional
#         Style of the plot. The default is "-o".
#     apply_hot_pixel_mask : bool, optional
#         Switch for applying the mask on warm/hot pixels. The default is
#         True.
#     color : str, optional
#         Color for the plot. The default is 'rebeccapurple'.
#     correct_pix_address : bool, optional
#         Switch for correcting pixel addressing for the faulty firmware
#         version for the 23 side of the daughterboard. The default is
#         False.
#     pickle_fig : bool, optional
#         Switch for pickling the figure. Can be used when plotting takes
#         a lot of time. The default is False.
#     absolute_timestamps : bool, optional
#         Indicator for data files with absolute timestamps. Default is
#         False.
#     single_file : optional
#         Switch for using only the first file in the folder. Can be
#         utilized for a quick plot. The default is False.

#     Returns
#     -------
#     None.

#     Examples
#     -------
#     An example how the function can be used to get the sensor
#     occupation from a single file while looking for peaks - the most
#     quick and straightforward approach to find where the beams were
#     focused and get the peak position for further use in, e.g., delta_t
#     functions. Here, the data were collected using the '23'-side
#     sensor half which required correction of the pixel addressing.
#     Offset calibration for the sensor is not available therefore
#     it should be skipped.

#     First, get the absolute path to where the '.dat' files are.
#     >>> path = r'C:/Path/To/Data'

#     Now to the function itself.
#     >>> plot_sensor_popuation(
#     >>> path,
#     >>> daughterboard_number="NL11",
#     >>> motherboard_number="#21",
#     >>> firmware_version="2212s",
#     >>> timestamps = 1000,
#     >>> correct_pix_address = True,
#     >>> find_peaks = True,
#     >>> single_file = True,
#     >>> )
#     """
#     # parameter type check
#     if not isinstance(firmware_version, str):
#         raise TypeError(
#             "'firmware_version' should be a string, '2212b' or '2212s'"
#         )
#     if not isinstance(daughterboard_number, str):
#         raise TypeError(
#             "'daughterboard_number' should be a string, 'NL11' or 'A5'"
#         )
#     if not isinstance(motherboard_number, str):
#         raise TypeError("'motherboard_number' should be a string")

#     os.chdir(path)

#     files = glob.glob("*.dat*")
#     files.sort(key=os.path.getmtime)

#     if single_file:
#         files = files[0]
#         plot_name = files[:-4]
#     else:
#         plot_name = files[0][:-4] + "-" + files[-1][:-4]

#     print(
#         "\n> > > Collecting data for sensor population plot,"
#         f"Working in {path} < < <\n"
#     )

#     # If fitting the peaks, calculate the photon rates in each peak as
#     # well
#     _, rates = collect_data_and_apply_mask(
#         files,
#         daughterboard_number,
#         motherboard_number,
#         firmware_version,
#         timestamps,
#         apply_hot_pixel_mask,
#         absolute_timestamps,
#         save_to_file=False,
#         correct_pix_address=correct_pix_address,
#         calculate_rates=True,
#     )

#     # Plotting
#     print("\n> > > Plotting < < <\n")
#     plt.rcParams.update({"font.size": 30})
#     fig = plt.figure(figsize=(16, 10))
#     fig.subplots_adjust(top=0.94, right=0.93)
#     if y_scale == "log":
#         plt.yy_scale("log")
#     if np.max(rates) > 1e3:
#         plt.plot(rates / 1e3, style, color=color)
#         plt.ylabel("Photon rate (kHz)")
#     elif np.max(rates) > 1e6:
#         plt.plot(rates / 1e6, style, color=color)
#         plt.ylabel("Photon rate (MHz)")
#     else:
#         plt.plot(rates, style, color=color)
#         plt.ylabel("Photon rate (Hz)")
#     plt.xlabel("Pixel number (-)")

#     # Save the figure
#     try:
#         os.chdir("results/sensor_population")
#     except FileNotFoundError:
#         os.makedirs("results/sensor_population")
#         os.chdir("results/sensor_population")
#     # fig.tight_layout()
#     if single_file:
#         plt.savefig(f"{plot_name}_single_file_ver2.png")
#         print(
#             "> > > The plot is saved as '{plot_name}_rates_single_file.png'"
#             "in {os.getcwd()} < < <"
#         )
#         if pickle_fig:
#             pickle.dump(
#                 fig, open(f"{plot_name}_rates_single_file.pickle", "wb")
#             )
#     else:
#         plt.savefig(f"{plot_name}.png")
#         print(
#             f"> > > The plot is saved as '{plot_name}_rates.png' "
#             f"in {os.getcwd()} < < <"
#         )
#         if pickle_fig:
#             pickle.dump(fig, open(f"{plot_name}_rates.pickle", "wb"))

#     os.chdir("../..")

#     return fig


def plot_sensor_population_full_sensor(
    path,
    daughterboard_number: str,
    motherboard_number1: str,
    motherboard_number2: str,
    firmware_version: str,
    timestamps: int = 512,
    y_scale: str = "linear",
    style: str = "-o",
    apply_hot_pixel_mask: bool = True,
    color: str = "salmon",
    find_peaks: bool = False,
    peak_threshold: int = 10,
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
    y_scale : str, optional
        Scale for the y-axis of the plot. Use "log" for logarithmic.
        The default is "linear".
    style : str, optional
        Style of the plot. The default is "-o".
    apply_hot_pixel_mask : bool, optional
        Switch for applying the mask on warm/hot pixels. The default is
        True.
    color : str, optional
        Color for the plot. The default is 'salmon'.
    find_peaks : bool, optional
        Switch for finding the highest peaks and fitting them with a
        Gaussian to provide their position. The default is False.
    peak_threshold : int, optional
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
        apply_hot_pixel_mask,
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
        apply_hot_pixel_mask,
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

    plt.rcParams.update({"font.size": 30})
    fig = plt.figure(figsize=(16, 10))
    fig.subplots_adjust(top=0.94, right=0.93)
    if y_scale == "log":
        plt.yy_scale("log")
    plt.plot(valid_per_pixel, style, color=color)
    plt.xlabel("Pixel number (-)")
    plt.ylabel("Photons (-)")

    # Find and fit peaks if find_peaks is True
    if find_peaks:
        threshold = np.median(valid_per_pixel) * peak_threshold
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
