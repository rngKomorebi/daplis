"""Module for checking quality of data from LinoSPAD2.

The functions implemented in this module can be used to check data
quality, i.e., to study more closely how sensor population changes
between cycles or how the total number of timestamps in the given pixel
changes over time (cycles).

This file can also be imported as a module and contains the following
functions:

    * sensor_population_by_cycle - unpack data from the chosen file,
    collect number of timestamps in each pixel and cycle and plot the
    sensor population for either chosen cycles or for the starting
    cycle and 3 cycles on each side. The starting cycle can be found
    based on the given threshold, e.g., 0 for any positive signal or
    15 for a signal stronger than 3.5 kHz per pixel. The plots
    are then saved.

    * pixel_population_by_cycle - unpack data for all files in the given
    folder, collect number of timestamps accross all cycles for the
    given pixel and plot them against the cycle number. For clearer
    picture how the signal oscilates, also plot a moving average in a
    window of 100 cycles. The plot is then saved.

    * get_time_ratios - unpacks data for all files in given folder, calculates
    the data collection time as the number of cycles in one file * the 
    length of one cycle,  calculates the total acquisition time as the 
    difference between the modification times of two subsequent files, 
    calculates the ratios of these times and calculates the average with 
    error.
 
    * save_file_time - saves the modification and creation times of data 
    files into a feather file.
    
    * load_data_from_feather - loads the modification and creation time 
    of data files from the feather file.

"""

import glob
import os
import sys

import numpy as np
from matplotlib import pyplot as plt
from tqdm import tqdm
import pandas as pd
import pyarrow.feather as feather
from datetime import datetime


from LinoSPAD2.functions import unpack as f_up
from LinoSPAD2.functions import utils


def sensor_population_by_cycle(
    path: str,
    daughterboard_number: str,
    motherboard_number: str,
    firmware_version: str,
    timestamps: int = 512,
    app_mask: bool = True,
    include_offset: bool = True,
    apply_calibration: bool = True,
    absolute_timestamps: bool = False,
    correct_pixel_addressing: bool = False,
    cycle_range: list = None,
    threshold: int = 0,
    chosen_file: int = 0,
) -> Tuple[np.ndarray, list]:
    """
    Collect sensor population data by acquisition cycle.

    Parameters
    ----------
    path : str
        Path to the data files.
    daughterboard_number : str
        LinoSPAD2 daughterboard number.
    motherboard_number : str
        LinoSPAD2 motherboard (FPGA) number.
    firmware_version : str
        LinoSPAD2 firmware version. Versions '2212s' (skip) and '2212b'
        (block) are recognized.
    timestamps : int, optional
        Number of timestamps per acquisition cycle per pixel. The default
        is 512.
    app_mask : bool, optional
        Switch for applying pixel mask. The default is True.
    include_offset : bool, optional
        Switch for applying offset calibration. The default is True.
    apply_calibration : bool, optional
        Switch for applying TDC and offset calibration. If set to 'True'
        while include_offset is set to 'False', only the TDC calibration is
        applied. The default is True.
    absolute_timestamps : bool, optional
        Indicator for data with absolute timestamps. The default is
        False.
    correct_pixel_addressing : bool, optional
        Switch for correcting pixel addressing. The default is False.
    cycle_range : list, optional
        List of specific acquisition cycles to plot. The default is None.
    threshold : int, optional
        Threshold for finding the starting cycle: 0 for the first
        cycle where any signal appears. This parameter should be changed
        only in the case no cycles to plot were given manually. The
        default is 0.
    chosen_file : int, optional
        Index of the chosen file. The default is 0.

    Returns
    -------
    Tuple[np.ndarray, list]
        Tuple containing the sensor population array and the list of
        absolute timestamps.
    """

    os.chdir(path)

    files_all = glob.glob("*.dat*")
    # files_all.sort(key=lambda x: os.path.getctime(x))
    files_all.sort(key=os.path.getmtime)

    if firmware_version == "2212s":
        pix_coor = np.arange(256).reshape(4, 64).T
    elif firmware_version == "2212b":
        pix_coor = np.arange(256).reshape(64, 4)
    else:
        raise ValueError("Firmware version is not recognized.")

    if app_mask is True:
        mask = utils.apply_mask(daughterboard_number, motherboard_number)

    file = files_all[chosen_file]

    if not absolute_timestamps:
        data = f_up.unpack_binary_data(
            file,
            daughterboard_number,
            motherboard_number,
            firmware_version,
            timestamps,
            include_offset,
            apply_calibration,
        )
        sensor_population = np.zeros((256, int(len(data[0]) / (timestamps))))
    else:
        data, _ = f_up.unpack_binary_data_with_absolute_timestamps(
            file,
            daughterboard_number,
            motherboard_number,
            firmware_version,
            timestamps,
            include_offset,
            apply_calibration,
        )
        sensor_population = np.zeros(
            (256, int(len(data[0]) / (timestamps + 1)))
        )

    cycle_ends = np.where(data[0].T[1] == -2)[0]
    cycle_ends = np.insert(cycle_ends, 0, 0)
    indices = np.where(data[0].T[1])[0]

    for j in range(len(cycle_ends) - 1):
        cycle_indices = indices[
            (indices >= cycle_ends[j]) & (indices < cycle_ends[j + 1])
        ]
        for k in range(256):
            if k in mask:
                continue
            tdc, pix = np.argwhere(pix_coor == k)[0]
            pix_index = np.where(data[tdc].T[0][cycle_indices] == pix)[0]
            pix_index_pos = np.where(
                data[tdc].T[1][cycle_indices][pix_index] > 0
            )[0]

            sensor_population[k][j] += len(
                data[tdc].T[1][cycle_indices][pix_index[pix_index_pos]]
            )

    if correct_pixel_addressing:
        fix = np.zeros((256, 2200))
        fix[:128, :] = sensor_population[128:, :]
        fix[128:, :] = np.flip(sensor_population[:128, :])
        sensor_population = fix
        del fix

    try:
        os.chdir(os.path.join(path, "results", "data_quality", "senpop_cycle"))
    except FileNotFoundError:
        os.makedirs(
            os.path.join(path, "results", "data_quality", "senpop_cycle")
        )
        os.chdir(os.path.join(path, "results", "data_quality", "senpop_cycle"))

    # If cycles are not manually given, find the starting cycle
    if cycle_range is None:
        # Find the pixel where a peak is
        pix_peak = np.argmax(sensor_population[:, -1])
        # Find first cycle where height of the peak is above the threshold
        # which can be set to 15 for signal above 3.5 kHz
        cycle_start = np.where(sensor_population[pix_peak, :] > threshold)[
            0
        ].min()
        cycle_range = [x for x in range(cycle_start - 3, cycle_start + 3)]

    plt.rcParams.update({"font.size": 22})
    for _, cycle in enumerate(cycle_range):
        fig = plt.figure(figsize=(10, 6))
        plt.plot(sensor_population[:, cycle], "-.", color="salmon")
        plt.xlabel("Pixel Index")
        plt.ylabel("Sensor Population")
        plt.title(f"Cycle {cycle}")
        plt.tight_layout()

        # Save the figure
        fig.savefig(f"{os.path.splitext(file)[0]}_cycle_{cycle}.png")

        # Close all the figures to free up resources
        plt.close("all")


def pixel_population_by_cycle(
    path,
    pix_to_plot,
    daughterboard_number: str,
    motherboard_number: str,
    firmware_version: str,
    timestamps: int = 512,
    include_offset: bool = True,
    apply_calibration: bool = True,
    absolute_timestamps: bool = False,
    color: str = "salmon",
):
    """
    Collect and plot pixel population data by acquisition cycle.



    Parameters
    ----------
    path : str
        Path to the data files.
    pix_to_plot : int
        Pixel number to plot.
    daughterboard_number : str
        LinoSPAD2 daughterboard number.
    motherboard_number : str
        LinoSPAD2 motherboard (FPGA) number.
    firmware_version : str
        LinoSPAD2 firmware version. Versions "2212s" (skip) and "2212b"
        (block) are recognized.
    timestamps : int, optional
        Number of timestamps per acquisition cycle per pixel. The default
        is 512.
    include_offset : bool, optional
        Switch for applying offset calibration. The default is True.
    apply_calibration : bool, optional
        Switch for applying TDC and offset calibration. If set to 'True'
        while include_offset is set to 'False', only the TDC calibration is
        applied. The default is True.
    absolute_timestamps : bool, optional
        Indicator for data with absolute timestamps. The default is
        False.
    color : str, optional
        Color for the plot. The default is "salmon".

    Returns
    -------
    None
    """
    os.chdir(path)

    files_all = glob.glob("*.dat*")
    # files_all.sort(key=lambda x: os.path.getmtime(x))
    files_all.sort(key=os.path.getmtime)

    # Define matrix of pixel coordinates, where rows are numbers of TDCs
    # and columns are the pixels that connected to these TDCs
    if firmware_version == "2212s":
        pix_coor = np.arange(256).reshape(4, 64).T
    elif firmware_version == "2212b":
        pix_coor = np.arange(256).reshape(64, 4)
    else:
        print("\nFirmware version is not recognized.")
        sys.exit()

    pixel_pop = []

    for i in tqdm(range(len(files_all)), desc="Collecting data"):
        # First board, unpack data
        file = files_all[i]

        if not absolute_timestamps:
            data = f_up.unpack_binary_data(
                file,
                daughterboard_number,
                motherboard_number,
                firmware_version,
                timestamps,
                include_offset,
                apply_calibration,
            )
        else:
            data, _ = f_up.unpack_binary_data_with_absolute_timestamps(
                file,
                daughterboard_number,
                motherboard_number,
                firmware_version,
                timestamps,
                include_offset,
                apply_calibration,
            )

        tdc, pix_c = np.argwhere(pix_coor == pix_to_plot)[0]
        pix = np.where(data[tdc].T[0] == pix_c)[0]

        column_data = data[:].T[1]

        cycle_ends = np.insert(np.where(data[tdc].T[1] == -2)[0], 0, 0)

        for i in range(len(cycle_ends) - 1):
            cycle_indices = pix[
                (pix >= cycle_ends[i]) & (pix < cycle_ends[i + 1])
            ]
            pixel_pop.append(
                len(column_data[cycle_indices][column_data[cycle_indices] > 0])
            )

    # Moving average of number of timestamps per 100 cycles
    moving_average = np.convolve(
        np.array(pixel_pop), np.ones(100) / 100, mode="valid"
    )

    try:
        os.chdir(os.path.join(path, "results", "data_quality", "pixpop_cycle"))
    except FileNotFoundError:
        os.makedirs(
            os.path.join(path, "results", "data_quality", "pixpop_cycle")
        )
        os.chdir(os.path.join(path, "results", "data_quality", "pixpop_cycle"))

    plt.rcParams.update({"font.size": 22})
    plt.figure(figsize=(16, 10))
    plt.plot(pixel_pop, "-.", color=color, label="Pixel population")
    plt.plot(moving_average, color="teal", label="Average in 100 cycles")
    plt.xlabel("Acquisition cycle [-]")
    plt.ylabel("# of timestamps [-]")
    plt.legend()
    plt.title(f"Pixel {pix_to_plot}")

    plot_name = (
        os.path.splitext(files_all[0])[0]
        + "-"
        + os.path.splitext(files_all[-1])[0]
    )
    plt.savefig(f"{plot_name}_{pix_to_plot}.png")


def get_time_ratios(
    path: str,
    daughterboard_number: str,
    motherboard_number: str,
    firmware_version: str,
    timestamps: int,
    include_offset: bool = True,
    apply_calibration: bool = True,
        
):
    """
    Calculate the ratio of time of actual data collection and the time it 
    takes to create and save the files.

    Parameters
    ----------
    path : str
        Path to the binary data file.
    daughterboard_number : str
        LinoSPAD2 daughterboard number.
    motherboard_number : str
        LinoSPAD2 motherboard (FPGA) number.
    firmware_version : str
        LinoSPAD2 firmware version. Either '2212s' (skip) or '2212b' (block).
    timestamps : int, optional
        Number of timestamps per cycle per TDC per acquisition cycle.
    include_offset : bool, optional
        Switch for applying offset calibration. The default is True.
    apply_calibration : bool, optional
        Switch for applying TDC and offset calibration. If set to 'True'
        while include_offset is set to 'False', only the TDC
        calibration is applied. The default is True.
    
    """
    

    os.chdir(path)

    files = glob.glob("*.dat*")
    files.sort(key=os.path.getmtime)


    timestamps_only = []
    number_of_cycles = []
    collecting_times = []

    tmsp_maxes = []

    for i in range(len(files) - 1):
        unpacked_data = f_up.unpack_binary_data(
            files[i],
            daughterboard_number,
            motherboard_number,
            firmware_version,
            timestamps,
            include_offset,
            apply_calibration,
        )

      
        timestamps_only = unpacked_data.T[1]

        tmsp_maxes.append(np.max(timestamps_only))

        number_of_cycles.append(len(np.where(unpacked_data[0].T[0] == -2)[0]))

        collecting_times.append(os.path.getmtime(files[i + 1]) - os.path.getmtime(files[i]))


    tp_max = max(tmsp_maxes) / 1e12

    ratios = [
        number_of_cycles[i] * tp_max / collecting_times[i]
        for i in range(len(collecting_times))
    ]

    ratio_average = np.mean(ratios)
    error = np.std(ratios) / np.sqrt(len(ratios))

    intercycle_time = [
        collecting_times[i] - number_of_cycles[i] * tp_max
        for i in range(len(collecting_times))
    ]

    print("Number of cycles:", number_of_cycles[1])
    print("Ratio average: ", ratio_average)
    print("Ratio error: ", error)


def save_file_times(path):
    """
    Save the creation and modification times of data files into a .feather file.

    Parameters
    ----------
    
    path : str
        Path to the data files.
    
    """
    file_list = glob.glob(os.path.join(path, "*.dat*"))
    files_data = []

    file_list.sort()
    
    first_file_name = os.path.basename(file_list[0])[:-4]  
    last_file_name = os.path.basename(file_list[-1])[:-4]
   
    for file_path in file_list:
        file_name = os.path.basename(file_path)
        creation_time = os.path.getctime(file_path)
        modification_time = os.path.getmtime(file_path)
        files_data.append(
            (
                file_name,
                datetime.fromtimestamp(creation_time),
                datetime.fromtimestamp(modification_time),
            )
        )

    df = pd.DataFrame(
        files_data, columns=["File Name", "Creation Time", "Modification Time"]
    )

    feather_filename = f"{first_file_name}_-_{last_file_name}.feather"
    feather.write_feather(df, os.path.join(path, feather_filename))

    return df


def load_data_from_feather(path):
    """
    Loads the creation and modification time from feather file.

    Parameters
    ----------
    
    path : str
        Path to the data files.
        
    """
    feather_files = glob.glob(os.path.join(path, "*.feather"))
    if feather_files:
        df_loaded = feather.read_feather(feather_files[0])
        return df_loaded
    else:
        print("No feather files found.")
        return None
