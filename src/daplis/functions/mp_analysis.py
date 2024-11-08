"""This module contains functions for LS2 data analysis that utilize
the multiprocessing Python library for speeding up the analysis
by using all available CPU cores instead of a single one.

This module can be imported with the following functions:

* calculate_and_save_timestamp_differences_mp - unpacks the binary data,
    calculates timestamp differences, and saves into a '.feather' file.
    Works with firmware versions '2208' and '2212b'. The multiprocessing
    version that utilizes all available CPU cores.
    
* calculate_and_save_timestamp_differences_full_sensor_mp - unpacks the
    binary data, calculates timestamp differences and saves into a
    '.feather' file. Works with firmware versions '2208', '2212s' and
    '2212b'. Analyzes data from both sensor halves/both FPGAs. Useful
    for data where the signal is at 0 across the whole sensor from the
    start of data collecting. The multiprocessing version that
    utilizies all CPU cores available.

* compact_share_mp - unpacks all '.dat' files in the given folder,
    collects the number of timestamps in each pixel and packs it into a
    '.txt' file, calculates timestamp differences and packs them into a 
    '.feather' file. The multiprocessing version that utilizes all
    available CPU cores.

"""

import functools
import glob
import multiprocessing
import os
import sys
from dataclasses import dataclass
from zipfile import ZipFile

import numpy as np
import pandas as pd
from daplis.functions import calc_diff as cd
from daplis.functions import unpack as f_up
from daplis.functions import utils
from pyarrow import feather as ft

# from tqdm import tqdm


@dataclass
class DataParamsConfig:
    """Configuration parameters for timestamp differences calculation.

    Parameters
    ----------
    pixels : list
        List of pixel numbers for which the timestamp differences should
        be calculated and saved or list of two lists with pixel numbers
        for peak vs. peak calculations.
    path : str
        Path to data files.
    daughterboard_number : str
        LinoSPAD2 daughterboard number.
    motherboard_number : str
        LinoSPAD2 motherboard (FPGA) number, including the "#".
    motherboard_number1 : str
        First LinoSPAD2 motherboard (FPGA) number, including the "#".
        Used for full sensor data analysis.
    motherboard_number2 : str
        Second LinoSPAD2 motherboard (FPGA) number. Used for full sensor
        data analysis.
    firmware_version : str
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
    absolute_timestamps : bool, optional
        Indicator for data with absolute timestamps. The default is
        False.
    """

    pixels: list
    path: str = ""
    daughterboard_number: str = ""
    motherboard_number: str = ""
    motherboard_number1: str = ""
    motherboard_number2: str = ""
    firmware_version: str = ""
    timestamps: int = 512
    delta_window: float = 50e3
    app_mask: bool = True
    include_offset: bool = True
    apply_calibration: bool = True
    absolute_timestamps: bool = False


def _calculate_timestamps_differences(
    file: str,
    result_queue: multiprocessing.Queue,
    data_params: DataParamsConfig,
) -> None:
    """Unpack data and collect timestamps differences for single file.

    Unpack the '.dat' data file, collect timestamps differences in the
    given window ("delta_window") and add them to the shared
    interprocess queue for further manipulations.

    Parameters
    ----------
    file : str
        Data file name, relative path.
    result_queue : multiprocessing.Queue
        Interprocess queue created using the multiprocessing.Queue method.
    data_params : DataParamsConfig
        An instance of the DataParamsConfig dataclass containing various
        configuration parameters for data processing.

    Returns
    -------
    None.
    """

    if data_params.firmware_version == "2212s":
        pix_coor = np.arange(256).reshape(4, 64).T
    elif data_params.firmware_version == "2212b":
        pix_coor = np.arange(256).reshape(64, 4)
    else:
        print("\nFirmware version is not recognized.")
        sys.exit()

    if data_params.app_mask is True:
        mask = utils.apply_mask(
            data_params.daughterboard_number, data_params.motherboard_number
        )
        if isinstance(data_params.pixels[0], int) and isinstance(
            data_params.pixels[1], int
        ):
            pixels = [pix for pix in data_params.pixels if pix not in mask]
        else:
            pixels = [pix for pix in data_params.pixels[0] if pix not in mask]
            pixels.extend(pix for pix in data_params.pixels[1] if pix not in mask)

    deltas_all = {}

    data_all = f_up.unpack_binary_data(
        file,
        data_params.daughterboard_number,
        data_params.motherboard_number,
        data_params.firmware_version,
        data_params.timestamps,
        data_params.include_offset,
        data_params.apply_calibration,
    )

    deltas_all = cd.calculate_differences_2212(data_all, pixels, pix_coor)

    data_for_plot_df = pd.DataFrame.from_dict(deltas_all, orient="index")

    result_queue.put(data_for_plot_df.T)


def _calculate_timestamps_differences_full_sensor(
    files: str,
    result_queue: multiprocessing.Queue,
    data_params: DataParamsConfig,
):
    # TODO add option for collecting from more than just two pixels
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
        Switch for rewriting the '.feather' file if it already exists.
    daughterboard_number : str
        LinoSPAD2 daughterboard number.
    motherboard_number1 : str
        First LinoSPAD2 motherboard (FPGA) number.
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
    app_mask : bool, optional
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
    # # parameter type check
    # if isinstance(data_params.pixels, list) is False:
    #     raise TypeError(
    #         "'pixels' should be a list of integers or a list of two lists"
    #     )
    # if isinstance(data_params.firmware_version, str) is False:
    #     raise TypeError(
    #         "'firmware_version' should be string, '2212s', '2212b' or" "'2208'"
    #     )
    # if isinstance(data_params.rewrite, bool) is False:
    #     raise TypeError("'rewrite' should be boolean")
    # if isinstance(data_params.daughterboard_number, str) is False:
    #     raise TypeError("'daughterboard_number' should be string")

    # Define matrix of pixel coordinates, where rows are numbers of TDCs
    # and columns are the pixels that connected to these TDCs
    if data_params.firmware_version == "2212s":
        pix_coor = np.arange(256).reshape(4, 64).T
    elif data_params.firmware_version == "2212b":
        pix_coor = np.arange(256).reshape(64, 4)
    else:
        print("\nFirmware version is not recognized.")
        sys.exit()

    # abs_tmsp_list1 = []
    # abs_tmsp_list2 = []

    deltas_all = {}

    # First board, unpack data
    os.chdir(os.path.join(data_params.path, f"{data_params.motherboard_number1}"))

    if not data_params.absolute_timestamps:
        data_all1 = f_up.unpack_binary_data(
            files[0],
            data_params.daughterboard_number,
            data_params.motherboard_number1,
            data_params.firmware_version,
            data_params.timestamps,
            data_params.include_offset,
            data_params.apply_calibration,
        )
    else:
        (
            data_all1,
            abs_tmsp1,
        ) = f_up.unpack_binary_data_with_absolute_timestamps(
            files[0],
            data_params.daughterboard_number,
            data_params.motherboard_number1,
            data_params.firmware_version,
            data_params.timestamps,
            data_params.include_offset,
            data_params.apply_calibration,
        )
    # abs_tmsp_list1.append(abs_tmsp1)

    # Collect indices of cycle ends (the '-2's)
    cycle_ends1 = np.where(data_all1[0].T[1] == -2)[0]
    cyc1 = np.argmin(np.abs(cycle_ends1 - np.where(data_all1[:].T[1] > 0)[0].min()))
    if cycle_ends1[cyc1] > np.where(data_all1[:].T[1] > 0)[0].min():
        cycle_start1 = cycle_ends1[cyc1 - 1]
    else:
        cycle_start1 = cycle_ends1[cyc1]

    # Second board, unpack data
    os.chdir(os.path.join(data_params.path, f"{data_params.motherboard_number2}"))

    if not data_params.absolute_timestamps:
        data_all2 = f_up.unpack_binary_data(
            files[1],
            data_params.daughterboard_number,
            data_params.motherboard_number2,
            data_params.firmware_version,
            data_params.timestamps,
            data_params.include_offset,
            data_params.apply_calibration,
        )
    else:
        (
            data_all2,
            abs_tmsp2,
        ) = f_up.unpack_binary_data_with_absolute_timestamps(
            files[1],
            data_params.daughterboard_number,
            data_params.motherboard_number2,
            data_params.firmware_version,
            data_params.timestamps,
            data_params.include_offset,
            data_params.apply_calibration,
        )
    # abs_tmsp_list2.append(abs_tmsp2)

    # Collect indices of cycle ends (the '-2's)
    cycle_ends2 = np.where(data_all2[0].T[1] == -2)[0]
    cyc2 = np.argmin(np.abs(cycle_ends2 - np.where(data_all2[:].T[1] > 0)[0].min()))
    if cycle_ends2[cyc2] > np.where(data_all2[:].T[1] > 0)[0].min():
        cycle_start2 = cycle_ends2[cyc2 - 1]
    else:
        cycle_start2 = cycle_ends2[cyc2]

    pix_left_peak = data_params.pixels[0]
    # The following piece of code take any value for the pixel from
    # the second motherboard, either in terms of full sensor (so
    # a value >=256) or in terms of single sensor half

    if data_params.pixels[1] >= 256:
        if data_params.pixels[1] > 256 + 127:
            pix_right_peak = 255 - (data_params.pixels[1] - 256)
        else:
            pix_right_peak = data_params.pixels[1] - 256 + 128
    elif data_params.pixels[1] > 127:
        pix_right_peak = 255 - data_params.pixels[1]
    else:
        pix_right_peak = data_params.pixels[1] + 128

    # Get the data from the requested pixel only
    deltas_all[f"{pix_left_peak},{pix_right_peak}"] = []
    tdc1, pix_c1 = np.argwhere(pix_coor == pix_left_peak)[0]
    pix1 = np.where(data_all1[tdc1].T[0] == pix_c1)[0]
    tdc2, pix_c2 = np.argwhere(pix_coor == pix_right_peak)[0]
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
            np.logical_and(pix1 >= cycle_ends1[cyc], pix1 < cycle_ends1[cyc + 1])
        ]
        if not np.any(pix1_):
            continue
        pix2_ = pix2[
            np.logical_and(pix2 >= cycle_ends2[cyc], pix2 < cycle_ends2[cyc + 1])
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
            ind = np.where(np.abs(deltas) < data_params.delta_window)[0]
            deltas_all[f"{pix_left_peak},{pix_right_peak}"].extend(deltas[ind])

    # Save data to a feather file in a cycle so data is not lost
    # in the case of failure close to the end
    data_for_plot_df = pd.DataFrame.from_dict(deltas_all, orient="index")
    del deltas_all

    result_queue.put(data_for_plot_df.T)


def _write_results_to_feather(result_queue, feather_file, lock) -> None:
    """Save or append data to a Feather file.

    Parameters
    ----------
    result_queue : multiprocessing.Queue
        Interprocess queue containing data to be saved or appended to
        the Feather file.
    feather_file : str
        Absolute path to the Feather file.
    lock : multiprocessing.Lock
        Shared interprocess lock for synchronizing access to the Feather
        file.

    Notes
    -----
    The data in `result_queue` is retrieved and combined with the
    existing data in the Feather file, or a new file is created if it
    doesn't exist.
    The process is synchronized using the shared `lock` to prevent
    conflicts during file access.
    """
    while True:
        result_df = result_queue.get()
        if result_df is None:
            break

        # Use a lock to prevent conflicts when writing to the file
        with lock:
            if os.path.exists(feather_file):
                existing_data = ft.read_feather(feather_file)
                combined_data = pd.concat([existing_data, result_df], axis=0)
            else:
                combined_data = result_df.copy()

            # Reset the index to avoid issues during feather.write_feather
            combined_data.reset_index(drop=True, inplace=True)

            # Write the combined data to the Feather file
            ft.write_feather(combined_data, feather_file)


def _write_results_to_txt(result_queue_txt, txt_file, lock) -> None:
    """Save or append data to a txt file.

    Parameters
    ----------
    result_queue : multiprocessing.Queue
        Interprocess queue containing data to be saved or appended to
        the txt file.
    txt_file : str
        Absolute path to the txt file.
    lock : multiprocessing.Lock
        Shared interprocess lock for synchronizing access to the Feather
        file.

    Notes
    -----
    The data in `result_queue` is retrieved and combined with the
    existing data in the txt file, or a new file is created if it
    doesn't exist.
    The process is synchronized using the shared `lock` to prevent
    conflicts during file access.
    """
    while True:
        result_df = result_queue_txt.get()
        if result_df is None:
            break

        # Use a lock to prevent conflicts when writing to the file
        with lock:
            if os.path.exists(txt_file):
                existing_data = np.genfromtxt(txt_file, delimiter="\t", dtype=int)
                combined_data = existing_data + result_df
            else:
                combined_data = result_df.copy()

            # Write the combined data to the Feather file
            np.savetxt(
                txt_file,
                combined_data,
                fmt="%d",
                delimiter="\t",
            )


def calculate_and_save_timestamp_differences_mp(
    path: str,
    pixels: list,
    rewrite: bool,
    daughterboard_number: str,
    motherboard_number: str,
    firmware_version: str,
    timestamps: int,
    delta_window: float = 50e3,
    app_mask: bool = True,
    include_offset: bool = False,
    apply_calibration: bool = True,
    chunksize: int = 20,
    number_of_cores: int = 4,
    maxtasksperchild: int = 1000,
) -> None:
    """Unpack data and collect timestamps differences using all CPU cores.

    Unpack data files and collect timestamps differences using all
    available CPU cores to speed up the process, while saving the results
    to a single Feather file on the go.

    Parameters
    ----------
    path : str
        Path to data files.
    pixels : list
        List of pixel numbers for which the timestamp differences should
        be calculated and saved or list of two lists with pixel numbers
        for peak vs. peak calculations.
    rewrite : bool
        Switch for rewriting the '.feather' file if it already exists.
    daughterboard_number : str
        LinoSPAD2 daughterboard number.
    motherboard_number : str
        LinoSPAD2 motherboard (FPGA) number, including the "#".
    firmware_version: str
        LinoSPAD2 firmware version. Versions "2212s" (skip) and "2212b"
        (block) are recognized.
    timestamps : int, optional
        Number of timestamps per acquisition cycle per pixel.
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
    chunksize : int, optional
        Number of files processed in each iteration. The default is 20.
    number_of_cores : int, optional
        Number of cores to use for multiprocessing. The default is 4.
    maxtasksperchild : int, optional
        Number of tasks per core. The default is 1000.
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
    """
    # parameter type check
    if isinstance(pixels, list) is False:
        raise TypeError("'pixels' should be a list of integers or a list of two lists")
    if isinstance(firmware_version, str) is False:
        raise TypeError(
            "'firmware_version' should be string, '2212s', '2212b' or '2208'"
        )
    if isinstance(rewrite, bool) is False:
        raise TypeError("'rewrite' should be boolean")
    if isinstance(daughterboard_number, str) is False:
        raise TypeError("'daughterboard_number' should be string")

    # Generate a dataclass object
    data_params = DataParamsConfig(
        pixels=pixels,
        daughterboard_number=daughterboard_number,
        motherboard_number=motherboard_number,
        firmware_version=firmware_version,
        timestamps=timestamps,
        delta_window=delta_window,
        app_mask=app_mask,
        include_offset=include_offset,
        apply_calibration=apply_calibration,
        absolute_timestamps=True,
    )

    os.chdir(path)

    # Find all LinoSPAD2 data files
    # files = sorted(glob.glob("*.dat"))
    files = glob.glob("*.dat*")
    files.sort(key=os.path.getmtime)
    # Get the resulting Feather file name based on the data files
    # found
    feather_file_name = files[0][:-4] + "-" + files[-1][:-4] + ".feather"
    # Construct absolute path to the Feather file
    feather_file = os.path.join(path, "delta_ts_data", feather_file_name)
    # Handle the rewrite parameter based on the file existence to avoid
    # accidental file overwritting
    utils.file_rewrite_handling(feather_file, rewrite)

    with multiprocessing.Manager() as manager:
        shared_result_queue = manager.Queue()
        shared_lock = manager.Lock()

        with multiprocessing.Pool(
            processes=number_of_cores, maxtasksperchild=maxtasksperchild
        ) as pool:
            # Start the writer process
            writer_process = multiprocessing.Process(
                target=_write_results_to_feather,
                args=(shared_result_queue, feather_file, shared_lock),
            )
            writer_process.start()

            # Create a partial function with fixed arguments for
            # process_file
            partial_process_file = functools.partial(
                _calculate_timestamps_differences,
                result_queue=shared_result_queue,
                data_params=data_params,
            )

            # Start the multicore analysis of the files
            pool.map(partial_process_file, files, chunksize=chunksize)

            # pool.close()
            # pool.join()
            # Use tqdm to create a progress bar for the file processing
            # TODO Can't configure tqdm to update the progress bar
            # correctly while using chunksize in pool

            # with tqdm(total=len(files), desc="Processing files") as pbar:
            #     for _ in pool.imap_unordered(partial_process_file, files):
            #         pbar.update(1)

            # Signal the writer process that no more results will be
            # added to the queue
            shared_result_queue.put(None)
            writer_process.join()


def calculate_and_save_timestamp_differences_full_sensor_mp(
    path: str,
    pixels: list,
    rewrite: bool,
    daughterboard_number: str,
    motherboard_number1: str,
    motherboard_number2: str,
    firmware_version: str,
    timestamps: int,
    delta_window: float = 50e3,
    app_mask: bool = True,
    include_offset: bool = False,
    apply_calibration: bool = True,
    absolute_timestamps: bool = True,
    chunksize: int = 20,
) -> None:
    """Unpack data and collect timestamps differences using all CPU cores.

    Unpack data files and collect timestamps differences using all
    available CPU cores to speed up the process, while saving the results
    to a single Feather file on the go. Version for analyzing data
    from both halves of the sensor.

    Parameters
    ----------
    path : str
        Path to data files.
    pixels : list
        List of pixel numbers for which the timestamp differences should
        be calculated and saved or list of two lists with pixel numbers
        for peak vs. peak calculations.
    rewrite : bool
        Switch for rewriting the '.feather' file if it already exists.
    daughterboard_number : str
        LinoSPAD2 daughterboard number.
    motherboard_number1 : str
        First LinoSPAD2 motherboard (FPGA) number.
    motherboard_number2 : str
        Second LinoSPAD2 motherboard (FPGA) number.
    firmware_version: str
        LinoSPAD2 firmware version. Versions "2212s" (skip) and "2212b"
        (block) are recognized.
    timestamps : int, optional
        Number of timestamps per acquisition cycle per pixel.
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
    absolute_timestamps : bool, False
        Indicator for data with absolute timestamps. The default is
        False.
    chunksize : int, optional
        Number of files processed in each iteration. The default is 20.

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
    """
    # parameter type check
    if isinstance(pixels, list) is False:
        raise TypeError("'pixels' should be a list of integers or a list of two lists")
    if isinstance(firmware_version, str) is False:
        raise TypeError(
            "'firmware_version' should be string, '2212s', '2212b' or '2208'"
        )
    if isinstance(rewrite, bool) is False:
        raise TypeError("'rewrite' should be boolean")
    if isinstance(daughterboard_number, str) is False:
        raise TypeError("'daughterboard_number' should be string")

    # Generate a dataclass object
    data_params = DataParamsConfig(
        pixels=pixels,
        path=path,
        daughterboard_number=daughterboard_number,
        motherboard_number1=motherboard_number1,
        motherboard_number2=motherboard_number2,
        firmware_version=firmware_version,
        timestamps=timestamps,
        delta_window=delta_window,
        app_mask=app_mask,
        include_offset=include_offset,
        apply_calibration=apply_calibration,
        absolute_timestamps=absolute_timestamps,
    )

    # Check the data from the first FPGA board
    try:
        os.chdir(os.path.join(path, f"{motherboard_number1}"))
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"Data from {motherboard_number1} not found") from exc
    # files_all1 = sorted(glob.glob("*.dat*"))
    files_all1 = glob.glob("*.dat*")
    files_all1.sort(key=os.path.getmtime)
    out_file_name = files_all1[0][:-4]

    # Check the data from the second FPGA board
    try:
        # os.chdir(f"{motherboard_number2}")
        os.chdir(os.path.join(path, f"{motherboard_number2}"))
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"Data from {motherboard_number2} not found") from exc
    # files_all2 = sorted(glob.glob("*.dat*"))
    files_all2 = glob.glob("*.dat*")
    files_all2.sort(key=os.path.getmtime)
    out_file_name = out_file_name + "-" + files_all2[-1][:-4]

    # Check if '.feather' file with timestamps differences already
    # exists
    out_file_name = os.path.join(path, "delta_ts_data", out_file_name)
    feather_file = f"{out_file_name}.feather"

    utils.file_rewrite_handling(feather_file, rewrite)

    files = zip(files_all1, files_all2)

    with multiprocessing.Manager() as manager:
        shared_result_queue = manager.Queue()
        shared_lock = manager.Lock()

        with multiprocessing.Pool() as pool:
            # Start the writer process
            writer_process = multiprocessing.Process(
                target=_write_results_to_feather,
                args=(shared_result_queue, feather_file, shared_lock),
            )
            writer_process.start()

            # Create a partial function with fixed arguments for
            # process_file
            partial_process_file = functools.partial(
                _calculate_timestamps_differences_full_sensor,
                result_queue=shared_result_queue,
                data_params=data_params,
            )

            # Start the multicore analysis of the files
            pool.map(partial_process_file, list(files), chunksize=chunksize)

            # Use tqdm to create a progress bar for the file processing
            # TODO Can't configure tqdm to update the progress bar
            # correctly while using chunksize in pool

            # with tqdm(total=len(files), desc="Processing files") as pbar:
            #     for _ in pool.imap_unordered(partial_process_file, files):
            #         pbar.update(1)

            # Signal the writer process that no more results will be
            # added to the queue
            shared_result_queue.put(None)
            writer_process.join()


def _compact_share_collect_data(
    file: str,
    result_queue_feather: multiprocessing.Queue,
    result_queue_txt: multiprocessing.Queue,
    data_params: DataParamsConfig,
):
    """Collect delta timestamp differences and sensor population from
    unpacked data, saving results to '.feather' and '.txt' files.

    This function processes data from a single .dat file, calculates
    timestamp differences and collects sensor population numbers. The
    resulting arrays are put each to a separate multiprocessing queue
    for saving via an appropriate function.

    Parameters
    ----------
    file : str
        Path to the data file to be processed.
    result_queue_feather : multiprocessing.Queue
        Queue for storing the timestamp differences in '.feather' format.
    result_queue_txt : multiprocessing.Queue
        Queue for storing the sensor population in '.txt' format.
    data_params : DataParamsConfig
        Configuration object containing parameters for data processing.

    Returns
    -------
    None.
    """

    os.chdir(data_params.path)

    # Collect the data for the required pixels

    # for transforming pixel number into TDC number + pixel
    # coordinates in that TDC
    if data_params.firmware_version == "2212s":
        pix_coor = np.arange(256).reshape(4, 64).T
    elif data_params.firmware_version == "2212b":
        pix_coor = np.arange(256).reshape(64, 4)
    else:
        sys.exit()

    # Prepare array for sensor population
    valid_per_pixel = np.zeros(256, dtype=int)

    # Prepare a dictionary for timestamp differences
    deltas_all = {}

    # Unpack data
    if not data_params.absolute_timestamps:
        data_all = f_up.unpack_binary_data(
            file,
            data_params.daughterboard_number,
            data_params.motherboard_number,
            data_params.firmware_version,
            data_params.timestamps,
            data_params.include_offset,
            data_params.apply_calibration,
        )
    else:
        data_all, _ = f_up.unpack_binary_data_with_absolute_timestamps(
            file,
            data_params.daughterboard_number,
            data_params.motherboard_number,
            data_params.firmware_version,
            data_params.timestamps,
            data_params.include_offset,
            data_params.apply_calibration,
        )

    deltas_all = cd.calculate_differences_2212(
        data_all, data_params.pixels, pix_coor, data_params.delta_window
    )

    # Collect sensor population
    for k in range(256):
        tdc, pix = np.argwhere(pix_coor == k)[0]
        valid_per_pixel[k] += np.count_nonzero(data_all[tdc][:, 0] == pix)

    # Save data as a .feather file in a cycle so data is not lost
    # in the case of failure close to the end
    data_for_plot_df = pd.DataFrame.from_dict(deltas_all, orient="index")
    del deltas_all

    result_queue_feather.put(data_for_plot_df.T)
    result_queue_txt.put(valid_per_pixel)


def compact_share_mp(
    path: str,
    pixels: list,
    rewrite: bool,
    daughterboard_number: str,
    motherboard_number: str,
    firmware_version: str,
    timestamps: int,
    delta_window: float = 50e3,
    include_offset: bool = False,
    apply_calibration: bool = True,
    absolute_timestamps: bool = False,
    chunksize=None,
):
    """Collect timestamp differences and sensor population using
    multiprocessing, saving results to '.feather' and '.txt' files.

    This function parallelizes the processing of data in the specified
    path using multiprocessing. It calculates timestamp differences and
    sensor population for the specified pixels based on the provided
    parameters, saving the timestamp differences to a '.feather' file and
    the sensor population to a '.txt' file. Both files are then zipped for
    compact output ready to share.

    Parameters
    ----------
    path : str
        Path to the directory containing data files.
    pixels : list
        List of pixel numbers for which the timestamp differences should
        be calculated and saved, or list of two lists with pixel numbers
        for peak vs. peak calculations.
    rewrite : bool
        Switch for rewriting the '.feather' file if it already exists.
    daughterboard_number : str
        LinoSPAD2 daughterboard number.
    motherboard_number : str
        LinoSPAD2 motherboard (FPGA) number, including the "#".
    firmware_version: str
        LinoSPAD2 firmware version. Accepted values are "2212s" (skip)
        and "2212b" (block).
    timestamps : int
        Number of timestamps per acquisition cycle per pixel.
    delta_window : float, optional
        Size of a window (in nanoseconds) to which timestamp differences
        are compared (default is 50e3 nanoseconds).
    include_offset : bool, optional
        Switch for applying offset calibration (default is True).
    apply_calibration : bool, optional
        Switch for applying TDC and offset calibration. If set to 'True'
        while include_offset is set to 'False', only the TDC calibration is
        applied (default is True).
    absolute_timestamps : bool, optional
        Indicator for data with absolute timestamps (default is False).
    chunksize : int, optional
        The number of data points processed in each batch by each worker.
        If None, the default chunk size is determined automatically.

    Raises
    ------
    TypeError
        Only boolean values of 'rewrite' and string values of
        'firmware_version' are accepted. The first error is raised so
        that the files are not accidentally overwritten in the case of
        unclear input.

    Returns
    -------
    None.
    """

    # parameter type check
    if isinstance(pixels, list) is False:
        raise TypeError("'pixels' should be a list of integers or a list of two lists")
    if isinstance(firmware_version, str) is False:
        raise TypeError(
            "'firmware_version' should be string, '2212s', '2212b' or '2208'"
        )
    if isinstance(rewrite, bool) is False:
        raise TypeError("'rewrite' should be boolean")
    if isinstance(daughterboard_number, str) is False:
        raise TypeError("'daughterboard_number' should be string")
    if isinstance(motherboard_number, str) is False:
        raise TypeError("'motherboard_number' should be string")

    # Generate a dataclass object
    data_params = DataParamsConfig(
        pixels=pixels,
        path=path,
        daughterboard_number=daughterboard_number,
        motherboard_number=motherboard_number,
        firmware_version=firmware_version,
        timestamps=timestamps,
        delta_window=delta_window,
        include_offset=include_offset,
        apply_calibration=apply_calibration,
        absolute_timestamps=absolute_timestamps,
    )

    os.chdir(path)

    # Find all LinoSPAD2 data files
    # files = sorted(glob.glob("*.dat"))
    files = glob.glob("*.dat*")
    files.sort(key=os.path.getmtime)
    # Get the resulting Feather file name based on the data files
    # found
    feather_file_name = files[0][:-4] + "-" + files[-1][:-4] + ".feather"
    txt_file_name = files[0][:-4] + "-" + files[-1][:-4] + ".txt"
    # Construct absolute path to the Feather file
    feather_file = os.path.join(path, "compact_share", feather_file_name)
    txt_file = os.path.join(path, "compact_share", txt_file_name)
    # Handle the rewrite parameter based on the file existence to avoid
    # accidental file overwritting
    utils.file_rewrite_handling(feather_file, rewrite)
    utils.file_rewrite_handling(txt_file, rewrite)

    with multiprocessing.Manager() as manager:
        shared_result_feather = manager.Queue()
        shared_result_txt = manager.Queue()
        shared_lock_feather = manager.Lock()
        shared_lock_txt = manager.Lock()

        with multiprocessing.Pool() as pool:
            # Start the writer process
            writer_process_feather = multiprocessing.Process(
                target=_write_results_to_feather,
                args=(
                    shared_result_feather,
                    feather_file,
                    shared_lock_feather,
                ),
            )

            writer_process_txt = multiprocessing.Process(
                target=_write_results_to_txt,
                args=(shared_result_txt, txt_file, shared_lock_txt),
            )

            writer_process_feather.start()
            writer_process_txt.start()

            # Create a partial function with fixed arguments for
            # process_file
            partial_process_file = functools.partial(
                _compact_share_collect_data,
                result_queue_feather=shared_result_feather,
                result_queue_txt=shared_result_txt,
                data_params=data_params,
            )

            # Start the multicore analysis of the files
            if chunksize is None:
                pool.map(partial_process_file, files)

            else:
                pool.map(partial_process_file, files, chunksize=chunksize)

            # Signal the writer process that no more results will be
            # added to the queue
            shared_result_feather.put(None)
            shared_result_txt.put(None)
            writer_process_feather.join()
            writer_process_txt.join()

        # Create a ZipFile Object
        os.chdir(os.path.join(path, "compact_share"))
        out_file_name, _ = os.path.splitext(feather_file_name)

        with ZipFile(f"{out_file_name}.zip", "w") as zip_object:
            # Adding files that need to be zipped
            zip_object.write(f"{out_file_name}.feather")
            zip_object.write(f"{out_file_name}.txt")

            print(
                "\n> > > Timestamp differences are saved as {feather_file} and "
                "sensor population as {txt_file} in "
                "{path} < < <".format(
                    feather_file=feather_file,
                    txt_file=txt_file,
                    path=path + "\delta_ts_data",
                )
            )
