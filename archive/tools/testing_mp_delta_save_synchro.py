import functools
import glob
import multiprocessing
import os
import sys
import time
from dataclasses import dataclass

import numpy as np
import pandas as pd
from pyarrow import feather as ft

from daplis.functions import calc_diff as cd
from daplis.functions import unpack as f_up
from daplis.functions import utils


@dataclass
class DataParamsConfig:
    """Configuration parameters for timestamp differences calculation.

    Parameters
    ----------
    pixels : list
        List of pixel numbers for which the timestamp differences should
        be calculated and saved or list of two lists with pixel numbers
        for peak vs. peak calculations.
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
    """

    path: str
    pixels: list
    daughterboard_number: str
    motherboard_number1: str
    motherboard_number2: str
    firmware_version: str
    timestamps: int = 512
    delta_window: float = 50e3
    app_mask: bool = True
    include_offset: bool = True
    apply_calibration: bool = True
    absolute_timestamps: bool = True


def calculate_timestamps_differences_full_sensor(
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
    os.chdir(os.path.join(path, f"{data_params.motherboard_number2}"))

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


def write_results_to_feather(result_queue, feather_file, lock) -> None:
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
    include_offset: bool = True,
    apply_calibration: bool = True,
    chunksize: int = 20,
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
        LinoSPAD2 motherboard (FPGA) number.
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
        path=path,
        pixels=pixels,
        daughterboard_number=daughterboard_number,
        motherboard_number1=motherboard_number1,
        motherboard_number2=motherboard_number2,
        firmware_version=firmware_version,
        timestamps=timestamps,
        delta_window=delta_window,
        app_mask=app_mask,
        include_offset=include_offset,
        apply_calibration=apply_calibration,
    )

    # os.chdir(path)

    # Check the data from the first FPGA board
    try:
        os.chdir(os.path.join(path, f"{motherboard_number1}"))
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"Data from {motherboard_number1} not found") from exc
    files_all1 = sorted(glob.glob("*.dat*"))
    out_file_name = files_all1[0][:-4]
    # os.chdir("..")

    # Check the data from the second FPGA board
    try:
        # os.chdir(f"{motherboard_number2}")
        os.chdir(os.path.join(path, f"{motherboard_number2}"))
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"Data from {motherboard_number2} not found") from exc
    files_all2 = sorted(glob.glob("*.dat*"))
    out_file_name = out_file_name + "-" + files_all2[-1][:-4]
    # os.chdir("..")

    # Check if '.feather' file with timestamps differences already
    # exists
    # os.chdir(os.path.join(path, "delta_ts_data"))
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
                target=write_results_to_feather,
                args=(shared_result_queue, feather_file, shared_lock),
            )
            writer_process.start()

            # Create a partial function with fixed arguments for
            # process_file
            partial_process_file = functools.partial(
                calculate_timestamps_differences_full_sensor,
                result_queue=shared_result_queue,
                data_params=data_params,
            )

            # Start the multicore analysis of the files
            pool.map(partial_process_file, list(files), chunksize=chunksize)

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


if __name__ == "__main__":
    path = r"/home/sj/Shared/05.01.24"
    pixels = [88, 399]
    daughterboard_number = "NL11"
    motherboard_number1 = "#33"
    motherboard_number2 = "#21"
    firmware_version = "2212b"
    timestamps = 300
    delta_window = 50e3
    app_mask = True
    include_offset = False
    apply_calibration = True
    start_time = time.time()
    # multiprocessing.set_start_method('spawn', force=True)
    multiprocessing.freeze_support()
    calculate_and_save_timestamp_differences_full_sensor_mp(
        path,
        pixels,
        True,
        daughterboard_number,
        motherboard_number1,
        motherboard_number2,
        firmware_version,
        timestamps,
        include_offset=include_offset,
        chunksize=2,
    )
    end_time = time.time()
    elapsed_time = end_time - start_time

    print(f"Multiprocessing (all CPU cores), Execution time: {elapsed_time} seconds")
