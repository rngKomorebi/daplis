import functools
import glob
import multiprocessing
import os
import sys
import time

import numpy as np
import pandas as pd
import pyarrow.feather as feather
from tqdm import tqdm

from daplis.functions import calc_diff as cd
from daplis.functions import unpack as f_up
from daplis.functions import utils


def process_file(
    file: str,
    result_queue,
    pixels: list,
    daughterboard_number: str,
    motherboard_number: str,
    firmware_version: str,
    timestamps: int = 512,
    delta_window: float = 50e3,
    app_mask: bool = True,
    include_offset: bool = True,
    apply_calibration: bool = True,
):
    # pixels = [58, 191]
    # daughterboard_number = "NL11"
    # motherboard_number = "#33"
    # firmware_version = "2212b"
    # timestamps = 300
    # delta_window = 50e3
    # app_mask = True
    # include_offset = False
    # apply_calibration = True

    if isinstance(pixels, list) is False:
        raise TypeError(
            "'pixels' should be a list of integers or a list of two lists"
        )
    if isinstance(firmware_version, str) is False:
        raise TypeError(
            "'firmware_version' should be string, '2212s', '2212b' or '2208'"
        )
    if isinstance(daughterboard_number, str) is False:
        raise TypeError("'daughterboard_number' should be string")

    if firmware_version == "2212s":
        pix_coor = np.arange(256).reshape(4, 64).T
    elif firmware_version == "2212b":
        pix_coor = np.arange(256).reshape(64, 4)
    else:
        print("\nFirmware version is not recognized.")
        sys.exit()

    if app_mask is True:
        mask = utils.apply_mask(daughterboard_number, motherboard_number)
        if isinstance(pixels[0], int) and isinstance(pixels[1], int):
            pixels = [pix for pix in pixels if pix not in mask]
        else:
            pixels[0] = [pix for pix in pixels[0] if pix not in mask]
            pixels[1] = [pix for pix in pixels[1] if pix not in mask]

    pixels_left, pixels_right = utils.pixel_list_transform(pixels)

    deltas_all = {}

    data_all = f_up.unpack_binary_data(
        file,
        daughterboard_number,
        motherboard_number,
        firmware_version,
        timestamps,
        include_offset,
        apply_calibration,
    )

    deltas_all = cd.calculate_differences(data_all, pixels, pix_coor)

    for q in pixels_left:
        for w in pixels_right:
            if w <= q:
                continue
            deltas_all["{},{}".format(q, w)] = []
            cycle_ends = np.argwhere(data_all[0].T[0] == -2)
            cycle_ends = np.insert(cycle_ends, 0, 0)
            tdc1, pix_c1 = np.argwhere(pix_coor == q)[0]
            pix1 = np.where(data_all[tdc1].T[0] == pix_c1)[0]
            tdc2, pix_c2 = np.argwhere(pix_coor == w)[0]
            pix2 = np.where(data_all[tdc2].T[0] == pix_c2)[0]

            for cyc in range(len(cycle_ends) - 1):
                pix1_ = pix1[
                    np.logical_and(
                        pix1 >= cycle_ends[cyc], pix1 < cycle_ends[cyc + 1]
                    )
                ]
                if not np.any(pix1_):
                    continue
                pix2_ = pix2[
                    np.logical_and(
                        pix2 >= cycle_ends[cyc], pix2 < cycle_ends[cyc + 1]
                    )
                ]
                if not np.any(pix2_):
                    continue

                tmsp1 = data_all[tdc1].T[1][
                    pix1_[np.where(data_all[tdc1].T[1][pix1_] > 0)[0]]
                ]
                tmsp2 = data_all[tdc2].T[1][
                    pix2_[np.where(data_all[tdc2].T[1][pix2_] > 0)[0]]
                ]
                for t1 in tmsp1:
                    deltas = tmsp2 - t1
                    ind = np.where(np.abs(deltas) < delta_window)[0]
                    deltas_all["{},{}".format(q, w)].extend(deltas[ind])

    data_for_plot_df = pd.DataFrame.from_dict(deltas_all, orient="index")

    result_queue.put(data_for_plot_df.T)


def write_results_to_feather(result_queue, feather_file, lock):
    while True:
        result_df = result_queue.get()
        if result_df is None:
            break

        # Use a lock to prevent conflicts when writing to the file
        with lock:
            if os.path.exists(feather_file):
                existing_data = feather.read_feather(feather_file)
                combined_data = pd.concat([existing_data, result_df], axis=0)
            else:
                combined_data = result_df.copy()

            # Reset the index to avoid issues during feather.write_feather
            combined_data.reset_index(drop=True, inplace=True)

            # Write the combined data to the Feather file
            feather.write_feather(combined_data, feather_file)


def calculate_and_save_timestamp_differences_mp(
    path: str,
    pixels: list,
    daughterboard_number: str,
    motherboard_number: str,
    firmware_version: str,
    timestamps: int = 512,
    delta_window: float = 50e3,
    app_mask: bool = True,
    include_offset: bool = True,
    apply_calibration: bool = True,
    rewrite: bool = False,
):
    os.chdir(path)

    files = sorted(glob.glob("*.dat"))
    print(files)

    feather_file_name = files[0][:-4] + "-" + files[-1][:-4] + ".feather"

    feather_file = os.path.join(path, "delta_ts_data", feather_file_name)

    utils.file_rewrite_handling(feather_file, rewrite)

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
                process_file,
                result_queue=shared_result_queue,
                pixels=pixels,
                daughterboard_number=daughterboard_number,
                motherboard_number=motherboard_number,
                firmware_version=firmware_version,
                timestamps=timestamps,
                delta_window=delta_window,
                app_mask=app_mask,
                include_offset=include_offset,
                apply_calibration=apply_calibration,
            )

            arg_list = [
                (
                    file,
                    shared_result_queue,
                    pixels,
                    daughterboard_number,
                    motherboard_number,
                    firmware_version,
                    timestamps,
                    delta_window,
                    app_mask,
                    include_offset,
                    apply_calibration,
                )
                for file in files
            ]

            pool.starmap(partial_process_file, arg_list, chunksize=30)

            # Use tqdm to create a progress bar for the file processing
            # Can't configure tqdm to update the progress bar correctly
            # while using chunksize in pool

            # with tqdm(total=len(files), desc="Processing files") as pbar:
            #     for _ in pool.imap_unordered(partial_process_file, files):
            #         pbar.update(1)

            # Signal the writer process that no more results will be
            # added to the queue
            shared_result_queue.put(None)
            writer_process.join()


if __name__ == "__main__":
    path = "/home/sj/LS2_Data/703"
    pixels = [58, 191]
    daughterboard_number = "NL11"
    motherboard_number = "#33"
    firmware_version = "2212b"
    timestamps = 300
    delta_window = 50e3
    app_mask = True
    include_offset = False
    apply_calibration = True
    start_time = time.time()
    # multiprocessing.set_start_method('spawn', force=True)
    multiprocessing.freeze_support()
    calculate_and_save_timestamp_differences_mp(
        path,
        pixels,
        daughterboard_number,
        motherboard_number,
        firmware_version,
        timestamps,
        include_offset=include_offset,
        rewrite=True,
    )
    end_time = time.time()
    elapsed_time = end_time - start_time

    print(
        f"Multiprocessing (all CPU cores), Execution time: {elapsed_time} seconds"
    )


# file = "/home/sj/LS2_Data/703/MP_RESULTS/out.feather"
# # file1 = "/home/sj/LS2_Data/703/delta_ts_data/0000004613-0000004847.feather"
# data = feather.read_feather(file)
