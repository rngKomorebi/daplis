import glob
import multiprocessing
import os
import sys
import time

import numpy as np

from daplis.functions import unpack as f_up
from daplis.functions import utils


# Step 1: Function to Process a Single File
def process_file(file) -> np.ndarray:
    daughterboard_number = "NL11"
    motherboard_number = "#33"
    firmware_version = "2212b"
    timestamps = 300
    include_offset = False
    apply_calibration = True
    app_mask = True

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

    data = f_up.unpack_binary_data(
        file,
        daughterboard_number,
        motherboard_number,
        firmware_version,
        timestamps,
        include_offset,
        apply_calibration,
    )
    for i in range(256):
        tdc, pix = np.argwhere(pix_coor == i)[0]
        ind = np.where(data[tdc].T[0] == pix)[0]
        ind1 = np.where(data[tdc].T[1][ind] > 0)[0]
        timestamps_per_pixel[i] += len(data[tdc].T[1][ind[ind1]])

    # Apply mask if requested
    if app_mask:
        mask = utils.apply_mask(daughterboard_number, motherboard_number)
        timestamps_per_pixel[mask] = 0

    # timestamps_per_pixel = timestamps_per_pixel
    return timestamps_per_pixel


def write_results_to_file(result_queue, output_file, lock):
    with open(output_file, "ab") as f:  # 'ab' for appending in binary mode
        while True:
            result_array = result_queue.get()
            if result_array is None:
                break

            # Use a lock to prevent conflicts when writing to the file
            with lock:
                np.savetxt(f, [result_array], fmt="%d")


# def main():
#     input_folder = "/home/sj/LS2_Data/703"
#     output_file = "/home/sj/LS2_Data/703/MP_RESULTS/out.txt"

#     os.chdir(input_folder)

#     files = glob.glob("*.dat")

#     # Create a multiprocessing queue to pass results from worker processes to the writer process
#     result_queue = multiprocessing.Queue()

#     # Create a lock to synchronize access to the output file
#     lock = multiprocessing.Lock()

#     # Use a manager to create a shared result queue and lock
#     with multiprocessing.Manager() as manager:
#         shared_result_queue = manager.Queue()
#         shared_lock = manager.Lock()

#         # Create a pool of worker processes
#         with multiprocessing.Pool() as pool:
#             # Start the writer process
#             writer_process = multiprocessing.Process(
#                 target=write_results_to_file,
#                 args=(shared_result_queue, output_file, shared_lock),
#             )
#             writer_process.start()

#             # Submit the file processing function for each file to the pool
#             pool.starmap(
#                 process_file, [(file, shared_result_queue) for file in files]
#             )

#             # Signal the writer process that no more results will be added to the queue
#             shared_result_queue.put(None)
#             writer_process.join()


def main():
    input_folder = "/home/sj/LS2_Data/703"
    # output_file = "/home/sj/LS2_Data/703/MP_RESULTS/out.txt"

    os.chdir(input_folder)

    files = glob.glob("*.dat")

    final_result_array = np.zeros(256)

    with multiprocessing.Pool() as pool:
        final_result_array = sum(pool.map(process_file, files, chunksize=50))

    return final_result_array


if __name__ == "__main__":
    start_time = time.time()
    result = main()
    end_time = time.time()
    elapsed_time = end_time - start_time

    print(f"Execution time: {elapsed_time} seconds")
