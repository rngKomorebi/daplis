"""This module contains functions for LS2 data analysis that utilize
the multiprocessing Python library for speeding up the analysis
by using all available CPU cores instead of a single one.

This module can be imported with the class MpWizard and its internal
functions for data analysis using multiple CPU cores.

"""

import glob
import multiprocessing
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from pyarrow import feather as ft

from daplis.functions import calc_diff as cd
from daplis.functions import utils
from daplis.functions.calibrate import load_calibration_data

# def _compact_share_collect_data(
#     file: str,
#     result_queue_feather: multiprocessing.Queue,
#     result_queue_txt: multiprocessing.Queue,
#     data_params: DataParamsConfig,
# ):
#     """Collect delta timestamp differences and sensor population from
#     unpacked data, saving results to '.feather' and '.txt' files.

#     This function processes data from a single .dat file, calculates
#     timestamp differences and collects sensor population numbers. The
#     resulting arrays are put each to a separate multiprocessing queue
#     for saving via an appropriate function.

#     Parameters
#     ----------
#     file : str
#         Path to the data file to be processed.
#     result_queue_feather : multiprocessing.Queue
#         Queue for storing the timestamp differences in '.feather' format.
#     result_queue_txt : multiprocessing.Queue
#         Queue for storing the sensor population in '.txt' format.
#     data_params : DataParamsConfig
#         Configuration object containing parameters for data processing.

#     Returns
#     -------
#     None.
#     """

#     os.chdir(data_params.path)

#     # Collect the data for the required pixels

#     # for transforming pixel number into TDC number + pixel
#     # coordinates in that TDC
#     if data_params.firmware_version == "2212s":
#         pix_coor = np.arange(256).reshape(4, 64).T
#     elif data_params.firmware_version == "2212b":
#         pix_coor = np.arange(256).reshape(64, 4)
#     else:
#         sys.exit()

#     # Prepare array for sensor population
#     valid_per_pixel = np.zeros(256, dtype=int)

#     # Prepare a dictionary for timestamp differences
#     deltas_all = {}

#     # Unpack data
#     if not data_params.absolute_timestamps:
#         data_all = f_up.unpack_binary_data(
#             file,
#             data_params.daughterboard_number,
#             data_params.motherboard_number,
#             data_params.firmware_version,
#             data_params.timestamps,
#             data_params.include_offset,
#             data_params.apply_calibration,
#         )
#     else:
#         data_all, _ = f_up.unpack_binary_data_with_absolute_timestamps(
#             file,
#             data_params.daughterboard_number,
#             data_params.motherboard_number,
#             data_params.firmware_version,
#             data_params.timestamps,
#             data_params.include_offset,
#             data_params.apply_calibration,
#         )

#     deltas_all = cd.calculate_differences_2212(
#         data_all, data_params.pixels, pix_coor, data_params.delta_window
#     )

#     # Collect sensor population
#     for k in range(256):
#         tdc, pix = np.argwhere(pix_coor == k)[0]
#         valid_per_pixel[k] += np.count_nonzero(data_all[tdc][:, 0] == pix)

#     # Save data as a .feather file in a cycle so data is not lost
#     # in the case of failure close to the end
#     data_for_plot_df = pd.DataFrame.from_dict(deltas_all, orient="index")
#     del deltas_all

#     result_queue_feather.put(data_for_plot_df.T)
#     result_queue_txt.put(valid_per_pixel)


# def compact_share_mp(
#     path: str,
#     pixels: list,
#     rewrite: bool,
#     daughterboard_number: str,
#     motherboard_number: str,
#     firmware_version: str,
#     timestamps: int,
#     delta_window: float = 50e3,
#     include_offset: bool = False,
#     apply_calibration: bool = True,
#     absolute_timestamps: bool = False,
#     chunksize=None,
# ):
#     """Collect timestamp differences and sensor population using
#     multiprocessing, saving results to '.feather' and '.txt' files.

#     This function parallelizes the processing of data in the specified
#     path using multiprocessing. It calculates timestamp differences and
#     sensor population for the specified pixels based on the provided
#     parameters, saving the timestamp differences to a '.feather' file and
#     the sensor population to a '.txt' file. Both files are then zipped for
#     compact output ready to share.

#     Parameters
#     ----------
#     path : str
#         Path to the directory containing data files.
#     pixels : list
#         List of pixel numbers for which the timestamp differences should
#         be calculated and saved, or list of two lists with pixel numbers
#         for peak vs. peak calculations.
#     rewrite : bool
#         Switch for rewriting the '.feather' file if it already exists.
#     daughterboard_number : str
#         LinoSPAD2 daughterboard number.
#     motherboard_number : str
#         LinoSPAD2 motherboard (FPGA) number, including the "#".
#     firmware_version: str
#         LinoSPAD2 firmware version. Accepted values are "2212s" (skip)
#         and "2212b" (block).
#     timestamps : int
#         Number of timestamps per acquisition cycle per pixel.
#     delta_window : float, optional
#         Size of a window (in nanoseconds) to which timestamp differences
#         are compared (default is 50e3 nanoseconds).
#     include_offset : bool, optional
#         Switch for applying offset calibration (default is True).
#     apply_calibration : bool, optional
#         Switch for applying TDC and offset calibration. If set to 'True'
#         while include_offset is set to 'False', only the TDC calibration is
#         applied (default is True).
#     absolute_timestamps : bool, optional
#         Indicator for data with absolute timestamps (default is False).
#     chunksize : int, optional
#         The number of data points processed in each batch by each worker.
#         If None, the default chunk size is determined automatically.

#     Raises
#     ------
#     TypeError
#         Only boolean values of 'rewrite' and string values of
#         'firmware_version' are accepted. The first error is raised so
#         that the files are not accidentally overwritten in the case of
#         unclear input.

#     Returns
#     -------
#     None.
#     """

#     # parameter type check
#     if isinstance(pixels, list) is False:
#         raise TypeError(
#             "'pixels' should be a list of integers or a list of two lists"
#         )
#     if isinstance(firmware_version, str) is False:
#         raise TypeError(
#             "'firmware_version' should be string, '2212s', '2212b' or '2208'"
#         )
#     if isinstance(rewrite, bool) is False:
#         raise TypeError("'rewrite' should be boolean")
#     if isinstance(daughterboard_number, str) is False:
#         raise TypeError("'daughterboard_number' should be string")
#     if isinstance(motherboard_number, str) is False:
#         raise TypeError("'motherboard_number' should be string")

#     # Generate a dataclass object
#     data_params = DataParamsConfig(
#         pixels=pixels,
#         path=path,
#         daughterboard_number=daughterboard_number,
#         motherboard_number=motherboard_number,
#         firmware_version=firmware_version,
#         timestamps=timestamps,
#         delta_window=delta_window,
#         include_offset=include_offset,
#         apply_calibration=apply_calibration,
#         absolute_timestamps=absolute_timestamps,
#     )

#     os.chdir(path)

#     # Find all LinoSPAD2 data files
#     # files = sorted(glob.glob("*.dat"))
#     files = glob.glob("*.dat*")
#     files.sort(key=os.path.getmtime)
#     # Get the resulting Feather file name based on the data files
#     # found
#     feather_file_name = files[0][:-4] + "-" + files[-1][:-4] + ".feather"
#     txt_file_name = files[0][:-4] + "-" + files[-1][:-4] + ".txt"
#     # Construct absolute path to the Feather file
#     feather_file = os.path.join(path, "compact_share", feather_file_name)
#     txt_file = os.path.join(path, "compact_share", txt_file_name)
#     # Handle the rewrite parameter based on the file existence to avoid
#     # accidental file overwritting
#     utils.file_rewrite_handling(feather_file, rewrite)
#     utils.file_rewrite_handling(txt_file, rewrite)

#     with multiprocessing.Manager() as manager:
#         shared_result_feather = manager.Queue()
#         shared_result_txt = manager.Queue()
#         shared_lock_feather = manager.Lock()
#         shared_lock_txt = manager.Lock()

#         with multiprocessing.Pool() as pool:
#             # Start the writer process
#             writer_process_feather = multiprocessing.Process(
#                 target=_write_results_to_feather,
#                 args=(
#                     shared_result_feather,
#                     feather_file,
#                     shared_lock_feather,
#                 ),
#             )

#             writer_process_txt = multiprocessing.Process(
#                 target=_write_results_to_txt,
#                 args=(shared_result_txt, txt_file, shared_lock_txt),
#             )

#             writer_process_feather.start()
#             writer_process_txt.start()

#             # Create a partial function with fixed arguments for
#             # process_file
#             partial_process_file = functools.partial(
#                 _compact_share_collect_data,
#                 result_queue_feather=shared_result_feather,
#                 result_queue_txt=shared_result_txt,
#                 data_params=data_params,
#             )

#             # Start the multicore analysis of the files
#             if chunksize is None:
#                 pool.map(partial_process_file, files)

#             else:
#                 pool.map(partial_process_file, files, chunksize=chunksize)

#             # Signal the writer process that no more results will be
#             # added to the queue
#             shared_result_feather.put(None)
#             shared_result_txt.put(None)
#             writer_process_feather.join()
#             writer_process_txt.join()

#         # Create a ZipFile Object
#         os.chdir(os.path.join(path, "compact_share"))
#         out_file_name, _ = os.path.splitext(feather_file_name)

#         with ZipFile(f"{out_file_name}.zip", "w") as zip_object:
#             # Adding files that need to be zipped
#             zip_object.write(f"{out_file_name}.feather")
#             zip_object.write(f"{out_file_name}.txt")

#             print(
#                 "\n> > > Timestamp differences are saved as {feather_file} and "
#                 "sensor population as {txt_file} in "
#                 "{path} < < <".format(
#                     feather_file=feather_file,
#                     txt_file=txt_file,
#                     path=path + "\delta_ts_data",
#                 )
#             )


class MpWizard:

    # Initialize by passing the input parameters which later will be
    # passed into all internal functions
    def __init__(
        self,
        path: str = "",
        pixels: list = [],
        daughterboard_number: str = "",
        motherboard_number: str = "",
        firmware_version: str = "",
        timestamps: int = 512,
        delta_window: float = 50e3,
        include_offset: bool = False,
        apply_calibration: bool = True,
        apply_mask: bool = True,
        absolute_timestamps: bool = False,
        number_of_cores: int = 1,
    ):

        self.path = path
        self.pixels = pixels
        self.daughterboard_number = daughterboard_number
        self.motherboard_number = motherboard_number
        self.firmware_version = firmware_version
        self.timestamps = timestamps
        self.delta_window = delta_window
        self.include_offset = include_offset
        self.apply_calibration = apply_calibration
        self.apply_mask = apply_mask
        self.absolute_timestamps = absolute_timestamps
        self.number_of_cores = number_of_cores

        os.chdir(self.path)

        # Load calibration if requested
        if self.apply_calibration:

            work_dir = Path(__file__).resolve().parent.parent

            path_calibration_data = os.path.join(
                work_dir, r"params\calibration_data"
            )

            calibration_data = load_calibration_data(
                path_calibration_data,
                daughterboard_number,
                motherboard_number,
                firmware_version,
                include_offset,
            )

            if self.include_offset:
                self.calibration_matrix, self.offset_array = calibration_data
            else:
                self.calibration_matrix = calibration_data

        # Apply mask if requested
        if self.apply_mask:
            mask = utils.apply_mask(
                self.daughterboard_number,
                self.motherboard_number,
            )
            if isinstance(self.pixels[0], int) and isinstance(
                self.pixels[1], int
            ):
                pixels = [pix for pix in self.pixels if pix not in mask]
            else:
                pixels = [pix for pix in self.pixels[0] if pix not in mask]
                pixels.extend(pix for pix in self.pixels[1] if pix not in mask)

        # Check the firmware version and set the pixel coordinates accordingly
        if self.firmware_version == "2212s":
            self.pix_coor = np.arange(256).reshape(4, 64).T
        elif firmware_version == "2212b":
            self.pix_coor = np.arange(256).reshape(64, 4)
        else:
            print("\nFirmware version is not recognized.")
            sys.exit()

    def _unpack_binary_data(
        self,
        file: str,
    ) -> np.ndarray:
        """Unpack binary data from LinoSPAD2.

        Return a 3D matrix of pixel numbers and timestamps.

        Parameters
        ----------
        file : str
            A '.dat' data file from LinoSPAD2 with the binary-encoded
            data.

        Returns
        -------
        np.ndarray
            A 3D matrix of the pixel numbers where timestamp was
            recorded and timestamps themselves.
        """
        # Unpack binary data
        raw_data = np.memmap(file, dtype=np.uint32)
        # Timestamps are stored in the lower 28 bits
        data_timestamps = (raw_data & 0xFFFFFFF).astype(np.int64)
        # Pixel address in the given TDC is 2 bits above timestamp
        data_pixels = ((raw_data >> 28) & 0x3).astype(np.int8)
        # Check the top bit, assign '-1' to invalid timestamps
        data_timestamps[raw_data < 0x80000000] = -1

        # Number of acquisition cycles in each data file
        cycles = len(data_timestamps) // (self.timestamps * 65)
        # Transform into a matrix of size 65 by cycles*timestamps
        data_pixels = (
            data_pixels.reshape(cycles, 65, self.timestamps)
            .transpose((1, 0, 2))
            .reshape(65, -1)
        )

        data_timestamps = (
            data_timestamps.reshape(cycles, 65, self.timestamps)
            .transpose((1, 0, 2))
            .reshape(65, -1)
        )

        # Cut the 65th TDC that does not hold any actual data from pixels
        data_pixels = data_pixels[:-1]
        data_timestamps = data_timestamps[:-1]

        # Insert '-2' at the end of each cycle
        insert_indices = np.linspace(
            self.timestamps, cycles * self.timestamps, cycles
        ).astype(np.int64)

        data_pixels = np.insert(
            data_pixels,
            insert_indices,
            -2,
            1,
        )
        data_timestamps = np.insert(
            data_timestamps,
            insert_indices,
            -2,
            1,
        )

        # Combine both matrices into a single one, where each cell holds pixel
        # coordinates in the TDC and the timestamp
        data_all = np.stack((data_pixels, data_timestamps), axis=2).astype(
            np.int64
        )

        if self.apply_calibration is False:
            data_all[:, :, 1] = data_all[:, :, 1] * 2500 / 140
        else:
            # Path to the calibration data
            pix_coordinates = np.arange(256).reshape(64, 4)
            for i in range(256):
                # Transform pixel number to TDC number and pixel
                # coordinates in that TDC (from 0 to 3)
                tdc, pix = np.argwhere(pix_coordinates == i)[0]
                # Find data from that pixel
                ind = np.where(data_all[tdc].T[0] == pix)[0]
                # Cut non-valid timestamps ('-1's)
                ind = ind[data_all[tdc].T[1][ind] >= 0]
                if not np.any(ind):
                    continue
                data_cut = data_all[tdc].T[1][ind]
                # Apply calibration; offset is added due to how delta
                # ts are calculated
                if self.include_offset:
                    data_all[tdc].T[1][ind] = (
                        (data_cut - data_cut % 140) * 2500 / 140
                        + self.calibration_matrix[i, (data_cut % 140)]
                        + self.offset_array[i]
                    )
                else:
                    data_all[tdc].T[1][ind] = (
                        data_cut - data_cut % 140
                    ) * 2500 / 140 + self.calibration_matrix[
                        i, (data_cut % 140)
                    ]

        return data_all

    def _calculate_timestamps_differences(self, files):
        """Calculate photon coincidences and save to '.feather'.

        Parameters
        ----------
        files : List
            '.dat' data files to analyze.
        """

        # try-except railguard for a function that goes to separate
        # cores
        try:
            # Check if the 'delta_ts_data' folder exists
            output_dir = Path(self.path) / "delta_ts_data"
            output_dir.mkdir(exist_ok=True)

            for file in files:
                # Unpack data from binary files
                data_all = self._unpack_binary_data(file)

                # Calculate the differences and convert them to a pandas
                # dataframe
                deltas_all = cd.calculate_differences(
                    data_all, self.pixels, self.pix_coor
                )
                data_for_plot_df = pd.DataFrame.from_dict(
                    deltas_all, orient="index"
                ).T

                # Save the data to a '.feather' file
                file_name = os.path.basename(file)

                output_file = os.path.join(
                    self.path, str(file_name.replace(".dat", ".feather"))
                )
                data_for_plot_df.reset_index(drop=True, inplace=True)

                file_name = Path(file).stem
                output_file = output_dir / f"{file_name}.feather"
                ft.write_feather(
                    data_for_plot_df.reset_index(drop=True), output_file
                )
        except Exception as e:
            print(f"Error processing files {files}: {e}")

    def calculate_and_save_timestamp_differences_mp(self):

        # Find all LinoSPAD2 data files
        files = glob.glob("*.dat")
        num_of_files = len(files)

        if not files:
            raise ValueError("No .dat files found in the specified path.")

        # Split files into chunks of equal length
        # (files / number_of_cores)
        files = np.array_split(files, self.number_of_cores)

        print("Starting analysis of the files")
        start_time = time.time()

        processes = []
        # Create processes (number of cores) and assign to each process
        # its specified files chunk (files/number_of_cores)
        # each process will run the _calculate_timestamps_differences
        # function with its own parameters target: the function to be run
        for i in range(self.number_of_cores):
            p = multiprocessing.Process(
                target=self._calculate_timestamps_differences, args=(files[i],)
            )

            p.start()
            # Add the process to the list so we can wait for all of them to finish
            processes.append(p)

        # Wait for all the processes to finish, and only then continue to the next step
        for process in processes:
            process.join()

        end_time = time.time()

        print(
            f"Parallel processing of {num_of_files} "
            "files (with each writing to its file) finished "
            f"in: {round(end_time - start_time, 2)} s"
        )

        # Combine '.feather' files from separate cores
        path_to_feathers = os.path.join(self.path, "delta_ts_data")

        utils.combine_feather_files(path_to_feathers)

        path_output = os.path.join(self.path, "delta_ts_data")

        print(
            "The feather files with the timestamp differences were "
            f"combined into the 'combined.feather' file in {path_output}"
        )
