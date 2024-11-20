import glob
import os
import sys
from math import ceil

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from pyarrow import feather as ft
from tqdm import tqdm

from daplis.functions import calc_diff as cd
from daplis.functions import unpack as f_up
from daplis.functions import utils


def calculate_and_save_timestamp_differences_full_sensor(
    path,
    pixels: list,
    rewrite: bool,
    daughterboard_number: str,
    motherboard_number1: str,
    motherboard_number2: str,
    firmware_version: str,
    timestamps: int = 512,
    delta_window: float = 50e3,
    app_mask: bool = True,
    include_offset: bool = True,
    apply_calibration: bool = True,
    absolute_timestamps: bool = False,
):
    os.chdir(path)

    # Check the data from the first FPGA board
    try:
        os.chdir(f"{motherboard_number1}")
    except FileNotFoundError as exc:
        raise FileNotFoundError(
            f"Data from {motherboard_number1} not found"
        ) from exc

    files_all1 = sorted(glob.glob("*.dat*"))
    out_file_name = files_all1[0][:-4]
    os.chdir("..")

    # Check the data from the second FPGA board
    try:
        os.chdir(f"{motherboard_number2}")
    except FileNotFoundError as exc:
        raise FileNotFoundError(
            f"Data from {motherboard_number2} not found"
        ) from exc

    files_all2 = sorted(glob.glob("*.dat*"))
    out_file_name = out_file_name + "-" + files_all2[-1][:-4]
    os.chdir("..")

    # Define matrix of pixel coordinates, where rows are numbers of TDCs
    # and columns are the pixels that connected to these TDCs
    if firmware_version == "2212s":
        pix_coor = np.arange(256).reshape(4, 64).T
    elif firmware_version == "2212b":
        pix_coor = np.arange(256).reshape(64, 4)
    else:
        print("\nFirmware version is not recognized.")
        sys.exit()

    # Check if '.feather' file with timestamps differences already
    # exists
    feather_file = f"{out_file_name}.feather"

    utils.file_rewrite_handling(feather_file, rewrite)

    abs_tmsp_list1 = []
    abs_tmsp_list2 = []
    for i in tqdm(range(ceil(len(files_all1))), desc="Collecting data"):
        deltas_all = {}

        # First board, unpack data
        os.chdir(os.path.join(path, motherboard_number1))

        file = files_all1[i]
        if not absolute_timestamps:
            data_all1 = f_up.unpack_binary_data(
                file,
                daughterboard_number,
                motherboard_number1,
                firmware_version,
                timestamps,
                include_offset,
                apply_calibration,
            )
        else:
            (
                data_all1,
                abs_tmsp1,
            ) = f_up.unpack_binary_data_with_absolute_timestamps(
                file,
                daughterboard_number,
                motherboard_number1,
                firmware_version,
                timestamps,
                include_offset,
                apply_calibration,
            )
        # abs_tmsp_list1.append(abs_tmsp1)
        # Collect indices of cycle ends (the '-2's)
        cycle_ends1 = np.where(data_all1[0].T[1] == -2)[0]
        cyc1 = np.argmin(
            np.abs(cycle_ends1 - np.where(data_all1[:].T[1] > 0)[0].min())
        )
        if cycle_ends1[cyc1] > np.where(data_all1[:].T[1] > 0)[0].min():
            cycle_start1 = cycle_ends1[cyc1 - 1]
            abs_tmsp_list1.append(abs_tmsp1[(cyc1 - 1):])
        else:
            cycle_start1 = cycle_ends1[cyc1]
            abs_tmsp_list1.append(abs_tmsp1[cyc1:])
        print(f"Cycle start 1: {cycle_start1}, {cyc1}")
            

        os.chdir("..")

        # Second board, unpack data
        os.chdir(f"{motherboard_number2}")

        # Second board, unpack data
        # os.chdir(f"{motherboard_number2}")
        file = files_all2[i]
        if not absolute_timestamps:
            data_all2 = f_up.unpack_binary_data(
                file,
                daughterboard_number,
                motherboard_number2,
                firmware_version,
                timestamps,
                include_offset,
                apply_calibration,
            )
        else:
            (
                data_all2,
                abs_tmsp2,
            ) = f_up.unpack_binary_data_with_absolute_timestamps(
                file,
                daughterboard_number,
                motherboard_number2,
                firmware_version,
                timestamps,
                include_offset,
                apply_calibration,
            )
        # abs_tmsp_list2.append(abs_tmsp2)
            
        # Collect indices of cycle ends (the '-2's)
        cycle_ends2 = np.where(data_all2[0].T[1] == -2)[0]
        cyc2 = np.argmin(
            np.abs(cycle_ends2 - np.where(data_all2[:].T[1] > 0)[0].min())
        )
        if cycle_ends2[cyc2] > np.where(data_all2[:].T[1] > 0)[0].min():
            cycle_start2 = cycle_ends2[cyc2 - 1]
            abs_tmsp_list2.append(abs_tmsp2[(cyc2 - 1):])
            
        else:
            cycle_start2 = cycle_ends2[cyc2]
            abs_tmsp_list2.append(abs_tmsp2[cyc2:])
            
        print(f"Cycle start 2: {cycle_start2}, {cyc2}")
        
        # if cycle_start1 > cycle_start2:
        #     cyc = len(data_all1[0].T[1]) - cycle_start1 + cycle_start2
        #     abs_tmsp_list1 = abs_tmsp_list1[cycle_ends1 >= cycle_start1]
        #     abs_tmsp_list2 = np.intersect1d(
        #         abs_tmsp_list2[cycle_ends2 >= cycle_start2],
        #         abs_tmsp_list2[cycle_ends2 <= cyc],
        #     )
        # else:
        #     cyc = len(data_all1[0].T[1]) - cycle_start2 + cycle_start1
        #     abs_tmsp_list2 = abs_tmsp_list2[cycle_ends2 >= cycle_start2]
        #     abs_tmsp_list1 = np.intersect1d(
        #         abs_tmsp_list1[cycle_ends1 >= cycle_start1],
        #         abs_tmsp_list1[cycle_ends1 <= cyc],
        #     )


    return abs_tmsp_list1, abs_tmsp_list2


path = r"/home/sj/Shared/05.01.24"


abs1, abs2 = calculate_and_save_timestamp_differences_full_sensor(
    path,
    pixels=[88, 399],
    rewrite=True,
    daughterboard_number="NL11",
    motherboard_number1="#33",
    motherboard_number2="#21",
    firmware_version="2212b",
    timestamps=300,
    include_offset=False,
    absolute_timestamps=True,
)


########

from matplotlib import pyplot as plt
import numpy as np
plt.plot(abs1[0], 'o')
# plt.plot(abs2)
plt.show()

%matplotlib qt
# absss = np.concatenate((abs1[0], abs1[1]))
abs_diff = np.array([abs1[0][i+1] - abs1[0][i] for i in range(len(abs1[0])-1)])

# absss2 = np.concatenate((abs2[0], abs2[1]))
abs_diff2 = np.array([abs2[0][i+1] - abs2[0][i] for i in range(len(abs2[0])-1)])

# plt.plot(absss, '.')
# plt.plot(absss2, '.')
# plt.show()

plt.plot(abs_diff)
plt.plot(abs_diff2[33:])
plt.show()