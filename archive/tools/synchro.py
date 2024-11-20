import glob
import os
import sys
import time
from math import ceil

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from tqdm import tqdm

from daplis.functions import unpack as f_up

# path1 = r"D:\LinoSPAD2\Data\board_NL11\Prague\Synchro\synchro_delay_test\#21"
# path2 = r"D:\LinoSPAD2\Data\board_NL11\Prague\Synchro\synchro_delay_test\#33"


# path1 = r"D:\LinoSPAD2\Data\board_NL11\Prague\Synchro\Ne beams\#21"
# path2 = r"D:\LinoSPAD2\Data\board_NL11\Prague\Synchro\Ne beams\#33"


# os.chdir(path1)
# files = glob.glob("*.dat*")

# print("In {}".format(path1[-3:]))

# for j, file in enumerate(files):
#     data = f_up.unpack_bin(
#         file,
#         db_num="NL11",
#         mb_num=path1[-3:],
#         fw_ver="2212b",
#         timestamps=300,
#         inc_offset=False,
#     )

#     valid_per_pixel = np.zeros(256)
#     pix_coor = np.arange(256).reshape(64, 4)
#     for i in range(256):
#         tdc, pix = np.argwhere(pix_coor == i)[0]
#         ind = np.where(data[tdc].T[0] == pix)[0]
#         ind1 = np.where(data[tdc].T[1][ind] > 0)[0]
#         valid_per_pixel[i] += len(data[tdc].T[1][ind[ind1]])

#     if np.sum(valid_per_pixel) > 0:
#         cycle_ends = np.where(data[0].T[1] == -2)[0]
#         cyc = np.argmin(
#             np.abs(cycle_ends - np.where(data[:].T[1] > 0)[0].min())
#         )
#     if cycle_ends[cyc] > np.where(data[:].T[1] > 0)[0].min():
#         cycle_start = cycle_ends[cyc - 1]
#     else:
#         cycle_start = cycle_ends[cyc]
#         # file_start = j
#     break

# print(j, cycle_start, cycle_ends, cyc)

# os.chdir(path2)
# files = glob.glob("*.dat*")

# print("In {}".format(path2[-3:]))

# for j, file in enumerate(files):
#     data = f_up.unpack_bin(
#         file,
#         db_num="NL11",
#         mb_num=path2[-3:],
#         fw_ver="2212b",
#         timestamps=300,
#         inc_offset=False,
#     )

#     # valid_per_pixel = np.zeros(256)
#     pix_coor = np.arange(256).reshape(64, 4)
#     for i in range(256):
#         tdc, pix = np.argwhere(pix_coor == i)[0]
#         ind = np.where(data[tdc].T[0] == pix)[0]
#         ind1 = np.where(data[tdc].T[1][ind] > 0)[0]
#         valid_per_pixel[i] += len(data[tdc].T[1][ind[ind1]])

#     if np.sum(valid_per_pixel) > 0:
#         cycle_ends = np.where(data[0].T[1] == -2)[0]
#         cyc = np.argmin(
#             np.abs(cycle_ends - np.where(data[:].T[1] > 0)[0].min())
#         )
#     if cycle_ends[cyc] > np.where(data[:].T[1] > 0)[0].min():
#         cycle_start = cycle_ends[cyc - 1]
#     else:
#         cycle_start = cycle_ends[cyc]
#     # file_start = j
#     break

# print(j, cycle_start, cycle_ends, cyc)

#####


def delta_save_full(
    path,
    pixels: list,
    rewrite: bool,
    db_num: str,
    mb_num1: str,
    mb_num2: str,
    fw_ver: str,
    timestamps: int = 512,
    delta_window: float = 50e3,
    app_mask: bool = True,
    inc_offset: bool = True,
    app_calib: bool = True,
):
    os.chdir(path)

    try:
        os.chdir("{}".format(mb_num1))
    except FileNotFoundError:
        raise FileNotFoundError("Path to data from {} is not found".format(mb_num1))
    files_all1 = glob.glob("*.dat*")
    out_file_name = files_all1[0][:-4]
    os.chdir("..")
    try:
        os.chdir("{}".format(mb_num2))
    except FileNotFoundError:
        raise FileNotFoundError("Path to data from {} is not found".format(mb_num2))
    files_all2 = glob.glob("*.dat*")
    out_file_name = out_file_name + "-" + files_all2[-1][:-4]
    os.chdir("..")

    if fw_ver == "2212s":
        # for transforming pixel number into TDC number + pixel
        # coordinates in that TDC
        pix_coor = np.arange(256).reshape(4, 64).T
    elif fw_ver == "2212b":
        pix_coor = np.arange(256).reshape(64, 4)
    else:
        print("\nFirmware version is not recognized.")
        sys.exit()

    # # Mask the hot/warm pixels
    # if app_mask is True:
    #     path_to_back = os.getcwd()
    #     path_to_mask = os.path.realpath(__file__) + "/../.." + "/params/masks"
    #     os.chdir(path_to_mask)
    #     file_mask1 = glob.glob("*{}_{}*".format(db_num, mb_num1))[0]
    #     mask1 = np.genfromtxt(file_mask1).astype(int)
    #     file_mask2 = glob.glob("*{}_{}*".format(db_num, mb_num2))[0]
    #     mask2 = np.genfromtxt(file_mask2).astype(int)
    #     os.chdir(path_to_back)

    for i in tqdm(range(ceil(len(files_all1))), desc="Collecting data"):
        deltas_all = {}
        # First board
        os.chdir("{}".format(mb_num1))
        file = files_all1[i]
        data_all1 = f_up.unpack_bin(
            file, db_num, mb_num1, fw_ver, timestamps, inc_offset, app_calib
        )
        cycle_ends1 = np.where(data_all1[0].T[1] == -2)[0]
        cyc1 = np.argmin(np.abs(cycle_ends1 - np.where(data_all1[:].T[1] > 0)[0].min()))
        if cycle_ends1[cyc1] > np.where(data_all1[:].T[1] > 0)[0].min():
            cycle_start1 = cycle_ends1[cyc1 - 1]
        else:
            cycle_start1 = cycle_ends1[cyc1]

        os.chdir("..")

        # Second board
        os.chdir("{}".format(mb_num2))
        file = files_all2[i]
        data_all2 = f_up.unpack_bin(
            file, db_num, mb_num2, fw_ver, timestamps, inc_offset, app_calib
        )
        cycle_ends2 = np.where(data_all2[0].T[1] == -2)[0]
        cyc2 = np.argmin(np.abs(cycle_ends2 - np.where(data_all2[:].T[1] > 0)[0].min()))
        if cycle_ends2[cyc2] > np.where(data_all2[:].T[1] > 0)[0].min():
            cycle_start2 = cycle_ends2[cyc2 - 1]
        else:
            cycle_start2 = cycle_ends2[cyc2]

        os.chdir("..")

        deltas_all["{},{}".format(pixels[0], pixels[1])] = []
        tdc1, pix_c1 = np.argwhere(pix_coor == pixels[0])[0]
        pix1 = np.where(data_all1[tdc1].T[0] == pix_c1)[0]
        tdc2, pix_c2 = np.argwhere(pix_coor == pixels[1])[0]
        pix2 = np.where(data_all2[tdc2].T[0] == pix_c2)[0]

        if cycle_start1 > cycle_start2:
            cyc = len(data_all1[0].T[1]) - cycle_start1 + cycle_start2
            cycle_ends1 = cycle_ends1[cycle_ends1 > cycle_start1]
            cycle_ends2 = np.intersect1d(
                cycle_ends2[cycle_ends2 > cycle_start2],
                cycle_ends2[cycle_ends2 < cyc],
            )
        else:
            cyc = len(data_all1[0].T[1]) - cycle_start2 + cycle_start1
            cycle_ends2 = cycle_ends2[cycle_ends2 > cycle_start2]
            cycle_ends1 = np.intersect1d(
                cycle_ends1[cycle_ends1 > cycle_start1],
                cycle_ends1[cycle_ends1 < cyc],
            )
        # get timestamp for both pixels in the given cycle
        for cyc in range(len(cycle_ends1) - 1):
            pix1_ = pix1[
                np.logical_and(pix1 > cycle_ends1[cyc], pix1 < cycle_ends1[cyc + 1])
            ]
            if not np.any(pix1_):
                continue
            pix2_ = pix2[
                np.logical_and(pix2 > cycle_ends2[cyc], pix2 < cycle_ends2[cyc + 1])
            ]

            if not np.any(pix2_):
                continue
            # calculate delta t
            tmsp1 = data_all1[tdc1].T[1][
                pix1_[np.where(data_all1[tdc1].T[1][pix1_] > 0)[0]]
            ]
            tmsp2 = data_all2[tdc2].T[1][
                pix2_[np.where(data_all2[tdc2].T[1][pix2_] > 0)[0]]
            ]
            for t1 in tmsp1:
                deltas = tmsp2 - t1
                ind = np.where(np.abs(deltas) < delta_window)[0]
                deltas_all["{},{}".format(pixels[0], pixels[1])].extend(deltas[ind])
        # Save data as a .csv file in a cycle so data is not lost
        # in the case of failure close to the end
        data_for_plot_df = pd.DataFrame.from_dict(deltas_all, orient="index")
        del deltas_all
        data_for_plot_df = data_for_plot_df.T
        try:
            os.chdir("delta_ts_data")
        except FileNotFoundError:
            os.mkdir("delta_ts_data")
            os.chdir("delta_ts_data")
        csv_file = glob.glob("*{}.csv*".format(out_file_name))
        if csv_file != []:
            data_for_plot_df.to_csv(
                "{}.csv".format(out_file_name),
                mode="a",
                index=False,
                header=False,
            )
        else:
            data_for_plot_df.to_csv("{}.csv".format(out_file_name), index=False)
        os.chdir("..")

    if os.path.isfile(path + "/delta_ts_data/{}.csv".format(out_file_name)) is True:
        print(
            "\n> > > Timestamp differences are saved as {file}.csv in "
            "{path} < < <".format(
                file=out_file_name,
                path=path + "\delta_ts_data",
            )
        )
    else:
        print("File wasn't generated. Check input parameters.")


path = r"D:\LinoSPAD2\Data\board_NL11\Prague\Synchro\Ne beams"
delta_save_full(
    path,
    pixels=[207, 58],
    # pixels=[158, 138],
    rewrite=True,
    db_num="NL11",
    mb_num1="#21",
    mb_num2="#33",
    fw_ver="2212b",
    timestamps=300,
    inc_offset=False,
    app_mask=False,
)
