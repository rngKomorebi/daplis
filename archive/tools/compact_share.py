import glob
import os
import sys
import time
from math import ceil
from zipfile import ZipFile

import numpy as np
import pandas as pd
from tqdm import tqdm

from daplis.functions import unpack as f_up


def compact_share(
    path: str,
    pixels: list,
    rewrite: bool,
    board_number: str,
    fw_ver: str,
    timestamps: int,
    delta_window: float = 50e3,
):
    """Collect delta ts and sensor population to a zip file.

    Unpacks data in the given folder, collects all timestamp differences
    and sensor population, saving the first one to a '.csv' file and
    the second to a '.txt' one, zipping both for compact output ready
    to share.

    Parameters
    ----------
    path : str
        Path to data files.
    pixels : list
        List of pixel numbers for which the timestamp differences should
        be calculated and saved or list of two lists with pixel numbers
        for peak vs. peak calculations.
    rewrite : bool
        Switch for rewriting the '.csv' file if it already exists.
    board_number : str
        The LinoSPAD2 daughterboard number.
    fw_ver: str
        LinoSPAD2 firmware version. Versions "2212s" (skip) and "2212b"
        (block) are recognized.
    timestamps : int, optional
        Number of timestamps per acquisition cycle per pixel. The default
        is 512.
    delta_window : float, optional
        Size of a window to which timestamp differences are compared.
        Differences in that window are saved. The default is 50e3 (50 ns).

    Raises
    ------
    TypeError
        Only boolean values of 'rewrite' and string values of 'fw_ver'
        are accepted. First error is raised so that the plot does not
        accidentally gets rewritten in the case no clear input was
        given.

    Returns
    -------
    None.
    """
    # parameter type check
    if isinstance(pixels, list) is False:
        raise TypeError(
            "'pixels' should be a list of integers or a list of two lists"
        )
    if isinstance(fw_ver, str) is False:
        raise TypeError("'fw_ver' should be string, '2212b' or '2208'")
    if isinstance(rewrite, bool) is False:
        raise TypeError("'rewrite' should be boolean")
    if isinstance(board_number, str) is False:
        raise TypeError(
            "'board_number' should be string, either 'NL11' or 'A5'"
        )

    os.chdir(path)

    files_all = glob.glob("*.dat*")

    out_file_name = files_all[0][:-4] + "-" + files_all[-1][:-4]

    # check if csv file exists and if it should be rewrited
    try:
        os.chdir("delta_ts_data")
        if os.path.isfile("{name}.csv".format(name=out_file_name)):
            if rewrite is True:
                print(
                    "\n! ! ! csv file with timestamps differences already "
                    "exists and will be rewritten ! ! !\n"
                )
                for i in range(5):
                    print(
                        "\n! ! ! Deleting the file in {} ! ! !\n".format(5 - i)
                    )
                    time.sleep(1)
                os.remove("{}.csv".format(out_file_name))
            else:
                print(
                    "\n csv file already exists, 'rewrite' set to"
                    "'False', passing"
                )
                pass
        os.chdir("..")
    except FileNotFoundError:
        pass

    # Collect the data for the required pixels
    if rewrite is True:
        print(
            "\n> > > Collecting data for delta t plot for the requested "
            "pixels and saving it to .csv in a cycle < < <\n"
        )
        if fw_ver == "2212s":
            # for transforming pixel number into TDC number + pixel
            # coordinates in that TDC
            pix_coor = np.arange(256).reshape(4, 64).T
        elif fw_ver == "2212b":
            pix_coor = np.arange(256).reshape(64, 4)
        else:
            print("\nFirmware version is not recognized.")
            sys.exit()

        # Mask the hot/warm pixels
        path_to_back = os.getcwd()
        path_to_mask = (
            os.path.realpath(__file__)
            + "/../../.."
            + "/src/LinoSPAD2/params/masks"
        )
        os.chdir(path_to_mask)
        file_mask = glob.glob("*{}*".format(board_number))[0]
        mask = np.genfromtxt(file_mask).astype(int)
        os.chdir(path_to_back)

        # Check if 'pixels' is one or two peaks, swap their positions if
        # needed
        if isinstance(pixels[0], list) is True:
            pixels_left = sorted(pixels[0])
            pixels_right = sorted(pixels[1])
            # Check if pixels from first list are to the left of the right
            # (peaks are not mixed up)
            if pixels_left[-1] > pixels_right[0]:
                plc_hld = pixels_left
                pixels_left = pixels_right
                pixels_right = plc_hld
                del plc_hld
        elif isinstance(pixels[0], int) is True:
            pixels_left = pixels.sort()
            pixels_right = pixels

        # Prepare array for sensor population
        valid_per_pixel = np.zeros(256)

        for i in tqdm(range(ceil(len(files_all))), desc="Collecting data"):
            file = files_all[i]

            # Prepare a dictionary for output
            deltas_all = {}

            # Unpack data for the requested pixels into dictionary
            data_all = f_up.unpack_bin(file, board_number, timestamps)

            # Calculate and collect timestamp differences
            # for q in pixels:
            for q in pixels_left:
                # for w in pixels:
                for w in pixels_right:
                    if w <= q:
                        continue
                    if q in mask or w in mask:
                        continue
                    deltas_all["{},{}".format(q, w)] = []
                    # find end of cycles
                    cycler = np.argwhere(data_all[0].T[0] == -2)
                    cycler = np.insert(cycler, 0, 0)
                    # first pixel in the pair
                    tdc1, pix_c1 = np.argwhere(pix_coor == q)[0]
                    pix1 = np.where(data_all[tdc1].T[0] == pix_c1)[0]
                    # second pixel in the pair
                    tdc2, pix_c2 = np.argwhere(pix_coor == w)[0]
                    pix2 = np.where(data_all[tdc2].T[0] == pix_c2)[0]
                    # get timestamp for both pixels in the given cycle
                    for cyc in range(len(cycler) - 1):
                        pix1_ = pix1[
                            np.logical_and(
                                pix1 > cycler[cyc], pix1 < cycler[cyc + 1]
                            )
                        ]
                        if not np.any(pix1_):
                            continue
                        pix2_ = pix2[
                            np.logical_and(
                                pix2 > cycler[cyc], pix2 < cycler[cyc + 1]
                            )
                        ]
                        if not np.any(pix2_):
                            continue
                        # calculate delta t
                        tmsp1 = data_all[tdc1].T[1][
                            pix1_[np.where(data_all[tdc1].T[1][pix1_] > 0)[0]]
                        ]
                        tmsp2 = data_all[tdc2].T[1][
                            pix2_[np.where(data_all[tdc2].T[1][pix2_] > 0)[0]]
                        ]
                        for t1 in tmsp1:
                            deltas = tmsp2 - t1
                            ind = np.where(np.abs(deltas) < delta_window)[0]
                            deltas_all["{},{}".format(q, w)].extend(
                                deltas[ind]
                            )
            # Collect sensor population
            for k in range(256):
                tdc, pix = np.argwhere(pix_coor == k)[0]
                ind = np.where(data_all[tdc].T[0] == pix)[0]
                ind1 = np.where(data_all[tdc].T[1][ind] > 0)[0]
                valid_per_pixel[k] += len(data_all[tdc].T[1][ind[ind1]])

            # Save data as a .csv file in a cycle so data is not lost
            # in the case of failure close to the end
            data_for_plot_df = pd.DataFrame.from_dict(
                deltas_all, orient="index"
            )
            del deltas_all
            data_for_plot_df = data_for_plot_df.T
            try:
                os.chdir("compact_share")
            except FileNotFoundError:
                os.mkdir("compact_share")
                os.chdir("compact_share")
            csv_file = glob.glob("*deltas_{}.csv*".format(out_file_name))
            if csv_file != []:
                data_for_plot_df.to_csv(
                    "deltas_{}.csv".format(out_file_name),
                    mode="a",
                    index=False,
                    header=False,
                )
            else:
                data_for_plot_df.to_csv(
                    "deltas_{}.csv".format(out_file_name), index=False
                )
            os.chdir("..")

        os.chdir("compact_share")
        np.savetxt("sen_pop_{}.txt".format(out_file_name), valid_per_pixel)
        # Create a ZipFile Object
        with ZipFile("{}.zip".format(out_file_name), "w") as zip_object:
            # Adding files that need to be zipped
            zip_object.write("deltas_{}.csv".format(out_file_name))
            zip_object.write("sen_pop_{}.txt".format(out_file_name))

            print(
                "\n> > > Timestamp differences are saved as {file}.csv and "
                "sensor population as sen_pop.txt in "
                "{path} < < <".format(
                    file=out_file_name,
                    path=path + "\delta_ts_data",
                )
            )
                )
            )
                )
            )
                )
            )
