"""Module for computing timestamp differences.

Compares all timestamps from the same cycle for the given pair of pixels
against a given value (delta_window). Differences in that window are saved
and returned as a list.

This file can also be imported as a module and contains the following
functions:

    * calculate_differences - calculate timestamp differences
    for the given pair of pixels. Works only with firmware version
    '2212'. Uses a faster algorithm than the standard version.

    * calculate_differences_1v1 - calculate timestamp
    differences for the given pair of pixels. Works only with firmware
    version '2212'. Uses a faster algorithm than the standard version.
    Calculated only for the diagonal pixels, i.e., 1-1, 2-2, etc., and
    not for all pairs.

"""

from __future__ import annotations

from typing import List

import numpy as np
from numpy import ndarray

from daplis.functions import utils

# def calculate_differences(
#     data: ndarray,
#     pixels: List[int] | List[List[int]],
#     pix_coor: ndarray,
#     delta_window: float = 50e3,
#     cycle_length: float = 4e9,
# ):
#     """Calculate timestamp differences for firmware version 2212.

#     Calculate timestamp differences for the given pixels and LinoSPAD2
#     firmware version 2212.

#     Parameters
#     ----------
#     data : ndarray
#         Matrix of timestamps, where rows correspond to the TDCs.
#     pixels : List[int] | List[List[int]]
#         List of pixel numbers for which the timestamp differences should
#         be calculated or list of two lists with pixel numbers for peak
#         vs. peak calculations.
#     pix_coor : ndarray
#         Array for transforming the pixel address in terms of TDC (0 to 3)
#         to pixel number in terms of half of the sensor (0 to 255).
#     delta_window : float, optional
#         Width of the time window for counting timestamp differences.
#         The default is 50e3 (50 ns).
#     cycle_length : float, optional
#         Length of each acquisition cycle. The default is 4e9 (4 ms).

#     Returns
#     -------
#     deltas_all : dict
#         Dictionary containing timestamp differences for each pair of pixels.

#     """

#     # Dictionary for the timestamp differences, where keys are the
#     # pixel numbers of the requested pairs
#     deltas_all = {}

#     pixels_left, pixels_right = utils.pixel_list_transform(pixels)

#     # Find ends of cycles
#     cycle_ends = np.argwhere(data[0].T[0] == -2)
#     cycle_ends = np.insert(cycle_ends, 0, 0)

#     for q in pixels_left:
#         # First pixel in the pair
#         tdc1, pix_c1 = np.argwhere(pix_coor == q)[0]
#         pix1 = np.where(data[tdc1].T[0] == pix_c1)[0]
#         for w in pixels_right:
#             if w <= q:
#                 continue
#             deltas_all[f"{q},{w}"] = []

#             timestamps_1 = []
#             timestamps_2 = []

#             # Second pixel in the pair
#             tdc2, pix_c2 = np.argwhere(pix_coor == w)[0]
#             pix2 = np.where(data[tdc2].T[0] == pix_c2)[0]

#             # Go over cycles, shifting the timestamps from each next
#             # cycle by lengths of cycles before (e.g., for the 4th cycle
#             # add 12 ms)
#             for i, _ in enumerate(cycle_ends[:-1]):
#                 slice_from = cycle_ends[i]
#                 slice_to = cycle_ends[i + 1]
#                 pix1_slice = pix1[(pix1 >= slice_from) & (pix1 < slice_to)]
#                 if not np.any(pix1_slice):
#                     continue
#                 pix2_slice = pix2[(pix2 >= slice_from) & (pix2 < slice_to)]
#                 if not np.any(pix2_slice):
#                     continue

#                 # Shift timestamps by cycle length
#                 tmsp1 = data[tdc1].T[1][pix1_slice]
#                 tmsp1 = tmsp1[tmsp1 > 0]
#                 tmsp1 = tmsp1 + cycle_length * i

#                 tmsp2 = data[tdc2].T[1][pix2_slice]
#                 tmsp2 = tmsp2[tmsp2 > 0]
#                 tmsp2 = tmsp2 + cycle_length * i

#                 timestamps_1.extend(tmsp1)
#                 timestamps_2.extend(tmsp2)

#             timestamps_1 = np.array(timestamps_1)
#             timestamps_2 = np.array(timestamps_2)

#             # Indicators for each pixel: 0 for timestamps from one pixel
#             # 1 - from the other
#             pix1_ind = np.zeros(len(timestamps_1), dtype=np.int32)
#             pix2_ind = np.ones(len(timestamps_2), dtype=np.int32)

#             pix1_data = np.vstack((pix1_ind, timestamps_1))
#             pix2_data = np.vstack((pix2_ind, timestamps_2))

#             # Dataframe for each pixel with pixel indicator and
#             # timestamps
#             df1 = pd.DataFrame(
#                 pix1_data.T, columns=["Pixel_index", "Timestamp"]
#             )
#             df2 = pd.DataFrame(
#                 pix2_data.T, columns=["Pixel_index", "Timestamp"]
#             )

#             # Combine the two dataframes
#             df_combined = pd.concat((df1, df2), ignore_index=True)

#             # Sort the timestamps
#             df_combined.sort_values("Timestamp", inplace=True)

#             # Subtract pixel indicators of neighbors; values of 0
#             # correspond to timestamp differences for the same pixel
#             # '-1' and '1' - to differences from different pixels
#             df_combined["Pixel_index_diff"] = df_combined["Pixel_index"].diff()

#             # Calculate timestamp difference between neighbors
#             df_combined["Timestamp_diff"] = df_combined["Timestamp"].diff()

#             # Get the correct timestamp difference sign
#             df_combined["Timestamp_diff"] = (
#                 df_combined["Timestamp_diff"] * df_combined["Pixel_index_diff"]
#             )

#             # Collect timestamp differences where timestamps are from
#             # different pixels
#             filtered_df = df_combined[
#                 abs(df_combined["Pixel_index_diff"]) == 1
#             ]

#             # Save only timestamps differences in the requested window
#             delta_ts = filtered_df[
#                 abs(filtered_df["Timestamp_diff"]) < delta_window
#             ]["Timestamp_diff"].values

#             deltas_all[f"{q},{w}"].extend(delta_ts)

#     return deltas_all


def calculate_differences(
    data: ndarray,
    pixels: List[int] | List[List[int]],
    delta_window: float = 50e3,
    cycle_length: float = 4e9,
):
    """Calculate timestamp differences for firmware version 2212.

    Calculate timestamp differences for the given pixels and LinoSPAD2
    firmware version 2212. Uses moving window algorithm for coincidence
    calculation.

    Parameters
    ----------
    data : ndarray
        Matrix of timestamps, where rows correspond to the TDCs.
    pixels : List[int] | List[List[int]]
        List of pixel numbers for which the timestamp differences should
        be calculated and saved or list of two lists with pixel numbers
        for peak vs. peak calculations.
    delta_window : float, optional
        Width of the time window for counting timestamp differences.
        The default is 50e3 (50 ns).
    cycle_length : float, optional
        Length of each acquisition cycle. The default is 4e9 (4 ms).

    Returns
    -------
    deltas_all : dict
        Dictionary containing timestamp differences for each pair of pixels.

    """

    # Dictionary for the timestamp differences, where keys are the
    # pixel numbers of the requested pairs
    deltas_all = {}

    pixels_left, pixels_right = utils.pixel_list_transform(pixels)

    seen_pairs = set()

    for q in pixels_left:
        for w in pixels_right:
            if q == w:
                continue
            # Canonical order of the pair, so that the column "a,b"
            # always means t_b - t_a with a < b, no matter in which
            # order the two lists of pixels were given
            pix_first, pix_second = (q, w) if q < w else (w, q)
            if (pix_first, pix_second) in seen_pairs:
                continue
            seen_pairs.add((pix_first, pix_second))

            deltas_all[f"{pix_first},{pix_second}"] = []

            # Timestamp lag calculation in a shifting window
            window_start_point = 0
            n2 = len(data[f"{pix_second}"])
            for t_first_pixel in data[f"{pix_first}"]:
                window_limit_left = t_first_pixel - delta_window
                window_limit_right = t_first_pixel + delta_window

                # Advance window_start_point
                while (
                    window_start_point < n2
                    and data[f"{pix_second}"][window_start_point]
                    < window_limit_left
                ):
                    window_start_point += 1

                pointer_in_window = window_start_point
                while (
                    pointer_in_window < n2
                    and data[f"{pix_second}"][pointer_in_window]
                    <= window_limit_right
                ):
                    dt = (
                        data[f"{pix_second}"][pointer_in_window]
                        - t_first_pixel
                    )
                    deltas_all[f"{pix_first},{pix_second}"].append(dt)
                    pointer_in_window += 1

    return deltas_all


def calculate_differences_1v1(
    data: ndarray,
    pixels: List[int] | List[List[int]],
    delta_window: float = 50e3,
    cycle_length: float = 4e9,
):
    """Calculate timestamp differences for explicit 1-to-1 pixel pairs.

    Like calculate_differences but pairs pixels_left[i] with
    pixels_right[i] only — no cross product. Requires both lists to have
    the same length.

    Parameters
    ----------
    data : ndarray
        Dict of timestamp arrays keyed by pixel number string, as returned
        by the current unpacker pipeline.
    pixels : List[int] | List[List[int]]
        Two lists of equal length: [[q0, q1, ...], [w0, w1, ...]].
        Pair i is (pixels_left[i], pixels_right[i]).
    delta_window : float, optional
        Width of the coincidence window in ps. The default is 50e3 (50 ns).
    cycle_length : float, optional
        Length of each acquisition cycle in ps. The default is 4e9 (4 ms).

    Returns
    -------
    deltas_all : dict
        Timestamp differences keyed by ``"q,w"`` for each pair.
    """

    deltas_all = {}

    # A flat list of pixel numbers means "all combinations" and cannot
    # be split into explicit 1-to-1 pairs
    if all(
        isinstance(pixel, (int, np.integer)) and not isinstance(pixel, bool)
        for pixel in pixels
    ):
        raise TypeError(
            "1v1 requires two lists of equal length, "
            "e.g. [[q0, q1], [w0, w1]]"
        )

    pixels_left, pixels_right = utils.pixel_list_transform(pixels)

    if len(pixels_left) != len(pixels_right):
        raise ValueError(
            "pixels_left and pixels_right must have the same length for 1v1."
        )

    for q, w in zip(pixels_left, pixels_right):
        deltas_all[f"{q},{w}"] = []

        window_start_point = 0
        n2 = len(data[f"{w}"])
        for t_first_pixel in data[f"{q}"]:
            window_limit_left  = t_first_pixel - delta_window
            window_limit_right = t_first_pixel + delta_window

            while (
                window_start_point < n2
                and data[f"{w}"][window_start_point] < window_limit_left
            ):
                window_start_point += 1

            pointer_in_window = window_start_point
            while (
                pointer_in_window < n2
                and data[f"{w}"][pointer_in_window] <= window_limit_right
            ):
                dt = data[f"{w}"][pointer_in_window] - t_first_pixel
                deltas_all[f"{q},{w}"].append(dt)
                pointer_in_window += 1

    return deltas_all
