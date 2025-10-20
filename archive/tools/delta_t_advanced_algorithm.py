def _flatten(input_list: List):
    """Flatten the input list.

    Flatten the input list, which can be a list of numbers, lists,
    or a combination of the two above, and return a list of
    numbers only, unpacking the lists inside.

    Parameters
    ----------
    input_list : List
        Input list that can contain numbers, lists of numbers, or a
        combination of both.

    Returns
    -------
    list
        Flattened list of numbers only.
    """
    flattened = []
    for item in input_list:
        if isinstance(item, list):
            flattened.extend(item)
        else:
            flattened.append(item)
    return flattened


def calculate_differences_mix(
    data: ndarray,
    pixels: List[int] | List[List[int]],
    pix_coor: ndarray,
    delta_window: float = 50e3,
    cycle_length: float = 4e9,
):
    # TODO update docstring
    """Calculate timestamp differences for firmware version 2212.

    Calculate timestamp differences for the given pixels and LinoSPAD2
    firmware version 2212.

    Parameters
    ----------
    data : ndarray
        Matrix of timestamps, where rows correspond to the TDCs.
    pixels : List[int] | List[List[int]]
        List of pixel numbers for which the timestamp differences should
        be calculated or list of two lists with pixel numbers for peak
        vs. peak calculations.
    pix_coor : ndarray
        Array for transforming the pixel address in terms of TDC (0 to 3)
        to pixel number in terms of half of the sensor (0 to 255).
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
    pixels_left, pixels_right = utils.pixel_list_transform(pixels)

    pixels = _flatten(pixels)

    # Find ends of cycles
    cycle_ends = np.argwhere(data[0].T[0] == -2)
    cycle_ends = np.insert(cycle_ends, 0, 0)

    df_combined = pd.DataFrame([])

    for j, pixel in enumerate(pixels):

        tdc, pix_c = np.argwhere(pix_coor == pixel)[0]
        pix = np.where(data[tdc].T[0] == pix_c)[0]

        timestamps_filtered = []

        for i, _ in enumerate(cycle_ends[:-1]):
            slice_from = cycle_ends[i]
            slice_to = cycle_ends[i + 1]
            pix_slice = pix[(pix >= slice_from) & (pix < slice_to)]
            if not np.any(pix_slice):
                continue

            # Shift timestamps by cycle length
            tmsp = data[tdc].T[1][pix_slice]
            tmsp = tmsp[tmsp > 0]
            tmsp = tmsp + cycle_length * i

            timestamps_filtered.extend(tmsp)

        # Asign even numbers to pixels from the left beam and
        # odd - from the right
        if pixel in pixels_left:
            pix_ind = (
                np.zeros(len(timestamps_filtered), dtype=np.int32) + j * 2
            )
        else:
            pix_ind = np.zeros(len(timestamps_filtered), dtype=np.int32) + (
                j * 2 + 1
            )

        pix_data = np.vstack((pix_ind, timestamps_filtered))
        df = pd.DataFrame(pix_data.T, columns=["Pixel_index", "Timestamp"])

        df_combined = pd.concat((df_combined, df), ignore_index=True)

        # Sort the timestamps
        df_combined.sort_values("Timestamp", inplace=True)

        # Subtract pixel indicators of neighbors; values of 0
        # correspond to timestamp differences for the same pixel
        # '-1' and '1' - to differences from different pixels
        df_combined["Pixel_index_diff"] = df_combined["Pixel_index"].diff()

        # Calculate timestamp difference between neighbors
        df_combined["Timestamp_diff"] = df_combined["Timestamp"].diff()

        # Get the correct timestamp difference sign
        df_combined["Timestamp_diff"] = df_combined[
            "Timestamp_diff"
        ] * np.sign(df_combined["Pixel_index_diff"])

        # Collect timestamp differences where timestamps are from
        # different pixels
        filtered_df = df_combined[
            abs(df_combined["Pixel_index_diff"]) % 2 == 1
        ]

        # Save only timestamps differences in the requested window
        deltas_out = filtered_df[
            abs(filtered_df["Timestamp_diff"]) < delta_window
        ]["Timestamp_diff"].values

    return deltas_out


def calculate_and_save_timestamp_differences_mix(
    path: str,
    pixels: List[int] | List[List[int]],
    rewrite: bool,
    daughterboard_number: str,
    motherboard_number: str,
    firmware_version: str,
    timestamps: int = 512,
    delta_window: float = 50e3,
    cycle_length: float = None,
    app_mask: bool = True,
    include_offset: bool = False,
    apply_calibration: bool = True,
    absolute_timestamps: bool = False,
    correct_pix_address: bool = False,
):
    """Calculate and save timestamp differences into '.feather' file.

    Unpacks data into a dictionary, calculates timestamp differences for
    the requested pixels, and saves them into a '.feather' table. Works with
    firmware version 2212. Uses a faster algorithm.

    Parameters
    ----------
    path : str
        Path to the folder with '.dat' data files.
    pixels : List[int] | List[List[int]]
        List of pixel numbers for which the timestamp differences should
        be calculated and saved or list of two lists with pixel numbers
        for peak vs. peak calculations.
    rewrite : bool
        switch for rewriting the plot if it already exists. used as a
        safeguard to avoid unwanted overwriting of the previous results.
        Switch for rewriting the '.feather' file if it already exists.
    daughterboard_number : str
        LinoSPAD2 daughterboard number.
    motherboard_number : str
        LinoSPAD2 motherboard (FPGA) number, including the '#'.
    firmware_version: str
        LinoSPAD2 firmware version. Versions "2212s" (skip) and "2212b"
        (block) are recognized.
    timestamps : int, optional
        Number of timestamps per acquisition cycle per pixel. The default
        is 512.
    delta_window : float, optional
        Size of a window to which timestamp differences are compared.
        Differences in that window are saved. The default is 50e3 (50 ns).
    cycle_length: float, optional
        Length of the acquisition cycle. The default is None.
    app_mask : bool, optional
        Switch for applying the mask for hot pixels. The default is True.
    include_offset : bool, optional
        Switch for applying offset calibration. The default is True.
    apply_calibration : bool, optional
        Switch for applying TDC and offset calibration. If set to 'True'
        while apply_offset_calibration is set to 'False', only the TDC
        calibration is applied. The default is True.
    absolute_timestamps: bool, optional
        Indicator for data with absolute timestamps. The default is
        False.
    correct_pix_address : bool, optional
        Correct pixel address for the sensor half on side 23 of the
        daughterboard. The default is False.

    Raises
    ------
    TypeError
        Raised if "pixels" is not a list.
    TypeError
        Raised if "firmware_version" is not a string.
    TypeError
        Raised if "rewrite" is not a boolean.
    TypeError
        Raised if "daughterboard_number" is not a string.
    """
    # Parameter type check
    if isinstance(pixels, list) is False:
        raise TypeError(
            "'pixels' should be a list of integers or a list of two lists"
        )
    if isinstance(firmware_version, str) is False:
        raise TypeError(
            "'firmware_version' should be string, '2212s', '2212b' or '2208'"
        )
    if isinstance(rewrite, bool) is False:
        raise TypeError("'rewrite' should be boolean")
    if isinstance(daughterboard_number, str) is False:
        raise TypeError("'daughterboard_number' should be string")

    os.chdir(path)

    # Handle the input list
    pixels = utils.pixel_list_transform(pixels)
    files_all = glob.glob("*.dat")

    files_all = sorted(files_all)

    out_file_name = files_all[0][:-4] + "-" + files_all[-1][:-4]

    # Feather file counter for saving delta ts into separate files
    # of up to 100 MB
    ft_file_number = 0

    # Check if the feather file exists and if it should be rewrited
    feather_file = os.path.join(
        path, "delta_ts_data", f"{out_file_name}_mix.feather"
    )

    # Remove the old '.feather' files with the pattern
    # for ft_file in feather_files:
    utils.file_rewrite_handling(feather_file, rewrite)

    # Go back to the folder with '.dat' files
    os.chdir(path)

    # Collect the data for the required pixels
    print(
        "\n> > > Collecting data for delta t plot for the requested "
        "pixels and saving it to .feather in a cycle < < <\n"
    )
    # Define matrix of pixel coordinates, where rows are numbers of TDCs
    # and columns are the pixels that connected to these TDCs
    if firmware_version == "2212s":
        pix_coor = np.arange(256).reshape(4, 64).T
    elif firmware_version == "2212b":
        pix_coor = np.arange(256).reshape(64, 4)
    else:
        print("\nFirmware version is not recognized.")
        sys.exit()

    # Correct pixel addressing for motherboard on side '23'
    if correct_pix_address:
        pixels = utils.correct_pixels_address(pixels)

    # Mask the hot/warm pixels
    if app_mask is True:
        mask = utils.apply_mask(daughterboard_number, motherboard_number)
        if isinstance(pixels[0], int) and isinstance(pixels[1], int):
            pixels = [pix for pix in pixels if pix not in mask]
        else:
            pixels[0] = [pix for pix in pixels[0] if pix not in mask]
            pixels[1] = [pix for pix in pixels[1] if pix not in mask]

    for i in tqdm(range(ceil(len(files_all))), desc="Collecting data"):
        file = files_all[i]

        # Unpack data for the requested pixels into dictionary
        if not absolute_timestamps:
            data_all = f_up.unpack_binary_data(
                file,
                daughterboard_number,
                motherboard_number,
                firmware_version,
                timestamps,
                include_offset,
                apply_calibration,
            )
        else:
            data_all, _ = f_up.unpack_binary_data_with_absolute_timestamps(
                file,
                daughterboard_number,
                motherboard_number,
                firmware_version,
                timestamps,
                include_offset,
                apply_calibration,
            )

        # If cycle_length is not given manually, estimate from the data
        if cycle_length is None:
            cycle_length = np.max(data_all)

        delta_ts = cd.calculate_differences_mix(
            data_all, pixels, pix_coor, delta_window, cycle_length
        )

        # Save data as a .feather file in a cycle so data is not lost
        # in the case of failure close to the end
        delta_ts = pd.DataFrame(delta_ts)
        # delta_ts = delta_ts.T

        try:
            os.chdir("delta_ts_data")
        except FileNotFoundError:
            os.mkdir("delta_ts_data")
            os.chdir("delta_ts_data")

        # Check if feather file exists
        feather_file = f"mix_{out_file_name}_{ft_file_number}.feather"
        if os.path.isfile(feather_file):
            # Check the size of the existing '.feather', if larger
            # than 100 MB, create new one
            if os.path.getsize(feather_file) / 1024 / 1024 < 100:
                # Load existing feather file
                existing_data = ft.read_feather(feather_file)

                # Append new data to the existing feather file
                combined_data = pd.concat([existing_data, delta_ts], axis=0)
                ft.write_feather(combined_data, feather_file)
            else:
                ft_file_number += 1
                feather_file = f"mix_{out_file_name}_{ft_file_number}.feather"
                ft.write_feather(delta_ts, feather_file)

        else:
            # Save as a new feather file
            ft.write_feather(delta_ts, feather_file)
        os.chdir("..")

    # Combine the numbered feather files into a single one
    _combine_intermediate_feather_files(path)

    # Check, if the file was created
    if (
        os.path.isfile(path + f"/delta_ts_data/mix_{out_file_name}.feather")
        is True
    ):
        print(
            "\n> > > Timestamp differences are saved as"
            f"mix_{out_file_name}.feather in "
            f"{os.path.join(path, 'delta_ts_data')} < < <"
        )

    else:
        print("File wasn't generated. Check input parameters.")
