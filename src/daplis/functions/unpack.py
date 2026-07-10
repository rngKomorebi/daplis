"""Module with scripts for unpacking data from LinoSPAD2.

This file can also be imported as a module and contains the following
functions:

    * unpack_binary_data - function for unpacking data from LinoSPAD2,
    firmware version 2212. Utilizes the numpy library to speed up the
    process.

    * unpack_binary_data_with_absolute_timestamps - function for
    unpacking data from LinoSPAD2, including the absolute timestamps,
    works with firmware version 2212.

"""

from __future__ import annotations

import numpy as np


def unpack_binary_data(
    file: str,
    daughterboard_number: str,
    motherboard_number: str,
    firmware_version: str,
    timestamps: int = 512,
) -> tuple[np.ndarray, np.ndarray]:
    """Unpacks binary-encoded data from LinoSPAD2 firmware version 2212.

    Parameters
    ----------
    file : str
        Path to the binary data file.
    daughterboard_number : str
        LinoSPAD2 daughterboard number.
    motherboard_number : str
        LinoSPAD2 motherboard (FPGA) number, including the "#".
    firmware_version : str
        LinoSPAD2 firmware version. Either '2212s' (skip) or '2212b' (block).
    timestamps : int, optional
        Number of timestamps per cycle per TDC per acquisition cycle.
        The default is 512.

    Returns
    -------
    data_pixels : array-like
        2D array of pixel coordinates in the TDC.
    data_timestamps : array-like
        2D array of photon timestamps.
    Raises
    ------
    TypeError
        If 'daughterboard_number', 'motherboard_number', or 'firmware_version'
        parameters are not of string type.

    Notes
    -----
    The returned data are two 2D arrays where rows represent TDC numbers,
    columns represent the data, and each cell contains a pixel number in
    the TDC (from 0 to 3) or the timestamp recorded by that pixel.
    """
    # Parameter type check
    if not isinstance(daughterboard_number, str):
        raise TypeError("'daughterboard_number' should be a string.")
    if not isinstance(motherboard_number, str):
        raise TypeError("'motherboard_number' should be a string.")
    if not isinstance(firmware_version, str):
        raise TypeError("'firmware_version' should be a string.")

    # Unpack binary data
    raw_data = np.fromfile(file, dtype=np.uint32)
    # Timestamps are stored in the lower 28 bits
    data_timestamps = (raw_data & 0xFFFFFFF).astype(np.int64)
    # Pixel address in the given TDC is 2 bits above timestamp
    data_pixels = ((raw_data >> 28) & 0x3).astype(np.int8)
    # Check the top bit, assign '-1' to invalid timestamps
    data_timestamps[raw_data < 0x80000000] = -1
    # Free up memory
    del raw_data

    # Number of acquisition cycles in each data file
    cycles = len(data_timestamps) // (timestamps * 65)
    # Transform into a matrix of size 65 by cycles*timestamps
    data_pixels = (
        data_pixels.reshape(cycles, 65, timestamps)
        .transpose((1, 0, 2))
        .reshape(65, -1)
    )

    data_timestamps = (
        data_timestamps.reshape(cycles, 65, timestamps)
        .transpose((1, 0, 2))
        .reshape(65, -1)
    )

    # Cut the 65th TDC that does not hold any actual data from pixels
    data_pixels = data_pixels[:-1]
    data_timestamps = data_timestamps[:-1]

    return data_pixels, data_timestamps


def unpack_binary_data_with_absolute_timestamps(
    file_path: str,
    daughterboard_number: str,
    motherboard_number: str,
    firmware_version: str,
    timestamps: int = 512,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Unpacks binary-encoded data from LinoSPAD2 firmware version 2212
    with absolute timestamps.

    Parameters
    ----------
    file_path : str
        Path to the binary data file.
    daughterboard_number : str
        LinoSPAD2 daughterboard number.
    motherboard_number : str
        LinoSPAD2 motherboard (FPGA) number, including the "#".
    firmware_version : str
        LinoSPAD2 firmware version. Either '2212s' (skip) or '2212b'
        (block) are accepted.
    timestamps : int, optional
        Number of timestamps per TDC per acquisition cycle.
        The default is 512.

    Returns
    -------
    data_pixels : ndarray
        2D array of pixel coordinates in the TDC, shape (64, cycles*timestamps).
    data_timestamps : ndarray
        2D array of photon timestamps, shape (64, cycles*timestamps).
    absolute_timestamps : ndarray
        Array of absolute timestamps in clock counts, one per acquisition cycle.

    Raises
    ------
    TypeError
        If 'daughterboard_number', 'motherboard_number', or
        'firmware_version' parameters are not of string type.

    Notes
    -----
    Each acquisition cycle is preceded by two full 32-bit words that
    together encode a 64-bit absolute timestamp: first word = counter
    bits 31:0, second word = counter bits 63:32 (verified against the
    firmware, tdc_array.vhd). The counter runs at 133.333 MHz, so one
    tick is 7.5 ns. It is free-running from FPGA configuration and is
    latched at the start of each acquisition window.
    """
    # Parameter type check
    if not isinstance(daughterboard_number, str):
        raise TypeError("'daughterboard_number' should be a string.")
    if not isinstance(motherboard_number, str):
        raise TypeError("'motherboard_number' should be a string.")
    if not isinstance(firmware_version, str):
        raise TypeError("'firmware_version' should be a string.")

    raw_data = np.fromfile(file_path, dtype=np.uint32)

    # Each cycle: 2 absolute-timestamp words + 65 TDCs * timestamps words
    cycle_words = timestamps * 65 + 2
    cycles = len(raw_data) // cycle_words

    # Extract absolute timestamps: first 2 words of every cycle.
    # Both words are full 32 bits of the 64-bit counter (low word first);
    # masking them to 28 bits like the photon words corrupts the value.
    cycle_starts = np.arange(cycles, dtype=np.int64) * cycle_words
    abs_low = raw_data[cycle_starts].astype(np.uint64)
    abs_high = raw_data[cycle_starts + 1].astype(np.uint64)
    absolute_timestamps = (abs_high << np.uint64(32)) | abs_low

    # Remove the absolute-timestamp words, leaving only TDC data
    abs_indices = np.sort(np.concatenate([cycle_starts, cycle_starts + 1]))
    raw_data = np.delete(raw_data, abs_indices)

    # Timestamps are stored in the lower 28 bits
    data_timestamps = (raw_data & 0xFFFFFFF).astype(np.int64)
    # Pixel address in the given TDC is 2 bits above timestamp
    data_pixels = ((raw_data >> 28) & 0x3).astype(np.int8)
    # Check the top bit, assign '-1' to invalid timestamps
    data_timestamps[raw_data < 0x80000000] = -1
    del raw_data

    # Transform into a matrix of shape (65, cycles*timestamps)
    data_pixels = (
        data_pixels.reshape(cycles, 65, timestamps)
        .transpose((1, 0, 2))
        .reshape(65, -1)
    )
    data_timestamps = (
        data_timestamps.reshape(cycles, 65, timestamps)
        .transpose((1, 0, 2))
        .reshape(65, -1)
    )

    # Cut the 65th TDC that does not hold any actual data from pixels
    data_pixels = data_pixels[:-1]
    data_timestamps = data_timestamps[:-1]

    return data_pixels, data_timestamps, absolute_timestamps
