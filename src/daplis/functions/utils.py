"""Module of utility functions.

This module provides a set of functions used in other LinoSPAD2 functions,
covering tasks such as applying masks, unpickling and displaying
matplotlib figures, defining Gaussian functions, fitting them to data,
and transforming lists of pixels.

This file can also be imported as a module and contains the following
functions:

    * apply_mask - Apply a mask to the given data based on the
    daughterboard and motherboard numbers.

    * unpickle_plot - Unpickle and display a matplotlib figure based on
    the specified type.

    * gaussian - Gaussian function for curve fitting.

    * fit_gaussian - Fit Gaussian function to data and return optimal
    parameters and covariance.

    * pixel_list_transform - Transform a list of pixels into two separate
    lists based on input type.
    
    * correct_pixels_address - correct the pixel addressing, the output
    has the same dimensions as the input. Should be used for motherboard
    on side "23" of the daughterboard.

    * file_rewrite_handling - based on the file name given and the
    boolean paramter 'rewrite', handles the file overwriting based on
    its' existence. Introduced for clearer code and modularity.
    
    * error_propagation_division - propagate error for the division
    operation
    
    * combine_feather_files - Combine multiple '.feather' files into one.
    Can be used in delta_t functions for collecting timestamp differences
    to alleviate bottlenecks in saving results or for combining files
    manually for testing. 
    

"""

import os
import pickle
import sys
import time
from glob import glob
from typing import List

import matplotlib
import numpy as np
import pandas as pd
from pyarrow import feather as ft
from scipy.optimize import curve_fit


def apply_mask(
    daughterboard_number: str, motherboard_number: str
) -> np.ndarray:
    """Find and return mask for the requested motherboard.

    Parameters
    ----------
    daughterboard_number : str
        The LinoSPAD2 daughterboard number.
    motherboard_number : str
        LinoSPAD2 motherboard (FPGA) number, including the "#".

    Returns
    -------
    mask : np.ndarray
        The mask array generated from the given daughterboard and motherboard numbers.
    """

    path_to_back = os.getcwd()
    path_to_mask = os.path.join(
        os.path.dirname(os.path.realpath(__file__)),
        "..",
        "params",
        "masks",
    )
    os.chdir(path_to_mask)
    file_mask = glob(f"*{daughterboard_number}_{motherboard_number}*")[0]
    mask = np.genfromtxt(file_mask).astype(int)
    os.chdir(path_to_back)

    return mask


def unpickle_plot(path: str, plot_type: str, interactive: bool = True) -> None:
    """Unpickle and display a matplotlib figure.

    This function loads a previously pickled matplotlib figure from the
    specified path and displays it. Optionally, it can switch to an
    interactive backend.

    Parameters
    ----------
    path : str
        Path to the folder containing the pickled matplotlib figure.
    plot_type : str
        Type of the plot. Should be one of 'sensor_population', 'fits',
        or 'delta_t'.
    interactive : bool, optional
        Switch for making the plot interactive by using QtAgg matplotlib
        backend. The default is True.

    Raises
    ------
    TypeError
        Raised if the 'type' parameter is not one of 'sensor_population',
        'fits', or 'delta_t'.

    Returns
    -------
    None.

    """
    if plot_type not in ["sensor_population", "fits", "delta_t"]:
        raise TypeError(
            "Type should be 'sensor_population', 'fits', or 'delta_t'"
        )

    os.chdir(path)
    files = glob("*.dat*")
    plot_name = files[0][:-4] + "-" + files[-1][:-4]

    os.chdir(f"results/{plot_type}")

    if interactive:
        matplotlib.use("QtAgg")

    with open(f"{plot_name}.pickle", "rb") as file:
        fig = pickle.load(file)
        fig.show()


def gaussian(x, amp, mu, sigma, bkg):
    """Gaussian function.

    Parameters
    ----------
    x : array-like
        The input data.
    amp : float
        Amplitude of the Gaussian.
    mu : float
        Mean (center) of the Gaussian.
    sigma : float
        Standard deviation of the Gaussian.
    bkg : float
        Background offset.

    Returns
    -------
    array-like
        The computed Gaussian values.

    """
    return amp * np.exp(-((x - mu) ** 2) / (2 * sigma**2)) + bkg


def fit_gaussian(x, y):
    """Fit Gaussian function to data.

    Parameters
    ----------
    x : array-like
        The x-axis data.
    y : array-like
        The y-axis data.

    Returns
    -------
    tuple
        A tuple containing two elements:
        - popt : array-like
            Optimal values for the parameters so that the sum of the
            squared residuals of `gaussian(x, *popt) - y` is minimized.
        - pcov : 2D array-like
            Estimated covariance of `popt`. The diagonal elements represent
            the variance of the parameter estimates.

    """
    # Initial guess for the parameters
    amp_guess = np.max(y)
    mu_guess = x[np.argmax(y)]
    # As std sometimes gives nonsense, 150 ps is added as balancing
    sigma_guess = min(np.std(x), 150)
    bkg_guess = np.median(y)

    # Perform the curve fitting
    popt, pcov = curve_fit(
        gaussian, x, y, p0=[amp_guess, mu_guess, sigma_guess, bkg_guess]
    )

    return popt, pcov


def pixel_list_transform(pixels: list):
    """Transform a list of pixels into two separate lists.

    Transform the given list of pixels into two separate lists,
    based on the input type (list of integers, of lists, or a mix of
    both).

    Parameters:
        pixels : list
            A list of pixels.

    Returns:
        list: A list of the left pixels.
        list: A list of the right pixels.
    """

    if isinstance(pixels[0], list) and isinstance(pixels[1], list) is True:
        pixels_left, pixels_right = sorted(pixels)
    elif isinstance(pixels[0], int) and isinstance(pixels[1], list) is True:
        pixels_left, pixels_right = sorted([[pixels[0]], pixels[1]])
    elif isinstance(pixels[0], list) and isinstance(pixels[1], int) is True:
        pixels_left, pixels_right = sorted([pixels[0], [pixels[1]]])
    elif isinstance(pixels[0], int) and isinstance(pixels[1], int) is True:
        pixels_left = [pixels[0]]
        pixels_right = [pixels[1]]

    return [pixels_left, pixels_right]


def __correct_pix_address(pix: int):
    """Pixel address correction.

    Should be used internally only with the "correct_pixel_address"
    function. Transforms the pixel address based on its position.

    Parameters
    ----------
    pix : int
        Pixel address.

    Returns
    -------
    int
        Transformed pixel address.
    """
    if pix > 127:
        pix = 255 - pix
    else:
        pix = pix + 128
    return pix


def correct_pixels_address(pixels: List[int] | List[List[int]]):
    """Correct pixel address for all given pixels.

    Return the list with the same dimensions as the input.

    Parameters
    ----------
    pixels : List[int] | List[List[int]]
        List of pixel addresses.

    Returns
    -------
    List[int] | List[List[int]]
        List of transformed pixel addresses.
    """
    if isinstance(pixels, list):
        return [correct_pixels_address(item) for item in pixels]
    else:
        return __correct_pix_address(pixels)


def file_rewrite_handling(file: str, rewrite: bool):
    """Handle file rewriting based on the 'rewrite' parameter.

    The function checks if the specified file exists in the
    'delta_ts_data' directory.
    If it exists and 'rewrite' is True, the file is deleted after a
    countdown (5 sec), this is a good time to stop execution if needed.
    If 'rewrite' is False and the file exists, the function exits with a
    system exit error message.

    Parameters
    ----------
    file : str
        The file path or name to check for existence and potentially
        rewrite.
    rewrite : bool
        If True, the function attempts to rewrite the file after
        deleting it.
        If False and the file already exists, the function exits with an
        error message.

    Returns
    -------
    None

    Raises
    ------
    SystemExit
        If 'rewrite' is False and the file already exists, the function
        exits with an error message.

    """
    try:
        # os.chdir("delta_ts_data")
        if os.path.isfile(file):
            if rewrite is True:
                print(
                    "\n! ! ! Feather file with timestamps differences already "
                    "exists and will be rewritten ! ! !\n"
                )
                for i in range(5):
                    print(f"\n! ! ! Deleting the file in {5 - i} ! ! !\n")
                    time.sleep(1)
                os.remove(file)
            else:
                sys.exit(
                    "\n Feather file already exists, 'rewrite' set to"
                    "'False', exiting."
                )
        # os.chdir("..")
    except FileNotFoundError:
        pass


def error_propagation_division(x, sigma_x, y, sigma_y, rho_xy=0):
    """Calculate error propagation for division operation.

    Parameters
    ----------
    x : float
        Numerator value.
    sigma_x : float
        Uncertainty (standard deviation) associated with x.
    y : float
        Denominator value.
    sigma_y : float
        Uncertainty (standard deviation) associated with y.
    rho_xy : float, optional
        Correlation coefficient between x and y (default is 0).

    Returns
    -------
    float
        The uncertainty (standard deviation) of the result of the division.

    Raises
    ------
    ValueError
        If any of the input values are non-numeric.
    """
    partial_derivative_x = 1 / y
    partial_derivative_y = -x / (y**2)

    term1 = (partial_derivative_x**2) * (sigma_x**2)
    term2 = (partial_derivative_y**2) * (sigma_y**2)
    term3 = (
        2
        * partial_derivative_x
        * partial_derivative_y
        * rho_xy
        * sigma_x
        * sigma_y
    )

    sigma_f = np.sqrt(term1 + term2 - term3)

    return sigma_f


def combine_feather_files(path: str, return_data: bool = False):
    """Find all '.feather' files in the folder and combine into one.

    Unpacks the data from each '.feather' file found in the folder,
    concatenates them in a loop, and saved as a 'combined.feather' file.
    In successful runs, the pre-existing 'combined.feather' file is
    ignored and is rewritten. Can be used to combine multiple '.feather'
    files into a single one.

    Parameters
    ----------
    path : str
        Path to the folder with the '.feather' files that need to be
        combined.
    return_data : bool, optional
        Switch for returning the combined dataframe, by default False.

    Returns
    -------
    pandas.DataFrame
        Pandas dataframe with the combined data from all '.feather'
        files found in the folder. Returned only if requested via the
        'return_data' parameter.
    """
    os.chdir(path)

    ft_files = glob("*.feather")

    if "combined.feather" in ft_files:
        ft_files.remove("combined.feather")

    data_all = pd.DataFrame()

    for file in ft_files:
        data = ft.read_feather(file)
        data_all = pd.concat((data_all, data), ignore_index=True)

    data_all.to_feather("combined.feather")

    if return_data:
        return data_all
