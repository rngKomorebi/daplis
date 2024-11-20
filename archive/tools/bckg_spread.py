import os
from glob import glob

import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
from pyarrow import feather as ft

from daplis.functions import utils

def _extend_spread_range(spread_bins, spread_counts, extension: int):
    """Extend the background spread bins and counts.

    Can be used to extend the background histogram to both sides while
    keeping the same bin size and assigning zero counts to the newly
    added bins. Should improve the fit and, therefore, the sigma value.

    Parameters
    ----------
    spread_bins : array-like
        Original bins of the background spread histogram.
    spread_counts : array-like
        Original counts of the background spread histogram.
    extension : int
        Number of elements to add to the original bins and counts from
        each side.

    Returns
    -------
    array-like, array-like
        Extended bins and counts of the background spread histogram.
    """
    # Calculate the step size between elements
    step_size = spread_bins[1] - spread_bins[0]

    # Create arrays to add on both sides
    left_extension = np.arange(
        spread_bins[0] - extension * step_size,
        spread_bins[0],
        step_size,
    )
    right_extension = np.arange(
        spread_bins[-1] + step_size,
        spread_bins[-1] + (extension + 1) * step_size,
        step_size,
    )

    # Concatenate arrays
    extended_array = np.concatenate(
        (left_extension, spread_bins, right_extension)
    )
    extended_counts = np.concatenate(
        (np.zeros(extension), spread_counts, np.zeros(extension))
    )

    return extended_array, extended_counts


def sigma_of_count_spread_to_average(
    path: str,
    pixels: list,
    step: int = 10,
    bins_sigma: int = 20,
    extend: int = 0,
):
    """Plot and fit background spread from the feather file.

    Collect timestamp differences for the requested pixels from the
    given feather file, plot histogram of the background signal,
    plot a jointplot of the background with the spread of the
    background, fit the spread and calculate the ratio of the sigma of
    that spread to average background signal.

    Parameters
    ----------
    path : str
        Path to the data files.
    ft_file : str
        Feather file with timestamp differences.
    pixels : list
        Pixels which should be used for analysis.
    step : int, optional
        Multiplier of the average LinoSPAD2 TDC width of 17.857 ps that
        is used for histograms. The default is 10.
    bins_sigma : int, optional
        Number of bins used for plotting the histogram of the background
        spread. The default 20.
    extend: int, optional
        Number of elements to add to the background spread histogram
        for a better fit. The default is 0, when no elements are added.
    """
    os.chdir(path)

    ft_file = glob.glob("*.feather")[0]

    ft_file_name = ft_file.split(".")[0]

    data = ft.read_feather(ft_file)

    data_cut = data[f"{pixels[0]},{pixels[1]}"]

    # Cut the data from the background only; without the offset
    # calibration, the delta t peak rarely goes outside the 10 ns mark
    data_cut = data_cut[(data_cut > 20e3) & (data_cut < 40e3)]

    # Bins in units of 17.857 ps of the average LinoSPAD2 TDC bin width
    bins = np.arange(
        np.min(data_cut), np.max(data_cut), 2.5 / 140 * 1e3 * step
    )

    counts, bin_edges = np.histogram(data_cut, bins=bins)

    bin_centers = (bin_edges - 2.5 / 140 * 1e3 * step / 2)[1:]

    plt.rcParams.update({"font.size": 22})

    try:
        os.chdir("results/bckg_spread")
    except Exception:
        os.makedirs("results/bckg_spread")
        os.chdir("results/bckg_spread")

    # Background histogram
    plt.figure(figsize=(10, 7))
    plt.step(bin_centers, counts, color="tomato")
    plt.title(f"Histogram of delta ts\nBin size is {bins[1] - bins[0]:.2f} ps")
    plt.xlabel(r"$\Delta$t [ps]")
    plt.ylabel("# of coincidences [-]")
    plt.savefig(f"{ft_file_name}_bckg_hist.png")

    # Seaborn join histograms of background including the spread
    sns.jointplot(
        x=bin_centers, y=counts, height=10, marginal_kws=dict(bins=bins_sigma)
    )
    plt.title("Histogram of delta ts with histograms of spread", fontsize=20)
    plt.xlabel(r"$\Delta$t [ps]", fontsize=20)
    plt.ylabel("# of coincidences [-]", fontsize=20)
    plt.savefig(f"{ft_file_name}_bckg_hist_joint.png")

    # Histogram of the spread plus Gaussian fit
    counts_spread, bin_edges_spread = np.histogram(counts, bins=bins_sigma)
    bin_centers_spread = (
        bin_edges_spread - (bin_edges_spread[1] - bin_edges_spread[0]) / 2
    )[1:]

    if extend > 0:
        bin_centers_spread, counts_spread = _extend_spread_range(
            bin_centers_spread, counts_spread, extend
        )

    pars, covs = utils.fit_gaussian(bin_centers_spread, counts_spread)

    fig, ax = plt.subplots(figsize=(10, 7))
    ax.step(
        bin_centers_spread,
        counts_spread,
        label="Spread of counts",
        color="darkorchid",
    )
    ax.plot(
        bin_centers_spread,
        utils.gaussian(bin_centers_spread, *pars),
        label="Fit",
        color="#cc8c32",
    )
    ax.set_title(
        f"Ratio of spread to average: {pars[2] / np.mean(counts) * 100:.1f} %"
    )
    ax.set_xlabel("Spread [-]")
    ax.set_ylabel("Counts [-]")
    ax.text(
        0.07,
        0.9,
        f"\u03C3={pars[2]:.2f}\u00B1{np.sqrt(covs[2,2]):.2f}",
        transform=ax.transAxes,
        fontsize=25,
        bbox=dict(
            facecolor="white", edgecolor="black", boxstyle="round,pad=0.5"
        ),
    )
    plt.savefig(f"{ft_file_name}_bckg_spread_hist_.png")


def sigma_of_count_spread_to_average_from_ft_file(
    path: str,
    ft_file: str,
    pixels: list,
    step: int = 10,
    bins_sigma: int = 20,
    extend: int = 0,
):
    """Plot and fit background spread from the feather file.

    Collect timestamp differences for the requested pixels from the
    given feather file, plot histogram of the background signal,
    plot a jointplot of the background with the spread of the
    background, fit the spread and calculate the ratio of the sigma of
    that spread to average background signal. Only the feather file
    is required without the raw data and the feather file is chosen
    directly.

    Parameters
    ----------
    path : str
        Path to where the feather file is.
    ft_file : str
        Feather file with timestamp differences.
    pixels : list
        Pixels which should be used for analysis.
    step : int, optional
        Multiplier of the average LinoSPAD2 TDC width of 17.857 ps that
        is used for histograms. The default is 10.
    bins_sigma : int, optional
        Number of bins used for plotting the histogram of the background
        spread. The default 20.
    extend: int, optional
        Number of elements to add to the background spread histogram
        for a better fit. The default is 0, when no elements are added.
    """

    ft_file_name = ft_file.split(".")[0]

    os.chdir(path)

    data = ft.read_feather(ft_file)

    data_cut = data[f"{pixels[0]},{pixels[1]}"]

    # Cut the data from the background only; without the offset
    # calibration, the delta t peak rarely goes outside the 10 ns mark
    data_cut = data_cut[(data_cut > 20e3) & (data_cut < 40e3)]

    # Bins in units of 17.857 ps of the average LinoSPAD2 TDC bin width
    bins = np.arange(
        np.min(data_cut), np.max(data_cut), 2.5 / 140 * 1e3 * step
    )

    counts, bin_edges = np.histogram(data_cut, bins=bins)

    bin_centers = (bin_edges - 2.5 / 140 * 1e3 * step / 2)[1:]

    plt.rcParams.update({"font.size": 22})

    try:
        os.chdir("results/bckg_spread")
    except FileNotFoundError:
        os.makedirs("results/bckg_spread")
        os.chdir("results/bckg_spread")

    # Background histogram
    plt.figure(figsize=(10, 7))
    plt.step(bin_centers, counts, color="tomato")
    plt.title(f"Histogram of delta ts\nBin size is {bins[1] - bins[0]:.2f} ps")
    plt.xlabel(r"$\Delta$t [ps]")
    plt.ylabel("# of coincidences [-]")
    plt.savefig(f"{ft_file_name}_bckg_hist.png")

    # Seaborn join histograms of background including the spread
    sns.jointplot(
        x=bin_centers, y=counts, height=10, marginal_kws=dict(bins=bins_sigma)
    )
    plt.title("Histogram of delta ts with histograms of spread", fontsize=20)
    plt.xlabel(r"$\Delta$t [ps]", fontsize=20)
    plt.ylabel("# of coincidences [-]", fontsize=20)
    plt.savefig(f"{ft_file_name}_bckg_hist_joint.png")

    # Histogram of the spread plus Gaussian fit
    counts_spread, bin_edges_spread = np.histogram(counts, bins=bins_sigma)
    bin_centers_spread = (
        bin_edges_spread - (bin_edges_spread[1] - bin_edges_spread[0]) / 2
    )[1:]

    # Extend the range (if required) for the background spread for a
    # better fit
    if extend > 0:
        bin_centers_spread, counts_spread = _extend_spread_range(
            bin_centers_spread, counts_spread, extend
        )

    pars, covs = utils.fit_gaussian(bin_centers_spread, counts_spread)

    fig, ax = plt.subplots(figsize=(10, 7))
    ax.step(
        bin_centers_spread,
        counts_spread,
        label="Spread of counts",
        color="darkorchid",
    )
    ax.plot(
        bin_centers_spread,
        utils.gaussian(bin_centers_spread, *pars),
        label="Fit",
        color="#cc8c32",
    )
    ax.set_title(
        f"Ratio of spread to average: {pars[2] / np.mean(counts) * 100:.1f} %"
    )
    ax.set_xlabel("Spread [-]")
    ax.set_ylabel("Counts [-]")
    ax.text(
        0.59,
        0.9,
        f"\u03C3={pars[2]:.2f}\u00B1{np.sqrt(covs[2,2]):.2f}",
        transform=ax.transAxes,
        fontsize=25,
        bbox=dict(
            facecolor="white", edgecolor="black", boxstyle="round,pad=0.5"
        ),
    )
    plt.savefig(f"{ft_file_name}_bckg_spread_hist_.png")


# path = r'/home/sj/Shared/Halogen_HBT/05.04.2024/delta_ts_data'
path = r'/media/sj/King4TB/LS2_Data/Halogen HBT/delayed/feather_combine'

# ft_file = r"0000248690-0000248869.feather"
ft_file = r'combined_file.feather'


sigma_of_count_spread_to_average_from_ft_file(path,
    ft_file, pixels=[144, 171], step=10, bins_sigma=30, extend=50
)

#####


from daplis.functions import fits, delta_t

# path = r"D:\LinoSPAD2\Data\board_NL11\Prague\Halogen_HBT"
path = r'C:\Users\bruce\Downloads\Ha_HBT(2)'

ft_file = r"combined_file.feather"

# ft_file = r"combined_file_grid.feather"

fits.fit_with_gaussian(path, ft_file=ft_file, pix_pair=[144, 171], window=20e3, step=11)

# delta_t.collect_and_plot_timestamp_differences(
#     path,
#     pixels=[[143, 144, 145], [170, 171, 172]],
#     rewrite=True,
#     ft_file=ft_file,
#     step=5,
# )

%matplotlib qt

delta_t.collect_and_plot_timestamp_differences(
    path,
    pixels=[145, 171],
    rewrite=True,
    ft_file=ft_file,
    step=7,
)


