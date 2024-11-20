"""Module for fitting timestamp differences with Gaussian function.

This file can also be imported as a module and contains the following
functions:

    * fit_wg - fit timestamp differences of a pair of pixels with a
    gaussian function and plot both a histogram of timestamp
    differences and the fit in a single figure.

"""

import glob
import os

import numpy as np
import pandas as pd
from lmfit.models import ConstantModel, GaussianModel
from matplotlib import pyplot as plt
from pyarrow import feather as ft
from scipy import signal as sg
from scipy.optimize import curve_fit

from daplis.functions.utils import error_propagation_division, gaussian


def fit_wg_double(
    path,
    pix_pair: list,
    thrs: float = 1.2,
    window: float = 5e3,
    step: int = 1,
    color_d: str = "salmon",
    color_f: str = "teal",
    title_on: bool = True,
):
    """Fit with Gaussian function and plot it.
    #TODO

    Fits timestamp differences of a pair of pixels with Gaussian
    function and plots it next to the histogram of the differences.
    Timestamp differences are collected from a '.csv' file with those if
    such exists.

    Parameters
    ----------
    path : str
        Path to datafiles.
    pix_pair : list
        Two pixel numbers for which fit is done.
    window : float, optional
        Time range in which timestamp differences are fitted. The
        default is 5e3.
    step : int, optional
        Bins of delta t histogram should be in units of 17.857 (average
        LinoSPAD2 TDC bin width). Default is 1.

    Raises
    ------
    FileNotFoundError
        Raised when no '.dat' data files are found.
    FileNotFoundError
        Raised when no '.csv' file with timestamp differences is found.
    ValueError
        Raised when no data for the requested pair of pixels was found
        in the '.csv' file.

    Returns
    -------
    None.

    """
    plt.ion()

    def gauss(x, A, x0, sigma, C):
        return A * np.exp(-((x - x0) ** 2) / (2 * sigma**2)) + C

    os.chdir(path)

    files = glob.glob("*.dat*")
    file_name = files[0][:-4] + "-" + files[-1][:-4]

    try:
        os.chdir("delta_ts_data")
    except FileNotFoundError:
        raise ("\nFile with data not found")

    csv_file_name = glob.glob("*{}*".format(file_name))[0]
    if csv_file_name == []:
        raise FileNotFoundError("\nFile with data not found")

    data = pd.read_csv(
        "{}".format(csv_file_name),
        usecols=["{},{}".format(pix_pair[0], pix_pair[1])],
    )
    try:
        data_to_plot = data["{},{}".format(pix_pair[0], pix_pair[1])]
    except KeyError:
        print("\nThe requested pixel pair is not found")
    del data
    # Check if there any finite values
    if not np.any(~np.isnan(data_to_plot)):
        raise ValueError("\nNo data for the requested pixel pair available")

    data_to_plot = data_to_plot.dropna()
    data_to_plot = np.array(data_to_plot)
    # Use te given window for trimming the data for fitting
    data_to_plot = np.delete(data_to_plot, np.argwhere(data_to_plot < -window))
    data_to_plot = np.delete(data_to_plot, np.argwhere(data_to_plot > window))

    os.chdir("..")
    plt.rcParams.update({"font.size": 22})

    # bins must be in units of 17.857 ps
    bins = np.arange(np.min(data_to_plot), np.max(data_to_plot), 17.857 * step)

    # Calculate histogram of timestamp differences for primary guess
    # of fit parameters and selecting a narrower window for the fit
    # n, b, p = plt.hist(data_to_plot, bins=bins, color="teal")
    # plt.close("all")
    n, b = np.histogram(data_to_plot, bins)

    peak_pos = sg.find_peaks(n, height=np.median(n) * thrs)[0]
    # print(peak_pos, n[peak_pos])

    # try:
    #     n_argmax = np.argmax(n)
    #     cp_pos = (bins[n_argmax] + bins[n_argmax + 1]) / 2
    # except ValueError:
    #     print("Couldn't find position of histogram max")

    data_to_fit1 = np.delete(
        data_to_plot, np.argwhere(data_to_plot < b[peak_pos[0]] - window / 2)
    )
    data_to_fit1 = np.delete(
        data_to_plot, np.argwhere(data_to_plot > b[peak_pos[0]] + window / 2)
    )

    data_to_fit2 = np.delete(
        data_to_plot, np.argwhere(data_to_plot < b[peak_pos[1]] - window / 2)
    )
    data_to_fit2 = np.delete(
        data_to_plot, np.argwhere(data_to_plot > b[peak_pos[1]] + window / 2)
    )

    # bins must be in units of 17.857 ps
    bins = np.arange(np.min(data_to_plot), np.max(data_to_plot), 17.857 * step)

    n1, b1 = np.histogram(data_to_fit1, bins)
    n2, b2 = np.histogram(data_to_fit2, bins)

    b11 = (b1 - 17.857 * step / 2)[1:]
    b22 = (b2 - 17.857 * step / 2)[1:]

    sigma = 150

    av_bkg = np.average(n)

    par1, pcov1 = curve_fit(gauss, b11, n, p0=[max(n), b[peak_pos[0]], sigma, av_bkg])
    par2, pcov2 = curve_fit(gauss, b22, n, p0=[max(n), b[peak_pos[1]], sigma, av_bkg])

    # interpolate for smoother fit plot
    to_fit_b1 = np.linspace(np.min(b11), np.max(b11), len(b11) * 100)
    to_fit_n1 = gauss(to_fit_b1, par1[0], par1[1], par1[2], par1[3])

    to_fit_b2 = np.linspace(np.min(b22), np.max(b22), len(b22) * 100)
    to_fit_n2 = gauss(to_fit_b2, par2[0], par2[1], par2[2], par2[3])

    perr1 = np.sqrt(np.diag(pcov1))
    vis_er1 = par1[0] / par1[3] ** 2 * 100 * perr1[-1]
    perr2 = np.sqrt(np.diag(pcov2))
    vis_er2 = par2[0] / par2[3] ** 2 * 100 * perr2[-1]

    plt.figure(figsize=(12, 8))
    plt.xlabel(r"$\Delta$t [ps]")
    plt.ylabel("# of coincidences [-]")
    plt.step(
        b1[1:],
        n,
        color=color_d,
        label="data",
    )
    plt.plot(
        # b,
        # fit_plot,
        to_fit_b1,
        to_fit_n1,
        "-",
        color=color_f,
        label="CT\n"
        "\u03C3={p1}\u00B1{pe1} ps\n"
        "\u03BC={p2}\u00B1{pe2} ps\n"
        "vis={vis}\u00B1{vis_er} %\n"
        "bkg={bkg}\u00B1{bkg_er}".format(
            # "\u03C3_s={p3} ps".format(
            p1=format(par1[2], ".1f"),
            p2=format(par1[1], ".1f"),
            pe1=format(perr1[2], ".1f"),
            pe2=format(perr1[1], ".1f"),
            bkg=format(par1[3], ".1f"),
            bkg_er=format(perr1[3], ".1f"),
            vis=format(par1[0] / par1[3] * 100, ".1f"),
            vis_er=format(vis_er1, ".1f"),
            # p3=format(st_dev, ".2f"),
        ),
    )

    plt.plot(
        # b,
        # fit_plot,
        to_fit_b2,
        to_fit_n2,
        "-",
        color="navy",
        label="HBT\n"
        "\u03C3={p1}\u00B1{pe1} ps\n"
        "\u03BC={p2}\u00B1{pe2} ps\n"
        "vis={vis}\u00B1{vis_er} %\n"
        "bkg={bkg}\u00B1{bkg_er}".format(
            # "\u03C3_s={p3} ps".format(
            p1=format(par2[2], ".1f"),
            p2=format(par2[1], ".1f"),
            pe1=format(perr2[2], ".1f"),
            pe2=format(perr2[1], ".1f"),
            bkg=format(par2[3], ".1f"),
            bkg_er=format(perr2[3], ".1f"),
            vis=format(par2[0] / par2[3] * 100, ".1f"),
            vis_er=format(vis_er2, ".1f"),
            # p3=format(st_dev, ".2f"),
        ),
    )
    plt.legend(loc="best")
    if title_on is True:
        plt.title(
            "Gaussian fit of delta t histogram, pixels {}, {}".format(
                pix_pair[0], pix_pair[1]
            )
        )

    try:
        os.chdir("results/fits")
    except Exception:
        os.makedirs("results/fits")
        os.chdir("results/fits")

    plt.savefig(
        "{file}_pixels_"
        "{pix1},{pix2}_fit.png".format(
            file=file_name, pix1=pix_pair[0], pix2=pix_pair[1]
        )
    )

    plt.pause(0.1)
    os.chdir("../..")


def fit_wg_all(
    path,
    pix_pair: list,
    thrs: float = 1.2,
    window: float = 5e3,
    step: int = 1,
    color_d: str = "salmon",
    color_f: str = "teal",
    title_on: bool = True,
):
    """Fit with Gaussian function and plot it.
    #TODO

    Fits timestamp differences of a pair of pixels with Gaussian
    function and plots it next to the histogram of the differences.
    Timestamp differences are collected from a '.csv' file with those if
    such exists.

    Parameters
    ----------
    path : str
        Path to datafiles.
    pix_pair : list
        Two pixel numbers for which fit is done.
    window : float, optional
        Time range in which timestamp differences are fitted. The
        default is 5e3.
    step : int, optional
        Bins of delta t histogram should be in units of 17.857 (average
        LinoSPAD2 TDC bin width). Default is 1.

    Raises
    ------
    FileNotFoundError
        Raised when no '.dat' data files are found.
    FileNotFoundError
        Raised when no '.csv' file with timestamp differences is found.
    ValueError
        Raised when no data for the requested pair of pixels was found
        in the '.csv' file.

    Returns
    -------
    None.

    """

    def gauss(x, A, x0, sigma, C):
        return A * np.exp(-((x - x0) ** 2) / (2 * sigma**2)) + C

    # def gauss_CT(x, A, x0, C):
    #     return A * np.exp(-((x - x0) ** 2) / (2 * 90**2)) + C

    # def gauss_HBT(x, A, x0, C):
    #     return A * np.exp(-((x - x0) ** 2) / (2 * 120**2)) + C

    os.chdir(path)

    files = glob.glob("*.dat*")
    file_name = files[0][:-4] + "-" + files[-1][:-4]

    try:
        os.chdir("delta_ts_data")
    except FileNotFoundError:
        raise ("\nFile with data not found")

    feather_file_name = glob.glob("*{}.feather*".format(file_name))[0]
    if not feather_file_name:
        raise FileNotFoundError("\nFile with data not found")

    data_to_plot = ft.read_feather(
        "{}".format(feather_file_name),
        columns=["{},{}".format(pix_pair[0], pix_pair[1])],
    ).dropna()

    # Check if there any finite values
    if not np.any(~np.isnan(data_to_plot)):
        raise ValueError("\nNo data for the requested pixel pair available")

    data_to_plot = data_to_plot.dropna()
    data_to_plot = np.array(data_to_plot)
    # Use the given window for trimming the data for fitting
    data_to_plot = np.delete(data_to_plot, np.argwhere(data_to_plot < -window))
    data_to_plot = np.delete(data_to_plot, np.argwhere(data_to_plot > window))

    os.chdir("..")
    plt.rcParams.update({"font.size": 22})

    # bins must be in units of 17.857 ps
    bins = np.arange(np.min(data_to_plot), np.max(data_to_plot), 2500 / 140 * step)

    # Calculate histogram of timestamp differences for primary guess
    # of fit parameters and selecting a narrower window for the fit
    n, b = np.histogram(data_to_plot, bins)

    peak_pos = sg.find_peaks(n, height=np.median(n) * thrs)[0]

    # print(peak_pos)

    plt.figure(figsize=(12, 8))
    plt.xlabel(r"$\Delta$t [ps]")
    plt.ylabel("# of coincidences [-]")
    plt.step(
        b[1:],
        n,
        color=color_d,
        label="data",
    )

    color_f = [
        "teal",
        "navy",
        "limegreen",
        "orchid",
        "paleturquoise",
        "gold",
        "royalblue",
    ]
    plt.xticks(
        [-20e3, -10e3, 0, 10e3, 20e3],
        [
            f"{-20e3:.0f}",
            f"{-10e3:.0f}",
            f"{0:.0f}",
            f"{10e3:.0f}",
            f"{20e3:.0f}",
        ],
    )
    labels = ["HBT", "Cross-talk", "BS1", "BS2", "BS3"]

    for k, peak_ind in enumerate(peak_pos):
        data_to_fit1 = np.delete(
            data_to_plot, np.argwhere(data_to_plot < b[peak_ind] - window / 2)
        )
        data_to_fit1 = np.delete(
            data_to_plot, np.argwhere(data_to_plot > b[peak_ind] + window / 2)
        )

        # bins must be in units of 17.857 ps
        bins = np.arange(np.min(data_to_plot), np.max(data_to_plot), 2500 / 140 * step)

        n1, b1 = np.histogram(data_to_fit1, bins)

        b11 = (b1 - 2500 / 140 * step / 2)[1:]

        if np.std(b11) > 150:
            sigma = 150
        else:
            sigma = np.std(b11)

        av_bkg = np.average(n)

        # sigma_values = np.sqrt(n)
        if peak_ind < 0:
            par1, pcov1 = curve_fit(
                # gauss_HBT,
                gauss,
                b11,
                n,
                p0=[max(n), b[peak_pos[k]], sigma, av_bkg],
                # p0=[max(n), b[peak_pos[k]], av_bkg],
                # sigma=sigma_values,
            )
        else:
            par1, pcov1 = curve_fit(
                # gauss_CT,
                gauss,
                b11,
                n,
                p0=[max(n), b[peak_pos[k]], sigma, av_bkg],
                # p0=[max(n), b[peak_pos[k]], av_bkg],
                # sigma=sigma_values,
            )

        # interpolate for smoother fit plot
        to_fit_b1 = np.linspace(np.min(b11), np.max(b11), len(b11) * 1)
        to_fit_n1 = gauss(to_fit_b1, par1[0], par1[1], par1[2], par1[3])
        # if peak_ind < 0:
        #     to_fit_n1 = gauss_HBT(to_fit_b1, par1[0], par1[1], par1[-1])
        # else:
        #     to_fit_n1 = gauss_CT(to_fit_b1, par1[0], par1[1], par1[-1])

        perr1 = np.sqrt(np.diag(pcov1))

        vis = par1[0] / par1[-1] * 100

        vis_er1 = error_propagation_division(par1[0], perr1[0], par1[3], perr1[3])
        vis_er1 = vis_er1 / (vis / 100) * 100

        plt.plot(
            # b,
            # fit_plot,
            to_fit_b1,
            to_fit_n1,
            "-",
            color=color_f[k],
            label=f"{labels[k]}\n"
            "\u03C3={p1}\u00B1{pe1} ps\n"
            "\u03BC={p2}\u00B1{pe2} ps\n"
            "C={vis}\u00B1{vis_er} %"
            # "bkg={bkg}\u00B1{bkg_er}"
            .format(
                labels,
                # "\u03C3_s={p3} ps".format(
                p1=format(par1[2], ".0f"),
                p2=format(par1[1], ".0f"),
                pe1=format(perr1[2], ".0f"),
                pe2=format(perr1[1], ".0f"),
                # bkg=format(par1[-1], ".1f"),
                # bkg_er=format(perr1[-1], ".1f"),
                vis=format(vis, ".1f"),
                vis_er=format(vis_er1, ".1f"),
                # p3=format(st_dev, ".2f"),
            ),
        )
        # plt.vlines((par1[1]-2*par1[2], par1[1]+2*par1[2]), ymin =par1[3] ,ymax=par1[0]+par1[3])
        lower_limit = par1[1] - 2 * par1[2]
        upper_limit = par1[1] + 2 * par1[2]
        # if peak_ind < 0:
        #     lower_limit = par1[1] - 2 * 120
        #     upper_limit = par1[1] + 2 * 120
        # else:
        #     lower_limit = par1[1] - 2 * 90
        #     upper_limit = par1[1] + 2 * 90

        data_in_interval = data_to_plot[
            (data_to_plot >= lower_limit) & (data_to_plot <= upper_limit)
        ]

        bckg_center_position = par1[1] - 7 * par1[2]
        # if peak_ind < 0:
        #     bckg_center_position = par1[1] - 7 * 120
        #     bckg_in_2sigma = data_to_plot[
        #         (data_to_plot > bckg_center_position - 2 * 120)
        #         & (data_to_plot < bckg_center_position + 2 * 120)
        #     ]
        # else:
        #     bckg_center_position = par1[1] + 7 * 90
        #     bckg_in_2sigma = data_to_plot[
        #         (data_to_plot > bckg_center_position - 2 * 90)
        #         & (data_to_plot < bckg_center_position + 2 * 90)
        # ]
        bckg_in_2sigma = data_to_plot[
            (data_to_plot > bckg_center_position - 2 * par1[2])
            & (data_to_plot < bckg_center_position + 2 * par1[2])
        ]

        # Plot the Gaussian fit and the 2-sigma interval
        # plt.axvline(lower_limit, color="gray", linestyle="--")
        # plt.axvline(upper_limit, color="gray", linestyle="--")
        # print(len(data_in_interval), len(bckg_in_2sigma))
        er1 = np.sqrt(len(data_in_interval))
        er2 = np.sqrt(len(bckg_in_2sigma))
        print(
            f"Population of the peak at {b11[peak_ind]} in a 2sigma interval is: {len(data_in_interval) - len(bckg_in_2sigma)}"
            + "\u00B1"
            + f"{np.sqrt(er1**2+er2**2):.2f}"
        )

    plt.xlim(-window, window)
    plt.legend(loc="best")
    if title_on is True:
        plt.title(
            "Gaussian fit of delta t histogram, pixels {}, {}".format(
                pix_pair[0], pix_pair[1]
            )
        )

    try:
        os.chdir("results/fits")
    except Exception:
        os.makedirs("results/fits")
        os.chdir("results/fits")

    plt.savefig(
        "{file}_pixels_"
        "{pix1},{pix2}_fit.png".format(
            file=file_name, pix1=pix_pair[0], pix2=pix_pair[1]
        )
    )

    plt.pause(0.1)
    os.chdir("../..")


# def fit_all_lmfit(
#     path,
#     pix_pair: list,
#     thrs: float = 1.2,
#     window: float = 5e3,
#     step: int = 1,
#     color_d: str = "salmon",
#     color_f: str = "teal",
#     title_on: bool = True,
# ):
#     os.chdir(path)

#     files = glob.glob("*.dat*")
#     file_name = files[0][:-4] + "-" + files[-1][:-4]

#     try:
#         os.chdir("delta_ts_data")
#         feather_file_name = glob.glob("*{}.feather*".format(file_name))[0]
#     except (FileNotFoundError, IndexError):
#         raise FileNotFoundError("\nFile with data not found")

#     data_to_plot = ft.read_feather(
#         "{}".format(feather_file_name),
#         columns=["{},{}".format(pix_pair[0], pix_pair[1])],
#     ).dropna()

#     # Check if there any finite values
#     if not np.any(~np.isnan(data_to_plot)):
#         raise ValueError("\nNo data for the requested pixel pair available")

#     data_to_plot = data_to_plot.dropna()
#     data_to_plot = np.array(data_to_plot)
#     # Use the given window for trimming the data for fitting
#     data_to_plot = np.delete(
#         data_to_plot, np.argwhere(data_to_plot < -window / 2)
#     )
#     data_to_plot = np.delete(
#         data_to_plot, np.argwhere(data_to_plot > window / 2)
#     )

#     # Data is ready, time to do the histograms

#     # bins must be in units of 17.857 ps
#     bins = np.arange(np.min(data_to_plot), np.max(data_to_plot), 17.857 * step)

#     # Calculate histogram of timestamp differences for primary guess
#     # of fit parameters and selecting a narrower window for the fit
#     counts, bin_edges = np.histogram(data_to_plot, bins)

#     # bin_centers = (bin_edges - 17.857 * step / 2)[1:]
#     bin_centers = bin_edges[:-1] + 0.5 * 17.857 * step

#     peak_pos = sg.find_peaks(counts, height=np.median(counts) * thrs)[0]

#     plt.figure(figsize=(16, 10))
#     plt.xlabel(r"$\Delta$t [ps]")
#     plt.ylabel("# of coincidences [-]")
#     plt.step(
#         bin_centers,
#         counts,
#         color=color_d,
#         label="data",
#     )

#     color_f = [
#         "teal",
#         "navy",
#         "limegreen",
#         "orchid",
#         "paleturquoise",
#         "gold",
#         "royalblue",
#     ]

#     for k, peak_ind in enumerate(peak_pos):
#         counts_flatten = np.copy(counts)

#         peaks_to_flatten = np.delete(peak_pos, k)

#         for peak_go_flat in peaks_to_flatten:
#             counts_flatten[peak_go_flat - 3 : peak_go_flat + 3] = np.median(
#                 counts
#             )

#         av_bkg = np.average(counts)

#         mod_peak = GaussianModel(prefix="peak_")

#         mod_bckg = ConstantModel(prefix="bkg_")

#         gmodel = mod_peak + mod_bckg

#         center = bin_centers[peak_ind]
#         # amplitude = max(counts_flatten)
#         amplitude = counts[peak_ind]
#         if np.std(bin_centers) > 150:
#             sigma = 150
#         else:
#             sigma = np.std(bin_centers)

#         # Initial parameter values for the Gaussian model
#         params_res = gmodel.make_params(
#             peak_center=center,
#             bkg_c=av_bkg,
#             peak_amplitude=dict(
#                 value=amplitude, min=av_bkg, max=amplitude * 2
#             ),
#             peak_sigma=dict(value=sigma, min=40, max=200),
#         )

#         # params_res = mod_peak.guess(counts_flatten, x=bin_centers)
#         # params_res += mod_bckg.guess(np.median(counts), x=bin_centers)

#         res = gmodel.fit(counts_flatten, params_res, x=bin_centers)

#         amp_fit = res.params["peak_amplitude"].value
#         mean_fit = res.params["peak_center"].value
#         sigma_fit = res.params["peak_sigma"].value
#         background_fit = res.params["bkg_c"].value

#         amp_stderr = res.params["peak_amplitude"].stderr
#         mean_stderr = res.params["peak_center"].stderr
#         sigma_stderr = res.params["peak_sigma"].stderr
#         background_stderr = res.params["bkg_c"].stderr

#         print(amp_fit, mean_fit, sigma_fit, background_fit)
#         print(amp_stderr, mean_stderr, sigma_stderr, background_stderr)
#         try:
#             vis_er = amp_fit / background_fit**2 * 100 * amp_stderr
#         except TypeError:
#             vis_er = 0
#         # interpolate for smoother fit plot
#         to_fit_b1 = np.linspace(
#             np.min(bin_centers), np.max(bin_centers), len(bin_centers) * 100
#         )
#         to_fit_n1 = gaussian(
#             to_fit_b1, amp_fit, mean_fit, sigma_fit, background_fit
#         )

#         plt.plot(
#             to_fit_b1,
#             to_fit_n1,
#             "-",
#             color=color_f[k],
#             # label="CT\n"
#             # "\u03C3={p1}\u00B1{pe1} ps\n"
#             # "\u03BC={p2}\u00B1{pe2} ps\n"
#             # "vis={vis}\u00B1{vis_er} %\n"
#             # "bkg={bkg}\u00B1{bkg_er}".format(
#             #     p1=format(sigma_fit, ".1f"),
#             #     p2=format(mean_fit, ".1f"),
#             #     pe1=format(sigma_stderr, ".1f"),
#             #     pe2=format(mean_stderr, ".1f"),
#             #     bkg=format(background_fit, ".1f"),
#             #     bkg_er=format(background_stderr, ".1f"),
#             #     vis=format(amp_fit / background_fit * 100, ".1f"),
#             #     vis_er=format(vis_er, ".1f"),
#             # ),
#         )

#         lower_limit = mean_fit - 2 * sigma_fit
#         upper_limit = mean_fit + 2 * sigma_fit

#         data_in_interval = data_to_plot[
#             (data_to_plot >= lower_limit) & (data_to_plot <= upper_limit)
#         ]

#         bckg_center_position = mean_fit - 7 * sigma_fit
#         bckg_in_2sigma = data_to_plot[
#             (data_to_plot > bckg_center_position - 2 * sigma_fit)
#             & (data_to_plot < bckg_center_position + 2 * sigma_fit)
#         ]

#         # Plot the Gaussian fit and the 2-sigma interval
#         plt.axvline(lower_limit, color="gray", linestyle="--")
#         plt.axvline(upper_limit, color="gray", linestyle="--")
#         print(len(data_in_interval), len(bckg_in_2sigma))
#         print(
#             f"Population of the peak at {bin_centers[peak_ind]} in a 2sigma interval is: {len(data_in_interval) - len(bckg_in_2sigma)}"
#         )

#     plt.legend(loc="best")
#     if title_on is True:
#         plt.title(
#             "Gaussian fit of delta t histogram, pixels {}, {}".format(
#                 pix_pair[0], pix_pair[1]
#             )
#         )

#     try:
#         os.chdir("results/fits")
#     except FileNotFoundError as _:
#         os.makedirs("results/fits")
#         os.chdir("results/fits")

#     plt.savefig(
#         "{file}_pixels_"
#         "{pix1},{pix2}_fit.png".format(
#             file=file_name, pix1=pix_pair[0], pix2=pix_pair[1]
#         )
#     )

#     plt.pause(0.1)
#     os.chdir("../..")


# %matplotlib qt

path0 = r"D:\LinoSPAD2\Data\board_NL11\Prague\CT_HBT\Second try\CT_HBT_1-0m_full_int"
path1 = r"D:\LinoSPAD2\Data\board_NL11\Prague\CT_HBT\Second try\CT_HBT_2-0m_full_int"

path2 = r"D:\LinoSPAD2\Data\board_NL11\Prague\CT_HBT\Second try\CT_HBT_1-0m_80%"
path3 = r"D:\LinoSPAD2\Data\board_NL11\Prague\CT_HBT\Second try\CT_HBT_1-0m_70%"
path4 = r"D:\LinoSPAD2\Data\board_NL11\Prague\CT_HBT\Second try\CT_HBT_1-0m_60%"
path5 = r"D:\LinoSPAD2\Data\board_NL11\Prague\CT_HBT\Second try\CT_HBT_1-0m_50%"

path6 = r"D:\LinoSPAD2\Data\board_NL11\Prague\CT_HBT\Second try\CT_HBT_2-0m_80%_int"
path7 = r"D:\LinoSPAD2\Data\board_NL11\Prague\CT_HBT\Second try\CT_HBT_2-0m_70%_int"
path8 = r"D:\LinoSPAD2\Data\board_NL11\Prague\CT_HBT\Second try\CT_HBT_2-0m_60%_int"
path9 = r"D:\LinoSPAD2\Data\board_NL11\Prague\CT_HBT\Second try\CT_HBT_2-0m_50%_int"

# path0 = r"D:\LinoSPAD2\Data\board_NL11\Prague\CT_HBT\Second try\CT_HBT_3-0m_full_int"
# path1 = (
#     r"D:\LinoSPAD2\Data\board_NL11\Prague\CT_HBT\Second try\CT_HBT_3-0m_90%"
# )
# path2 = (
#     r"D:\LinoSPAD2\Data\board_NL11\Prague\CT_HBT\Second try\CT_HBT_3-0m_70%"
# )
# path3 = (
#     r"D:\LinoSPAD2\Data\board_NL11\Prague\CT_HBT\Second try\CT_HBT_3-0m_60%"
# )
# path4 = (
#     r"D:\LinoSPAD2\Data\board_NL11\Prague\CT_HBT\Second try\CT_HBT_3-0m_50%"
# )

paths = [path0, path1, path2, path3, path4, path5, path6, path7, path8, path9]
# paths = [path0, path1, path2, path3, path4]

from daplis.functions import delta_t, sensor_plot

for path in paths:
    # plot_tmsp.plot_sensor_population(
    #     path,
    #     daughterboard_number="NL11",
    #     motherboard_number="#33",
    #     firmware_version="2212s",
    #     timestamps=300,
    #     include_offset=False,
    #     fit_peaks=True,
    #     pickle_fig=True,
    #     show_fig=True,
    # )

    # delta_t.calculate_and_save_timestamp_differences(
    #     path,
    #     pixels=[
    #         [x for x in range(170 - 2, 170 + 3)],
    #         [x for x in range(174 - 2, 174 + 3)],
    #     ],
    #     rewrite=True,
    #     daughterboard_number="NL11",
    #     motherboard_number="#33",
    #     firmware_version="2212s",
    #     timestamps=300,
    #     include_offset=False,
    # )

    # delta_t.collect_and_plot_timestamp_differences(
    #     path,
    #     pixels=[168, 169, 170, 171, 172, 172, 173, 174, 175, 176],
    #     rewrite=True,
    #     step=3,
    # )

    fit_wg_all(
        path,
        pix_pair=[170, 174],
        window=20e3,
        step=10,
        thrs=1.25,
        title_on=False,
    )


### Full intensity only

path0 = r"D:\LinoSPAD2\Data\board_NL11\Prague\CT_HBT\Second try\CT_HBT_1-0m_full_int"
path1 = r"D:\LinoSPAD2\Data\board_NL11\Prague\CT_HBT\Second try\CT_HBT_2-0m_full_int"
path2 = r"D:\LinoSPAD2\Data\board_NL11\Prague\CT_HBT\Second try\CT_HBT_3-0m_full_int"

paths = [path0, path1, path2]

import itertools

pix_pairs = list(itertools.product([169, 170, 171], [173, 174, 175]))
pix_pairs = [[x, y] for x, y in pix_pairs]

# for path in paths:
for pix_pair in pix_pairs:
    fit_wg_all(
        path1,
        pix_pair=[171, 173],
        # pix_pair=pix_pair,
        window=20e3,
        step=7,
        thrs=1.35,
        title_on=False,
    )
