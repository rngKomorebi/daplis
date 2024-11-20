import os
from glob import glob

import numpy as np
from matplotlib import pyplot as plt
from pyarrow import feather as ft
from scipy import signal as sg

from daplis.functions import utils

init_path = r"D:\LinoSPAD2\Data\board_NL11\Prague\CT_HBT\Second try"
os.chdir(init_path)
folders = glob("*/")

folders = [folder for folder in folders if "CT_HBT" in folder]
paths = []
for folder in folders:
    paths.append(os.path.join(init_path, folder, "delta_ts_data"))

for path in paths[:-1]:

    os.chdir(path)

    ft_file = glob("*.feather")[0]

    data = ft.read_feather(
        ft_file,
        columns=["170,174"],
    ).dropna()

    step = 10
    window = 203

    data = np.array(data).T[0]

    data = data[(data > -20e3) & (data < 20e3)]

    bins = np.arange(np.min(data), np.max(data), 17.857 * step)

    counts, bin_edges = np.histogram(data, bins=bins)

    bin_centers = (bin_edges - 17.857 * step / 2)[1:]

    thrs = 1.35

    peak_pos = sg.find_peaks(counts, height=np.median(counts) * thrs)[0]

    # for k, peak_ind in enumerate(peak_pos):
    peak_ind = peak_pos[0]
    data_to_fit1 = np.delete(
        data, np.argwhere(data < bin_centers[peak_ind] - window / 2)
    )
    data_to_fit1 = np.delete(
        data, np.argwhere(data > bin_centers[peak_ind] + window / 2)
    )

    # bins must be in units of 17.857 ps
    bins = np.arange(np.min(data), np.max(data), 17.857 * step)

    n1, b1 = np.histogram(data_to_fit1, bins)

    b11 = (b1 - 17.857 * step / 2)[1:]

    if np.std(b11) > 150:
        sigma = 150
    else:
        sigma = np.std(b11)

    av_bkg = np.average(counts)

    sigma_values = np.sqrt(counts)

    par1, pcov1 = utils.fit_gaussian(bin_centers, counts)

    # interpolate for smoother fit plot
    to_fit_b1 = np.linspace(np.min(b11), np.max(b11), len(b11) * 1)
    to_fit_n1 = utils.gaussian(to_fit_b1, par1[0], par1[1], par1[2], par1[3])

    perr1 = np.sqrt(np.diag(pcov1))
    vis_er1 = par1[0] / par1[3] ** 2 * 100 * perr1[-1]

    ratios = []
    steps = [1, 1.25, 1.5, 1.75, 2, 2.25, 2.5, 2.75, 3]

    for step in steps:
        lower_limit = par1[1] - step * par1[2]
        upper_limit = par1[1] + step * par1[2]

        data_in_interval = data[(data >= lower_limit) & (data <= upper_limit)]

        bckg_center_position = par1[1] - 7 * par1[2]
        bckg_in_2sigma = data[
            (data > bckg_center_position - step * par1[2])
            & (data < bckg_center_position + step * par1[2])
        ]

        # Plot the Gaussian fit and the 2-sigma interval
        # plt.figure(figsize=(10, 7))
        # plt.plot(bin_centers, counts, "o", color="salmon")
        # plt.plot(to_fit_b1, to_fit_n1, "--", color="teal")
        # plt.axvline(lower_limit, color="gray", linestyle="--")
        # plt.axvline(upper_limit, color="gray", linestyle="--")
        ratios.append(len(data_in_interval) / np.sqrt(len(bckg_in_2sigma)))
        print(f"Ratio of peak population to sqrt of bckg: {ratios[-1]:.2f}")
        print(
            f"Population of the peak at {b11[peak_ind]:.2f} in a {step}-sigma interval is: {len(data_in_interval) - len(bckg_in_2sigma)}"
        )

    plt.figure(figsize=(10, 7))
    plt.plot(steps, ratios)
    plt.title(f"Peak population in n*σ / √(bckg in n*σ)")
    plt.xlabel("Sigma multiplier [-]")
    plt.ylabel("Ratio [-]")
