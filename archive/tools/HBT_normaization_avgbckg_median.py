import os

import numpy as np
from matplotlib import pyplot as plt
from pyarrow import feather as ft
from scipy.optimize import curve_fit

from daplis.functions import fits
from daplis.functions.utils import gaussian

file = r"D:\LinoSPAD2\Data\NL11\2025.11.03\2025.11.03\2025.11.03\delta_ts_data\0000246530-0000247129.feather"

data = ft.read_feather(file)

plt.hist(data, bins=300, range=(-8e3, -2e3))
plt.xlabel("$\Delta$t (ps)")

bin = 2500 / 140

bins = np.arange(-8e3, -2e3, 2500 / 140 * 3)
hist = plt.hist(data, bins=bins, range=(-8e3, -2e3))

# len(hist[0])
# len(hist[1])

# plt.plot(hist[0], hist[1])


####
bins = np.arange(-2e3, 5e3, 2500 / 140 * 3)
plt.hist(data, bins=bins, range=(-2e3, 5e3))

data_bckg = data[(data >= 2e3) & (data <= 12e3)].dropna()


bins = np.arange(np.min(data_bckg), np.max(data_bckg), 2500 / 140 * 3)
counts, _ = np.histogram(data_bckg, bins=bins)


bckg_avg = np.average(counts)
total_median = np.median(hist[0])

counts_hbt = hist[0] / 1
bin_edges = hist[1]
bin_centers = bin_edges[1:] - (bin_edges[1] - bin_edges[0]) / 2

plt.plot(bin_centers, counts_hbt, "o-")

params, covs = curve_fit(
    gaussian, bin_centers, counts_hbt, p0=[0.5, -5.2e3, 100, 1]
)

# sigma(C) = sqrt(Var(C)) = sqrt(Cov(C,C)) - how to get the error

std_err = np.sqrt(covs[2, 2])

params[2]
std_err


plt.figure()
plt.plot(bin_centers, counts_hbt, "o-", label="Data")
plt.plot(
    bin_centers, gaussian(bin_centers, *params), label="Gaussian fit\n$\sigma$"
)
plt.legend()

params_normalized_avg = params


####

path = r"D:\LinoSPAD2\Data\NL11\2025.11.03\2025.11.03\2025.11.03"

params_normalized_median = fits.fit_with_gaussian(
    path,
    pixels=[88, 190],
    multiplier=3,
    range_left=-8e3,
    range_right=-2e3,
    normalize=1,
    return_fit_params=1,
)
