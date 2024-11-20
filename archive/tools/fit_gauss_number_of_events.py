# Testing fits with gaussian with number of events as a parameter instead
# of amplitude
import os
from glob import glob

import numpy as np
from matplotlib import pyplot as plt
from pyarrow import feather as ft
from scipy.optimize import curve_fit


def gaussian(x, mu, sigma, N, C):
    return (
        N
        / (np.sqrt(2 * np.pi) * sigma)
        * np.exp(-0.5 * ((x - mu) / sigma) ** 2)
        + C
    )


path = r"D:\LinoSPAD2\Data\board_NL11\Prague\CT_HBT\Second try\CT_HBT_1-0m_80%"
os.chdir(os.path.join(path, "delta_ts_data"))

ft_file = glob("*.feather")[0]

data = ft.read_feather(ft_file, columns=["170,174"]).dropna()
step = 20
counts, bin_edges = np.histogram(
    data, bins=np.arange(np.min(data), np.max(data), step * 17.857)
)

x_data = (bin_edges - 17.857 * step / 2)[1:]
y_data = counts
# Fit the Gaussian function to data
pos_try = x_data[np.argmax(counts)]
sigma_try = 150
popul_try = len(data)
bckg_try = np.median(counts)
popt, pcov = curve_fit(
    gaussian, x_data, y_data, p0=[pos_try, sigma_try, popul_try, bckg_try]
)

# Extract the fitted parameters
mu_fit, sigma_fit, N_fit, bckg_fit = popt

print("Fitted parameters:")
print("Mean (mu):", mu_fit)
print("Standard deviation (sigma):", sigma_fit)
print("Peak population (N):", N_fit)

fig, ax = plt.subplots(figsize=(12, 8))
ax.plot(x_data, y_data, "o")
ax.plot(x_data, gaussian(x_data, *popt), "--")
