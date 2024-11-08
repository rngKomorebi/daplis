import glob
import os

import numpy as np
from matplotlib import pyplot as plt
from scipy import signal as sg
from scipy.optimize import curve_fit
from tqdm import tqdm

from daplis.functions import unpack

path = "D:/LinoSPAD2/Data/board_A5/FW 2212 block/Spectrometer/Ar"
os.chdir(path)

files = glob.glob("*.dat*")

valid_per_pixel = np.zeros(256)

pix_coor = np.arange(256).reshape(64, 4)

for i in tqdm(range(len(files)), desc="Going through files"):
    data = unpack.unpack_binary_data(
        files[i],
        daughterboard_number="A5",
        motherboard_number="#36",
        firmware_version="2212b",
        timestamps=200,
        include_offset=False,
    )

    for i in range(0, 256):
        tdc, pix = np.argwhere(pix_coor == i)[0]
        ind = np.where(data[tdc].T[0] == pix)[0]
        valid_per_pixel[i] += len(np.where(data[tdc].T[1][ind] > 0)[0])

mask = [70, 205, 212, 95, 157, 165, 57, 123, 187, 118, 251]

for i in mask:
    valid_per_pixel[i] = 0

v_max = np.max(valid_per_pixel)

peak_pos = sg.find_peaks(valid_per_pixel, threshold=v_max / 10)[0]

pixels = np.arange(0, 256, 1)
print(peak_pos[-1] - peak_pos[-2])
nm_per_pix = (811.5311 / 1.0003 - 810.3692 / 1.0003) / 11
x_nm = nm_per_pix * pixels + 811.5311 / 1.0003 - nm_per_pix * peak_pos[-1]

peak_pos_nm = (
    np.array(peak_pos) * nm_per_pix + 811.5311 / 1.0003 - nm_per_pix * peak_pos[-1]
)


def gauss(x, A, x0, sigma, C):
    return A * np.exp(-((x - x0) ** 2) / (2 * sigma**2)) + C


sigma = 0.1
valid_bckg = np.copy(valid_per_pixel)
for i in range(len(peak_pos)):
    valid_bckg[peak_pos[i] - 3 : peak_pos[i] + 3] = 0

av_bkg = np.average(valid_bckg)

x_nm_cut = np.zeros((len(peak_pos), len(valid_per_pixel)))
valid_per_pixel_cut = np.zeros((len(peak_pos), len(valid_per_pixel)))
fit_plot = np.zeros((len(peak_pos), len(valid_per_pixel)))
par = np.zeros((len(peak_pos), 4))
pcov = np.zeros((len(peak_pos), 4, 4))
perr = np.zeros((len(peak_pos), 4))

for i in range(len(valid_per_pixel_cut)):
    valid_per_pixel_cut[i] = np.copy(valid_per_pixel)
    for j in np.delete(np.arange(len(peak_pos)), i):
        valid_per_pixel_cut[i][peak_pos[j] - 5 : peak_pos[j] + 5] = av_bkg


for i in range(len(peak_pos)):
    par[i], pcov[i] = curve_fit(
        gauss,
        x_nm,
        valid_per_pixel_cut[i],
        p0=[max(valid_per_pixel_cut[i]), peak_pos_nm[i], sigma, av_bkg],
    )
    perr[i] = np.sqrt(np.diag(pcov[i]))
    fit_plot[i] = gauss(x_nm, par[i][0], par[i][1], par[i][2], par[i][3])

colors = ["#008080", "#009480", "#00a880", "#00bc80", "#00d080", "#00e480"]
colors1 = ["#cd9bd8", "#da91c5", "#e189ae", "#e48397", "#e08080", "#ffda9e"]

plt.ion()
plt.figure(figsize=(16, 10))
plt.rcParams.update({"font.size": 16})
plt.xlabel("Wavelength [nm]")
plt.ylabel("Counts [-]")
plt.plot(x_nm, valid_per_pixel, "o-", color="salmon", label="Data")
for i in range(len(peak_pos)):
    plt.plot(
        x_nm[peak_pos[i] - 10 : peak_pos[i] + 10],
        fit_plot[i][peak_pos[i] - 10 : peak_pos[i] + 10],
        color=colors[i],
        label="\n"
        "\u03C3={p1}\u00B1{pe1} nm\n"
        "\u03BC={p2}\u00B1{pe2} nm".format(
            p1=format(par[i][2], ".3f"),
            p2=format(par[i][1], ".3f"),
            pe1=format(perr[i][2], ".3f"),
            pe2=format(perr[i][1], ".3f"),
        ),
    )


# plt.legend(loc="best", fontsize=12)

# =============================================================================
# Plot for a paper
# =============================================================================

plt.ioff()

fig, ax = plt.subplots(figsize=(16, 10))
plt.rcParams.update({"font.size": 28})
ax.set_xlabel("Wavelength [nm]")
ax.set_ylabel("Counts [-]")
ax.minorticks_on()
ax.tick_params(axis="both", width=1.5)
ax.tick_params(which="major", length=8)
ax.tick_params(which="minor", length=6)
ax.set_xlim(790, 820)
ax.set_ylim(-0.15e6, np.max(valid_per_pixel) + 0.5e6)
ax.plot(x_nm, valid_per_pixel, "o-", color="steelblue", label="Data")
for i in range(len(peak_pos)):
    ax.plot(
        x_nm[peak_pos[i] - 10 : peak_pos[i] + 10],
        fit_plot[i][peak_pos[i] - 10 : peak_pos[i] + 10],
        color=colors1[i],
        linewidth=2,
        label="\n"
        "\u03C3={p1}\u00B1{pe1} nm\n"
        "\u03bb={p2}\u00B1{pe2} nm".format(
            p1=format(par[i][2], ".3f"),
            p2=format(par[i][1], ".3f"),
            pe1=format(perr[i][2], ".3f"),
            pe2=format(perr[i][1], ".3f"),
        ),
    )
ax.legend(loc="best", fontsize=22)
plt.tight_layout()
try:
    os.chdir("results")
except:
    pass
plt.savefig("Ar_spec_for_paper.pdf", bbox_inches="tight")
# os.chdir("..")
