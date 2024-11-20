import os
from glob import glob

import numpy as np
from matplotlib import pyplot as plt

plt.rcParams.update({"font.size": 27})


# os.chdir(r"D:\LinoSPAD2\Data\B7d\2024.10.14\TDC_calib_2212b")
os.chdir(r"D:\LinoSPAD2\Data\B7d\2024.10.29\TDC_2212b")

files = glob("*.dat")

timestamps = 1000

M_DNL_global = np.zeros((64, 140, 140))

for tdc in range(64):
    counts_all = []
    print(f"doing TDC {tdc}")
    for file in files:

        # Unpack binary data
        raw_data = np.fromfile(file, dtype=np.uint32)
        # Timestamps are stored in the lower 28 bits
        data_timestamps = (raw_data & 0xFFFFFFF).astype(np.int64)
        # Pixel address in the given TDC is 2 bits above timestamp
        data_timestamps[raw_data < 0x80000000] = -1
        # Free up memory
        del raw_data

        # Number of acquisition cycles in each data file
        cycles = len(data_timestamps) // (timestamps * 65)

        data_timestamps = (
            data_timestamps.reshape(cycles, 65, timestamps)
            .transpose((1, 0, 2))
            .reshape(65, -1)
        )

        data_to_plot = data_timestamps[tdc]

        data_to_plot = data_to_plot[data_to_plot > 0] % 140

        bins = np.arange(0, 141, 1)

        counts, _ = np.histogram(data_to_plot, bins=bins)

        counts_all.append(counts)

    counts_corr = np.sum(counts_all, axis=0) / np.sum(counts_all) * 2500
    nonzero_start_counts = np.where(counts_corr > 0)[0][0]

    ### Correcting DNL ###

    M = np.zeros((140 - nonzero_start_counts, 140 - nonzero_start_counts))
    S_in = counts_corr

    S_in_pos = S_in[nonzero_start_counts:]

    S_out = 2500 / (140 - nonzero_start_counts)

    for i in range(140 - nonzero_start_counts):
        P_in_i = np.sum(S_in_pos[:i])

        for j in range(140 - nonzero_start_counts):
            P_out_j = j * 2500 / (140 - nonzero_start_counts)
            stuff = (
                np.min((P_in_i + S_in_pos[i], P_out_j + S_out))
                - np.max((P_in_i, P_out_j))
            ) / S_in_pos[i]

            M[i, j] = np.max((0, stuff))

    M = M.T

    M_DNL_global[tdc][nonzero_start_counts:140, nonzero_start_counts:140] = M

np.save("B7d_#28_2212b_DNL.npy", M_DNL_global)
