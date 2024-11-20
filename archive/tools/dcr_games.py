import os
import pickle

import numpy as np
from matplotlib import pyplot as plt

from daplis.functions import cross_talk, sensor_plot

# path33 = r"D:\LinoSPAD2\Data\board_NL11\Prague\DCR\#33"
# path21 = r"D:\LinoSPAD2\Data\board_NL11\Prague\DCR\#21"
path33 = r"/media/sj/King4TB/LS2_Data/CT/#33"
path21 = r"/media/sj/King4TB/LS2_Data/CT/#21"

dcr33 = cross_talk.collect_dcr_by_file(
    path33,
    daughterboard_number="NL11",
    motherboard_number="#33",
    firmware_version="2212s",
    timestamps=1000,
)

file_path = "DCR_#33.pkl"
# os.chdir(file_path)
with open(file_path, "wb") as file:
    pickle.dump(dcr33, file)

dcr21 = cross_talk.collect_dcr_by_file(
    path21,
    daughterboard_number="NL11",
    motherboard_number="#21",
    firmware_version="2212s",
    timestamps=1000,
)

file_path = "DCR_#21.pkl"

with open(file_path, "wb") as file:
    pickle.dump(dcr21, file)


path = r"D:\LinoSPAD2\Data\board_NL11\Prague\DCR"

sensor_plot.plot_sensor_population_full_sensor(
    path,
    daughterboard_number="NL11",
    motherboard_number1="#33",
    motherboard_number2="#21",
    firmware_version="2212s",
    timestamps=1000,
    include_offset=False,
    app_mask=False,
)

##################
hot_pixels_21 = [5, 7, 35, 100, 121, 189, 198, 220, 225, 228, 229, 247]
hot_pixels_33 = [15, 50, 52, 66, 93, 98, 109, 122, 210, 231, 236]

# 33
plt.figure(figsize=(10, 8))
plt.rcParams.update({"font.size": 20})
x_axis = [x * 64 / 60 for x in range(0, 648)]
plt.xlabel("Time [min]")
plt.ylabel("Normalized DCR [-]")
plt.plot(
    x_axis,
    [x[15] for x in dcr33] / np.max([x[15] for x in dcr33]),
    "--",
    label="Hot pixel 15",
)
plt.plot(
    x_axis,
    [x[50] for x in dcr33] / np.max([x[50] for x in dcr33]),
    "--",
    label="Hot pixel 50",
)
plt.plot(
    x_axis,
    [x[122] for x in dcr33] / np.max([x[122] for x in dcr33]),
    "--",
    label="Hot pixel 121",
)
plt.plot(
    x_axis,
    [x[210] for x in dcr33] / np.max([x[210] for x in dcr33]),
    "--",
    label="Hot pixel 220",
)
plt.legend()

plt.figure(figsize=(10, 8))
plt.rcParams.update({"font.size": 20})
x_axis = [x * 64 / 60 for x in range(0, 648)]
plt.xlabel("Time [min]")
plt.ylabel("Normalized DCR [-]")
plt.plot(x_axis, [x[15] for x in dcr33], "--", label="Hot pixel 15")
plt.plot(x_axis, [x[50] for x in dcr33], "--", label="Hot pixel 50")
plt.plot(x_axis, [x[122] for x in dcr33], "--", label="Hot pixel 122")
plt.plot(x_axis, [x[210] for x in dcr33], "--", label="Hot pixel 210")
plt.legend()

# 21
plt.figure(figsize=(10, 8))
plt.rcParams.update({"font.size": 20})
x_axis = [x * 64 / 60 for x in range(0, 600)]
plt.xlabel("Time [min]")
plt.ylabel("Normalized DCR [-]")
plt.plot(
    x_axis,
    [x[5] for x in dcr] / np.max([x[5] for x in dcr]),
    "--",
    label="Hot pixel 5",
)
plt.plot(
    x_axis,
    [x[35] for x in dcr] / np.max([x[35] for x in dcr]),
    "--",
    label="Hot pixel 35",
)
plt.plot(
    x_axis,
    [x[100] for x in dcr] / np.max([x[100] for x in dcr]),
    "--",
    label="Hot pixel 100",
)
plt.plot(
    x_axis,
    [x[228] for x in dcr] / np.max([x[228] for x in dcr]),
    "--",
    label="Hot pixel 228",
)
plt.legend()


### OPEN PKL
os.chdir(r"/media/sj/King4TB/LS2_Data/CT/#33")
# os.chdir(r"/media/sj/King4TB/LS2_Data/CT/#21")

# os.chdir(r'/home/sj/Documents/Quantum_astrometry/CT/DCR')

with open("DCR_#33.pkl", "rb") as f:
    data = pickle.load(f)


plt.rcParams.update({"font.size": 22})
plt.figure(figsize=(12, 8))
plt.plot(
    [x for x in range(len(data))],
    np.median(data, axis=1),
    color="darkslateblue",
)
plt.title("NL11 #21")
plt.xlabel("Time [min]")
plt.ylabel("Median DCR [cps]")


#### TESTING PLOTTING AND MASKING ####
path = r"/media/sj/King4TB/LS2_Data/CT/#33/CT_#33"
os.chdir(path)
files = r"0000015843.dat
data = sensor_plot.collect_data_and_apply_mask(
    files,
    daughterboard_number="NL11",
    motherboard_number="#33",
    firmware_version="2212s",
    timestamps=1000,
    include_offset=False,
    app_mask=False,
)

# plot_tmsp.plot_sensor_population(
#     path,
#     daughterboard_number="NL11",
#     motherboard_number="#33",
#     firmware_version="2212s",
#     timestamps=1000,
#     include_offset=False,
#     single_file=True,
# )
plt.plot(data)

# os.chdir("/home/sj/GitHub/LinoSPAD2/src/LinoSPAD2/params/masks")
# mask = np.genfromtxt("mask_NL11_#33.txt").astype(int)


### DCR histogram

plt.figure(figsize=(12, 8))
plt.rcParams.update({"font.size": 22})
plt.hist(
    np.average(data, axis=0),
    # np.concatenate(data),
    # data,
    bins=np.logspace(np.log10(0.1), np.log10(np.max(data)), 200),
    # stacked=True,
    color='teal'
)
plt.xlim(10)
plt.xscale("log")
plt.yscale("log")
plt.xlabel("DCR [cps/pixel]")
plt.ylabel("Count [-]")

# custom_xticks = [1, 10, 100, 1000, 10000]
# custom_xlabels = ["{:.0e}".format(y) for y in custom_xticks]
# plt.xticks(custom_xticks, custom_xlabels)

index_hot = np.where(data[0]> 1000 )[0]
data_hot = data[0][index_hot]

index_cut = np.unique(np.concatenate([np.arange(x-1, x+2, 1) for x in np.where(data[0]> 1000 )[0]]))
index_left = np.delete(np.arange(0,256,1), index_cut)

data_cut = data[0][index_cut]
data_left = data[0][index_left]

plt.hist(
    data_cut,
    bins=np.logspace(np.log10(0.1), np.log10(np.max(data_cut)), 200),
)
plt.xscale("log")
plt.yscale("log")

plt.hist(
    data_left,
    bins=np.logspace(np.log10(0.1), np.log10(np.max(data_cut)), 200),
)
plt.xscale("log")
plt.yscale("log")

### Full and hot+neighbors
plt.figure(figsize=(12, 8))
plt.rcParams.update({"font.size": 22})
plt.hist(
    data[0],
    bins=np.logspace(np.log10(0.1), np.log10(np.max(data[0])), 200), label='All'
)
# plt.hist(
#         data_hot,
#         bins=np.logspace(np.log10(0.1), np.log10(np.max(data_hot)), 200),
#         color='lightblue', alpha=0.5, label="Hot"
# )
# plt.hist(
#         data_cut,
#         bins=np.logspace(np.log10(0.1), np.log10(np.max(data_cut)), 200),
#         color='orange', alpha=0.5, label="Hot+/-1"
# )
plt.plot(bins[1:], cumul)
# plt.hist(
#     [data[0], data_hot, data_cut],
#     bins=np.logspace(np.log10(0.1), np.log10(np.max(data[0])), 200),
#     stacked=True,
#     color=['blue', 'purple', 'orange'],
#     label=['All', 'Hot', 'Hot+/-1']
# )
plt.legend(loc="best")
plt.xlim(10)
plt.xscale("log")
plt.yscale("log")
plt.xlabel("DCR [cps/pixel]")
plt.ylabel("Count [-]")

custom_xticks = [1, 10, 100, 1000, 10000]
custom_xlabels = ["{:.0e}".format(y) for y in custom_xticks]


### With integral of histogram
plt.rcParams.update({"font.size": 22})

bins = np.logspace(np.log10(0.1), np.log10(np.max(data[0])), 200)

hist, bin_edges = np.histogram(np.average(data, axis=0), bins=bins)

bin_centers = (bin_edges - (bin_edges[1] - bin_edges[0])/2)[1:]

# Plot the histogram
fig, ax = plt.subplots(figsize=(12,8))
# ax.bar(bin_centers, hist, width=np.diff(bin_edges), edgecolor='black', align='edge', label='All', color='salmon')
ax.bar(bin_centers, hist, width=np.diff(bin_edges), label='All', color='salmon')
cumul = np.cumsum(hist)
ax1 = ax.twinx()
ax1.plot(bin_centers, cumul/256*100, color="teal", linewidth=3)
# ax.step(bin_centers, hist, label='All', color='salmon', fill=True,  alpha=0.7)

ax.set_xlim(10)
ax1.set_xlim(10)
ax.set_ylim(0)
ax1.set_ylim(0)
ax.set_xscale('log')
# ax.set_yscale('log')
# ax1.set_yscale('log')
ax.set_xlabel('DCR [cps/pixel]')
ax.set_ylabel("Count [-]")
ax1.set_ylabel('Integral [%]')
plt.show()
