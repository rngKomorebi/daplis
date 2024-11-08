import os
import pickle
from datetime import datetime
from glob import glob

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from scipy.optimize import curve_fit

from daplis.functions import cross_talk, delta_t

# path = r"D:\LinoSPAD2\Data\B7d\DCR\2024.09.24"

path = r"D:\LinoSPAD2\Data\B7d\DCR_CT"

# cross_talk.collect_dcr_by_file(
#     path,
#     daughterboard_number="B7d",
#     motherboard_number="#28",
#     firmware_version="2212s",
#     timestamps=500,
# )

cross_talk.plot_dcr_histogram_and_stability(path)

###
os.chdir(path)
files = glob("*.dat")
ctimes = []
mtimes = []
for file in files:
    mod_ctime = os.path.getmtime(file)
    mod_mtime = os.path.getmtime(file)

    formatted_ctime = datetime.fromtimestamp(mod_ctime).strftime("%H:%M:%S")
    formatted_mtime = datetime.fromtimestamp(mod_mtime).strftime("%H:%M:%S")

    ctimes.append(formatted_ctime)
    mtimes.append(formatted_mtime)


os.chdir(r"D:\LinoSPAD2\Data\B7d\DCR_CT\dcr_data")
file = r"0000009196-0000010912_dcr_data.pkl"

with open(file, "rb") as f:
    data = pickle.load(f)

DCR = {}
for i, row in enumerate(data):
    # DCR[f"{ctimes[i]}"] = round(np.median(row))
    DCR[f"{ctimes[i]}"] = np.median(row)


df_DCR = pd.DataFrame(list(DCR.items()), columns=["Time", "Value"])
df_DCR.columns = ["Time", "Median_DCR"]

df_DCR["Time_dt"] = pd.to_datetime(df_DCR["Time"])

# # #
txt_file = os.path.join(r"D:\LinoSPAD2\Data\B7d\DCR_CT", r"2024-09-26 Storing.txt")

df = pd.read_csv(txt_file, sep="\t", encoding="ANSI", skiprows=1)

df.columns = ["Date", "Time", "Temperature_C"]

df["Temperature_value"] = (
    df["Temperature_C"].str.extract(r"(\d+\.\d+)").astype(float)
).dropna()

df_filtered = df[~pd.isna(df["Temperature_value"])]

df_filtered["Time_dt"] = pd.to_datetime(df_filtered["Time"])

temp_to_add = []

for i in range(len(df_DCR)):
    index = np.abs(df_DCR["Time_dt"][i] - df_filtered["Time_dt"]).argmin()
    temp_to_add.append(df_filtered["Temperature_value"].iloc[index])

df_DCR["Temperature"] = temp_to_add

colors = ["#7f2704", "#a34704", "#d55d0e", "#e6801a", "#f29e50", "#f7c193"]

plt.rcParams.update({"font.size": 27})
fig, ax = plt.subplots(figsize=(16, 10))
for i in range(6):
    ax.plot(
        df_DCR["Time"][i * 280 : (i + 1) * 280],
        df_DCR["Median_DCR"][i * 280 : (i + 1) * 280],
        "o",
        color=colors[i],
    )
ax.plot(
    df_DCR["Time"][6 * 280 :],
    df_DCR["Median_DCR"][6 * 280 :],
    "o",
    color=colors[-1],
)
ax2 = ax.twinx()
ax2.plot(df_DCR["Time"], df_DCR["Temperature"], color="#0e87d8", linewidth=2)
ax.set_xticks(df_DCR["Time"][::80], [f"{x}" for x in df_DCR["Time"][::80]], rotation=50)
ax.set_xlabel("Time (H:M:S)")
ax.set_ylabel("Median DCR (cps/pixel)")
ax2.set_ylabel("Temperature ($^\circ$C)")

# plt.figure(figsize=(16,10))
# plt.plot(a, b, 'o', color='rebeccapurple')
# plt.plot(a, line(a))

# Using seaborn
import seaborn as sns

palette = sns.color_palette("flare", as_cmap=True).reversed()

fig, ax = plt.subplots(figsize=(16, 10))
sns.scatterplot(
    x=df_DCR["Time"],
    y=df_DCR["Median_DCR"],
    hue=df_DCR["Temperature"],
    palette=palette,
)
plt.xticks(df_DCR["Time"][::80], [f"{x}" for x in df_DCR["Time"][::80]], rotation=50)
plt.xlabel("Time (H:M:S)")
plt.ylabel("Median DCR (cps/pixel)")

# ax2 = ax.twinx()
# ax2.plot(df_DCR["Time"], df_DCR["Temperature"], color='#0e87d8', linewidth=2)
# ax2.set_xticks([], [])
# ax.set_xticks(df_DCR["Time"][::80], [f"{x}" for x in df_DCR["Time"][::80]], rotation=50)


# New and old DCR data
# os.chdir(r"D:\LinoSPAD2\Data\B7d\DCR\2024.09.24\dcr_data")
# file = r"0000008156-0000008755_dcr_data.pkl"

# with open(file, "rb") as f:
#     data = pickle.load(f)

# plt.rcParams.update({"font.size": 27})
# fig, ax = plt.subplots(figsize=(16,10))

# for i in range(6):
#     ax.plot(df_DCR["Time"][i*280:(i+1)*280], df_DCR["Median_DCR"][i*280:(i+1)*280], color=colors[i])
# ax.plot(df_DCR["Time"][6*280:], df_DCR["Median_DCR"][6*280:], color=colors[-1])
# ax2 = ax.twiny()
# ax2.plot(np.median(data, axis=1))

# ax.set_xticks(df_DCR["Time"][::80], [f"{x}" for x in df_DCR["Time"][::80]], rotation=50)
# ax.set_xlabel("Time (H:M:S)")
# ax.set_ylabel("Median DCR (cps/pixel)")
# ax2.set_ylabel("Temperature ($^\circ$C)")

# # #


temperatures = [25.7, 26.6, 27.2, 27.4, 27.6, 28.1]
colors = ["#7f2704", "#a34704", "#d55d0e", "#e6801a", "#f29e50", "#f7c193"]

plt.rcParams.update({"font.size": 27})
plt.figure(figsize=(16, 10))
for i in range(len(colors)):
    plt.plot(
        times[i * 100 : (i + 1) * 100],
        np.median(data, axis=1)[i * 100 : (i + 1) * 100],
        color=colors[i],
        label=f"{temperatures[i]} $^\circ$C",
    )
# plt.legend(loc="best")
plt.xticks(times[::30], [f"{x}" for x in times[::30]], rotation=50)
# plt.xlabel("Time (-)")
plt.ylabel("Median DCR (cps/pixel)")
plt.title(f"Median DCR: {np.median(data):.0f} cps/pixel")

med_DCR_0 = np.median(data[0:100])
rise = []
temp_rise = []

for i in range(1, 6):
    med_DCR = np.median(data[i * 100 : (i + 1) * 100])
    rise.append(med_DCR - med_DCR_0)
    # rise.append((med_DCR - med_DCR_0) / med_DCR_0 * 100)
    temp_rise.append(temperatures[i] - temperatures[0])

rise = np.array(rise)
temp_rise = np.array(temp_rise)


# Line fit
def line(x, a, b):
    return a * x + b


params, covs = curve_fit(line, xdata=temp_rise, ydata=rise, p0=[3, 6])

plt.rcParams.update({"font.size": 27})
plt.figure(figsize=(16, 10))
plt.plot(temp_rise, rise, "o", color="darkorange")
plt.plot(
    temp_rise,
    line(temp_rise, *params),
    color="rebeccapurple",
    label=f"{params[0]:0.1f}*x-{abs(params[1]):0.1f}",
)
plt.xlabel("Temperature change ($^\circ$C)")
plt.ylabel("Median DCR change (cps/pixel)")
plt.title(f"DCR change on temperature change")
plt.legend(loc="best")
plt.title(f"DCR change on temperature change")
plt.legend(loc="best")
