import os
import shutil
from glob import glob

import numpy as np

from daplis.functions import delta_t, fits

# path = r"D:\LinoSPAD2\Data\D2b_board\28.06.2024\second_try"

path = r"D:\LinoSPAD2\Data\board_NL11\Prague\14.05"

os.chdir(path)
files = glob("*.dat")

# Step or number of files to group

step = 20

# Group the files by two
if len(files) % step == 0:
    num_of_folders = len(files) // step
else:
    num_of_folders = len(files) // step + 1

new_folders = []

# Create new folders
for i in range(num_of_folders):
    new_folder = f"{step*i}-{(step+1)*i}"
    try:
        os.mkdir(new_folder)
    except FileExistsError as _:
        pass
    new_folders.append(f"{new_folder}")

# Copy the data files to the new folders
for i, file in enumerate(files):
    j = i // step
    destination = os.path.join(path, new_folders[j], file)
    shutil.copyfile(file, destination)


###

# Collect the timestamp differences in eachf newly generated folder

os.chdir(path)

for k, folder in enumerate(new_folders):

    delta_t.calculate_and_save_timestamp_differences(
        folder,
        pixels=[12, 58],
        rewrite=True,
        daughterboard_number="D2b",
        motherboard_number="#4",
        firmware_version="2212s",
        timestamps=300,
        include_offset=False,
        correct_pix_address=True,
    )

    os.chdir("..")

###

# Go in steps of 50, plot the delta t plot with fit

import pandas as pd
from pyarrow import feather as ft

os.chdir(path)

try:
    os.mkdir("combinations")
except FileExistsError as _:
    pass

destination_to_combinations = os.path.join(path, "combinations")

for k, folder in enumerate(new_folders):

    try:
        os.remove(f"deltas_combined_{0}-{k}")
    except FileNotFoundError as _:
        pass

    path_to_delta = os.path.join(path, f"{folder}\delta_ts_data")

    os.chdir(path_to_delta)

    ft_file = glob("*.feather")[0]

    shutil.copy(ft_file, destination_to_combinations)


os.chdir(destination_to_combinations)

delta_files = glob("*.feather")

for k, _ in enumerate(delta_files):

    for j in range(k):

        try:
            ft_file_previous = glob(f"deltas_combined_{0}-{k}.feather")[0]
            data_previous = ft.read_feather(ft_file_previous).dropna()
            data_next = ft.read_feather(delta_files[j]).dropna()
            data_combined = pd.concat([data_previous, data_next])
        except (FileNotFoundError, IndexError) as _:
            data_next = ft.read_feather(delta_files[j]).dropna()
            data_combined = data_next

        ft.write_feather(data_combined, f"deltas_combined_{0}-{k}.feather")


###

combined_feathers = glob("*combined*")
combined_feathers_sorted = combined_feathers
combined_feathers_sorted.sort(key=lambda x: os.path.getctime(x))

a = []
for i, file in enumerate(combined_feathers):

    # delta_t.collect_and_plot_timestamp_differences(
    #     path=destination_to_combinations,
    #     pixels=[12, 58],
    #     rewrite=True,
    #     ft_file=file,
    #     step=10,
    #     range_left=-20e3,
    #     range_right=20e3,
    #     correct_pix_address=True,
    # )

    a.append(
        fits.fit_with_gaussian(
            path=destination_to_combinations,
            pix_pair=[153, 170],
            ft_file=file,
            window=20e3,
            step=10,
            # correct_pix_address=True,
            return_fit_params=True,
        )
    )

a = np.array(a)

a_clean = np.copy(a)
a_clean[np.where(a[:, 1, 2] > 100)[0]] = None

from matplotlib import pyplot as plt

from daplis.functions import utils

plt.figure(figsize=(12, 8))
for i in range(len(a_clean)):
    plt.errorbar(i, a_clean[i][0][2], yerr=a_clean[i][1][2], fmt="o")
plt.xticks([x for x in range(29)], [x * 30 for x in range(1, 30)], rotation=65)
plt.xlabel("Number of files [-]")
plt.ylabel("$\sigma$ [ps]")
plt.title(f"Median $\sigma$ for last 10 is {np.median(a[-10:,0,2]):.2f} ps")


contrasts = []

plt.figure(figsize=(12, 8))
for i in range(len(a)):
    contrast = a_clean[i][0][0] / a_clean[i][0][3] * 100
    contrasts.append(contrast)
    vis_er = utils.error_propagation_division(
        a_clean[i][0][0], a_clean[i][1][0], a_clean[i][0][3], a_clean[i][1][3]
    )

    # Contrast error in %
    vis_er = vis_er / (contrast / 100) * 100
    plt.errorbar(i, contrast, yerr=vis_er, fmt="o")

plt.xticks([x for x in range(29)], [x * 30 for x in range(1, 30)], rotation=65)
plt.xlabel("Number of files [-]")
plt.ylabel("Contrast [%]")
plt.title(f"Median contrast of last 10 is {np.median(contrasts[-10:]):.2f}")
