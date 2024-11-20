import os
from glob import glob

from matplotlib import pyplot as plt
from PIL import Image

from daplis.functions import sensor_plot

path = r"D:\LinoSPAD2\Data\board_NL11\Prague\MZI\05.08.24"
os.chdir(path)

files_all = glob("*.dat")[:100]

plt.rcParams.update({"font.size": 25})
fig = plt.figure(figsize=(12, 8))
plt.xlabel("# of file [-]")
plt.ylabel("# of photons [-]")
plt.title("MZI, pixels 135 and 174")

for i, files in enumerate(files_all):
    timestamps = sensor_plot.collect_data_and_apply_mask(
        files,
        daughterboard_number="NL11",
        motherboard_number="#33",
        firmware_version="2212s",
        timestamps=40,
        include_offset=False,
    )

    # just save the first without plotting
    if i == 0:
        tmsp_1 = timestamps[135]
        tmsp_2 = timestamps[174]
    # plot previous+current for adding lines between points, then save
    # current for the next iteration
    else:
        plt.plot([i - 1, i], [tmsp_1, timestamps[135]], "o-", color="salmon")
        plt.plot([i - 1, i], [tmsp_2, timestamps[174]], "o-", color="teal")
        tmsp_1 = timestamps[135]
        tmsp_2 = timestamps[174]
    try:
        os.chdir("results/MZI")
    except FileNotFoundError:
        os.makedirs("results/MZI")
        os.chdir("results/MZI")
    plt.savefig(f"{i}.png")
    os.chdir("../..")

os.chdir(os.path.join(path, "results/MZI"))

images = []

# collect .pngs and save as gif
for file in sorted(glob("*.png"), key=os.path.getctime):
    im = Image.open(file)
    images.append(im)

images[0].save("gif.gif", save_all=True, append_images=images[1:], duration=500, loop=0)
