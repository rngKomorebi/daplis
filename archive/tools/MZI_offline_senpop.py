import os
from glob import glob

import mpl_styler
import numpy as np
from matplotlib import pyplot as plt

plt.style.use("night_wave")

path = r"D:\LinoSPAD2\Data\B7d\2025.09.24\plan AA\senpop_data"

os.chdir(path)

files = glob("*.txt")

pix_100 = []
pix_200 = []

for file in files:
    data = np.loadtxt(file)
    pix_100.append(data[50])
    pix_200.append(data[250])


plt.figure()
plt.plot(pix_100)
plt.plot(pix_200)
