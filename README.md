## Data Analysis Package for LInoSpad (DAPLIS)

Package for unpacking and analyzing the binary data from the timestamping mode of the LinoSPAD2 detector.

![Tests](https://github.com/rngKomorebi/LinoSPAD2/actions/workflows/tests.yml/badge.svg)
![Documentation](https://github.com/rngKomorebi/LinoSPAD2/actions/workflows/documentation.yml/badge.svg)

## Introduction

This package was written for data analysis for LinoSPAD2, mainly for
analysis of the timestamp output. The key functions are ones for
unpacking the binary output of the detector that utilizes the numpy
Python library for quick unpacking of .dat files to matrices,
dictionaries, or data frames.

The "functions" folder holds all functions from unpacking to plotting
numerous types of graphs (pixel population, histograms of timestamp
differences, etc.)

The "params" folder holds masks (used to mask some of the noisiest
pixels) and calibration data (compensating for TDC nonlinearities and
offset) for LinoSPAD2 daughterboards.

The "archive" folder is a collection of scripts for debugging, tests,
older versions of functions, etc.

The "examples" folder contains a few jupyter notebooks with examples
on how to use the main functions, showcasing how to work with the
most important function parameters.

Full documentation, including examples and full documentation of
modules and functions, can be found [here](https://rngkomorebi.github.io/daplis/).

Some functions (mainly the plotting ones) save plots as pictures in the
.png format, creating a folder for the output in the same folder that
holds the data. Others (such as delta_t.py for collecting timestamp differences
in the given time window) save .csv or .feather files with the processed data for
easier and faster plotting.

Additionally, a standalone repo with an application for online plotting
of the sensor population can be found [here](https://github.com/rngKomorebi/LinoSPAD2-app).

## Installation and usage

A fresh, separate virtual environment is highly recommended before installing the package.
This can be done using pip, see, e.g., [this](https://packaging.python.org/en/latest/guides/installing-using-pip-and-virtual-environments/).
This can help to avoid any dependency conflicts and ensure smooth operation of the
package.

First, check if the virtualenv package is installed. To do this, one can run:
```
pip show virtualenv
```
If the package was not found, it can be installed using:
```
pip instal virtualenv
```
To create a new environment, run the following (if 'py -m' does not work,
 try 'python -m', 'python3 -m'):
```
py -m venv PATH/TO/ENVIRONMENT/ENVIRONMENT_NAME
```
To activate the environment (on Windows):
```
PATH/TO/ENVIRONMENT/ENVIRONMENT_NAME/Scripts/activate
```
and on Linux:
```
source PATH/TO/ENVIRONMENT/ENVIRONMENT_NAME/bin/activate
```

Then, package itself can be installed using pip inside the environment:
```
pip install daplis
```

Alternatively, to start using the package, one can download the whole repo. "requirements.txt"
lists all packages required for this project to run. One can create
an environment for this project either using conda or pip.
```
cd PATH/TO/GITHUB/CODES/daplis
pip install -r requirements.txt
```
or, using conda:
```
conda create --name NEW_ENVIRONMENT_NAME --file /PATH/TO/requirements.txt -c conda-forge
```
To install the package, first, switch to the created environment:
```
conda activate NEW_ENVIRONMENT_NAME
```
and run
```
pip install -e .
```
where '-e' stands for editable: any changes introduced to the package will
instantly become a part of the package and can be used without the need
of reinstalling the whole thing. After that, one can import any function 
from the daplis package:
```
from daplis.functions import sensor_plot, delta_t, fits
```

## How to contribute

This repo consists of two branches: 'main' serves as the release version
of the package, tested, proven to be functional, and ready to use, while
the 'develop' branch serves as the main hub for testing new stuff. To
contribute, the best way would be to fork the repository and use the 'develop'
branch for new introductions, submitting the results via pull requests. 
Everyone willing to contribute is kindly asked to follow the 
[PEP 8](https://peps.python.org/pep-0008/) and 
[PEP 257](https://peps.python.org/pep-0257/) conventions.

## License and contact info

This package is available under the MIT license. See LICENSE for more
information. If you'd like to contact me, the author, feel free to
write at sergei.kulkov23@gmail.com.
