## Data Analysis Package for LInoSpad (DAPLIS)

Package for unpacking and analyzing the binary data from the timestamping mode of the LinoSPAD2 detector.

![Tests](https://github.com/rngKomorebi/LinoSPAD2/actions/workflows/tests.yml/badge.svg)
![Documentation](https://github.com/rngKomorebi/LinoSPAD2/actions/workflows/documentation.yml/badge.svg)
![PyPI - Version](https://img.shields.io/pypi/v/daplis)
![PyPI - License](https://img.shields.io/pypi/l/daplis)

## Introduction

This package was written for data analysis for LinoSPAD2, mainly for
analysis of the timestamp output. The key functions are ones for
unpacking the binary output of the detector that utilizes the numpy
Python library for quick unpacking of .dat files to matrices,
dictionaries, or data frames.

This library greatly streamlines working with raw timestamps from 
the LinoSPAD2 camera. With this package, one can analyze dark current 
rate, cross-talk probability, plot sensor population, and analyze the
raw data to look for photon coincidences.

## Structure of the package

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
pip install virtualenv
```
To create a new environment, run the following:
```
virtualenv PATH/TO/NEW/ENVIRONMENT
```
To activate the environment (on Windows):
```
PATH/TO/NEW/ENVIRONMENT/Scripts/activate
```
and on Linux:
```
source PATH/TO/NEW/ENVIRONMENT/bin/activate
```

Then, package itself can be installed using pip inside the environment:
```
pip install daplis
```

Alternatively, to start using the package, one can download the whole repo. "requirements.txt" 
lists all packages required for this project to run. One can create 
an environment for this project either using conda or pip following the instruction 
above. Once the new environmnt is activated, run the following to install 
the required packages:
```
cd PATH/TO/GITHUB/CODES/daplis
pip install -r requirements.txt
```
Now, the package can be installed via
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

For conda users, the new environment can be installed using the 'requirements' 
text file directly:
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

For a fast introduction on how to use the package, please see the
jupyter notebooks with examples on the main functions at "daplis/examples/".

## Versioning and releases

There is no version number to bump anywhere in the source tree: `setuptools_scm`
derives it from the git tag, so **creating the tag is the version bump**.
`daplis.__version__` reads it back from the installed distribution.

The changelog is edited by hand, and it has to be edited *before* the tag
exists - the workflow reads `CHANGELOG.md` as it was at the tagged commit, so
notes still sitting under `[Unreleased]` cannot ship.

1. In `CHANGELOG.md`, rename the `## [Unreleased]` heading, leaving an empty
   `[Unreleased]` above it for the next cycle:

   ```markdown
   ## [Unreleased]

   ## [1.5.0] - 2026-09-01
   ```

   and update the link definitions at the foot of the file:

   ```markdown
   [Unreleased]: https://github.com/rngKomorebi/daplis/compare/v1.5.0...HEAD
   [1.5.0]: https://github.com/rngKomorebi/daplis/compare/v1.4.5...v1.5.0
   ```

   `python tools/changelog.py 1.5.0` prints exactly what the release notes will
   say, and exits non-zero if the section is missing - run it before tagging.

2. Merge that into `main`.
3. On GitHub, *Releases -> Draft a new release*, create the tag `v1.5.0`
   **there** - it must point at the commit from step 2 - and publish.

Publishing runs `publish.yml`, which validates the tag, refuses to ship a
version with no changelog entry, lints and runs the tests, checks the built
version matches the tag, uploads to PyPI, and rewrites the release body from
the changelog.

If the changelog entry is missing, the run fails at the first job, so nothing
is built and nothing reaches PyPI - no version number is lost. Fix the
changelog, push, then **delete the tag as well as the release** before
retrying: deleting a release leaves its tag behind, and re-creating one with
the same name silently reuses that tag and its old commit. PyPI versions
themselves cannot be reused, so a release that *does* upload is final.

After a release, merge `main` back into `develop`. Tags are only visible to
`setuptools_scm` from commits that descend from them, so a `develop` that has
never seen the release commit keeps deriving versions from an older tag - a
local build would report a *lower* version than the one already on PyPI.

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
