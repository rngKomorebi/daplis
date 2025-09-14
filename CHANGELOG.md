# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.3.0] - 2025-09-14

Now compatible with python-3.9, new sensor_plot function.

### Added

- New function in the sensor_plot that returns rates on the y-axis.

### Changed

- Updated the mask for the B7d-#28 boards combination: new hot pixel added.


## [1.2.0] - 2025-06-14

Added offset calibration, removed deprecated functions, updated doscstrings, added error workarounds.

### Added

- Function to the 'fits' module for combining the '.feather' files for the requested pixels before doing a fit.

- Option for including the offset calibration post-factum during the fitting via the 'fits' module or plotting the coincidence histogram using the 'delta_t' module. Can be useful when timestamp differences were calculated without the offset calibration but it is available and can be utilized.

### Fixed

- Error handling for the fitting functions when no data was available for the requested pixels.

### Changed

- Commented out the offset calibration functions from the 'calibrate' module. Those were found to be not working properly and outdated.

### Removed

- Deprecated functions from the 'calc_diff' and 'delta_t' modules.

- Function for plotting sensor population from SPDC data as the usual function can be used for that as well.

- Test for plotting the sensor population plot for the SPDC data.

- Jypyter notebook with the example of offset calibration using cross-talk data. This will be reworked and will return later.

## [1.1.2] - 2025-03-16

Bug fixing, offset calibration for FW2212s.

### Added

- Example on how to do offset calibration using cross-talk data and 
an offset calibration file for the 2212s firmware version.

### Fixed

- A bug in the unpack version, where only the TDC calibration for the 2212b firmware version was applied.

### Changed

- "_mod" function versions in the delta_t and calc_diff modules are a part 
of the main roster of function, renamed to "_1v1" and can be used for 
calculating timestamp differences for diagonal pixels (1-1, 2-2, etc.).

## [1.1.1] - 2025-01-15

Updated the documentation, cleaned up the code, removed the repetitions of pieces of code in some of the functions which appeared most probably due to an incorrect merge.

### Fixed

- As a security update, changed the requirement on the version of the tqdm package.

- Code style and strings in the documentation and comments.

### Changed

- Closer window for the background relative to the signal in cross-talk calculation. The previous shift was too far away from the peak and could cause incorrect cross-talk numbers due to data at high photon rates.

### Removed

- Old functions from mp_analysis.py which became incompatible with the latest version of the package and due to numerous bottlenecks in the code operation.

## [1.1.0] - 2024-12-04

Added new examples, improved the documentation. Updated and corrected tests. Updated security with requirements on more fresh versions of some of the packages.

### Added

- More examples on how to use the main functions of the package.

- Function for unpacking the pickled cross-talk plots. Can be used to
change the plot or extract the data and replot it completely.

### Changed

- As a security update, changes the requirement for the version of the setuptools: now it should be above 70.0.

- Updated documentation, mainly the instruction on how to install the library.

- Updated tests to also run on the Python 3.12 and 3.13.

## [1.0.0] - 2024-11-24

Creation of DAPLIS - Data Analysis Package for LinoSpad2. Plus the long-awaited merge
of the develop branch with the main one, adding more features and faster functions to the release version of the package.

## [1.0.1] - 2024-11-20

Prepared the current version of the develop branch for merging with the main one.
After that, the changes introduced to the main branch will become incompatible with
the previous version of the main branch. That would mark the second major
release of the package. However, since the proper version numbering came much later after the first release, the version will become 1.0.0.

### Added

- This changelog.

- 'pickle_figure' boolean to the collect_and_plot_timestamp_differences and unpickle_delta_t_plot in the 'delta_t' module for more control over the plots.

- Two jupyter books with examples on cross-talk and dark count rate analysis.

### Fixed

- Fixed gitignore that covered two files with the calibration data for the
B7d LinoSPAD2 daughterboard.

- Code style and format to follow the 72 elements per line for comments and 79 for code.

### Changed

- Function 'combine_feather_files' from the utils module was generalized
to combine all '.feather' files found in the given folder. The previous
version was moved to the delta_t module and renamed to '_combine_intermediate_feather_files', as it was used only inside that module.

- Updated the documentation for the package as a whole.

### Removed

- Unused jupyter books with examples.

- Examples from the documentation, since there are much more detailed jupyter books.

## [0.9.9] - 2024-11-11

### Added

- fits_examples.ipynb --- jupyter notebook with examples on how to use
the functions from the fits module, showcasing different parameters and
use cases.



### Fixed

- Bugs in cross_talk module as unused parameters and mix-up in parameter
names.

- Histogram binning in the cross_talk module to include more numbers
after the decimal point for better precision.

- Moved firmware version check up the pipeline so that it performed at
the start of the function call and before the inital data unpacking.

### Removed

- Unused "mask_NL11_all.txt".
- Unused masks in "params/masks/old".
- Test leftovers in "tests/test_data/results".
