# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.0.0] - 2024-11-11

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
