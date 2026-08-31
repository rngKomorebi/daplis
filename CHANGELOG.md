# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).


## [1.5.0] - 2026-08-31

### Added

- `scan_window` in `calculate_and_save_timestamp_differences_full_sensor_alt`
  (default 1e6, i.e. +-1 us): half-width of the window used while locating the
  coarse board-to-board offset. It is separate from `delta_window`, which now
  only sets how much is kept around the located offset.

- The located cross-board alignment is reported back. The function returns a
  dictionary with `feather` (path to the saved file), `cycle_lag_N`,
  `delta_epoch_ps`, `delta_epoch_s` and `pair_counts` (differences saved per
  pixel pair), and prints the lag it settled on together with the coincidence
  count and its significance.

- Guards that fail loudly instead of quietly saving an empty table:
  `RuntimeError` if the absolute-timestamp counters are not monotonic across
  the run (a board power-cycled mid-run), if the requested pixels hold no
  photons, or if no coincidence peak reaches 5 sigma above the baseline of the
  neighbouring lags. A warning is printed when the sub-cycle phase between the
  boards is not constant.

- `calculate_differences_1v1` and
  `calculate_and_save_timestamp_differences_1v1` raise `TypeError` when given a
  flat list of pixels. A flat list means "all combinations" and cannot be split
  into explicit 1-to-1 pairs, which used to fail further in with a much less
  obvious message.

### Changed

-`calculate_and_save_timestamp_differences_full_sensor_alt` is
  rewritten around a single global board-to-board offset.** The two boards are
  taken to share an external clock but no trigger (CLK_IN/J11 only), so each
  FPGA's free-running 133.333 MHz absolute-timestamp counter (7.5 ns per tick)
  starts at its own power-up moment and the two are related by one constant,
  Delta_epoch. Every requested pixel's photons are pooled onto each board's
  continuous timeline, the integer cycle lag that maximizes coincidences within
  `scan_window` fixes Delta_epoch, and all differences within `delta_window` of
  it are then kept per pixel pair. This replaces the old per-cycle scheme,
  which looked for the first cycle above `threshold` on each board, paired
  cycles by index from there, corrected each cycle from its own absolute
  timestamps, and estimated the epoch offset from a histogram of whichever
  pixel pair happened to have data first.

- `threshold`, `apply_mask` and `epoch_offset_ps` are gone and `scan_window` is
  new (see Removed), and it returns the alignment dictionary described above
  instead of the epoch offset as a float. Only firmware versions "2212s" and
  "2212b" are offered in the error message now; "2208" was never supported
  here.

- `pixels` accepts two lists of any length, `[[left, ...], [right, ...]]`,
  rather than one pixel per sensor half; a plain `[left, right]` still works.
  Right-half pixels are given as full-sensor indices (256..511) and remapped to
  raw pixels on the second board by the new `_remap_full_sensor_pixel` helper.

- Each data file is unpacked once instead of once per pixel pair, and the
  differences are gathered with `numpy.searchsorted` in blocks instead of a
  Python loop over cycles and timestamps.

- Data files are collected without `os.chdir` and sorted by name instead of
  modification time. File names are timestamped, so this matches acquisition
  order and agrees with the '.feather' name the fit functions reconstruct.
  Results go into one '.feather' file written at the end, instead of numbered
  per-file chunks concatenated and deleted afterwards.

- `collect_and_plot_timestamp_differences` no longer forces `figsize=(16, 10)`
  on the single-pair figure, so the house style's figure size applies there
  too.

- **The package version now comes from the git tag**, via `setuptools_scm`.
  `pyproject.toml` no longer carries a `version = "..."` line, so creating the
  release tag *is* the version bump. `daplis.__version__` reads it back from
  the installed distribution, and the Sphinx docs take their `release` from the
  same place instead of a hardcoded string.

- **Releases are cut by publishing a GitHub Release**, not by pushing to
  `main`. The auto-tagging `release.yml` is gone; `publish.yml` now gates the
  whole pipeline on the tag having a matching `CHANGELOG.md` section, lints and
  tests before building, verifies the built version matches the tag, and fills
  the release body from the changelog. See "Versioning and releases" in the
  README.

- Ruff replaces ad-hoc style checking as the lint gate: `[tool.ruff]` in
  `pyproject.toml` configures it (pycodestyle, pyflakes, isort, pyupgrade,
  bugbear and numpy-convention pydocstyle), and CI runs `ruff check .` on every
  push as well as before a release.

- The bundled `daplis.mplstyle` now lives in
  [komorebi_mpl](https://github.com/rngKomorebi/komorebi_mpl) as the registered
  `daplis` style, so it shares one source of truth with the other house styles.
  `komorebi_mpl>=0.0.5` is now a dependency.

- The style is applied on import via `komorebi_mpl.apply_default("daplis")`
  instead of an unconditional `plt.style.use`. An explicit
  `komorebi_mpl.use(...)` in your own script now wins, whatever the import
  order. `daplis.style_path` still works and points at the bundled sheet.

- Following that move, daplis plots pick up the house treatment: outward-facing
  ticks with minor ticks, a grid, and the shared typography scale. The
  colorblind-accessible eight-colour cycle is unchanged.

### Fixed

- `calculate_differences` dropped every pixel pair whose right-hand index was
  not greater than the left-hand one (`if w <= q: continue`). A full-sensor
  call whose right list holds lower indices than its left therefore returned
  nothing at all for those pairs. Pairs are now skipped only when both pixels
  are the same, each pair is emitted once under a canonical key - `"a,b"`
  always means `t_b - t_a` with `a < b` - and pairs repeated by overlapping
  input lists are de-duplicated.

### Removed

- `threshold` and `apply_mask` from
  `calculate_and_save_timestamp_differences_full_sensor_alt`. The global
  alignment needs no threshold to pick a reference cycle, and `apply_mask` was
  accepted and documented but never used - it only ever sat next to a "TODO add
  check for masked/noisy pixels".

- `epoch_offset_ps` from the same function. The offset is always measured from
  the data now, so there is nothing to pass in or to carry over from a previous
  run; the value that was used comes back in the returned dictionary instead.

## [1.4.5] - 2026-07-10

Working on full sensor function, trying to synchronize the two boards.

### Changed

- How the absolute timestamps are handled, using them for calibrating the offset between the two LinoSPAD2 motherboards.

- Fixing how full_sensor in unpack, sensor_plot, delta_t, fits are working.

## [1.4.4] - 2026-06-03

New tests, fixing tests, cleaning up, fixing 1v1 functions

### Added

- New functions in fits module, fit_with_gaussian_lmfit_with_stats. Fits the histogram of coincidences with a Gaussian function using lmfit but also returns, chi2red, residuals for checking fit quality.

- New tests for delta_t, sensor_plot, cross_talk modules, covering the functions checks for which were missing, updating old ones based on the new functionality.

### Changed

- In the unpack module, removed the old version of the unpacking function, rewrote the unpack_binary_data_with_absolute_timestamps function using the same new logic as for the normal function.

- the plot_sensor_population_full_sensor function in the sensor_plot module now saves two figures: one with the number of photons, one with the photon rate. Added functionality for looking for peaks above threshold.

- How 1v1 (calc_diff, delta_t) functions operate: they now use the same logic as normal functions, also updated so that they work with new unpack/calibrate routine.

- fit_with_gaussian function to handle unsuccessful fits.

- fit_with_gaussian_combine now works more reliably and saves the plot in a cleaner format.

- Updated the offset calibration for B7d, #28, 2212b combination.

## [1.4.3] - 2026-04-01

Fixed tests, dropped python-3.8 support

## [1.4.2] - 2026-03-27

More tests, flexible mpl style

### Added

- More tests

- daplis mpl style - keeps the design as before but without hardcoded colors and styles

## [1.4.1] - 2026-03-26

New example, functionality; cleaning up

### Added

- New jupyter notebook with SNR analysis estimating time to expected SNR for HBT peak; plus how long to collect to see HBT

- Plotting both rates and # of photons by default for plot_sensor_population

## [1.4.0] - 2025-12-01

Faster functions, cleaner implementation.

### Added

- New function in calc_diff with a new algorithm for coincidence calculation that works on a moving window instead of merging two lists, sorting, and taking a difference between neighbors; the new one is much faster. The old one is commented out for now, plan to remove.

- New function in delta_t that works with the new algorithm. The old one is commented out for now, plan to remove.

- A new function to the archive/tools that normalizes HBT coincidence histogram to median of the whole histogram and separately to the average background and compares the two.

### Changed

- The unpacking function which does not apply calibration now at all and does not merge the pixel and timestamp data into a single matrix - faster, cleaner, easier to work with. Calibration is then applied where needed: cross_talk function, delta_t, etc. sensor_plot function do not require calibration so it's an added overhead when plotting sensor population plots.

- Docstrings, some parameters' names for consistency and readability.

- Tests that work with the new function.

## [1.3.1] - 2025-10-20

Better plots, more control over fitting functions.

### Added

- Normalization to each fitting function.

### Changed

- Font size for all single plots (so except for the delta_t plot which can be a grid of plots) from 27 to 30 for better readability.

- The 'window' parameter to 'range_left' and 'range_right' to all fitting functions for more control over the window where the fit is done.

- Cleaned up the 'fancy' lmfit fitting function for better readability and contrast without normalization which makes sense.

### Removed

- Redundant parameters from some of the plotting functions, like 'show_fig' which sometimes brakes how the plots are shown after running the plotting functions.


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

[Unreleased]: https://github.com/rngKomorebi/daplis/compare/v1.5.0...HEAD
[1.5.0]: https://github.com/rngKomorebi/daplis/compare/v1.4.5...v1.5.0
[1.4.5]: https://github.com/rngKomorebi/daplis/compare/v1.4.4...v1.4.5
[1.4.4]: https://github.com/rngKomorebi/daplis/compare/v1.4.3...v1.4.4
[1.4.3]: https://github.com/rngKomorebi/daplis/compare/v1.4.2...v1.4.3
[1.4.2]: https://github.com/rngKomorebi/daplis/compare/v1.4.1...v1.4.2
[1.4.1]: https://github.com/rngKomorebi/daplis/compare/v1.4.0...v1.4.1
[1.4.0]: https://github.com/rngKomorebi/daplis/compare/v1.3.1...v1.4.0
[1.3.1]: https://github.com/rngKomorebi/daplis/compare/v1.3.0...v1.3.1
[1.3.0]: https://github.com/rngKomorebi/daplis/compare/v1.2.0...v1.3.0
[1.2.0]: https://github.com/rngKomorebi/daplis/compare/v1.1.2...v1.2.0
[1.1.2]: https://github.com/rngKomorebi/daplis/compare/v1.1.1...v1.1.2
[1.1.1]: https://github.com/rngKomorebi/daplis/compare/v1.0.1...v1.1.1
[1.0.1]: https://github.com/rngKomorebi/daplis/compare/v0.9.0...v1.0.1
[0.9.0]: https://github.com/rngKomorebi/daplis/releases/tag/v0.9.0
