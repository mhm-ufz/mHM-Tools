# Changelog

## [v0.2.1]

### Fixed

- Calculate hydrograph KGE/NSE from the cropped overlapping discharge period instead of stale pre-crop arrays.
- Keep hydrograph objective and catchment state per `Hydrograph` instance to avoid stale metrics leaking between runs.
- Limit mHM restart tile-mask discovery to the restart output folder and owning tile folder to avoid unrelated parent masks affecting merges.
- Apply gridded ESP and SPAEF-like metrics per timestep before averaging, instead of flattening the full time-space array first.
- Handle single-point temporal overlaps in xarray utilities and improve the related crop error logging.
- Fix `create_header()` output handling for explicit file paths, missing parent directories, and existing directories with dots in the name.
- Prevent file output helpers from replacing existing files when the requested output path has no file suffix.
- Handle dotted gauge output directories in catchment creation.
- Use lon/lat box resolution as fallback for L0 resolution in `create-catchment`.
- Added ("longitude", "latitude") to possible xy coordinates in discharge file

### Changed

- Write catchment gauge-correction `score`, `shape_error`, and `method` columns to the gauge info CSV.
- Refactor NetCDF writing into `write_xarray_to_netcdf()` and shared helpers in `mhm_tools.common.netcdf`.
- Update install instructions in the README.

### Tests

- Add catchment gauge info CSV coverage for correction score, shape error, and method metadata.
- Add regression coverage for hydrograph KGE after cropping.
- Add `create_header()` path handling coverage, including CLI `--only-header` output to an explicit file.
- Add and update NetCDF encoding tests for the refactored NetCDF helper functions.
- Update spatial metric tests for corrected ESP/SPAEF output names and timestep-wise behavior.
- Add xarray overlap regression coverage.

## [v0.2]

### Added

- Initial official release for mHM 5 pre-processing, post-processing, and evaluation workflows.
- Add `create-catchment` to delineate basins from DEM or flow-direction data, correct gauge outlets by area or shape, and write mHM/mRM basin, mask, idgauges, and gauge metadata outputs.
- Add `crop-mhm-setup` to crop existing mHM setups to domains, masks, or bounding boxes while preserving required grid files and headers.
- Add `create-header` and `latlon` tools to generate mHM-compatible ASCII headers and lon/lat NetCDF grids from setup extents and resolutions.
- Add `prepare-mhm-forcings` and `calculate-pet` to prepare meteorological forcings, normalize units, handle temporal frequency, and derive PET fields.
- Add `create-mhm-restart-file` and `create-mhm-restart-from-setup` to build restart files from target grids or tiled setup runs, including masking and merge support.
- Add `create-subdomain-masks`, `create-idgauges`, and region/catchment masking helpers for domain partitioning and routing setup preparation.
- Add data-processing tools for file conversion, merging many files, regridding, filling missing values, calculating long-term means, ratios, differences, and relative differences.
- Add `discharge-evaluation` to compare observed and simulated discharge, match gauges, calculate metrics, and create CDF and map outputs.
- Add `hydrograph` to read discharge series, calculate objective metrics, and create hydrograph, seasonality, scatter, and flow-duration plots.
- Add `gridded-data-evaluation` for spatial and temporal comparison of gridded model outputs with metrics such as ESP, SPAEF, MSPAEF, and WASPAEF.
- Add `mhm-run-overview`, `2d-map`, and `taylor-diagram` tools for run summaries and visualization.
- Add utility commands such as `link-folder-tree` and initial mHM 5 to mHM 6 land-cover ASCII-to-NetCDF conversion support.

### Changed

- Switch to the Click-based CLI with grouped commands, aliases, typo suggestions, and optional Trogon support.
- Improve NetCDF metadata, coordinate handling, mask handling, and output provenance across generated files.

### Fixed

- Stabilize catchment shape and area correction, gridded evaluation masking, restart creation from setup tiles, hydrograph reading, and header generation.