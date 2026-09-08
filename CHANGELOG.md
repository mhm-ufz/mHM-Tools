# Changelog

## [Unreleased]

### Added

#### New tools
- `evaluation twsa-evaluation` compares a simulated total water storage anomaly against a gridded reference. It accepts an mHM fluxes-and-states record, a total water storage or a ready anomaly, brings both onto the coarser grid and onto the reference's own observation windows rather than calendar months, scores every cell with the metrics `--metrics` selects (`all` by default, `none` or a comma separated list of `kge` and `rmse`), and writes the metrics, maps, anomaly fields and one time series plot per `WMO_REGION_BOUNDS` region. A metric that is not selected is neither calculated, written nor plotted, and without the KGE the per region maps are dropped as well while the cell series plots stay. `--metrics none` together with `--no-write-twsa` is refused before anything is read, since such a run would compute nothing. Windows come from CF time bounds where present and from evenly spaced stamps otherwise; a gap, a missing unit, a reference coarser than the input or a record not covering the baseline is refused rather than guessed. Storage levels are added back before scoring, because a KGE bias over two near-zero anomaly means explodes; where one record arrived as an anomaly its level is unknown, both get the same offset, and a warning states that the bias no longer measures a storage bias.
- `legacy-tools snow-evaluation` compares a simulated snow field against a gridded snow reference. It brings both onto the coarser grid and calendar (never finer than 8 day composites), reduces them to a binary snow flag, and writes per snow year and cell the first and last snow covered day, the season span and the number of snow covered days, plus classification accuracy maps, the snow covered share over time per hemisphere, and an animated three panel comparison - each repeated per `WMO_REGION_BOUNDS` region holding data. It streams throughout: records are probed for their coordinates instead of opened, read one file at a time, and reduced in blocks sized against `--max-memory-gib`, and a step counts as snow covered when any step it was aggregated from exceeded the threshold, so peak memory follows the compared grid rather than the record (0.51 instead of 2.26 GiB over 500 daily files).
- `create-wmo-region-masks` delineates (or reuses) global basins, rasterizes the 6 WMO regions from a shapefile set and a GeoJSON, and maps every basin to exactly one region through a basin-adjacency graph seeded from anchor basins, which keeps regions contiguous and stops endorheic basins from forming islands of the wrong region as the per-basin majority vote of `legacy create-subdomain-masks` could. Greenland is excluded. It writes one mask per region plus a 4-panel overview plot.
- `discharge-eval-comparison` compares two or more `discharge-evaluation` `results.csv` files with CDF, violin, catchment-map and a new per-gauge `map-diff` plot, each broken down per GRDC region derived from the gauge id and collected into one overview PDF. Values that cannot occur by construction (`kge`/`nse` above 1) are discarded before plotting so one diverged run cannot distort an axis or a KDE.
- `data-converter format-data` prepares every categorical mHM input - soil, geology, LAI and land cover - onto the exact DEM grid from one raster or a CSV/TXT manifest, writing `nc` (default), `asc` or `tif` plus the `*_classdefinition.txt` files mHM expects. Categories are reclassified at the source resolution before the grid warp, so no interpolation invents a class. Multi-horizon soil quantizes raster windows into profiles (v5 profile classes for `asc`, v6 per-horizon classes for `nc`) and accepts six bulk density units; gridded LAI aggregates in time to `daily`, `monthly`, `annual` or `long-term-mean-monthly` and streams one row block and time step at a time; land cover validates that the manifest periods are gapless and non-overlapping and writes one file per period or a single CF time stack. Every output is restricted to the DEM domain, with nodata inside it filled from the nearest classified neighbour per input layer unless `--no-fill-nodata` is given, and a run that cannot fit on the output volume is refused up front. Peak memory stays flat as the grid or the record grows.
- `data-converter rasterize-map` burns a vector attribute onto the exact grid of a reference DEM and writes ASCII, NetCDF or GeoTIFF, reading the burn values from the vector or through a lookup table for textual categories, and refusing to overwrite one of its own inputs.
- `setup-creation create-dem-derivatives` derives the filled DEM, slope, aspect, flow accumulation and flow direction in one pass, into one `nc` file or one file per derivative for `asc`/`tif`. Aspect, which `pyflwdir` does not provide, uses Horn's 3x3 gradient as a compass bearing, with flat cells written as nodata.

#### Existing tools
- `create-catchment` gains `--max-distance-m` to select outlet candidates within a radial distance in meters, mutually exclusive with the existing cell limit and its five-cell default, and `--no-max-error` to rank candidates without disqualifying any of them.
- `--ncpus` sizes a local dask cluster in `twsa-evaluation` and spreads files over cores in `snow-evaluation` (0 for every core the machine reports); one core skips the cluster. Chunks are sized against `--max-memory-gib`, so a large budget yields fewer, larger chunks and less parallelism.

#### Shared modules
- `mhm_tools.common.crs_handler` collects the CRS handling: `resolve_crs` uses a supplied CRS only when the object carries none and raises when the two disagree, `set_spatial_dims` rejects axes that are not one-dimensional, finite and regularly spaced, and `write_object_crs` attaches a CRS to a Dataset or DataArray.
- `mhm_tools.common.lookup_handler` reads lookup tables and CSV manifests, matching column names case-, punctuation- and unit-suffix-insensitively and validating every value with the offending row number.
- `mhm_tools.common.parallel` provides `create_dask_cluster` and `resolve_n_cpus`.
- `mhm_tools.common.metrics.kge` holds the Kling-Gupta efficiency as `calculate_kling_gupta_efficiency` for a pair of series and `calculate_kling_gupta_efficiency_per_cell` for a whole grid.
- `get_file_res` takes a dataset through `ds` and reports a missing resolution through `raise_exception` instead of always raising.
- `convert_meters_to_degrees` in `common.utils` and `EARTH_RADIUS_M`, `EARTH_RADIUS_KM`, `METERS_PER_DEGREE` and `MIN_COS_LATITUDE` in `common.constants` put every distance on one sphere, replacing three separate literals. The conversion is latitude aware, since a meridional degree covers only 70 % of the needed span in longitude at 60 degrees north.
- Helpers promoted into `common` for reuse across tools: the grid, time and region helpers of `snow-evaluation` (`common.xarray_utils`, `common.time_utils`, `common.utils`, `common.plotter`), `write_mask_to_file` and `create_mask_from_polygon` for the region masks, `normalize_cli_sequence` for the CLI, and the WMO region tables and `GREENLAND_COORDS` into `common.constants`.
- `read_dataset`, `get_dataset_from_path` and `get_xarray_ds_from_file` take one variable name or a sequence of them and restrict what is read instead of only sizing the chunks. An mHM fluxes-and-states file holds 28 variables of which an evaluation needs one, so a multi-year record moved roughly 25 times more data than the tool used.
- `resample_to_target_freq` gains `resample_origin`, `time_anchor` and `drop_partial_edges`, all defaulting to the previous behavior, so records starting on different days share bins, a label describes the steps it aggregates, and an edge step only one record covers can be dropped. A multi-day step stays anchored to its origin under both pandas 2 and 3, which stopped treating `Day` as Tick-like and silently ignored `origin`.

### Changed
- `Hydrograph.calc_kling_gupta_efficiency` delegates to `common.metrics.kge` instead of holding its own copy, so a discharge KGE and a gridded KGE are the same number by construction. The shared version filters to the finite pairs before correlating, which the method relied on its caller to do, and returns NaN instead of an infinity for a constant or zero-mean series.
- SPAEF and TSM share one paired NaN filter, and the temporal correlation of TSM is a blocked Spearman computed in `common.xarray_utils`.
- KGE plots and catchment maps end at `KGE_CONSTANT_MEAN_BOUND` (-0.41), the score of a constant mean predictor, and paint everything below it in a distinct under-colour instead of stretching the scale to an outlier.
- The coordinate resolution is derived from the average step with a median fallback for irregular grids, through `calculate_coordinate_resolution` everywhere it was derived inline.
- `gridded-data-evaluation` holds its streamed monthly series in a quarter of the memory: `float32` sums with `uint16` counts, subsets and buckets popped as they merge, and one preallocated field for the period mean. `monthly_counts` stays `uint32`, which cannot wrap on a long record.
- `create-catchment` crops the flow direction data to the reference shape's bounds plus a buffer scaling with the shape - `SHAPE_BUFFER_EXTENT_FACTOR` times its larger half extent, never less than the outlet search distance - instead of a fixed 1 degree, because a shape disagreeing with the flow network is off by a fraction of the catchment it describes. For a 194 km2 Alpine catchment the working domain drops from 1264x1304 to 241x282 L0 cells and the delineation from 4 min 18 s to 5 s. The written extent is unaffected: it still comes from the delineated mask, `--frame` and the L2 alignment.
- `create-catchment` scores outlet candidates by shape overlap alone (`1 - IoU`), since the IoU can never exceed the ratio of the smaller to the larger area and therefore already carries the area discrepancy, and restricts them to cells draining within `CANDIDATE_AREA_AGREEMENT_FACTOR` of the reference area, which drops candidates whose IoU could never reach that factor. Without a reference shape the area error still ranks them. `--no-max-error` means the same in both searches: the criterion only ranks, and the best candidate is always taken.
- `create-catchment` reports every unknown number in its gauge outputs as `nan` instead of mixing `-9999`, `None` and empty cells, names the matching methods `shape_iou`, `area_basinex` and `area_burek`, and documents in `write_gauges_to_csv` what each column holds. In `gauges_info.nc` the area is described as the upstream area at the delineated outlet and carries a `delineation` attribute naming the methods used, instead of attributing every area to the Burek correction.
- Outlet distances are measured coordinate-aware, with latitude-dependent distances on geographic grids, instead of the Burek implementation's square-cell approximation.
- CDF plots use a small low-opacity `+` marker instead of large high-opacity `o` markers.

### Fixed
- `norm_deviation` of the TSM metric subtracted the spatial mean before dividing by it only for the first term: `data - mean / mean` instead of `(data - mean) / mean`. Every TSM value was wrong.
- `get_std_from_ds` had its climatology test inverted, so the deseasonalising branch never ran and the standard deviation was the raw one. `std` and every `rel_std` and std panel change. A dead branch guarding on the standard library `array` module, which no value can be an instance of, is gone.
- `align_bounds_to_l2` overshot the upper bound by one L0 cell, which left the domain indivisible by the upscaling factor and fell back to a crop ignoring the meteo grid, offsetting `mask_l2` against the forcing.
- Mask upscaling works for domains narrower than a single target cell, and resolution mismatches from float32 coordinate rounding no longer occur.
- A gridded file whose coordinates are not plain dimension coordinates is read correctly again: selecting variables keeps the coordinate variables, `normalize_lat_lon` promotes a coordinate stored as a data variable and indexes it, and both read paths normalize before reading the axis order, where an unindexed coordinate always compared as ascending and left a descending grid unflipped.
- A reference on a 0 to 360 longitude axis, as GRACE ships it, is compared against a model grid again; a global axis is rotated onto -180 to 180 and re-sorted, while a regional axis crossing the antimeridian is refused rather than torn in two.
- `get_ds_extend` no longer raises `KeyError` when a coordinate names a bounds variable that a variable selection left behind.
- `--available-mem` is read as a budget in GiB and keeps a fractional value.
- The CLI builds and behaves under a stricter click: a repeatable option with a single default no longer takes the whole CLI down while building, `store_false` flags are no longer stuck on their off value, and a repeatable option splits a quoted value on whitespace as well as on commas, so `--mask-paths="a.txt b.txt"` names two files again. Label options are only split on commas.

### Tests
- Snow evaluation: the snow flag threshold, classification accuracy with missing steps, the season metrics and their snow-year window, the 8 day calendar normalization, grid aggregation, hemisphere split, region selection and an end-to-end run.
- Streamed monthly series of `gridded-data-evaluation`: per-bucket division, a calendar month split across subsets, an end-to-end check against a direct monthly resample on one and two cores, and the accumulator dtypes and subset release.
- Catchment candidate selection and CLI: meter limits, radial filtering, optional area delimiting, coordinate-aware distances, distance argument validation, and the coordinate layouts of a read dataset.

## [v0.2.3]

### Added

- Add CF-conventions `long_name` (and `units` where derivable from the source variable) to the outputs of `calc_diff`, `calc_ratio`, and `calc_rel_diff`, and to `gridded-data-evaluation`'s climatology/std/mean/time-series outputs, instead of shipping unlabeled arrays.
- Add a `--update-tile-masks` flag to `create-mhm-restart-from-setup` that rewrites each reused tile's `mask_tile.nc` from the current mask instead of keeping the existing one, so `--no-tile-creation`/`--skip-mhm-run` reruns with a changed mask no longer silently mask restart data with a stale per-tile mask.

### Fixed

- Restore variable attributes that xarray's `.where()`/arithmetic operations silently drop when masking DEM cells (`_mask_dem_with_l0_file`, `crop_file`), merging catchments (`merge_catchment`), and converting forcing units (`convert_units`), so masked/merged/unit-converted output keeps its original CF metadata.
- Fix `_fill_recreated_restart_inputs` passing meteo/input fill values to `fill_nearest`'s `default_value` parameter as `fill_value` instead, which set the wrong NetCDF `_FillValue`/`missing_value` metadata instead of the intended out-of-domain default.
- Fix `merge_files`'s final merge staging its temp directory under a stale subfolder path left over from the last-processed input folder instead of the actual output directory, breaking the "same filesystem, atomic rename" assumption and risking a cross-device link error.
- Remove intermediate per-folder merge outputs once `preserve_folders` is disabled in `merge_files`, instead of leaving redundant per-folder files and directories behind after everything has already been merged into the final output.
- Fix stale/broken symlinks being invisible to `Path.exists()`, which crashed `merge_files`, `Grid.migrate_grid_using_systemlink`, and `link_folder_tree` with `FileExistsError` when rerun after a symlink's original target had moved or been removed.
- Fix the final restart file written by `create-mhm-restart-from-setup` declaring `NaN` as its `_FillValue` instead of `-9999.0`, so mHM's Fortran reader (which expects `-9999.0`, matching the tile restart files read in) could no longer recognize missing cells in the merged output.
- Fix `gridded-data-evaluation`'s `get_file_stats` building its climatology/std/mean output with a disconnected `lat`/`lon` coordinate whenever the input file's real spatial dimension used an alias (e.g. `northing`/`y`) instead of `lat`/`lon`: the statistics stayed keyed by the alias while a separate, unrelated `lat`/`lon` coordinate was attached alongside them, so `apply_spatial_mask` silently skipped masking/cropping the actual data (only cropping the orphan coordinate) and the mismatch later crashed with `CoordinateValidationError: conflicting sizes for dimension 'lat'`.

## [v0.2.2]

### Added

- Always derive `L1_soilMoist` in the merged restart as the midpoint between `L1_wiltingPoint` and `L1_soilMoistFC` when both are available, overwriting any value a native tile restart may already carry.

### Changed

- Extend `generate_bounds`/`generate_bounds_for_all_coords` to accept an explicit resolution fallback for coordinates with only one point, where a cell width can no longer be derived by differencing.
- Add a `create_bounds` option to `get_dataset_from_path`, mirroring the existing option on `get_xarray_ds_from_file`, to attach real coordinate bounds to a dataset at open time, before any cropping.
- Move `regrid_mask` from `mhm_tools.pre.crop_mhm_setup` to `mhm_tools.common.xarray_utils`, since it's pure xarray grid-matching logic already used by both pre- and post-processing modules.
- `write_mask_file` now writes the catchment mask once (as `mask`) and adds `land_mask` as an HDF5 hard link to the same on-disk data via the new `add_variable_hard_link` helper (`mhm_tools.common.netcdf`), instead of writing the identical array twice under both names. Both variables remain fully and independently readable by any NetCDF/HDF5 tool, including all attributes, with no duplicated storage.
- `write_metric_plots` now always writes the same per-variable/per-realisation statistics table (n, min, max, mean, median) to `metric_summary.csv` via the new `write_metric_summary_csv`, independent of which `--plot-types` were requested, so the numbers are available as a machine-readable file rather than only inside the optional overview PDF.
- `write_metric_plots`'s `metric_plots_overview.pdf` is now written whenever there is at least one plot or summary row to show, instead of being skipped for the common case of a single variable and a single plot type; it now also embeds every plot type actually produced in that call (previously it could end up containing only catchment-map plots).

### Fixed

- Fix `cut_to_filled_area` producing an empty crop, and silently dropping the last filled row/column even when multiple cells are filled, due to an off-by-one in its bounding-box-to-slice conversion. A single-cell spatial mask previously crashed `gridded-data-evaluation` with `Cannot determine file resolution: no valid lon or lat coordinates provided`.
- Fix `generate_bounds_for_all_coords` writing the `bounds` attribute onto a discarded copy of the dataset instead of the one it returns, so the generated `*_bnds` coordinate existed but nothing pointing to it.
- Keep real CF coordinate bounds on `gridded-data-evaluation` statistics through cropping, including crops down to a single cell, instead of relying on resolution being re-derivable from the already-cropped coordinates afterward.
- Fix `regrid_mask` crashing with `IndexError` when a target or mask coordinate had been cropped to a single point: it now honors a caller-supplied `target_res`/`mask_res` instead of always re-deriving them from the raw coordinate arrays, and its own fallback now derives resolution via `get_file_res` (trying both lon and lat) instead of assuming `lon` has at least two points.
- Fix `get_file_res` returning a negative resolution for descending coordinates, which also silently broke matching against configured `l0`/`l1`/`l2`/`l11` resolutions for such files.
- Fix `create-mhm-restart-from-setup` merge producing an mHM-unreadable restart file: soil-horizon, land-cover-period, and LAI-timestep boundaries were looked up under invented native variable names, and a missing `return` corrupted the default six-horizon boundaries — both silently replaced real depths/periods/months with meaningless index values that mHM rejected on restart.
- Fix border-clobbering when merging restart tiles shared across multiple mask/parameter runs: a later tile's fill/NaN cells could overwrite an earlier tile's valid data at the same position.
- Mask each tile restart file in place with its own tile mask before it is moved or merged, so unmasked edge values can no longer leak into relocated or merged restart output.
- Write a tile's mask section on demand when reusing an existing tile setup (`--no-tile-creation`) whose `mask_tile.nc` predates this file, instead of requiring the whole tile to be recreated.
- crop_mhm_setup tryed to read mask regardless of whether it was provided or not
- avoid division by zero in SPEAF by returning nan whenever std of one array is zero
- Fix `resample_to_target_freq` silently dropping all data before the final overlap check: its closing `xr.align(..., join="inner")` joined on every shared dimension, so `lat`/`lon` labels that were already positionally matched by earlier cropping/regridding but still differed by float32-vs-float64 precision (e.g. `68.1500015258789` vs `68.15`) failed an exact-equality join and collapsed to near-empty, leaving `gridded-data-evaluation` unable to find any temporal overlap between input and reference.

### Removed

- Remove the unused `merge_mhm_restart_files` restart-merge pipeline and the ~15 helper functions reachable only from it (superseded in production by `merge_restart_files`), plus dead entries in the internal variable/dimension rename lookup tables that never matched real mHM restart output.

### Tests

- Add direct unit coverage for `cut_to_filled_area`: single-cell crops, inclusion of the last filled row/column, buffer clipping at both array bounds, and an upscaling edge case at the array boundary.
- Add coverage for `create_bounds` on `get_xarray_ds_from_file`/`get_dataset_from_path`, including that the coordinate's `bounds` attribute is set correctly.
- Rewrite `gridded-data-evaluation` single-point-crop regression tests to check for real coordinate bounds rather than a cached resolution attribute.
- Add `regrid_mask` coverage (moved to `tests/test_xarray_utils.py`) for resolution-fallback correctness: falling back across axes when a coordinate has a single point, honoring an explicitly provided resolution over a diffed one, and single-cell domains with a resolution supplied directly or derived from CF bounds.
- Add regression coverage for `resample_to_target_freq` confirming that lat/lon labels differing only by float32-vs-float64 precision no longer collapse the aligned output, using coordinate arrays reproducing the exact mismatch (`68.15` vs `68.1500015258789`).
- Add coverage for `write_mask_file` confirming `land_mask` and `mask` stay independently readable with identical data while verifying, via direct HDF5 object-address inspection, that `land_mask` is a genuine hard link rather than a separately stored duplicate.

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
