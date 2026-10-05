# Changelog

## [Unreleased]

### Added

#### New tools
- `evaluation twsa-evaluation` compares a simulated total water storage anomaly against a gridded reference. It accepts an mHM fluxes-and-states record, a total water storage or a ready anomaly, brings both onto the coarser grid and onto the reference's own observation windows rather than calendar months, scores every cell with the metrics `--metrics` selects (`all` by default, `none` or a comma separated list of `kge` and `rmse`), and writes the metrics, the anomaly fields and a map of every selected metric for the whole domain and per `WMO_REGION_BOUNDS` region. Maps of the KGE components and a time series plot of every cell per region are written on request (`--plot-kge-components`, `--plot-region-cells`). A metric that is not selected is neither calculated, written nor plotted. `--metrics none` together with `--no-write-twsa` is refused before anything is read, since such a run would compute nothing. Windows come from CF time bounds where present and from evenly spaced stamps otherwise; a gap, a missing unit, a reference coarser than the input or a record not covering the baseline is refused rather than guessed. Storage levels are added back before scoring, because a KGE bias over two near-zero anomaly means explodes; where one record arrived as an anomaly its level is unknown, both get the same offset, and a warning states that the bias no longer measures a storage bias.
- `legacy-tools snow-evaluation` compares a simulated snow field against a gridded snow reference. It brings both onto the coarser grid and calendar (never finer than 8 day composites), reduces them to a binary snow flag, and writes per snow year and cell the first and last snow covered day, the season span and the number of snow covered days, plus classification accuracy maps, the snow covered share over time per hemisphere, and an animated three panel comparison - each repeated per `WMO_REGION_BOUNDS` region holding data. It streams throughout: records are probed for their coordinates instead of opened, read one file at a time, and reduced in blocks sized against `--max-memory-gib`, and a step counts as snow covered when any step it was aggregated from exceeded the threshold, so peak memory follows the compared grid rather than the record (0.51 instead of 2.26 GiB over 500 daily files).
- `create-wmo-region-masks` delineates (or reuses) global basins, rasterizes the 6 WMO regions from a shapefile set and a GeoJSON, and maps every basin to exactly one region through a basin-adjacency graph seeded from anchor basins, which keeps regions contiguous and stops endorheic basins from forming islands of the wrong region as the per-basin majority vote of `legacy create-subdomain-masks` could. Greenland is excluded. It writes one mask per region plus a 4-panel overview plot.
- `discharge-eval-comparison` compares two or more `discharge-evaluation` `results.csv` files with CDF, catchment-map and a new per-gauge `map-diff` plot (violin plots on request), each broken down per GRDC region derived from the gauge id and collected into one overview PDF. Values that cannot occur by construction (`kge`/`nse` above 1) are discarded before plotting so one diverged run cannot distort an axis or a KDE.
- `data-converter format-data` prepares every categorical mHM input - soil, geology, LAI and land cover - onto the exact DEM grid from one raster or a CSV/TXT manifest, writing `nc` (default), `asc` or `tif` plus the `*_classdefinition.txt` files mHM expects. Categories are reclassified at the source resolution before the grid warp, so no interpolation invents a class. Multi-horizon soil quantizes raster windows into profiles (v5 profile classes for `asc`, v6 per-horizon classes for `nc`) and accepts six bulk density units; gridded LAI aggregates in time to `daily`, `monthly`, `annual` or `long-term-mean-monthly` and streams one row block and time step at a time; land cover validates that the manifest periods are gapless and non-overlapping and writes one file per period or a single CF time stack. Every output is restricted to the DEM domain, with nodata inside it filled from the nearest classified neighbour per input layer unless `--no-fill-nodata` is given, and a run that cannot fit on the output volume is refused up front. Peak memory stays flat as the grid or the record grows.
- `data-converter rasterize-map` burns a vector attribute onto the exact grid of a reference DEM and writes ASCII, NetCDF or GeoTIFF, reading the burn values from the vector or through a lookup table for textual categories, and refusing to overwrite one of its own inputs.
- `setup-creation create-dem-derivatives` derives the filled DEM, slope, aspect, flow accumulation and flow direction in one pass, into one `nc` file or one file per derivative for `asc`/`tif`. Aspect, which `pyflwdir` does not provide, uses Horn's 3x3 gradient as a compass bearing, with flat cells written as nodata.

#### Existing tools
- Every tool writing gridded NetCDF gains `--compression` (0-9), `--no-shuffle`, `--significant-digits` and `--quantize-mode` in one `netcdf output` help group, across 17 commands from `latlon`, `crop-mhm-setup` and `fill-nearest` to the four evaluation tools. Omitting them keeps the compression the input file carries - the zlib level and, through netcdf-c's `_Quantize*` attribute, the lossy precision as well - and falls back to level 1 for a dataset built from scratch, which carries most of the saving at a fraction of the cost. A given option wins over the input, and only over its own axis: `--compression 9` on a file quantized to 3 digits changes the level and keeps the 3 digits, while `--compression 0` writes uncompressed and `--significant-digits 0` writes full precision, each switching one axis off the way the other does. Lossy quantization zeroes the insignificant mantissa bits so the zlib pass has far less entropy to encode, taking `mHM_Fluxes_States.nc` from 308,511 to 271,428 bytes at a maximum relative error of 2.7e-05 (`--significant-digits 4`) or to 242,091 bytes at 4.2e-03 (`--significant-digits 3 --quantize-mode GranularBitRound`). Integer data such as masks, every coordinate, and data variables that hold positions - a discharge evaluation's gauge `x`/`y`, which shifted by 27 m before the guard - are never quantized, and an engine other than netcdf4 drops quantization with a warning rather than appearing to apply it. The three raw `netCDF4` writers behind `format-data` keep their own fixed levels.
- `create-catchment` gains `--max-distance-m` to select outlet candidates within a radial distance in meters, mutually exclusive with the existing cell limit and its five-cell default, and `--no-max-error` to rank candidates without disqualifying any of them.
- `discharge-evaluation`, `gridded-data-evaluation`, `snow-evaluation` and `twsa-evaluation` log every statistic they calculate as a table with a `Global` row followed by one row per WMO region, holding the medians over the gauges or grid cells. `--write-region-stats` also writes the tables of the gridded, snow and TWSA evaluations to CSV. `gridded-data-evaluation` gains `--regions` and recalculates its spatial metrics (SPAEF, ESP, ...) for every region.
- `discharge-evaluation` draws its catchment maps again per WMO region (`catchment_map_region_<region>_<metric>.png`), framed by the region's bounds, and matches the catchment geometries only once for all of them.
- `--ncpus` sizes a local dask cluster in `twsa-evaluation` and spreads files over cores in `snow-evaluation` (0 for every core the machine reports); one core skips the cluster. Chunks are sized against `--max-memory-gib`, so a large budget yields fewer, larger chunks and less parallelism.

#### Shared modules
- `mhm_tools.common.netcdf.NetcdfCompression` carries the lossless (`complevel`, `shuffle`) and lossy (`significant_digits`, `quantize_mode`) settings as one object the writers and tools pass instead of separate arguments, validating them before any data is read. `add_netcdf_compression_args` and `get_netcdf_compression` in `common.cli_utils` declare the options and resolve them into that object, returning None when none was given so the input's compression applies; they are the first shared argument helpers in the CLI. `is_coordinate_like_variable` recognises a variable holding positions by name, `standard_name`, `units` or a bounds reference.
- `mhm_tools.common.crs_handler` collects the CRS handling: `resolve_crs` uses a supplied CRS only when the object carries none and raises when the two disagree, `set_spatial_dims` rejects axes that are not one-dimensional, finite and regularly spaced, and `write_object_crs` attaches a CRS to a Dataset or DataArray.
- `mhm_tools.common.lookup_handler` reads lookup tables and CSV manifests, matching column names case-, punctuation- and unit-suffix-insensitively and validating every value with the offending row number.
- `mhm_tools.common.parallel` provides `create_dask_cluster` and `resolve_n_cpus`.
- `mhm_tools.common.metrics.kge` holds the Kling-Gupta efficiency as `calculate_kling_gupta_efficiency` for a pair of series and `calculate_kling_gupta_efficiency_per_cell` for a whole grid.
- `mhm_tools.common.plotter` holds the shared plot building blocks: `plot_single_map` and `round_sensibly` (moved from `gridded_data_evaluation`, which still provides them), `create_discrete_colour_norm`, `calculate_colour_limits`, `get_metric_plot_style`, `add_map_colorbar`, `style_map_axes`, `create_comparison_title`, `create_axis_label`, `create_summary_text`, `create_table_text` and `get_percent_diff_bounds`, which picks the percent difference bins of `PERCENT_DIFF_BOUNDS` that `plot_map` uses with `percent_difference`. `create_bin_colormap` builds the bin colours of `create_discrete_colour_norm`, with the neutral centre bin of diverging maps. `NSE_CONSTANT_MEAN_BOUND` joins `KGE_CONSTANT_MEAN_BOUND` in `common.constants`.
- `mhm_tools.common.units` reads and converts the CF style units of amounts and rates: volumes and depths of water (1 kg m-2 of water is 1 mm) per s, min, h, d, month or year (a year is 365.25 days), in their common spellings (`m3 s-1`, `m³/s`, `mm/d`, `kg/m2/s`, ...). It provides `split_units`, `calculate_conversion_factor`, `calculate_amount_factor` (a rate over one step as an amount, a depth as a volume via the catchment area), `convert_rate_time_unit`, `get_closest_time_unit` and `convert_temperature_to_celsius`. `prepare-mhm-forcings`, `mhm-run-overview`, `twsa-evaluation` and `discharge-evaluation` use it instead of their own unit tables; `WATER_STORAGE_UNIT_FACTORS_MM` gives way to `AMOUNT_UNITS` and `TIME_UNIT_SECONDS` in `common.constants`.
- `calculate_median_time_step_seconds` in `common.time_utils` reads the step length of datetime64 and cftime axes; `convert_dataarray_units` and `calculate_region_medians` in `common.xarray_utils` convert a DataArray keeping its attributes and summarise variables per WMO region; `write_stats_table` in `common.utils` logs a table and optionally writes it to CSV. `create_results_csv` returns the results it writes and can leave out its log table.
- `get_file_res` takes a dataset through `ds` and reports a missing resolution through `raise_exception` instead of always raising.
- `convert_meters_to_degrees` in `common.utils` and `EARTH_RADIUS_M`, `EARTH_RADIUS_KM`, `METERS_PER_DEGREE` and `MIN_COS_LATITUDE` in `common.constants` put every distance on one sphere, replacing three separate literals. The conversion is latitude aware, since a meridional degree covers only 70 % of the needed span in longitude at 60 degrees north.
- Helpers promoted into `common` for reuse across tools: the grid, time and region helpers of `snow-evaluation` (`common.xarray_utils`, `common.time_utils`, `common.utils`, `common.plotter`), `write_mask_to_file` and `create_mask_from_polygon` for the region masks, `normalize_cli_sequence` for the CLI, and the WMO region tables and `GREENLAND_COORDS` into `common.constants`.
- `read_dataset`, `get_dataset_from_path` and `get_xarray_ds_from_file` take one variable name or a sequence of them and restrict what is read instead of only sizing the chunks. An mHM fluxes-and-states file holds 28 variables of which an evaluation needs one, so a multi-year record moved roughly 25 times more data than the tool used.
- `resample_to_target_freq` gains `resample_origin`, `time_anchor` and `drop_partial_edges`, all defaulting to the previous behavior, so records starting on different days share bins, a label describes the steps it aggregates, and an edge step only one record covers can be dropped. A multi-day step stays anchored to its origin under both pandas 2 and 3, which stopped treating `Day` as Tick-like and silently ignored `origin`.

### Changed
- `write_xarray_to_file` owns the compression of every NetCDF output, so the `zlib`/`complevel`/`shuffle` literals that 12 write sites repeated are gone - `subdomain_masks` alone held seven, and `regrid`, `twsa-evaluation` and `snow-evaluation` silently omitted `shuffle`, compressing differently for no stated reason - along with the unused `COMPRESSION_DICT`. Each caller's `dtype` and fill values are left untouched.
- Coordinates follow their own rule, so grid geometry stays exact: a one-dimensional axis is never compressed, a two-dimensional coordinate of at least `MIN_COORD_COMPRESSION_VALUES` values is compressed losslessly, and none is ever quantized. Below that threshold a coordinate stays uncompressed, because chunking a twelve-element bounds array costs more than it saves.
- `Hydrograph.calc_kling_gupta_efficiency` delegates to `common.metrics.kge` instead of holding its own copy, so a discharge KGE and a gridded KGE are the same number by construction. The shared version filters to the finite pairs before correlating, which the method relied on its caller to do, and returns NaN instead of an infinity for a constant or zero-mean series.
- SPAEF and TSM share one paired NaN filter, and the temporal correlation of TSM is a blocked Spearman computed in `common.xarray_utils`.
- Every plot of `discharge-evaluation`, `discharge-eval-comparison`, `snow-evaluation`, `twsa-evaluation` and `metric-plots`, the catchment maps and the maps of `2d-map`, `ratio`, `difference` and `relative-difference` follows one style, written down in `AGENTS.md`: a "Comparison <input> with <reference> for years <first>-<last>" title, units in square brackets, a), b), ... panel labels, median and mean in the panel title, a 10.5 in width at 400 dpi (also the new default of `--dpi` and `--map-diff-dpi`), maps without axis ticks, and the input and reference drawn in `#79A3E6` and `#008176`. Output file names are unchanged.
- Map colorbars are discrete with about 9 bins and ticks on the dividers between the colours. Skill and correlation metrics (KGE, NSE, gamma, snow classification accuracy) use `viridis_r`, ratios and differences `coolwarm_r` (also the new default of `--map-diff-cmap`) with one neutral grey bin (`NEUTRAL_COLOR`) centred on 1 or 0, so values close to it stand out, and RMSE `magma_r`. The centre bin stays on the colormap's midpoint when values extend beyond only one limit. Limits follow the data min/max, or the 1st/99th percentile where outliers stretch the range, and are rounded outwards so rounding never moves a value off the scale; values beyond a limit get a colour of their own. KGE scales end at `KGE_CONSTANT_MEAN_BOUND` (-0.41) and NSE scales at `NSE_CONSTANT_MEAN_BOUND` (0.0), the scores of a constant mean predictor, with everything below them in light grey, and a correlation's lower limit drops to 0 once values fall below it. The `gridded-data-evaluation` maps share these colorbars; their fixed ratio scale now spans 0.45 to 1.55. Percent differences - the relative mean and monthly relative climatology difference of `gridded-data-evaluation`, `relative-difference`, and the `rel_diff` gauge and catchment maps of `discharge-evaluation`, now mapped in percent with `coolwarm_r` instead of as a fraction with `viridis_r` - use fixed bins with clean edges instead of limits from the data: ±25 % (steps of 5 around a ±2.5 % centre bin), ±50 % (steps of 10 around ±5 %), ±75 % (steps of 15 around ±5 %) or ±100 % (-100, -75, -50, -25, -10, 10, 25, 50, 75, 100), the narrowest holding the differences once outliers are ignored as on the other maps. Values beyond the bins get an arrow instead of widening the scale, and the twelve monthly panels share one set of bins.
- `relative-difference` calculates and writes the relative difference in percent, `100 * (ref - mod) / ref` with units `%`, instead of as a fraction with units `1`. `--vmin` or `--vmax` replace the percent bins.
- The coordinate resolution is derived from the average step with a median fallback for irregular grids, through `calculate_coordinate_resolution` everywhere it was derived inline.
- `gridded-data-evaluation` holds its streamed monthly series in a quarter of the memory: `float32` sums with `uint16` counts, subsets and buckets popped as they merge, and one preallocated field for the period mean. `monthly_counts` stays `uint32`, which cannot wrap on a long record.
- `create-catchment` crops the flow direction data to the reference shape's bounds plus a buffer scaling with the shape - `SHAPE_BUFFER_EXTENT_FACTOR` times its larger half extent, never less than the outlet search distance - instead of a fixed 1 degree, because a shape disagreeing with the flow network is off by a fraction of the catchment it describes. For a 194 km2 Alpine catchment the working domain drops from 1264x1304 to 241x282 L0 cells and the delineation from 4 min 18 s to 5 s. The written extent is unaffected: it still comes from the delineated mask, `--frame` and the L2 alignment.
- `create-catchment` scores outlet candidates by shape overlap alone (`1 - IoU`), since the IoU can never exceed the ratio of the smaller to the larger area and therefore already carries the area discrepancy, and restricts them to cells draining within `CANDIDATE_AREA_AGREEMENT_FACTOR` of the reference area, which drops candidates whose IoU could never reach that factor. Without a reference shape the area error still ranks them. `--no-max-error` means the same in both searches: the criterion only ranks, and the best candidate is always taken.
- `create-catchment` reports every unknown number in its gauge outputs as `nan` instead of mixing `-9999`, `None` and empty cells, names the matching methods `shape_iou`, `area_basinex` and `area_burek`, and documents in `write_gauges_to_csv` what each column holds. In `gauges_info.nc` the area is described as the upstream area at the delineated outlet and carries a `delineation` attribute naming the methods used, instead of attributing every area to the Burek correction.
- Outlet distances are measured coordinate-aware, with latitude-dependent distances on geographic grids, instead of the Burek implementation's square-cell approximation.
- CDF plots fill a landscape page of the overview PDF, use a small low-opacity `+` marker instead of large high-opacity `o` markers, mark the median with a line at 0.5, show the solid CDF lines in the legend, and stop the axis at -0.41 for KGE and 0.0 for NSE.
- `discharge-evaluation` converts the simulated discharge to the unit of the observed one before scoring, e.g. `m3 d-1` to `m3 s-1`. A record without units is taken as m³/s with a warning, and a depth rate compared with a volume rate is refused.
- `diff` of `discharge-evaluation` and the hydrographs is the volume difference of sim - obs in m³, the summed rate differences times the time step length, instead of a sum of rates labelled m³; a depth rate is turned into a volume with the catchment area. `rel_diff` is unchanged.
- Catchment maps are titled "Metrics at the catchment outlets", since a polygon shows the metric of its outlet gauge, and "Median metrics per catchment" where `metric-plots` takes medians.
- Violin plots are only made on request: `discharge-evaluation --metric-plot-types` defaults to `cdf map catchment-map` and `metric-plots --plot-types` to `cdf`.

### Fixed
- `latlon`'s `-x/--compression`  had no effect. It built a flat encoding dict that the writer was never handed, and the coordinate path stripped compression regardless, so every latlon file was written uncompressed whatever the flag said. Its lat/lon grids are coordinates rather than data variables, and the file drops from 6,705,750 to 113,929 bytes, or 72,523 with `--compression 9`.
- A dataset read from a compressed file imposed that file's zlib level on every write, so the documented default never applied. The requested level now wins, and the level is only inherited when none was requested.
- `norm_deviation` of the TSM metric subtracted the spatial mean before dividing by it only for the first term: `data - mean / mean` instead of `(data - mean) / mean`. Every TSM value was wrong.
- `get_std_from_ds` had its climatology test inverted, so the deseasonalising branch never ran and the standard deviation was the raw one. `std` and every `rel_std` and std panel change. A dead branch guarding on the standard library `array` module, which no value can be an instance of, is gone.
- `align_bounds_to_l2` overshot the upper bound by one L0 cell, which left the domain indivisible by the upscaling factor and fell back to a crop ignoring the meteo grid, offsetting `mask_l2` against the forcing.
- Mask upscaling works for domains narrower than a single target cell, and resolution mismatches from float32 coordinate rounding no longer occur.
- A gridded file whose coordinates are not plain dimension coordinates is read correctly again: selecting variables keeps the coordinate variables, `normalize_lat_lon` promotes a coordinate stored as a data variable and indexes it, and both read paths normalize before reading the axis order, where an unindexed coordinate always compared as ascending and left a descending grid unflipped.
- A reference on a 0 to 360 longitude axis, as GRACE ships it, is compared against a model grid again; a global axis is rotated onto -180 to 180 and re-sorted, while a regional axis crossing the antimeridian is refused rather than torn in two.
- `get_ds_extend` no longer raises `KeyError` when a coordinate names a bounds variable that a variable selection left behind.
- `--available-mem` is read as a budget in GiB and keeps a fractional value.
- `prepare-mhm-forcings` multiplied precipitation in `kg m-2` by 1000, though 1 kg m-2 of water is 1 mm, and rates in `mm s-1` on a daily step by 90000 instead of 86400. Rates needed a time step `pd.infer_freq` recognised, which failed on gaps, on multi-hour steps and on the older `H` alias; the step now comes from the time axis, and units such as `mm/d`, `kg/m2/s` or `degree_Celsius` are understood.
- `mhm-run-overview --convert-units` knew no hours, so rates of an hourly setup were expressed per second, and it took a month as 30.4 days.
- `pretty_print_df` showed every value below 10 with one decimal, so a KGE of 0.453 read 0.5; values now carry 4 significant digits and integers stay whole.
- `discharge-evaluation` without `--catchment-map-variables` selected no metric at all, so it drew no catchment maps and wrote an empty summary table into the overview PDF.
- Metric CDF plots drew every line and median line twice.
- The title of the relative mean difference map of `gridded-data-evaluation` gave the median in percent but the mean as the ratio input / reference, e.g. 1.05 instead of 5 %. Both are now in percent.
- `plot_map`, behind `2d-map`, `ratio`, `difference` and `relative-difference`, took the colour limits and colorbar extension of a map zoomed to a region from the whole field, and a constant field painted its empty cells and ignored the region.
- `discharge-evaluation` dropped a gauge from its KGE and NSE CDFs as soon as one KGE component of it was missing.
- The CLI builds and behaves under a stricter click: a repeatable option with a single default no longer takes the whole CLI down while building, `store_false` flags are no longer stuck on their off value, and a repeatable option splits a quoted value on whitespace as well as on commas, so `--mask-paths="a.txt b.txt"` names two files again. Label options are only split on commas.

### Removed
- `discharge-evaluation` no longer writes `cdf_<var>_global_color_by_region.png`; `cdf_<var>_regions.png` shows the same breakdown as one CDF per region. The unused `plot_kde` is gone.

### Tests
- NetCDF compression: the CLI helpers, including that no option yields no settings so the input is inherited; the coordinate rules by dimension count and size; inheritance, override, partial override and both off-switches checked against the `_Quantize*` attribute the file actually carries; every quantize mode; integer safety; and the engine guard.
- Snow evaluation: the snow flag threshold, classification accuracy with missing steps, the season metrics and their snow-year window, the 8 day calendar normalization, grid aggregation, hemisphere split, region selection and an end-to-end run.
- Streamed monthly series of `gridded-data-evaluation`: per-bucket division, a calendar month split across subsets, an end-to-end check against a direct monthly resample on one and two cores, and the accumulator dtypes and subset release.
- Unit conversion: the unit spellings, conversion factors, rates turned into amounts with and without an area, rates per another time unit, temperatures, the median time step and the DataArray conversion, plus the forcing conversion, the run-overview rate conversion, the storage conversion to mm, the discharge unit conversion and the volume `diff` of the hydrographs.
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
