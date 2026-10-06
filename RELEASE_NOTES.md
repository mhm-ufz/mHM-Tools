# Release notes

## v0.3

v0.3 adds seven tools for setup preparation and evaluation, puts every evaluation on one plot style with statistics per WMO region, adds compression options to all gridded NetCDF output, and fixes several unit and statistics errors that changed results.

This enables the preparation of mHM setups from scratch as well as a broader range of evaluations, including comparisons between different model realisations or calibrations.

### Data preparation
Where privious versions focussed on the creation of local setups from existing global setups, version 0.3 builds the basis for mHM setup creation from raw data and serves as the backend for the upcoming mHM QGIS tool.

#### New preparation tools
- **`format-data`**: formats categorical input data for mHM on the exact grid and CRS of a DEM, writing `nc` (default), `asc` or `tif`. Categories are reclassified at the source resolution before warping onto the DEM grid, so no interpolation invents a class, and nodata inside the DEM domain is filled from the nearest valid neighbour unless `--no-fill-nodata` is given.
  - **Soil** (`--type soil`): either a soil raster mapped through a lookup table and written with `soil_classdefinition.txt`, or a CSV/TXT manifest of clay, sand, silt and bulk density rasters per horizon. Horizons are classified into v5 profile classes (`asc`) or v6 per-horizon classes (`nc`), at class intervals set by `--composition-step` and `--bulkdensity-step`, and six bulk density units are accepted. Gaps are filled per input layer, so a hole in one layer no longer drops the cell from every horizon.
  - **Geology** (`--type geology`): a geology raster mapped through a lookup table onto geology classes. The same table supplies the class order and karstic flag of the `geology_classdefinition.txt` written next to it.
  - **Land cover** (`--type lc`): a single land cover raster mapped through a lookup table onto mHM land cover classes, or a manifest of historical periods that follow each other without gaps or overlaps. Periods are written as one ASCII file each or as one NetCDF time stack (`lc_periods.nc`).
  - **LAI** (`--type lai`): either a land cover raster mapped to LAI classes and written with `LAI_classdefinition.txt`, or a gridded LAI NetCDF aggregated in time to `daily`, `monthly`, `annual` or `long-term-mean-monthly` (default, `--output-temporal-resolution`). Gridded LAI is streamed, so memory doesn't grow with the record length, and a run whose output won't fit on disk is refused before it starts.
  - Gridded LAI is aggregated in time to `daily`, `monthly`, `annual` or `long-term-mean-monthly` (`--output-temporal-resolution`).
  - Land cover periods from a manifest must be gapless and non-overlapping and are written as one file per period or as one time stack.
  - Nodata inside the DEM domain is filled from the nearest classified neighbour unless `--no-fill-nodata` is given.
- **`rasterize-map`**: burns an attribute of a vector map, e.g. a shapefile, onto the exact grid of a reference DEM.
  - Takes the values from a vector attribute (`--burn-field`) or, for text categories, through a lookup table (`--lookup-table`).
  - Writes ASCII, NetCDF or GeoTIFF, and refuses to overwrite one of its own inputs.
- **`create-dem-derivatives`**: derives the filled DEM, slope, aspect, flow accumulation and flow direction from a DEM in one pass, using `pyflwdir`.
  - Writes every derivative into one NetCDF file, or one file per derivative for `asc` and `tif` (`--output-extension`).
  - Aspect, which `pyflwdir` does not provide, is calculated from Horn's 3×3 gradient as a compass bearing; flat cells are written as nodata.
- **`create-wmo-region-masks`**: creates hydrologically distinct masks of the six WMO regions, whose borders follow river basins, so every basin belongs to exactly one region.
  - Delineates global basins with `pyflwdir`, or reuses existing ones, and rasterizes the rough WMO regions from a shapefile set and a GeoJSON (`--shape-dir`, `--region-geojson`).
  - Assigns each basin through a basin-adjacency graph seeded from anchor basins, which keeps regions contiguous and stops endorheic basins from forming islands of the wrong region.
  - Excludes Greenland, and writes one mask per region plus a four-panel overview plot.

### Evaluation
#### New evaluation tools
- **`twsa-evaluation`**: compares simulated total water storage anomalies (TWSA) against a gridded reference such as GRACE.
  - Accepts an mHM fluxes-and-states record, a total water storage or a ready anomaly, and brings both records onto the coarser grid and the reference's own observation windows.
  - Scores every grid cell with KGE and RMSE (`--metrics`), with anomalies relative to a baseline period (`--baseline-start-year`, `--baseline-end-year`).
  - Writes the metrics, the anomaly fields and a map of each metric for the whole domain and per WMO region; maps of the KGE components and time series per region on request (`--plot-kge-components`, `--plot-region-cells`).
  - Refuses a gap in the record, a missing unit or a reference coarser than the input instead of guessing.
- **`snow-evaluation`**: compares a simulated snow field against a gridded snow reference.
  - Brings both onto the coarser grid and calendar (8-day composites at the finest) and reduces them to a binary snow flag (`--snow-threshold`).
  - Writes per snow year and grid cell the first and last snow-covered day, the season length and the number of snow-covered days, plus classification accuracy maps and the snow-covered share over time per hemisphere.
  - Adds an animated three-panel comparison (skip it with `--no-gif`) and repeats every output per WMO region.
  - Streams through the record, so memory follows the compared grid rather than the record length (`--max-memory-gib`).
- **`discharge-eval-comparison`**: compares the `results.csv` of two or more `discharge-evaluation` runs side by side, e.g. different model realisations or calibrations.
  - Draws CDFs, catchment maps and a per-gauge difference map against a reference run (`--reference-name`); violin plots on request (`--plot-types`).
  - Breaks every plot down per GRDC region, derived from the gauge id, and collects them in one overview PDF.
  - Discards values that cannot occur, such as a KGE or NSE above 1, so one diverged run cannot distort an axis.

#### Harmonization of plots
- **One plot style**: shared titles, units, colours and discrete colorbars. Relative differences use fixed percent bins of ±25, ±50, ±75 or ±100 %.
- **Units**: one converter understands the common CF spellings (`m3 s-1`, `mm/d`, `kg/m2/s`, …), and simulated discharge is converted to the observed unit before scoring.
- **Statistics per WMO region**: every evaluation logs a `Global` row and one row per WMO region; `--write-region-stats` also writes them to CSV.

### Additional Highlights
- **`create-catchment`**: outlets are matched by shape overlap and the crop is sized to the reference shape, taking a 194 km² catchment from 4 min 18 s to 5 s. New option `--max-distance-m`.
- **NetCDF compression**: `--compression`, `--significant-digits` and `--quantize-mode` on every tool writing gridded NetCDF. Lossy quantization never touches integers or coordinates, and an input file's compression is kept unless you override it.

### Upgrade notes
- `converter-nc-ascii` is now called `file-converter`.
- New dependency: `rasterio`.
- Violin plots are only drawn on request, and `cdf_<var>_global_color_by_region.png` is no longer written; `cdf_<var>_regions.png` shows the same breakdown.

See `CHANGELOG.md` for the full list of changes.

---
---

## v0.2.3
### Added 
- CF `long_name`/`units` metadata on `calc_diff`/`calc_ratio`/`calc_rel_diff` and gridded-data-evaluation outputs
- `--update-tile-masks` flag for `create-mhm-restart-from-setup` to refresh reused tiles' masks

### Fixed
- Restore CF attributes dropped by xarray ops in DEM masking, catchment merging, unit conversion
- Fix wrong `_FillValue`/`missing_value` from a swapped `fill_nearest` argument
- Fix `merge_files` staging to a stale path (cross-device link risk)
- Clean up leftover per-folder outputs when `preserve_folders` is off
- Fix broken symlinks crashing `exists()` checks in `merge_files`/`migrate_grid_using_systemlink`/`link_folder_tree`
- Fix restart file's `_FillValue` (`NaN` → `-9999.0`) so mHM can read it back
- Fix `gridded-data-evaluation` crashing/silently skipping the mask on inputs with non-`lat`/`lon` spatial dim names (e.g. `latitude`/`longitude`)

---
## v0.2.2 
### Added

- Always derive `L1_soilMoist` in the merged restart as the midpoint between `L1_wiltingPoint` and `L1_soilMoistFC` when both are available, overwriting any value a native tile restart may already carry.

### Changed
- Extend `generate_bounds`/`generate_bounds_for_all_coords` to accept an explicit resolution fallback for coordinates with only one point, where a cell width can no longer be derived by differencing.
- Add a `create_bounds` option to `get_dataset_from_path`, mirroring the existing option on `get_xarray_ds_from_file`, to attach real coordinate bounds to a dataset at open time, before any cropping.
- Move `regrid_mask` from `mhm_tools.pre.crop_mhm_setup` to `mhm_tools.common.xarray_utils`, since it's pure xarray grid-matching logic already used by both pre- and post-processing modules.
- `write_mask_file` now writes the catchment mask once (as `mask`) and adds `land_mask` as an HDF5 hard link to the same on-disk data via the new `add_variable_hard_link` helper (`mhm_tools.common.netcdf`), instead of writing the identical array twice under both names. Both variables remain fully and independently readable by any NetCDF/HDF5 tool, including all attributes, with no duplicated storage.

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

---
## v0.2.1

### Fixed
- missing documentation
- write shape error in catchment creation to csv
- fix spatial metrics (calculation per tilmestep then average not flatten)
- small bug fixes
- cleaner code
- better change log
---
## v0.2
First official release adding pre-processing tools as well as evaluation tools for mHM version 5. 
This includes: 
setup-creation: create domains by delineation (based on area and/or shape similarity), crop mHM setups, create headers, latlon files and prepare forcings including pet calculation.
data-processing: convert, merge, regrid, fill, aggregate, compare, and derive gridded data.
evaluation: evaluate discharge (cdf and map plots of KGE and NSE), hydrographs, gridded data (with multiple metrics and maps), and run overviews.
visualization: create 2D maps and Taylor diagrams.
utilities: helper commands like linking folder trees.
first mhm-v5-v6-converter tools: convert mHM5 landcover ASCII files to NetCDF.