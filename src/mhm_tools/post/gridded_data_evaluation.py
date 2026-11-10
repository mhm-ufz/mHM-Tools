"""Evaluate gridded model data against gridded reference data.

The module loads one or more input and reference datasets, crops or masks them
to a target domain, computes direct or bootstrap comparison metrics, and writes
spatial diagnostics, seasonal summaries, and result tables.

Authors
-------
- Simon Lüdke
"""

import logging
import random
import re
import warnings
from pathlib import Path
from typing import Iterable, Tuple

import dask
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
from joblib import Parallel, delayed
from mpl_toolkits.axes_grid1 import make_axes_locatable

from mhm_tools.common.file_handler import (
    ChunkType,
    get_coord_values,
    get_dataset_from_path,
    get_xarray_ds_from_file,
    write_xarray_to_file,
)
from mhm_tools.common.logger import ErrorLogger, log_arguments, log_errors
from mhm_tools.common.metrics.metrics_handler import (
    calculate_results_metric,
    create_results_csv,
)
from mhm_tools.common.netcdf import generate_bounds_for_all_coords
from mhm_tools.common.plotter import (
    KGE_UNDER_COLOR,
    add_map_colorbar,
    create_axis_label,
    create_comparison_title,
    get_metric_plot_style,
    get_percent_diff_bounds,
    plot_single_map,
    round_sensibly,
)
from mhm_tools.common.resolution_handler import Resolution, get_file_res
from mhm_tools.common.time_utils import (
    resample_to_target_freq,
    timedelta_to_alias,
)
from mhm_tools.common.utils import (
    cut_to_filled_area,
    format_region_title,
    sanitize_name,
    select_regions,
    split_file_list,
    write_stats_table,
)
from mhm_tools.common.xarray_utils import (
    SPEARMAN_BYTES_PER_ELEMENT,
    calculate_region_medians,
    calculate_spearman_over_time,
    crop_ds,
    crop_to_region,
    get_clim_from_ds,
    get_coord_key,
    get_ds_extend,
    get_overlapping_time_slice,
    normalize_lat_lon,
    regrid_mask,
    regrid_to_coarser_grid,
    select_time_steps_by_year_and_month,
)

logger = logging.getLogger(__name__)
# Working set of one spearman block. Small on purpose: the runtime is flat
# down to a few MiB, so a bigger block would only raise the peak.
SPEARMAN_BLOCK_BUDGET_MIB = 32
# Anomalies from a monthly climatology need several values per calendar month;
# with fewer years the seasonal cycle is kept in the std and the correlation.
MIN_YEARS_TO_REMOVE_SEASONAL_CYCLE = 5
# the correlation of the 12 climatology values measures nothing but the cycle
CLIMATOLOGY_SEASONALITY_LABEL = "seasonal cycle only"


class EvalDataset:
    """Dataset containing all the necessary information about a evaluation dataset."""

    path = None
    name = None
    var = None
    factor = 1
    file_name = "*.*"

    def __init__(self, path, name, var, factor, file_name):
        self.path = Path(path) if path is not None else None
        self.name = name
        self.var = var
        self.factor = factor
        self.file_name = file_name


def calculate_spearman_map(input_data, ref_data) -> Tuple[np.ndarray, np.ndarray]:
    """Correlate two stacks cell by cell over their leading axis.

    Vectorizing the correlation costs several float64 copies of whatever is
    handed to the kernel, so the stack is fed to it in latitude blocks. The
    block size is deliberately small: measured over a 200x280 grid the runtime
    is flat from a 1 GiB block down to a 4 MiB one, so a larger block would
    only raise the peak.

    Parameters
    ----------
    input_data, ref_data : numpy.ndarray
        Stacks of the same shape, with time or month as the leading dimension.

    Returns
    -------
    tuple of numpy.ndarray
        Spearman rho and two-sided p-value, both of shape (lat, lon).
    """
    if np.shape(input_data) != np.shape(ref_data) or len(np.shape(input_data)) != 3:
        with ErrorLogger(logger):
            msg = "Wrong shape for spatial spearman correlation!"
            raise ValueError(msg)
    # the kernel reduces the trailing axis, so the leading one is moved back
    input_np = np.moveaxis(np.asarray(input_data), 0, -1)
    ref_np = np.moveaxis(np.asarray(ref_data), 0, -1)
    n_lat, n_lon, n_steps = input_np.shape
    block_rows = max(
        1,
        int(
            SPEARMAN_BLOCK_BUDGET_MIB
            * 1024**2
            // (n_lon * n_steps * SPEARMAN_BYTES_PER_ELEMENT)
        ),
    )
    logger.info(
        f"Correlating {n_lat}x{n_lon} cells over {n_steps} steps "
        f"in blocks of {min(block_rows, n_lat)} rows."
    )
    rho = np.empty((n_lat, n_lon), dtype=np.float32)
    pval = np.empty((n_lat, n_lon), dtype=np.float32)
    for start in range(0, n_lat, block_rows):
        stop = min(start + block_rows, n_lat)
        rho[start:stop], pval[start:stop] = calculate_spearman_over_time(
            input_np[start:stop], ref_np[start:stop]
        )
    return rho, pval


def crop_datasets_to_spatial_overlap(input_ds, ref_ds):
    """Crop input and reference datasets to their common spatial overlap."""
    input_lon_min, input_lon_max, input_lat_min, input_lat_max = get_ds_extend(input_ds)
    ref_lon_min, ref_lon_max, ref_lat_min, ref_lat_max = get_ds_extend(ref_ds)

    overlap_lon_min = max(input_lon_min, ref_lon_min)
    overlap_lon_max = min(input_lon_max, ref_lon_max)
    overlap_lat_min = max(input_lat_min, ref_lat_min)
    overlap_lat_max = min(input_lat_max, ref_lat_max)

    if overlap_lon_min > overlap_lon_max or overlap_lat_min > overlap_lat_max:
        msg = (
            "Input and reference datasets do not share a common spatial overlap; "
            "cannot compare them."
        )
        with ErrorLogger(logger):
            raise ValueError(msg)

    cropped_input = crop_ds(
        input_ds,
        overlap_lon_min,
        overlap_lon_max,
        overlap_lat_min,
        overlap_lat_max,
    )
    cropped_ref = crop_ds(
        ref_ds,
        overlap_lon_min,
        overlap_lon_max,
        overlap_lat_min,
        overlap_lat_max,
    )

    input_cropped = (
        cropped_input["lat"].size != input_ds["lat"].size
        or cropped_input["lon"].size != input_ds["lon"].size
    )
    ref_cropped = (
        cropped_ref["lat"].size != ref_ds["lat"].size
        or cropped_ref["lon"].size != ref_ds["lon"].size
    )

    input_bounds_match_overlap = (
        np.isclose(input_lon_min, overlap_lon_min)
        and np.isclose(input_lon_max, overlap_lon_max)
        and np.isclose(input_lat_min, overlap_lat_min)
        and np.isclose(input_lat_max, overlap_lat_max)
    )
    ref_bounds_match_overlap = (
        np.isclose(ref_lon_min, overlap_lon_min)
        and np.isclose(ref_lon_max, overlap_lon_max)
        and np.isclose(ref_lat_min, overlap_lat_min)
        and np.isclose(ref_lat_max, overlap_lat_max)
    )

    if input_cropped or ref_cropped:
        if input_bounds_match_overlap and not ref_bounds_match_overlap:
            logger.info(
                "Input spatial extent is a subset of reference; cropping reference to the common overlap."
            )
        elif ref_bounds_match_overlap and not input_bounds_match_overlap:
            logger.warning(
                "Reference spatial extent is a subset of input; cropping input to the common overlap."
            )
        else:
            logger.warning(
                "Cropping input and reference datasets to their common overlapping spatial extent."
            )

    return cropped_input, cropped_ref


def get_std_from_ds(ds, input_var=None, clim=None, factor=1):
    """Calculate maps of temporal standard deviation from an DataArray.

    If a climatology is provided the timeseries can be detrended by
    seasonality.
    """
    # Retrieve data and apply factor
    data = ds * factor if input_var is None else ds[input_var] * factor

    # Subtract climatology for each month
    if clim is not None:
        data_reduced = data.groupby("time.month") - clim
        std = data_reduced.std(dim="time", skipna=True)
    else:
        std = data.std(dim="time", skipna=True)

    return std


def get_file_stats(
    ds_in,
    input_var,
    factor=1,
    coordinate_slice=None,
    output_path=None,
    avaiable_years=None,
    direct_comp=False,
    compression=None,
    remove_seasonal_cycle_from_std=False,
    season_months=None,
):
    """Get statistics for one file.

    With `remove_seasonal_cycle_from_std` the std is that of the anomalies from
    the monthly climatology, otherwise that of the raw values (both ddof=0).
    Only steps in `avaiable_years` and `season_months` are used.
    """
    # logger.debug(f"Get file stats {file}")
    lat_key = get_coord_key(ds_in, lat=True)
    lon_key = get_coord_key(ds_in, lon=True)
    try:
        spatial_resolution = _spatial_resolution(ds_in, lon_key, lat_key)
    except ValueError:
        spatial_resolution = None

    # Normalize to canonical 'lat'/'lon' dimension names, since the statistics
    # dataset built below always coordinates itself as 'lat'/'lon'.
    if lat_key != "lat" and "lat" in ds_in.coords:
        ds_in = ds_in.drop_vars("lat")
    if lon_key != "lon" and "lon" in ds_in.coords:
        ds_in = ds_in.drop_vars("lon")
    ds_in = normalize_lat_lon(ds_in, lat_key=lat_key, lon_key=lon_key)

    # Apply coordinate slicing if needed
    logger.debug(f"before cropping the file {ds_in}")
    # make sure that latitude order is from highest to lowest value
    if ds_in["lat"].shape[0] > 1 and ds_in["lat"][1] > ds_in["lat"][0]:
        ds_croped = ds_in.isel(lat=slice(None, None, -1))
    else:
        ds_croped = ds_in
    if coordinate_slice is not None:
        ds_croped = ds_croped.sel(
            {"lat": coordinate_slice["lat"], "lon": coordinate_slice["lon"]}
        )
    ds_croped = select_time_steps_by_year_and_month(
        ds_croped, avaiable_years, season_months
    )
    # the f-string is built even at INFO level, which is a full extra read of a
    # lazily opened record
    if logger.isEnabledFor(logging.DEBUG):
        logger.debug(
            f"After cropping/year selection {input_var} is all nan: "
            f"{bool(ds_croped[input_var].isnull().all().compute().item())}; "
            f"shape={ds_croped[input_var].shape}"
        )

    # Two passes over a lazy record: the anomalies of the std need the finished
    # climatology, and one graph for both would hold every chunk until then.
    # Computing here also keeps later reads of the stats from re-reading it.
    clim = get_clim_from_ds(ds_croped, input_var, factor)
    mean = ds_croped[input_var].mean(dim="time", skipna=True) * factor
    clim, mean = dask.compute(clim, mean)
    std = get_std_from_ds(
        ds_croped, input_var, clim if remove_seasonal_cycle_from_std else None, factor
    ).compute()
    logger.debug(
        f"After stats calculation {input_var} is all nan: "
        f"clim={bool(clim.isnull().all().compute().item())}, "
        f"std={bool(std.isnull().all().compute().item())}, "
        f"mean={bool(mean.isnull().all().compute().item())}"
    )

    # Aggregation/arithmetic above drops attrs; give each output CF metadata
    # describing the statistic, copying units from the source variable when known.
    source_units = ds_croped[input_var].attrs.get("units")
    clim.attrs = {"long_name": f"Monthly climatology of {input_var}"}
    std.attrs = {
        "long_name": create_std_long_name(input_var, remove_seasonal_cycle_from_std)
    }
    mean.attrs = {"long_name": f"Temporal mean of {input_var}"}
    if source_units is not None:
        clim.attrs["units"] = source_units
        std.attrs["units"] = source_units
        mean.attrs["units"] = source_units

    # Construct the output dataset with lazy evaluations
    output = xr.Dataset(
        {"clim": clim, "std": std, "mean": mean},
        coords={
            "month": np.arange(1, 13, 1),
            "lat": get_coord_values(ds_croped, lat=True),
            "lon": get_coord_values(ds_croped, lon=True),
        },
    )
    output = generate_bounds_for_all_coords(output, res=spatial_resolution)
    if direct_comp:
        ts = ds_croped[input_var] * factor
        ts.attrs = dict(ds_croped[input_var].attrs)
        ts.name = "time_series"
        output = xr.merge([output, ts])
    if output_path is not None:
        write_xarray_to_file(ds=output, file_path=output_path, compression=compression)
    return output


def _spatial_resolution(obj, lon_key, lat_key):
    """Return the absolute spatial resolution for a dataset or data array."""
    return abs(
        get_file_res(lon=obj[lon_key], lat=obj[lat_key], resolutions=Resolution())
    )


def _first_2d_spatial_slice(mask_da, lat_key, lon_key):
    """Reduce non-spatial mask dimensions by selecting the first slice."""
    mask_2d = mask_da
    extra_dims = [d for d in mask_2d.dims if d not in [lat_key, lon_key]]
    for dim in extra_dims:
        mask_2d = mask_2d.isel({dim: 0}, drop=True)
    return mask_2d


def _select_mask_for_grid(mask_ds, target_ds, mask_var=None):
    """Select the mask variable matching the target grid resolution."""
    lat_key_ds = get_coord_key(target_ds, lat=True)
    lon_key_ds = get_coord_key(target_ds, lon=True)
    target_res = _spatial_resolution(target_ds, lon_key_ds, lat_key_ds)

    if mask_var is not None:
        if mask_var not in mask_ds.data_vars:
            msg = f"Mask variable {mask_var!r} not found in mask dataset."
            with ErrorLogger(logger):
                raise ValueError(msg)
        mask_da = mask_ds[mask_var]
        lat_key_mask = get_coord_key(mask_da, lat=True)
        lon_key_mask = get_coord_key(mask_da, lon=True)
        mask_res = _spatial_resolution(mask_da, lon_key_mask, lat_key_mask)
        logger.info(
            f"Using explicitly selected mask variable {mask_var!r} "
            f"with resolution {mask_res} for target resolution {target_res}."
        )
        return (
            _first_2d_spatial_slice(mask_da, lat_key_mask, lon_key_mask),
            mask_var,
            lat_key_mask,
            lon_key_mask,
            mask_res,
            target_res,
        )

    exact_match = None
    finer_matches = []
    coarser_matches = []
    for var_name in mask_ds.data_vars:
        mask_da = mask_ds[var_name]
        try:
            lat_key_mask = get_coord_key(mask_da, lat=True)
            lon_key_mask = get_coord_key(mask_da, lon=True)
            mask_res = _spatial_resolution(mask_da, lon_key_mask, lat_key_mask)
        except Exception:
            logger.debug(
                f"Skipping mask variable {var_name!r}; no usable spatial resolution.",
                exc_info=True,
            )
            continue
        candidate = (
            mask_da,
            var_name,
            lat_key_mask,
            lon_key_mask,
            mask_res,
            target_res,
        )
        if abs(mask_res - target_res) <= 1e-5:
            exact_match = candidate
            break
        if mask_res < target_res:
            finer_matches.append(candidate)
        else:
            coarser_matches.append(candidate)

    selected = exact_match
    if selected is None and finer_matches:
        selected = min(finer_matches, key=lambda item: item[4])
    if selected is None:
        available = [item[4] for item in coarser_matches]
        msg = (
            "Could not select a mask variable for gridded data evaluation: "
            f"target resolution is {target_res}, and only coarser mask "
            f"resolutions are available: {available}."
        )
        with ErrorLogger(logger):
            raise ValueError(msg)

    mask_da, var_name, lat_key_mask, lon_key_mask, mask_res, target_res = selected
    logger.info(
        f"Selected mask variable {var_name!r} with resolution {mask_res} "
        f"for target resolution {target_res}."
    )
    return (
        _first_2d_spatial_slice(mask_da, lat_key_mask, lon_key_mask),
        var_name,
        lat_key_mask,
        lon_key_mask,
        mask_res,
        target_res,
    )


def apply_spatial_mask(ds, mask_da, mask_var=None):
    """Mask all spatial data variables in a dataset with a spatial mask."""
    if mask_da is None:
        return ds
    lat_key_ds = get_coord_key(ds, lat=True)
    lon_key_ds = get_coord_key(ds, lon=True)

    if isinstance(mask_da, xr.Dataset):
        (
            mask_2d,
            selected_mask_var,
            lat_key_mask,
            lon_key_mask,
            mask_res,
            target_res,
        ) = _select_mask_for_grid(mask_da, ds, mask_var=mask_var)
        mask_on_ds = regrid_mask(
            mask_ds=mask_2d,
            lon_key_mask=lon_key_mask,
            lat_key_mask=lat_key_mask,
            target_lon=ds[lon_key_ds].data,
            target_lat=ds[lat_key_ds].data,
            lon_key_target=lon_key_ds,
            lat_key_target=lat_key_ds,
            target_res=target_res,
            mask_res=mask_res,
        )
        logger.debug(
            f"Applied mask variable {selected_mask_var!r} to target grid "
            f"with shape {mask_on_ds.shape}."
        )
    else:
        lat_key_mask = get_coord_key(mask_da, lat=True)
        lon_key_mask = get_coord_key(mask_da, lon=True)
        mask_2d = _first_2d_spatial_slice(mask_da, lat_key_mask, lon_key_mask)

        if (
            mask_2d.sizes.get(lat_key_mask, 0) == 0
            or mask_2d.sizes.get(lon_key_mask, 0) == 0
        ):
            return ds
        target_coords = {
            lat_key_mask: ds[lat_key_ds].values,
            lon_key_mask: ds[lon_key_ds].values,
        }
        mask_on_ds = mask_2d.interp(target_coords, method="nearest")
        mask_on_ds = mask_on_ds.rename(
            {lat_key_mask: lat_key_ds, lon_key_mask: lon_key_ds}
        )

    valid_mask = np.isfinite(mask_on_ds) & (mask_on_ds > 0)
    logger.info(
        f"Spatial mask has {int(np.count_nonzero(valid_mask.values))} valid cells "
        f"on target grid with shape {valid_mask.shape}."
    )
    if not np.any(valid_mask.values):
        msg = (
            "Spatial mask does not select any valid cells on the target grid. "
            "Check the mask extent, selected mask variable, and target dataset "
            "resolution."
        )
        with ErrorLogger(logger):
            raise ValueError(msg)
    logger.debug(
        f"Mask after regrid is all nan: "
        f"{bool(mask_on_ds.isnull().all().compute().item())}; "
        f"valid cells={int(np.count_nonzero(valid_mask.values))}"
    )
    logger.debug(
        f"Before spatial mask stats are all nan: "
        f"clim={bool(ds['clim'].isnull().all().compute().item()) if 'clim' in ds else None}, "
        f"std={bool(ds['std'].isnull().all().compute().item()) if 'std' in ds else None}, "
        f"mean={bool(ds['mean'].isnull().all().compute().item()) if 'mean' in ds else None}"
    )

    out = ds.copy()
    for var_name in out.data_vars:
        da = out[var_name]
        if lat_key_ds in da.dims and lon_key_ds in da.dims:
            original_attrs = dict(da.attrs)
            out[var_name] = da.where(valid_mask)
            out[var_name].attrs = original_attrs
    logger.debug(
        f"After spatial mask stats are all nan: "
        f"clim={bool(out['clim'].isnull().all().compute().item()) if 'clim' in out else None}, "
        f"std={bool(out['std'].isnull().all().compute().item()) if 'std' in out else None}, "
        f"mean={bool(out['mean'].isnull().all().compute().item()) if 'mean' in out else None}"
    )

    # Crop to the bounding box of the filled mask area.
    mask_np = np.asarray(valid_mask.values)
    if np.any(mask_np):
        l0_resolution = None
        if out[lon_key_ds].size > 1:
            l0_resolution = float(
                np.abs(out[lon_key_ds].values[1] - out[lon_key_ds].values[0])
            )
        resolutions = Resolution(l0_resolution=l0_resolution)
        cut_ds = xr.Dataset(coords={"lat": out[lat_key_ds], "lon": out[lon_key_ds]})
        buffer = min(5, cut_ds[lon_key_ds].size // 20, cut_ds[lat_key_ds].size // 20)
        lat_slice_idx, lon_slice_idx = cut_to_filled_area(
            ds=cut_ds,
            resolutions=resolutions,
            catchment_mask=mask_np,
            buffer=buffer,
        )
        out = out.isel({lat_key_ds: lat_slice_idx, lon_key_ds: lon_slice_idx})
        logger.debug(
            f"After mask crop stats are all nan: "
            f"clim={bool(out['clim'].isnull().all().compute().item()) if 'clim' in out else None}, "
            f"std={bool(out['std'].isnull().all().compute().item()) if 'std' in out else None}, "
            f"mean={bool(out['mean'].isnull().all().compute().item()) if 'mean' in out else None}; "
            f"shape={out.sizes}"
        )

    return out


def get_files(path, n_bootstrap_years=None, available_years=None, file_name="*.*"):
    """Recursevely find all netcdf files in directory."""
    nc_files = []
    # Search for .nc files at each depth level
    if len(year_structure_paths(path, file_name=file_name)) > 0:
        all_years = get_years_from_path(path, file_name=file_name)
        if available_years is not None:
            selectable_years = [y for y in all_years if y in available_years]
        else:
            selectable_years = all_years
        logger.debug(
            f"selectable years are {selectable_years} - n_bootstrap_years is {n_bootstrap_years}"
        )
        if n_bootstrap_years is not None:
            # needs fixed folder structure of y/m/file
            selectable_years = random.choices(selectable_years, k=n_bootstrap_years)
        for year in selectable_years:
            folder_path = path / str(year)
            nc_files.extend(list(folder_path.rglob(file_name)))
    else:
        nc_files = list(path.rglob(file_name))
    return nc_files


def combine_results(results):
    """Combine the statistics calculated for subsets into one.

    Args:
        results: One tuple per subset, as returned by
            `get_stats_one_pass_subset`. Consumed in place, so the caller must
            not read it again.

    Returns
    -------
        Tuple of the total step count, the summed monthly sums, square sums
        and monthly counts, and the merged period sums and counts.
    """
    step_count = sum(count for count, _, _, _, _, _ in results)
    if step_count == 0:
        msg = (
            "No time step was read for the one-pass statistics. Check the "
            "selected years and season."
        )
        with ErrorLogger(logger):
            raise ValueError(msg)
    monthly_sums = sum(sums for _, sums, _, _, _, _ in results)
    square_sums = sum(squares for _, _, squares, _, _, _ in results)
    monthly_counts = sum(counts for _, _, _, counts, _, _ in results)
    # each subset streams its own files, so a calendar month can be split
    # across subsets and its sums have to be added up per key. Both the subset
    # and its buckets are popped as they are merged, so a subset's copy of the
    # record is released here instead of outliving the merge.
    period_sums, period_counts = {}, {}
    while results:
        _, _, _, _, subset_sums, subset_counts = results.pop(0)
        for key in list(subset_sums):
            values, counts = subset_sums.pop(key), subset_counts.pop(key)
            if key in period_sums:
                period_sums[key] += values
                period_counts[key] += counts
            else:
                period_sums[key] = values
                period_counts[key] = counts
    return (
        step_count,
        monthly_sums,
        square_sums,
        monthly_counts,
        period_sums,
        period_counts,
    )


def get_stats_one_pass_subset(
    files,
    input_var,
    factor=1,
    coordinate_slice=None,
    with_period_series=False,
    remove_seasonal_cycle_from_std=False,
    available_years=None,
    season_months=None,
):
    """Accumulate per-month sums, square sums and counts from a list of files.

    Iterates through the NetCDF files one time step at a time, optionally
    applies a spatial slice, multiplies by `factor` and adds every step to the
    sums of its calendar month. NaN steps are skipped per cell.

    Parameters
    ----------
    files : list
        NetCDF files to stream through.
    input_var : str
        Variable to read.
    factor : float
        Unit conversion factor applied to every time step.
    coordinate_slice : dict, optional
        Lat/lon slices to crop each time step to.
    with_period_series : bool
        Also accumulate one field per calendar year and month, which gives the
        monthly means the spatial metrics need without ever holding the record.
    remove_seasonal_cycle_from_std : bool
        Keep the square sums per calendar month, which the std of the
        anomalies from the monthly climatology needs. Otherwise one total.
    available_years : list of int, optional
        Years to read; steps of other years are skipped.
    season_months : list of int, optional
        Calendar months to read; steps of other months are skipped.

    Returns
    -------
    step_count : int
        Number of time steps read.
    monthly_sums : ndarray
        Sum per calendar month, shape (12, ...).
    square_sums : ndarray
        Sum of squares, float64, per calendar month (12, ...) or in total (1, ...).
    monthly_counts : ndarray
        Valid-count per calendar month, shape (12, ...).
    period_sums : dict
        Sum per (year, month) key, empty unless `with_period_series`.
    period_counts : dict
        Valid-count per (year, month) key, empty unless `with_period_series`.
    """
    da = None
    if not isinstance(files, Iterable):
        # logger.warning(f"Files not a list of files but one file {files}.")
        files = [files]
    files = list(files)
    if len(files) == 0:
        msg = "Received an empty file subset for one-pass statistics."
        with ErrorLogger(logger):
            raise ValueError(msg)
    logger.debug(files)
    with get_xarray_ds_from_file(
        files[0], var_name=input_var, engine="netcdf4", force_decending_y=True
    ) as ds:
        # Apply coordinate slicing if needed
        if coordinate_slice is not None:
            lat_key = get_coord_key(ds, lat=True)
            lon_key = get_coord_key(ds, lon=True)
            da = ds.sel(
                {lat_key: coordinate_slice["lat"], lon_key: coordinate_slice["lon"]}
            )[input_var]
        else:
            da = ds[input_var]
    step_count = 0
    grid_shape = da.shape[1:]
    monthly_sums = np.zeros((12, *grid_shape))
    # only the std without the seasonal cycle needs the squares per month
    square_sums = np.zeros((12 if remove_seasonal_cycle_from_std else 1, *grid_shape))
    monthly_counts = np.zeros((12, *grid_shape), dtype=np.uint32)
    period_sums = {}
    period_counts = {}
    for f, file in enumerate(files):
        with get_xarray_ds_from_file(
            file, var_name=input_var, engine="netcdf4", force_decending_y=True
        ) as ds:
            logger.info(
                f"timestep {step_count} in file {f + 1} / {len(files)} from {file}"
            )
            if coordinate_slice is not None:
                lat_key = get_coord_key(ds, lat=True)
                lon_key = get_coord_key(ds, lon=True)
                da = ds.sel(
                    {lat_key: coordinate_slice["lat"], lon_key: coordinate_slice["lon"]}
                )[input_var]
            else:
                da = ds[input_var]
            # selected per step, so the years also apply to directories without
            # year folders and to files holding several years
            da = select_time_steps_by_year_and_month(da, available_years, season_months)
            if da.sizes["time"] == 0:
                continue
            for _time_value, sub in da.groupby("time", squeeze=False):
                data_slice = sub.isel(time=0)
                try:
                    step_count += 1
                    month = int(data_slice.time.dt.month.item()) - 1
                    values = data_slice.values * factor
                    is_valid = ~np.isnan(values)
                    filled = np.where(is_valid, values, 0)
                    monthly_sums[month] += filled
                    square_index = month if remove_seasonal_cycle_from_std else 0
                    square_sums[square_index] += np.square(filled, dtype=np.float64)
                    monthly_counts[month] += is_valid
                    if with_period_series:
                        # one bucket per calendar month of the record, so the
                        # monthly means fall out of this same pass
                        key = (int(data_slice.time.dt.year.item()), month + 1)
                        if key not in period_sums:
                            # a bucket only ever holds one month, so float32
                            # sums and uint16 counts carry the record at a
                            # quarter of the float64 footprint
                            period_sums[key] = np.zeros(values.shape, dtype=np.float32)
                            period_counts[key] = np.zeros(values.shape, dtype=np.uint16)
                        period_sums[key] += filled
                        period_counts[key] += is_valid
                except Exception as e:
                    logger.error(data_slice)
                    with ErrorLogger(logger):
                        raise e
    logger.debug(f"Read {step_count} time steps from {len(files)} files.")
    return (
        step_count,
        monthly_sums,
        square_sums,
        monthly_counts,
        period_sums,
        period_counts,
    )


def create_period_mean_series(period_sums, period_counts, lat, lon, var, units=None):
    """Create a monthly mean time series from streamed per-month buckets.

    Parameters
    ----------
    period_sums, period_counts : dict
        Sum and valid-count fields keyed by ``(year, month)``. Emptied in
        place, so the caller must not read them again.
    lat, lon : array-like
        Coordinate values of the streamed grid.
    var : str
        Source variable name, used for the long name.
    units : str, optional
        Units copied onto the series.

    Returns
    -------
    xarray.Dataset
        Dataset with a single ``time_series`` variable on a monthly time axis.
    """
    keys = sorted(period_sums)
    # filled bucket by bucket, popping each one once it has been divided, so
    # the record is never stacked into a second full copy
    means = np.full((len(keys), *period_sums[keys[0]].shape), np.nan, dtype=np.float32)
    for index, key in enumerate(keys):
        sums, counts = period_sums.pop(key), period_counts.pop(key)
        np.divide(sums, counts, out=means[index], where=counts > 0)
    times = pd.to_datetime([f"{year:04d}-{month:02d}-01" for year, month in keys])
    series = xr.DataArray(
        means,
        coords={"time": times, "lat": lat, "lon": lon},
        dims=["time", "lat", "lon"],
        name="time_series",
    )
    series.attrs = {"long_name": f"Monthly mean of {var}"}
    if units is not None:
        series.attrs["units"] = units
    return series.to_dataset()


def create_std_long_name(var, remove_seasonal_cycle_from_std):
    """Create the long name of a temporal standard deviation.

    Args:
        var (str): Name of the source variable.
        remove_seasonal_cycle_from_std (bool): Whether the std was taken from
            the anomalies of the monthly climatology.

    Returns
    -------
        The long name, stating whether the seasonal cycle is included.
    """
    if remove_seasonal_cycle_from_std:
        return (
            f"Temporal standard deviation of {var} anomalies "
            "from the monthly climatology"
        )
    return f"Temporal standard deviation of {var} incl. the seasonal cycle"


def create_seasonality_label(remove_seasonal_cycle):
    """Create the label stating whether the seasonal cycle was removed.

    It ends the std and Spearman panel titles and is stored as `seasonality`
    attribute of `rel_std` and `spearman`. The correlation of the climatologies
    uses `CLIMATOLOGY_SEASONALITY_LABEL` instead.

    Args:
        remove_seasonal_cycle (bool): Whether the anomalies from the monthly
            climatology were used.

    Returns
    -------
        "deseasonalised" or "with seasonal cycle".
    """
    return "deseasonalised" if remove_seasonal_cycle else "with seasonal cycle"


def create_region_file_suffix(region_name):
    """Create the file name suffix of a figure of one WMO region.

    Args:
        region_name (str or None): WMO region, None for the whole domain.

    Returns
    -------
        "_region_<region>" with a file-safe region name, "" for the domain.
    """
    return f"_region_{sanitize_name(region_name)}" if region_name else ""


def get_stats_one_pass(
    path,
    var,
    factor=1,
    coordinate_slice=None,
    ncpus=1,
    n_bootstrap_years=None,
    bootstrap_index=None,
    output_path=None,
    available_years=None,
    file_name="*.*",
    with_period_series=False,
    compression=None,
    remove_seasonal_cycle_from_std=False,
    season_months=None,
):
    """Compute streaming statistics from monthly/yearly files.

    Reads one file at a time and accumulates per-month sums, square sums and
    counts, from which the mean, standard deviation and monthly climatology
    follow. NaN steps are skipped per cell, as in `get_file_stats`. Optionally
    slices coordinates, applies a multiplicative factor, supports bootstrapping
    over years, and can write the result to disk.

    With `remove_seasonal_cycle_from_std` the std is that of the anomalies from
    the monthly climatology (ddof=0), otherwise that of the raw values (ddof=0).
    Only steps in `available_years` and `season_months` are used.

    With `with_period_series` the same pass also returns a monthly
    `time_series`, so the spatial metrics can be calculated without the record
    ever being held in memory.
    """
    files = []
    if path.is_dir():
        files = get_files(
            path,
            n_bootstrap_years=n_bootstrap_years,
            available_years=available_years,
            file_name=file_name,
        )
    elif path.is_file():
        files = [path]
    if len(files) == 0:
        msg = (
            f"No input files found for statistics at {path} "
            f"(file pattern: {file_name}, years: {available_years})."
        )
        with ErrorLogger(logger):
            raise FileNotFoundError(msg)
    logger.debug(f"List of files: {files}")
    file_subsets = split_file_list(files, ncpus) if ncpus > 1 else [files]
    logger.info("creating statistics one pass...")
    subset_results = Parallel(n_jobs=ncpus, backend="loky")(
        delayed(get_stats_one_pass_subset)(
            file_subset,
            var,
            factor,
            coordinate_slice,
            with_period_series,
            remove_seasonal_cycle_from_std,
            available_years,
            season_months,
        )
        for file_subset in file_subsets
    )
    logger.info("combining results...")
    (
        step_count,
        monthly_sums,
        square_sums,
        monthly_counts,
        period_sums,
        period_counts,
    ) = combine_results(subset_results)
    logger.debug(f"Combined {step_count} time steps.")
    valid_counts = monthly_counts.sum(axis=0)
    total_sums = monthly_sums.sum(axis=0)
    mean = np.divide(
        total_sums,
        valid_counts,
        out=np.full(total_sums.shape, np.nan),
        where=valid_counts > 0,
    )
    climatology = np.divide(
        monthly_sums,
        monthly_counts,
        out=np.full(monthly_sums.shape, np.nan),
        where=monthly_counts > 0,
    )
    if remove_seasonal_cycle_from_std:
        # squared anomalies of a month: S2 - 2*clim*S1 + n*clim^2 = S2 - S1*clim,
        # and the anomalies add up to zero, so their mean needs no subtraction
        squared_deviations = np.nansum(square_sums - monthly_sums * climatology, axis=0)
    else:
        squared_deviations = square_sums.sum(axis=0) - total_sums * mean
    # clipped, because rounding can push a variance of zero just below it
    variance = np.divide(
        np.maximum(squared_deviations, 0),
        valid_counts,
        out=np.full(total_sums.shape, np.nan),
        where=valid_counts > 0,
    )
    std_dev = np.sqrt(variance)
    with get_dataset_from_path(
        files[0],
        var_name=var,
        engine="netcdf4",
        force_decending_y=True,
    ) as ds_in:
        lat_key = get_coord_key(ds_in, lat=True)
        lon_key = get_coord_key(ds_in, lon=True)
        try:
            spatial_resolution = _spatial_resolution(ds_in, lon_key, lat_key)
        except ValueError:
            spatial_resolution = None
        # Apply coordinate slicing if needed
        ds = (
            ds_in.sel(
                {lat_key: coordinate_slice["lat"], lon_key: coordinate_slice["lon"]}
            )
            if coordinate_slice is not None
            else ds_in
        )
        lat = get_coord_values(ds, lat=True)
        lon = get_coord_values(ds, lon=True)
        source_units = ds[var].attrs.get("units") if var in ds else None
    # Calculate climatology and standard deviation along the time dimension
    # Construct the output dataset with lazy evaluations
    # climatology = climatology.rename({get_coord_key(climatology, lat=True): "lat", get_coord_key(climatology, lon=True): "lon"})

    std = xr.DataArray(std_dev, coords={"lat": lat, "lon": lon}, dims=["lat", "lon"])
    clim = xr.DataArray(
        climatology,
        coords={"month": np.arange(1, 13, 1), "lat": lat, "lon": lon},
        dims=["month", "lat", "lon"],
    )
    mean = xr.DataArray(mean, coords={"lat": lat, "lon": lon}, dims=["lat", "lon"])
    clim.attrs = {"long_name": f"Monthly climatology of {var}"}
    std.attrs = {"long_name": create_std_long_name(var, remove_seasonal_cycle_from_std)}
    mean.attrs = {"long_name": f"Temporal mean of {var}"}
    if source_units is not None:
        clim.attrs["units"] = source_units
        std.attrs["units"] = source_units
        mean.attrs["units"] = source_units
    output = xr.Dataset(
        {"clim": clim, "std": std, "mean": mean},
        coords={"month": np.arange(1, 13, 1), "lat": lat, "lon": lon},
    )
    output = generate_bounds_for_all_coords(output, res=spatial_resolution)
    if period_sums:
        output = xr.merge(
            [
                output,
                create_period_mean_series(
                    period_sums, period_counts, lat, lon, var, source_units
                ),
            ]
        )
    # Trigger computation if needed
    if output_path is not None:
        output_file = (
            output_path.parent / f"{output_path.stem}_{bootstrap_index}.nc"
            if bootstrap_index is not None
            else output_path
        )
        logger.info(f"Writing output to {output_file}")
        write_xarray_to_file(ds=output, file_path=output_file, compression=compression)
    return output


def resample_to_coarser_calendar(
    ds_input: xr.Dataset, ds_ref: xr.Dataset
) -> Tuple[xr.Dataset, xr.Dataset]:
    """Resample the higher-resolution dataset to the coarser calendar.

    Compare the two datasets' median time steps, convert them to pandas/xarray
    frequency aliases, and resample the finer one up to the coarser one using
    calendar-aware frequencies (e.g., 'ME' not '720H').
    """
    hours_in, alias_in = timedelta_to_alias(ds_input)
    hours_ref, alias_ref = timedelta_to_alias(ds_ref)
    if hours_in > hours_ref:
        # input is coarser (e.g. monthly) → bring ref up to that
        target_alias = alias_in
    elif hours_ref > hours_in:
        # ref is coarser → bring input up to that
        target_alias = alias_ref
    else:
        # same resolution, nothing to do
        target_alias = alias_in
        logger.info(f"Both are already {target_alias}")
    return resample_to_target_freq(ds_input, ds_ref, target_alias)


def crop_data_to_overlapping_time(input_ds, ref_ds):
    """Crop data to overlapping time."""
    time_slice = get_overlapping_time_slice(input_ds, ref_ds)
    # Slice both datasets to that time range
    logger.debug(f"Input time before crop: {input_ds.time}")
    input_ds = input_ds.sel(time=time_slice)
    logger.debug(f"Input time after crop: {input_ds.time}")
    logger.debug(f"Ref time before crop: {ref_ds.time}")
    ref_ds = ref_ds.sel(time=time_slice)
    logger.debug(f"Ref time after crop: {ref_ds.time}")
    # Normalize anchors at the current resolution to avoid mismatches.
    # _, alias = timedelta_to_alias(input_ds)
    # input_ds = normalize_time_axis(input_ds, alias)
    # ref_ds = normalize_time_axis(ref_ds, alias)
    # # Ensure identical time axis after slicing (e.g. monthly midpoints can differ)
    # input_ds, ref_ds = xr.align(input_ds, ref_ds, join="inner")
    return input_ds, ref_ds, time_slice


@log_errors(raise_exceptions=True)
def plot_map(
    rel_mean,
    rel_std,
    spearman,
    ref_clim,
    input_clim,
    input_name,
    ref_name,
    output_path,
    overlapping_years=None,
    season_months=None,
    std_seasonality=None,
    spearman_seasonality=None,
    region_name=None,
):
    """Create a 2x2 figure of relative mean, relative std, Spearman correlation, and seasonal means.

    Generates four subplots: (1) relative temporal mean, (2) relative temporal
    standard deviation, (3) Spearman correlation, of the (by default
    deseasonalised) time series or, without a direct comparison, of the
    climatologies, and (4) monthly seasonal means for both datasets.
    `season_months` names the compared season in the title, `std_seasonality`
    and `spearman_seasonality` (see `create_seasonality_label`) state in the
    panel titles whether the seasonal cycle was removed.
    `region_name` names the WMO region in the title and the file names.
    """
    rel_mean = np.where(rel_mean == np.inf, np.nan, rel_mean)
    rel_mean = np.where(rel_mean < 0, np.nan, rel_mean)
    rel_std = np.where(rel_std == np.inf, np.nan, rel_std)
    rel_std = np.where(rel_std < 0, np.nan, rel_std)
    fig, axes = plt.subplots(2, 2, figsize=(10.5, 4.68))
    title = create_comparison_title(
        input_name, ref_name, overlapping_years, season_months
    ) + format_region_title(region_name)
    fig.suptitle(title, fontweight="normal", fontsize="x-large")

    # the ratios look like the alpha and beta maps of discharge-evaluation
    im0, bounds0, extend0, ticks0 = plot_single_map(
        axes[0, 0], rel_mean, bounds_type="data", **get_metric_plot_style("beta")
    )
    axes[0, 0].set_title(
        f"Relative temporal Mean (median={np.nanmedian(rel_mean):.2f})"
    )
    im1, bounds1, extend1, ticks1 = plot_single_map(
        axes[0, 1], rel_std, bounds_type="data", **get_metric_plot_style("alpha")
    )
    std_label = f", {std_seasonality}" if std_seasonality else ""
    axes[0, 1].set_title(
        f"Relative temporal Standarddeviation{std_label} "
        f"(median={np.nanmedian(rel_std):.2f})"
    )

    correlation_style = get_metric_plot_style("spearman")
    if np.nanmin(spearman) < 0:
        # the scale runs from 0 to 1 and every negative correlation gets the
        # under colour, instead of the scale following them below 0
        correlation_style.update(vmin=0.0, under_color=KGE_UNDER_COLOR)
    im2, bounds2, extend2, ticks2 = plot_single_map(
        axes[1, 0], spearman, bounds_type="data", **correlation_style
    )

    spearman_label = f", {spearman_seasonality}" if spearman_seasonality else ""
    axes[1, 0].set_title(
        f"Spearman Correlation{spearman_label} (median={np.nanmedian(spearman):.2f})"
    )

    # Plot for the seasonality
    months = np.arange(1, 13, 1)
    bar_width = 0.4
    axes[1, 1].bar(
        months - bar_width / 2,
        np.nanmean(ref_clim, axis=(1, 2)),
        width=bar_width,
        color="#008176",
        label=ref_name,
        alpha=0.8,
    )
    axes[1, 1].bar(
        months + bar_width / 2,
        np.nanmean(input_clim, axis=(1, 2)),
        width=bar_width,
        color="#79A3E6",
        label=input_name,
        alpha=0.8,
    )

    ax_twy = axes[1, 1].twinx()
    ref_clim = np.where(ref_clim != 0, ref_clim, np.nan)
    rel_clim = np.nanmean(input_clim, axis=(1, 2)) / np.nanmean(ref_clim, axis=(1, 2))
    rel_clim_diff_1 = max(
        np.abs(1 - np.nanmin(rel_clim)), np.abs(1 - np.nanmax(rel_clim))
    )
    ax_twy.errorbar(
        months, rel_clim, label=f"{input_name}/{ref_name}", color="#0000A7", fmt="--"
    )
    ax_twy.axhline(y=1, color="#0000A7", linewidth=0.5)
    axes[1, 1].set_xlabel("month of year")
    handles, labels = [], []
    for ax in [axes[1, 1], ax_twy]:
        for handle, label in zip(*ax.get_legend_handles_labels()):
            handles.append(handle)
            labels.append(label)

    axes[1, 1].legend(handles, labels, loc="upper right")
    axes[1, 1].set_title("Seasonality")
    axes[1, 1].set_ylabel("ET [mm/day]")
    axes[1, 1].tick_params(axis="y", labelcolor="black")
    axes[1, 1].set_xlim(1 - (1.1 * bar_width), 12 + (1.1 * bar_width))
    axes[1, 1].set_xticks(months)
    axes[1, 1].set_xticklabels(months)
    ymax = 1 + rel_clim_diff_1 * 1.05 if not np.isnan(rel_clim_diff_1) else 1
    ax_twy.set_ylim(
        np.nanmax([0, 1 - rel_clim_diff_1 * 1.05]), ymax
    )  # Example range for the ratio
    ax_twy.set_ylabel("Ratio (Input / Reference)", color="#0000A7")
    ax_twy.tick_params(axis="y", labelcolor="#0000A7")

    ratio_label = create_axis_label(f"{input_name} / {ref_name}")
    add_map_colorbar(fig, axes[0, 0], im0, bounds0, extend0, ticks0, ratio_label)
    add_map_colorbar(fig, axes[0, 1], im1, bounds1, extend1, ticks1, ratio_label)
    add_map_colorbar(
        fig, axes[1, 0], im2, bounds2, extend2, ticks2, create_axis_label("rho")
    )

    for ax in axes.flat:
        for spine in ax.spines.values():
            spine.set_linewidth(0.5)
    axes[1, 1].spines["top"].set_linewidth(0.25)
    for ax in [axes[0, 0], axes[1, 0], axes[0, 1]]:
        ax.set_yticks([])
        ax.set_xticks([])
        ax.yaxis.labelpad = 0
        for spine in ax.spines.values():
            spine.set_linewidth(0.25)
    plt.tight_layout()
    region_suffix = create_region_file_suffix(region_name)
    file_name = f"et_map_{input_name}_{ref_name}{region_suffix}.png"
    file_name = file_name.replace(" ", "_")
    plt.savefig(output_path / file_name, dpi=800)
    plt.close(fig)
    logger.info(f"created et_map {output_path / file_name}")


@log_errors(raise_exceptions=True)
def plot_map_bias_only(
    rel_mean,
    ref_clim,
    input_clim,
    input_name,
    ref_name,
    output_path,
    overlapping_years=None,
    total_rel_mean=None,
    season_months=None,
    region_name=None,
):
    """Create a 3-panel plot with relative mean, mean difference, and climatology.

    `season_months` names the compared season in the title.
    `region_name` names the WMO region in the title and the file names.
    """
    rel_mean = np.where(rel_mean == np.inf, np.nan, rel_mean)
    ref_mean_map = np.asarray(np.nanmean(ref_clim, axis=0))
    input_mean_map = np.asarray(np.nanmean(input_clim, axis=0))
    diff_mean = input_mean_map - ref_mean_map
    diff_mean = np.where(diff_mean == np.inf, np.nan, diff_mean)

    fig = plt.figure(figsize=(10.5, 5.6))
    gs = fig.add_gridspec(2, 4, height_ratios=[1.0, 0.5])
    ax_rel_mean = fig.add_subplot(gs[0, 0:2])
    ax_diff_mean = fig.add_subplot(gs[0, 2:4])
    ax_clim = fig.add_subplot(gs[1, 1:3])

    title = create_comparison_title(
        input_name, ref_name, overlapping_years, season_months
    ) + format_region_title(region_name)
    fig.suptitle(title, fontweight="normal", fontsize="x-large")

    # the ratio looks like the beta maps of discharge-evaluation
    im0, bounds0, extend0, ticks0 = plot_single_map(
        ax_rel_mean, rel_mean, bounds_type="data", **get_metric_plot_style("beta")
    )
    title_rel_mean = f"a) Relative Mean (median={np.nanmedian(rel_mean):.2f}"
    if total_rel_mean is not None:
        title_rel_mean += f", mean={total_rel_mean:.2f}"
    title_rel_mean += ")"
    ax_rel_mean.set_title(title_rel_mean)

    # diff_abs_max = int(round(np.nanmax(np.abs(diff_mean)) + 0.5))
    # if not np.isfinite(diff_abs_max) or diff_abs_max == 0:
    #     diff_abs_max = 1.0
    vmin = np.nanquantile(diff_mean, 0.01)
    vmax = np.nanquantile(diff_mean, 0.99)
    diff_diff_1 = max(abs(vmin), abs(vmax))
    if diff_diff_1 > 1.4:
        # round to next 0.5
        diff_diff_1 = round((diff_diff_1 * 2) / 2) / 4 * 4.5
    else:
        # round to next 0.2
        diff_diff_1 = round((diff_diff_1 * 5) / 5) / 4 * 4.5
    im1, bounds1, extend1, ticks1 = plot_single_map(
        ax_diff_mean,
        diff_mean,
        vmin=-diff_diff_1,
        vmax=diff_diff_1,
        center=0,
        bounds_type="max",
    )
    ax_diff_mean.set_title(
        f"b) Difference (median={np.nanmedian(diff_mean):.2f}mm/day, mean={np.nanmean(diff_mean):.2f}mm/day)"
    )

    months = np.arange(1, 13, 1)
    bar_width = 0.4
    ref_monthly_mean = np.asarray(np.nanmean(ref_clim, axis=(1, 2)))
    input_monthly_mean = np.asarray(np.nanmean(input_clim, axis=(1, 2)))
    ax_clim.bar(
        months - bar_width / 2,
        ref_monthly_mean,
        width=bar_width,
        color="#008176",
        label=ref_name,
        alpha=0.8,
    )
    ax_clim.bar(
        months + bar_width / 2,
        input_monthly_mean,
        width=bar_width,
        color="#79A3E6",
        label=input_name,
        alpha=0.8,
    )

    ax_twy = ax_clim.twinx()
    ref_clim = np.where(ref_clim != 0, ref_clim, np.nan)
    rel_clim = input_monthly_mean / np.asarray(np.nanmean(ref_clim, axis=(1, 2)))
    rel_clim_diff_1 = max(
        np.abs(1 - np.nanmin(rel_clim)), np.abs(1 - np.nanmax(rel_clim))
    )
    ax_twy.errorbar(
        months, rel_clim, label=f"{input_name}/{ref_name}", color="#0000A7", fmt="--"
    )
    ax_twy.axhline(y=1, color="#0000A7", linewidth=0.5)
    ax_clim.set_xlabel("month of year")
    handles, labels = [], []
    for ax in [ax_clim, ax_twy]:
        for handle, label in zip(*ax.get_legend_handles_labels()):
            handles.append(handle)
            labels.append(label)

    ax_clim.legend(handles, labels, loc="upper right")
    ax_clim.set_title("c) Climatology")
    ax_clim.set_ylabel("ET [mm/day]")
    ax_clim.tick_params(axis="y", labelcolor="black")
    ax_clim.set_xlim(1 - (1.1 * bar_width), 12 + (1.1 * bar_width))
    ax_clim.set_xticks(months)
    ax_clim.set_xticklabels(months)
    ymax = 1 + rel_clim_diff_1 * 1.05 if not np.isnan(rel_clim_diff_1) else 1
    ax_twy.set_ylim(
        np.nanmax([0, 1 - rel_clim_diff_1 * 1.05]), ymax
    )  # Example range for the ratio
    ax_twy.set_ylabel("Ratio (Input / Reference)", color="#0000A7")
    ax_twy.tick_params(axis="y", labelcolor="#0000A7")

    add_map_colorbar(
        fig,
        ax_rel_mean,
        im0,
        bounds0,
        extend0,
        ticks0,
        create_axis_label(f"{input_name} / {ref_name}"),
    )

    divider1 = make_axes_locatable(ax_diff_mean)
    cax1 = divider1.append_axes("right", size="5%", pad=0.1)
    fig.colorbar(
        im1,
        cax=cax1,
        label="mm/day",
        boundaries=bounds1,
        extend=extend1,
        ticks=ticks1,
    )

    for ax in [ax_rel_mean, ax_diff_mean, ax_clim]:
        for spine in ax.spines.values():
            spine.set_linewidth(0.5)
    ax_clim.spines["top"].set_linewidth(0.25)
    for ax in [ax_rel_mean, ax_diff_mean]:
        ax.set_yticks([])
        ax.set_xticks([])
        ax.yaxis.labelpad = 0
    for spine in ax_rel_mean.spines.values():
        spine.set_linewidth(0.25)
    for spine in ax_diff_mean.spines.values():
        spine.set_linewidth(0.25)
    plt.tight_layout()

    region_suffix = create_region_file_suffix(region_name)
    file_name = f"et_map_bias_only_{input_name}_{ref_name}{region_suffix}.png"
    file_name = file_name.replace(" ", "_")
    plt.savefig(output_path / file_name, dpi=800)
    plt.close(fig)
    logger.info(f"created et_map {output_path / file_name}")


@log_errors(raise_exceptions=True)
def plot_map_global_climate(
    rel_mean,
    rel_std,
    input_name,
    ref_name,
    output_path,
    overlapping_years=None,
    season_months=None,
    region_name=None,
):
    """Create a plot showing relative mean, relative std and seasonality.

    `season_months` names the compared season in the title.
    `region_name` names the WMO region in the title and the file names.
    """
    rel_mean = np.where(rel_mean == np.inf, np.nan, rel_mean)
    rel_std = np.where(rel_std == np.inf, np.nan, rel_std)
    fig, axes = plt.subplots(2, 1, figsize=(10.5, 6.2))
    title = create_comparison_title(
        input_name, ref_name, overlapping_years, season_months
    ) + format_region_title(region_name)
    fig.suptitle(title, fontweight="normal", fontsize="x-large")

    mean_diff_1 = max(np.abs(1 - np.nanmin(rel_mean)), np.abs(1 - np.nanmax(rel_mean)))
    im0, bounds0, extend0, ticks0 = plot_single_map(
        axes[0], rel_mean, mean_diff_1, bounds_type="fixed"
    )
    axes[0].set_title(f"Relative temporal Mean (median={np.nanmedian(rel_mean):.2f})")

    std_diff_1 = max(np.abs(1 - np.nanmin(rel_std)), np.abs(1 - np.nanmax(rel_std)))
    im1, bounds1, extend1, ticks1 = plot_single_map(
        axes[1], rel_std, std_diff_1, bounds_type="quantiles"
    )
    axes[1].set_title(
        f"Relative temporal Standarddeviation (median={np.nanmedian(rel_std):.2f})"
    )

    divider0 = make_axes_locatable(axes[0])
    cax0 = divider0.append_axes("right", size="5%", pad=0.1)
    fig.colorbar(
        im0, cax=cax0, label="", boundaries=bounds0, extend=extend0, ticks=ticks0
    )

    divider1 = make_axes_locatable(axes[1])
    cax1 = divider1.append_axes("right", size="5%", pad=0.1)
    fig.colorbar(
        im1, cax=cax1, label="", boundaries=bounds1, extend=extend1, ticks=ticks1
    )

    for ax in axes.flat:
        for spine in ax.spines.values():
            spine.set_linewidth(0.5)
    for ax in [axes[0], axes[1]]:
        ax.set_yticks([])
        ax.set_xticks([])
        ax.yaxis.labelpad = 0
        for spine in ax.spines.values():
            spine.set_linewidth(0.25)
    plt.tight_layout()
    region_suffix = create_region_file_suffix(region_name)
    file_name = f"et_map_global_climate_{input_name}_{ref_name}{region_suffix}.png"
    file_name = file_name.replace(" ", "_")
    plt.savefig(output_path / file_name, dpi=800)
    plt.close(fig)
    logger.info(f"created et_map {output_path / file_name}")


@log_errors(raise_exceptions=True)
def plot_map_global_climate2(
    rel_mean,
    diff_mean,
    input_name,
    ref_name,
    output_path,
    overlapping_years=None,
    total_rel_mean=None,
    season_months=None,
    region_name=None,
):
    """Create a plot showing relative mean and monthly climatology difference.

    `season_months` names the compared season in the title.
    `region_name` names the WMO region in the title and the file names.
    """
    rel_mean = np.where(rel_mean == np.inf, np.nan, rel_mean)
    diff_mean = np.asarray(np.where(diff_mean == np.inf, np.nan, diff_mean))
    fig = plt.figure(figsize=(10.5, 6.2))
    gs = fig.add_gridspec(2, 4, height_ratios=[1.0, 1.0])
    ax_rel_mean = fig.add_subplot(gs[1, 1:3])
    ax_diff = fig.add_subplot(gs[0, 1:3])
    title = create_comparison_title(
        input_name, ref_name, overlapping_years, season_months
    ) + format_region_title(region_name)
    fig.suptitle(title, fontweight="normal", fontsize="x-large")

    im0, bounds0, extend0, ticks0 = plot_single_map(
        ax_rel_mean, rel_mean, bounds_type="percent"
    )
    title_rel_mean = f"b) Relative Mean Difference (median={np.nanmedian(rel_mean):.1f}"
    if total_rel_mean is not None:
        title_rel_mean += f", mean={total_rel_mean:.1f}"
    title_rel_mean += ")"
    ax_rel_mean.set_title(title_rel_mean)

    vmin = np.nanquantile(diff_mean, 0.01)
    vmax = np.nanquantile(diff_mean, 0.99)
    diff_diff_1 = max(abs(vmin), abs(vmax))
    diff_diff_1, round_dec = round_sensibly(diff_diff_1)
    logger.debug(
        f"diff_diff_1={diff_diff_1}, vmin={vmin}, vmax={vmax}, round_dec={round_dec}"
    )
    im1, bounds1, extend1, ticks1 = plot_single_map(
        ax_diff, diff_mean, diff_diff_1, center=0, bounds_type="max"
    )
    ax_diff.set_title(
        f"a) Mean Difference (median={np.nanmedian(diff_mean):.{round_dec}f}mm/day, mean={np.nanmean(diff_mean):.{round_dec}f}mm/day)"
    )
    divider1 = make_axes_locatable(ax_diff)
    cax1 = divider1.append_axes("right", size="5%", pad=0.1)
    fig.colorbar(
        im1, cax=cax1, label="mm/day", boundaries=bounds1, extend=extend1, ticks=ticks1
    )

    divider0 = make_axes_locatable(ax_rel_mean)
    cax0 = divider0.append_axes("right", size="5%", pad=0.1)
    cbar0 = fig.colorbar(
        im0, cax=cax0, label="%", boundaries=bounds0, extend=extend0, ticks=ticks0
    )
    # shortest form of every tick, so 25 and 2.5 instead of 25.0 and 2.5
    cbar0.set_ticks(ticks0, labels=[f"{tick + 0.0:g}" for tick in ticks0])

    for ax in [ax_rel_mean, ax_diff]:
        for spine in ax.spines.values():
            spine.set_linewidth(0.5)
    ax_rel_mean.set_yticks([])
    ax_rel_mean.set_xticks([])
    ax_rel_mean.yaxis.labelpad = 0
    for spine in ax_rel_mean.spines.values():
        spine.set_linewidth(0.25)
    if diff_mean.ndim != 1:
        ax_diff.set_yticks([])
        ax_diff.set_xticks([])
        ax_diff.yaxis.labelpad = 0
        for spine in ax_diff.spines.values():
            spine.set_linewidth(0.25)
    plt.tight_layout()

    region_suffix = create_region_file_suffix(region_name)
    file_name = f"et_map_global_climate2_{input_name}_{ref_name}{region_suffix}.png"
    file_name = file_name.replace(" ", "_")
    plt.savefig(output_path / file_name, dpi=800)
    plt.close(fig)
    logger.info(f"created et_map {output_path / file_name}")


@log_errors(raise_exceptions=True)
def plot_map_local_climate(
    input_clim,
    ref_clim,
    input_name,
    ref_name,
    output_path,
    overlapping_years=None,
    season_months=None,
    region_name=None,
):
    """Create monthly local-climate maps for relative and absolute differences.

    Produces two figures with 12 small maps (one per month):
    - Relative climatology difference in percent
    - Absolute climatology difference in native units (e.g. mm/day)
    `season_months` names the compared season in the title.
    `region_name` names the WMO region in the title and the file names.
    """
    month_labels = (
        "Jan",
        "Feb",
        "Mar",
        "Apr",
        "May",
        "Jun",
        "Jul",
        "Aug",
        "Sep",
        "Oct",
        "Nov",
        "Dec",
    )
    input_clim = np.asarray(input_clim)
    ref_clim = np.asarray(ref_clim)
    if input_clim.shape != ref_clim.shape or input_clim.ndim != 3:
        msg = (
            "input_clim and ref_clim must both be 3D arrays with equal shape "
            "(month, lat, lon). "
            f"Got {input_clim.shape} and {ref_clim.shape}."
        )
        with ErrorLogger(logger):
            raise ValueError(msg)
    if input_clim.shape[0] != 12:
        msg = (
            "Monthly local-climate plotting expects exactly 12 climatology slices. "
            f"Got {input_clim.shape[0]}."
        )
        with ErrorLogger(logger):
            raise ValueError(msg)

    # Avoid unstable percentages where reference climatology is close to zero.
    ref_safe = np.where(np.abs(ref_clim) < 0.1, np.nan, ref_clim)
    rel_monthly = 100.0 * (input_clim - ref_safe) / ref_safe
    abs_monthly = input_clim - ref_clim

    def _plot_monthly_panels(
        values, panel_title, output_file_name, colorbar_label, bounds_type="max"
    ):
        values = np.where(values == np.inf, np.nan, values)
        values = np.where(values == -np.inf, np.nan, values)
        finite_values = values[np.isfinite(values)]
        if finite_values.size == 0:
            diff_to_mean = 1e-6
        elif bounds_type == "percent":
            # the months share one colorbar, so their bins come from all months
            diff_to_mean = float(get_percent_diff_bounds(finite_values)[-1])
        else:
            q01 = float(np.nanquantile(finite_values, 0.01))
            q99 = float(np.nanquantile(finite_values, 0.99))
            diff_to_mean = max(abs(q01), abs(q99))
            diff_to_mean, _ = round_sensibly(diff_to_mean)

        fig = plt.figure(figsize=(14.0, 8.5))
        gs = fig.add_gridspec(
            3,
            5,
            width_ratios=[1.0, 1.0, 1.0, 1.0, 0.085],
            wspace=0.06,
            hspace=0.18,
        )
        axes = np.array(
            [[fig.add_subplot(gs[i, j]) for j in range(4)] for i in range(3)]
        )
        cax = fig.add_subplot(gs[:, 4])
        if input_name is not None and ref_name is not None:
            title = create_comparison_title(
                input_name, ref_name, overlapping_years, season_months
            )
        else:
            title = "Local Climate Comparison"
        title += format_region_title(region_name)
        fig.suptitle(f"{title}\n{panel_title}", fontsize="x-large", fontweight="normal")

        im = None
        bounds = None
        extend = None
        ticks = None
        for month_idx, ax in enumerate(axes.flat):
            month_values = values[month_idx]
            im, bounds, extend, ticks = plot_single_map(
                ax,
                month_values,
                diff_to_mean=diff_to_mean,
                center=0,
                bounds_type=bounds_type,
            )
            ax.set_title(month_labels[month_idx], fontsize="medium")
            ax.set_xticks([])
            ax.set_yticks([])
            ax.yaxis.labelpad = 0
            for spine in ax.spines.values():
                spine.set_linewidth(0.25)

        cbar = fig.colorbar(
            im,
            cax=cax,
            boundaries=bounds,
            extend=extend,
            ticks=ticks,
            label=colorbar_label,
        )
        # shortest form of every tick, so 25 and 2.5 instead of 25.0 and 2.5
        cbar.set_ticks(ticks, labels=[f"{tick + 0.0:g}" for tick in ticks])
        cbar.ax.tick_params(labelsize=8)
        fig.subplots_adjust(left=0.03, right=0.94, bottom=0.04, top=0.89)
        plt.savefig(output_path / output_file_name, dpi=400)
        plt.close(fig)
        logger.info(f"created et_map {output_path / output_file_name}")

    rel_file_name = (
        f"et_map_local_climate_rel_{input_name}_{ref_name}"
        f"{create_region_file_suffix(region_name)}.png"
    ).replace(" ", "_")
    abs_file_name = (
        f"et_map_local_climate_abs_{input_name}_{ref_name}"
        f"{create_region_file_suffix(region_name)}.png"
    ).replace(" ", "_")
    _plot_monthly_panels(
        rel_monthly,
        "Monthly Relative Climatology Difference",
        rel_file_name,
        "Relative Climatology Difference [%]",
        bounds_type="percent",
    )
    _plot_monthly_panels(
        abs_monthly,
        "Monthly Absolute Climatology Difference",
        abs_file_name,
        "Absolute Climatology Difference [mm/day]",
    )


def write_comparison_plots(
    rel_mean,
    input_clim,
    ref_clim,
    input_name,
    ref_name,
    output_path,
    rel_std=None,
    spearman=None,
    bias_only=False,
    global_climate=False,
    overlapping_years=None,
    season_months=None,
    std_seasonality=None,
    spearman_seasonality=None,
    region_name=None,
):
    """Write the figures of one comparison for the whole domain or one WMO region.

    The overview map needs `rel_std` and `spearman`, `global_climate` writes the
    climate difference and the monthly local climate maps, and every other run
    the bias-only map. With `region_name` every field is cropped to the region
    first, so its colour limits, medians and seasonality are the region's own.

    Args:
        rel_mean (xr.DataArray): Relative mean (input / reference).
        input_clim, ref_clim (xr.DataArray): Monthly climatologies of both datasets.
        input_name, ref_name (str): Names of the datasets.
        output_path (Path): Directory the figures are written to.
        rel_std, spearman (xr.DataArray, optional): Relative std and Spearman map.
        bias_only, global_climate (bool): Mode of the comparison.
        overlapping_years, season_months (list of int, optional): Compared years
            and calendar months, named in the titles.
        std_seasonality, spearman_seasonality (str, optional): Seasonality labels
            of the std and Spearman panels, see `create_seasonality_label`.
        region_name (str, optional): WMO region to crop to, None for the domain.
    """
    fields = {
        "rel_mean": rel_mean,
        "input_clim": input_clim,
        "ref_clim": ref_clim,
        "rel_std": rel_std,
        "spearman": spearman,
    }
    if region_name is not None:
        fields = {
            name: None if field is None else crop_to_region(field, region_name)
            for name, field in fields.items()
        }
        if not bool(np.isfinite(fields["rel_mean"]).any()):
            logger.info(f"Skipping the figures of {region_name}: no valid cell in it.")
            return
    rel_mean, input_clim, ref_clim, rel_std, spearman = fields.values()
    shared_args = {
        "input_name": input_name,
        "ref_name": ref_name,
        "output_path": output_path,
        "overlapping_years": overlapping_years,
        "season_months": season_months,
        "region_name": region_name,
    }
    input_total, ref_total = np.nanmean(input_clim), np.nanmean(ref_clim)
    if (
        rel_std is not None
        and spearman is not None
        and not bias_only
        and not global_climate
    ):
        plot_map(
            rel_std=rel_std,
            rel_mean=rel_mean,
            spearman=spearman,
            ref_clim=ref_clim,
            input_clim=input_clim,
            std_seasonality=std_seasonality,
            spearman_seasonality=spearman_seasonality,
            **shared_args,
        )
    elif global_climate:
        input_mean_map = np.nanmean(input_clim, axis=0)
        ref_mean_map = np.nanmean(ref_clim, axis=0)
        rel_mean_diff = 100 * (input_mean_map - ref_mean_map) / ref_mean_map
        rel_mean_diff = np.where(
            np.nanmax(ref_clim, axis=0) < 0.1, np.nan, rel_mean_diff
        )
        plot_map_global_climate2(
            rel_mean=rel_mean_diff,
            diff_mean=input_mean_map - ref_mean_map,
            total_rel_mean=100 * (input_total - ref_total) / ref_total,
            **shared_args,
        )
        plot_map_local_climate(input_clim=input_clim, ref_clim=ref_clim, **shared_args)
    else:
        plot_map_bias_only(
            rel_mean=rel_mean,
            ref_clim=ref_clim,
            input_clim=input_clim,
            total_rel_mean=input_total / ref_total,
            **shared_args,
        )


def create_map_from_output(
    output_path,
    input_name,
    ref_name,
    bias_only=False,
    global_climate=False,
    region_names=(),
):
    """Read the relative statistics file and draw its figures again.

    The figures are drawn for the whole domain and once per WMO region in
    `region_names`; the season and the seasonality labels come from the file.
    """
    file = get_rel_stat_file(output_path, input_name, ref_name)
    logger.info(f"Plotting data from {file}")
    with get_xarray_ds_from_file(file, force_decending_y=True) as ds:
        input_clim_key = (
            f"{input_name}_clim" if f"{input_name}_clim" in ds else "input_clim"
        )
        ref_clim_key = f"{ref_name}_clim" if f"{ref_name}_clim" in ds else "ref_clim"
        fields = {
            name: ds[key].load() if key in ds else None
            for name, key in {
                "rel_mean": "rel_mean",
                "rel_std": "rel_std",
                "spearman": "spearman",
                "input_clim": input_clim_key,
                "ref_clim": ref_clim_key,
            }.items()
        }
        season_months = ds.attrs.get("season_months")
    for region_name in [None, *region_names]:
        write_comparison_plots(
            input_name=input_name,
            ref_name=ref_name,
            output_path=output_path,
            bias_only=bias_only,
            global_climate=global_climate,
            season_months=season_months,
            std_seasonality=getattr(fields["rel_std"], "attrs", {}).get("seasonality"),
            spearman_seasonality=getattr(fields["spearman"], "attrs", {}).get(
                "seasonality"
            ),
            region_name=region_name,
            **fields,
        )


def get_stats(  # noqa: PLR0913
    path,
    var,
    factor,
    coordinate_slice,
    n_bootstrap_years,
    ncpus,
    output_file,
    available_years=None,
    direct_comp=False,
    available_mem=None,
    mask_da=None,
    mask_var=None,
    file_name="*.*",
    stream_metrics=False,
    compression=None,
    remove_seasonal_cycle_from_std=False,
    season_months=None,
):
    """Get statistics dataset from a path to a file or directory with files.

    `remove_seasonal_cycle_from_std` and `season_months` are passed on to
    `get_stats_one_pass` or `get_file_stats`, see there.
    """
    logger.info(f"Get stats for {path}")
    if var is not None:
        # a single file is streamed too when the monthly series is needed,
        # since only the one-pass statistics build it outside the direct mode
        if (path.is_dir() or stream_metrics) and not direct_comp:
            stats_ds = get_stats_one_pass(
                path,
                var,
                factor,
                coordinate_slice,
                n_bootstrap_years=n_bootstrap_years,
                ncpus=ncpus,
                output_path=output_file,
                available_years=available_years,
                file_name=file_name,
                with_period_series=stream_metrics,
                compression=compression,
                remove_seasonal_cycle_from_std=remove_seasonal_cycle_from_std,
                season_months=season_months,
            )
        elif path.is_dir() or path.is_file():
            if path.is_file() and path.suffix == ".nc":
                chunking = available_mem is not None
                ds = get_dataset_from_path(
                    path,
                    chunking=chunking,
                    available_mem_gib=available_mem,
                    chunk_type=ChunkType.SPACE,
                    force_decending_y=True,
                    var_name=var,
                )
                stats_ds = get_file_stats(
                    ds,
                    var,
                    factor,
                    coordinate_slice,
                    output_path=output_file,
                    avaiable_years=available_years,
                    direct_comp=direct_comp,
                    compression=compression,
                    remove_seasonal_cycle_from_std=remove_seasonal_cycle_from_std,
                    season_months=season_months,
                )
            if path.is_dir():
                file_list = get_files(
                    path, available_years=available_years, file_name=file_name
                )
                # open_mfdataset keeps the record lazy, while the default opens
                # every file eagerly. The files split the record in time, so
                # chunks in space only would need a piece of every file at once.
                with get_dataset_from_path(
                    file_list,
                    var_name=var,
                    available_mem=available_mem,
                    file_name=file_name,
                    use_mfdataset=True,
                    chunk_type=ChunkType.TIME,
                ) as ds_in:
                    stats_ds = get_file_stats(
                        ds_in,
                        var,
                        factor,
                        coordinate_slice,
                        output_path=output_file,
                        avaiable_years=available_years,
                        direct_comp=direct_comp,
                        compression=compression,
                        remove_seasonal_cycle_from_std=remove_seasonal_cycle_from_std,
                        season_months=season_months,
                    )
        else:
            msg = f"Path {path} is neither dir nor file."
            with ErrorLogger(logger):
                raise ValueError(msg)
    else:
        with get_dataset_from_path(
            path,
            var_name=var,
            engine="netcdf4",
            force_decending_y=True,
            create_bounds=True,
        ) as ds_input:
            ds = ds_input
            if coordinate_slice is not None:
                ds = ds.sel(
                    {
                        get_coord_key(ds, lat=True): coordinate_slice["lat"],
                        get_coord_key(ds, lon=True): coordinate_slice["lon"],
                    }
                )
            if available_years is not None:
                ds = ds.sel(time=ds_input.time.dt.year.isin(available_years))
            if "clim" in ds and "std" in ds and "mean" in ds:
                stats_ds = ds
            else:
                with ErrorLogger(logger):
                    msg = "Wrong statisitcs file. If you want to create new statistics you have to provide a var."
                    raise KeyError(msg)

    # get_file_stats/get_stats_one_pass already attach real bounds to their
    # own output before returning (deriving cell width from the still-
    # uncropped source), and the direct-open branch above opens its file
    # with create_bounds=True — so stats_ds already carries real *_bnds
    # coordinates by this point, and apply_spatial_mask's crop preserves
    # them (including when it narrows an axis to a single point).
    masked_ds = apply_spatial_mask(stats_ds, mask_da, mask_var=mask_var)
    return generate_bounds_for_all_coords(masked_ds)


def create_relative_stats_table(
    relative_stats,
    input_clim,
    ref_clim,
    region_names,
    input_name,
    ref_name,
    output_file=None,
):
    """Log the median relative statistics per WMO region and for the domain.

    Args:
        relative_stats: Mapping holding ``rel_mean`` and optionally ``rel_std``
            and ``spearman`` as (lat, lon) DataArrays.
        input_clim: Monthly input climatology as a (month, lat, lon) DataArray.
        ref_clim: Monthly reference climatology on the same grid.
        region_names: Keys of ``WMO_REGION_BOUNDS``.
        input_name: Name of the input dataset.
        ref_name: Name of the reference dataset.
        output_file: CSV file path, or None to only log the table.

    Returns
    -------
        DataFrame of the medians per region.
    """
    stats = xr.Dataset(
        {
            name: relative_stats[name]
            for name in ("rel_mean", "rel_std", "spearman")
            if relative_stats.get(name) is not None
        }
    )
    stats["mean_diff"] = (input_clim - ref_clim).mean("month", skipna=True)
    units = input_clim.attrs.get("units")
    stats_df = calculate_region_medians(stats, list(stats.data_vars), region_names)
    write_stats_table(
        stats_df,
        title=(
            f"Median relative statistics of {input_name} against {ref_name} "
            f"(rel_* = input / reference, mean_diff = input - reference"
            f"{f' [{units}]' if units else ''})"
        ),
        output_file=output_file,
    )
    return stats_df


def create_region_metric_tables(
    input_maps,
    ref_maps,
    lat,
    lon,
    global_results,
    input_name,
    ref_name,
    region_names,
    output_stem=None,
):
    """Log every spatial metric per WMO region and for the domain as a table.

    The domain values are taken from `global_results`, so only the regions are
    calculated here.

    Args:
        input_maps: Input array with shape (time, lat, lon).
        ref_maps: Reference array with the same shape and missing pattern.
        lat: Latitudes of the map rows.
        lon: Longitudes of the map columns.
        global_results: Dict of the domain result dict per metric name, as
            returned by ``create_results_csv``.
        input_name: Name of the input dataset.
        ref_name: Name of the reference dataset.
        region_names: Keys of ``WMO_REGION_BOUNDS``.
        output_stem: Path prefix of the CSV files, or None to only log them.

    Returns
    -------
        Dict of the table per metric name.
    """
    coords = {"lat": lat, "lon": lon}
    input_da = xr.DataArray(input_maps, dims=("time", "lat", "lon"), coords=coords)
    ref_da = xr.DataArray(ref_maps, dims=("time", "lat", "lon"), coords=coords)
    tables = {}
    for metric, global_result in global_results.items():
        rows = [{"region": "Global", **global_result}]
        for region_name in region_names:
            region_input = crop_to_region(input_da, region_name).values
            region_ref = crop_to_region(ref_da, region_name).values
            if not (np.isfinite(region_input) & np.isfinite(region_ref)).any():
                continue
            try:
                result = calculate_results_metric(
                    region_input, region_ref, input_name, ref_name, metric=metric
                )
            except ValueError as error:
                logger.warning(f"Skipping {metric} for region {region_name}: {error}")
                continue
            rows.append({"region": region_name, **result})
        metric_df = pd.DataFrame(rows).drop(columns="name", errors="ignore")
        tables[metric] = metric_df
        write_stats_table(
            metric_df,
            title=f"{metric} of {input_name} against {ref_name}",
            output_file=(
                None
                if output_stem is None
                else f"{output_stem}_{metric.lower()}_regions.csv"
            ),
        )
    return tables


def compare_input_with_ref(  # noqa: PLR0912, PLR0913, PLR0915
    input_path,
    input_var,
    output_path,
    ref_path,
    ref_var,
    input_name="input",
    ref_name="ref",
    input_factor=1,
    ref_factor=1,
    coordinate_slice=None,
    n_bootstrap_years=None,
    bootstrap_index=None,
    ncpus=1,
    available_years=None,
    direct_comp=False,
    available_mem=None,
    target_freq=None,
    plot=True,
    bias_only=False,
    global_climate=False,
    compare_on_coarser_grid=False,
    stream_metrics=False,
    mask_da=None,
    mask_var=None,
    input_file_name=None,
    ref_file_name=None,
    result_metric="all",
    compression=None,
    region_names=(),
    write_region_stats=False,
    remove_seasonal_cycle_from_std=False,
    remove_seasonal_cycle_from_corr=True,
    season_months=None,
):
    """Compare the two datasets.

    The relative statistics and the spatial metrics are logged as tables, for
    the whole domain and per region in `region_names`, and written to CSV when
    `write_region_stats` is set. A bootstrap run logs only the spatial metrics;
    its relative statistics are tabled once all bootstrap runs are combined.

    `remove_seasonal_cycle_from_std` takes the std from the anomalies of the
    monthly climatology; `remove_seasonal_cycle_from_corr` correlates the
    monthly anomalies of the time series instead of the series themselves.
    Only the months in `season_months` are compared.
    """
    output_path = Path(output_path)
    if bootstrap_index is not None:
        random.seed(bootstrap_index)
    overlapping_years = None

    # get input statistics
    input_stats_file = None
    ref_stats_file = None

    input = get_stats(
        path=input_path,
        var=input_var,
        factor=input_factor,
        coordinate_slice=coordinate_slice,
        n_bootstrap_years=n_bootstrap_years,
        ncpus=ncpus,
        output_file=input_stats_file,
        available_years=available_years,
        direct_comp=direct_comp,
        available_mem=available_mem,
        mask_da=mask_da,
        mask_var=mask_var,
        file_name=input_file_name,
        stream_metrics=stream_metrics,
        compression=compression,
        remove_seasonal_cycle_from_std=remove_seasonal_cycle_from_std,
        season_months=season_months,
    )
    logger.debug(f"input ds: {input}")

    # get reference statistics
    ref = get_stats(
        path=ref_path,
        var=ref_var,
        factor=ref_factor,
        coordinate_slice=coordinate_slice,
        n_bootstrap_years=n_bootstrap_years,
        ncpus=ncpus,
        output_file=ref_stats_file,
        available_years=available_years,
        direct_comp=direct_comp,
        available_mem=available_mem,
        mask_da=mask_da,
        mask_var=mask_var,
        file_name=ref_file_name,
        stream_metrics=stream_metrics,
        compression=compression,
        remove_seasonal_cycle_from_std=remove_seasonal_cycle_from_std,
        season_months=season_months,
    )
    logger.debug(f"ref ds: {ref}")
    logger.debug(
        f"After get_stats input/ref are all nan: "
        f"input_mean={bool(input['mean'].isnull().all().compute().item())}, "
        f"input_clim={bool(input['clim'].isnull().all().compute().item())}, "
        f"ref_mean={bool(ref['mean'].isnull().all().compute().item())}, "
        f"ref_clim={bool(ref['clim'].isnull().all().compute().item())}"
    )
    # regrid spatial resoution
    # regridd to same spatial resolution
    if len(input["lat"].data) < 1 or len(input["lon"].data) < 1:
        logger.error("Input dataset has empty coordinate.")
    if len(ref["lat"].data) < 1 or len(ref["lon"].data) < 1:
        logger.error("Ref dataset has empty coordinate.")

    input, ref = crop_datasets_to_spatial_overlap(input, ref)
    if compare_on_coarser_grid:
        input, ref = regrid_datasets_to_coarser_grid(input, ref)
        # the aggregation can shift the outer edges, so the overlap is taken again
        input, ref = crop_datasets_to_spatial_overlap(input, ref)
    else:
        input, ref = regridd_to_higher_spatial_resolution(input, ref)
    logger.debug(
        f"After spatial overlap/regrid input/ref are all nan: "
        f"input_mean={bool(input['mean'].isnull().all().compute().item())}, "
        f"input_clim={bool(input['clim'].isnull().all().compute().item())}, "
        f"ref_mean={bool(ref['mean'].isnull().all().compute().item())}, "
        f"ref_clim={bool(ref['clim'].isnull().all().compute().item())}"
    )
    output_name = f"{input_name}-{ref_name}".replace(" ", "_")
    region_output_stem = None
    if write_region_stats:
        region_output_stem = output_path / output_name
        if bootstrap_index is not None:
            region_output_stem = output_path / f"{output_name}_{bootstrap_index}"
    # compare and save statistics
    full_metrics = not bias_only and not global_climate
    with_std = not bias_only

    if direct_comp or stream_metrics:
        input_ts, ref_ts = input["time_series"], ref["time_series"]

        # If we already know the target frequency, resample to that; otherwise pick the coarser one.
        if target_freq is not None:
            input_ts, ref_ts = resample_to_target_freq(input_ts, ref_ts, target_freq)
        else:
            input_ts, ref_ts = resample_to_coarser_calendar(input_ts, ref_ts)

        input_ts, ref_ts, time_slice = crop_data_to_overlapping_time(input_ts, ref_ts)
        overlapping_years = [time_slice.start.year, time_slice.stop.year]
        # resampling to the coarser calendar fills the months between two
        # seasons with empty steps, so the season is selected again
        if season_months is not None:
            input_ts = select_time_steps_by_year_and_month(
                input_ts, months=season_months
            )
            ref_ts = select_time_steps_by_year_and_month(ref_ts, months=season_months)
        if input_ts.shape != ref_ts.shape:
            msg = f"Input and ref time_series shapes differ after resampling/cropping: {input_ts.shape} vs {ref_ts.shape}"
            with ErrorLogger(logger):
                raise ValueError(msg)
        logger.info(
            f"Creating data from timeseries with shape {input_ts.shape} "
            f"and {ref_ts.shape} as {input_ts.dtype}"
        )
        if input_ts.sizes.get("time", 0) == 0 or ref_ts.sizes.get("time", 0) == 0:
            msg = "No overlapping time steps after alignment; cannot compare input and reference."
            logger.error(msg)
            raise ValueError(msg)
        no_pair_msg = (
            "Input and reference data have no paired finite values after "
            "time alignment, spatial overlap, regridding, and masking."
        )
        if not full_metrics:
            # nothing below reads the stacks, so this stays a reduction that
            # runs chunk by chunk instead of materializing both of them
            if not bool((np.isfinite(input_ts) & np.isfinite(ref_ts)).any()):
                with ErrorLogger(logger):
                    raise ValueError(no_pair_msg)
        else:
            # astype always copies, so the metrics below own their arrays. Each
            # record is released right after its copy exists, because a float64
            # record and its float32 copy would otherwise both be held.
            input_time, ref_time = input_ts.time, ref_ts.time
            input_ts_np = input_ts.astype(np.float32).values
            del input_ts
            input = input.drop_vars("time_series")
            ref_ts_np = ref_ts.astype(np.float32).values
            del ref_ts
            ref = ref.drop_vars("time_series")
            # One buffer does both jobs: it holds the paired steps first and is
            # then inverted into the mask that gives both stacks the same
            # missing pattern, as the metrics below expect.
            paired_valid = np.isfinite(input_ts_np)
            paired_valid &= np.isfinite(ref_ts_np)
            if not paired_valid.any():
                with ErrorLogger(logger):
                    raise ValueError(no_pair_msg)
            np.logical_not(paired_valid, out=paired_valid)
            input_ts_np[paired_valid] = np.nan
            ref_ts_np[paired_valid] = np.nan
            del paired_valid
            try:
                global_results = create_results_csv(
                    map1=input_ts_np,
                    map2=ref_ts_np,
                    ds1_name=input_name,
                    ds2_name=ref_name,
                    out_dir=output_path,
                    out_name=output_name,
                    metric=result_metric,
                    log_table=False,
                )
                create_region_metric_tables(
                    input_ts_np,
                    ref_ts_np,
                    get_coord_values(input, lat=True),
                    get_coord_values(input, lon=True),
                    global_results,
                    input_name,
                    ref_name,
                    region_names,
                    output_stem=region_output_stem,
                )
                spearman_basis = "the time series"
                if remove_seasonal_cycle_from_corr:
                    # after the spatial metrics, which need the raw maps: each
                    # series loses its own monthly means in place, one month at
                    # a time, so the correlation follows the anomalies
                    months = input_time.dt.month.values
                    with warnings.catch_warnings():
                        # cells without any paired step warn as empty slices
                        warnings.simplefilter("ignore", category=RuntimeWarning)
                        for month in np.unique(months):
                            in_month = months == month
                            for stack in (input_ts_np, ref_ts_np):
                                stack[in_month] -= np.nanmean(stack[in_month], axis=0)
                    spearman_basis = "the monthly anomalies of the time series"
                spearman_seasonality = create_seasonality_label(
                    remove_seasonal_cycle_from_corr
                )
                spearman, spearman_pval = calculate_spearman_map(input_ts_np, ref_ts_np)
            except ValueError as ve:
                logger.error("Input and ref do not have the same temporal extent.")
                logger.info(input_time)
                logger.info(ref_time)
                raise ve
            del input_ts_np, ref_ts_np
    elif full_metrics:
        logger.info("Calculating spearman correlation from seasonalities.")
        input_clim_np = np.asarray(input["clim"].values)
        ref_clim_np = np.asarray(ref["clim"].values)
        if season_months is not None:
            # the climatologies keep all 12 months, those outside the season empty
            season_index = np.asarray(season_months) - 1
            input_clim_np = input_clim_np[season_index]
            ref_clim_np = ref_clim_np[season_index]
        spearman_basis = "the monthly climatologies"
        spearman_seasonality = CLIMATOLOGY_SEASONALITY_LABEL
        if not np.any(np.isfinite(input_clim_np) & np.isfinite(ref_clim_np)):
            msg = (
                "Input and reference climatologies have no paired finite values "
                "after spatial overlap, regridding, and masking."
            )
            with ErrorLogger(logger):
                raise ValueError(msg)
        spearman, spearman_pval = calculate_spearman_map(input_clim_np, ref_clim_np)
        global_results = create_results_csv(
            map1=input_clim_np,
            map2=ref_clim_np,
            ds1_name=input_name,
            ds2_name=ref_name,
            out_dir=output_path,
            out_name=output_name,
            metric=result_metric,
            log_table=False,
        )
        create_region_metric_tables(
            input_clim_np,
            ref_clim_np,
            get_coord_values(input, lat=True),
            get_coord_values(input, lon=True),
            global_results,
            input_name,
            ref_name,
            region_names,
            output_stem=region_output_stem,
        )

    if input["mean"].shape != ref["mean"].shape:
        msg = f"Input and ref mean shapes differ after regridding: {input['mean'].shape} vs {ref['mean'].shape}"
        with ErrorLogger(logger):
            raise ValueError(msg)
    mean_valid = (
        np.isfinite(input["mean"].values)
        & np.isfinite(ref["mean"].values)
        & (ref["mean"].values != 0)
    )
    logger.debug(
        f"Mean comparison valid cells: "
        f"{int(np.count_nonzero(mean_valid))}/{mean_valid.size}; "
        f"input_mean_all_nan={bool(input['mean'].isnull().all().compute().item())}; "
        f"ref_mean_all_nan={bool(ref['mean'].isnull().all().compute().item())}"
    )
    rel_mean_values = np.full(input["mean"].shape, np.nan, dtype=np.float64)
    np.divide(
        input["mean"].values,
        ref["mean"].values,
        out=rel_mean_values,
        where=mean_valid,
    )
    rel_mean = xr.DataArray(
        rel_mean_values, coords=input["mean"].coords, dims=input["mean"].dims
    )
    if with_std:
        if input["std"].shape != ref["std"].shape:
            msg = f"Input and ref std shapes differ after regridding: {input['std'].shape} vs {ref['std'].shape}"
            with ErrorLogger(logger):
                raise ValueError(msg)
        std_valid = (
            np.isfinite(input["std"].values)
            & np.isfinite(ref["std"].values)
            & (ref["std"].values != 0)
        )
        rel_std_values = np.full(input["std"].shape, np.nan, dtype=np.float64)
        np.divide(
            input["std"].values,
            ref["std"].values,
            out=rel_std_values,
            where=std_valid,
        )
        rel_std = xr.DataArray(
            rel_std_values, coords=input["std"].coords, dims=input["std"].dims
        )
    else:
        rel_std = None
    if full_metrics:
        spearman = xr.DataArray(
            spearman.data, coords=input["std"].coords, dims=input["std"].dims
        )
        spearman_pval = xr.DataArray(
            spearman_pval.data, coords=input["std"].coords, dims=input["std"].dims
        )
    else:
        spearman, spearman_pval = None, None
    if input["clim"].shape != ref["clim"].shape:
        msg = f"Input and ref clim shapes differ after regridding: {input['clim'].shape} vs {ref['clim'].shape}"
        with ErrorLogger(logger):
            raise ValueError(msg)
    clim_valid = np.isfinite(input["clim"].values) & np.isfinite(ref["clim"].values)
    logger.debug(
        f"Climatology comparison valid cells: "
        f"{int(np.count_nonzero(clim_valid))}/{clim_valid.size}; "
        f"input_clim_all_nan={bool(input['clim'].isnull().all().compute().item())}; "
        f"ref_clim_all_nan={bool(ref['clim'].isnull().all().compute().item())}"
    )
    input_clim = xr.DataArray(
        np.where(clim_valid, input["clim"].values, np.nan),
        coords={
            "month": np.arange(1, 13, 1),
            "lat": get_coord_values(input, lat=True),
            "lon": get_coord_values(input, lon=True),
        },
        dims=["month", "lat", "lon"],
        attrs=dict(input["clim"].attrs),
    )
    ref_clim = xr.DataArray(
        np.where(clim_valid, ref["clim"].values, np.nan),
        coords={
            "month": np.arange(1, 13, 1),
            "lat": get_coord_values(input, lat=True),
            "lon": get_coord_values(input, lon=True),
        },
        dims=["month", "lat", "lon"],
        attrs=dict(ref["clim"].attrs),
    )
    rel_mean = rel_mean.where(np.isfinite(rel_mean) & (rel_mean >= 0))
    rel_mean.attrs = {
        "units": "1",
        "long_name": "Relative mean (input / reference)",
    }
    output = xr.Dataset(
        {
            "rel_mean": rel_mean,
        },
        coords={
            "month": np.arange(1, 13, 1),
            "lat": get_coord_values(input, lat=True),
            "lon": get_coord_values(input, lon=True),
        },
    )
    if with_std:
        rel_std = rel_std.where(np.isfinite(rel_std) & (rel_std >= 0))
        rel_std.attrs = {
            "units": "1",
            "long_name": "Relative standard deviation (input / reference)",
            "seasonality": create_seasonality_label(remove_seasonal_cycle_from_std),
        }
        output["rel_std"] = rel_std
    if full_metrics:
        spearman.attrs = {
            "units": "1",
            "long_name": (
                f"Spearman rank correlation of {spearman_basis} (input vs reference)"
            ),
            "seasonality": spearman_seasonality,
        }
        spearman_pval.attrs = {
            "units": "1",
            "long_name": (
                f"p-value of Spearman rank correlation of {spearman_basis} "
                "(input vs reference)"
            ),
        }
        output["spearman"] = spearman
        output["spearman_pval"] = spearman_pval
    if season_months is not None:
        output.attrs["season_months"] = np.asarray(season_months, dtype=np.int32)
    file_name = "relative_stats"
    if input_name is not None:
        file_name += f"_{input_name}"
        output[f"{input_name}_clim"] = input_clim
    else:
        output["input_clim"] = input_clim
    if ref_name is not None:
        file_name += f"_{ref_name}"
        output[f"{ref_name}_clim"] = ref_clim
    else:
        output["ref_clim"] = ref_clim
    logger.debug(
        f"Before writing relative output is all nan: "
        f"rel_mean={bool(output['rel_mean'].isnull().all().compute().item())}; "
        f"input_clim={bool(input_clim.isnull().all().compute().item())}; "
        f"ref_clim={bool(ref_clim.isnull().all().compute().item())}"
    )
    file_name = file_name.replace(" ", "_")
    if bootstrap_index is not None:
        file_name = output_path / f"{file_name}_{bootstrap_index}.nc"
    else:
        file_name = output_path / f"{file_name}.nc"

    write_xarray_to_file(ds=output, file_path=file_name, compression=compression)
    logger.info(f"Written output to {file_name}")
    if bootstrap_index is None:
        create_relative_stats_table(
            output,
            input_clim,
            ref_clim,
            region_names,
            input_name,
            ref_name,
            output_file=(
                file_name.with_name(f"{file_name.stem}_regions.csv")
                if write_region_stats
                else None
            ),
        )
    if plot:
        for region_name in [None, *region_names]:
            write_comparison_plots(
                rel_mean=rel_mean,
                input_clim=input_clim,
                ref_clim=ref_clim,
                input_name=input_name,
                ref_name=ref_name,
                output_path=output_path,
                rel_std=rel_std,
                spearman=spearman,
                bias_only=bias_only,
                global_climate=global_climate,
                overlapping_years=overlapping_years,
                season_months=season_months,
                std_seasonality=rel_std.attrs["seasonality"] if with_std else None,
                spearman_seasonality=spearman_seasonality if full_metrics else None,
                region_name=region_name,
            )
    return file_name


def get_rel_stat_file(output_path, input_name, ref_name):
    """Return the path for the relative-statistics file for the two datasets."""
    file_name = "relative_stats"
    if input_name is not None:
        file_name += f"_{input_name}"
    if ref_name is not None:
        file_name += f"_{ref_name}"
    file_name = file_name.replace(" ", "_")
    return output_path / f"{file_name}.nc"


def evaluate_boostraping_stat_files(stat_files, input_name, ref_name):
    """Evaluate bootstrapped stats and return medians across iterations."""
    # Open the first file to initialize dimensions and weights
    try:
        with get_xarray_ds_from_file(
            stat_files[0], force_decending_y=True
        ) as first_file:
            shape = first_file["rel_mean"].shape
            n_bootstrap = len(stat_files)
            # the coordinates let the combined fields be cropped to regions
            coords = {
                "lat": first_file["rel_mean"]["lat"].values,
                "lon": first_file["rel_mean"]["lon"].values,
            }

            # Determine keys for climatology fields
            input_clim_key = (
                f"{input_name}_clim"
                if f"{input_name}_clim" in first_file
                else "input_clim"
            )
            ref_clim_key = (
                f"{ref_name}_clim" if f"{ref_name}_clim" in first_file else "ref_clim"
            )

            has_rel_std = "rel_std" in first_file
            has_spearman = "spearman" in first_file

            # Preallocate arrays for all available variables
            mean = np.empty((n_bootstrap, *shape))
            std = np.empty((n_bootstrap, *shape)) if has_rel_std else None
            spearman = np.empty((n_bootstrap, *shape)) if has_spearman else None
            input_clim = np.empty((n_bootstrap, 12, *shape))  # Month x lat x lon
            ref_clim = np.empty((n_bootstrap, 12, *shape))  # Month x lat x lon
    except ValueError as ve:
        logger.error(f"opening file {stat_files[0]} as first file failed.")
        raise ve
    # Fill the preallocated arrays with bootstrap data
    for i, file in enumerate(stat_files):
        with get_xarray_ds_from_file(file, force_decending_y=True) as ds:
            mean[i] = ds["rel_mean"].values
            if std is not None:
                std[i] = ds["rel_std"].values
            if spearman is not None:
                spearman[i] = ds["spearman"].values
            input_clim[i] = ds[input_clim_key].values
            ref_clim[i] = ds[ref_clim_key].values

    # Convert the arrays into xarray DataArrays
    mean_da = xr.DataArray(mean, dims=["bootstrap", "lat", "lon"], coords=coords)
    std_da = (
        xr.DataArray(std, dims=["bootstrap", "lat", "lon"], coords=coords)
        if std is not None
        else None
    )
    spearman_da = (
        xr.DataArray(spearman, dims=["bootstrap", "lat", "lon"], coords=coords)
        if spearman is not None
        else None
    )
    input_clim_da = xr.DataArray(
        input_clim, dims=["bootstrap", "month", "lat", "lon"], coords=coords
    )
    ref_clim_da = xr.DataArray(
        ref_clim, dims=["bootstrap", "month", "lat", "lon"], coords=coords
    )

    # Combine results into an xarray Dataset
    results = {
        "rel_mean": mean_da.median(dim="bootstrap"),
        "input_clim": input_clim_da.median(dim="bootstrap"),
        "ref_clim": ref_clim_da.median(dim="bootstrap"),
    }
    if std_da is not None:
        results["rel_std"] = std_da.median(dim="bootstrap")
    if spearman_da is not None:
        results["spearman"] = spearman_da.median(dim="bootstrap")
    return results


def regridd_to_higher_spatial_resolution(ds1, ds2):
    """Regrid the coarser dataset to the finer dataset's resolution using nearest neighbor.

    Parameters
    ----------
    ds1 : xarray.Dataset
        First dataset.
    ds2 : xarray.Dataset
        Second dataset. Both should have latitude ('lat') and longitude ('lon') as coordinates.

    Returns
    -------
    xarray.Dataset
        Regridded version of the coarser dataset to match the finer dataset.
    xarray.Dataset
        The finer dataset (unchanged).
    """
    # lat_res_1 = abs(ds1["lat"][1] - ds1["lat"][0]).item()
    # lon_res_1 = abs(ds1["lon"][1] - ds1["lon"][0]).item()
    # lat_res_2 = abs(ds2["lat"][1] - ds2["lat"][0]).item()
    # lon_res_2 = abs(ds2["lon"][1] - ds2["lon"][0]).item()

    # as they are croped to the same extend the shape can be used as proxy:
    lat_shape_1 = ds1["lat"].shape[0]
    lon_shape_1 = ds1["lon"].shape[0]
    lat_shape_2 = ds2["lat"].shape[0]
    lon_shape_2 = ds2["lon"].shape[0]

    # Identify the finer and coarser datasets
    if (lat_shape_1 * lon_shape_1) == (lat_shape_2 * lon_shape_2):
        return ds1, ds2
    if (lat_shape_1 * lon_shape_1) < (lat_shape_2 * lon_shape_2):
        coarse_ds, fine_ds = ds1, ds2
    else:
        coarse_ds, fine_ds = ds2, ds1
    coarse_res = get_file_res(ds=coarse_ds, raise_exception=False)
    # a NaN tolerance would let every nearest match fail and blank the whole grid
    if np.isnan(coarse_res):
        regridded_ds = coarse_ds.reindex(
            lat=fine_ds["lat"], lon=fine_ds["lon"], method="nearest"
        )
    else:
        regridded_ds = coarse_ds.reindex(
            lat=fine_ds["lat"],
            lon=fine_ds["lon"],
            method="nearest",
            tolerance=coarse_res,
        )
    if (
        regridded_ds["lat"].shape != fine_ds["lat"].shape
        or regridded_ds["lon"].shape != fine_ds["lon"].shape
    ):
        debug_msg = (
            f"Regridding resulted in unexpected coordinate shapes. "
            f"Coarse lat shape: {coarse_ds['lat'].shape}, Coarse lon shape: {coarse_ds['lon'].shape}, "
            f"Fine lat shape: {fine_ds['lat'].shape}, Fine lon shape: {fine_ds['lon'].shape}, "
            f"Regridded lat shape: {regridded_ds['lat'].shape}, Regridded lon shape: {regridded_ds['lon'].shape}."
            f"With bounds coarse lat: ({coarse_ds['lat'].min().item()}, {coarse_ds['lat'].max().item()}), "
            f"With bounds fine lat: ({fine_ds['lat'].min().item()}, {fine_ds['lat'].max().item()}), "
            f"With bounds regridded lat: ({regridded_ds['lat'].min().item()}, {regridded_ds['lat'].max().item()})"
        )
        logger.debug(debug_msg)
        try:
            aligned_coarse, aligned_fine = xr.align(regridded_ds, fine_ds, join="inner")
        except Exception as e:
            # If align fails for any reason, fall back to returning the best-effort regridded dataset
            with ErrorLogger(logger):
                logger.error(
                    f"Alignment of regridded dataset with fine dataset failed. Returning best-effort regridded dataset. Debug info: {debug_msg}"
                )
                raise e
    else:
        aligned_coarse, aligned_fine = regridded_ds, fine_ds
    if coarse_ds is ds1:
        return aligned_coarse, aligned_fine
    return aligned_fine, aligned_coarse


def regrid_datasets_to_coarser_grid(ds1, ds2):
    """Bring two statistics datasets onto the coarser of their two grids.

    The coordinate bounds are dropped before aggregating and regenerated after,
    because averaging bounds values would not describe the new cells.

    Parameters
    ----------
    ds1, ds2 : xarray.Dataset
        Statistics datasets with 'lat' and 'lon' coordinates.

    Returns
    -------
    xarray.Dataset
        The first dataset on the common coarse grid.
    xarray.Dataset
        The second dataset on the common coarse grid.
    """
    bounds_vars = [
        name
        for name in set(ds1.variables) | set(ds2.variables)
        if str(name).endswith("_bnds")
    ]
    coarse_ds1, coarse_ds2 = regrid_to_coarser_grid(
        ds1.drop_vars(bounds_vars, errors="ignore"),
        ds2.drop_vars(bounds_vars, errors="ignore"),
    )
    return (
        generate_bounds_for_all_coords(coarse_ds1),
        generate_bounds_for_all_coords(coarse_ds2),
    )


def get_years_from_path(path, raise_exception=True, file_name="*.*"):
    """Get available years from a dataset folder structure or file."""
    if path.is_dir():
        if len(year_structure_paths(path, file_name=file_name)) == 0:
            matching_files = list(path.rglob(file_name))
            if not matching_files:
                msg = f"No files matching pattern '{file_name}' found in directory {path}."
                with ErrorLogger(logger):
                    raise ValueError(msg)
            with get_dataset_from_path(matching_files, file_name=file_name) as ds:
                if "time" in ds.coords:
                    return [int(y) for y in np.unique(ds.time.dt.year.data)]
                msg = f"No year structure found in directory {path} and files do not contain a time coordinate to determine years."
                with ErrorLogger(logger):
                    raise ValueError(msg)
        return [int(p.name) for p in year_structure_paths(path, file_name=file_name)]
    if path.is_file():
        with get_xarray_ds_from_file(path, force_decending_y=True) as input_ds:
            if "time" in input_ds.coords:
                return [int(y) for y in np.unique(input_ds.time.dt.year.data)]
            msg = f"File {path} does not contain a time coordinate to determine years."
            with ErrorLogger(logger):
                raise ValueError(msg)
    if raise_exception:
        msg = f"The provided path {path} is neither file nor directory."
        with ErrorLogger(logger):
            raise ValueError(msg)
    return []


def get_available_years(
    input_path,
    ref_path,
    year_slice=None,
    direct_comp=True,
    input_file_name="*.*",
    ref_file_name="*.*",
):
    """Determine available years from constrains and datasets.

    If no reference data is given it will only be the input years inside
    the year slice. See `select_available_years`.
    """
    logger.info("Determining available years.")
    years_in = get_years_from_path(input_path, file_name=input_file_name)
    years_ref = (
        get_years_from_path(ref_path, raise_exception=False, file_name=ref_file_name)
        if ref_path is not None
        else []
    )
    logger.debug(f"Input years: {sorted(years_in)}; ref years: {sorted(years_ref)}")
    return select_available_years(years_in, years_ref, year_slice, direct_comp)


def select_available_years(years_in, years_ref, year_slice=None, direct_comp=True):
    """Select the years to compare from the years of both datasets.

    Args:
        years_in (list of int): Years of the input dataset.
        years_ref (list of int): Years of the reference dataset, empty without one.
        year_slice (slice, optional): First and last allowed year.
        direct_comp (bool): Keep only the years both datasets cover; otherwise
            the years of either dataset.

    Returns
    -------
        Sorted list of the selected years.
    """
    year_slice = year_slice if year_slice is not None else slice(None, None)
    candidates = set(years_in)
    if years_ref:
        if direct_comp:
            candidates &= set(years_ref)
        else:
            candidates |= set(years_ref)
    return [
        year
        for year in sorted(candidates)
        if (year_slice.start is None or year >= year_slice.start)
        and (year_slice.stop is None or year <= year_slice.stop)
    ]


def year_structure_paths(path: Path, file_name="*.*") -> bool:
    """Return any subdirectory of `path` with year structur.

    That means any subdirectory whose name is exactly four digits (a
    valid “year”), and that subdirectory contains at least one entry.
    """
    if not path.is_dir():
        return False
    year_paths = []
    year_re = re.compile(r"^\d{4}$")
    for sub in path.iterdir():
        # check that sub is dir, has a year as name and contains one dir or file
        if (
            sub.is_dir()
            and year_re.fullmatch(sub.name)
            and any(list(sub.rglob(file_name)))
        ):
            year_paths.append(sub)
    return year_paths


def infer_time_resolution_hours_from_files(files):
    """Infer dataset time resolution (hours) from one or more files."""
    file_list = sorted(files)
    if not file_list:
        return None
    max_probe_files = 24
    if len(file_list) > max_probe_files:
        # Probe a small consecutive subset to preserve local cadence.
        file_list = file_list[:max_probe_files]

    intra_file_diffs = []
    start_times = []

    for file in file_list:
        with get_xarray_ds_from_file(file, force_decending_y=True) as ds:
            if "time" not in ds.coords:
                continue
            times = np.asarray(ds.time.values)
            if times.size == 0:
                continue

            start_times.append(times[0])
            if times.size > 1:
                times = times.astype("datetime64[ns]")
                diffs = np.diff(times) / np.timedelta64(1, "h")
                diffs = diffs[np.isfinite(diffs) & (diffs > 0)]
                if diffs.size > 0:
                    intra_file_diffs.append(float(np.median(diffs)))

    if intra_file_diffs:
        return float(np.median(np.asarray(intra_file_diffs)))

    if len(start_times) > 1:
        start_times = np.unique(np.asarray(start_times, dtype="datetime64[ns]"))
        start_times.sort()
        diffs = np.diff(start_times) / np.timedelta64(1, "h")
        diffs = diffs[np.isfinite(diffs) & (diffs > 0)]
        if diffs.size > 0:
            return float(np.median(diffs))
    return None


@log_arguments()
def gridded_data_evaluation(  # noqa: PLR0913
    input: EvalDataset,
    ref: EvalDataset,
    output_path,
    coordinate_slice=None,
    year_slice=None,
    mask_da=None,
    mask_var=None,
    direct_comp=True,
    n_bootstrap_years=None,
    n_bootstrap_selections=None,
    target_time_freq=None,
    bias_only=False,
    global_climate=False,
    compare_on_coarser_grid=False,
    stream_metrics=False,
    only_plot=False,
    result_metric="all",
    avaiable_mem=None,
    n_cpus=1,
    compression=None,
    regions="all",
    create_wmo_region_output=False,
    remove_seasonal_cycle_from_std=False,
    remove_seasonal_cycle_from_corr=True,
    season_months=None,
):
    """Validate a spatial variable by comparing dataset climatologies.

    With `create_wmo_region_output` the statistics tables get a row per WMO
    region in `regions` ("all", "none" or a comma separated list), are written
    to CSV, and every figure is drawn again for each region, cropped to it.
    Otherwise only the whole domain is evaluated.

    `remove_seasonal_cycle_from_std`/`_corr` take the std and the Spearman
    correlation from the anomalies of the monthly climatology. With fewer than
    `MIN_YEARS_TO_REMOVE_SEASONAL_CYCLE` years covered by both datasets (per
    bootstrap selection) the seasonal cycle is kept in both. Bootstrapping always
    correlates with the seasonal cycle. `season_months` restricts all
    statistics and metrics to these calendar months.
    """
    output_path = Path(output_path)
    region_names = select_regions(regions) if create_wmo_region_output else []
    input_path = input.path
    ref_path = ref.path
    logger.debug(
        f"gridded_data_evaluation received mask as {type(mask_da).__name__}; "
        f"data_vars={list(mask_da.data_vars) if hasattr(mask_da, 'data_vars') else None}; "
        f"name={getattr(mask_da, 'name', None)}; mask_var={mask_var}"
    )

    output_path.mkdir(parents=True, exist_ok=True)

    if only_plot and get_rel_stat_file(output_path, input.name, ref.name).is_file():
        create_map_from_output(
            output_path=output_path,
            input_name=input.name,
            ref_name=ref.name,
            bias_only=bias_only,
            global_climate=global_climate,
            region_names=region_names,
        )
        return

    # when reading both datasets into memory, we need to ensure that we have enough memory for both
    avaiable_mem = (
        avaiable_mem / 2
        if avaiable_mem is not None and ref_path is not None
        else avaiable_mem
    )

    bootstrap_requested = (
        n_bootstrap_years is not None
        and n_bootstrap_selections is not None
        and n_bootstrap_selections > 0
    )

    if bootstrap_requested:
        if input_path.is_dir() and len(year_structure_paths(input_path)) == 0:
            msg = (
                "Bootstrapping requires year-structured input data directories (YYYY/...). "
                f"Input path '{input_path}' has no year-folder structure."
            )
            with ErrorLogger(logger):
                raise ValueError(msg)
        if (
            ref_path is not None
            and ref_path.is_dir()
            and len(year_structure_paths(ref_path)) == 0
        ):
            msg = (
                "Bootstrapping requires year-structured reference data directories (YYYY/...). "
                f"Reference path '{ref_path}' has no year-folder structure."
            )
            with ErrorLogger(logger):
                raise ValueError(msg)

    logger.info("Determining available years.")
    years_in = get_years_from_path(input_path, file_name=input.file_name)
    years_ref = (
        get_years_from_path(ref_path, raise_exception=False, file_name=ref.file_name)
        if ref_path is not None
        else []
    )
    # the streamed metrics compare the common years like the direct comparison
    available_years = select_available_years(
        years_in, years_ref, year_slice, direct_comp or stream_metrics
    )
    logger.info(f"Years {available_years} are available for comparison.")
    if not available_years:
        logger.error("Since no data is available the program is stoped.")
        return
    # both anomalies need enough years, so only the years both datasets cover
    # count, and a bootstrap selection holds at most n_bootstrap_years of them
    n_common_years = len(select_available_years(years_in, years_ref, year_slice))
    n_common_years = (
        min(n_common_years, n_bootstrap_years)
        if bootstrap_requested
        else n_common_years
    )
    if n_common_years < MIN_YEARS_TO_REMOVE_SEASONAL_CYCLE and (
        remove_seasonal_cycle_from_std or remove_seasonal_cycle_from_corr
    ):
        logger.warning(
            f"Input and reference share only {n_common_years} years, fewer than "
            f"{MIN_YEARS_TO_REMOVE_SEASONAL_CYCLE}, so the seasonal cycle is kept "
            "in the standard deviation and the Spearman correlation."
        )
        remove_seasonal_cycle_from_std = False
        remove_seasonal_cycle_from_corr = False

    if ref_path is None:
        # Only create statistics; do not compare
        logger.info(
            f"No ref file provided. Only creating a stat file for {input.name}."
        )
        output_name = f"{input.name}_stats.nc" if input.name is not None else "stats.nc"
        output_name = output_name.replace(" ", "_")
        if input_path.is_file():
            # Write file stats to file
            with get_xarray_ds_from_file(
                input_path, chunking=True, available_mem_gib=10, force_decending_y=True
            ) as ds_in:
                get_file_stats(
                    ds_in,
                    input.var,
                    input.factor,
                    coordinate_slice,
                    output_path=output_path / output_name,
                    avaiable_years=available_years,  # keep parameter name as used elsewhere
                    compression=compression,
                    remove_seasonal_cycle_from_std=remove_seasonal_cycle_from_std,
                    season_months=season_months,
                )

        elif (
            n_bootstrap_years is not None
            and n_bootstrap_selections > 0
            and input_path.is_dir()
        ):
            # Write file stats for each bootstrap selection
            _ = Parallel(n_jobs=n_cpus)(
                delayed(get_stats_one_pass)(
                    input_path,
                    input.var,
                    input.factor,
                    coordinate_slice,
                    n_bootstrap_years,
                    bootstrap_index,
                    output_path / output_name,
                    available_years=available_years,
                    file_name=input.file_name,
                    compression=compression,
                    remove_seasonal_cycle_from_std=remove_seasonal_cycle_from_std,
                    season_months=season_months,
                )
                for bootstrap_index in range(n_bootstrap_selections)
            )

        elif input_path.is_dir():
            # Stats for one dataset read from multiple files in a directory
            get_stats_one_pass(
                input_path,
                input.var,
                input.factor,
                coordinate_slice,
                output_path=output_path / output_name,
                available_years=available_years,
                file_name=input.file_name,
                compression=compression,
                remove_seasonal_cycle_from_std=remove_seasonal_cycle_from_std,
                season_months=season_months,
            )
        else:
            with ErrorLogger(logger):
                msg = "input path does not exist"
                raise ValueError(msg)

    elif (
        n_bootstrap_years is not None
        and n_bootstrap_selections > 0
        and input_path.is_dir()
        and ref_path.is_dir()
        and not direct_comp
    ):
        # Compare via bootstrapping
        if only_plot:
            stat_files = list(
                output_path.glob(f"relative_stats_{input.name}_{ref.name}_*")
            )
        else:
            stat_files = Parallel(n_jobs=n_cpus)(
                delayed(compare_input_with_ref)(
                    input_path,
                    input.var,
                    output_path,
                    ref_path,
                    ref.var,
                    input.name,
                    ref.name,
                    input.factor,
                    ref.factor,
                    coordinate_slice,
                    n_bootstrap_years,
                    bootstrap_index,
                    available_years=available_years,
                    input_file_name=input.file_name,
                    ref_file_name=ref.file_name,
                    target_freq=target_time_freq,
                    bias_only=bias_only,
                    global_climate=global_climate,
                    compare_on_coarser_grid=compare_on_coarser_grid,
                    stream_metrics=stream_metrics,
                    mask_da=mask_da,
                    mask_var=mask_var,
                    result_metric=result_metric,
                    compression=compression,
                    region_names=region_names,
                    write_region_stats=create_wmo_region_output,
                    # the figures are drawn once from the medians of all runs
                    plot=False,
                    remove_seasonal_cycle_from_std=remove_seasonal_cycle_from_std,
                    # bootstrapping correlates with the seasonal cycle on purpose
                    remove_seasonal_cycle_from_corr=False,
                    season_months=season_months,
                )
                for bootstrap_index in range(n_bootstrap_selections)
            )
        stat_files = [file for file in stat_files if file is not None]
        if stat_files:
            results = evaluate_boostraping_stat_files(
                stat_files, input_name=input.name, ref_name=ref.name
            )
            rel_stat_file = get_rel_stat_file(output_path, input.name, ref.name)
            create_relative_stats_table(
                results,
                results["input_clim"],
                results["ref_clim"],
                region_names,
                input.name,
                ref.name,
                output_file=(
                    rel_stat_file.with_name(f"{rel_stat_file.stem}_regions.csv")
                    if create_wmo_region_output
                    else None
                ),
            )
            for region_name in [None, *region_names]:
                write_comparison_plots(
                    input_name=input.name,
                    ref_name=ref.name,
                    output_path=output_path,
                    bias_only=bias_only,
                    global_climate=global_climate,
                    season_months=season_months,
                    std_seasonality=create_seasonality_label(
                        remove_seasonal_cycle_from_std
                    ),
                    # bootstrapping correlates the raw series or the climatologies
                    spearman_seasonality=(
                        create_seasonality_label(False)
                        if stream_metrics
                        else CLIMATOLOGY_SEASONALITY_LABEL
                    ),
                    region_name=region_name,
                    **results,
                )
        else:
            logger.error("There are no statfiles created from the evaluation.")
    else:
        logger.info("Compare without bootstrapping.")
        # every dask thread holds its own chunk, so --ncpus bounds the memory
        # that --available-mem sizes the chunks for
        with dask.config.set(num_workers=n_cpus):
            compare_input_with_ref(
                input_path,
                input.var,
                output_path,
                ref_path,
                ref.var,
                input.name,
                ref.name,
                input.factor,
                ref.factor,
                coordinate_slice,
                ncpus=n_cpus,
                available_years=available_years,
                direct_comp=direct_comp,
                available_mem=avaiable_mem,
                input_file_name=input.file_name,
                ref_file_name=ref.file_name,
                target_freq=target_time_freq,
                bias_only=bias_only,
                global_climate=global_climate,
                compare_on_coarser_grid=compare_on_coarser_grid,
                stream_metrics=stream_metrics,
                mask_da=mask_da,
                mask_var=mask_var,
                result_metric=result_metric,
                compression=compression,
                region_names=region_names,
                write_region_stats=create_wmo_region_output,
                remove_seasonal_cycle_from_std=remove_seasonal_cycle_from_std,
                remove_seasonal_cycle_from_corr=remove_seasonal_cycle_from_corr,
                season_months=season_months,
            )
