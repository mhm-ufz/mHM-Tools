"""Evaluate gridded model data against gridded reference data.

The module loads one or more input and reference datasets, crops or masks them
to a target domain, computes direct or bootstrap comparison metrics, and writes
spatial diagnostics, seasonal summaries, and result tables.

Authors
-------
- Simon Lüdke
"""

import array
import logging
import random
import re
from pathlib import Path
from typing import Iterable, Tuple

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from joblib import Parallel, delayed
from matplotlib.colors import BoundaryNorm
from mpl_toolkits.axes_grid1 import make_axes_locatable

from mhm_tools.common.file_handler import (
    ChunkType,
    get_coord_values,
    get_dataset_from_path,
    get_xarray_ds_from_file,
    write_xarray_to_file,
)
from mhm_tools.common.logger import ErrorLogger, log_arguments, log_errors
from mhm_tools.common.metrics.metrics_handler import create_results_csv
from mhm_tools.common.netcdf import generate_bounds_for_all_coords
from mhm_tools.common.resolution_handler import Resolution
from mhm_tools.common.utils import cut_to_filled_area
from mhm_tools.common.xarray_utils import (
    crop_ds,
    get_clim_from_ds,
    get_coord_key,
    get_ds_extend,
    get_overlapping_time_slice,
    spearman_correlation,
    timedelta_to_alias,
)

logger = logging.getLogger(__name__)


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


def spearman_spatial(data1, data2):
    """Calculate pixel-wise Spearman correlation maps for two DataArrays.

    Both inputs must have shape (12, Y, X): one climatology value per month.
    """
    if len(np.shape(data1)) != len(np.shape(data2)) or len(np.shape(data1)) != 3:
        with ErrorLogger(logger):
            msg = "Wrong shape for spatial spearman correlation!"
            raise ValueError(msg)
    res = np.full(np.shape(data1[0]), np.nan)
    pval = np.full(np.shape(data1[0]), np.nan)
    for i, row in enumerate(data1[0]):
        for j, _col in enumerate(row):
            sp_corr, sp_pval = spearman_correlation(data1[:, i, j], data2[:, i, j])
            res[i, j] = sp_corr
            pval[i, j] = sp_pval
    return res, pval


def spearman_spatial_joblib(
    data1: np.ndarray, data2: np.ndarray, spearman_correlation, n_jobs: int = -1
) -> Tuple[np.ndarray, np.ndarray]:
    """Parallel pixel-wise Spearman correlation over two arrays of shape (T, Y, X).

    Parameters
    ----------
    data1, data2 : ndarray, shape (T, Y, X)
        The two time-series stacks to correlate.
    spearman_correlation : Callable
        A function f(a: 1D, b: 1D) -> (rho, pval).
    n_jobs : int
        Number of parallel workers (-1 = all CPUs).

    Returns
    -------
    res : ndarray, shape (Y, X)
        Spearman rho for each pixel.
    pval : ndarray, shape (Y, X)
        Two-tailed p-value for each pixel.
    """
    # materialize once to avoid expensive lazy per-pixel loads
    data1 = np.asarray(data1)
    data2 = np.asarray(data2)

    # get spatial shape
    _, ny, nx = data1.shape

    # pre-allocate outputs
    res = np.full((ny, nx), np.nan, dtype=np.float32)
    pval = np.full((ny, nx), np.nan, dtype=np.float32)

    # list of all pixel indices
    indices = [(i, j) for i in range(ny) for j in range(nx)]

    # worker for a single pixel
    def _worker(i, j):
        rho, p = spearman_correlation(data1[:, i, j], data2[:, i, j])
        return i, j, rho, p

    if n_jobs == 1:
        for i, j in indices:
            rho, p = spearman_correlation(data1[:, i, j], data2[:, i, j])
            res[i, j] = rho
            pval[i, j] = p
        return res, pval

    # dispatch in parallel
    results = Parallel(n_jobs=n_jobs)(delayed(_worker)(i, j) for i, j in indices)

    # scatter results back
    for i, j, rho, p in results:
        res[i, j] = rho
        pval[i, j] = p

    return res, pval


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
    if clim is None:
        data_reduced = data.groupby("time.month") - clim
        std = data_reduced.std(dim="time", skipna=True)
    else:
        std = data.std(dim="time", skipna=True)

    # Return as DataArray with appropriate coordinates
    if type(std) is array and len(std.shape) == 2:
        return xr.DataArray(
            std,
            coords={
                "lat": get_coord_values(ds, lat=True),
                "lon": get_coord_values(ds, lon=True),
            },
            dims=["lat", "lon"],
        )
    return std


def get_file_stats(
    ds_in,
    input_var,
    factor=1,
    coordinate_slice=None,
    output_path=None,
    avaiable_years=None,
    direct_comp=False,
):
    """Get statistics for one file."""
    # logger.debug(f"Get file stats {file}")

    # Apply coordinate slicing if needed
    logger.debug(f"before cropping the file {ds_in}")
    lat_key = get_coord_key(ds_in, lat=True)
    lon_key = get_coord_key(ds_in, lon=True)
    # make sure that latitude order is from highest to lowest value
    if ds_in[lat_key].shape[0] > 1 and ds_in[lat_key][1] > ds_in[lat_key][0]:
        ds_croped = ds_in.isel(lat=slice(None, None, -1))
    else:
        ds_croped = ds_in
    if coordinate_slice is not None:
        ds_croped = ds_croped.sel(
            {lat_key: coordinate_slice["lat"], lon_key: coordinate_slice["lon"]}
        )
    if avaiable_years is not None:
        ds_croped = ds_croped.sel(time=ds_croped.time.dt.year.isin(avaiable_years))

    # Calculate climatology and standard deviation along the time dimension
    clim = get_clim_from_ds(ds_croped, input_var, factor)
    std = get_std_from_ds(ds_croped, input_var, clim, factor)

    mean = ds_croped[input_var].mean(dim="time", skipna=True) * factor

    # Construct the output dataset with lazy evaluations
    output = xr.Dataset(
        {"clim": clim, "std": std, "mean": mean},
        coords={
            "month": np.arange(1, 13, 1),
            "lat": get_coord_values(ds_croped, lat=True),
            "lon": get_coord_values(ds_croped, lon=True),
        },
    )
    output = generate_bounds_for_all_coords(output)
    if direct_comp:
        ts = ds_croped[input_var] * factor
        ts.name = "time_series"
        output = xr.merge([output, ts])
    if output_path is not None:
        write_xarray_to_file(ds=output, file_path=output_path)
    return output


def apply_spatial_mask(ds, mask_da):
    """Mask all spatial data variables in a dataset with a 2D mask DataArray."""
    if mask_da is None:
        return ds
    lat_key_ds = get_coord_key(ds, lat=True)
    lon_key_ds = get_coord_key(ds, lon=True)
    lat_key_mask = get_coord_key(mask_da, lat=True)
    lon_key_mask = get_coord_key(mask_da, lon=True)

    # Reduce to a 2D mask over lat/lon if extra dims exist.
    mask_2d = mask_da
    extra_dims = [d for d in mask_2d.dims if d not in [lat_key_mask, lon_key_mask]]
    for dim in extra_dims:
        mask_2d = mask_2d.isel({dim: 0}, drop=True)

    target_coords = {lat_key_mask: ds[lat_key_ds], lon_key_mask: ds[lon_key_ds]}
    if (
        mask_2d.sizes.get(lat_key_mask, 0) == 0
        or mask_2d.sizes.get(lon_key_mask, 0) == 0
    ):
        return ds
    mask_on_ds = mask_2d.interp(target_coords, method="nearest")
    mask_on_ds = mask_on_ds.rename({lat_key_mask: lat_key_ds, lon_key_mask: lon_key_ds})
    valid_mask = np.isfinite(mask_on_ds) & (mask_on_ds > 0)

    out = ds.copy()
    for var_name in out.data_vars:
        da = out[var_name]
        if lat_key_ds in da.dims and lon_key_ds in da.dims:
            out[var_name] = da.where(valid_mask)

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
    """Combine the statistics calculated for subsets into one."""
    total_count = sum(count for _, _, count, _, _ in results)
    if total_count == 0:
        msg = "Total count of number of results is 0"
        with ErrorLogger(logger):
            raise ValueError(msg)
    total_mean = sum(mean * count for mean, _, count, _, _ in results) / total_count
    total_M2 = sum(M2 for _, M2, _, _, _ in results)
    total_M2 += sum(
        count * (mean - total_mean) ** 2 for mean, _, count, _, _ in results
    )
    monthly_sums = sum(ms for _, _, _, ms, _ in results)
    monthly_counts = sum(mc for _, _, _, _, mc in results)
    return total_mean, total_M2, total_count, monthly_sums, monthly_counts


def get_stats_one_pass_subset(files, input_var, factor=1, coordinate_slice=None):
    """Compute running statistics from a list of monthly files.

    Iterates through NetCDF files (each containing one month's data for
    `input_var`), optionally applies a spatial slice, multiplies by `factor`,
    and updates running aggregates.

    Returns
    -------
    mean : ndarray
        Running mean over all time steps.
    sum_square_diff : ndarray
        Sum of squared deviations (for variance via Welford's algorithm).
    count : int
        Number of time steps processed.
    monthly_sums : ndarray
        Sum per calendar month, shape (12, ...).
    monthly_counts : ndarray
        Valid-count per calendar month, shape (12, ...).
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
        files[0], engine="netcdf4", force_decending_y=True
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
    count = 0  # xr.DataArray(np.ones(mean.shape, dtype=int).copy(), coords=mean.coords, dims=mean.dims).expand_dims(dim='time', axis=0)
    mean = np.zeros(da.shape[1:])
    sum_square_diff = np.zeros(da.shape[1:])
    monthly_sums = np.zeros((12, *da.shape[1:]))
    monthly_counts = np.zeros((12, *da.shape[1:]))
    for f, file in enumerate(files):
        with get_xarray_ds_from_file(
            file, engine="netcdf4", force_decending_y=True
        ) as ds:
            logger.info(f"timestep {count} in file {f+1} / {len(files)} from {file}")
            if coordinate_slice is not None:
                lat_key = get_coord_key(ds, lat=True)
                lon_key = get_coord_key(ds, lon=True)
                da = ds.sel(
                    {lat_key: coordinate_slice["lat"], lon_key: coordinate_slice["lon"]}
                )[input_var]
            else:
                da = ds[input_var]
            for _time_value, sub in da.groupby("time", squeeze=False):
                data_slice = sub.isel(time=0)
                try:
                    data_values = data_slice.values * factor
                    # logger.debug(f"{count} - {np.shape(data_values)}")
                    count += 1
                    delta = data_values - mean
                    mean += delta / count
                    delta2 = data_values - mean
                    sum_square_diff += delta * delta2
                    # climatology
                    month = int(data_slice.time.dt.month.item()) - 1
                    monthly_sums[month] += data_slice.fillna(0).values * factor
                    monthly_counts[month] += ~np.isnan(data_slice.values)
                except Exception as e:
                    logger.error(data_slice)
                    with ErrorLogger(logger):
                        raise e
    logger.debug(
        f"{np.nanmean(mean)}, {np.nanmean(sum_square_diff)}, {count}, {np.nanmean(monthly_sums)}, {np.nanmean(monthly_counts)}"
    )
    return mean, sum_square_diff, count, monthly_sums, monthly_counts


def split_file_list(file_list, n_processes):
    """Split a list into sublists."""
    file_list = list(file_list)
    if n_processes > 1:
        return [
            subset
            for subset in (file_list[i::n_processes] for i in range(n_processes))
            if subset
        ]
    return file_list


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
):
    """Compute streaming statistics from monthly/yearly files.

    Reads one file at a time and updates running aggregates to produce mean,
    standard deviation, and monthly climatology. Optionally slices coordinates,
    applies a multiplicative factor, supports bootstrapping over years, and can
    write the result to disk.
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
        delayed(get_stats_one_pass_subset)(file_subset, var, factor, coordinate_slice)
        for file_subset in file_subsets
    )
    logger.info("combining results...")
    mean, sum_square_diff, count, monthly_sums, monthly_counts = combine_results(
        subset_results
    )
    logger.debug(
        f"{mean.mean()}, {sum_square_diff.mean()}, {count}, {monthly_sums.mean()}, {monthly_counts.mean()}"
    )
    variance = sum_square_diff / (count - 1)
    std_dev = np.sqrt(variance)
    monthly_sums = np.where(monthly_counts > 0, monthly_sums, np.nan)
    monthly_counts = np.where(monthly_counts > 0, monthly_counts, np.nan)
    climatology = monthly_sums / monthly_counts
    climatology = np.where(monthly_counts > 0, climatology, np.nan)
    with get_xarray_ds_from_file(
        files[0], engine="netcdf4", force_decending_y=True
    ) as ds_in:
        lat_key = get_coord_key(ds_in, lat=True)
        lon_key = get_coord_key(ds_in, lon=True)
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
    output = xr.Dataset(
        {"clim": clim, "std": std, "mean": mean},
        coords={"month": np.arange(1, 13, 1), "lat": lat, "lon": lon},
    )
    output = generate_bounds_for_all_coords(output)
    # Trigger computation if needed
    if output_path is not None:
        output_file = (
            output_path.parent / f"{output_path.stem}_{bootstrap_index}.nc"
            if bootstrap_index is not None
            else output_path
        )
        logger.info(f"Writing output to {output_file}")
        write_xarray_to_file(ds=output, file_path=output_file)
    return output


def plot_single_map(
    ax,
    values,
    diff_to_mean=None,
    center=1,
    vmin=0,
    vmax=1,
    cmap=plt.cm.coolwarm_r,
    # cmap = plt.cm.RdBu,
    bounds_type="fixed",
):
    """Plot a single map on a Matplotlib Axes.

    Handles bounds and colormap selection. Behavior by `bounds_type`:
    - "max": set vmin=1 - diff_to_mean and vmax=1 + diff_to_mean
    - "quantiles": set vmin/vmax to the 5th/95th percentiles of `values`
    - "fixed": set vmin=0.5 and vmax=1.5

    Returns
    -------
    im : AxesImage
        The image artist.
    bounds : ndarray
        Bin edges used by BoundaryNorm.
    extent : {"neither", "min", "max", "both"}
        Whether data extend beyond bounds.
    ticks : ndarray
        Tick centers for colorbar labels (every second bin center).
    """

    def _step_decimals(step):
        step_abs = abs(float(step))
        if step_abs == 0:
            return 0
        decimals = int(max(0, -np.floor(np.log10(step_abs))))
        if not np.isclose(step_abs * (10**decimals), round(step_abs * (10**decimals))):
            decimals += 1
        return min(decimals, 6)

    if bounds_type == "max" and diff_to_mean is not None:
        vmin = center - diff_to_mean
        vmax = center + diff_to_mean
    if bounds_type == "quantiles":
        vmin, vmax = (
            np.nanquantile(values, 0.05),
            np.nanquantile(values, 0.95),
        )
        if abs(vmax - vmin) < abs(vmax / 3) or vmin == vmax:
            vmin, vmax = (float(np.nanmin(values)), float(np.nanmax(values)))
        if abs(vmax - vmin) < abs(vmax / 3) or vmin == vmax:
            vmin, vmax = (
                vmin - abs(vmin / 3),
                vmax + abs(vmax / 3),
            )
    if bounds_type == "fixed":
        # vmin, vmax = 0.5, 1.5
        vmin, vmax = center - 0.5625, center + 0.5625

    values_np = np.asarray(values)
    if not np.isfinite(vmin) or not np.isfinite(vmax):
        finite_values = values_np[np.isfinite(values_np)]
        if finite_values.size == 0:
            vmin, vmax = 0.0, 1.0
        else:
            vmin, vmax = float(np.nanmin(finite_values)), float(
                np.nanmax(finite_values)
            )
    if np.isclose(vmax, vmin):
        delta = max(abs(vmax), 1.0) * 0.5
        vmin, vmax = vmin - delta, vmax + delta

    target_bins = 9
    step = (vmax - vmin) / max(target_bins, 1)
    if not np.isfinite(step) or step <= 0:
        step = 1.0
    tick_anchor = center if bounds_type in {"fixed", "max"} else 0.0
    if not (vmin <= tick_anchor <= vmax):
        tick_anchor = 0.0 if (vmin < 0 < vmax) else vmin

    centers = np.linspace(vmin + step / 2, vmax - step / 2, target_bins)
    if vmin <= tick_anchor <= vmax and centers.size > 0:
        anchor_idx = int(np.argmin(np.abs(centers - tick_anchor)))
        centers = centers + (tick_anchor - centers[anchor_idx])

    bounds = np.concatenate(
        (
            [centers[0] - step / 2],
            0.5 * (centers[:-1] + centers[1:]),
            [centers[-1] + step / 2],
        )
    )
    if centers.size <= 2:
        ticks = centers
    else:
        anchor_idx = np.where(np.isclose(centers, tick_anchor))[0]
        start_idx = int(anchor_idx[0] % 2) if anchor_idx.size else 0
        ticks = centers[start_idx::2]
        if ticks.size == 0:
            ticks = centers
    decimals = _step_decimals(step)
    ticks = np.round(ticks, decimals=decimals)
    bounds = np.round(bounds, decimals=decimals + 1)

    extent = "neither"
    if np.nanquantile(values_np, 0.96) > bounds[-1]:
        extent = "max"
    if np.nanquantile(values_np, 0.049) < bounds[0]:
        extent = "min" if extent == "neither" else "both"

    norm = BoundaryNorm(bounds, cmap.N)
    im = ax.imshow(values_np, cmap=cmap, norm=norm)
    return im, bounds, extent, ticks


def round_sensibly(value):
    """Round map half-range to sensible steps and return decimals for labels.

    Returns
    -------
    tuple[float, int]
        (rounded_value, round_dec)
    """
    value = float(abs(value))
    if not np.isfinite(value) or value == 0:
        return 1e-6, 6

    thresholds = [
        (1.4, 2, 2),  # step 0.5
        (0.15, 5, 2),  # step 0.2
        (0.015, 50, 3),  # step 0.02
        (0.0015, 500, 4),  # step 0.002
        (0.00015, 5000, 5),  # step 0.0002
        (0.0, 50000, 6),  # step 0.00002
    ]
    for threshold, scale, round_dec in thresholds:
        if value > threshold:
            rounded = round(value * scale) / scale
            rounded = max(rounded, 1 / scale)
            return rounded, round_dec
    return value, 6


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


def resample_to_target_freq(
    ds_input: xr.Dataset, ds_ref: xr.Dataset, target_freq
) -> Tuple[xr.Dataset, xr.Dataset]:
    """Resample both datasets to the provided target freq."""
    hours_in, alias_in = timedelta_to_alias(ds_input)
    hours_ref, alias_ref = timedelta_to_alias(ds_ref)

    if target_freq != alias_ref:
        # input is coarser (e.g. monthly) → bring ref up to that
        logger.info(f"Resampling ref from {alias_ref} to {target_freq}")
        ds_ref = ds_ref.resample(time=target_freq).mean()
    if target_freq != alias_in:
        # ref is coarser → bring input up to that
        logger.info(f"Resampling input from {alias_in} to {target_freq}")
        ds_input = ds_input.resample(time=target_freq).mean()

    # Normalize anchors so both datasets share identical timestamps.
    ds_input = normalize_time_axis(ds_input, target_freq)
    ds_ref = normalize_time_axis(ds_ref, target_freq)

    # finally, force them onto exactly the same time-axis
    ds_input, ds_ref = xr.align(ds_input, ds_ref, join="inner")
    # logger.debug(f"Input file after align {ds_input}")
    return ds_input, ds_ref


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


def normalize_time_axis(ds: xr.Dataset, alias: str) -> xr.Dataset:
    """Normalize time stamps to a consistent anchor for the given frequency alias."""

    def _period_timestamp_index(period_freq: str, timestamp_freq: str):
        time_index = ds.indexes.get("time")
        if time_index is None:
            return None
        try:
            period_index = time_index.to_period(period_freq)
        except Exception:
            try:
                period_index = time_index.to_datetimeindex().to_period(period_freq)
            except Exception:
                return None
        try:
            return period_index.to_timestamp(timestamp_freq)
        except Exception:
            return period_index.to_timestamp(freq=timestamp_freq)

    alias = alias.upper()
    if "time" not in ds.coords:
        ds_out = ds
    elif alias.endswith("H"):
        try:
            ds_out = ds.assign_coords(time=ds.time.dt.floor("h"))
        except ValueError:
            ds_out = ds.assign_coords(time=ds.time.dt.floor("h"))
    elif alias == "D":
        ds_out = ds.assign_coords(time=ds.time.dt.floor("D"))
    elif alias.startswith("W"):
        new_time = _period_timestamp_index("W-MON", "W-MON")
        ds_out = ds.assign_coords(time=new_time) if new_time is not None else ds
    elif alias == "ME":
        new_time = _period_timestamp_index("M", "M")
        ds_out = ds.assign_coords(time=new_time) if new_time is not None else ds
    elif alias == "MS":
        new_time = _period_timestamp_index("M", "MS")
        ds_out = ds.assign_coords(time=new_time) if new_time is not None else ds
    else:
        ds_out = ds
    return ds_out


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
):
    """Create a 2x2 figure of relative mean, relative std, Spearman correlation, and seasonal means.

    Generates four subplots: (1) relative temporal mean, (2) relative temporal
    standard deviation, (3) Spearman correlation of climatologies, and (4) monthly
    seasonal means for both datasets.
    """
    rel_mean = np.where(rel_mean == np.inf, np.nan, rel_mean)
    rel_mean = np.where(rel_mean < 0, np.nan, rel_mean)
    rel_std = np.where(rel_std == np.inf, np.nan, rel_std)
    rel_std = np.where(rel_std < 0, np.nan, rel_std)
    fig, axes = plt.subplots(2, 2, figsize=(10.5, 4.68))
    if input_name is not None and ref_name is not None:
        title = f"Comparision {input_name} with {ref_name}"
    if overlapping_years is not None and len(overlapping_years) > 1:
        title += f" for years {overlapping_years[0]}-{overlapping_years[-1]}"
    fig.suptitle(title, fontweight="normal", fontsize="x-large")

    # Set common colormap and normalization limits for mean_et and mean_aet
    mean_diff_1 = max(np.abs(1 - np.nanmin(rel_mean)), np.abs(1 - np.nanmax(rel_mean)))
    im0, bounds0, extend0, ticks0 = plot_single_map(
        axes[0, 0], rel_mean, mean_diff_1, bounds_type="fixed"
    )
    axes[0, 0].set_title(
        f"Relative temporal Mean (median={np.nanmedian(rel_mean):.2f})"
    )
    std_diff_1 = max(np.abs(1 - np.nanmin(rel_std)), np.abs(1 - np.nanmax(rel_std)))
    im1, bounds1, extend1, ticks1 = plot_single_map(
        axes[0, 1], rel_std, std_diff_1, bounds_type="quantiles"
    )
    axes[0, 1].set_title(
        f"Relative temporal Standarddeviation (median={np.nanmedian(rel_std):.2f})"
    )

    im2, bounds2, extend2, ticks2 = plot_single_map(
        axes[1, 0],
        spearman,
        vmin=np.nanmin(spearman),
        vmax=1,
        cmap=plt.cm.viridis_r,
        bounds_type="quantiles",
    )

    axes[1, 0].set_title(f"Spearman Correlation (median={np.nanmedian(spearman):.2f})")

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

    divider0 = make_axes_locatable(axes[0, 0])
    cax = divider0.append_axes("right", size="5%", pad=0.1)
    fig.colorbar(
        im0, cax=cax, label="", boundaries=bounds0, extend=extend0, ticks=ticks0
    )

    divider1 = make_axes_locatable(axes[0, 1])
    cax2 = divider1.append_axes("right", size="5%", pad=0.1)
    fig.colorbar(
        im1, cax=cax2, label="", boundaries=bounds1, extend=extend1, ticks=ticks1
    )

    divider2 = make_axes_locatable(axes[1, 0])
    cax = divider2.append_axes("right", size="5%", pad=0.1)
    fig.colorbar(
        im2, cax=cax, label="", boundaries=bounds2, extend=extend2, ticks=ticks2
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
    file_name = f"et_map_{input_name}_{ref_name}.png".replace(" ", "_")
    plt.savefig(output_path / file_name, dpi=800)
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
):
    """Create a 3-panel plot with relative mean, mean difference, and climatology."""
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

    if input_name is not None and ref_name is not None:
        title = f"Comparision {input_name} with {ref_name}"
    if overlapping_years is not None and len(overlapping_years) > 1:
        title += f" for years {overlapping_years[0]}-{overlapping_years[-1]}"
    fig.suptitle(title, fontweight="normal", fontsize="x-large")

    mean_diff_1 = max(np.abs(1 - np.nanmin(rel_mean)), np.abs(1 - np.nanmax(rel_mean)))
    im0, bounds0, extend0, ticks0 = plot_single_map(
        ax_rel_mean, rel_mean, mean_diff_1, bounds_type="fixed"
    )
    ax_rel_mean.set_title(
        f"a) Relative Mean (median={np.nanmedian(rel_mean):.2f}, mean={total_rel_mean:.2f})"
    )

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

    divider0 = make_axes_locatable(ax_rel_mean)
    cax = divider0.append_axes("right", size="5%", pad=0.1)
    fig.colorbar(
        im0, cax=cax, label="", boundaries=bounds0, extend=extend0, ticks=ticks0
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

    file_name = f"et_map_bias_only_{input_name}_{ref_name}.png".replace(" ", "_")
    plt.savefig(output_path / file_name, dpi=800)
    logger.info(f"created et_map {output_path / file_name}")


@log_errors(raise_exceptions=True)
def plot_map_global_climate(
    rel_mean,
    rel_std,
    input_name,
    ref_name,
    output_path,
    overlapping_years=None,
):
    """Create a plot showing relative mean, relative std and seasonality."""
    rel_mean = np.where(rel_mean == np.inf, np.nan, rel_mean)
    rel_std = np.where(rel_std == np.inf, np.nan, rel_std)
    fig, axes = plt.subplots(2, 1, figsize=(10.5, 6.2))
    if input_name is not None and ref_name is not None:
        title = f"Comparision {input_name} with {ref_name}"
    if overlapping_years is not None and len(overlapping_years) > 1:
        title += f" for years {overlapping_years[0]}-{overlapping_years[-1]}"
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
    file_name = f"et_map_global_climate_{input_name}_{ref_name}.png".replace(" ", "_")
    plt.savefig(output_path / file_name, dpi=800)
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
):
    """Create a plot showing relative mean and monthly climatology difference."""
    rel_mean = np.where(rel_mean == np.inf, np.nan, rel_mean)
    diff_mean = np.asarray(np.where(diff_mean == np.inf, np.nan, diff_mean))
    fig = plt.figure(figsize=(10.5, 6.2))
    gs = fig.add_gridspec(2, 4, height_ratios=[1.0, 1.0])
    ax_rel_mean = fig.add_subplot(gs[1, 1:3])
    ax_diff = fig.add_subplot(gs[0, 1:3])
    if input_name is not None and ref_name is not None:
        title = f"Comparision {input_name} with {ref_name}"
    if overlapping_years is not None and len(overlapping_years) > 1:
        title += f" for years {overlapping_years[0]}-{overlapping_years[-1]}"
    fig.suptitle(title, fontweight="normal", fontsize="x-large")

    vmin = np.nanquantile(rel_mean, 0.01)
    vmax = np.nanquantile(rel_mean, 0.99)
    mean_diff_1 = max(abs(vmin), abs(vmax))
    mean_diff_1, round_dec = round_sensibly(mean_diff_1)
    logger.debug(
        f"mean_diff_1={mean_diff_1}, vmin={vmin}, vmax={vmax}, "
        f"round_dec={round_dec}"
    )
    im0, bounds0, extend0, ticks0 = plot_single_map(
        ax_rel_mean, rel_mean, mean_diff_1, bounds_type="max", center=0
    )
    title_rel_mean = (
        f"b) Relative Mean Difference (median={np.nanmedian(rel_mean):.{round_dec}f}"
    )
    if total_rel_mean is not None:
        title_rel_mean += f", mean={total_rel_mean:.{round_dec}f}"
    title_rel_mean += ")"
    ax_rel_mean.set_title(title_rel_mean)

    vmin = np.nanquantile(diff_mean, 0.01)
    vmax = np.nanquantile(diff_mean, 0.99)
    diff_diff_1 = max(abs(vmin), abs(vmax))
    diff_diff_1, round_dec = round_sensibly(diff_diff_1)
    logger.debug(
        f"diff_diff_1={diff_diff_1}, vmin={vmin}, vmax={vmax}, "
        f"round_dec={round_dec}"
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
    fig.colorbar(
        im0, cax=cax0, label="%", boundaries=bounds0, extend=extend0, ticks=ticks0
    )

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

    file_name = f"et_map_global_climate2_{input_name}_{ref_name}.png".replace(" ", "_")
    plt.savefig(output_path / file_name, dpi=800)
    logger.info(f"created et_map {output_path / file_name}")


@log_errors(raise_exceptions=True)
def plot_map_local_climate(
    input_clim,
    ref_clim,
    input_name,
    ref_name,
    output_path,
    overlapping_years=None,
):
    """Create monthly local-climate maps for relative and absolute differences.

    Produces two figures with 12 small maps (one per month):
    - Relative climatology difference in percent
    - Absolute climatology difference in native units (e.g. mm/day)
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

    def _plot_monthly_panels(values, panel_title, output_file_name, colorbar_label):
        values = np.where(values == np.inf, np.nan, values)
        values = np.where(values == -np.inf, np.nan, values)
        finite_values = values[np.isfinite(values)]
        if finite_values.size == 0:
            diff_to_mean = 1e-6
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
            title = f"Comparision {input_name} with {ref_name}"
        else:
            title = "Local Climate Comparison"
        if overlapping_years is not None and len(overlapping_years) > 1:
            title += f" for years {overlapping_years[0]}-{overlapping_years[-1]}"
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
                bounds_type="max",
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
        cbar.ax.tick_params(labelsize=8)
        fig.subplots_adjust(left=0.03, right=0.94, bottom=0.04, top=0.89)
        plt.savefig(output_path / output_file_name, dpi=400)
        logger.info(f"created et_map {output_path / output_file_name}")

    rel_file_name = f"et_map_local_climate_rel_{input_name}_{ref_name}.png".replace(
        " ", "_"
    )
    abs_file_name = f"et_map_local_climate_abs_{input_name}_{ref_name}.png".replace(
        " ", "_"
    )
    _plot_monthly_panels(
        rel_monthly,
        "Monthly Relative Climatology Difference",
        rel_file_name,
        "Relative Climatology Difference [%]",
    )
    _plot_monthly_panels(
        abs_monthly,
        "Monthly Absolute Climatology Difference",
        abs_file_name,
        "Absolute Climatology Difference [mm/day]",
    )


def create_map_from_output(
    output_path, input_name, ref_name, bias_only=False, global_climate=False
):
    """Read in statistics netcdf and create a map plots from it."""
    file = get_rel_stat_file(output_path, input_name, ref_name)
    logger.info(f"Plotting data from {file}")
    with get_xarray_ds_from_file(file, force_decending_y=True) as ds:
        rel_mean = ds["rel_mean"]
        rel_mean = np.where(rel_mean == np.inf, np.nan, rel_mean)
        rel_std = ds.get("rel_std", None)
        rel_std = (
            np.where(rel_std == np.inf, np.nan, rel_std)
            if rel_std is not None
            else None
        )
        spearman = ds.get("spearman", None)
        input_clim = (
            ds[f"{input_name}_clim"] if f"{input_name}_clim" in ds else ds["input_clim"]
        )
        ref_clim = (
            ds[f"{ref_name}_clim"] if f"{ref_name}_clim" in ds else ds["ref_clim"]
        )
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
            input_name=input_name,
            ref_name=ref_name,
            output_path=output_path,
        )
    elif global_climate:
        rel_mean_diff = (
            100
            * (np.nanmean(input_clim, axis=0) - np.nanmean(ref_clim, axis=0))
            / np.nanmean(ref_clim, axis=0)
        )
        rel_mean_diff = np.where(
            np.nanmax(ref_clim, axis=0) < 0.1, np.nan, rel_mean_diff
        )
        plot_map_global_climate2(
            rel_mean=rel_mean_diff,
            diff_mean=np.nanmean(input_clim, axis=0) - np.nanmean(ref_clim, axis=0),
            input_name=input_name,
            ref_name=ref_name,
            output_path=output_path,
            total_rel_mean=np.nanmean(input_clim) / np.nanmean(ref_clim),
        )
        plot_map_local_climate(
            input_clim=input_clim,
            ref_clim=ref_clim,
            input_name=input_name,
            ref_name=ref_name,
            output_path=output_path,
        )
    else:
        plot_map_bias_only(
            rel_mean=rel_mean,
            ref_clim=ref_clim,
            input_clim=input_clim,
            input_name=input_name,
            ref_name=ref_name,
            output_path=output_path,
            total_rel_mean=np.nanmean(input_clim) / np.nanmean(ref_clim),
        )


def get_stats(
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
    file_name="*.*",
):
    """Get statistics dataset from a path to a file or directory with files."""
    logger.info(f"Get stats for {path}")
    if var is not None:
        if path.is_dir() and not direct_comp:
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
            )
        elif path.is_dir() or path.is_file():
            if path.is_file() and path.suffix == ".nc":
                chunking = available_mem is not None
                ds = get_xarray_ds_from_file(
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
                )
            if path.is_dir():
                file_list = get_files(
                    path, available_years=available_years, file_name=file_name
                )
                with get_dataset_from_path(
                    file_list, available_mem=available_mem, file_name=file_name
                ) as ds_in:
                    stats_ds = get_file_stats(
                        ds_in,
                        var,
                        factor,
                        coordinate_slice,
                        output_path=output_file,
                        avaiable_years=available_years,
                        direct_comp=direct_comp,
                    )
        else:
            msg = f"Path {path} is neither dir nor file."
            with ErrorLogger(logger):
                raise ValueError(msg)
    else:
        with get_xarray_ds_from_file(
            path, engine="netcdf4", force_decending_y=True
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
    masked_ds = apply_spatial_mask(stats_ds, mask_da)
    return generate_bounds_for_all_coords(masked_ds)


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
    mask_da=None,
    input_file_name=None,
    ref_file_name=None,
    result_metric="all",
):
    """Compare the two datasets."""
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
        file_name=input_file_name,
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
        file_name=ref_file_name,
    )
    logger.debug(f"ref ds: {ref}")
    # regrid spatial resoution
    # regridd to same spatial resolution
    if len(input["lat"].data) < 1 or len(input["lon"].data) < 1:
        logger.error("Input dataset has empty coordinate.")
    if len(ref["lat"].data) < 1 or len(ref["lon"].data) < 1:
        logger.error("Ref dataset has empty coordinate.")

    input, ref = crop_datasets_to_spatial_overlap(input, ref)
    input, ref = regridd_to_higher_spatial_resolution(input, ref)
    output_name = f"{input_name}-{ref_name}".replace(" ", "_")
    # compare and save statistics
    full_metrics = not bias_only and not global_climate
    with_std = not bias_only

    if direct_comp:
        input_ts, ref_ts = input["time_series"], ref["time_series"]

        # If we already know the target frequency, resample to that; otherwise pick the coarser one.
        if target_freq is not None:
            input_ts, ref_ts = resample_to_target_freq(input_ts, ref_ts, target_freq)
        else:
            input_ts, ref_ts = resample_to_coarser_calendar(input_ts, ref_ts)

        input_ts, ref_ts, time_slice = crop_data_to_overlapping_time(input_ts, ref_ts)
        overlapping_years = [time_slice.start.year, time_slice.stop.year]
        if input_ts.shape != ref_ts.shape:
            msg = f"Input and ref time_series shapes differ after resampling/cropping: {input_ts.shape} vs {ref_ts.shape}"
            with ErrorLogger(logger):
                raise ValueError(msg)
        pairwise_valid = np.isfinite(input_ts.values) & np.isfinite(ref_ts.values)
        input_ts = input_ts.where(pairwise_valid)
        ref_ts = ref_ts.where(pairwise_valid)
        logger.info(
            f"Creating data from timeseries with shape {input_ts.shape} and {ref_ts.shape}"
        )
        if input_ts.sizes.get("time", 0) == 0 or ref_ts.sizes.get("time", 0) == 0:
            msg = "No overlapping time steps after alignment; cannot compare input and reference."
            logger.error(msg)
            raise ValueError(msg)
        try:
            if full_metrics:
                input_ts_np = np.asarray(input_ts.values)
                ref_ts_np = np.asarray(ref_ts.values)
                spearman, spearman_pval = spearman_spatial_joblib(
                    input_ts_np, ref_ts_np, spearman_correlation, ncpus
                )
                create_results_csv(
                    map1=input_ts_np,
                    map2=ref_ts_np,
                    ds1_name=input_name,
                    ds2_name=ref_name,
                    out_dir=output_path,
                    out_name=output_name,
                    metric=result_metric,
                )
        except ValueError as ve:
            logger.error("Input and ref do not have the same temporal extent.")
            logger.info(input_ts.time)
            logger.info(ref_ts.time)
            raise ve
    elif full_metrics:
        logger.info("Calculating spearman correlation from seasonalities.")
        input_clim_np = np.asarray(input["clim"].values)
        ref_clim_np = np.asarray(ref["clim"].values)
        spearman, spearman_pval = spearman_spatial_joblib(
            input_clim_np, ref_clim_np, spearman_correlation, ncpus
        )
        create_results_csv(
            map1=input_clim_np,
            map2=ref_clim_np,
            ds1_name=input_name,
            ds2_name=ref_name,
            out_dir=output_path,
            out_name=output_name,
            metric=result_metric,
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
    input_clim = xr.DataArray(
        np.where(clim_valid, input["clim"].values, np.nan),
        coords={
            "month": np.arange(1, 13, 1),
            "lat": get_coord_values(input, lat=True),
            "lon": get_coord_values(input, lon=True),
        },
        dims=["month", "lat", "lon"],
    )
    ref_clim = xr.DataArray(
        np.where(clim_valid, ref["clim"].values, np.nan),
        coords={
            "month": np.arange(1, 13, 1),
            "lat": get_coord_values(input, lat=True),
            "lon": get_coord_values(input, lon=True),
        },
        dims=["month", "lat", "lon"],
    )
    rel_mean = rel_mean.where(np.isfinite(rel_mean) & (rel_mean >= 0))
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
        output["rel_std"] = rel_std
    if full_metrics:
        output["spearman"] = spearman
        output["spearman_pval"] = spearman_pval
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
    file_name = file_name.replace(" ", "_")
    if bootstrap_index is not None:
        file_name = output_path / f"{file_name}_{bootstrap_index}.nc"
    else:
        file_name = output_path / f"{file_name}.nc"

    write_xarray_to_file(ds=output, file_path=file_name)
    logger.info(f"Written output to {file_name}")
    if plot:
        if full_metrics:
            plot_map(
                rel_std=rel_std,
                rel_mean=rel_mean,
                spearman=spearman,
                ref_clim=ref_clim,
                input_clim=input_clim,
                input_name=input_name,
                ref_name=ref_name,
                output_path=output_path,
                overlapping_years=overlapping_years,
            )
        elif global_climate:
            rel_mean_diff = (
                100
                * (np.nanmean(input_clim, axis=0) - np.nanmean(ref_clim, axis=0))
                / np.nanmean(ref_clim, axis=0)
            )
            rel_mean_diff = np.where(
                np.nanmax(ref_clim, axis=0) < 0.1, np.nan, rel_mean_diff
            )
            plot_map_global_climate2(
                rel_mean=rel_mean_diff,
                diff_mean=np.nanmean(input_clim, axis=0) - np.nanmean(ref_clim, axis=0),
                input_name=input_name,
                ref_name=ref_name,
                output_path=output_path,
                overlapping_years=overlapping_years,
            )
            plot_map_local_climate(
                input_clim=input_clim,
                ref_clim=ref_clim,
                input_name=input_name,
                ref_name=ref_name,
                output_path=output_path,
            )
        else:
            plot_map_bias_only(
                rel_mean=rel_mean,
                ref_clim=ref_clim,
                input_clim=input_clim,
                input_name=input_name,
                ref_name=ref_name,
                output_path=output_path,
                overlapping_years=overlapping_years,
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
    mean_da = xr.DataArray(mean, dims=["bootstrap", "lat", "lon"])
    std_da = (
        xr.DataArray(std, dims=["bootstrap", "lat", "lon"]) if std is not None else None
    )
    spearman_da = (
        xr.DataArray(spearman, dims=["bootstrap", "lat", "lon"])
        if spearman is not None
        else None
    )
    input_clim_da = xr.DataArray(input_clim, dims=["bootstrap", "month", "lat", "lon"])
    ref_clim_da = xr.DataArray(ref_clim, dims=["bootstrap", "month", "lat", "lon"])

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

    def _coord_spacing(coord):
        if coord is None or coord.size < 2:
            return None
        diffs = np.diff(np.asarray(coord))
        diffs = np.abs(diffs[~np.isnan(diffs)])
        if diffs.size == 0:
            return None
        return float(np.nanmedian(diffs))

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
    coarse_res = _coord_spacing(coarse_ds.get("lat")) or _coord_spacing(
        coarse_ds.get("lon")
    )
    if coarse_res is None:
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
    the year slice.
    """
    logger.info("Determining available years.")
    # get all years from input data
    years_in = get_years_from_path(input_path, file_name=input_file_name)
    years_in.sort()
    logger.debug(f"Input years: {years_in}")

    # get all years from reference data
    years_ref = get_years_from_path(
        ref_path, raise_exception=False, file_name=ref_file_name
    )
    years_ref.sort()
    logger.debug(f"Ref years: {years_ref}")

    # find overlapping years
    years = []
    possible_years = years_in
    if not direct_comp and years_ref:
        possible_years.extend([y for y in years_ref if y not in possible_years])
        possible_years.sort()
    for year in possible_years:
        if year_slice.start is not None and year < year_slice.start:
            continue
        if year_slice.stop is not None and year > year_slice.stop:
            continue
        if not years_ref or year in years_ref or not direct_comp:
            years.append(year)
    return years


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
    direct_comp=True,
    n_bootstrap_years=None,
    n_bootstrap_selections=None,
    target_time_freq=None,
    bias_only=False,
    global_climate=False,
    only_plot=False,
    result_metric="all",
    avaiable_mem=None,
    n_cpus=1,
):
    """Validate a spatial variable by comparing dataset climatologies."""
    output_path = Path(output_path)
    input_path = input.path
    ref_path = ref.path

    if not output_path.is_dir():
        output_path.mkdir(parents=True)

    if only_plot and get_rel_stat_file(output_path, input.name, ref.name).is_file():
        create_map_from_output(
            output_path=output_path,
            input_name=input.name,
            ref_name=ref.name,
            bias_only=bias_only,
            global_climate=global_climate,
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

    available_years = get_available_years(
        input_path,
        ref_path,
        year_slice,
        direct_comp,
        input_file_name=input.file_name,
        ref_file_name=ref.file_name,
    )
    logger.info(f"Years {available_years} are available for comparison.")
    if not available_years:
        logger.error("Since no data is available the program is stoped.")
        return

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
                    output=output_path / output_name,
                    avaiable_years=available_years,  # keep parameter name as used elsewhere
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
                    mask_da=mask_da,
                    result_metric=result_metric,
                )
                for bootstrap_index in range(n_bootstrap_selections)
            )
        stat_files = [file for file in stat_files if file is not None]
        if stat_files:
            results = evaluate_boostraping_stat_files(
                stat_files, input_name=input.name, ref_name=ref.name
            )
            if (
                "spearman" in results
                and "rel_std" in results
                and not bias_only
                and not global_climate
            ):
                plot_map(
                    **results,
                    output_path=output_path,
                    input_name=input.name,
                    ref_name=ref.name,
                )
            elif global_climate:
                rel_mean_diff = (
                    100
                    * (
                        np.nanmean(results["input_clim"], axis=0)
                        - np.nanmean(results["ref_clim"], axis=0)
                    )
                    / np.nanmean(results["ref_clim"], axis=0)
                )
                rel_mean_diff = np.where(
                    np.nanmax(results["ref_clim"], axis=0) < 0.1, np.nan, rel_mean_diff
                )
                plot_map_global_climate2(
                    rel_mean=rel_mean_diff,
                    diff_mean=np.nanmean(results["input_clim"], axis=0)
                    - np.nanmean(results["ref_clim"], axis=0),
                    output_path=output_path,
                    input_name=input.name,
                    ref_name=ref.name,
                )
                plot_map_local_climate(
                    input_clim=results["input_clim"],
                    ref_clim=results["ref_clim"],
                    input_name=input.name,
                    ref_name=ref.name,
                    output_path=output_path,
                )
            else:
                plot_map_bias_only(
                    rel_mean=results["rel_mean"],
                    input_clim=results["input_clim"],
                    ref_clim=results["ref_clim"],
                    output_path=output_path,
                    input_name=input.name,
                    ref_name=ref.name,
                )
        else:
            logger.error("There are no statfiles created from the evaluation.")
    else:
        logger.info("Compare without bootstrapping.")
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
            mask_da=mask_da,
            result_metric=result_metric,
        )
