"""Compare a spatial variable between two datasets using the climatology of that variable."""

import array
import logging
import random
import re
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from joblib import Parallel, delayed
from matplotlib.colors import BoundaryNorm
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.stats import spearmanr

from mhm_tools.common.file_handler import (
    ChunkType,
    get_coord_values,
    get_xarray_ds_from_file,
)
from mhm_tools.common.logger import ErrorLogger, log_arguments, log_errors
from mhm_tools.common.xarray_utils import (
    get_coord_key,
    get_overlapping_time_slice,
    timedelta_to_alias,
)

logger = logging.getLogger(__name__)


def spearman_correlation(data1, data2):
    """Calculate Spearman rank correlation between two xarray DataArrays."""
    # Check that both arrays are of the same size and flatten them
    if data1.shape != data2.shape:
        with ErrorLogger(logger):
            msg = "Both DataArrays must have the same shape"
            raise ValueError(msg)
    data1 = data1.values.flatten()
    data2 = data2.values.flatten()
    # Calculate Spearman rank correlation using scipy
    corr, p_value = spearmanr(data1, data2)
    return corr, p_value


# def _spearman_1d(a: np.ndarray, b: np.ndarray):
#     """Compute Spearman’s ρ and p-value for two 1D arrays."""
#     # nan_policy='omit' will drop NaNs in the calculation
#     rho, p = spearmanr(a, b, nan_policy="omit")
#     return np.float32(rho), np.float32(p)


# def spearman_spatial(
#     ds1: xr.DataArray, ds2: xr.DataArray
# ) -> tuple[xr.DataArray, xr.DataArray]:
#     """
#     Compute a per-pixel Spearman correlation map between two DataArrays
#     of shape (time, y, x), leveraging Dask & xarray chunking.
#     Returns (rho, pvalue) each with dims (y, x).
#     """
#     # make sure they align on time
#     ds1, ds2 = xr.align(ds1, ds2)

#     rho, p = xr.apply_ufunc(
#         _spearman_1d,  # the 1D function
#         ds1,
#         ds2,  # inputs
#         input_core_dims=[["time"], ["time"]],
#         output_core_dims=[[], []],
#         vectorize=True,  # vectorize over y, x
#         dask="parallelized",  # dispatch one Dask task per chunk
#         output_dtypes=[np.float32, np.float32],
#     )
#     # give them nice names
#     rho.name = "spearman_rho"
#     p.name = "spearman_p"
#     return rho, p


def spearman_spatial(data1, data2):
    """Calculate maps of Spearman rank correlation between two xarray DataArrays of shape(12,n,m)."""
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
) -> tuple[np.ndarray, np.ndarray]:
    """
    Parallel pixel‐wise Spearman correlation over two arrays of shape (T, Y, X).

    Parameters
    ----------
    data1, data2 : ndarray, shape (T, Y, X)
        The two time‐series stacks to correlate.
    spearman_correlation : Callable
        A function f(a: 1D, b: 1D) -> (rho, pval).
    n_jobs : int
        Number of parallel workers (−1 = all CPUs).

    Returns
    -------
    res : ndarray, shape (Y, X)
        Spearman ρ for each pixel.
    pval : ndarray, shape (Y, X)
        Two‐tailed p‐value for each pixel.
    """
    # get spatial shape
    _, ny, nx = data1.shape

    # pre‐allocate outputs
    res = np.full((ny, nx), np.nan, dtype=np.float32)
    pval = np.full((ny, nx), np.nan, dtype=np.float32)

    # list of all pixel indices
    indices = [(i, j) for i in range(ny) for j in range(nx)]

    # worker for a single pixel
    def _worker(i, j):
        rho, p = spearman_correlation(data1[:, i, j], data2[:, i, j])
        return i, j, rho, p

    # dispatch in parallel
    results = Parallel(n_jobs=n_jobs)(delayed(_worker)(i, j) for i, j in indices)

    # scatter results back
    for i, j, rho, p in results:
        res[i, j] = rho
        pval[i, j] = p

    return res, pval


def climatology(data):
    """Calculate the climatology from a xarray DataArray."""
    if "time" not in data.dims or data.sizes["time"] == 0:
        msg = "Input data for climatology calculation has no valid time dimension."
        with ErrorLogger(logger):
            raise ValueError(msg)
    # group into monthly mean data
    data_clim = data.groupby("time.month").mean(dim="time", skipna=True)
    # Ensure the climatology has all 12 months, filling missing months with NaNs
    return data_clim.reindex(month=np.arange(1, 13), fill_value=np.nan)


def get_clim_from_ds(ds, input_var=None, factor=1):
    """Calculate climatology from DataSet with variable or DataArray while mulitplying with a provided factor."""
    data = ds * factor if input_var is None else ds[input_var] * factor
    return climatology(data)


def get_std_from_ds(ds, input_var=None, clim=None, factor=1):
    """Calculate maps of temporal standard deviation from an DataArray.

    If a climatology is provided the timeseries can be detrended by seasonality.
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
    if ds_in[lat_key][1] > ds_in[lat_key][0]:
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
    if direct_comp:
        ts = ds_croped[input_var] * factor
        ts.name = "time_series"
        output = xr.merge([output, ts])
    if output_path is not None:
        output.to_netcdf(output_path)
    return output


def get_files(path, n_bootstrap_years=None, available_years=None, file_name="*.*"):
    """Recursevely find all netcdf files in directory."""
    nc_files = []
    # Search for .nc files at each depth level
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
        nc_files.extend(folder_path.rglob(file_name))
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
    """Take a list of files with all containing data for one month and creating statisitcs while reading them one by one."""
    da = None
    if not isinstance(files, Iterable):
        # logger.warning(f"Files not a list of files but one file {files}.")
        files = [files]
    logger.debug(files)
    with get_xarray_ds_from_file(files[0], engine="netcdf4") as ds:
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
        with get_xarray_ds_from_file(file, engine="netcdf4") as ds:
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
    if n_processes > 1:
        return [file_list[i::n_processes] for i in range(n_processes)]
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
    """Create dataset statistics by reading in one monthly or yearly file at a time and updating the statistics."""
    if path.is_dir():
        files = get_files(
            path,
            n_bootstrap_years=n_bootstrap_years,
            available_years=available_years,
            file_name=file_name,
        )
    logger.debug(f"List of files: {files}")
    file_subsets = split_file_list(files, ncpus) if ncpus > 1 else [files]
    logger.info("creating statistics...")
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
    with get_xarray_ds_from_file(files[0], engine="netcdf4") as ds_in:
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
    # Trigger computation if needed
    if output_path is not None:
        output_file = (
            output_path.parent / f"{output_path.stem}_{bootstrap_index}.nc"
            if bootstrap_index is not None
            else output_path
        )
        logger.info(f"Writing output to {output_file}")
        output.to_netcdf(output_file)
    return output


def plot_single_map(
    ax,
    values,
    diff_to_mean=None,
    vmin=0,
    vmax=1,
    cmap=plt.cm.coolwarm,
    bounds_type="fixed",
):
    """Plot one map to an matplotlib axis, taking care of the bounds and colormap."""
    n_bins = 10
    if bounds_type == "max" and diff_to_mean is not None:
        vmin = 1 - diff_to_mean
        vmax = 1 + diff_to_mean
    if bounds_type == "quantiles":
        vmin, vmax = (
            np.nanquantile(values, 0.05),
            np.nanquantile(values, 0.95),
        )
    if bounds_type == "fixed":
        vmin, vmax = 0.5, 1.5
    bounds = np.linspace(vmin, vmax, n_bins + 1)
    bounds = [np.round(b, 2) for b in bounds]

    extent = "neither"
    if np.nanquantile(values, 0.75) > vmax:
        extent = "max"
    if np.nanquantile(values, 0.25) < vmin:
        extent = "min" if extent == "neither" else "both"

    norm = BoundaryNorm(bounds, cmap.N)
    im = ax.imshow(values, cmap=cmap, norm=norm)
    return im, bounds, extent


def resample_to_coarser_calendar(
    ds_input: xr.Dataset, ds_ref: xr.Dataset
) -> tuple[xr.Dataset, xr.Dataset]:
    """
    Resampler the dataset with higher temporal resolution to the resolution of the other.

    Compare the two datasets’ median time‐steps, turn them into pandas/xarray
    freq aliases, and then resample the *finer* one up to the *coarser* one
    using calendar‐aware frequencies (e.g. 'M' not '720H').
    """
    hours_in, alias_in = timedelta_to_alias(ds_input)
    hours_ref, alias_ref = timedelta_to_alias(ds_ref)

    if hours_in > hours_ref:
        # input is coarser (e.g. monthly) → bring ref up to that
        logger.info(f"Resampling ref from {alias_ref} to {alias_in}")
        ds_ref = ds_ref.resample(time=alias_in).mean()
    elif hours_ref > hours_in:
        # ref is coarser → bring input up to that
        logger.info(f"Resampling input from {alias_in} to {alias_ref}")
        ds_input = ds_input.resample(time=alias_ref).mean()
    else:
        # same resolution, nothing to do
        logger.info(f"Both are already {alias_in}")

    # finally, force them onto exactly the same time‐axis
    ds_input, ds_ref = xr.align(ds_input, ds_ref)
    return ds_input, ds_ref


def crop_data_to_overlapping_time(input_ds, ref_ds):
    """Crop data to overlapping time."""
    time_slice = get_overlapping_time_slice(input_ds, ref_ds)
    # Slice both datasets to that time range
    input_ds = input_ds.sel(time=time_slice)
    ref_ds = ref_ds.sel(time=time_slice)
    return input_ds, ref_ds


@log_errors()
def plot_map(
    rel_mean, rel_std, spearman, ref_clim, input_clim, input_name, ref_name, output_path
):
    """Create a plot with four subplots showing relative mean, standard deviation, the spearman correlation of the climatologies and the seasonal mean of both datasets."""
    rel_mean = np.where(rel_mean == np.inf, np.nan, rel_mean)
    rel_std = np.where(rel_std == np.inf, np.nan, rel_std)
    fig, axes = plt.subplots(2, 2, figsize=(10.5, 4.68))
    if input_name is not None and ref_name is not None:
        fig.suptitle(f"Comparision {input_name} with {ref_name}")

    # Set common colormap and normalization limits for mean_et and mean_aet
    mean_diff_1 = max(np.abs(1 - np.nanmin(rel_mean)), np.abs(1 - np.nanmax(rel_mean)))
    im0, bounds0, extend0 = plot_single_map(
        axes[0, 0], rel_mean, mean_diff_1, bounds_type="fixed"
    )
    axes[0, 0].set_title(
        f"Relative temporal Mean (median={np.nanmedian(rel_mean):.2f})"
    )
    std_diff_1 = max(np.abs(1 - np.nanmin(rel_std)), np.abs(1 - np.nanmax(rel_std)))
    im1, bounds1, extend1 = plot_single_map(
        axes[0, 1], rel_std, std_diff_1, bounds_type="quantiles"
    )
    axes[0, 1].set_title(
        f"Relative temporal Standarddeviation (median={np.nanmedian(rel_std):.2f})"
    )

    im2 = axes[1, 0].imshow(spearman, vmin=np.nanmin(spearman), vmax=1)
    im2, bounds2, extend2 = plot_single_map(
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
    ax_twy.set_ylim(
        max(0, 1 - rel_clim_diff_1 * 1.05), 1 + rel_clim_diff_1 * 1.05
    )  # Example range for the ratio
    ax_twy.set_ylabel("Ratio (Input / Reference)", color="#0000A7")
    ax_twy.tick_params(axis="y", labelcolor="#0000A7")

    divider0 = make_axes_locatable(axes[0, 0])
    cax = divider0.append_axes("right", size="5%", pad=0.1)
    fig.colorbar(im0, cax=cax, label="", boundaries=bounds0, extend=extend0)

    divider1 = make_axes_locatable(axes[0, 1])
    cax2 = divider1.append_axes("right", size="5%", pad=0.1)
    fig.colorbar(im1, cax=cax2, label="", boundaries=bounds1, extend=extend1)

    divider2 = make_axes_locatable(axes[1, 0])
    cax = divider2.append_axes("right", size="5%", pad=0.1)
    fig.colorbar(im2, cax=cax, label="", boundaries=bounds2, extend=extend2)

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

    plt.savefig(output_path / "et_map.png", dpi=800)
    logger.info("created et_map")


def create_map_from_output(output_path, input_name, ref_name):
    """Read in statistics netcdf and create a map plots from it."""
    file = get_rel_stat_file(output_path, input_name, ref_name)
    logger.info(f"Plotting data from {file}")
    with get_xarray_ds_from_file(file) as ds:
        rel_std = ds["rel_std"]
        rel_mean = ds["rel_mean"]
        spearman = ds["spearman"]
        input_clim = (
            ds[f"{input_name}_clim"] if f"{input_name}_clim" in ds else ds["input_clim"]
        )
        ref_clim = (
            ds[f"{ref_name}_clim"] if f"{ref_name}_clim" in ds else ds["ref_clim"]
        )
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
            with get_dataset_from_path(
                path, available_mem=available_mem, file_name=file_name
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
        with get_xarray_ds_from_filerom_file(path, engine="netcdf4") as ds_input:
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
    return stats_ds


def compare_input_with_ref(
    input_path,
    input_var,
    output_path,
    ref_path,
    ref_var,
    input_name=None,
    ref_name=None,
    input_factor=1,
    ref_factor=1,
    coordinate_slice=None,
    n_bootstrap_years=None,
    bootstrap_index=None,
    ncpus=1,
    available_years=None,
    direct_comp=False,
    available_mem=None,
    input_file_name='*.*',
    ref_file_name='*.*',
    target_freq=None
):
    """Compare the two datasets."""
    if bootstrap_index is not None:
        random.seed(bootstrap_index)
    # get input statistics
    input_stats_file = None  # output_path / f"{input_name}_stats.nc" if input_name is not None else "input_stats.nc"
    ref_stats_file = None  # output_path / f"{ref_name}_stats.nc" if ref_name is not None else "ref_stats.nc"
    # TODO: Add index to file names
    # TODO If direct comparison load all files in mem to calculate spearman correlation. FOr this use target_freq to resample the times
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
        file_name=input_file_name,
    )
    logger.debug(f"input ds: {input}")

    # get output statistics
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
        file_name=ref_file_name,
    )
    logger.debug(f"ref ds: {ref}")
    # regrid spatial resoution
    # regridd to same spatial resolution
    input, ref = regridd_to_higher_spatial_resolution(input, ref)
    # regridd to same spatial resolution

    # compare and save statistics
    rel_mean = input["mean"].values / ref["mean"].values
    rel_std = input["std"].values / ref["std"].values
    if direct_comp:
        input_ts, ref_ts = input["time_series"], ref["time_series"]
        input_ts, ref_ts = resample_to_coarser_calendar(input_ts, ref_ts)
        input_ts, ref_ts = crop_data_to_overlapping_time(input_ts, ref_ts)
        logger.info(
            f"Creating data from timeseries with shape {input_ts.shape} and {ref_ts.shape}"
        )
        try:
            spearman, spearman_pval = spearman_spatial_joblib(
                input_ts, ref_ts, spearman_correlation, ncpus
            )
        except ValueError as ve:
            logger.error("Input and ref do not have the same temporal extend.")
            logger.info(input_ts.time)
            logger.info(ref_ts.time)
            raise (ve)
    else:
        logger.info("Calculating spearman correlation from seasonalities.")
        spearman, spearman_pval = spearman_spatial_joblib(
            input["clim"], ref["clim"], spearman_correlation, ncpus
        )

    rel_mean = xr.DataArray(
        rel_mean,
        coords={
            "lat": get_coord_values(input, lat=True),
            "lon": get_coord_values(input, lon=True),
        },
        dims=["lat", "lon"],
    )
    rel_std = xr.DataArray(
        rel_std,
        coords={
            "lat": get_coord_values(input, lat=True),
            "lon": get_coord_values(input, lon=True),
        },
        dims=["lat", "lon"],
    )
    spearman = xr.DataArray(
        spearman.data,
        coords={
            "lat": get_coord_values(input, lat=True),
            "lon": get_coord_values(input, lon=True),
        },
        dims=["lat", "lon"],
    )
    spearman_pval = xr.DataArray(
        spearman_pval.data,
        coords={
            "lat": get_coord_values(input, lat=True),
            "lon": get_coord_values(input, lon=True),
        },
        dims=["lat", "lon"],
    )
    input_clim = xr.DataArray(
        input["clim"].data,
        coords={
            "month": np.arange(1, 13, 1),
            "lat": get_coord_values(input, lat=True),
            "lon": get_coord_values(input, lon=True),
        },
        dims=["month", "lat", "lon"],
    )
    ref_clim = xr.DataArray(
        ref["clim"].data,
        coords={
            "month": np.arange(1, 13, 1),
            "lat": get_coord_values(input, lat=True),
            "lon": get_coord_values(input, lon=True),
        },
        dims=["month", "lat", "lon"],
    )
    output = xr.Dataset(
        {
            "spearman": spearman,
            "rel_std": rel_std,
            "rel_mean": rel_mean,
            "spearman_pval": spearman_pval,
        },
        coords={
            "month": np.arange(1, 13, 1),
            "lat": get_coord_values(input, lat=True),
            "lon": get_coord_values(input, lon=True),
        },
    )
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
    if bootstrap_index is not None:
        file_name = output_path / f"{file_name}_{bootstrap_index}.nc"
    else:
        file_name = output_path / f"{file_name}.nc"
    output.to_netcdf(file_name)
    logger.info(f"Written output to {file_name}")
    plot_map(
        rel_std=rel_std,
        rel_mean=rel_mean,
        spearman=spearman,
        ref_clim=ref["clim"],
        input_clim=input["clim"],
        input_name=input_name,
        ref_name=ref_name,
        output_path=output_path,
    )
    return file_name


def get_rel_stat_file(output_path, input_name, ref_name):
    """Create the file name for the file  contatining relative statistics of the two datasets."""
    file_name = "relative_stats"
    if input_name is not None:
        file_name += f"_{input_name}"
    if ref_name is not None:
        file_name += f"_{ref_name}"
    return output_path / f"{file_name}.nc"


def evaluate_boostraping_stat_files(stat_files, input_name, ref_name):
    """Evaluate bootstrapped statistics and compute median across bootstrap iterations."""
    # Open the first file to initialize dimensions and weights
    try:
        with get_xarray_ds_from_filerom_file(stat_files[0]) as first_file:
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

            # Preallocate arrays for all variables
            mean = np.empty((n_bootstrap, *shape))
            std = np.empty((n_bootstrap, *shape))
            spearman = np.empty((n_bootstrap, *shape))
            input_clim = np.empty((n_bootstrap, 12, *shape))  # Month x lat x lon
            ref_clim = np.empty((n_bootstrap, 12, *shape))  # Month x lat x lon
    except ValueError as ve:
        logger.error(f"opening file {stat_files[0]} as first file failed.")
        raise ve
    # Fill the preallocated arrays with bootstrap data
    for i, file in enumerate(stat_files):
        with get_xarray_ds_from_filerom_file(file) as ds:
            mean[i] = ds["rel_mean"].values
            std[i] = ds["rel_std"].values
            spearman[i] = ds["spearman"].values
            input_clim[i] = ds[input_clim_key].values
            ref_clim[i] = ds[ref_clim_key].values

    # Convert the arrays into xarray DataArrays
    mean_da = xr.DataArray(mean, dims=["bootstrap", "lat", "lon"])
    std_da = xr.DataArray(std, dims=["bootstrap", "lat", "lon"])
    spearman_da = xr.DataArray(spearman, dims=["bootstrap", "lat", "lon"])
    input_clim_da = xr.DataArray(input_clim, dims=["bootstrap", "month", "lat", "lon"])
    ref_clim_da = xr.DataArray(ref_clim, dims=["bootstrap", "month", "lat", "lon"])

    # Combine results into an xarray Dataset
    return {
        "rel_mean": mean_da.median(dim="bootstrap"),
        "rel_std": std_da.median(dim="bootstrap"),
        "spearman": spearman_da.median(dim="bootstrap"),
        "input_clim": input_clim_da.median(dim="bootstrap"),
        "ref_clim": ref_clim_da.median(dim="bootstrap"),
    }


def get_dataset_from_path(
    path, available_years=None, available_mem=None, file_name="*.*"
):
    """Get a dataset from a given path whether that is a file or a directory."""
    if path.is_file() and path.suffix == ".nc":
        chunking = available_mem is not None
        return get_xarray_ds_from_file(
            path,
            chunking=chunking,
            available_mem_gib=available_mem,
            chunk_type=ChunkType.SPACE,
        )
    if path.is_dir():
        file_list = get_files(
            path, available_years=available_years, file_name=file_name
        )
        logger.debug(file_list)
        logger.debug("combining files by coords ...")
        return xr.open_mfdataset(
            file_list,
            combine="by_coords",  # Ensures files are combined based on shared coordinates
        )
    with ErrorLogger(logger):
        msg = f"Path {path} does not exist."
        raise ValueError(msg)


def regridd_to_higher_spatial_resolution(ds1, ds2):
    """
    Regrids the coarser dataset to the resolution of the finer dataset using nearest-neighbor method.

    Parameters
    ----------
        ds1 (xarray.Dataset): First dataset.
        ds2 (xarray.Dataset): Second dataset.
            - Both should have latitude ('lat') and longitude ('lon') as coordinates.

    Returns
    -------
        xarray.Dataset: Regridded version of the coarser dataset to match the resolution of the finer dataset.
        xarray.Dataset: The finer dataset (unchanged).
    """
    # Determine which dataset is coarser
    lat_res_1 = abs(ds1["lat"][1] - ds1["lat"][0]).item()
    lon_res_1 = abs(ds1["lon"][1] - ds1["lon"][0]).item()
    lat_res_2 = abs(ds2["lat"][1] - ds2["lat"][0]).item()
    lon_res_2 = abs(ds2["lon"][1] - ds2["lon"][0]).item()

    # Identify the finer and coarser datasets
    if (lat_res_1 * lon_res_1) == (lat_res_2 * lon_res_2):
        return ds1, ds2
    if (lat_res_1 * lon_res_1) > (lat_res_2 * lon_res_2):
        coarse_ds, fine_ds = ds1, ds2
    else:
        coarse_ds, fine_ds = ds2, ds1

    # Perform nearest-neighbor regridding
    regridded_ds = coarse_ds.interp(
        lat=fine_ds["lat"], lon=fine_ds["lon"], method="nearest"
    )

    # Return regridded dataset and finer dataset
    logger.info(
        f'Regridded the two datasets to the resolution lat: {coarse_ds["lat"].data[1]-coarse_ds["lat"].data[0]}, lon: {coarse_ds["lon"].data[1]-coarse_ds["lon"].data[0]}'
    )
    if coarse_ds is ds1:
        return regridded_ds, ds2
    return ds1, regridded_ds


def get_years_from_path(path, raise_exception=True, file_name="*.*"):
    """Get years for one dataset from the folder structure or the xarray dataset."""
    if path.is_dir():
        return [int(p.name) for p in year_structure_paths(path, file_name=file_name)]
    if path.is_file():
        with get_xarray_ds_from_file(path) as input_ds:
            return [int(y) for y in np.unique(input_ds.time.dt.year.data)]
    if raise_exception:
        msg = f"The provided path {path} is neither file nor directory."
        with ErrorLogger(logger):
            raise ValueError(msg)
    return []


def get_available_years(input_path, ref_path, year_slice=None, direct_comp=True):
    """
    Determine available years from constrains and datasets.

    If no reference data is given it will only be the input years inside the year slice.
    """
    logger.info("Determining overlapping years.")
    # get all years from input data
    years_in = get_years_from_path(input_path)
    years_in.sort()
    logger.debug(f"Input years: {years_in}")

    # get all years from reference data
    years_ref = get_years_from_path(ref_path, raise_exception=False)
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
        if not years_ref or year in years_ref:
            years.append(year)
    return years


def year_structure_paths(path: Path, file_name="*.*") -> bool:
    """
    Return any subdirectory of `path` with year structur.

    That means any subdirectory whose name is exactly four digits (a valid “year”),
    and that subdirectory contains at least one entry.
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

def get_target_time_res_from_files(input_file, ref_file):
    """Get time resolution for resampling from two files."""
    with get_xarray_ds_from_file(input_file) as input_in:
        res_sim = input_in.time.diff('time').median()
    with get_xarray_ds_from_file(ref_file) as ref_in:
        res_obs = ref_in.time.diff('time').median()
    target_res = res_obs if res_obs > res_sim else res_sim
    return f"{int(target_res / np.timedelta64(1, 'h'))}h"



def get_target_time_res(input_path, ref_path, folder_name=''):
    """Get coarser time resolution from two datasets with files in folder structur."""
    input_files = (Path(input_path) / folder_name).glob('*nc')
    ref_files = (Path(ref_path) / folder_name).glob('*nc')
    if not list(input_files) or not list(ref_files):
        logger.error("One of the datasets has no files.")
        return None
    return get_target_time_res_from_files(next(input_files), next(ref_files))

@log_arguments()
def gridded_data_validation(
    input_path,
    input_var,
    output_path,
    ref_path,
    ref_var,
    input_name=None,
    ref_name=None,
    input_factor=1,
    ref_factor=1,
    only_plot=False,
    coordinate_slice=None,
    n_cpus=1,
    n_bootstrap_years=None,
    n_bootstrap_selections=None,
    direct_comp=True,
    year_slice=None,
    avaiable_mem=None,
    input_file_name="*.*",
    ref_file_name="*.*",
):
    """Validate a spatial variable from two datasets by comparing the climatology of that variable."""
    output_path = Path(output_path)
    input_path = Path(input_path)
    ref_path = Path(ref_path) if ref_path is not None else None

    if not output_path.is_dir():
        output_path.mkdir(parents=True)
    if only_plot and get_rel_stat_file(output_path, input_name, ref_name).is_file():
        create_map_from_output(
            output_path=output_path, input_name=input_name, ref_name=ref_name
        )
        return
    if input_path.is_dir() and len(year_structure_paths(input_path)) == 0:
        # check if the input path has the right structure
        input_path = input_path / "mHM_Fluxes_States.nc"
    available_years = get_available_years(input_path, ref_path, year_slice, direct_comp)
    logger.info(f"Years {available_years} are available for comparison.")
    if direct_comp: 
        target_time_res = get_target_time_res(input_path, ref_path, next(iter(years)))
        logger.info(f"Years {years} are overlapping. Data should be resampled to {target_time_res}")


    if ref_path is None:
        # Only create statistics do not compare
        logger.info(
            f"No ref file provided. Only creating a stat file for {input_name}."
        )
        output_name = f"{input_name}_stats.nc" if input_name is not None else "stats.nc"
        if input_path.is_file():
            # Write file stats to file
            with get_xarray_ds_from_file(
                input_path, chunking=True, available_mem_gib=10
            ) as ds_in:
                get_file_stats(
                    ds_in,
                    input_var,
                    input_factor,
                    coordinate_slice,
                    output=output_path / output_name,
                    avaiable_years=available_years,
                )

        elif (
            n_bootstrap_years is not None
            and n_bootstrap_selections > 0
            and input_path.is_dir()
        ):
            # Write file stats for each bootstrap selection
            stat_files = Parallel(n_jobs=n_cpus)(
                delayed(get_stats_one_pass)(
                    input_path,
                    input_var,
                    input_factor,
                    coordinate_slice,
                    n_bootstrap_years,
                    bootstrap_index,
                    output_path / output_name,
                    available_years=available_years,
                    input_file_name=input_file_name,
                )
                for bootstrap_index in range(n_bootstrap_selections)
            )
        elif input_path.is_dir():
            # Write file stats for one dataset read in from multiple files in a directory
            get_stats_one_pass(
                input_path,
                input_var,
                input_factor,
                coordinate_slice,
                output_path=output_path / output_name,
                available_years=available_years,
                file_name=input_file_name,
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
        # Compare by bootstraping
        if only_plot:
            stat_files = list(
                output_path.glob(f"relative_stats_{input_name}_{ref_name}_*")
            )
        else:
            ref_path = Path(ref_path)
            stat_files = Parallel(n_jobs=n_cpus)(
                delayed(compare_input_with_ref)(
                    input_path,
                    input_var,
                    output_path,
                    ref_path,
                    ref_var,
                    input_name,
                    ref_name,
                    input_factor,
                    ref_factor,
                    coordinate_slice,
                    n_bootstrap_years,
                    bootstrap_index,
                    available_years=available_years,
                    input_file_name=input_file_name,
                    ref_file_name=ref_file_name,
                    target_freq=target_time_res,
                )
                for bootstrap_index in range(n_bootstrap_selections)
            )
        stat_files = [file for file in stat_files if file is not None]
        if stat_files:
            results = evaluate_boostraping_stat_files(
                stat_files, input_name=input_name, ref_name=ref_name
            )
            plot_map(
                **results,
                output_path=output_path,
                input_name=input_name,
                ref_name=ref_name,
            )
        else:
            logger.error("There are no statfiles created from the evaluation.")
    else:
        logger.info("compare without bootstraping")
        compare_input_with_ref(
            input_path,
            input_var,
            output_path,
            ref_path,
            ref_var,
            input_name,
            ref_name,
            input_factor,
            ref_factor,
            coordinate_slice,
            ncpus=n_cpus,
            available_years=available_years,
            direct_comp=direct_comp,
            available_mem=avaiable_mem,
            input_file_name=input_file_name,
            ref_file_name=ref_file_name,
            target_freq=target_time_res,
        )
