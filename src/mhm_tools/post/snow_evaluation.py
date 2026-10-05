"""Evaluate simulated snow cover against a gridded snow reference.

The tool reads two gridded datasets (input and reference) from files or from
directories that are searched recursively, brings them onto a common calendar
and a common grid, reduces both to a binary snow-cover flag, and derives the
date of the first and last snow-covered day as well as the snow season
duration for every snow year and every grid cell. It also maps the snow
presence classification accuracy and renders an animated comparison of both
snow-cover fields and their disagreement.

Normalization rules
-------------------
- Calendar: both datasets are resampled to the coarser of the two calendars.
  If that calendar is still finer than 8 days, 8 day composites are used
  instead. Multi day steps are binned from the 1970 epoch, so both datasets
  land on the same bin grid regardless of where their records start.
- Grid: the finer dataset is aggregated onto the coarser grid by spatial mean.
- Snow flag: every step is compared against the snow threshold on its own, and
  a time step of the comparison counts as snow-covered when at least one of the
  steps it was aggregated from exceeded it. A step where nothing valid was
  observed stays missing. Thresholding each step rather than the mean of a
  composite makes the flag independent of how the record is split into files,
  and at a threshold of 0 the two give the same answer, because the mean of
  non-negative values only exceeds 0 when one of them does.
- Year window: a snow year from September to August, labelled by its start
  year, so a northern winter stays inside one window. ``calendar_year``
  switches to 1 January until 31 December.

Outputs
-------
- ``snow_season_metrics_<name>.nc`` per dataset, with the first and last
  snow-covered date, the same two days counted from the window start, the
  season span and the number of snow-covered days per year and cell.
- ``snow_season_metrics_difference_<input>_minus_<ref>.nc`` with the input
  minus reference bias of the numeric metrics in days.
- ``classification_accuracy_<input>_vs_<ref>.nc`` and ``.png`` with the map of
  CA = 1/n * sum(1 where SPF_input == SPF_ref), the share of compared time
  steps in which both snow presence flags agree, plus the n per cell. See
  :func:`calculate_classification_accuracy` for the definition, what enters n
  and why a high CA does not by itself mean a good simulation.
- ``snow_cover_percentage_<input>_vs_<ref>.png`` with the snow covered share of
  the cells over the full period. Domains reaching over the equator get one
  panel per hemisphere, all others a single coverage panel.
- ``snow_cover_<name>.nc`` per dataset with the normalized binary snow cover.
- ``snow_cover_comparison_<input>_vs_<ref>.gif`` with the input snow cover,
  the reference snow cover and their disagreement.
- the coverage plot, the accuracy map and the gif again per evaluation region,
  reusing the regions of the discharge evaluation.

Authors
-------
- Simon Lüdke
"""

import logging
from pathlib import Path

import dask.array as darr
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
from joblib import Parallel, delayed
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.patches import Patch

from mhm_tools.common.file_handler import (
    ChunkType,
    get_dataset_from_path,
    write_xarray_to_file,
)
from mhm_tools.common.logger import ErrorLogger, log_errors
from mhm_tools.common.parallel import resolve_ncpus
from mhm_tools.common.plotter import (
    AXIS_COLOR,
    FIGURE_WIDTH,
    INPUT_COLOR,
    KGE_UNDER_COLOR,
    PLOT_DPI,
    REF_COLOR,
    add_map_colorbar,
    calculate_map_figure_size,
    create_axis_label,
    create_comparison_title,
    create_summary_text,
    get_map_extent_and_origin,
    plot_single_map,
    style_axes,
    style_map_axes,
)
from mhm_tools.common.time_utils import (
    get_data_coverage,
    get_origin_anchored_freq,
    get_period_freq,
    get_step_days,
    normalize_time_axis,
    resample_to_target_freq,
    timedelta_to_alias,
)
from mhm_tools.common.utils import (
    format_region_title,
    sanitize_name,
    select_regions,
    split_file_list,
    write_stats_table,
)
from mhm_tools.common.xarray_utils import (
    aggregate_to_target_grid,
    align_to_target_grid,
    calculate_coordinate_resolution,
    calculate_region_medians,
    crop_to_region,
    get_coord_key,
    get_single_data_var,
    normalize_lat_lon,
    normalize_time,
    regrid_to_coarser_grid,
)
from mhm_tools.post.gridded_data_evaluation import (
    crop_datasets_to_spatial_overlap,
)

logger = logging.getLogger(__name__)

# the comparison never runs finer than an 8 day composite
EIGHT_DAY_FREQ = "8D"
EIGHT_DAY_HOURS = 192

NO_SNOW_VALUE = 0
SNOW_VALUE = 1
# A snow flag holds one bit of information, so it is kept as one signed byte
# with a sentinel for missing data instead of a float carrying NaN. That is
# eight times smaller than a float64 and keeps whole records in reach.
MISSING_VALUE = -1
SNOW_FLAG_DTYPE = "int8"
# a file is opened in blocks of a sixteenth of the budget, so several blocks
# may be computed at once without the sum of them leaving it
FILE_BLOCKS_IN_BUDGET = 16
NANOSECONDS_PER_DAY = 86_400_000_000_000

# Neutral state ramp for the binary panels. The light end is dark enough to
# stay visible against the page, which also separates it from missing data.
NO_SNOW_COLOR = "#9fb0b8"
SNOW_COLOR = "#37474f"
NO_DATA_COLOR = "#ffffff"
# light neutral, so agreement stays apart from the light input colour
AGREEMENT_COLOR = "#d9d9d9"
# accuracy is a skill score and uses the sequential skill colormap
ACCURACY_COLORMAP = "viridis_r"
ACCURACY_UNDER_COLOR = KGE_UNDER_COLOR


# ---------------------------------------------------------------------------
# reading and writing
# ---------------------------------------------------------------------------


def read_snow_data_array(data_path, file_name, var_name, label, max_memory_gib=8.0):
    """Read a snow variable from a file or recursively from a directory.

    The record is chunked along time rather than read as one block, so a long
    record streams through the evaluation instead of having to fit in memory.
    Where the input is one file per year, a chunk is roughly one file.

    Args:
        data_path: Path to a NetCDF file or to a directory searched recursively.
        file_name: Glob pattern used for the recursive directory search.
        var_name: Variable to select, or None to detect the single data variable.
        label: Name used in log messages and error messages.
        max_memory_gib: Memory budget one chunk is sized against.

    Returns
    -------
        The DataArray with ``time``, ``lat`` and ``lon`` dimensions.
    """
    logger.info(f"Reading {label} data from {data_path}")
    # open_mfdataset keeps the record lazy and gives one chunk per file; the
    # default path opens every file eagerly and only chunks afterwards, which
    # for a multi-year mHM record means reading it all before chunking it.
    # Naming the variable keeps the rest out of the read, and an mHM
    # fluxes-and-states file holds dozens of variables of which one is snow.
    ds = get_dataset_from_path(
        data_path,
        var_name=var_name,
        file_name=file_name,
        normalize_latlon_coords=True,
        use_mfdataset=True,
        chunking=True,
        available_mem_gib=max_memory_gib,
        chunk_type=ChunkType.TIME,
    )
    if var_name is None:
        var_name = get_single_data_var(ds)
        if var_name is None:
            msg = (
                f"Could not determine the {label} variable automatically. "
                f"Available data variables: {list(ds.data_vars)}. Pass it explicitly."
            )
            with ErrorLogger(logger):
                raise ValueError(msg)
        logger.info(f"Using {label} variable '{var_name}'.")
    elif var_name not in ds.data_vars:
        msg = (
            f"Variable '{var_name}' not found in the {label} dataset. "
            f"Available data variables: {list(ds.data_vars)}."
        )
        with ErrorLogger(logger):
            raise ValueError(msg)

    da = normalize_time(normalize_lat_lon(ds[var_name]))
    # a file holding a single time step must keep its time dimension, so only
    # the dimensions that are not part of the grid are squeezed away
    squeezable = [
        name
        for name in da.dims
        if name not in ("time", "lat", "lon") and da.sizes[name] == 1
    ]
    if squeezable:
        da = da.squeeze(squeezable, drop=True)
    missing_dims = {"time", "lat", "lon"} - set(da.dims)
    if missing_dims:
        msg = (
            f"The {label} variable '{var_name}' is missing the dimensions "
            f"{missing_dims}."
        )
        with ErrorLogger(logger):
            raise ValueError(msg)
    extra_dims = set(da.dims) - {"time", "lat", "lon"}
    if extra_dims:
        logger.warning(
            f"The {label} variable '{var_name}' has the additional dimensions "
            f"{extra_dims} which are kept as they are."
        )
    return da.transpose("time", "lat", "lon", ...)


def write_snow_dataset(ds, file_path, compression=None):
    """Write a snow evaluation dataset to a compressed NetCDF file.

    Date variables get explicit time units, so the file states one calendar
    instead of letting every variable pick its own. Integer variables carry the
    snow flag's missing sentinel as their fill value, so a reader decodes it
    back to NaN instead of reading -1 as data.

    Args:
        ds: Dataset to write.
        file_path: Target file path; parent folders are created.
        compression: NetCDF compression settings, or None.

    Returns
    -------
        The written file path.
    """
    encoding = {}
    for name in list(ds.data_vars) + list(ds.coords):
        if np.issubdtype(ds[name].dtype, np.datetime64):
            encoding[name] = {
                "units": "seconds since 1970-01-01",
                "calendar": "standard",
            }
        elif name in ds.data_vars and np.issubdtype(ds[name].dtype, np.integer):
            encoding[name] = {"_FillValue": MISSING_VALUE}
    logger.info(f"Writing {file_path}")
    write_xarray_to_file(ds, file_path, encoding=encoding, compression=compression)
    return Path(file_path)


def resample_snow_to_target_frequency(input_da, ref_da, target_freq):
    """Resample both snow fields to the target frequency on a shared bin grid.

    A step of several days is binned from an origin instead of from the
    calendar, so the epoch origin is required to keep two records that start on
    different days on the same bins. Steps that only one dataset covers
    completely are dropped.

    Args:
        input_da: Input DataArray with a ``time`` dimension.
        ref_da: Reference DataArray with a ``time`` dimension.
        target_freq: Pandas frequency alias to resample to.

    Returns
    -------
        Tuple of the two resampled and time-aligned DataArrays.
    """
    step_days = get_step_days(target_freq)
    resample_origin = "epoch" if step_days is not None and step_days > 1 else None
    input_da, ref_da = resample_to_target_freq(
        input_da,
        ref_da,
        target_freq,
        resample_origin=resample_origin,
        time_anchor="period_start",
        drop_partial_edges=True,
    )
    if input_da.sizes["time"] == 0:
        msg = f"No common time steps remain after resampling to {target_freq}."
        with ErrorLogger(logger):
            raise ValueError(msg)
    logger.info(
        f"Both datasets share {input_da.sizes['time']} {target_freq} time steps."
    )
    return input_da, ref_da


# -------------------------------------------------------------------------
# grid and calendar normalization
# -------------------------------------------------------------------------


def get_snow_alias(obj):
    """Detect the calendar of a snow field, naming an 8 day step ``8D``.

    ``timedelta_to_alias`` maps an 8 day step to its hour count, because it
    only names daily, weekly and monthly calendars. The snow evaluation uses
    8 day composites, so that step gets its own alias here.

    Args:
        obj: Object with a ``time`` coordinate.

    Returns
    -------
        Tuple of (median time step in hours, frequency alias).
    """
    hours, alias = timedelta_to_alias(obj)
    if abs(hours - EIGHT_DAY_HOURS) <= 12:
        return hours, EIGHT_DAY_FREQ
    return hours, alias


def get_target_frequency(input_da, ref_da):
    """Determine the common target frequency of two time series.

    The coarser of the two calendars is used. If that is still finer than an
    8 day composite, ``8D`` is returned.

    Args:
        input_da: Input DataArray with a ``time`` dimension.
        ref_da: Reference DataArray with a ``time`` dimension.

    Returns
    -------
        The pandas frequency alias to resample both datasets to.
    """
    input_hours, input_alias = get_snow_alias(input_da)
    ref_hours, ref_alias = get_snow_alias(ref_da)
    logger.info(
        f"Detected calendars: input {input_alias} ({input_hours} h), "
        f"reference {ref_alias} ({ref_hours} h)."
    )
    coarser_alias = input_alias if input_hours >= ref_hours else ref_alias
    if max(input_hours, ref_hours) < EIGHT_DAY_HOURS:
        logger.info(
            f"Both calendars are finer than 8 days; using {EIGHT_DAY_FREQ} instead."
        )
        return EIGHT_DAY_FREQ
    return coarser_alias


# -------------------------------------------------------------------------
# snow cover and season metrics
# -------------------------------------------------------------------------


def create_snow_cover_flag(da, snow_threshold):
    """Reduce a snow variable to a binary snow-cover flag.

    The flag is a signed byte rather than a float with NaN, because it carries
    one bit of information per cell. Missing data becomes ``MISSING_VALUE``;
    use :func:`get_valid_snow_mask` instead of ``notnull`` to find it.

    Args:
        da: DataArray of snow depth, snow water equivalent or snow cover.
        snow_threshold: Values strictly above this count as snow-covered.

    Returns
    -------
        DataArray with 1 for snow, 0 for no snow and -1 for missing data.
    """
    flag = xr.where(da > snow_threshold, SNOW_VALUE, NO_SNOW_VALUE)
    flag = xr.where(da.notnull(), flag, MISSING_VALUE).astype(SNOW_FLAG_DTYPE)
    flag.name = "snow_cover"
    flag.attrs = {
        "long_name": "binary snow cover flag",
        "units": "1",
        "flag_values": f"{NO_SNOW_VALUE}, {SNOW_VALUE}",
        "flag_meanings": "no_snow snow",
        "snow_threshold": snow_threshold,
    }
    return flag


def get_snow_record_files(data_path, file_name):
    """List the files one snow record is read from, in time order of their names.

    Args:
        data_path: Path to a NetCDF file or a directory searched recursively.
        file_name: Glob pattern used for the recursive directory search.

    Returns
    -------
        Sorted list of file paths.
    """
    data_path = Path(data_path)
    files = sorted(data_path.rglob(file_name)) if data_path.is_dir() else [data_path]
    if not files:
        msg = f"No file matches {file_name!r} below {data_path}."
        with ErrorLogger(logger):
            raise FileNotFoundError(msg)
    return files


def get_file_time_block(da, max_memory_gib):
    """Return how many time steps of one file to reduce at a time.

    A record kept in a single file would otherwise be reduced in one go, which
    is the whole record again. Because the flags are merged by a maximum, a
    block may cover any steps in any order, so the block size only has to fit
    the budget.

    Args:
        da: DataArray of one file, with ``lat`` and ``lon``.
        max_memory_gib: Memory budget of the run.

    Returns
    -------
        The number of time steps per block, at least one.
    """
    step_bytes = da.sizes["lat"] * da.sizes["lon"] * max(da.dtype.itemsize, 1)
    block_bytes = max(max_memory_gib, 0.1) * 1024**3 / FILE_BLOCKS_IN_BUDGET
    return max(1, min(da.sizes["time"], int(block_bytes // max(step_bytes, 1))))


def read_snow_file_times(file_path):
    """Read the time axis of one file without touching its values.

    Opening a dataset does not read a data variable, so only the coordinate is
    fetched here. This is what keeps a probe of thousands of files cheap.

    Args:
        file_path: Path to a NetCDF file.

    Returns
    -------
        The time stamps of that file as an array.
    """
    with xr.open_dataset(file_path) as ds:
        return np.asarray(ds[get_coord_key(ds, time=True)].values)


def probe_snow_record(
    data_path, file_name, var_name, label, ncpus=1, max_memory_gib=8.0
):
    """Describe a record by its grid and calendar without reading its values.

    The shared grid, period and calendar are all that is needed before the
    files can be binned, and none of it needs the data. Opening the whole
    record instead would hold every file at once, which on a record of
    thousands of files costs more than the binning it prepares.

    Args:
        data_path: Path to a NetCDF file or a directory searched recursively.
        file_name: Glob pattern used for the recursive directory search.
        var_name: Variable to read, or None to detect the single one.
        label: Name used in the log and error messages.
        ncpus: Cores used to read the time axes in parallel.
        max_memory_gib: Memory budget the single probed file is sized against.

    Returns
    -------
        An empty DataArray carrying the real grid and time axis of the record.
    """
    files = get_snow_record_files(data_path, file_name)
    # one file settles the grid, the variable and its type
    first = read_snow_data_array(
        files[0], "*.nc", var_name, f"{label} grid", max_memory_gib
    )
    latitudes = first["lat"].values
    longitudes = first["lon"].values
    if len(files) == 1:
        times = pd.DatetimeIndex(first["time"].values)
    else:
        cores = min(resolve_ncpus(ncpus), len(files))
        logger.info(
            f"Probing the time axis of {len(files)} {label} file(s) on "
            f"{cores} core(s)."
        )
        if cores > 1:
            per_file = Parallel(n_jobs=cores, backend="loky")(
                delayed(read_snow_file_times)(file_path) for file_path in files
            )
        else:
            per_file = [read_snow_file_times(file_path) for file_path in files]
        times = pd.DatetimeIndex(np.concatenate(per_file)).sort_values()
    logger.info(
        f"The {label} record holds {len(times)} steps from "
        f"{times[0].date()} to {times[-1].date()} on a "
        f"{len(latitudes)}x{len(longitudes)} grid."
    )
    # the values stay empty and lazy; only the coordinates carry information
    skeleton = darr.zeros(
        (len(times), len(latitudes), len(longitudes)),
        dtype=first.dtype,
        chunks=(1, len(latitudes), len(longitudes)),
    )
    return xr.DataArray(
        skeleton,
        dims=("time", "lat", "lon"),
        coords={"time": times, "lat": latitudes, "lon": longitudes},
        name=first.name,
    )


def create_target_bins(time_slice, target_freq):
    """Create the shared time bins every file's values are aggregated into.

    A fixed day multiple is binned from the 1970 epoch, so a file only covering
    part of the record still lands on the same bins as every other file.

    Args:
        time_slice: ``slice`` of the compared period.
        target_freq: Pandas frequency alias of one bin.

    Returns
    -------
        DatetimeIndex of the first instant of every bin.
    """
    start, end = pd.Timestamp(time_slice.start), pd.Timestamp(time_slice.stop)
    step_days = get_step_days(target_freq)
    if step_days is not None:
        epoch = pd.Timestamp("1970-01-01")
        first = epoch + pd.Timedelta(
            days=((start - epoch).days // step_days) * step_days
        )
        labels = pd.date_range(first, end, freq=f"{step_days}D")
        bin_ends = labels + pd.Timedelta(days=step_days)
    else:
        period_freq = get_period_freq(target_freq) or "D"
        periods = pd.period_range(start, end, freq=period_freq)
        labels = pd.DatetimeIndex(periods.start_time)
        bin_ends = pd.DatetimeIndex(periods.end_time)
    # a bin reaching outside the compared period is aggregated from fewer steps
    # than a full one, which would compare a full period against a shorter one
    fully_covered = (labels >= start) & (bin_ends <= end)
    dropped = len(labels) - int(fully_covered.sum())
    if dropped:
        logger.info(
            f"Dropping {dropped} bin(s) the compared period only covers partly."
        )
    return labels[fully_covered]


def regrid_da_to_target_grid(da, target_lat, target_lon):
    """Bring one field onto the shared target grid.

    Args:
        da: DataArray with ``lat`` and ``lon``.
        target_lat: Target latitude values.
        target_lon: Target longitude values.

    Returns
    -------
        The DataArray on the target grid.
    """
    source_cell = calculate_coordinate_resolution(
        da["lat"]
    ) * calculate_coordinate_resolution(da["lon"])
    target_cell = calculate_coordinate_resolution(
        target_lat
    ) * calculate_coordinate_resolution(target_lon)
    if np.isclose(source_cell, target_cell, rtol=1e-3):
        return align_to_target_grid(da, target_lat, target_lon)
    return aggregate_to_target_grid(da, target_lat, target_lon)


def accumulate_snow_flag_subset(
    files,
    var_name,
    label,
    target_grid,
    target_freq,
    time_slice,
    snow_threshold,
    max_memory_gib=8.0,
):
    """Reduce a subset of files to the binary flag of the bins it fills.

    One file is read, regridded and binned at a time, and only the totals of
    the shared bins are kept, so the raw record never has to fit in memory.

    Args:
        files: File paths of this subset.
        var_name: Variable to read, or None to detect the single one.
        label: Name used in the log and error messages.
        target_grid: Tuple of (bin labels, target latitudes, target longitudes).
        target_freq: Pandas frequency alias of one bin.
        time_slice: ``slice`` of the compared period.
        snow_threshold: Values strictly above this count as snow-covered.
        max_memory_gib: Memory budget one chunk of a file is sized against.

    Returns
    -------
        The binary snow cover flag of the bins this subset filled.
    """
    bin_labels, target_lat, target_lon = target_grid
    # Every step is thresholded on its own and the bin keeps the largest flag it
    # saw, so -1 stays only where nothing valid fell, 0 where snow never did and
    # 1 as soon as one step exceeded the threshold. That merge is a maximum, so
    # it does not care in which order the files arrive and needs no sums.
    flag = np.full(
        (len(bin_labels), len(target_lat), len(target_lon)),
        MISSING_VALUE,
        dtype=SNOW_FLAG_DTYPE,
    )
    bin_position = {label_value: index for index, label_value in enumerate(bin_labels)}
    step_days = get_step_days(target_freq)
    resample_origin = "epoch" if step_days is not None and step_days > 1 else None
    resample_kwargs = {} if resample_origin is None else {"origin": resample_origin}
    resample_freq = (
        target_freq
        if resample_origin is None
        else get_origin_anchored_freq(target_freq)
    )

    for number, file_path in enumerate(files, start=1):
        da = read_snow_data_array(
            file_path,
            "*.nc",
            var_name,
            f"{label} file {number}",
            max_memory_gib / FILE_BLOCKS_IN_BUDGET,
        )
        da = da.sel(time=time_slice)
        if da.sizes["time"] == 0:
            continue
        da = regrid_da_to_target_grid(da, target_lat, target_lon)
        # a block at a time, so a record kept in one file is streamed as well
        block = get_file_time_block(da, max_memory_gib)
        for first_step in range(0, da.sizes["time"], block):
            part = da.isel(time=slice(first_step, first_step + block))
            step_flag = create_snow_cover_flag(part, snow_threshold)
            # a bin without any time step comes back as NaN, which is missing data
            binned = (
                step_flag.resample(time=resample_freq, **resample_kwargs)
                .max()
                .fillna(MISSING_VALUE)
                .astype(SNOW_FLAG_DTYPE)
            )
            binned = normalize_time_axis(binned, target_freq, anchor="period_start")
            binned_values = binned.compute().values
            for step, bin_start in enumerate(pd.DatetimeIndex(binned["time"].values)):
                position = bin_position.get(bin_start)
                if position is None:
                    continue
                np.maximum(flag[position], binned_values[step], out=flag[position])
        logger.info(f"Binned {label} file {number} of {len(files)}: {file_path.name}")
    return flag


def create_snow_flag_one_pass(
    data_path,
    file_name,
    var_name,
    label,
    target_grid,
    target_freq,
    time_slice,
    snow_threshold,
    ncpus=1,
    max_memory_gib=8.0,
):
    """Read a record one file at a time and reduce it to a binary snow flag.

    Only the totals of the shared bins and the flag itself are held, so the
    memory a run needs follows the compared grid rather than the size of the
    record. The files are independent, so subsets of them are binned in
    parallel and their totals summed.

    Args:
        data_path: Path to a NetCDF file or a directory searched recursively.
        file_name: Glob pattern used for the recursive directory search.
        var_name: Variable to read, or None to detect the single one.
        label: Name used in the log and error messages.
        target_grid: Tuple of (bin labels, target latitudes, target longitudes).
        target_freq: Pandas frequency alias of one bin.
        time_slice: ``slice`` of the compared period.
        snow_threshold: Values strictly above this count as snow-covered.
        ncpus: Cores used to bin the files in parallel.
        max_memory_gib: Memory budget one chunk of a file is sized against.

    Returns
    -------
        The binary snow cover flag on the target grid and bins.
    """
    files = get_snow_record_files(data_path, file_name)
    cores = min(resolve_ncpus(ncpus), len(files))
    # merging two subsets is a maximum, so the files may be split any way
    subsets = split_file_list(files, cores) if cores > 1 else [files]
    logger.info(f"Binning {len(files)} {label} file(s) on {cores} core(s).")
    arguments = (
        var_name,
        label,
        target_grid,
        target_freq,
        time_slice,
        snow_threshold,
        max_memory_gib,
    )
    if cores > 1:
        results = Parallel(n_jobs=cores, backend="loky")(
            delayed(accumulate_snow_flag_subset)(subset, *arguments)
            for subset in subsets
        )
    else:
        results = [accumulate_snow_flag_subset(subsets[0], *arguments)]

    bin_labels, target_lat, target_lon = target_grid
    flag = results[0]
    for subset_flag in results[1:]:
        np.maximum(flag, subset_flag, out=flag)
    flag_da = xr.DataArray(
        flag,
        dims=("time", "lat", "lon"),
        coords={"time": bin_labels, "lat": target_lat, "lon": target_lon},
        name="snow_cover",
    )
    flag_da.attrs = {
        "long_name": "binary snow cover flag",
        "units": "1",
        "flag_values": f"{NO_SNOW_VALUE}, {SNOW_VALUE}",
        "flag_meanings": "no_snow snow",
        "snow_threshold": snow_threshold,
    }
    held_gib = flag_da.size * flag_da.dtype.itemsize / 1024**3
    logger.info(f"Holding the {label} snow cover flag ({held_gib:.2f} GiB).")
    return flag_da


def get_valid_snow_mask(snow_flag):
    """Return where a snow cover flag holds data rather than the missing sentinel.

    Args:
        snow_flag: Snow cover flag DataArray.

    Returns
    -------
        Boolean DataArray, True where the cell was observed.
    """
    return snow_flag >= 0


def drop_uncovered_bins(input_flag, ref_flag, target_freq):
    """Drop the bins that only one of the two records has data in.

    A bin at the edge of one record is aggregated from fewer steps than the
    same bin of the other, which would compare a full period against a shorter
    one.

    Args:
        input_flag: Input snow cover flag on the shared bins.
        ref_flag: Reference snow cover flag on the shared bins.
        target_freq: Pandas frequency alias of one bin.

    Returns
    -------
        Tuple of the two flags reduced to the bins both records cover.
    """
    covered = (get_valid_snow_mask(input_flag).any(("lat", "lon"))) & (
        get_valid_snow_mask(ref_flag).any(("lat", "lon"))
    )
    kept = int(covered.sum())
    if kept == 0:
        msg = f"No common time steps remain after resampling to {target_freq}."
        with ErrorLogger(logger):
            raise ValueError(msg)
    dropped = input_flag.sizes["time"] - kept
    if dropped:
        logger.info(f"Dropping {dropped} bin(s) only one of the records covers.")
    logger.info(f"Both datasets share {kept} {target_freq} time steps.")
    return input_flag.isel(time=covered.values), ref_flag.isel(time=covered.values)


def create_year_windows(time_index, year_mode, snow_year_start_month, step_days):
    """Create the yearly evaluation windows covered by a time index.

    Args:
        time_index: DatetimeIndex of the normalized data.
        year_mode: Either "snow_year" or "calendar_year".
        snow_year_start_month: First month of a snow year, e.g. 9 for September.
        step_days: Length of one time step in days, used as coverage tolerance.

    Returns
    -------
        List of (year_label, window_start, window_end) tuples.
    """
    start_month = 1 if year_mode == "calendar_year" else snow_year_start_month
    year_labels = np.where(
        time_index.month >= start_month, time_index.year, time_index.year - 1
    )
    tolerance = pd.Timedelta(days=step_days)
    windows = []
    for year_label in sorted({int(year) for year in year_labels}):
        window_start = pd.Timestamp(year=year_label, month=start_month, day=1)
        window_end = window_start + pd.DateOffset(years=1) - pd.Timedelta(1, "ns")
        covered = time_index[(time_index >= window_start) & (time_index <= window_end)]
        if covered.empty:
            continue
        # one time step of slack, because a step label never hits the window end
        if (covered.min() - window_start) > tolerance or (
            window_end - covered.max()
        ) > tolerance:
            logger.warning(
                f"Year {year_label} is only partly covered: data run from "
                f"{covered.min()} to {covered.max()} inside the window "
                f"{window_start.date()} to {window_end.date()}."
            )
        windows.append((year_label, window_start, window_end))
    return windows


def create_step_bounds_in_days(time_index, target_freq, window_start, window_end):
    """Return start and end of every time step as days since the window start.

    Using the true period bounds instead of a median step keeps month lengths
    exact, so a snow-covered February contributes 28 and not 31 days. Steps
    reaching over a window edge are clipped to the window.

    Args:
        time_index: DatetimeIndex of the time steps inside the window.
        target_freq: Pandas frequency alias the data was resampled to.
        window_start: First time stamp of the year window.
        window_end: Last time stamp of the year window.

    Returns
    -------
        Tuple of (step start, step end) arrays in days since ``window_start``.
    """
    window_length = (window_end + pd.Timedelta(1, "ns") - window_start) / pd.Timedelta(
        days=1
    )
    step_days = get_step_days(target_freq)
    period_freq = get_period_freq(target_freq)
    if step_days is not None:
        # a fixed day multiple is labelled by its first day, so the bounds follow
        starts = (time_index - window_start) / pd.Timedelta(days=1)
        ends = starts + step_days
    elif period_freq is None:
        # sub-daily or irregular data: fall back to the spacing of the stamps
        starts = (time_index - window_start) / pd.Timedelta(days=1)
        step = float(np.median(np.diff(starts))) if len(starts) > 1 else 1.0
        ends = starts + step
    else:
        periods = time_index.to_period(period_freq)
        starts = (periods.start_time - window_start) / pd.Timedelta(days=1)
        ends = (periods.end_time + pd.Timedelta(1, "ns") - window_start) / pd.Timedelta(
            days=1
        )
    starts = np.clip(np.asarray(starts, dtype=float), 0.0, window_length)
    ends = np.clip(np.asarray(ends, dtype=float), 0.0, window_length)
    return starts, ends


def create_date_from_day_of_window(day_of_window, window_start):
    """Convert days since a window start into dates, mapping NaN to NaT.

    Args:
        day_of_window: DataArray of days since ``window_start``.
        window_start: Timestamp the offsets refer to.

    Returns
    -------
        DataArray of datetime64 values.
    """
    offset = (day_of_window * NANOSECONDS_PER_DAY).round().astype("timedelta64[ns]")
    return np.datetime64(pd.Timestamp(window_start), "ns") + offset


def calculate_snow_season_metrics(snow_flag, year_windows, target_freq):
    """Calculate first snow day, last snow day and season duration per year and cell.

    The first and last snow day are the first and last calendar day covered by
    a snow-covered time step, so monthly data resolves to the first and last
    day of the respective month. Cells that never carry snow inside a window
    get NaN for the dates and the span and 0 snow-covered days. Cells without
    any valid data stay NaN everywhere.

    Args:
        snow_flag: Binary snow cover DataArray with ``time``, ``lat``, ``lon``.
        year_windows: List of (year_label, window_start, window_end) tuples.
        target_freq: Pandas frequency alias the data was resampled to.

    Returns
    -------
        Dataset with a ``year`` dimension holding the season metrics.
    """
    yearly_metrics = []
    for year_label, window_start, window_end in year_windows:
        window = snow_flag.sel(time=slice(window_start, window_end))
        if window.sizes["time"] == 0:
            logger.warning(
                f"Skipping year {year_label} because it holds no time steps."
            )
            continue
        is_snow = window == SNOW_VALUE
        has_valid_data = get_valid_snow_mask(window).any("time")
        step_start, step_end = create_step_bounds_in_days(
            pd.DatetimeIndex(window.time.values), target_freq, window_start, window_end
        )
        step_start = xr.DataArray(step_start, dims="time", coords={"time": window.time})
        step_end = xr.DataArray(step_end, dims="time", coords={"time": window.time})

        first_day = xr.where(is_snow, step_start, np.nan).min("time")
        last_end = xr.where(is_snow, step_end, np.nan).max("time")
        first_day = first_day.where(has_valid_data)
        # the last snow-covered day itself, not the exclusive end of its step
        last_day = (last_end - 1).where(has_valid_data)
        year_ds = xr.Dataset(
            {
                "first_snow_date": create_date_from_day_of_window(
                    first_day, window_start
                ),
                "last_snow_date": create_date_from_day_of_window(
                    last_day, window_start
                ),
                "first_snow_day_of_window": first_day,
                "last_snow_day_of_window": last_day,
                "season_span_days": (last_end - first_day).where(has_valid_data),
                "snow_cover_days": (is_snow * (step_end - step_start))
                .sum("time")
                .where(has_valid_data),
            }
        )
        year_ds = year_ds.expand_dims(year=[year_label])
        year_ds = year_ds.assign_coords(
            window_start=("year", [np.datetime64(window_start, "ns")]),
        )
        yearly_metrics.append(year_ds)

    if not yearly_metrics:
        msg = "No evaluation year holds any time step; cannot calculate season metrics."
        with ErrorLogger(logger):
            raise ValueError(msg)
    metrics = xr.concat(yearly_metrics, dim="year")
    return set_snow_metric_attributes(metrics)


def set_snow_metric_attributes(metrics):
    """Add descriptive attributes to the snow season metric variables.

    Args:
        metrics: Dataset of snow season metrics.

    Returns
    -------
        The dataset with attributes set.
    """
    # "d" instead of "days" keeps xarray from decoding these into timedelta64
    attributes = {
        "first_snow_date": {"long_name": "date of the first snow-covered day"},
        "last_snow_date": {"long_name": "date of the last snow-covered day"},
        "first_snow_day_of_window": {
            "long_name": "first snow-covered day counted from the start of the year window",
            "units": "d",
        },
        "last_snow_day_of_window": {
            "long_name": "last snow-covered day counted from the start of the year window",
            "units": "d",
        },
        "season_span_days": {
            "long_name": "snow season duration from the first to the last snow-covered day",
            "units": "d",
        },
        "snow_cover_days": {
            "long_name": "number of snow-covered days inside the year window",
            "units": "d",
        },
    }
    for var_name, var_attrs in attributes.items():
        metrics[var_name].attrs.update(var_attrs)
    metrics["year"].attrs["long_name"] = "label of the year window"
    metrics["window_start"].attrs["long_name"] = "first day of the year window"
    return metrics


def compare_snow_season_metrics(input_metrics, ref_metrics):
    """Difference the numeric snow season metrics of input and reference.

    Args:
        input_metrics: Snow season metrics of the input dataset.
        ref_metrics: Snow season metrics of the reference dataset.

    Returns
    -------
        Dataset of input minus reference for every numeric metric.
    """
    numeric_vars = [
        "first_snow_day_of_window",
        "last_snow_day_of_window",
        "season_span_days",
        "snow_cover_days",
    ]
    difference = xr.Dataset(
        {var: input_metrics[var] - ref_metrics[var] for var in numeric_vars}
    )
    for var in numeric_vars:
        difference[var].attrs = {
            "long_name": f"input minus reference {input_metrics[var].attrs['long_name']}",
            "units": "d",
        }
    difference.attrs["comment"] = (
        "Positive values mean the input value is later or longer."
    )
    return difference


def create_snow_stats_table(
    accuracy_ds,
    difference_metrics,
    region_names,
    input_name,
    ref_name,
    output_file=None,
):
    """Log the median snow statistics per WMO region and for the domain.

    Args:
        accuracy_ds: Dataset from ``calculate_classification_accuracy``.
        difference_metrics: Dataset from ``compare_snow_season_metrics``.
        region_names: Keys of ``WMO_REGION_BOUNDS``.
        input_name: Label of the input dataset.
        ref_name: Label of the reference dataset.
        output_file: CSV file path, or None to only log the table.

    Returns
    -------
        DataFrame of the medians per region.
    """
    # short names keep the table narrow, the title explains them
    stats = xr.Dataset(
        {
            "accuracy": accuracy_ds["classification_accuracy"],
            "first_snow_day_diff": difference_metrics["first_snow_day_of_window"],
            "last_snow_day_diff": difference_metrics["last_snow_day_of_window"],
            "season_span_diff": difference_metrics["season_span_days"],
            "snow_cover_days_diff": difference_metrics["snow_cover_days"],
        }
    )
    stats_df = calculate_region_medians(stats, list(stats.data_vars), region_names)
    write_stats_table(
        stats_df,
        title=(
            f"Median snow statistics of {input_name} against {ref_name} "
            "(accuracy = share of matching time steps, *_diff = input - reference "
            "in days, taken over every cell and year)"
        ),
        output_file=output_file,
    )
    return stats_df


def calculate_classification_accuracy(input_flag, ref_flag):
    """Calculate the share of time steps in which both snow flags agree per cell.

    Definition
    ----------
    For one grid cell with the snow presence flags ``SPF_input`` and ``SPF_ref``
    (1 snow, 0 no snow), over the ``n`` time steps ``i`` of the common period::

        CA = 1/n * sum_i  1(SPF_input_i == SPF_ref_i)

    The indicator is 1 when both datasets classify the step the same way. In
    confusion-matrix terms the numerator is ``hits + correct negatives``, so CA
    is the overall accuracy of the snow/no-snow classification. It is computed
    per cell and reduces the time dimension away, which is what makes it a map.

    What counts towards n
    ---------------------
    Only steps in which *both* datasets hold valid data. A step masked in either
    dataset is dropped from the numerator and the denominator alike, so ``n``
    varies between cells and is written next to the accuracy as
    ``compared_time_steps``. A cell with no overlapping valid step at all stays
    NaN instead of 0, so "never comparable" cannot be mistaken for "never
    agreed". CA lies in [0, 1]; 1 means every compared step matched.

    Interpreting it
    ---------------
    CA measures agreement, not skill, and two properties matter when reading the
    map:

    - It is inflated by the base rate. A cell that is snow-free at every step in
      both datasets scores 1.00, and so does a permanently snow-covered cell,
      because agreeing about an unchanging state is free. Only cells that
      actually switch state carry information about the simulation, so a
      high-CA area is not automatically a well-simulated one.
    - It is symmetric and therefore blind to the direction of the error. Snow
      simulated where the reference has none costs exactly as much as the
      reverse. Use the difference panel of the gif, which separates
      input-only from reference-only cells, and the signed season metrics to see
      which way a cell is wrong.

    Because the flags are binary, CA also says nothing about the depth or the
    water equivalent of the snow; it only scores presence.

    Args:
        input_flag: Binary snow cover DataArray of the input dataset.
        ref_flag: Binary snow cover DataArray of the reference dataset.

    Returns
    -------
        Dataset with ``classification_accuracy`` and ``compared_time_steps``,
        both on the ``lat``/``lon`` grid of the inputs.
    """
    both_valid = get_valid_snow_mask(input_flag) & get_valid_snow_mask(ref_flag)
    compared_steps = both_valid.sum("time")
    matching_steps = ((input_flag == ref_flag) & both_valid).sum("time")
    accuracy = matching_steps / compared_steps.where(compared_steps > 0)
    empty_cells = int((compared_steps == 0).sum())
    if empty_cells:
        logger.warning(
            f"{empty_cells} cell(s) have no time step where both datasets are "
            "valid and stay undefined in the classification accuracy."
        )
    accuracy_ds = xr.Dataset(
        {
            "classification_accuracy": accuracy,
            "compared_time_steps": compared_steps.where(compared_steps > 0),
        }
    )
    accuracy_ds["classification_accuracy"].attrs = {
        "long_name": "snow presence classification accuracy",
        "units": "1",
        "valid_min": 0.0,
        "valid_max": 1.0,
        "comment": (
            "Share of compared time steps in which the input and the reference "
            "snow presence flag are equal."
        ),
    }
    accuracy_ds["compared_time_steps"].attrs = {
        "long_name": "number of time steps compared per cell",
        "units": "1",
        "comment": "The n of the classification accuracy.",
    }
    return accuracy_ds


def select_hemisphere(snow_flag, northern=True):
    """Select the cells of one hemisphere from a snow cover field.

    Args:
        snow_flag: DataArray with a ``lat`` coordinate.
        northern: Select latitudes at or above the equator, else below it.

    Returns
    -------
        The DataArray reduced to the cells of that hemisphere.
    """
    latitudes = snow_flag["lat"].values
    selected = latitudes >= 0 if northern else latitudes < 0
    return snow_flag.isel(lat=np.flatnonzero(selected))


def calculate_snow_covered_cell_percentage(snow_flag):
    """Calculate the share of valid cells that are snow covered per time step.

    Args:
        snow_flag: Binary snow cover DataArray with ``time``, ``lat``, ``lon``.

    Returns
    -------
        DataArray of percentages along ``time``, NaN where no cell is valid.
    """
    valid_cells = get_valid_snow_mask(snow_flag).sum(("lat", "lon"))
    snow_cells = (snow_flag == SNOW_VALUE).sum(("lat", "lon"))
    percentage = (100.0 * snow_cells / valid_cells).where(valid_cells > 0)
    percentage.attrs = {
        "long_name": "share of valid cells that are snow covered",
        "units": "%",
    }
    return percentage


# -------------------------------------------------------------------------
# output
# -------------------------------------------------------------------------


def create_plottable_frame(frame):
    """Turn one snow flag frame into floats, with NaN where data is missing.

    Only ``imshow`` needs the float form, and only ever for a single time step,
    so the conversion happens per frame instead of on the whole record.

    Args:
        frame: Snow cover flag array of one time step.

    Returns
    -------
        Float array with 1 for snow, 0 for no snow and NaN for missing data.
    """
    plottable = frame.astype(float)
    plottable[frame < 0] = np.nan
    return plottable


def create_snow_cover_difference(input_frame, ref_frame):
    """Categorize the disagreement between two binary snow cover frames.

    Args:
        input_frame: Binary snow cover array of the input dataset.
        ref_frame: Binary snow cover array of the reference dataset.

    Returns
    -------
        Array with 0 for agreement, 1 for input-only snow, 2 for reference-only
        snow and NaN where either frame is missing.
    """
    difference = np.full(input_frame.shape, np.nan)
    both_valid = (input_frame >= 0) & (ref_frame >= 0)
    input_snow = both_valid & (input_frame == SNOW_VALUE)
    ref_snow = both_valid & (ref_frame == SNOW_VALUE)
    difference[both_valid] = 0
    difference[input_snow & ~ref_snow] = 1
    difference[ref_snow & ~input_snow] = 2
    return difference


@log_errors(raise_exceptions=True)
def create_classification_accuracy_map(
    accuracy_ds,
    output_file,
    input_name,
    ref_name,
    target_freq,
    accuracy_vmin=None,
    region_name=None,
    years=None,
):
    """Plot the snow presence classification accuracy as a map.

    Args:
        accuracy_ds: Dataset from ``calculate_classification_accuracy``.
        output_file: Target image path; parent folders are created.
        input_name: Label of the input dataset.
        ref_name: Label of the reference dataset.
        target_freq: Frequency alias the compared time steps refer to.
        accuracy_vmin: Lower end of the colour scale, None to derive it.
        region_name: Region shown in the title, None for the whole domain.
        years: First and last year of the compared period, None to omit it.

    Returns
    -------
        The written file path.
    """
    output_file = Path(output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    accuracy = accuracy_ds["classification_accuracy"]
    values = np.asarray(accuracy.values, dtype=float)
    values = np.where(np.isinf(values), np.nan, values)
    extent, origin = get_map_extent_and_origin(
        accuracy["lon"].values, accuracy["lat"].values
    )

    fig, ax = plt.subplots(figsize=calculate_map_figure_size(extent))
    # the upper end stays at the meaningful 1, the lower end follows the data
    image, bounds, extend, ticks = plot_single_map(
        ax,
        values,
        center=None,
        vmin=accuracy_vmin,
        vmax=1.0,
        cmap=ACCURACY_COLORMAP,
        bounds_type="data",
        under_color=ACCURACY_UNDER_COLOR,
        extent=extent,
        origin=origin,
        interpolation="nearest",
    )
    add_map_colorbar(
        fig,
        ax,
        image,
        bounds,
        extend,
        ticks,
        create_axis_label("Classification accuracy", accuracy.attrs.get("units")),
    )
    style_map_axes(ax)
    fig.suptitle(
        f"{create_comparison_title(input_name, ref_name, years)}"
        f"{format_region_title(region_name)}",
        fontweight="normal",
        fontsize="x-large",
    )
    ax.set_title(
        "Snow presence classification accuracy "
        f"({create_summary_text(values, bounds=bounds)})"
    )
    compared_steps = accuracy_ds["compared_time_steps"]
    ax.annotate(
        f"n = {int(compared_steps.min()):d} to {int(compared_steps.max()):d} "
        f"{target_freq} steps per cell    white = no data",
        xy=(0.0, -0.03),
        xycoords="axes fraction",
        va="top",
        fontsize="small",
    )
    fig.tight_layout()
    fig.savefig(output_file, dpi=PLOT_DPI)
    plt.close(fig)
    logger.info(f"Wrote classification accuracy map to {output_file}")
    return output_file


@log_errors(raise_exceptions=True)
def create_snow_cover_percentage_plot(
    input_flag, ref_flag, output_file, input_name, ref_name, region_name=None
):
    """Plot the share of snow covered cells over time.

    Cells north and south of the equator get their own panel, because their
    snow seasons are half a year apart. A domain that stays on one side of the
    equator gets a single coverage panel instead.

    Args:
        input_flag: Binary snow cover DataArray of the input dataset.
        ref_flag: Binary snow cover DataArray of the reference dataset.
        output_file: Target image path; parent folders are created.
        input_name: Label of the input dataset.
        ref_name: Label of the reference dataset.
        region_name: Region shown in the title, None for the whole domain.

    Returns
    -------
        The written file path.
    """
    output_file = Path(output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    time_values = pd.DatetimeIndex(input_flag.time.values)
    crosses_equator = (
        select_hemisphere(input_flag, northern=True).sizes["lat"] > 0
        and select_hemisphere(input_flag, northern=False).sizes["lat"] > 0
    )
    panels = (
        [("a) Northern hemisphere", True), ("b) Southern hemisphere", False)]
        if crosses_equator
        else [("Snow covered cells", None)]
    )

    fig, axes = plt.subplots(
        len(panels),
        1,
        figsize=(FIGURE_WIDTH, 3.8 * len(panels) + 0.8),
        sharex=True,
        squeeze=False,
    )
    axes = axes[:, 0]
    for ax, (panel_title, northern) in zip(axes, panels):
        ax.set_ylabel(create_axis_label("Snow covered cells", "%"))
        style_axes(ax, show_grid=True)
        percentages = []
        for flag in (input_flag, ref_flag):
            cells = flag if northern is None else select_hemisphere(flag, northern)
            percentages.append(calculate_snow_covered_cell_percentage(cells).values)
        summary = ", ".join(
            f"mean {name}={np.nanmean(values):.1f}%"
            for name, values in zip((input_name, ref_name), percentages)
        )
        ax.set_title(f"{panel_title} ({summary})", loc="left")
        for values, color, label in zip(
            percentages, (INPUT_COLOR, REF_COLOR), (input_name, ref_name)
        ):
            ax.plot(
                time_values,
                values,
                color=color,
                linewidth=1.8,
                label=label,
                solid_joinstyle="round",
            )
        upper_limit = float(np.nanmax(np.concatenate(percentages)))
        ax.set_ylim(0, min(100.0, max(10.0, np.ceil(upper_limit / 10) * 10)))
        ax.margins(x=0)

    # year ticks, thinned out so a long period stays readable
    span_years = max(1, round((time_values[-1] - time_values[0]).days / 365.25))
    tick_step = max(1, int(np.ceil(span_years / 15)))
    axes[-1].xaxis.set_major_locator(mdates.YearLocator(base=tick_step))
    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    if tick_step > 1:
        axes[-1].xaxis.set_minor_locator(mdates.YearLocator())
    # nudge the limits out to the enclosing year starts, so the first and last
    # year still get a tick; a wide gap is left alone instead of padded
    axis_start, axis_end = time_values[0], time_values[-1]
    year_start = pd.Timestamp(year=axis_start.year, month=1, day=1)
    next_year_start = pd.Timestamp(year=axis_end.year + 1, month=1, day=1)
    tick_snap = pd.Timedelta(days=45)
    if axis_start - year_start <= tick_snap:
        axis_start = year_start
    if next_year_start - axis_end <= tick_snap:
        axis_end = next_year_start
    axes[0].set_xlim(axis_start, axis_end)
    axes[-1].set_xlabel("year")
    # one legend for both panels
    axes[0].legend(loc="upper right")
    years = (time_values[0].year, time_values[-1].year)
    fig.suptitle(
        f"{create_comparison_title(input_name, ref_name, years)}"
        f"{format_region_title(region_name)}",
        fontweight="normal",
        fontsize="x-large",
    )
    fig.tight_layout()
    fig.savefig(output_file, dpi=PLOT_DPI)
    plt.close(fig)
    logger.info(f"Wrote snow cover percentage plot to {output_file}")
    return output_file


@log_errors(raise_exceptions=True)
def create_snow_cover_gif(
    input_flag,
    ref_flag,
    output_file,
    input_name,
    ref_name,
    frames_per_second=4,
    max_frames=0,
    region_name=None,
):
    """Render an animated three panel comparison of two binary snow cover fields.

    Panel one shows the input snow cover, panel two the reference snow cover
    and panel three their disagreement, with the input colour where only the
    input has snow and the reference colour where only the reference has snow.

    Args:
        input_flag: Binary snow cover DataArray of the input dataset.
        ref_flag: Binary snow cover DataArray of the reference dataset.
        output_file: Target gif path; parent folders are created.
        input_name: Label of the input dataset.
        ref_name: Label of the reference dataset.
        frames_per_second: Playback speed of the gif.
        max_frames: Maximum number of frames, 0 for all time steps.
        region_name: Region shown in the panel titles, None for the whole domain.

    Returns
    -------
        The written file path.
    """
    output_file = Path(output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    time_values = pd.DatetimeIndex(input_flag.time.values)
    frame_count = len(time_values)
    if max_frames and frame_count > max_frames:
        logger.warning(
            f"Limiting the gif to the first {max_frames} of {frame_count} time steps."
        )
        frame_count = max_frames
    elif frame_count > 500:
        logger.warning(
            f"Rendering {frame_count} frames. Use --max-gif-frames to shorten the gif."
        )

    snow_cmap = ListedColormap([NO_SNOW_COLOR, SNOW_COLOR]).with_extremes(
        bad=NO_DATA_COLOR
    )
    snow_norm = BoundaryNorm([-0.5, 0.5, 1.5], snow_cmap.N)
    diff_cmap = ListedColormap([AGREEMENT_COLOR, INPUT_COLOR, REF_COLOR]).with_extremes(
        bad=NO_DATA_COLOR
    )
    diff_norm = BoundaryNorm([-0.5, 0.5, 1.5, 2.5], diff_cmap.N)
    extent, origin = get_map_extent_and_origin(
        input_flag["lon"].values, input_flag["lat"].values
    )

    fig, axes = plt.subplots(
        1,
        3,
        figsize=calculate_map_figure_size(
            extent, panel_count=3, extra_width=0.6, extra_height=1.6
        ),
        sharex=True,
        sharey=True,
        constrained_layout=True,
    )
    fig.get_layout_engine().set(w_pad=0.08, h_pad=0.06, wspace=0.03)
    input_frame = np.asarray(input_flag.isel(time=0).values)
    ref_frame = np.asarray(ref_flag.isel(time=0).values)
    # nearest keeps the grid cells crisp instead of blurring their edges
    map_style = {"extent": extent, "origin": origin, "interpolation": "nearest"}
    images = [
        axes[0].imshow(
            create_plottable_frame(input_frame),
            cmap=snow_cmap,
            norm=snow_norm,
            **map_style,
        ),
        axes[1].imshow(
            create_plottable_frame(ref_frame),
            cmap=snow_cmap,
            norm=snow_norm,
            **map_style,
        ),
        axes[2].imshow(
            create_snow_cover_difference(input_frame, ref_frame),
            cmap=diff_cmap,
            norm=diff_norm,
            **map_style,
        ),
    ]
    region_title = format_region_title(region_name)
    for ax, title in zip(
        axes,
        [
            f"a) {input_name} snow cover",
            f"b) {ref_name} snow cover",
            "c) Difference",
        ],
    ):
        ax.set_title(title, fontsize="medium")
        style_map_axes(ax)
    # one legend row per colour scheme, below its own panels
    legend_style = {
        "loc": "upper center",
        "bbox_to_anchor": (0.5, -0.12),
        "frameon": False,
        "fontsize": "small",
        "ncol": 3,
        "handlelength": 1.4,
        "columnspacing": 1.2,
    }
    axes[1].legend(
        handles=[
            Patch(facecolor=SNOW_COLOR, label="snow"),
            Patch(facecolor=NO_SNOW_COLOR, label="no snow"),
            Patch(facecolor=NO_DATA_COLOR, edgecolor=AXIS_COLOR, label="no data"),
        ],
        **legend_style,
    )
    axes[2].legend(
        handles=[
            Patch(facecolor=INPUT_COLOR, label=f"only {input_name}"),
            Patch(facecolor=REF_COLOR, label=f"only {ref_name}"),
            Patch(facecolor=AGREEMENT_COLOR, label="agreement"),
        ],
        **legend_style,
    )
    title = create_comparison_title(input_name, ref_name)
    suptitle = fig.suptitle(
        f"{title}, {time_values[0].date()}{region_title}",
        fontweight="normal",
        fontsize="x-large",
    )

    def _update_frame(frame_index):
        input_values = np.asarray(input_flag.isel(time=frame_index).values)
        ref_values = np.asarray(ref_flag.isel(time=frame_index).values)
        images[0].set_data(create_plottable_frame(input_values))
        images[1].set_data(create_plottable_frame(ref_values))
        images[2].set_data(create_snow_cover_difference(input_values, ref_values))
        suptitle.set_text(f"{title}, {time_values[frame_index].date()}{region_title}")
        return [*images, suptitle]

    animation = FuncAnimation(
        fig, _update_frame, frames=frame_count, blit=False, repeat=False
    )
    animation.save(output_file, writer=PillowWriter(fps=frames_per_second))
    plt.close(fig)
    logger.info(f"Wrote snow cover gif with {frame_count} frames to {output_file}")
    return output_file


# -------------------------------------------------------------------------
# regions
# -------------------------------------------------------------------------


def create_region_outputs(
    input_flag,
    ref_flag,
    output_dir,
    region_names,
    input_name,
    ref_name,
    target_freq,
    accuracy_vmin=None,
    write_gif=True,
    gif_fps=4,
    max_gif_frames=0,
):
    """Write the coverage plot, accuracy map and gif for every region with data.

    Regions whose bounds hold no cell, or no time step where both datasets
    hold valid data, are skipped with a log message.

    Args:
        input_flag: Binary snow cover DataArray of the input dataset.
        ref_flag: Binary snow cover DataArray of the reference dataset.
        output_dir: Directory the region outputs are written to.
        region_names: Region names to process.
        input_name: Label of the input dataset.
        ref_name: Label of the reference dataset.
        target_freq: Frequency alias the time steps refer to.
        accuracy_vmin: Lower end of the accuracy colour scale, None to derive it.
        write_gif: Render the animated comparison per region.
        gif_fps: Playback speed of the gifs.
        max_gif_frames: Maximum number of gif frames, 0 for all time steps.

    Returns
    -------
        Dict of the written output file paths per region.
    """
    written_files = {}
    for region_name in region_names:
        region_input = crop_to_region(input_flag, region_name)
        region_ref = crop_to_region(ref_flag, region_name)
        if region_input.sizes["lat"] == 0 or region_input.sizes["lon"] == 0:
            logger.info(f"Skipping region {region_name}: no cells inside its bounds.")
            continue
        # the accuracy map needs time steps where both datasets hold data
        both_valid = get_valid_snow_mask(region_input) & get_valid_snow_mask(region_ref)
        valid_cells = int(both_valid.any("time").sum())
        if valid_cells == 0:
            logger.info(
                f"Skipping region {region_name}: no time step with valid data in "
                "both datasets inside it."
            )
            continue
        logger.info(
            f"Region {region_name}: {region_input.sizes['lat']} x "
            f"{region_input.sizes['lon']} cells, {valid_cells} of them with data "
            "in both datasets."
        )
        safe_name = sanitize_name(region_name)
        suffix = f"{input_name}_vs_{ref_name}_region_{safe_name}"
        region_time = pd.DatetimeIndex(region_input.time.values)
        written_files[f"{region_name} cover_percentage_plot"] = (
            create_snow_cover_percentage_plot(
                region_input,
                region_ref,
                output_dir / f"snow_cover_percentage_{suffix}.png",
                input_name=input_name,
                ref_name=ref_name,
                region_name=region_name,
            )
        )
        written_files[f"{region_name} accuracy_map"] = (
            create_classification_accuracy_map(
                calculate_classification_accuracy(region_input, region_ref),
                output_dir / f"classification_accuracy_{suffix}.png",
                input_name=input_name,
                ref_name=ref_name,
                target_freq=target_freq,
                accuracy_vmin=accuracy_vmin,
                region_name=region_name,
                years=(region_time[0].year, region_time[-1].year),
            )
        )
        if write_gif:
            written_files[f"{region_name} gif"] = create_snow_cover_gif(
                region_input,
                region_ref,
                output_dir / f"snow_cover_comparison_{suffix}.gif",
                input_name=input_name,
                ref_name=ref_name,
                frames_per_second=gif_fps,
                max_frames=max_gif_frames,
                region_name=region_name,
            )
    return written_files


# -------------------------------------------------------------------------
# main worker
# -------------------------------------------------------------------------


def snow_evaluation(  # noqa: PLR0913
    input_path,
    ref_path,
    output_dir,
    input_file_name="*.nc",
    ref_file_name="*.nc",
    input_var=None,
    ref_var=None,
    input_name="input",
    ref_name="ref",
    input_snow_threshold=0.0,
    ref_snow_threshold=0.0,
    target_freq=None,
    year_mode="snow_year",
    snow_year_start_month=9,
    accuracy_vmin=None,
    regions="all",
    write_snow_cover=True,
    write_gif=True,
    gif_fps=4,
    max_gif_frames=0,
    max_memory_gib=8.0,
    ncpus=1,
    compression=None,
    write_region_stats=False,
):
    """Evaluate the snow cover of an input dataset against a reference dataset.

    Both datasets are normalized to a common calendar and grid, reduced to a
    binary snow cover flag and summarized into yearly first snow day, last snow
    day and season duration maps. Results are written as NetCDF files and as an
    animated comparison.

    Args:
        input_path: File or directory holding the input data.
        ref_path: File or directory holding the reference data.
        output_dir: Directory the results are written to.
        input_file_name: Glob pattern for the recursive input search.
        ref_file_name: Glob pattern for the recursive reference search.
        input_var: Input variable name, None to detect it automatically.
        ref_var: Reference variable name, None to detect it automatically.
        input_name: Label of the input dataset used in file names and plots.
        ref_name: Label of the reference dataset used in file names and plots.
        input_snow_threshold: Input values above this count as snow-covered.
        ref_snow_threshold: Reference values above this count as snow-covered.
        target_freq: Frequency alias to force, None to derive it from the data.
        year_mode: Either "snow_year" or "calendar_year".
        snow_year_start_month: First month of a snow year.
        accuracy_vmin: Lower end of the accuracy colour scale, None to derive it.
        regions: "all", "none" or a comma separated list of region names.
        write_snow_cover: Write the normalized binary snow cover fields.
        write_gif: Render the animated snow cover comparison.
        gif_fps: Playback speed of the gif.
        max_gif_frames: Maximum number of gif frames, 0 for all time steps.
        max_memory_gib: Memory budget one time chunk is sized against.
        compression: NetCDF compression settings, or None.
        write_region_stats: Also write the statistics table of the domain and
            the regions to CSV; it is always logged.

    Returns
    -------
        Dict of the written output file paths.
    """
    output_dir = Path(output_dir)
    # resolve the regions first, so a typo fails before the data is read
    region_names = select_regions(regions)
    input_da = probe_snow_record(
        input_path, input_file_name, input_var, "input", ncpus, max_memory_gib
    )
    ref_da = probe_snow_record(
        ref_path, ref_file_name, ref_var, "reference", ncpus, max_memory_gib
    )

    input_da, ref_da = crop_datasets_to_spatial_overlap(input_da, ref_da)
    input_da, ref_da = regrid_to_coarser_grid(input_da, ref_da)
    input_da, ref_da = crop_datasets_to_spatial_overlap(input_da, ref_da)

    # the overlap comes from the time coordinates alone; scanning the values for
    # empty steps would read both records in full, and the one-pass counts drop
    # a bin nothing fell into anyway
    input_coverage = get_data_coverage(input_da)
    ref_coverage = get_data_coverage(ref_da)
    time_slice = slice(
        max(input_coverage[0], ref_coverage[0]),
        min(input_coverage[1], ref_coverage[1]),
    )
    if time_slice.start >= time_slice.stop:
        msg = (
            f"The two records do not overlap in time: the input covers "
            f"{input_coverage[0].date()} to {input_coverage[1].date()} and the "
            f"reference {ref_coverage[0].date()} to {ref_coverage[1].date()}."
        )
        with ErrorLogger(logger):
            raise ValueError(msg)
    input_da = input_da.sel(time=time_slice)
    ref_da = ref_da.sel(time=time_slice)
    if target_freq is None:
        target_freq = get_target_frequency(input_da, ref_da)
    # only the shared grid and calendar come from the lazy records; the values
    # are read again one file at a time so the raw fields never have to fit
    bin_labels = create_target_bins(time_slice, target_freq)
    target_grid = (bin_labels, input_da["lat"].values, input_da["lon"].values)
    del input_da, ref_da

    input_flag = create_snow_flag_one_pass(
        input_path,
        input_file_name,
        input_var,
        "input",
        target_grid,
        target_freq,
        time_slice,
        input_snow_threshold,
        ncpus=ncpus,
        max_memory_gib=max_memory_gib,
    )
    ref_flag = create_snow_flag_one_pass(
        ref_path,
        ref_file_name,
        ref_var,
        "reference",
        target_grid,
        target_freq,
        time_slice,
        ref_snow_threshold,
        ncpus=ncpus,
        max_memory_gib=max_memory_gib,
    )
    input_flag, ref_flag = drop_uncovered_bins(input_flag, ref_flag, target_freq)

    step_days = float(
        np.median(np.diff(input_flag.time.values)) / np.timedelta64(1, "D")
    )
    year_windows = create_year_windows(
        pd.DatetimeIndex(input_flag.time.values),
        year_mode,
        snow_year_start_month,
        step_days,
    )
    logger.info(
        f"Evaluating {len(year_windows)} {year_mode} window(s) at {target_freq} "
        f"resolution with a median time step of {step_days:g} days."
    )
    input_metrics = calculate_snow_season_metrics(input_flag, year_windows, target_freq)
    ref_metrics = calculate_snow_season_metrics(ref_flag, year_windows, target_freq)
    difference_metrics = compare_snow_season_metrics(input_metrics, ref_metrics)
    accuracy = calculate_classification_accuracy(input_flag, ref_flag)

    common_attrs = {
        "target_frequency": target_freq,
        "year_mode": year_mode,
        "snow_year_start_month": snow_year_start_month,
        "time_step_days": step_days,
    }
    input_metrics.attrs.update(
        {**common_attrs, "dataset": input_name, "snow_threshold": input_snow_threshold}
    )
    ref_metrics.attrs.update(
        {**common_attrs, "dataset": ref_name, "snow_threshold": ref_snow_threshold}
    )
    difference_metrics.attrs.update(
        {**common_attrs, "dataset": f"{input_name} minus {ref_name}"}
    )
    accuracy.attrs.update({**common_attrs, "dataset": f"{input_name} vs {ref_name}"})
    region_stats_file = (
        output_dir / f"snow_region_stats_{input_name}_vs_{ref_name}.csv"
        if write_region_stats
        else None
    )
    create_snow_stats_table(
        accuracy,
        difference_metrics,
        region_names,
        input_name,
        ref_name,
        output_file=region_stats_file,
    )

    input_time = pd.DatetimeIndex(input_flag.time.values)
    written_files = {
        "input_metrics": write_snow_dataset(
            input_metrics,
            output_dir / f"snow_season_metrics_{input_name}.nc",
            compression=compression,
        ),
        "ref_metrics": write_snow_dataset(
            ref_metrics,
            output_dir / f"snow_season_metrics_{ref_name}.nc",
            compression=compression,
        ),
        "difference_metrics": write_snow_dataset(
            difference_metrics,
            output_dir
            / f"snow_season_metrics_difference_{input_name}_minus_{ref_name}.nc",
            compression=compression,
        ),
        "accuracy": write_snow_dataset(
            accuracy,
            output_dir / f"classification_accuracy_{input_name}_vs_{ref_name}.nc",
            compression=compression,
        ),
        "accuracy_map": create_classification_accuracy_map(
            accuracy,
            output_dir / f"classification_accuracy_{input_name}_vs_{ref_name}.png",
            input_name=input_name,
            ref_name=ref_name,
            target_freq=target_freq,
            accuracy_vmin=accuracy_vmin,
            years=(input_time[0].year, input_time[-1].year),
        ),
        "cover_percentage_plot": create_snow_cover_percentage_plot(
            input_flag,
            ref_flag,
            output_dir / f"snow_cover_percentage_{input_name}_vs_{ref_name}.png",
            input_name=input_name,
            ref_name=ref_name,
        ),
    }
    if region_stats_file is not None:
        written_files["region_stats"] = region_stats_file
    if write_snow_cover:
        written_files["input_snow_cover"] = write_snow_dataset(
            input_flag.to_dataset(name="snow_cover"),
            output_dir / f"snow_cover_{input_name}.nc",
            compression=compression,
        )
        written_files["ref_snow_cover"] = write_snow_dataset(
            ref_flag.to_dataset(name="snow_cover"),
            output_dir / f"snow_cover_{ref_name}.nc",
            compression=compression,
        )
    written_files.update(
        create_region_outputs(
            input_flag,
            ref_flag,
            output_dir,
            region_names,
            input_name=input_name,
            ref_name=ref_name,
            target_freq=target_freq,
            accuracy_vmin=accuracy_vmin,
            write_gif=write_gif,
            gif_fps=gif_fps,
            max_gif_frames=max_gif_frames,
        )
    )
    if write_gif:
        written_files["gif"] = create_snow_cover_gif(
            input_flag,
            ref_flag,
            output_dir / f"snow_cover_comparison_{input_name}_vs_{ref_name}.gif",
            input_name=input_name,
            ref_name=ref_name,
            frames_per_second=gif_fps,
            max_frames=max_gif_frames,
        )
    return written_files
