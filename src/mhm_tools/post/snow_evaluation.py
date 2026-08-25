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
- Snow flag: values above the snow threshold become 1 (snow), all other valid
  values become 0 (no snow), missing values stay missing.
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

import matplotlib as mpl
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.patches import Patch
from mpl_toolkits.axes_grid1 import make_axes_locatable

from mhm_tools.common.constants import WMO_REGION_BOUNDS
from mhm_tools.common.file_handler import (
    ChunkType,
    get_dataset_from_path,
    write_xarray_to_file,
)
from mhm_tools.common.logger import ErrorLogger
from mhm_tools.common.time_utils import (
    get_period_freq,
    get_step_days,
    resample_to_target_freq,
    timedelta_to_alias,
)
from mhm_tools.common.utils import sanitize_name
from mhm_tools.common.xarray_utils import (
    crop_ds,
    get_coord_key,
    get_ds_extend,
    get_overlapping_time_slice,
    get_single_data_var,
    normalize_lat_lon,
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
NANOSECONDS_PER_DAY = 86_400_000_000_000

# Neutral state ramp for the binary panels. The light end is dark enough to
# stay visible against the page, which also separates it from missing data.
NO_SNOW_COLOR = "#9fb0b8"
SNOW_COLOR = "#37474f"
NO_DATA_COLOR = "#ffffff"
# Dataset identity, one hue each, kept the same in every figure. Validated as a
# categorical pair: protan dE 22.9, normal dE 31.9, both above the surface floor.
INPUT_COLOR = "#1f6feb"
REF_COLOR = "#d1495b"
AGREEMENT_COLOR = NO_SNOW_COLOR
# single hue light to dark, so a high accuracy reads as a strong colour
ACCURACY_COLORMAP = "Blues"
ACCURACY_UNDER_COLOR = "#f7f4d8"
# recessive ink for axes, labels and captions
AXIS_COLOR = "#9aa5ab"
CAPTION_COLOR = "#5b6770"
GRID_COLOR = "#e4e8ea"


# ---------------------------------------------------------------------------
# reading and writing
# ---------------------------------------------------------------------------
def normalize_time(obj, new_time_key="time"):
    """Rename the time dimension and coordinate to ``time``.

    Args:
        obj: Dataset or DataArray to rename.
        new_time_key: Target time name.

    Returns
    -------
        The renamed object.
    """
    time_key = get_coord_key(obj, time=True)
    if time_key == new_time_key:
        return obj
    logger.info(f"Normalizing time coordinate name: {time_key} -> {new_time_key}")
    return obj.rename({time_key: new_time_key})


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

    da = normalize_time(normalize_lat_lon(ds[var_name])).squeeze(drop=True)
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


def write_snow_dataset(ds, file_path):
    """Write a snow evaluation dataset to a compressed NetCDF file.

    Date variables get explicit time units, so the file states one calendar
    instead of letting every variable pick its own. Integer variables carry the
    snow flag's missing sentinel as their fill value, so a reader decodes it
    back to NaN instead of reading -1 as data.

    Args:
        ds: Dataset to write.
        file_path: Target file path; parent folders are created.

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
        elif name in ds.data_vars:
            encoding[name] = {"zlib": True, "complevel": 4}
            if np.issubdtype(ds[name].dtype, np.integer):
                encoding[name]["_FillValue"] = MISSING_VALUE
    logger.info(f"Writing {file_path}")
    write_xarray_to_file(ds, file_path, encoding=encoding)
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


def calculate_coordinate_resolution(coord):
    """Return the median absolute spacing of a 1-D coordinate.

    Args:
        coord: Coordinate DataArray or array with at least two values.

    Returns
    -------
        The spacing as float.
    """
    values = np.asarray(coord)
    if values.ndim != 1 or values.size < 2:
        msg = (
            f"Cannot determine the resolution of a coordinate with shape "
            f"{values.shape}. Two-dimensional or single-cell grids are not supported."
        )
        with ErrorLogger(logger):
            raise ValueError(msg)
    return float(np.nanmedian(np.abs(np.diff(values))))


def create_cell_edges(centers):
    """Create cell edges from monotonic cell centers.

    Args:
        centers: 1-D array of cell centers.

    Returns
    -------
        Array of ``len(centers) + 1`` cell edges in the order of the centers.
    """
    centers = np.asarray(centers, dtype=float)
    if centers.size < 2:
        msg = "Cannot create cell edges from less than two cell centers."
        with ErrorLogger(logger):
            raise ValueError(msg)
    inner_edges = 0.5 * (centers[:-1] + centers[1:])
    first_edge = centers[0] - (inner_edges[0] - centers[0])
    last_edge = centers[-1] + (centers[-1] - inner_edges[-1])
    return np.concatenate([[first_edge], inner_edges, [last_edge]])


def align_to_target_grid(da, target_lat, target_lon):
    """Snap a DataArray onto target coordinates of the same resolution.

    Args:
        da: DataArray with ``lat`` and ``lon``.
        target_lat: Target latitude values.
        target_lon: Target longitude values.

    Returns
    -------
        The DataArray reindexed to the target coordinates.
    """
    tolerance = (
        max(
            calculate_coordinate_resolution(target_lat),
            calculate_coordinate_resolution(target_lon),
        )
        / 2
    )
    return da.reindex(
        lat=np.asarray(target_lat),
        lon=np.asarray(target_lon),
        method="nearest",
        tolerance=tolerance,
    )


def aggregate_to_target_grid(da, target_lat, target_lon):
    """Aggregate a fine DataArray onto a coarser target grid by spatial mean.

    Uses ``coarsen`` when the target grid is an aligned integer multiple of the
    source grid and falls back to binning the source cells into the target
    cells otherwise.

    Args:
        da: Fine DataArray with ``lat`` and ``lon``.
        target_lat: Coarse latitude values.
        target_lon: Coarse longitude values.

    Returns
    -------
        The aggregated DataArray on the target grid.
    """
    lat_factor = round(
        calculate_coordinate_resolution(target_lat)
        / calculate_coordinate_resolution(da["lat"])
    )
    lon_factor = round(
        calculate_coordinate_resolution(target_lon)
        / calculate_coordinate_resolution(da["lon"])
    )
    if (
        lat_factor > 1
        and lon_factor > 1
        and da.sizes["lat"] % lat_factor == 0
        and da.sizes["lon"] % lon_factor == 0
    ):
        coarsened = da.coarsen(lat=lat_factor, lon=lon_factor, boundary="trim").mean()
        if coarsened.sizes["lat"] == len(target_lat) and coarsened.sizes["lon"] == len(
            target_lon
        ):
            logger.info(
                f"Aggregating with coarsen by factors lat={lat_factor}, lon={lon_factor}."
            )
            return coarsened.assign_coords(
                lat=np.asarray(target_lat), lon=np.asarray(target_lon)
            )
        logger.debug(
            f"Coarsen produced {coarsened.sizes['lat']}x{coarsened.sizes['lon']} cells "
            f"instead of {len(target_lat)}x{len(target_lon)}; binning instead."
        )

    logger.info("Aggregating fine cells into the coarse target cells by mean.")
    lat_edges = create_cell_edges(target_lat)
    lon_edges = create_cell_edges(target_lon)
    # groupby_bins needs ascending edges, so reverse the labels for descending grids
    lat_labels = target_lat if lat_edges[0] < lat_edges[-1] else target_lat[::-1]
    lon_labels = target_lon if lon_edges[0] < lon_edges[-1] else target_lon[::-1]
    binned = (
        da.groupby_bins("lat", np.sort(lat_edges), labels=np.asarray(lat_labels))
        .mean("lat")
        .groupby_bins("lon", np.sort(lon_edges), labels=np.asarray(lon_labels))
        .mean("lon")
        .rename({"lat_bins": "lat", "lon_bins": "lon"})
    )
    return binned.reindex(lat=np.asarray(target_lat), lon=np.asarray(target_lon))


def regrid_to_coarser_grid(input_da, ref_da):
    """Bring two DataArrays onto the coarser of their two grids.

    Args:
        input_da: Input DataArray with ``lat`` and ``lon``.
        ref_da: Reference DataArray with ``lat`` and ``lon``.

    Returns
    -------
        Tuple of the two DataArrays on the common coarse grid.
    """
    input_cell_area = calculate_coordinate_resolution(
        input_da["lat"]
    ) * calculate_coordinate_resolution(input_da["lon"])
    ref_cell_area = calculate_coordinate_resolution(
        ref_da["lat"]
    ) * calculate_coordinate_resolution(ref_da["lon"])

    if np.isclose(input_cell_area, ref_cell_area, rtol=1e-3):
        logger.info("Input and reference share the same grid resolution.")
        return align_to_target_grid(input_da, ref_da["lat"], ref_da["lon"]), ref_da
    if input_cell_area > ref_cell_area:
        logger.info("Reference grid is finer; aggregating it onto the input grid.")
        return input_da, aggregate_to_target_grid(
            ref_da, input_da["lat"].values, input_da["lon"].values
        )
    logger.info("Input grid is finer; aggregating it onto the reference grid.")
    return (
        aggregate_to_target_grid(input_da, ref_da["lat"].values, ref_da["lon"].values),
        ref_da,
    )


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


def materialize_snow_flag(snow_flag, max_memory_gib, label):
    """Compute a snow cover flag once, if it fits the memory budget.

    Every consumer reads the flag again - once per year window, per region and
    per gif frame - so leaving it lazy makes dask rebuild the whole read,
    regrid and resample graph each time. Computing it once trades that for one
    array, and as a signed byte it is an eighth of the float it replaces. A
    record too large for the budget stays lazy and correct, only slower.

    Args:
        snow_flag: Snow cover flag DataArray.
        max_memory_gib: Memory budget in GiB.
        label: Name used in log messages.

    Returns
    -------
        The computed or the unchanged DataArray.
    """
    estimated_gib = snow_flag.size * snow_flag.dtype.itemsize / 1024**3
    if estimated_gib <= max_memory_gib:
        logger.info(f"Computing {label} once ({estimated_gib:.2f} GiB).")
        return snow_flag.load()
    logger.warning(
        f"Keeping {label} lazy because {estimated_gib:.2f} GiB exceeds the budget "
        f"of {max_memory_gib} GiB. It is recomputed per year, region and frame, "
        "so this is markedly slower; raise --max-memory-gib if the memory is there."
    )
    return snow_flag


def get_valid_snow_mask(snow_flag):
    """Return where a snow cover flag holds data rather than the missing sentinel.

    Args:
        snow_flag: Snow cover flag DataArray.

    Returns
    -------
        Boolean DataArray, True where the cell was observed.
    """
    return snow_flag >= 0


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


def format_region_title(region_name):
    """Format a region name as a title suffix.

    Args:
        region_name: Region name or None for the whole domain.

    Returns
    -------
        The suffix string, empty when no region is given.
    """
    return f" - {region_name}" if region_name else ""


def crop_to_region(snow_flag, region_name):
    """Crop a snow cover field to the bounds of an evaluation region.

    Args:
        snow_flag: DataArray with ``lat`` and ``lon`` coordinates.
        region_name: Key of ``WMO_REGION_BOUNDS``.

    Returns
    -------
        The DataArray reduced to the region's bounds.
    """
    bounds = WMO_REGION_BOUNDS[region_name]
    lon_slice = bounds["lon_slice"]
    lat_slice = bounds["lat_slice"]
    return crop_ds(
        snow_flag, lon_slice.start, lon_slice.stop, lat_slice.start, lat_slice.stop
    )


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


def calculate_map_figure_size(
    da,
    panel_count=1,
    max_panel_width=4.6,
    max_panel_height=6.5,
    extra_width=0.0,
    extra_height=1.2,
):
    """Size a map figure from the data aspect so the panels fill the canvas.

    ``imshow`` keeps geographic cells square, so a figure that ignores the
    lon/lat ratio leaves the map floating in white space. One panel is fitted
    into the given box at the data aspect and the figure follows from that.

    Args:
        da: DataArray with ``lat`` and ``lon`` coordinates.
        panel_count: Number of map panels side by side.
        max_panel_width: Largest width of one panel in inches.
        max_panel_height: Largest height of one panel in inches.
        extra_width: Inches reserved for labels, colorbars and padding.
        extra_height: Inches reserved for titles, captions and legends.

    Returns
    -------
        The (width, height) figure size in inches.
    """
    lon_min, lon_max, lat_min, lat_max = get_ds_extend(da)
    lat_span = lat_max - lat_min
    aspect = (lon_max - lon_min) / lat_span if lat_span > 0 else 1.0
    panel_height = min(max_panel_height, max_panel_width / aspect)
    # keep a very thin domain from collapsing into a line
    panel_width = max(panel_height * aspect, 2.0)
    return (panel_count * panel_width + extra_width, panel_height + extra_height)


def style_axes(ax, show_grid=False):
    """Apply the recessive axis styling shared by every figure.

    Args:
        ax: Matplotlib axes to style.
        show_grid: Draw a light horizontal grid behind the data.

    Returns
    -------
        The styled axes.
    """
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color(AXIS_COLOR)
        ax.spines[spine].set_linewidth(0.8)
    ax.tick_params(colors=CAPTION_COLOR, labelcolor=CAPTION_COLOR, length=3, width=0.8)
    if show_grid:
        ax.grid(axis="y", color=GRID_COLOR, linewidth=0.8)
        ax.set_axisbelow(True)
    return ax


def get_map_extent_and_origin(da):
    """Return the imshow extent and origin for a lat/lon DataArray.

    Args:
        da: DataArray with ``lat`` and ``lon`` coordinates.

    Returns
    -------
        Tuple of (extent tuple, origin string).
    """
    lon_min, lon_max, lat_min, lat_max = get_ds_extend(da)
    lat_values = np.asarray(da["lat"].values)
    origin = "lower" if lat_values[0] <= lat_values[-1] else "upper"
    return (lon_min, lon_max, lat_min, lat_max), origin


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


def create_accuracy_levels(accuracy, accuracy_vmin=None):
    """Create colour levels that resolve the spread of an accuracy field.

    A fixed 0 to 1 ramp leaves most of its steps unused, because accuracies
    usually sit in a narrow band near 1. The lower bound therefore follows the
    data unless it is given, while the upper bound stays at the meaningful 1.

    Args:
        accuracy: Accuracy DataArray with values between 0 and 1.
        accuracy_vmin: Lower bound to force, None to derive it from the data.

    Returns
    -------
        Array of level edges from the lower bound up to 1.
    """
    if accuracy_vmin is None:
        data_min = float(np.nanmin(accuracy))
        if not np.isfinite(data_min):
            data_min = 0.0
        # round down to a 0.05 step so the tick labels stay readable
        accuracy_vmin = np.floor(data_min * 20) / 20
    accuracy_vmin = min(max(float(accuracy_vmin), 0.0), 0.95)
    span = 1.0 - accuracy_vmin
    step = next(
        (
            candidate
            for candidate in (0.01, 0.02, 0.05, 0.1, 0.2)
            if span / candidate <= 10
        ),
        0.2,
    )
    return np.round(np.arange(accuracy_vmin, 1.0 + step / 2, step), 4)


def create_classification_accuracy_map(
    accuracy_ds,
    output_file,
    input_name,
    ref_name,
    target_freq,
    accuracy_vmin=None,
    region_name=None,
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

    Returns
    -------
        The written file path.
    """
    output_file = Path(output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    accuracy = accuracy_ds["classification_accuracy"]
    levels = create_accuracy_levels(accuracy, accuracy_vmin)
    colormap = mpl.colormaps[ACCURACY_COLORMAP].resampled(len(levels) - 1)
    colormap = colormap.with_extremes(bad="white", under=ACCURACY_UNDER_COLOR)
    norm = BoundaryNorm(levels, colormap.N)
    extent, origin = get_map_extent_and_origin(accuracy)
    data_min = float(np.nanmin(accuracy))
    data_max = float(np.nanmax(accuracy))

    fig, ax = plt.subplots(
        figsize=calculate_map_figure_size(
            accuracy,
            max_panel_width=8.0,
            max_panel_height=6.4,
            extra_width=2.6,
            extra_height=1.3,
        )
    )
    image = ax.imshow(
        np.asarray(accuracy.values, dtype=float),
        cmap=colormap,
        norm=norm,
        extent=extent,
        origin=origin,
        interpolation="nearest",
    )
    # a divided axis keeps the colorbar exactly as tall as the map
    colorbar_ax = make_axes_locatable(ax).append_axes("right", size="3.5%", pad=0.2)
    colorbar = fig.colorbar(
        image,
        cax=colorbar_ax,
        ticks=levels,
        extend="min" if data_min < levels[0] else "neither",
    )
    colorbar.set_label("share of matching time steps", color=CAPTION_COLOR, labelpad=10)
    colorbar.outline.set_visible(False)
    colorbar.ax.tick_params(colors=CAPTION_COLOR, labelcolor=CAPTION_COLOR, length=3)
    ax.set_xlabel("lon")
    ax.set_ylabel("lat")
    style_axes(ax)
    compared_steps = accuracy_ds["compared_time_steps"]
    ax.set_title(
        f"snow presence classification accuracy{format_region_title(region_name)}\n"
        f"{input_name} vs {ref_name}",
        fontsize="large",
        pad=12,
    )
    # state the real range, so a clipped colour scale cannot mislead
    ax.annotate(
        f"domain mean {float(accuracy.mean(skipna=True)):.2f}    "
        f"range {data_min:.2f} to {data_max:.2f}    "
        f"n = {int(compared_steps.min()):d} to {int(compared_steps.max()):d} "
        f"{target_freq} steps per cell    white = no data",
        xy=(0.0, -0.13),
        xycoords="axes fraction",
        fontsize="small",
        color=CAPTION_COLOR,
    )
    logger.info(f"Writing {output_file}")
    fig.savefig(output_file, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return output_file


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
        [("northern hemisphere", True), ("southern hemisphere", False)]
        if crosses_equator
        else [(None, None)]
    )

    fig, axes = plt.subplots(
        len(panels),
        1,
        figsize=(12, 3.8 * len(panels)),
        sharex=True,
        squeeze=False,
        constrained_layout=True,
    )
    axes = axes[:, 0]
    for ax, (panel_title, northern) in zip(axes, panels):
        if panel_title is not None:
            ax.set_title(
                panel_title, fontsize="medium", loc="left", color=CAPTION_COLOR, pad=6
            )
        ax.set_ylabel("snow covered cells [%]")
        style_axes(ax, show_grid=True)
        percentages = []
        for flag in (input_flag, ref_flag):
            cells = flag if northern is None else select_hemisphere(flag, northern)
            percentages.append(calculate_snow_covered_cell_percentage(cells).values)
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
    # one legend for both panels, below the axes so it never covers the lines
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="outside lower center",
        ncol=2,
        frameon=False,
        fontsize="medium",
    )
    fig.suptitle(
        f"snow covered share of the domain{format_region_title(region_name)}\n"
        f"{input_name} vs {ref_name}",
        fontsize="large",
    )
    logger.info(f"Writing {output_file}")
    fig.savefig(output_file, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return output_file


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
    extent, origin = get_map_extent_and_origin(input_flag)

    fig, axes = plt.subplots(
        1,
        3,
        figsize=calculate_map_figure_size(
            input_flag,
            panel_count=3,
            max_panel_width=4.6,
            max_panel_height=6.2,
            extra_width=1.4,
            extra_height=1.45,
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
            f"{input_name} snow cover",
            f"{ref_name} snow cover",
            "difference",
        ],
    ):
        ax.set_title(title, fontsize="medium", color=CAPTION_COLOR, pad=8)
        ax.set_xlabel("lon")
        style_axes(ax)
    axes[0].set_ylabel("lat")
    # one legend row per colour scheme, below its own panels
    legend_style = {
        "loc": "upper center",
        "bbox_to_anchor": (0.5, -0.12),
        "frameon": False,
        "fontsize": "small",
        "ncol": 3,
        "handlelength": 1.4,
        "columnspacing": 1.2,
        "labelcolor": CAPTION_COLOR,
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
    suptitle = fig.suptitle(f"{time_values[0].date()}{region_title}", fontsize="large")

    def _update_frame(frame_index):
        input_values = np.asarray(input_flag.isel(time=frame_index).values)
        ref_values = np.asarray(ref_flag.isel(time=frame_index).values)
        images[0].set_data(create_plottable_frame(input_values))
        images[1].set_data(create_plottable_frame(ref_values))
        images[2].set_data(create_snow_cover_difference(input_values, ref_values))
        suptitle.set_text(f"{time_values[frame_index].date()}{region_title}")
        return [*images, suptitle]

    logger.info(f"Writing {output_file} with {frame_count} frames.")
    animation = FuncAnimation(
        fig, _update_frame, frames=frame_count, blit=False, repeat=False
    )
    animation.save(output_file, writer=PillowWriter(fps=frames_per_second))
    plt.close(fig)
    return output_file


# -------------------------------------------------------------------------
# regions
# -------------------------------------------------------------------------


def select_regions(requested_regions):
    """Resolve the requested region names against the known regions.

    Args:
        requested_regions: "all", "none" or a comma separated list of names.

    Returns
    -------
        List of region names, empty when no regional output is wanted.
    """
    requested = str(requested_regions).strip()
    if requested.lower() in {"none", ""}:
        return []
    if requested.lower() == "all":
        return list(WMO_REGION_BOUNDS)
    selected = []
    lookup = {name.lower(): name for name in WMO_REGION_BOUNDS}
    for entry in requested.split(","):
        key = entry.strip().lower()
        if key in lookup:
            selected.append(lookup[key])
        elif key:
            msg = (
                f"Unknown region {entry.strip()!r}. "
                f"Available regions: {', '.join(WMO_REGION_BOUNDS)}."
            )
            with ErrorLogger(logger):
                raise ValueError(msg)
    return selected


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

    Regions whose bounds hold no cell or no valid time step of the normalized
    domain are skipped with a log message.

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
        valid_cells = int(get_valid_snow_mask(region_input).any("time").sum())
        if valid_cells == 0:
            logger.info(f"Skipping region {region_name}: no valid data inside it.")
            continue
        logger.info(
            f"Region {region_name}: {region_input.sizes['lat']} x "
            f"{region_input.sizes['lon']} cells, {valid_cells} of them with data."
        )
        safe_name = sanitize_name(region_name)
        suffix = f"{input_name}_vs_{ref_name}_region_{safe_name}"
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

    Returns
    -------
        Dict of the written output file paths.
    """
    output_dir = Path(output_dir)
    # resolve the regions first, so a typo fails before the data is read
    region_names = select_regions(regions)
    input_da = read_snow_data_array(
        input_path, input_file_name, input_var, "input", max_memory_gib
    )
    ref_da = read_snow_data_array(
        ref_path, ref_file_name, ref_var, "reference", max_memory_gib
    )

    input_da, ref_da = crop_datasets_to_spatial_overlap(input_da, ref_da)
    input_da, ref_da = regrid_to_coarser_grid(input_da, ref_da)
    input_da, ref_da = crop_datasets_to_spatial_overlap(input_da, ref_da)

    time_slice = get_overlapping_time_slice(input_da, ref_da)
    input_da = input_da.sel(time=time_slice)
    ref_da = ref_da.sel(time=time_slice)
    if target_freq is None:
        target_freq = get_target_frequency(input_da, ref_da)
    input_da, ref_da = resample_snow_to_target_frequency(input_da, ref_da, target_freq)

    input_flag = create_snow_cover_flag(input_da, input_snow_threshold)
    ref_flag = create_snow_cover_flag(ref_da, ref_snow_threshold)
    input_flag = materialize_snow_flag(input_flag, max_memory_gib, "input snow cover")
    ref_flag = materialize_snow_flag(ref_flag, max_memory_gib, "reference snow cover")
    # the raw fields are the largest arrays in play and are done with once the
    # flags are computed, so let them go before the metrics start
    del input_da, ref_da

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

    written_files = {
        "input_metrics": write_snow_dataset(
            input_metrics, output_dir / f"snow_season_metrics_{input_name}.nc"
        ),
        "ref_metrics": write_snow_dataset(
            ref_metrics, output_dir / f"snow_season_metrics_{ref_name}.nc"
        ),
        "difference_metrics": write_snow_dataset(
            difference_metrics,
            output_dir
            / f"snow_season_metrics_difference_{input_name}_minus_{ref_name}.nc",
        ),
        "accuracy": write_snow_dataset(
            accuracy,
            output_dir / f"classification_accuracy_{input_name}_vs_{ref_name}.nc",
        ),
        "accuracy_map": create_classification_accuracy_map(
            accuracy,
            output_dir / f"classification_accuracy_{input_name}_vs_{ref_name}.png",
            input_name=input_name,
            ref_name=ref_name,
            target_freq=target_freq,
            accuracy_vmin=accuracy_vmin,
        ),
        "cover_percentage_plot": create_snow_cover_percentage_plot(
            input_flag,
            ref_flag,
            output_dir / f"snow_cover_percentage_{input_name}_vs_{ref_name}.png",
            input_name=input_name,
            ref_name=ref_name,
        ),
    }
    if write_snow_cover:
        written_files["input_snow_cover"] = write_snow_dataset(
            input_flag.to_dataset(name="snow_cover"),
            output_dir / f"snow_cover_{input_name}.nc",
        )
        written_files["ref_snow_cover"] = write_snow_dataset(
            ref_flag.to_dataset(name="snow_cover"),
            output_dir / f"snow_cover_{ref_name}.nc",
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
