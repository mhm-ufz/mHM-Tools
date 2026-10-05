"""Time-related helpers for resampling datasets."""

import logging
import re
from typing import Literal, Tuple, Union

import numpy as np
import pandas as pd
import xarray as xr

from mhm_tools.common.logger import ErrorLogger
from mhm_tools.common.netcdf import generate_bounds

logger = logging.getLogger(__name__)

FIXED_DAY_ALIAS = re.compile(r"(\d*)D")

# A step this much larger than the median one is a gap rather than a rhythm,
# and the windows around it cannot be derived from the stamps.
IRREGULAR_SPACING_TOLERANCE = 1.5

# A CF bounds variable spans a second dimension, which a DataArray of one
# variable cannot carry, so the two edges travel as a pair of 1-D coordinates.
TIME_BOUNDS_START_COORD = "time_bounds_start"
TIME_BOUNDS_END_COORD = "time_bounds_end"


def get_step_days(target_freq):
    """Return the fixed length in days of a frequency, or None for calendar steps.

    ``D`` and ``8D`` cover a fixed number of days, while ``W`` and ``ME`` follow
    the calendar and need period arithmetic.

    Args:
        target_freq: Pandas frequency alias.

    Returns
    -------
        The number of days, or None when the step is a calendar period.
    """
    match = FIXED_DAY_ALIAS.fullmatch(str(target_freq).upper())
    return int(match.group(1) or 1) if match else None


def get_period_freq(target_freq):
    """Map a frequency alias to the pandas period alias of one time step.

    Args:
        target_freq: Pandas frequency alias.

    Returns
    -------
        The period alias ("D", "W" or "M"), or None for sub-daily data.
    """
    alias = str(target_freq).upper()
    if alias in {"ME", "MS", "M"}:
        return "M"
    if alias.startswith("W"):
        return "W"
    return "D" if alias == "D" else None


def calculate_median_time_step_seconds(time_values):
    """Calculate the median step length of a time axis in seconds.

    Args:
        time_values: Datetime64 or cftime values of the time axis.

    Returns
    -------
        The median of the positive steps in seconds, None for fewer than two
        distinct time values.
    """
    values = np.asarray(time_values)
    if values.size < 2:
        return None
    if np.issubdtype(values.dtype, np.datetime64):
        steps = np.diff(values.astype("datetime64[ns]").astype(np.int64)) / 1e9
    else:
        steps = np.array([step.total_seconds() for step in np.diff(values)])
    steps = steps[steps > 0]
    return float(np.median(steps)) if steps.size else None


def timedelta_to_alias(ds: xr.DataArray) -> str:
    """Map a median timedelta to a pandas frequency alias.

    - ~1 day -> 'D'
    - ~7 days -> 'W'
    - ~28-31 days -> 'ME'
    - otherwise: fall back to '<N>h'

    """
    time = getattr(ds, "time", None)
    if time is None:
        msg = "Object has no 'time' coordinate."
        with ErrorLogger(logger):
            raise ValueError(msg)
    if time.size < 2:
        msg = (
            "Cannot infer time frequency because only "
            f"{time.size} timestamp{'s' if time.size != 1 else ''} are present."
        )
        raise ValueError(msg)
    try:
        median_delta = ds.time.diff("time").median()
    except Exception as e:
        logger.error(ds)
        with ErrorLogger(logger):
            raise e
    days = median_delta / np.timedelta64(1, "D")
    hours = int(median_delta / np.timedelta64(1, "h"))
    if abs(days - 1) < 0.5:
        return hours, "D"
    if abs(days - 7) < 1:
        return hours, "W"
    if 27 < days < 32:
        return hours, "ME"
    # fallback: integer hours (lowercase for pandas >= 3.0)
    return hours, f"{hours}h"


def get_data_coverage(obj):
    """Return the first and last instant that an object's time steps cover.

    A time stamp labels a period, so a monthly stamp of 31 January covers all
    of January and a daily stamp covers its whole day. Comparing coverage
    instead of raw labels keeps a coarse dataset from looking like it starts
    late.

    Args:
        obj: Object with a ``time`` dimension.

    Returns
    -------
        Tuple of (first covered instant, first instant after the coverage).
    """
    time_index = pd.DatetimeIndex(obj.time.values)
    if time_index.size < 2:
        return time_index[0], time_index[-1]
    _, alias = timedelta_to_alias(obj)
    period_freq = get_period_freq(alias)
    if period_freq is None:
        step = pd.Timedelta(np.median(np.diff(time_index.values)))
        return time_index[0], time_index[-1] + step
    return (
        time_index[0].to_period(period_freq).start_time,
        time_index[-1].to_period(period_freq).end_time + pd.Timedelta(1, "ns"),
    )


# ------------------------ internals ------------------------


def _pick_da(obj: Union[xr.DataArray, xr.Dataset]) -> xr.DataArray:
    if isinstance(obj, xr.DataArray):
        return obj
    if not obj.data_vars:
        msg = "Dataset has no data variables."
        with ErrorLogger(logger):
            raise ValueError(msg)
    return obj[next(iter(obj.data_vars))]


def _ensure_time(obj, var=None):
    try_obj = obj[var] if var is not None else obj
    if "time" not in try_obj.dims and "time" not in try_obj.coords:
        msg = "Object needs a 'time' dimension."
        with ErrorLogger(logger):
            raise ValueError(msg)


def _is_intensive(var: xr.DataArray) -> bool:  # noqa: PLR0911
    """
    Heuristic: True = intensive, False = extensive.

    - If units include '/s', ' s-1', '/h', ' h-1' -> intensive (rate)
    - If CF names suggest totals (amount) -> extensive
    - If units look like pure totals per step (mm, kg m-2) -> extensive
    - cell_methods hint: 'time: mean' -> intensive, 'time: sum' -> extensive
    Fallback: intensive.
    """
    u = (var.attrs.get("units") or "").lower().strip()
    sn = (var.attrs.get("standard_name") or "").lower()
    cm = (var.attrs.get("cell_methods") or "").lower()

    if "time: sum" in cm:
        return False
    if "time: mean" in cm:
        return True

    if "amount" in sn or "accumulation" in sn or "thickness_of" in sn:
        return False
    if "precipitation_amount" in sn or "snowfall_amount" in sn:
        return False

    # rates (intensive)
    if any(t in u for t in ["/s", " s-1", "/h", " h-1", "/min", " min-1"]):
        return True
    if "flux" in sn:
        return True

    # totals per step (extensive): mm, kg m-2, m, j m-2 etc., but not per time
    looks_total = any(t in u for t in ["mm", "kg m-2", "kg/m2", "j m-2", "j/m2", "m"])
    has_per_time = any(t in u for t in ["/s", " s-1", "/h", " h-1", "/d", " d-1"])
    if looks_total and not has_per_time:
        logger.info(f"Unit {u} results in extensive resampling")
        return False

    # default: intensive
    logger.info(f"Unit {u} results in intensive resampling")
    return True


def _target_alias(which: Literal["daily", "hourly"]) -> str:
    return "D" if which == "daily" else "1h"


def _alias_and_hours(obj: Union[xr.DataArray, xr.Dataset]) -> tuple[int, str]:
    hours, alias = timedelta_to_alias(_pick_da(obj))
    return int(hours), alias


def _offset_for_alias(alias: str) -> pd.DateOffset:
    # Map our aliases to pandas offsets
    alias_upper = alias.upper()
    if alias_upper in ("D",):
        return pd.offsets.Day(1)
    if alias in ("W",):
        return pd.offsets.Week(1)
    if alias_upper in ("ME", "M"):
        return pd.offsets.MonthEnd(1)
    # e.g., "3H", "1H"
    if alias_upper.endswith("H"):
        return pd.offsets.Hour(int(alias[:-1]))
    msg = f"Unsupported alias '{alias}'"
    with ErrorLogger(logger):
        raise ValueError(msg)


def _per_step_duration_index(time: xr.DataArray, alias_in: str) -> pd.TimedeltaIndex:
    """
    Duration of each *source* step (right-open interval) as TimedeltaIndex.

    handling variable-length months when alias_in == 'ME'.
    """
    t = pd.DatetimeIndex(time.data)
    # duration to the next stamp
    dt = t[1:] - t[:-1]
    if len(t) == 0:
        return pd.to_timedelta([])
    # last step: extend by one calendar step
    dt_last = _offset_for_alias(alias_in)
    return dt.append(pd.TimedeltaIndex([pd.Timedelta(dt_last)]))


def _distribute_extensive_to_finer(
    da: xr.DataArray,  # totals per source step
    alias_in: str,
    alias_out: str,
) -> xr.DataArray:
    """
    Sum-preserving upsample for extensive variables.

    Evenly distributes each coarse-step total into its child finer bins.
    """
    # how many target bins per source step?
    dt_src = _per_step_duration_index(da["time"], alias_in)
    dt_out = _offset_for_alias(alias_out)
    # number of target bins within each source interval
    bins_per = (dt_src / pd.Timedelta(dt_out)).round().astype(int)
    # divide each total by its number of bins → per-target-bin value
    per_bin = da / xr.DataArray(
        bins_per.values, dims=["time"], coords={"time": da.time}
    )
    # now replicate into finer grid by resampling with ffill
    return per_bin.resample(time=alias_out).ffill()


# ---------------------- public function ----------------------


def resample_to_daily_or_hourly_adaptive(
    in_obj: Union[xr.DataArray, xr.Dataset],
    target: Literal["daily", "hourly"],
    upsample_for_intensive: Literal["linear", "ffill", "nearest"] = "linear",
    var: str | None = None,
) -> Union[xr.DataArray, xr.Dataset]:
    """
    Resample to daily or hourly with **adaptive** choice of aggregation.

      * Intensive vars → downsample = mean; upsample = interpolate/fill.
      * Extensive vars → downsample = sum;  upsample = sum-preserving distribution.

    Parameters
    ----------
    in_obj : xr.DataArray | xr.Dataset
    target : 'daily' | 'hourly'
    upsample_for_intensive : fill method for intensive vars when going finer.
    var : str | None

    Returns
    -------
    Same type as input, resampled to calendar-aware 'D' or '1h'.
    """
    in_obj = in_obj.copy()
    logger.info(f"Starting adaptive resampling to {target}")
    logger.info(f"Input object: {in_obj}")

    # If Dataset, keep only data_vars that have a time dimension/coord
    if isinstance(in_obj, xr.Dataset):
        if var:
            if in_obj[var].sizes.get("time", 0) < 2:
                logger.info(
                    f"Provided variable '{var}' has less than 2 time steps; cannot resample"
                )
                return in_obj  # nothing to resample
        else:
            vars_with_time = []
            for name, da in in_obj.data_vars.items():
                try:
                    _ensure_time(da)
                    if da.sizes.get("time", 0) >= 2:
                        vars_with_time.append(name)
                    else:
                        logger.info(
                            f"Variable '{name}' has less than 2 time steps; removing from object"
                        )
                except ValueError:
                    logger.info(
                        f"Variable '{name}' has no 'time' dimension; removing from object"
                    )
            if not vars_with_time:
                msg = "Dataset has no variables with a 'time' dimension."
                with ErrorLogger(logger):
                    raise ValueError(msg)
            if not vars_with_time:
                logger.info("No variables with sufficient time steps; cannot resample")
                return in_obj  # nothing to resample
            in_obj = in_obj[vars_with_time]
    _ensure_time(in_obj)
    alias_tgt = _target_alias(target)
    tgt_hours = 24 if target == "daily" else 1
    in_hours, alias_in = _alias_and_hours(in_obj)

    # If already at target cadence (1h or D), return
    if (target == "hourly" and in_hours == 1) or (
        target == "daily" and alias_in == "D"
    ):
        return in_obj

    logger.info(f"Adaptive regridding from {alias_in} to {target}")
    going_coarser = in_hours < tgt_hours  # e.g., 1H -> D
    going_finer = in_hours > tgt_hours  # e.g., D/ME/W/3H -> 1H or D

    def _resample_da(da: xr.DataArray) -> xr.DataArray:  # noqa: PLR0911
        intensive = _is_intensive(da)

        if going_coarser:
            if intensive:
                # average to the coarser calendar bins
                return da.resample(time=alias_tgt).mean()
            # sum totals into the coarser bins
            return da.resample(time=alias_tgt).sum()

        if going_finer:
            if intensive:
                if upsample_for_intensive == "linear":
                    return da.resample(time=alias_tgt).interpolate("linear")
                if upsample_for_intensive == "ffill":
                    return da.resample(time=alias_tgt).ffill()
                if upsample_for_intensive == "nearest":
                    return da.resample(time=alias_tgt).nearest()
                msg = f"Unknown upsample_for_intensive='{upsample_for_intensive}'"
                with ErrorLogger(logger):
                    raise ValueError(msg)
            # extensive → distribute evenly across finer bins (sum-preserving)
            return _distribute_extensive_to_finer(
                da, alias_in=alias_in, alias_out=alias_tgt
            )

        # Same nominal hours but different calendars (e.g., 24H -> D or D -> 1h)
        if target == "daily":
            return (
                da.resample(time="D").mean()
                if intensive
                else da.resample(time="D").sum()
            )
        if intensive:
            if upsample_for_intensive == "linear":
                return da.resample(time="1h").interpolate("linear")
            if upsample_for_intensive == "ffill":
                return da.resample(time="1h").ffill()
            if upsample_for_intensive == "nearest":
                return da.resample(time="1h").nearest()
            msg = f"Unknown upsample_for_intensive='{upsample_for_intensive}'"
            with ErrorLogger(logger):
                raise ValueError(msg)
        return _distribute_extensive_to_finer(da, alias_in=alias_in, alias_out="1h")

    if isinstance(in_obj, xr.DataArray):
        out = _resample_da(in_obj)
    else:
        # Dataset: apply variable-wise, preserving coords/attrs
        out_vars = {}
        for name, da in in_obj.data_vars.items():
            out_vars[name] = _resample_da(da)
        out = xr.Dataset(out_vars)
        # carry coordinates (besides resampled time) from original dataset
        for cname, coord in in_obj.coords.items():
            if cname == "time":
                out = out.assign_coords(time=out[next(iter(out_vars))].time)
            elif cname not in out.coords:
                out = out.assign_coords({cname: coord})
        out.attrs = in_obj.attrs
    in_hours, alias_in = _alias_and_hours(in_obj)
    logger.info(f"New resolution {alias_in} meaning {in_hours} hours per timestep")
    logger.debug(f"Resampled object: {out}")
    return out


def drop_partial_edge_steps(
    ds_input, ds_ref, target_freq, coverage_start, coverage_end
):
    """Drop leading or trailing steps that only one dataset fully covers.

    When the overlap starts or ends inside a target period, that period is
    averaged from fewer source steps for one of the datasets, which biases the
    comparison. Only the two edge steps can be affected, so only those are
    checked.

    Parameters
    ----------
    ds_input, ds_ref : xr.Dataset
        Resampled datasets sharing one time axis.
    target_freq : str
        Pandas frequency alias both were resampled to.
    coverage_start, coverage_end
        First instant both datasets cover and first instant after it.

    Returns
    -------
    tuple
        Both datasets without partially covered edge steps.
    """
    step_days = get_step_days(target_freq)
    period_freq = get_period_freq(target_freq)
    if (step_days is None and period_freq is None) or ds_input.sizes["time"] == 0:
        return ds_input, ds_ref
    time_index = pd.DatetimeIndex(ds_input.time.values)
    if step_days is not None:
        step_starts = time_index
        step_ends = time_index + pd.Timedelta(days=step_days)
    else:
        periods = time_index.to_period(period_freq)
        step_starts = periods.start_time
        step_ends = periods.end_time + pd.Timedelta(1, "ns")
    fully_covered = (step_starts >= coverage_start) & (step_ends <= coverage_end)
    keep = np.ones(len(time_index), dtype=bool)
    if not fully_covered[0]:
        keep[0] = False
    if len(time_index) > 1 and not fully_covered[-1]:
        keep[-1] = False
    if keep.all():
        return ds_input, ds_ref
    dropped = [str(stamp.date()) for stamp in time_index[~keep]]
    logger.warning(
        f"Dropping the {target_freq} step(s) {', '.join(dropped)} because the "
        "common period only covers them partly, which would compare a full "
        "period against a shorter one."
    )
    indices = np.flatnonzero(keep)
    return ds_input.isel(time=indices), ds_ref.isel(time=indices)


def normalize_time_axis(
    ds: xr.Dataset, alias: str, anchor: str = "default"
) -> xr.Dataset:
    """Normalize time stamps to a consistent anchor for the given frequency alias.

    Parameters
    ----------
    ds : xr.Dataset
        Object with a ``time`` coordinate.
    alias : str
        Pandas frequency alias the time stamps belong to.
    anchor : str, optional
        ``"default"`` keeps the historic anchors (month end for ``ME``, the
        ``W-MON`` stamp for weekly). ``"period_start"`` moves every stamp to the
        first instant of the period it covers, so the label describes the steps
        it was aggregated from. Resampling labels a period by its right edge, so
        only ``"period_start"`` lets a caller derive the covered days from the
        label without a systematic offset.

    Returns
    -------
    xr.Dataset
        The object with anchored time stamps.
    """
    if anchor == "period_start":
        if "time" not in ds.coords:
            return ds
        if get_step_days(alias) is not None:
            # a fixed day multiple is already labelled by its first day
            return ds.assign_coords(time=ds.time.dt.floor("D"))
        period_freq = get_period_freq(alias)
        if period_freq is None:
            return ds.assign_coords(time=ds.time.dt.floor("h"))
        return ds.assign_coords(
            time=ds.indexes["time"].to_period(period_freq).start_time
        )

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


def get_origin_anchored_freq(target_freq):
    """Translate a day multiple into the hour alias that ``origin`` still anchors.

    ``origin`` only takes effect for a Tick-like frequency, and pandas 3.0
    stopped counting ``Day`` as one, so ``"8D"`` silently ignores the origin
    while the exactly equivalent ``"192h"`` still honors it. A time axis here is
    naive UTC, where a day is a fixed 24 hours, so the translation does not
    change which stamps fall in a bin.

    Args:
        target_freq: Pandas frequency alias.

    Returns
    -------
        The equivalent hour alias, or ``target_freq`` unchanged when it does not
        cover a fixed number of days.
    """
    step_days = get_step_days(target_freq)
    return target_freq if step_days is None else f"{step_days * 24}h"


def get_time_bounds(obj):
    """Return the CF time bounds of a record as a (n, 2) array of timestamps.

    A time stamp only labels a period, and the bounds are the period itself.
    They are the only place a record states the window each of its values was
    observed or averaged over. A CF bounds variable spans a second dimension,
    which a DataArray of one variable cannot carry, so the two edges are also
    accepted as the pair of 1-D coordinates that `set_time_bounds` attaches.

    Args:
        obj: Dataset or DataArray with a ``time`` coordinate.

    Returns
    -------
        The (start, end) of every step, or None when the record has no bounds.
    """
    if "time" not in getattr(obj, "coords", {}):
        return None
    coords = obj.coords
    if TIME_BOUNDS_START_COORD in coords and TIME_BOUNDS_END_COORD in coords:
        return np.stack(
            [
                np.asarray(coords[TIME_BOUNDS_START_COORD].values),
                np.asarray(coords[TIME_BOUNDS_END_COORD].values),
            ],
            axis=1,
        )
    time_coord = obj["time"]
    bounds_name = time_coord.attrs.get("bounds") or time_coord.encoding.get("bounds")
    if not bounds_name:
        return None
    holder = obj if bounds_name in getattr(obj, "coords", {}) else None
    if holder is None and bounds_name in getattr(obj, "variables", {}):
        holder = obj
    if holder is None:
        logger.warning(
            f"The time coordinate names the bounds '{bounds_name}', but the "
            f"record does not carry that variable."
        )
        return None
    bounds = np.asarray(holder[bounds_name].values)
    if bounds.ndim != 2 or bounds.shape[0] != time_coord.size or bounds.shape[1] != 2:
        logger.warning(
            f"The time bounds '{bounds_name}' have shape {bounds.shape}, which "
            f"does not describe two edges per time step."
        )
        return None
    return bounds


def set_time_bounds(da, bounds):
    """Attach time bounds to a DataArray as a pair of 1-D coordinates.

    Selecting one variable out of a dataset drops its CF bounds variable,
    because that variable spans a second dimension the DataArray does not have.
    Two 1-D coordinates carry the same two edges along the time axis.

    Args:
        da: DataArray with a ``time`` dimension.
        bounds: (n, 2) array of the start and end of every step, or None.

    Returns
    -------
        The DataArray with the bounds attached, unchanged when none were given.
    """
    if bounds is None:
        return da
    return da.assign_coords(
        {
            TIME_BOUNDS_START_COORD: ("time", np.asarray(bounds)[:, 0]),
            TIME_BOUNDS_END_COORD: ("time", np.asarray(bounds)[:, 1]),
        }
    )


def infer_time_bounds(obj, tolerance=IRREGULAR_SPACING_TOLERANCE, label="reference"):
    """Infer the window of every step from evenly spaced stamps.

    The edges sit halfway between two stamps, which takes a stamp as the centre
    of its window. A gap would stretch the windows around it over instants the
    record never observed, so an unevenly spaced record has to state its own
    bounds instead of having them guessed.

    Args:
        obj: Dataset or DataArray with a ``time`` coordinate.
        tolerance: How much the largest step may exceed the median one.
        label: Name used in the error message.

    Returns
    -------
        The (start, end) of every step as a (n, 2) array.
    """
    time_index = pd.DatetimeIndex(obj["time"].values)
    if time_index.size < 2:
        msg = (
            f"The {label} record holds a single time step, so the window it "
            f"covers cannot be derived from the spacing of its stamps. Give its "
            f"time coordinate a CF 'bounds' attribute naming a (time, 2) "
            f"variable with the first and last instant of the step."
        )
        with ErrorLogger(logger):
            raise ValueError(msg)
    spacing = np.diff(time_index.values) / np.timedelta64(1, "D")
    median_spacing = float(np.median(spacing))
    largest_spacing = float(spacing.max())
    if largest_spacing > tolerance * median_spacing:
        gap_at = time_index[int(np.argmax(spacing))]
        msg = (
            f"The {label} record steps {median_spacing:.0f} days at a time but "
            f"jumps {largest_spacing:.0f} days after {gap_at.date()}, so its "
            f"steps do not state what they cover. Deriving the windows from the "
            f"stamps would stretch the ones around that jump over instants the "
            f"record never observed. Give its time coordinate a CF 'bounds' "
            f"attribute naming a (time, 2) variable with the first and last "
            f"instant of every step."
        )
        with ErrorLogger(logger):
            raise ValueError(msg)
    logger.info(
        f"Deriving the {label} windows from its {median_spacing:.0f} day spacing."
    )
    # the lower edges plus the last upper one are contiguous by construction,
    # while the upper column of generate_bounds is not for uneven spacing
    coord = xr.DataArray(
        time_index.values, dims=("time",), coords={"time": time_index.values}
    )
    raw_bounds = generate_bounds(coord).values
    edges = np.append(raw_bounds[:, 0], raw_bounds[-1, 1])
    return np.stack([edges[:-1], edges[1:]], axis=1)


def resample_to_reference_windows(input_ds, ref_ds, label="input"):
    """Average an input over each time window of a reference record.

    The reference keeps every one of its steps and its own time axis, and the
    input is averaged over exactly the window each reference value was observed
    over. A window the input does not cover completely is dropped from both,
    because a full period must not be compared against a shorter one.

    A reference that states its bounds is taken at its word. One that does not
    has its windows derived from its stamps, which only works while they are
    evenly spaced.

    Args:
        input_ds: Dataset or DataArray to average, with a ``time`` dimension.
        ref_ds: Reference whose time windows define the shared calendar.
        label: Name used in the log and error messages.

    Returns
    -------
        Tuple of the averaged input and the reference on the shared windows.
    """
    bounds = get_time_bounds(ref_ds)
    if bounds is None:
        bounds = infer_time_bounds(ref_ds)
    window_days = float(
        np.median((bounds[:, 1] - bounds[:, 0]) / np.timedelta64(1, "D"))
    )
    input_step_days, _ = timedelta_to_alias(input_ds)
    input_step_days = input_step_days / 24
    if input_step_days > window_days * IRREGULAR_SPACING_TOLERANCE:
        msg = (
            f"The {label} steps {input_step_days:.0f} days at a time, which is "
            f"coarser than the {window_days:.0f} day windows of the reference. "
            f"Averaging it over them would repeat one value across several "
            f"windows instead of comparing what each window covers."
        )
        with ErrorLogger(logger):
            raise ValueError(msg)
    input_start, input_end = get_data_coverage(input_ds)
    kept_steps, averages = [], []
    for step, (window_start, window_end) in enumerate(bounds):
        if pd.Timestamp(window_start) < input_start or (
            pd.Timestamp(window_end) > input_end
        ):
            continue
        window = input_ds.sel(time=slice(window_start, window_end))
        if window.sizes.get("time", 0) == 0:
            continue
        averages.append(window.mean("time", keep_attrs=True))
        kept_steps.append(step)
    if not kept_steps:
        msg = (
            f"No reference window is covered completely by the {label} record, "
            f"which spans {input_start.date()} to {input_end.date()}."
        )
        with ErrorLogger(logger):
            raise ValueError(msg)
    dropped = ref_ds.sizes["time"] - len(kept_steps)
    if dropped:
        logger.info(
            f"Dropping {dropped} reference window(s) the {label} record does not "
            f"cover completely."
        )
    ref_out = ref_ds.isel(time=kept_steps)
    input_out = xr.concat(averages, dim=ref_out["time"])
    logger.info(f"Averaged the {label} over {len(kept_steps)} reference windows.")
    return input_out, ref_out


def resample_to_target_freq(
    ds_input: xr.Dataset,
    ds_ref: xr.Dataset,
    target_freq,
    resample_origin=None,
    time_anchor: str = "default",
    drop_partial_edges: bool = False,
) -> Tuple[xr.Dataset, xr.Dataset]:
    """Resample both datasets to the provided target freq.

    Parameters
    ----------
    ds_input, ds_ref : xr.Dataset
        Datasets with a ``time`` dimension.
    target_freq : str
        Pandas frequency alias to resample to.
    resample_origin : optional
        Origin handed to ``resample``, e.g. ``"epoch"``. A multi-day step is
        binned from an origin instead of from the calendar, so without a shared
        origin two datasets whose records start on different days never produce
        the same bins. Setting it also forces resampling of a dataset that
        already reports the target frequency, because only resampling
        normalizes its bin phase.
    time_anchor : str, optional
        Anchor handed to :func:`normalize_time_axis`.
    drop_partial_edges : bool, optional
        Drop a first or last step that the two datasets do not both cover
        completely.

    Returns
    -------
    tuple
        Both resampled and time-aligned datasets.
    """
    _hours_in, alias_in = timedelta_to_alias(ds_input)
    _hours_ref, alias_ref = timedelta_to_alias(ds_ref)
    # record the real coverage before resampling blurs the edges
    coverage = None
    if drop_partial_edges:
        input_coverage = get_data_coverage(ds_input)
        ref_coverage = get_data_coverage(ds_ref)
        coverage = (
            max(input_coverage[0], ref_coverage[0]),
            min(input_coverage[1], ref_coverage[1]),
        )
    resample_kwargs = {} if resample_origin is None else {"origin": resample_origin}
    force_resample = resample_origin is not None
    # an origin only anchors a Tick-like frequency, see get_origin_anchored_freq
    resample_freq = (
        target_freq
        if resample_origin is None
        else get_origin_anchored_freq(target_freq)
    )

    if force_resample or target_freq != alias_ref:
        # input is coarser (e.g. monthly) → bring ref up to that
        logger.info(f"Resampling ref from {alias_ref} to {target_freq}")
        ds_ref = ds_ref.resample(time=resample_freq, **resample_kwargs).mean()
    if force_resample or target_freq != alias_in:
        # ref is coarser → bring input up to that
        logger.info(f"Resampling input from {alias_in} to {target_freq}")
        ds_input = ds_input.resample(time=resample_freq, **resample_kwargs).mean()

    # Normalize anchors so both datasets share identical timestamps.
    ds_input = normalize_time_axis(ds_input, target_freq, anchor=time_anchor)
    ds_ref = normalize_time_axis(ds_ref, target_freq, anchor=time_anchor)

    # Align the two datasets along the time dimension ensuring that they match exactly, while ignoring any other dimensions
    non_time_dims = (set(ds_input.dims) | set(ds_ref.dims)) - {"time"}
    ds_input, ds_ref = xr.align(ds_input, ds_ref, join="inner", exclude=non_time_dims)
    if coverage is not None:
        ds_input, ds_ref = drop_partial_edge_steps(
            ds_input, ds_ref, target_freq, coverage[0], coverage[1]
        )
    # logger.debug(f"Input file after align {ds_input}")
    return ds_input, ds_ref
