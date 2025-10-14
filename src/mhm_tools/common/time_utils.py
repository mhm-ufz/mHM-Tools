import logging
from mhm_tools.common.xarray_utils import timedelta_to_alias
import numpy as np
import pandas as pd
import xarray as xr
from typing import Literal, Union

logger = logging.getLogger(__name__)

# --- your helpers, used AS-IS somewhere in your codebase ---
# timedelta_to_alias(ds: xr.DataArray) -> Tuple[int, str]
# resample_to_coarser_calendar(...)

# ------------------------ internals ------------------------

def _pick_da(obj: Union[xr.DataArray, xr.Dataset]) -> xr.DataArray:
    if isinstance(obj, xr.DataArray):
        return obj
    if not obj.data_vars:
        raise ValueError("Dataset has no data variables.")
    return obj[list(obj.data_vars)[0]]

def _ensure_time(obj):
    if "time" not in obj.dims and "time" not in obj.coords:
        raise ValueError("Object needs a 'time' dimension.")

def _is_intensive(var: xr.DataArray) -> bool:
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
        logger.info(f'Unit {u} results in extensive resampling')
        return False

    # default: intensive
    logger.info(f'Unit {u} results in intensive resampling')
    return True

def _target_alias(which: Literal["daily", "hourly"]) -> str:
    return "D" if which == "daily" else "1H"

def _alias_and_hours(obj: Union[xr.DataArray, xr.Dataset]) -> tuple[int, str]:
    hours, alias = timedelta_to_alias(_pick_da(obj))
    return int(hours), alias

def _offset_for_alias(alias: str) -> pd.DateOffset:
    # Map our aliases to pandas offsets
    if alias in ("D",):
        return pd.offsets.Day(1)
    if alias in ("W",):
        return pd.offsets.Week(1)
    if alias in ("ME", "M"):
        return pd.offsets.MonthEnd(1)
    # e.g., "3H", "1H"
    if alias.endswith("H"):
        return pd.offsets.Hour(int(alias[:-1]))
    raise ValueError(f"Unsupported alias '{alias}'")

def _per_step_duration_index(time: xr.DataArray, alias_in: str) -> pd.TimedeltaIndex:
    """
    Duration of each *source* step (right-open interval) as TimedeltaIndex,
    handling variable-length months when alias_in == 'ME'.
    """
    t = pd.DatetimeIndex(time.values)
    # duration to the next stamp
    dt = (t[1:] - t[:-1])
    if len(t) == 0:
        return pd.to_timedelta([])
    # last step: extend by one calendar step
    dt_last = _offset_for_alias(alias_in)
    dt = dt.append(pd.TimedeltaIndex([pd.Timedelta(dt_last)]))
    return dt

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
    per_bin = da / xr.DataArray(bins_per.values, dims=["time"], coords={"time": da.time})
    # now replicate into finer grid by resampling with ffill
    out = per_bin.resample(time=alias_out).ffill()
    return out

# ---------------------- public function ----------------------

def resample_to_daily_or_hourly_adaptive(
    obj: Union[xr.DataArray, xr.Dataset],
    target: Literal["daily", "hourly"],
    upsample_for_intensive: Literal["linear", "ffill", "nearest"] = "linear",
) -> Union[xr.DataArray, xr.Dataset]:
    """
    Resample to daily or hourly with **adaptive** choice of aggregation:
      * Intensive vars → downsample = mean; upsample = interpolate/fill.
      * Extensive vars → downsample = sum;  upsample = sum-preserving distribution.

    Parameters
    ----------
    obj : xr.DataArray | xr.Dataset
    target : 'daily' | 'hourly'
    upsample_for_intensive : fill method for intensive vars when going finer.

    Returns
    -------
    Same type as input, resampled to calendar-aware 'D' or '1H'.
    """
    
    _ensure_time(obj)

    alias_tgt = _target_alias(target)
    tgt_hours = 24 if target == "daily" else 1
    in_hours, alias_in = _alias_and_hours(obj)
    
    # If already at target cadence (1H or D), return
    if (target == "hourly" and in_hours == 1) or (target == "daily" and alias_in == "D"):
        return obj

    logger.info(f'Adaptive regridding from {alias_in} to {target}')
    going_coarser = in_hours < tgt_hours   # e.g., 1H -> D
    going_finer   = in_hours > tgt_hours   # e.g., D/ME/W/3H -> 1H or D

    def _resample_da(da: xr.DataArray) -> xr.DataArray:
        intensive = _is_intensive(da)
        
        if going_coarser:
            if intensive:
                # average to the coarser calendar bins
                return da.resample(time=alias_tgt).mean()
            else:
                # sum totals into the coarser bins
                return da.resample(time=alias_tgt).sum()

        elif going_finer:
            if intensive:
                if upsample_for_intensive == "linear":
                    return da.resample(time=alias_tgt).interpolate("linear")
                elif upsample_for_intensive == "ffill":
                    return da.resample(time=alias_tgt).ffill()
                elif upsample_for_intensive == "nearest":
                    return da.resample(time=alias_tgt).nearest()
                else:
                    raise ValueError(f"Unknown upsample_for_intensive='{upsample_for_intensive}'")
            else:
                # extensive → distribute evenly across finer bins (sum-preserving)
                return _distribute_extensive_to_finer(da, alias_in=alias_in, alias_out=alias_tgt)

        else:
            # Same nominal hours but different calendars (e.g., 24H -> D or D -> 1H)
            if target == "daily":
                return da.resample(time="D").mean() if intensive else da.resample(time="D").sum()
            else:  # hourly target
                if intensive:
                    if upsample_for_intensive == "linear":
                        return da.resample(time="1H").interpolate("linear")
                    elif upsample_for_intensive == "ffill":
                        return da.resample(time="1H").ffill()
                    elif upsample_for_intensive == "nearest":
                        return da.resample(time="1H").nearest()
                else:
                    return _distribute_extensive_to_finer(da, alias_in=alias_in, alias_out="1H")

    if isinstance(obj, xr.DataArray):
        out = _resample_da(obj)
    else: 
        # Dataset: apply variable-wise, preserving coords/attrs
        out_vars = {}
        for name, da in obj.data_vars.items():
            out_vars[name] = _resample_da(da)
        out = xr.Dataset(out_vars)
        # carry coordinates (besides resampled time) from original dataset
        for cname, coord in obj.coords.items():
            if cname == "time":
                out = out.assign_coords(time=out[list(out_vars)[0]].time)
            elif cname not in out.coords:
                out = out.assign_coords({cname: coord})
        out.attrs = obj.attrs
    in_hours, alias_in = _alias_and_hours(obj)
    logger.info(f'New resolution {alias_in} meaning {in_hours} hours per timestep')
    return out