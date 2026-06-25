"""Provides basic xarray utils."""

import logging
from typing import Optional, Union

import numpy as np
import pandas as pd
import xarray as xr
from scipy.stats import spearmanr

from mhm_tools.common.constants import LAT_KEYS, LON_KEYS, TIME_KEYS
from mhm_tools.common.logger import ErrorLogger
from mhm_tools.common.netcdf import generate_bounds_for_all_coords

logger = logging.getLogger(__name__)


def normalize_lat_lon(
    ds: Union[xr.Dataset, xr.DataArray],
    lat_key: Optional[str] = None,
    lon_key: Optional[str] = None,
    new_lat_key: str = "lat",
    new_lon_key: str = "lon",
    raise_exceptions: bool = False,
    log_warning: bool = False,
) -> Union[xr.Dataset, xr.DataArray]:
    """
    Normalize latitude and longitude dimension and coordinate names to 'lat' and 'lon'.

    Handles both dimensions and coordinate variables.
    """
    try:
        rename_dict = {}
        if lat_key is None:
            lat_key = get_coord_key(ds, lat=True)
        if lon_key is None:
            lon_key = get_coord_key(ds, lon=True)

        coords_and_dims = list(ds.coords) + list(ds.dims)
        if log_warning and (lat_key != new_lat_key or lon_key != new_lon_key):
            logger.warning(
                f"The coordinates were normalised from {lon_key}->{new_lon_key} and {lat_key}->{new_lat_key}"
            )
        # Rename coordinate variables if needed
        if (
            lat_key is not None
            and new_lat_key not in coords_and_dims
            and lat_key in coords_and_dims
        ):
            rename_dict[lat_key] = new_lat_key
        if (
            lon_key is not None
            and new_lon_key not in coords_and_dims
            and lon_key in coords_and_dims
        ):
            rename_dict[lon_key] = new_lon_key

        return ds.rename(rename_dict)
    except Exception as e:
        if raise_exceptions:
            with ErrorLogger(logger):
                raise (e)
        else:
            logger.warning(f"Exception in normalize lat lon: {e}")
            return ds


def snap_to_target(
    ds: Union[xr.Dataset, xr.DataArray],
    lat_key: str,
    lon_key: str,
    target_lat_array,
    target_lon_array,
    new_lat_key: str = "lat",
    new_lon_key: str = "lon",
) -> Union[xr.Dataset, xr.DataArray]:
    """
    Rename latitude/longitude dimensions and assign exact target coordinates.

    This is useful after nearest-neighbor grid matching, where selected values
    are correct by position but coordinate labels still need to match exactly
    for xarray alignment.
    """
    ds = normalize_lat_lon(
        ds,
        lat_key=lat_key,
        lon_key=lon_key,
        new_lat_key=new_lat_key,
        new_lon_key=new_lon_key,
        raise_exceptions=True,
    )
    return ds.assign_coords(
        {
            new_lat_key: np.asarray(target_lat_array),
            new_lon_key: np.asarray(target_lon_array),
        }
    )


def get_coord_key(
    ds, lat=False, lon=False, time=False, raise_exception=True, is_retry=False
):
    """Return the lat or lon coordinate name used in the xarray dataset."""
    if lat + lon + time != 1:
        with ErrorLogger(logger):
            msg = f"one of lon, lat or time should be true but lon={lon} and lat={lat} and time={time}"
            raise ValueError(msg)
    ds_dims = ds.dims if isinstance(ds, xr.DataArray) or is_retry else ds.coords
    # first check if there are dimensions with a fitting axis attribute
    try:
        for dim in ds_dims:
            if (
                (lat and ds[dim].axis == "Y")
                or (lon and ds[dim].axis == "X")
                or (time and ds[dim].axis == "T")
            ):
                return dim
    except AttributeError:
        pass
    # then select possible keys from the following lists and try them until a fitting one is found.
    if lat:
        keys = LAT_KEYS
    elif lon:
        keys = LON_KEYS
    else:
        keys = TIME_KEYS
    for key in keys:
        if key in ds_dims and len(ds[key].shape) == 1:
            return key
    for key in keys:
        if key in ds_dims:
            logger.warning(
                f"{type(ds)} contains key: {key} but ds[key] has shape {ds[key].shape}."
            )
            return key
    if not is_retry and isinstance(ds, xr.Dataset):
        logger.warning(
            f"{type(ds)} does not contain fitting coordinates. Trying again looking for dimensions"
        )
        return get_coord_key(
            ds,
            lat=lat,
            lon=lon,
            time=time,
            raise_exception=raise_exception,
            is_retry=True,
        )
    if raise_exception:
        with ErrorLogger(logger):
            msg = f"None of {keys} in {type(ds).__name__} keys {ds_dims}."
            raise ValueError(msg)
    return None


def get_single_data_var(ds: xr.Dataset, proposed_vars: Optional[list] = None):
    """Get the data var name from a dataset that only contains one data variable."""
    data_vars = list(ds.data_vars)  # shallow copy is enough; entries are strings
    if isinstance(proposed_vars, list):
        for var in data_vars:
            if var in proposed_vars:
                return var
    if len(data_vars) > 1:
        # remove coords without mutating while iterating
        coords = [
            coord for coord in LAT_KEYS + LON_KEYS + TIME_KEYS if coord in data_vars
        ]
        for coord in coords:
            logger.debug(f"Removing coordinate data_var {coord} from consideration.")
        data_vars = [dv for dv in data_vars if dv not in coords]
        logger.debug(f"data_vars after removing coords: {data_vars}")

        # remove bounds variables; iterate over snapshot to avoid skipping
        bounds_removed = []
        filtered = []
        for data_var in data_vars.copy():
            if "bounds" in ds[data_var].attrs or data_var.endswith("_bnds"):
                bounds_removed.append(data_var)
            else:
                filtered.append(data_var)
                logger.debug(f"Keeping data_var {data_var}.")
        for bnd in bounds_removed:
            logger.debug(f"Removing bounds data_var {bnd} from consideration.")
        data_vars = filtered
        logger.debug(f"data_vars after removing bounds: {data_vars}")

        if len(data_vars) > 1:
            logger.error(f"Only single data_var allowed but has {data_vars}")
            return None
        if len(data_vars) == 0:
            logger.error("No datavar that is not coordinate.")
            return None
    logger.debug(f"data_vars: {data_vars}")
    if isinstance(data_vars, list) and len(data_vars) == 1:
        return data_vars[0]
    return None


def induce_data_var_from_file_name(ds, file_path):
    """Check if one of the data_vars is part of the file name and select it as the most probable data_var."""
    logger.info("Searching for more than one datavar by comparing with file name.")
    name = file_path.stem
    data_vars = list(ds.data_vars)
    logger.debug(f"{name} - {data_vars}")
    for dv in data_vars:
        if dv in name:
            return dv
        if name in dv:
            return dv
    return None


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


def get_overlapping_time_slice(input_ds, ref_ds):
    """Return the inclusive overlapping time window of two time-indexed objects.

    The overlap is computed from the first/last non-all-NaN timesteps of each
    input. Time-of-day components are dropped before creating the slice so
    small sub-daily offsets (e.g. 00:00 vs 11:00 for daily data) still produce
    a common calendar-day window.

    Args:
        input_ds: Simulation (or first) xarray object with a ``time`` dimension.
        ref_ds: Reference (or second) xarray object with a ``time`` dimension.

    Returns
    -------
        ``slice(start, end)`` with ``pandas.Timestamp`` endpoints.
    """
    input_non_nan_time = input_ds.dropna(dim="time", how="all").time.data
    reference_non_nan_time = ref_ds.dropna(dim="time", how="all").time.data

    logger.debug(f"input {input_non_nan_time[0]} till {input_non_nan_time[-1]}")
    logger.debug(f"ref {reference_non_nan_time[0]} till {reference_non_nan_time[-1]}")

    # Infer temporal buckets. We compare on calendar buckets (day/month/week)
    # instead of exact timestamps so small offsets like 00:00 vs 11:00 do not
    # break overlap detection for daily/monthly data.
    _, input_alias = timedelta_to_alias(input_ds)
    _, reference_alias = timedelta_to_alias(ref_ds)

    def _alias_to_bucket(alias):
        alias = str(alias).upper()
        if alias in {"ME", "MS", "M"}:
            return "M"
        if alias.startswith("W"):
            return "W"
        if alias == "D":
            return "D"
        return None

    input_bucket = _alias_to_bucket(input_alias)
    reference_bucket = _alias_to_bucket(reference_alias)
    bucket = (
        "M"
        if "M" in (input_bucket, reference_bucket)
        else (
            "W"
            if "W" in (input_bucket, reference_bucket)
            else "D" if "D" in (input_bucket, reference_bucket) else None
        )
    )

    # Find overlapping range between both available periods.
    only_nan_msg = "No non nan value data."
    if input_non_nan_time.any() and reference_non_nan_time.any():
        if bucket is None:
            # Sub-daily case: use exact timestamps.
            overlap_start = pd.to_datetime(
                max(input_non_nan_time[0], reference_non_nan_time[0])
            )
            overlap_end = pd.to_datetime(
                min(input_non_nan_time[-1], reference_non_nan_time[-1])
            )
        else:
            # Bucketed comparison (D/W/M): compare period overlap and expand to
            # full bucket boundaries for robust `.sel(time=slice(...))`.
            input_period = pd.DatetimeIndex(input_non_nan_time).to_period(bucket)
            reference_period = pd.DatetimeIndex(reference_non_nan_time).to_period(
                bucket
            )
            overlap_start_period = max(input_period.min(), reference_period.min())
            overlap_end_period = min(input_period.max(), reference_period.max())
            overlap_start = overlap_start_period.start_time
            overlap_end = overlap_end_period.end_time

        if overlap_end <= overlap_start:
            logger.warning(
                "The two datasets are not overlapping. "
                f"Sim data has non nan data from {input_non_nan_time[0]} to {input_non_nan_time[-1]} "
                f"and obs from {reference_non_nan_time[0]} to {reference_non_nan_time[-1]}."
            )
        logger.info(
            f"Cropping data to timeframe {overlap_start} to {overlap_end}"
            + (f" using {bucket} buckets." if bucket is not None else ".")
        )
    else:
        with ErrorLogger(logger):
            raise ValueError(only_nan_msg)
    return slice(overlap_start, overlap_end)


def crop_ds(
    ds: xr.Dataset,
    lon_min: float,
    lon_max: float,
    lat_min: float,
    lat_max: float,
    lon_name: str = "lon",
    lat_name: str = "lat",
) -> xr.Dataset:
    """Crop an xarray.Dataset to the given lon/lat bounds, handling coordinate order."""
    # ensure min < max
    lon_low, lon_high = sorted([lon_min, lon_max])
    lat_low, lat_high = sorted([lat_min, lat_max])

    # grab the coordinate arrays by name
    lon_vals = ds[lon_name].data
    lat_vals = ds[lat_name].data

    # if the coordinate axis is ascending, slice low->high; else high->low
    if lon_vals[0] <= lon_vals[-1]:
        lon_slice = slice(lon_low, lon_high)
    else:
        lon_slice = slice(lon_high, lon_low)

    if lat_vals[0] <= lat_vals[-1]:
        lat_slice = slice(lat_low, lat_high)
    else:
        lat_slice = slice(lat_high, lat_low)

    # select using a dict so named dims are respected
    return ds.sel({lon_name: lon_slice, lat_name: lat_slice})


def climatology(data):
    """Calculate the climatology from an xarray DataArray."""
    if "time" not in data.dims or data.sizes["time"] == 0:
        msg = "Input data for climatology calculation has no valid time dimension."
        with ErrorLogger(logger):
            raise ValueError(msg)
    # group into monthly mean data
    data_clim = data.groupby("time.month").mean(dim="time", skipna=True)
    # Ensure the climatology has all 12 months, filling missing months with NaNs
    return data_clim.reindex(month=np.arange(1, 13), fill_value=np.nan)


def get_clim_from_ds(ds, input_var=None, factor=1):
    """Calculate climatology from a Dataset or DataArray.

    Multiplies the selected data by `factor` before computing the monthly
    climatology.
    """
    data = ds * factor if input_var is None else ds[input_var] * factor
    return climatology(data)


def spearman_correlation(data1, data2):
    """Calculate Spearman rank correlation between two xarray DataArrays."""
    # Check that both arrays are of the same size and flatten them
    if data1.shape != data2.shape:
        with ErrorLogger(logger):
            msg = "Both DataArrays must have the same shape"
            raise ValueError(msg)
    # Accept either xarray objects or plain numpy arrays.
    data1 = np.asarray(getattr(data1, "values", data1)).flatten()
    data2 = np.asarray(getattr(data2, "values", data2)).flatten()
    valid = np.isfinite(data1) & np.isfinite(data2)
    data1 = data1[valid]
    data2 = data2[valid]
    if data1.size < 2:
        return np.nan, np.nan
    # Calculate Spearman rank correlation using scipy
    corr, p_value = spearmanr(data1, data2)
    return corr, p_value


def get_dtype(ds):
    """Return a simple dtype string without forcing data into memory."""
    try:
        if isinstance(ds, xr.Dataset):
            v = get_single_data_var(ds)
            if v is None:
                try:
                    dt = ds.dtype  # check if dataset has single dtype
                    da = ds  # use dataset directly
                except AttributeError as e:
                    msg = "Dataset has no single data variable to inspect."
                    raise ValueError(msg) from e
            else:
                da = ds[v]
        elif isinstance(ds, xr.DataArray):
            da = ds
        else:
            msg = f"Unsupported type {type(ds)}"
            raise ValueError(msg)

        dt = da.dtype  # cheap: does not load data
        if dt is None:
            return "f4"

        if np.issubdtype(dt, np.floating):
            return "f4" if dt.itemsize <= 4 else "f8"
        if np.issubdtype(dt, np.integer) or np.issubdtype(dt, np.unsignedinteger):
            # prefer signed integers for ASCII grid
            return "i4" if dt.itemsize <= 4 else "i8"

        msg = f"write_grid: cannot infer dtype from data with numpy dtype {dt}"
        with ErrorLogger(logger):
            raise ValueError(msg)
    except Exception:
        return "f4"


def _snap_coord_bound(value, resolution=None):
    """Snap coordinate bounds that only differ from the grid by roundoff."""
    value = float(value)
    if not np.isfinite(value):
        return value
    scale = max(abs(value), 1.0)
    tolerance = np.finfo(float).eps * scale * 4096
    if resolution is not None:
        resolution = abs(float(resolution))
        if np.isfinite(resolution) and resolution > 0:
            snapped = round(value / resolution) * resolution
            if abs(value - snapped) <= tolerance:
                value = float(snapped)
    nearest_integer = round(value)
    if abs(value - nearest_integer) <= tolerance:
        return float(nearest_integer)
    return value


def _snap_coord_bounds(bounds, resolution=None):
    return tuple(_snap_coord_bound(bound, resolution) for bound in bounds)


def _coord_bound_resolution(ds, lon_key, lat_key, lon_bnds_key=None, lat_bnds_key=None):
    if "spatial_resolution" in ds.attrs:
        return ds.attrs["spatial_resolution"]
    try:
        from mhm_tools.common.resolution_handler import get_file_res

        return get_file_res(ds[lon_key], ds[lat_key], None)
    except ValueError:
        pass
    for bounds_key in (lon_bnds_key, lat_bnds_key):
        if bounds_key is None or bounds_key not in ds:
            continue
        bounds = np.asarray(ds[bounds_key].values, dtype=float)
        widths = np.abs(bounds[..., 1] - bounds[..., 0])
        widths = widths[np.isfinite(widths) & (widths > 0)]
        if widths.size:
            return float(np.nanmedian(widths))
    return None


def get_ds_extend(ds, var=None, recursive_search=True, resolutions=None):
    """Get the spatial extent of a dataset as (lon_min, lon_max, lat_min, lat_max) from its bounds."""
    from mhm_tools.common.resolution_handler import get_file_res

    if var is not None:
        # get the coordinate keys from the variable if possible, otherwise from the dataset
        lon_key = get_coord_key(ds[var], lon=True)
        lat_key = get_coord_key(ds[var], lat=True)
    else:
        lon_key = get_coord_key(ds, lon=True)
        lat_key = get_coord_key(ds, lat=True)
    lon = ds[lon_key]
    lat = ds[lat_key]
    lon_bnds_key = lon.attrs.get("bounds", None)
    lat_bnds_key = lat.attrs.get("bounds", None)
    res = _coord_bound_resolution(ds, lon_key, lat_key, lon_bnds_key, lat_bnds_key)
    if lon_bnds_key is not None and lat_bnds_key is not None:
        lon_min = float(ds[lon_bnds_key].values.min())
        lon_max = float(ds[lon_bnds_key].values.max())
        lat_min = float(ds[lat_bnds_key].values.min())
        lat_max = float(ds[lat_bnds_key].values.max())
        return _snap_coord_bounds((lon_min, lon_max, lat_min, lat_max), res)
    if recursive_search:
        ds = generate_bounds_for_all_coords(ds)
        return get_ds_extend(ds, var=var, recursive_search=False)
    logger.warning(
        "Could not find coordinate bounds for dataset; estimating spatial extent from coordinate values."
    )
    lon_vals = np.asarray(ds[lon_key].values)
    lat_vals = np.asarray(ds[lat_key].values)
    res = (
        ds.attrs.get("spatial_resolution")
        if "spatial_resolution" in ds.attrs
        else (get_file_res(ds[lon_key], ds[lat_key], resolutions=resolutions))
    )
    if recursive_search:
        ds = generate_bounds_for_all_coords(ds, res=res)
        return get_ds_extend(
            ds, var=var, recursive_search=False, resolutions=resolutions
        )
    return _snap_coord_bounds(
        (
            float(np.nanmin(lon_vals)) - float(res) / 2,
            float(np.nanmax(lon_vals)) + float(res) / 2,
            float(np.nanmin(lat_vals)) - float(res) / 2,
            float(np.nanmax(lat_vals)) + float(res) / 2,
        ),
        res,
    )
