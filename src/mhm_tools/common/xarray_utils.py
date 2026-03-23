"""Provides basic xarray utils."""

import logging
from typing import Optional

import numpy as np
import pandas as pd
import xarray as xr
from scipy.stats import spearmanr

from mhm_tools.common.constants import LAT_KEYS, LON_KEYS, TIME_KEYS
from mhm_tools.common.logger import ErrorLogger

logger = logging.getLogger(__name__)


def normalize_lat_lon(
    ds: xr.Dataset,
    lat: Optional[str] = None,
    lon: Optional[str] = None,
    raise_exceptions: bool = False,
) -> xr.Dataset:
    """
    Normalize latitude and longitude dimension and coordinate names to 'lat' and 'lon'.

    Handles both dimensions and coordinate variables.
    """
    try:
        rename_dict = {}
        if lat is None:
            lat = get_coord_key(ds, lon=True)
        if lon is None:
            lon = get_coord_key(ds, lon=True)

        coords_and_dims = list(ds.coords) + list(ds.dims)

        # Rename coordinate variables if needed
        if lat is not None and "lat" not in coords_and_dims and lat in coords_and_dims:
            rename_dict[lat] = "lat"
        if lon is not None and "lon" not in coords_and_dims and lon in coords_and_dims:
            rename_dict[lon] = "lon"

        return ds.rename(rename_dict)
    except Exception as e:
        if raise_exceptions:
            with ErrorLogger(logger):
                raise (e)
        else:
            logger.warning(f"Exception in normalize lat lon: {e}")
            return ds


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
    """Crop data to overlapping time."""
    t1 = input_ds.dropna(dim="time", how="all").time.data
    t2 = ref_ds.dropna(dim="time", how="all").time.data
    logger.debug(f"Non-nan time values in input: {t1}")
    logger.debug(f"Non-nan time values in reference: {t2}")
    # Find overlapping range
    only_nan_msg = "No non nan value data."
    if t1.size == 0 or t2.size == 0:
        with ErrorLogger(logger):
            raise ValueError(only_nan_msg)

    logger.debug(f"input {t1[0]} till {t1[-1]}")
    logger.debug(f"ref {t2[0]} till {t2[-1]}")

    if t1.any() and t2.any():
        start = str(max(t1[0], t2[0]))
        end = str(min(t1[-1], t2[-1]))
        start = start.split("T")[0] if "T" in start else start
        end = end.split("T")[0] if "T" in end else end
        if end <= start:
            logger.warning(
                f"The two datasets are not overlapping. Sim data hass non nan data from {t1[0]} to {t1[-1]} and obs from {t2[0]} to {t2[-1]}."
            )
        logger.info(f"Cropping data to timeframe {start} to {end}")
    else:
        with ErrorLogger(logger):
            raise ValueError(only_nan_msg)
    start = pd.to_datetime(start)
    end = pd.to_datetime(end)
    return slice(start, end)


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
