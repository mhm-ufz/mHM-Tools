"""Provides basic xarray utils."""

import logging

import numpy as np
import xarray as xr

from mhm_tools.common.logger import ErrorLogger

logger = logging.getLogger(__name__)

def normalize_lat_lon(ds: xr.Dataset, lat: str, lon: float) -> xr.Dataset:
    """
    Normalize names for latitude/longitude.
    """
    if "lat" not in ds.coords:
        ds = ds.rename_vars({lat: "lat"})
    if "lon" not in ds.coords:
        ds = ds.rename_vars({lon: "lon"})
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
    # then select possible keys from the following lists and try them untill a fitting one is found.
    if lat:
        keys = ["lat", "latitude", "northing", "y", "new_y", "Y"]
    elif lon:
        keys = ["lon", "longitude", "easting", "x", "new_x", "X"]
    else:
        keys = ["time", "month_of_year"]
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
    return ""


def get_single_data_var(ds):
    """Get the data var name from da dataset that only contains one data variable."""
    data_vars = list(ds.data_vars)
    if len(data_vars) > 1:
        logger.error("Only single data_var allowed")
        return None
    logger.debug(f"data_vars: {data_vars}")
    return data_vars[0]


def induce_data_var_from_file_name(ds, file_path):
    """Check if one of the data_vars is part of the file name and select it as most probable data_var."""
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
    """
    Map a median timedelta to a pandas frequency alias.

    - ~1 day  → 'D'
    - ~7 days → 'W'
    - ~28–31 days → 'M'
    - otherwise: fall back to '<N>H'
    """
    median_delta = ds.time.diff("time").median()
    days = median_delta / np.timedelta64(1, "D")
    hours = int(median_delta / np.timedelta64(1, "h"))
    if abs(days - 1) < 0.5:
        return hours, "D"
    if abs(days - 7) < 1:
        return hours, "W"
    if 27 < days < 32:
        return hours, "ME"
    # fallback: integer hours
    return hours, f"{hours}H"


def get_overlapping_time_slice(input_ds, ref_ds):
    """Crop data to overlapping time."""
    t1 = input_ds.dropna(dim="time", how="all").time.data
    t2 = ref_ds.dropna(dim="time", how="all").time.data
    logger.debug(f"input {t1[0]} till {t1[-1]}")
    logger.debug(f"ref {t2[0]} till {t2[-1]}")
    # Find overlapping range
    only_nan_msg = "No non nan value data."
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
        with ErrorLogger:
            raise ValueError(only_nan_msg)
    return slice(start, end)

def crop_ds(
    ds: xr.Dataset,
    lon_min: float,
    lon_max: float,
    lat_min: float,
    lat_max: float
) -> xr.Dataset:
    """
    Crop an xarray.Dataset to the given lon/lat bounds, handling coordinate order.
    """
    # longitude
    lon_vals = ds['lon'].values
    if lon_vals[0] <= lon_vals[-1]:
        lon_slice = slice(lon_min, lon_max)
    else:
        lon_slice = slice(lon_max, lon_min)

    # latitude
    lat_vals = ds['lat'].values
    if lat_vals[0] <= lat_vals[-1]:
        lat_slice = slice(lat_min, lat_max)
    else:
        lat_slice = slice(lat_max, lat_min)

    return ds.sel(lon=lon_slice, lat=lat_slice)