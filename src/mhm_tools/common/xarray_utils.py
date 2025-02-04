"""Provides basic xarray utils."""

import logging

import xarray as xr

from mhm_tools.common.logger import ErrorLogger

logger = logging.getLogger(__name__)


def get_coord_key(ds, lat=False, lon=False, raise_exception=True, is_retry=False):
    """Return the lat or lon coordinate name used in the xarray dataset."""
    if (lon and lat) or not (lon or lat):
        with ErrorLogger(logger):
            msg = f"only lon or lat should be true but lon={lon} and lat={lat}"
            raise ValueError(msg)
    if lat:
        keys = ["lat", "latitude", "northing", "y", "new_y", "Y"]
    else:
        keys = ["lon", "longitude", "easting", "x", "new_x", "X"]
    ds_dims = ds.dims if isinstance(ds, xr.DataArray) or is_retry else ds.coords
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
            ds, lat=lat, lon=lon, raise_exception=raise_exception, is_retry=True
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
