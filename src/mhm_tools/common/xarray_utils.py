"""Provides basic xarray utils."""

import logging

import xarray as xr

from mhm_tools.common.logger import ErrorLogger

logger = logging.getLogger(__name__)


def get_coord_key(ds, lat=False, lon=False):
    """Return the lat or lon coordinate name used in the xarray dataset."""
    if (lon and lat) or not (lon or lat):
        with ErrorLogger(logger):
            msg = f"only lon or lat should be true but lon={lon} and lat={lat}"
            raise ValueError(msg)
    if lat:
        keys = ["lat", "latitude", "northing", "y", "new_y"]
    else:
        keys = ["lon", "longitude", "easting", "x", "new_x"]
    ds_dims = ds.dims if isinstance(ds, xr.DataArray) else ds.coords
    for key in keys:
        if key in ds_dims and len(ds[key].shape) == 1:
            return key
    for key in keys:
        if key in ds_dims:
            logger.warning(
                f"{type(ds)} contains key: {key} but ds[key] has shape {ds[key].shape}."
            )
            return key
    with ErrorLogger(logger):
        msg = f"None of {keys} in {type(ds).__name__} keys {ds_dims}."
        raise ValueError(msg)
