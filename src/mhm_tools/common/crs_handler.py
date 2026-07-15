"""Coordinate reference system helpers for raster data."""

import contextlib

import numpy as np
import xarray as xr
from rasterio.crs import CRS

from mhm_tools.common.xarray_utils import get_coord_key, get_single_data_var

__all__ = ["MissingCRSError", "resolve_crs"]


class MissingCRSError(ValueError):
    """Raised when a raster operation requires CRS metadata but none exists."""


def resolve_crs(obj, explicit_crs=None, *, required=False):
    """Resolve CRS metadata.

    An explicit CRS is used only when the object has no CRS. Supplying an
    explicit CRS that conflicts with embedded metadata raises ``ValueError``.
    """
    import rioxarray as rxr

    _ = rxr  # register the ``rio`` accessor
    embedded = None
    with contextlib.suppress(Exception):
        embedded = obj.rio.crs
    if embedded is None:
        candidates = [
            getattr(obj, "attrs", {}).get(key)
            for key in ("crs_wkt", "spatial_ref", "crs")
        ]
        for candidate in candidates:
            if candidate:
                embedded = CRS.from_user_input(candidate)
                break

    explicit = CRS.from_user_input(explicit_crs) if explicit_crs is not None else None
    embedded = CRS.from_user_input(embedded) if embedded is not None else None
    if embedded is not None and explicit is not None and embedded != explicit:
        msg = f"Explicit CRS {explicit} conflicts with embedded CRS {embedded}."
        raise ValueError(msg)
    resolved = embedded or explicit
    if required and resolved is None:
        msg = "Raster CRS is missing. Provide CRS metadata or an explicit CRS."
        raise MissingCRSError(msg)
    return resolved


def _set_spatial_dims(data: xr.DataArray) -> xr.DataArray:
    """Set rioxarray spatial dimensions on a two-dimensional array."""
    import rioxarray as rxr

    _ = rxr
    y_dim = get_coord_key(data, lat=True)
    x_dim = get_coord_key(data, lon=True)
    if y_dim == x_dim or y_dim not in data.dims or x_dim not in data.dims:
        msg = "Raster requires distinct one-dimensional X and Y coordinates."
        raise ValueError(msg)
    for dim in (y_dim, x_dim):
        if data[dim].ndim != 1:
            msg = f"Raster coordinate {dim!r} must be one-dimensional."
            raise ValueError(msg)
        _validate_regular_axis(data[dim].values, dim)
    return data.rio.set_spatial_dims(x_dim=x_dim, y_dim=y_dim, inplace=False)


def _validate_regular_axis(values, name):
    values = np.asarray(values)
    if values.size < 2:
        return
    deltas = np.diff(values.astype(float))
    if not np.all(np.isfinite(deltas)) or np.any(deltas == 0):
        msg = f"Raster coordinate {name!r} is invalid."
        raise ValueError(msg)
    if not np.allclose(deltas, deltas[0], rtol=1e-7, atol=1e-10):
        msg = f"Raster coordinate {name!r} is not regularly spaced."
        raise ValueError(msg)


def _write_object_crs(obj, crs):
    """Attach CRS metadata to a Dataset or DataArray."""
    if isinstance(obj, xr.DataArray):
        data = _set_spatial_dims(obj)
        return data.rio.write_crs(crs, inplace=False)
    name = get_single_data_var(obj)
    if name is None:
        msg = "CRS assignment requires exactly one payload data variable."
        raise ValueError(msg)
    data = _set_spatial_dims(obj[name]).rio.write_crs(crs, inplace=False)
    result = obj.copy(deep=False)
    result[name] = data
    return result.rio.set_spatial_dims(
        x_dim=data.rio.x_dim,
        y_dim=data.rio.y_dim,
        inplace=False,
    ).rio.write_crs(crs, inplace=False)
