"""
Coordinate reference system helpers for raster data.

Authors
-------
- Sanjeev Bashyal
"""

import contextlib

import numpy as np
import xarray as xr
from rasterio.crs import CRS

from mhm_tools.common.xarray_utils import get_coord_key, get_single_data_var

__all__ = [
    "MissingCRSError",
    "resolve_crs",
    "set_spatial_dims",
    "write_object_crs",
]

# Attributes that may carry a CRS on data rioxarray cannot georeference itself.
CRS_ATTRS = ("crs_wkt", "spatial_ref", "crs")


class MissingCRSError(ValueError):
    """Raised when a raster operation requires CRS metadata but none exists."""


def _rio(obj):
    """Return the ``rio`` accessor, importing rioxarray to register it."""
    import rioxarray  # noqa: F401  (imported for its accessor side effect)

    return obj.rio


def _embedded_crs(obj):
    """Return the CRS an object carries, by accessor first and attributes second."""
    with contextlib.suppress(Exception):
        crs = _rio(obj).crs
        if crs is not None:
            return crs
    for key in CRS_ATTRS:
        candidate = getattr(obj, "attrs", {}).get(key)
        if candidate:
            return CRS.from_user_input(candidate)
    return None


def resolve_crs(obj, explicit_crs=None, *, required=False):
    """Resolve CRS metadata.

    An explicit CRS is used only when the object has no CRS. Supplying an
    explicit CRS that conflicts with embedded metadata raises ``ValueError``.
    """
    embedded = _embedded_crs(obj)
    embedded = CRS.from_user_input(embedded) if embedded is not None else None
    explicit = CRS.from_user_input(explicit_crs) if explicit_crs is not None else None
    if embedded is not None and explicit is not None and embedded != explicit:
        msg = f"Explicit CRS {explicit} conflicts with embedded CRS {embedded}."
        raise ValueError(msg)
    resolved = embedded or explicit
    if required and resolved is None:
        msg = "Raster CRS is missing. Provide CRS metadata or an explicit CRS."
        raise MissingCRSError(msg)
    return resolved


def set_spatial_dims(data: xr.DataArray) -> xr.DataArray:
    """Tag the X and Y dimensions of a regularly spaced two-dimensional array."""
    y_dim = get_coord_key(data, lat=True)
    x_dim = get_coord_key(data, lon=True)
    if y_dim == x_dim or not {y_dim, x_dim}.issubset(data.dims):
        msg = "Raster requires distinct one-dimensional X and Y coordinates."
        raise ValueError(msg)
    for dim in (y_dim, x_dim):
        if data[dim].ndim != 1:
            msg = f"Raster coordinate {dim!r} must be one-dimensional."
            raise ValueError(msg)
        _validate_regular_axis(data[dim].values, dim)
    return _rio(data).set_spatial_dims(x_dim=x_dim, y_dim=y_dim, inplace=False)


def _validate_regular_axis(values, name):
    """Reject an axis that is not finite and evenly spaced."""
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


def write_object_crs(obj, crs):
    """Attach CRS metadata to a Dataset or DataArray."""
    if isinstance(obj, xr.DataArray):
        return set_spatial_dims(obj).rio.write_crs(crs, inplace=False)
    name = get_single_data_var(obj)
    if name is None:
        msg = "CRS assignment requires exactly one payload data variable."
        raise ValueError(msg)
    data = set_spatial_dims(obj[name]).rio.write_crs(crs, inplace=False)
    result = obj.copy(deep=False)
    result[name] = data
    return result.rio.set_spatial_dims(
        x_dim=data.rio.x_dim,
        y_dim=data.rio.y_dim,
        inplace=False,
    ).rio.write_crs(crs, inplace=False)
