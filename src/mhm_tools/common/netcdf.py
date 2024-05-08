"""Common NetCDF/xarray routines."""

import xarray as xr

from .constants import NC_ENCODE_DEFAULTS


def set_netcdf_encoding(ds, var_encoding=None):
    """
    Set default netcdf encoding settings to a xarray data-set.

    Parameters
    ----------
    ds : :class:`xarray.Dataset`
        xarray dataset to set the encoding.
    var_encoding : :class:`dict`, optional
        Encoding for variables within the given dataset,
        by default :any:`NC_ENCODE_DEFAULTS`
    """
    var_encoding = var_encoding or NC_ENCODE_DEFAULTS
    dims = set(ds.dims)
    all_coords = set(ds.coords)
    dim_coords = all_coords & dims  # intersection
    aux_coords = all_coords - dims  # difference
    bnds = {ds[c].attrs["bounds"] for c in all_coords if "bounds" in ds[c].attrs}
    vars = set(ds.data_vars) - bnds

    for v in aux_coords | vars:  # union
        ds[v].encoding = var_encoding

    # missing data not allowed for coordinates by CF convention
    # http://cfconventions.org/cf-conventions/cf-conventions.html#missing-data
    # no FillValue for dim-coords and bounds
    for v in dim_coords | bnds:  # union
        ds[v].encoding = {"_FillValue": None}


def generate_bounds(da, bounds_dim="bnds"):
    """
    Generate bounds for a given coordinate.

    Parameters
    ----------
    da : :class:`xarray.Dataarray`
        Data array holding a coordinate.

    Returns
    -------
    :class:`xarray.Dataarray`
        Bounds for the given coordinate.
    """
    (dim,) = da.dims
    diff = da.diff(dim)
    lower = da - diff / 2
    upper = da + diff / 2
    bounds = xr.concat([lower, upper], dim=bounds_dim)
    first = (bounds.isel({dim: 0}) - diff.isel({dim: 0})).assign_coords(
        {dim: da[dim][0]}
    )
    return xr.concat([first, bounds], dim=dim).transpose()
