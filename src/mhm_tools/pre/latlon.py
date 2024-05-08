"""
Create the latlon file for mHM.

Authors
-------
- Sebastian Müller
"""

import time

import numpy as np
import xarray as xr
from pyproj import Proj

from ..common import (
    NC_ENCODE_DEFAULTS,
    check_grid_compatibility,
    generate_bounds,
    read_header,
    rescale_grid,
    set_netcdf_encoding,
    standardize_header,
    write_header,
)


def xy_to_latlon(x, y, crs=None):
    """
    Convert cartesian coordinates to lat-lon.

    Parameters
    ----------
    x : arraylike
        x coordinates
    y : arraylike
        y coordinates
    crs : str, optional
        Coordinates reference system (e.g. 'epsg:3035'),
        by default None

    Returns
    -------
    tuple of arrays
        (lat, lon) arrays
    """
    if not crs:
        return x, y
    transform = Proj(crs)
    # longitude, latitude
    return transform(x, y, inverse=True)


def _create_grid(header, crs=None, dtype="f4"):
    # create x and y grid
    c_size = header["cellsize"]
    x = header["xllcorner"] + c_size / 2 + np.arange(header["ncols"]) * c_size
    y = header["yllcorner"] + c_size / 2 + np.arange(header["nrows"]) * c_size
    x_grid, y_grid = np.meshgrid(x, y)
    # determine latitude and longitude of the target grid
    lons, lats = xy_to_latlon(x_grid, y_grid, crs)
    return x.astype(dtype), y.astype(dtype), lons.astype(dtype), lats.astype(dtype)


#     """This function writes the latlon.nc file from given ASCII headers."""


def create_latlon(
    out_file,
    level0,
    level1,
    level11=None,
    level2=None,
    write_header_l0=None,
    write_header_l1=None,
    write_header_l11=None,
    write_header_l2=None,
    crs=None,
    dtype="f4",
    compression=9,
    add_bounds=False,
):
    """
    Create the latlon.nc file from given ASCII headers.

    Parameters
    ----------
    out_file : Pathlike
        The path of the output NetCDF file containing the latlon information.
    level0 : dict or Pathlike
        Level-0 (DEM) information. Either an ascii (header) file
        or a dictionary containing the header information.
    level1 : float or dict or Pathlike
        Level-1 (hydrology) information. Either an ascii (header) file,
        a dictionary containing the header information
        or a cell-size to determine information from level-0.
    level11 : float or dict or Pathlike, optional
        Level-11 (routing) information. Either an ascii (header) file,
        a dictionary containing the header information
        or a cell-size to determine information from level-0.
    level2 : float or dict or Pathlike, optional
        Level-2 (meteorology) information. Either an ascii (header) file,
        a dictionary containing the header information
        or a cell-size to determine information from level-0.
        Level-2 information wont be written to the latlon file.
    write_header_l0 : Pathlike, optional
        Write the level-0 header to a given file path.
    write_header_l1 : Pathlike, optional
        Write the level-1 header to a given file path.
    write_header_l11 : Pathlike, optional
        Write the level-11 header to a given file path.
    write_header_l2 : Pathlike, optional
        Write the level-2 header to a given file path.
    crs : str, optional
        Coordinates reference system (e.g. 'epsg:3035').
        If not given, headers will be interpreted as given in lat-lon ('epsg:4326').
    dtype : str, optional
        Data type for the latlon file and headers, by default "f4"
    compression : int, optional
        Compression level for the NetCDF file, by default 9
    add_bounds : bool, optional
        Add bounds to the NetCDF axis, by default False
    """
    coords = {}

    # read header information
    if not isinstance(level0, dict):
        level0 = read_header(level0)
    level0 = standardize_header(level0)
    if isinstance(level1, (int, float)):
        level1 = rescale_grid(level0, level1, in_name="L0", out_name="L1")
    elif not isinstance(level1, dict):
        level1 = read_header(level1)
    level1 = standardize_header(level1)
    # check L0/L1 compatibility
    check_grid_compatibility(level0, level1, "L0", "L1")

    if write_header_l0:
        write_header(write_header_l0, level0)
    if write_header_l1:
        write_header(write_header_l1, level1)

    # create grids
    x_l0, y_l0, lons_l0, lats_l0 = _create_grid(level0, crs, dtype)
    x_l1, y_l1, lons_l1, lats_l1 = _create_grid(level1, crs, dtype)
    coords["yc_l0"] = y_l0
    coords["xc_l0"] = x_l0
    coords["lat_l0"] = (["yc_l0", "xc_l0"], lats_l0)
    coords["lon_l0"] = (["yc_l0", "xc_l0"], lons_l0)
    coords["yc_l1"] = y_l1
    coords["xc_l1"] = x_l1
    coords["lat"] = (["yc_l1", "xc_l1"], lats_l1)
    coords["lon"] = (["yc_l1", "xc_l1"], lons_l1)

    # level11 is optional
    if level11:
        if isinstance(level11, (int, float)):
            level11 = rescale_grid(level0, level11, in_name="L0", out_name="L11")
        elif not isinstance(level11, dict):
            level11 = read_header(level11)
        level11 = standardize_header(level11)
        # check L0/L11 compatibility
        check_grid_compatibility(level0, level11, "L0", "L11")
        # check L1/L11 compatibility
        check_grid_compatibility(level1, level11, "L1", "L11")
        # create grids
        x_l11, y_l11, lons_l11, lats_l11 = _create_grid(level11, crs, dtype)
        coords["yc_l11"] = y_l11
        coords["xc_l11"] = x_l11
        coords["lat_l11"] = (["yc_l11", "xc_l11"], lats_l11)
        coords["lon_l11"] = (["yc_l11", "xc_l11"], lons_l11)
        if write_header_l11:
            write_header(write_header_l11, level11)

    # level2 is optional
    if level2:
        if isinstance(level2, (int, float)):
            level2 = rescale_grid(level0, level2, in_name="L0", out_name="L2")
        elif not isinstance(level2, dict):
            level2 = read_header(level2)
        level2 = standardize_header(level2)
        # check L0/L2 compatibility
        check_grid_compatibility(level0, level2, "L0", "L2")
        # check L1/L2 compatibility
        check_grid_compatibility(level1, level2, "L1", "L2")
        if write_header_l2:
            write_header(write_header_l2, level2)

    latlon = xr.Dataset(
        coords=coords,
        attrs={
            "description": "lat lon file",
            "projection": crs or "epsg:4326",
            "history": f"Created {time.ctime()}",
        },
    )
    dims = set(latlon.dims)
    all_coords = set(latlon.coords)
    dim_coords = all_coords & dims  # intersection
    aux_coords = all_coords - dims  # difference

    # add metadata
    for var in aux_coords:
        is_lat = var.startswith("lat")
        level = "11" if var.endswith("11") else ("0" if var.endswith("0") else "1")
        suffix = " at level " + level
        name = "latitude" if is_lat else "longitude"
        latlon[var].attrs["standard_name"] = name
        latlon[var].attrs["long_name"] = name + suffix
        latlon[var].attrs["units"] = "degrees_" + ("north" if is_lat else "east")
    for var in dim_coords:
        level = var.split("l")[1]
        suffix = "-coordinate in Cartesian system at level " + level
        latlon[var].attrs["axis"] = var[0].upper()
        latlon[var].attrs["long_name"] = var[0].lower() + suffix
        latlon[var].attrs["units"] = "m"
        if add_bounds:
            bounds_name = f"{var}_bnds"
            latlon.coords[bounds_name] = generate_bounds(latlon[var])
            latlon[var].attrs["bounds"] = bounds_name

    # compression
    encoding = NC_ENCODE_DEFAULTS.copy()
    if 0 < compression < 10:
        encoding.update({"zlib": True, "complevel": compression})
    set_netcdf_encoding(latlon, encoding)
    # save netcdf file
    latlon.to_netcdf(out_file)
