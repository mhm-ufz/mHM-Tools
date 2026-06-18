"""Create the latlon.nc grid description required by mHM.

The module builds compatible L0, L1, L11, and L2 grid definitions from headers,
cell sizes, or existing files, checks grid alignment, and writes the combined
latlon NetCDF file used by mHM/mRM setups.

Authors
-------
- Jeisson Leal
- Sebastian Müller
"""

import logging
import time
from pathlib import Path

import numpy as np
import xarray as xr
from pyproj import Proj

from mhm_tools.common.file_handler import (
    create_header,
    get_xarray_ds_from_file,
    write_xarray_to_file,
)
from mhm_tools.common.logger import ErrorLogger, log_arguments

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

logger = logging.getLogger(__name__)


def xy_to_latlon(x, y, crs=None):
    """Convert cartesian coordinates to lat-lon.

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
    """Create grid from ascii header."""
    c_size = header["cellsize"]
    x = header["xllcorner"] + c_size / 2 + np.arange(header["ncols"]) * c_size
    y = header["yllcorner"] + c_size / 2 + np.arange(header["nrows"]) * c_size
    y = np.flip(y)
    x_grid, y_grid = np.meshgrid(x, y)
    # determine latitude and longitude of the target grid
    lons, lats = xy_to_latlon(x_grid, y_grid, crs)
    return x.astype(dtype), y.astype(dtype), lons.astype(dtype), lats.astype(dtype)


def get_header_from_file(file):
    """Get level-0 header from file."""
    file = Path(file)
    if file.suffix.lower() == ".nc":
        with get_xarray_ds_from_file(file) as ds:
            return create_header(ds)
    elif file.suffix.lower() in [".asc", ".hdr", ".txt"]:
        return read_header(file)
    else:
        msg = f"Cannot read level-0 header from file {file!r}."
        with ErrorLogger(logger):
            raise ValueError(msg)


def add_optional_level(
    coords,
    level,
    level_name,
    level0_header,
    level1_header,
    crs,
    dtype,
    write_header_path=None,
):
    """Add optional level to the latlon dataset."""
    if isinstance(level, (int, float)):
        level_header = rescale_grid(
            level0_header, level, in_name="L0", out_name=level_name
        )
    elif not isinstance(level, dict):
        level_header = get_header_from_file(level)
    level_header = standardize_header(level_header)
    # check L0/level compatibility
    check_grid_compatibility(level_header, level0_header, "L0", level_name)
    # check L1/level compatibility
    check_grid_compatibility(level_header, level1_header, "L1", level_name)
    # create grids
    x_level, y_level, lons_level, lats_level = _create_grid(level_header, crs, dtype)
    coords[f"yc_{level_name}"] = y_level
    coords[f"xc_{level_name}"] = x_level
    coords[f"lat_{level_name}"] = (f"yc_{level_name}", f"xc_{level_name}"), lats_level
    coords[f"lon_{level_name}"] = (f"yc_{level_name}", f"xc_{level_name}"), lons_level
    if write_header_path:
        write_header(write_header_path, level_header)


@log_arguments()
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
    """Create the latlon.nc file from given ASCII headers.

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
        level0 = get_header_from_file(level0)
    level0 = standardize_header(level0)
    if isinstance(level1, (int, float)):
        level1 = rescale_grid(level0, level1, in_name="L0", out_name="L1")
    elif not isinstance(level1, dict):
        level1 = get_header_from_file(level1)
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
        add_optional_level(
            coords,
            level11,
            level_name="l11",
            level0_header=level0,
            level1_header=level1,
            crs=crs,
            dtype=dtype,
            write_header_path=write_header_l11,
        )

    # level2 is optional
    if level2:
        add_optional_level(
            coords,
            level2,
            level_name="l2",
            level0_header=level0,
            level1_header=level1,
            crs=crs,
            dtype=dtype,
            write_header_path=write_header_l2,
        )
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
    write_xarray_to_file(ds=latlon, file_path=out_file)  # , encoding=encoding)
