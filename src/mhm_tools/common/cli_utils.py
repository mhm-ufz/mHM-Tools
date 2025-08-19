"""
General CLI utility functions.

This module provides helpers for common command-line tasks such as:
- Parsing 'lat,lon' strings into float tuples
- Converting memory size strings (e.g., "10MB", "2GB") into bytes
- Determining coordinate extents from NetCDF mask datasets
- Consolidating coordinate inputs from strings, mask files, or explicit values
"""

import argparse
import logging

import xarray as xr

from mhm_tools.common.logger import ErrorLogger

logger = logging.getLogger(__name__)


def parse_coords(coords_str):
    """Split the input string of 'lat,lon' by comma and convert each part to a float."""
    try:
        lat, lon = map(float, coords_str.split(","))
        return lat, lon
    except ValueError as err:
        with ErrorLogger(logger):
            msg = "Coordinates must be two comma-separated floats."
            raise argparse.ArgumentTypeError(msg) from err


def get_available_mem_in_unit(available_mem):
    """Convert a memory string with units into an integer number of bytes.

    Accepts strings like '10MB', '2GB', or raw numbers (interpreted as bytes).
    Returns None if input is None.
    """
    if available_mem is None:
        return None
    mem_str = available_mem.lower().strip()
    if mem_str.endswith("mb"):
        return int(mem_str[:-2]) * 1_000_000
    if mem_str.endswith("gb"):
        return int(mem_str[:-2]) * 1_000_000_000
    return int(mem_str)


def get_coords_from_mask(mask):
    """Get the coordinate extents from a mask NetCDF file.

    Parameters
    ----------
    mask : str
        Path to the mask file.

    Returns
    -------
    tuple
        (lon_min, lon_max, lat_min, lat_max, mask_dataarray)
    """
    ds = xr.open_dataset(mask)
    lon = ds.lon
    lat = ds.lat
    lon_min = lon.min()
    lon_max = lon.max()
    lat_min = lat.min()
    lat_max = lat.max()

    # adjust from cell centers to corner boundaries
    resolution = ds.lon.values[1] - ds.lon.values[0]
    lon_min -= resolution / 2
    lon_max += resolution / 2
    lat_min -= resolution / 2
    lat_max += resolution / 2

    logger.debug(
        f"Read coord from mask file: lat ({lat_min} to {lat_max}) "
        f"{(lon_max - lon_min) / resolution} cells and lon ({lon_min} to {lon_max}) "
        f"{(lon_max - lon_min) / resolution} cells"
    )

    if lat_min > lat_max:
        lat_min, lat_max = lat_max, lat_min
    if lon_min > lon_max:
        lon_min, lon_max = lon_max, lon_min

    mask_key = next(key for key in ["mask", "land_mask"] if key in ds.data_vars)
    mask_da = ds[mask_key]
    return lon_min, lon_max, lat_min, lat_max, mask_da


def get_coords(
    lonlatbox=None,
    mask_file=None,
    lon_min=None,
    lon_max=None,
    lat_min=None,
    lat_max=None,
    raise_exception=True,
):
    """Get coordinate bounds from a lonlatbox string, mask file, or explicit values.

    Parameters
    ----------
    lonlatbox : str, optional
        Comma-separated 'lon_min,lon_max,lat_min,lat_max'.
    mask_file : str, optional
        Path to a mask NetCDF file.
    lon_min, lon_max, lat_min, lat_max : float, optional
        Explicit coordinate bounds.
    raise_exception : bool
        If True, raise ValueError when inputs are insufficient.

    Returns
    -------
    tuple
        (lon_min, lon_max, lat_min, lat_max, mask_dataarray or None)
    """
    mask = None
    if lonlatbox is not None:
        lon_min_val, lon_max_val, lat_min_val, lat_max_val = map(
            float, lonlatbox.split(",")
        )
        mask = None
    elif mask_file is not None:
        lon_min_val, lon_max_val, lat_min_val, lat_max_val, mask = get_coords_from_mask(
            mask_file
        )
    elif None not in (lon_min, lon_max, lat_min, lat_max):
        lon_min_val, lon_max_val, lat_min_val, lat_max_val = (
            lon_min,
            lon_max,
            lat_min,
            lat_max,
        )
    elif raise_exception:
        with ErrorLogger(logger):
            msg = "Either lonlatbox, mask_file, or all coordinate bounds must be provided."
            raise ValueError(msg)
    else:
        return None, None, None, None, None
    return lon_min_val, lon_max_val, lat_min_val, lat_max_val, mask
