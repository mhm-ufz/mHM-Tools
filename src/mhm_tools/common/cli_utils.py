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

from mhm_tools.common.file_handler import get_xarray_ds_from_file
from mhm_tools.common.logger import ErrorLogger
from mhm_tools.common.xarray_utils import get_coord_key

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
    logger.info(f"mem_string {mem_str}")
    if mem_str.endswith("kb"):
        return int(mem_str[:-2]) // 1000_000
    if mem_str.endswith("mb"):
        return int(mem_str[:-2]) // 1000
    if mem_str.endswith("gb"):
        return int(mem_str[:-2])
    return int(mem_str) * 1_000_000_000


def get_coords_from_mask(mask, mask_key=None):
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
    with get_xarray_ds_from_file(mask, normalize_latlon_coords=True) as mask_ds:

        if mask_key is None:
            mask_key = next(
                key
                for key in ["mask", "land_mask", "mask_l2"]
                if key in mask_ds.data_vars
            )
        mask_da = mask_ds[mask_key]
        lon_key = get_coord_key(mask_da, lon=True)
        lat_key = get_coord_key(mask_da, lat=True)
        lon = mask_da[lon_key]
        lat = mask_da[lat_key]
        lon_min_target_grid = lon.min()
        lon_max_target_grid = lon.max()
        lat_min_target_grid = lat.min()
        lat_max_target_grid = lat.max()

        # if bounds are avaiable use them:
        # try to get it from coordinate bounds
        if "bounds" in lon.attrs and "bounds" in lat.attrs:
            lon_bounds_key = lon.attrs["bounds"]
            lon_bounds = mask_ds[lon_bounds_key]
            lat_bounds_key = lat.attrs["bounds"]
            lat_bounds = mask_ds[lat_bounds_key]
            resolution = float(lon_bounds.values[0][1] - lon_bounds.values[0][0]) / 2
            lon_min_target_grid = lon_bounds.values[0][0]
            lon_max_target_grid = lon_bounds.values[-1][1]
            lat_min_target_grid = lat_bounds.values[0][0]
            lat_max_target_grid = lat_bounds.values[-1][1]
        else:
            # change values from center cell to corner values
            try:
                resolution = float(lon.values[1] - lon.values[0])
                lon_min_target_grid -= resolution / 2
                lon_max_target_grid += resolution / 2
                lat_min_target_grid -= resolution / 2
                lat_max_target_grid += resolution / 2
            except Exception as e:
                if mask_key == "mask_l2":
                    # switch to different mask key for min max calculation
                    mask_key = "mask"
                    mask_da_fine = mask_ds[mask_key]
                    lon_key_fine = get_coord_key(mask_da_fine, lon=True)
                    lat_key_fine = get_coord_key(mask_da_fine, lat=True)
                    lon_fine = mask_da_fine[lon_key_fine]
                    lat_fine = mask_da_fine[lat_key_fine]
                    resolution = float(lon_fine.values[1] - lon_fine.values[0])
                    lon_min_target_grid = lon_fine.min() - resolution / 2
                    lon_max_target_grid = lon_fine.max() + resolution / 2
                    lat_min_target_grid = lat_fine.min() - resolution / 2
                    lat_max_target_grid = lat_fine.max() + resolution / 2
                else:
                    raise e

        logger.debug(
            f"Read coord from mask file: lat ({lat_min_target_grid} to {lat_max_target_grid}) {(lon_max_target_grid-lat_min_target_grid)/resolution} cells and lon ({lon_min_target_grid} to {lon_max_target_grid}) {(lon_max_target_grid-lat_min_target_grid)/resolution} cells"
        )

        if lat_min_target_grid > lat_max_target_grid:
            lat_min_target_grid, lat_max_target_grid = (
                lat_max_target_grid,
                lat_min_target_grid,
            )
        if lon_min_target_grid > lon_max_target_grid:
            lon_min_target_grid, lon_max_target_grid = (
                lon_max_target_grid,
                lon_min_target_grid,
            )

        return (
            lon_min_target_grid,
            lon_max_target_grid,
            lat_min_target_grid,
            lat_max_target_grid,
            mask_da,
        )


def get_coords(
    lonlatbox=None,
    mask_file=None,
    lon_min=None,
    lon_max=None,
    lat_min=None,
    lat_max=None,
    raise_exception=True,
    mask_var=None,
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
        lonlat_split = lonlatbox.split(",")
        lon_min_val, lon_max_val, lat_min_val, lat_max_val = map(
            float, lonlat_split[:4]
        )
        mask = None
    elif mask_file is not None:
        lon_min_val, lon_max_val, lat_min_val, lat_max_val, mask = get_coords_from_mask(
            mask_file, mask_key=mask_var
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
    return (
        float(lon_min_val),
        float(lon_max_val),
        float(lat_min_val),
        float(lat_max_val),
        mask,
    )
