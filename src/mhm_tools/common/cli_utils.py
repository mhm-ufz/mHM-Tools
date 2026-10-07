"""
General CLI utility functions.

This module provides helpers for common command-line tasks such as:
- Parsing 'lat,lon' strings into float tuples
- Converting memory size strings (e.g., "10MB", "2GB") into a budget in GiB
- Determining coordinate extents from NetCDF mask datasets
- Consolidating coordinate inputs from strings, mask files, or explicit values
- Declaring the shared lossy NetCDF quantization options
"""

import argparse
import logging
import re

from mhm_tools.common.file_handler import get_xarray_ds_from_file
from mhm_tools.common.logger import ErrorLogger
from mhm_tools.common.netcdf import (
    DEFAULT_COMPLEVEL,
    MAX_COMPLEVEL,
    MIN_COMPLEVEL,
    QUANTIZE_MODES,
    NetcdfCompression,
)
from mhm_tools.common.resolution_handler import calculate_coordinate_resolution
from mhm_tools.common.xarray_utils import get_coord_key, get_ds_extend

logger = logging.getLogger(__name__)


def add_netcdf_compression_args(parser):
    """Add the NetCDF compression options to a parser or argument group.

    Beside the zlib level these declare lossy quantization, which zeroes the
    insignificant mantissa bits of floating point variables so the lossless
    compression that follows shrinks the file markedly. Integer data such as
    masks is never quantized.

    Parameters
    ----------
    parser : argparse.ArgumentParser or argparse._ArgumentGroup
        Parser or group the options are added to. A parser gets its own
        "netcdf output" group so the options stay together in the help.

    Returns
    -------
    None
    """
    group = (
        parser.add_argument_group("netcdf output")
        if isinstance(parser, argparse.ArgumentParser)
        else parser
    )
    group.add_argument(
        "-x",
        "--compression",
        required=False,
        default=None,
        type=int,
        choices=range(MIN_COMPLEVEL, MAX_COMPLEVEL + 1),
        help=(
            "Lossless zlib compression level for the output. 0 writes the data "
            "uncompressed, 9 compresses hardest and slowest. Omit to keep the "
            f"level of the input file, or {DEFAULT_COMPLEVEL} when it has none."
        ),
    )
    group.add_argument(
        "--no-shuffle",
        required=False,
        action="store_true",
        help=(
            "Skip the HDF5 shuffle filter. It normally helps zlib on floating "
            "point data, so there is rarely a reason to turn it off."
        ),
    )
    group.add_argument(
        "--significant-digits",
        required=False,
        default=None,
        type=int,
        help=(
            "Reduce the stored precision of floating point variables to this "
            "many significant decimal digits, which makes the file compress "
            "much better. For --quantize-mode BitRound this counts significant "
            "BITS instead (9 bits are roughly 3 decimal digits). Omit to keep "
            "the precision of the input file, or pass 0 to store the full "
            "precision. Integer data and coordinates are never affected."
        ),
    )
    group.add_argument(
        "--quantize-mode",
        required=False,
        default="BitGroom",
        choices=list(QUANTIZE_MODES),
        help=(
            "Quantization method used with --significant-digits. BitGroom is "
            "the conservative default, GranularBitRound compresses more at the "
            "same digit count, BitRound counts bits rather than digits."
        ),
    )


def get_netcdf_compression(args):
    """Return the NetCDF compression settings the parsed arguments ask for.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed arguments, which need not carry the compression options.

    Returns
    -------
    NetcdfCompression or None
        Settings for the NetCDF writers, or None when no option was given, so
        an input file's own compression is kept.
    """
    complevel = getattr(args, "compression", None)
    significant_digits = getattr(args, "significant_digits", None)
    shuffle = not getattr(args, "no_shuffle", False)
    if complevel is None and significant_digits is None and shuffle:
        return None
    return NetcdfCompression(
        complevel=complevel,
        shuffle=None if shuffle else False,
        significant_digits=significant_digits,
        quantize_mode=getattr(args, "quantize_mode", "BitGroom"),
    )


def normalize_cli_sequence(values, split_whitespace=True):
    """Normalize repeated, comma-separated and space-separated CLI values.

    A repeatable option takes one value per occurrence, so `--x "a b"` arrives
    as one string. Splitting it keeps that form working instead of turning it
    into a single value that no file matches.

    Parameters
    ----------
    values : str or Sequence[str] or None
        CLI value or values to normalize.
    split_whitespace : bool, optional
        Also split on whitespace, by default True. Pass False for a value that
        may legitimately contain a space, such as a plot label.

    Returns
    -------
    list[str] or None
        Normalized values, or None when no values were supplied.
    """
    if values is None:
        return None
    if isinstance(values, str):
        values = [values]
    separators = r"[,\s]+" if split_whitespace else ","
    normalized_values = []
    for value in values:
        for part in re.split(separators, str(value)):
            striped_part = part.strip()
            if striped_part:
                normalized_values.append(striped_part)
    if not normalized_values:
        return None
    return normalized_values


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
    """Convert a memory string into a budget in Gb.

    Args:
        available_mem: Memory as a string carrying a `kb`, `mb` or `gb`
            suffix, or a plain number that is already read as Gb. None
            passes through. The suffixes step by 1000, so 500mb is 0.5 Gb.

    Returns
    -------
        The budget in Gb as a float, or None when nothing was given.
    """
    if available_mem is None:
        return None
    mem_str = str(available_mem).lower().strip()
    logger.info(f"mem_string {mem_str}")
    # every caller consumes this as `available_mem_gib`, so the value stays a
    # float: flooring it turned any budget below one GiB into zero
    for suffix, per_gib in (("kb", 1000**2), ("mb", 1000), ("gb", 1)):
        if mem_str.endswith(suffix):
            value = float(mem_str[: -len(suffix)]) / per_gib
            break
    else:
        value = float(mem_str)
    if value <= 0:
        msg = f"available memory must be greater than zero, got {available_mem!r}"
        with ErrorLogger(logger):
            raise ValueError(msg)
    return value


def get_coords_from_mask(mask, mask_key=None, resolutions=None):
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
        mask_da = mask_ds[mask_key].load()
        (
            lon_min_target_grid,
            lon_max_target_grid,
            lat_min_target_grid,
            lat_max_target_grid,
        ) = get_ds_extend(mask_ds, mask_key, resolutions=resolutions)

        # the resolution is only needed to report cell counts, so derive it lazily
        if logger.isEnabledFor(logging.DEBUG):
            lon_coord = mask_ds[get_coord_key(mask_ds, lon=True)]
            # a single-cell mask has no spacing to derive it from
            resolution = (
                calculate_coordinate_resolution(lon_coord)
                if lon_coord.size > 1
                else mask_ds.attrs.get("spatial_resolution", float("nan"))
            )
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


def parse_lonlatbox(lonlatbox):
    """Parse a lonlatbox of four bounds and an optional L0 resolution.

    Args:
        lonlatbox (str): 'lon_min,lon_max,lat_min,lat_max', optionally followed
            by ',resolution_l0'.

    Returns
    -------
        Tuple of lon_min, lon_max, lat_min, lat_max and the L0 resolution,
        which is None when only the four bounds are given.
    """
    values = [float(value) for value in str(lonlatbox).split(",")]
    if len(values) not in (4, 5):
        msg = (
            "The lonlatbox takes 'lon_min,lon_max,lat_min,lat_max' and an optional "
            f"',resolution_l0', but got {len(values)} values: {lonlatbox}"
        )
        with ErrorLogger(logger):
            raise ValueError(msg)
    resolution_l0 = values[4] if len(values) == 5 else None
    return (*values[:4], resolution_l0)


def get_coords(
    lonlatbox=None,
    mask_file=None,
    lon_min=None,
    lon_max=None,
    lat_min=None,
    lat_max=None,
    raise_exception=True,
    mask_var=None,
    resolutions=None,
):
    """Get coordinate bounds from a lonlatbox string, mask file, or explicit values.

    Parameters
    ----------
    lonlatbox : str, optional
        Comma-separated 'lon_min,lon_max,lat_min,lat_max', an optional fifth
        L0 resolution is ignored.
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
        lon_min_val, lon_max_val, lat_min_val, lat_max_val, _ = parse_lonlatbox(
            lonlatbox
        )
        mask = None
    elif mask_file is not None:
        lon_min_val, lon_max_val, lat_min_val, lat_max_val, mask = get_coords_from_mask(
            mask_file, mask_key=mask_var, resolutions=resolutions
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
