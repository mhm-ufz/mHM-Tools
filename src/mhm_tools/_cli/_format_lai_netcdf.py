"""
Format gridded LAI NetCDF data onto the DEM grid for mHM.

Authors
-------
- Sanjeev Bashyal
"""

from argparse import ArgumentParser, Namespace
from pathlib import Path

import click

from mhm_tools.common.cli_utils import (
    add_format_grid_args,
    add_netcdf_compression_args,
    get_netcdf_compression,
)


def add_args(parser: ArgumentParser) -> None:
    """Add CLI arguments for the ``format-lai-netcdf`` command."""
    required = parser.add_argument_group("required arguments")
    optional = parser.add_argument_group("optional arguments")
    add_format_grid_args(
        required, optional, input_help="Path to the gridded LAI NetCDF file."
    )
    optional.add_argument(
        "--output-temporal-resolution",
        choices=("daily", "monthly", "annual", "long-term-mean-monthly"),
        default="long-term-mean-monthly",
        help="Temporal resolution of the output. Default: long-term-mean-monthly.",
    )
    optional.add_argument(
        "--resampling",
        choices=("nearest", "bilinear"),
        default="bilinear",
        help="Spatial resampling method. Default: bilinear.",
    )
    add_netcdf_compression_args(parser)


def run(args: Namespace) -> None:
    """Format a gridded LAI NetCDF file onto the DEM grid.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed command-line arguments.
    """
    input_file = Path(args.input_file)
    if input_file.suffix.lower() != ".nc":
        msg = "--input-file must be a gridded LAI NetCDF (.nc) file."
        raise click.UsageError(msg)
    from mhm_tools.pre.format_lai import format_lai_netcdf_data

    format_lai_netcdf_data(
        input_file=input_file,
        dem_file=Path(args.dem_file),
        output_path=Path(args.output_path),
        output_temporal_resolution=args.output_temporal_resolution,
        input_crs=args.input_crs,
        dem_crs=args.dem_crs,
        resampling=args.resampling,
        compression=get_netcdf_compression(args),
    )
