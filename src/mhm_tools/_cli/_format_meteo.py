"""
Format meteorological or observation NetCDF files onto an mHM grid.

Authors
-------
- Sanjeev Bashyal
"""

from argparse import ArgumentParser, Namespace
from pathlib import Path

import click

from mhm_tools.common.cli_utils import (
    add_netcdf_compression_args,
    get_netcdf_compression,
)

RESOLUTIONS = ("hourly", "daily", "monthly", "yearly")


def add_args(parser: ArgumentParser) -> None:
    """Add CLI arguments for the ``format-meteo`` command."""
    from mhm_tools.pre.format_meteo import VARIABLES

    required = parser.add_argument_group("required arguments")
    optional = parser.add_argument_group("optional arguments")
    required.add_argument(
        "-i",
        "--input-file",
        required=True,
        help="One input NetCDF file; the glob is matched in its folder.",
    )
    required.add_argument(
        "-g",
        "--glob",
        required=True,
        help="Glob of the NetCDF files to read, relative to the input file's folder.",
    )
    required.add_argument(
        "-v", "--input-var", required=True, help="Variable to read from the input."
    )
    required.add_argument(
        "--output-var",
        required=True,
        choices=tuple(VARIABLES),
        help="mHM variable to write; it sets the output unit and file name.",
    )
    required.add_argument(
        "--input-temporal-resolution",
        required=True,
        choices=RESOLUTIONS,
        help="Temporal resolution of the input.",
    )
    required.add_argument(
        "--output-temporal-resolution",
        required=True,
        choices=RESOLUTIONS,
        help="Temporal resolution of the output; not finer than the input.",
    )
    required.add_argument(
        "-u", "--input-unit", required=True, help="Unit of the input variable."
    )
    required.add_argument(
        "-s",
        "--input-crs",
        required=True,
        help="CRS of the input coordinates, used when the files carry none.",
    )
    required.add_argument(
        "-d",
        "--grid-file",
        required=True,
        help="ASCII grid header of the target (L2) grid.",
    )
    required.add_argument(
        "-r", "--grid-crs", required=True, help="CRS of the target grid."
    )
    required.add_argument(
        "-o",
        "--output-dir",
        required=True,
        help="Directory for <output-var>.nc and header.txt.",
    )
    optional.add_argument(
        "--aggregation",
        choices=("average", "sum", "minimum", "maximum"),
        help="Aggregation to a coarser output temporal resolution.",
    )
    optional.add_argument(
        "--cumulative-resolution",
        choices=RESOLUTIONS,
        help="The input accumulates and resets at this resolution (e.g. ERA5-Land).",
    )
    optional.add_argument(
        "--cumulative-reset",
        type=int,
        help="Last hour, day or month an accumulation covers. Default: the last one.",
    )
    optional.add_argument(
        "--clip-negative",
        action="store_true",
        help="Clip small negative amounts (rounding noise) to 0 instead of stopping.",
    )
    optional.add_argument(
        "--start",
        help="First time to keep. Default: the first complete output period.",
    )
    optional.add_argument(
        "--end",
        help=(
            "Exclusive end of the time to keep, e.g. 2002-01-01 to keep 2001. "
            "Default: the end of the last complete output period."
        ),
    )
    add_netcdf_compression_args(parser)


def run(args: Namespace) -> None:
    """Format the input files onto the target grid.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed command-line arguments.
    """
    if not Path(args.input_file).is_file():
        msg = f"--input-file is not a file: {args.input_file}"
        raise click.UsageError(msg)
    from mhm_tools.common.esri_grid import read_header, write_header
    from mhm_tools.pre.format_meteo import MeteoFolderSpec, format_meteo_data

    spec = MeteoFolderSpec(
        kind=args.output_var,
        folder=Path(args.input_file).parent,
        crs=args.input_crs,
        pattern=args.glob,
        temporal_resolution=args.input_temporal_resolution.capitalize(),
        output_temporal_resolution=args.output_temporal_resolution.capitalize(),
        aggregation=(args.aggregation or "").capitalize(),
        unit=args.input_unit,
        cumulative=args.cumulative_resolution is not None,
        cumulative_resolution=(args.cumulative_resolution or "").capitalize(),
        cumulative_reset=args.cumulative_reset,
        variable=args.input_var,
        clip_negative=args.clip_negative,
    )
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    format_meteo_data(
        spec,
        output_dir / f"{args.output_var}.nc",
        Path(args.grid_file),
        grid_crs=args.grid_crs,
        start_time=args.start,
        end_time=args.end,
        compression=get_netcdf_compression(args),
    )
    write_header(output_dir / "header.txt", read_header(args.grid_file))
