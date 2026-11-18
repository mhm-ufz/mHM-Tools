"""
Format categorical soil, geology, LAI, or land-cover rasters for mHM via a lookup table.

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

EPILOG = """Manifest input (-i pointing at a .csv or .txt file) formats several land-cover periods in one run and is accepted only for -t lc. It is a plain comma-separated file with a header row; column names are matched case- and punctuation-insensitively. Relative paths resolve against the folder holding the manifest.

\b
  StartDateTime,EndDateTime,FilePath
  2000-01-01T00:00:00,2005-07-01T12:00:00,landcover_2000.tif
  2005-07-01T12:00:00,2010-01-01T00:00:00,landcover_2005.tif

Periods use inclusive start and exclusive end boundaries. They must be ordered without gaps or overlaps: each StartDateTime equals the previous EndDateTime. Year-only values are accepted. Legacy StartYear/EndYear manifests remain supported with inclusive end years. -e asc writes one file per period; -e nc writes a single lc_periods.nc.
"""


def add_args(parser: ArgumentParser) -> None:
    """Add CLI arguments for the ``format-morph-categorical`` command."""
    required = parser.add_argument_group("required arguments")
    optional = parser.add_argument_group("optional arguments")
    required.add_argument(
        "-t",
        "--type",
        dest="data_type",
        required=True,
        choices=("soil", "geology", "lai", "lc"),
        help="Data type to format.",
    )
    add_format_grid_args(
        required,
        optional,
        input_help=(
            "Path to one input raster, or for -t lc a CSV/TXT manifest of "
            "land-cover periods described at the end of this help."
        ),
    )
    required.add_argument(
        "-l",
        "--lookup-table",
        required=True,
        help="Path to the categorical lookup table.",
    )
    required.add_argument(
        "-m",
        "--mapping-field",
        required=True,
        help="Lookup-table column containing the input raster values.",
    )
    required.add_argument(
        "-c",
        "--class-field",
        required=True,
        help="Lookup-table column containing the output class values.",
    )
    optional.add_argument(
        "-e",
        "--output-extension",
        "--extension",
        dest="extension",
        choices=("nc", "asc", "tif"),
        default="nc",
        help="Output raster extension. Default: nc.",
    )
    optional.add_argument(
        "--resampling",
        choices=("auto", "nearest", "bilinear", "cubic", "average", "mode"),
        help=(
            "Spatial resampling method. Default: nearest for soil, geology and "
            "lai, auto (majority/nearest for classes) for lc."
        ),
    )
    optional.add_argument(
        "--no-fill-nodata",
        dest="fill_nodata",
        action="store_false",
        help=(
            "Keep input nodata gaps instead of taking them from their nearest "
            "valid neighbour."
        ),
    )
    add_netcdf_compression_args(parser)


def run(args: Namespace) -> None:
    """Format one categorical raster, or a land-cover manifest, onto the DEM grid.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed command-line arguments.
    """
    input_file = Path(args.input_file)
    if input_file.is_dir():
        msg = "--input-file must be a raster, CSV, or TXT file, not a directory."
        raise click.UsageError(msg)
    if input_file.suffix.lower() in {".csv", ".txt"}:
        if args.data_type != "lc":
            msg = (
                "Manifest inputs are supported only for --type lc. "
                "Use format-soilgrids for soil horizon manifests."
            )
            raise click.UsageError(msg)
        from mhm_tools.pre.format_lc_data import format_lc_periods as formatter
    elif args.data_type == "soil":
        from mhm_tools.pre.format_soil import format_soil_data as formatter
    elif args.data_type == "geology":
        from mhm_tools.pre.format_geology import format_geology_data as formatter
    elif args.data_type == "lai":
        from mhm_tools.pre.format_lai import format_lai_data as formatter
    else:
        from mhm_tools.pre.format_lc_data import format_lc_data as formatter

    # formatters differ in their default resampling, so pass it only when given
    kwargs = {} if args.resampling is None else {"resampling": args.resampling}
    formatter(
        input_file=input_file,
        dem_file=Path(args.dem_file),
        output_path=Path(args.output_path),
        lookup_table=Path(args.lookup_table),
        mapping_field=args.mapping_field,
        class_field=args.class_field,
        output_type=args.extension,
        input_crs=args.input_crs,
        dem_crs=args.dem_crs,
        fill_nodata=args.fill_nodata,
        compression=get_netcdf_compression(args),
        **kwargs,
    )
