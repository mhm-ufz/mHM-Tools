"""
Format categorical soil, geology, or land-cover data for use by mHM.

Authors
-------
- Sanjeev Bashyal
"""

from argparse import ArgumentParser, Namespace
from pathlib import Path

import click

_MANIFEST_SUFFIXES = {".csv", ".txt"}


def add_args(parser: ArgumentParser) -> None:
    """Add CLI arguments for the ``format-data`` command."""
    required = parser.add_argument_group("required arguments")
    optional = parser.add_argument_group("optional arguments")
    required.add_argument(
        "-t",
        "--type",
        dest="data_type",
        required=True,
        choices=("soil", "geology", "lc"),
        help="Categorical data type to format.",
    )
    required.add_argument(
        "-i",
        "--input-file",
        dest="input_file",
        required=True,
        help="Path to one input raster or a CSV/TXT manifest file.",
    )
    required.add_argument(
        "-d",
        "--dem-file",
        required=True,
        help="DEM providing the output grid and coordinate reference system.",
    )
    required.add_argument(
        "-o",
        "--output-path",
        required=True,
        help="Directory where the formatted output files are written.",
    )
    optional.add_argument(
        "-l",
        "--lookup-table",
        help="Path to the categorical lookup table.",
    )
    optional.add_argument(
        "-m",
        "--mapping-field",
        help="Lookup-table column containing the input raster values.",
    )
    optional.add_argument(
        "-c",
        "--class-field",
        help="Lookup-table column containing the output class values.",
    )
    optional.add_argument(
        "-e",
        "--extension",
        choices=("nc", "asc", "tif"),
        default="nc",
        help="Output raster extension. Default: nc.",
    )
    optional.add_argument(
        "-s",
        "--input-crs",
        help="CRS to assign when the input raster has no CRS metadata.",
    )
    optional.add_argument(
        "-r",
        "--dem-crs",
        help="CRS to assign when the DEM has no CRS metadata.",
    )
    optional.add_argument(
        "--resampling",
        choices=("auto", "nearest", "bilinear", "cubic", "average", "mode"),
        help=(
            "Spatial resampling method. 'auto' uses majority/nearest for "
            "classes and average/bilinear for continuous soil layers."
        ),
    )
    optional.add_argument(
        "--no-fill-nodata",
        dest="fill_nodata",
        action="store_false",
        help=(
            "Keep input nodata gaps instead of taking them from their nearest "
            "valid neighbour. Soil gaps are filled per input layer, so a hole "
            "in one of clay, sand, silt, or bulk density no longer drops the "
            "cell from every horizon."
        ),
    )


def _require_lookup_options(args: Namespace) -> None:
    """Require lookup options only for categorical input workflows."""
    missing = [
        option
        for option, value in (
            ("--lookup-table", args.lookup_table),
            ("--mapping-field", args.mapping_field),
            ("--class-field", args.class_field),
        )
        if value is None
    ]
    if missing:
        names = ", ".join(missing)
        msg = f"Missing required option(s): {names}"
        raise click.UsageError(msg)


def run(args: Namespace) -> None:
    """Dispatch categorical formatting to the selected data formatter."""
    input_file = Path(args.input_file)
    if input_file.is_dir():
        msg = "--input-file must be a raster, CSV, or TXT file, not a directory."
        raise click.UsageError(msg)
    is_manifest_input = input_file.suffix.lower() in _MANIFEST_SUFFIXES

    if is_manifest_input and args.data_type == "soil":
        from mhm_tools.pre.format_soil import format_soil_horizons as formatter
    elif is_manifest_input and args.data_type == "lc":
        _require_lookup_options(args)
        from mhm_tools.pre.format_lc_data import format_lc_periods as formatter
    elif is_manifest_input:
        msg = "Manifest inputs are supported only for --type lc or soil."
        raise click.UsageError(msg)
    elif args.data_type == "soil":
        _require_lookup_options(args)
        from mhm_tools.pre.format_soil import format_soil_data as formatter
    elif args.data_type == "geology":
        _require_lookup_options(args)
        from mhm_tools.pre.format_geology import format_geology_data as formatter
    else:
        _require_lookup_options(args)
        from mhm_tools.pre.format_lc_data import format_lc_data as formatter

    kwargs = {
        "input_file": input_file,
        "dem_file": Path(args.dem_file),
        "output_path": Path(args.output_path),
        "output_type": args.extension,
        "input_crs": args.input_crs,
        "dem_crs": args.dem_crs,
    }
    if args.data_type != "soil" or not is_manifest_input:
        kwargs.update(
            lookup_table=Path(args.lookup_table),
            mapping_field=args.mapping_field,
            class_field=args.class_field,
        )
    if args.resampling is not None:
        kwargs["resampling"] = args.resampling
    kwargs["fill_nodata"] = args.fill_nodata
    formatter(**kwargs)
