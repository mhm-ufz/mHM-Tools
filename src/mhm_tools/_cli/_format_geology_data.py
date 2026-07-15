"""Format categorical geology data for use by mHM.

Authors
-------
- mHM-Tools Developers
"""

from argparse import ArgumentParser, Namespace
from pathlib import Path


def add_args(parser: ArgumentParser) -> None:
    """Add CLI arguments for the ``format-geology-data`` command."""
    required = parser.add_argument_group("required arguments")
    optional = parser.add_argument_group("optional arguments")
    required.add_argument(
        "-i",
        "--input-file",
        required=True,
        help="Path to the categorical geology raster.",
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
        help=(
            "Directory where geology_class.nc or geology_class.asc and "
            "geology_classdefinition.txt are written."
        ),
    )
    required.add_argument(
        "-l",
        "--lookup-table",
        required=True,
        help="Path to the geology lookup table.",
    )
    required.add_argument(
        "-m",
        "--mapping-field",
        required=True,
        help="Lookup-table column containing the input raster values.",
    )
    optional.add_argument(
        "-t",
        "--output-type",
        choices=("nc", "asc"),
        default="nc",
        help="Output file type. Default: nc.",
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


def run(args: Namespace) -> None:
    """Write a categorical geology raster and classdefinition from a lookup."""
    from mhm_tools.pre.format_geology import format_geology_data

    format_geology_data(
        input_file=Path(args.input_file),
        dem_file=Path(args.dem_file),
        output_path=Path(args.output_path),
        lookup_table=Path(args.lookup_table),
        mapping_field=args.mapping_field,
        output_type=args.output_type,
        input_crs=args.input_crs,
        dem_crs=args.dem_crs,
    )
