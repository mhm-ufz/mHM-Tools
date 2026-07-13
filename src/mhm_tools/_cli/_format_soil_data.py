"""Format categorical soil data for use by mHM.

Authors
-------
- mHM-Tools Developers
"""

from argparse import ArgumentParser, Namespace
from pathlib import Path


def add_args(parser: ArgumentParser) -> None:
    """Add CLI arguments for the ``format-soil-data`` command."""
    required = parser.add_argument_group("required arguments")
    optional = parser.add_argument_group("optional arguments")
    required.add_argument(
        "-i",
        "--input-file",
        required=True,
        help="Path to the categorical soil raster.",
    )
    required.add_argument(
        "-o",
        "--output-path",
        required=True,
        help=(
            "Directory where soil_class.nc or soil_class.asc and "
            "soil_classdefinition.txt are written."
        ),
    )
    required.add_argument(
        "-l",
        "--lookup-table",
        required=True,
        help="Path to the soil lookup table.",
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


def run(args: Namespace) -> None:
    """Write a categorical soil raster and classdefinition from a lookup."""
    from mhm_tools.pre.format_soil import format_soil_data

    format_soil_data(
        input_file=Path(args.input_file),
        output_path=Path(args.output_path),
        lookup_table=Path(args.lookup_table),
        mapping_field=args.mapping_field,
        output_type=args.output_type,
    )
