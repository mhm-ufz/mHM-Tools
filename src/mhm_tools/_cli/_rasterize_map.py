"""Rasterize vector map data on a reference DEM grid.

Authors
-------
- mHM-Tools Developers
"""

from argparse import ArgumentParser, Namespace
from pathlib import Path

import click


def add_args(parser: ArgumentParser) -> None:
    """Add CLI arguments for the ``rasterize-map`` command."""
    required = parser.add_argument_group("required arguments")
    optional = parser.add_argument_group("optional arguments")
    required.add_argument(
        "-i",
        "--input-file",
        required=True,
        help="Path to the input vector map.",
    )
    required.add_argument(
        "-d",
        "--dem-file",
        required=True,
        help="Reference DEM defining the output grid.",
    )
    required.add_argument(
        "-o",
        "--output-file",
        required=True,
        help="Output raster path; .asc, .nc, .tif, or .tiff selects the file format.",
    )
    required.add_argument(
        "-b",
        "--burn-field",
        required=True,
        help=(
            "Numeric vector attribute to rasterize directly, or numeric "
            "lookup-table column to burn."
        ),
    )
    optional.add_argument(
        "-l",
        "--lookup-table",
        help="Optional lookup table used to map vector categories.",
    )
    optional.add_argument(
        "-m",
        "--mapping-field",
        help=(
            "Category field shared by the input vector and lookup table. "
            "Required with --lookup-table."
        ),
    )
    optional.add_argument(
        "-s",
        "--input-crs",
        help="CRS to assign when the input vector has no CRS metadata.",
    )
    optional.add_argument(
        "-r",
        "--dem-crs",
        help="CRS to assign when the reference DEM has no CRS metadata.",
    )


def run(args: Namespace) -> None:
    """Rasterize direct numeric values or values mapped through a lookup."""
    from mhm_tools.pre.rasterize_map import rasterize_map_data

    if (args.lookup_table is None) != (args.mapping_field is None):
        msg = "Options --lookup-table and --mapping-field must be provided together."
        raise click.UsageError(msg)

    kwargs = {
        "input_file": Path(args.input_file),
        "dem_file": Path(args.dem_file),
        "output_file": Path(args.output_file),
        "mapping_field": args.mapping_field or args.burn_field,
    }
    if args.lookup_table is not None:
        kwargs.update(
            lookup_table=Path(args.lookup_table),
            lookup_mapping_field=args.mapping_field,
            lookup_value_field=args.burn_field,
        )
    if args.input_crs is not None:
        kwargs["input_crs"] = args.input_crs
    if args.dem_crs is not None:
        kwargs["dem_crs"] = args.dem_crs
    rasterize_map_data(**kwargs)
