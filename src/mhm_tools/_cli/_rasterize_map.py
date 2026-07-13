"""Rasterize vector map data on a reference DEM grid.

Authors
-------
- mHM-Tools Developers
"""

from argparse import ArgumentParser, Namespace
from pathlib import Path


def add_args(parser: ArgumentParser) -> None:
    """Add CLI arguments for the ``rasterize-map`` command."""
    required = parser.add_argument_group("required arguments")
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
        help="Path to the output GeoTIFF.",
    )
    required.add_argument(
        "-m",
        "--mapping-field",
        required=True,
        help="Numeric vector attribute to rasterize.",
    )


def run(args: Namespace) -> None:
    """Rasterize a numeric vector field on the reference DEM grid."""
    from mhm_tools.pre.rasterize_map import rasterize_map_data

    rasterize_map_data(
        input_file=Path(args.input_file),
        dem_file=Path(args.dem_file),
        output_file=Path(args.output_file),
        mapping_field=args.mapping_field,
    )
