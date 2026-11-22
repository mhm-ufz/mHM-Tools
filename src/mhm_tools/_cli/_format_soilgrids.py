"""
Format physical soil horizon layers (clay, sand, silt, bulk density) for mHM.

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

EPILOG = """The manifest is a plain comma-separated file with a header row; column names are matched case- and punctuation-insensitively. Relative paths resolve against the folder holding the manifest.

\b
  Horizon,Upper Depth,Lower Depth,Clay Layer,Sand Layer,Silt Layer,Bulk Density Layer,Bulk Density Unit
  1,0,100,clay1.tif,sand1.tif,silt1.tif,bd1.tif,kg/m3
  2,100,300,clay2.tif,sand2.tif,silt2.tif,bd2.tif,kg/m3

Horizons are numbered from 1 without gaps, the first starts at depth 0, and each Lower Depth is the next Upper Depth. Depths are in mm. Bulk Density Unit must be identical in every row, one of: g/cm3, kg/m3, cg/cm3, mg/cm3, g/dm3, kg/dm3. -e asc writes soil_class.asc plus soil_classdefinition.txt; -e nc writes soil_horizon_class.nc plus soil_classdefinition_iFlag_soilDB_1.txt.
"""


def add_args(parser: ArgumentParser) -> None:
    """Add CLI arguments for the ``format-soilgrids`` command."""
    required = parser.add_argument_group("required arguments")
    optional = parser.add_argument_group("optional arguments")
    add_format_grid_args(
        required,
        optional,
        input_help=(
            "Path to the CSV/TXT soil horizon manifest described at the end of "
            "this help."
        ),
    )
    optional.add_argument(
        "-e",
        "--output-extension",
        "--extension",
        dest="extension",
        choices=("nc", "asc"),
        default="nc",
        help="Output raster extension. Default: nc.",
    )
    optional.add_argument(
        "--resampling",
        choices=("auto", "nearest", "bilinear", "cubic", "average", "mode"),
        default="auto",
        help=(
            "Spatial resampling method. 'auto' uses average/bilinear for the "
            "continuous soil layers. Default: auto."
        ),
    )
    optional.add_argument(
        "--composition-step",
        type=float,
        default=5.0,
        help="Composition class interval. Default: 5.0.",
    )
    optional.add_argument(
        "--bulkdensity-step",
        type=float,
        default=0.1,
        help="Bulk-density class interval. Default: 0.1.",
    )
    optional.add_argument(
        "--no-fill-nodata",
        dest="fill_nodata",
        action="store_false",
        help=(
            "Keep input nodata gaps instead of taking them from their nearest "
            "valid neighbour. Gaps are filled per input layer, so a hole in one "
            "of clay, sand, silt, or bulk density no longer drops the cell from "
            "every horizon."
        ),
    )
    add_netcdf_compression_args(parser)


def run(args: Namespace) -> None:
    """Format the soil horizon layers of a manifest onto the DEM grid.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed command-line arguments.
    """
    input_file = Path(args.input_file)
    if input_file.suffix.lower() not in {".csv", ".txt"}:
        msg = "--input-file must be a CSV or TXT soil horizon manifest."
        raise click.UsageError(msg)
    from mhm_tools.pre.format_soil import format_soil_horizons

    format_soil_horizons(
        input_file=input_file,
        dem_file=Path(args.dem_file),
        output_path=Path(args.output_path),
        output_type=args.extension,
        input_crs=args.input_crs,
        dem_crs=args.dem_crs,
        resampling=args.resampling,
        composition_step=args.composition_step,
        bulkdensity_step=args.bulkdensity_step,
        fill_nodata=args.fill_nodata,
        compression=get_netcdf_compression(args),
    )
