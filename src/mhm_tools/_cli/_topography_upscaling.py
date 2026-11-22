"""Upscale a DEM and/or a flow direction grid to a coarser grid for mHM.

The tool averages the DEM per coarse cell, averages the slope calculated on the
input DEM, calculates the aspect on the upscaled DEM, and upscales the flow
direction with the outlet rule of mHM: each coarse cell drains where the cell
with the largest flow accumulation of its block drains to. The coarse grid keeps
the lower-left corner of the input grid, whose rows and columns must be integer
multiples of the upscale factor. Without a flow direction it is derived from the
depression-filled DEM. With a flow direction, a flow accumulation (e.g. MERIT
Hydro dir and upa) and --available-mem, every step runs in chunks, so global
grids fit into the memory budget.

Authors
-------
- Simon Lüdke
"""

from mhm_tools.common.cli_utils import (
    add_missing_aspect_arg,
    add_netcdf_compression_args,
    get_netcdf_compression,
)
from mhm_tools.common.constants import (
    DEM_DERIVATIVE_EXTENSIONS,
    FACC_UNITS,
    FDIR_CODE_BY_OFFSET,
)


def add_args(parser):
    """Add CLI arguments for the topography-upscaling subcommand.

    Args:
        parser (argparse.ArgumentParser): The subcommand parser.
    """
    required = parser.add_argument_group("required arguments")
    inputs = parser.add_argument_group(
        "input files (at least one of --dem-file and --fdir-file)"
    )
    target = parser.add_argument_group(
        "target grid (exactly one of --upscale-factor and --target-resolution)"
    )
    optional = parser.add_argument_group("optional arguments")
    flags = parser.add_argument_group("flags")
    required.add_argument(
        "-o",
        "--output-path",
        required=True,
        help=(
            "Output file for nc, or output directory for one file per variable "
            "(asc, tif or --one-file-per-variable)."
        ),
    )
    inputs.add_argument(
        "--dem-file",
        default=None,
        help="DEM (.nc, .asc, .tif). Gives the dem, slope and aspect output.",
    )
    inputs.add_argument(
        "--fdir-file",
        default=None,
        help="Flow direction grid. Without it the flow direction is derived from the DEM.",
    )
    inputs.add_argument(
        "--facc-file",
        default=None,
        help=(
            "Flow accumulation that belongs to --fdir-file, e.g. MERIT Hydro upa. "
            "It picks the outlets instead of a flow accumulation computed from the "
            "flow direction, which lets the flow direction run in chunks. The facc "
            "output copies its values and unit."
        ),
    )
    inputs.add_argument("--dem-var", default=None, help="Variable of --dem-file.")
    inputs.add_argument("--fdir-var", default=None, help="Variable of --fdir-file.")
    inputs.add_argument("--facc-var", default=None, help="Variable of --facc-file.")
    target.add_argument(
        "--upscale-factor",
        type=int,
        default=None,
        help="Number of input cells per coarse cell along each axis.",
    )
    target.add_argument(
        "--target-resolution",
        type=float,
        default=None,
        help="Coarse cell size, an integer multiple of the input cell size.",
    )
    optional.add_argument(
        "--facc-unit",
        default="km2",
        choices=list(FACC_UNITS),
        help=(
            "Unit of a computed facc output: upstream area in km2 or number of "
            "upstream input cells. A facc from --facc-file keeps its unit. "
            "Default: km2."
        ),
    )
    add_missing_aspect_arg(optional)
    optional.add_argument(
        "--ftype",
        default="d8",
        choices=list(FDIR_CODE_BY_OFFSET),
        help="Flow direction type of --fdir-file and of the output. Default: d8.",
    )
    optional.add_argument(
        "-e",
        "--output-extension",
        default="nc",
        choices=list(DEM_DERIVATIVE_EXTENSIONS),
        help=(
            "nc writes one file holding every variable plus grid and crs; asc and "
            "tif write one file per variable. Default: nc."
        ),
    )
    optional.add_argument(
        "--crs",
        default=None,
        help="CRS to use when the inputs have none (e.g. 'EPSG:4326').",
    )
    optional.add_argument(
        "--available-mem",
        "--max-memory",
        dest="available_mem",
        default=None,
        help=(
            "Memory budget in Gb, or with a unit such as 500mb. Turns on processing "
            "in chunks; flow routing still needs the whole input grid in memory."
        ),
    )
    optional.add_argument(
        "--ncpus",
        type=int,
        default=1,
        help="Cores for processing in chunks. 0 or -1 uses all cores. Default: 1.",
    )
    flags.add_argument(
        "--depression-fill-dem",
        action="store_true",
        help=(
            "Fill the depressions of the DEM and derive dem, slope and aspect from "
            "the filled DEM. Not needed for a hydrologically conditioned DEM such as "
            "MERIT Hydro elv."
        ),
    )
    flags.add_argument(
        "--write-facc",
        action="store_true",
        help="Also write the flow accumulation of the outlet of each coarse cell.",
    )
    flags.add_argument(
        "--keep-edge-outflow",
        action="store_true",
        help=(
            "Keep the direction of outlets that drain off the grid edge instead of "
            "making them sinks, so tiles upscaled one by one can be mosaicked."
        ),
    )
    flags.add_argument(
        "--one-file-per-variable",
        action="store_true",
        help=(
            "With nc, write one file per variable named after it into the "
            "--output-path directory, each with lat, lon and crs."
        ),
    )
    add_netcdf_compression_args(parser)


def run(args):
    """Upscale the topography given on the command line.

    Args:
        args (argparse.Namespace): Parsed command line arguments.
    """
    import logging

    from mhm_tools.common.cli_utils import get_available_mem_in_unit
    from mhm_tools.pre.topography_upscaling import upscale_topography

    logger = logging.getLogger(__name__)
    written = upscale_topography(
        output_path=args.output_path,
        dem_file=args.dem_file,
        fdir_file=args.fdir_file,
        facc_file=args.facc_file,
        upscale_factor=args.upscale_factor,
        target_resolution=args.target_resolution,
        depression_fill_dem=args.depression_fill_dem,
        write_facc=args.write_facc,
        facc_unit=args.facc_unit,
        keep_edge_outflow=args.keep_edge_outflow,
        missing_aspect=args.missing_aspect,
        ftype=args.ftype,
        output_extension=args.output_extension,
        one_file_per_variable=args.one_file_per_variable,
        crs=args.crs,
        dem_var=args.dem_var,
        fdir_var=args.fdir_var,
        facc_var=args.facc_var,
        compression=get_netcdf_compression(args),
        available_mem_gib=get_available_mem_in_unit(args.available_mem),
        ncpus=args.ncpus,
    )
    for path in written:
        logger.info(f"Wrote {path}")
