"""Create DEM derivatives (filled DEM, slope, aspect, facc, fdir) from a DEM.

The tool creates a filled DEM, slope, aspect, flow accumulation, and flow direction from a DEM.
With --input-type fdir it creates only the flow accumulation from an existing flow direction grid,
so facc follows the same network as the fdir of an mHM setup, also after upscaling.
The output can be written to a single netCDF file, or to one file per derivative in netCDF, ASCII or GeoTIFF format.

Authors
-------
- Sanjeev Bashyal
"""

from mhm_tools.common.cli_utils import (
    add_netcdf_compression_args,
    get_netcdf_compression,
)
from mhm_tools.common.constants import (
    DEM_DERIVATIVE_EXTENSIONS,
    FACC_UNITS,
    FDIR_CODES,
)


def add_args(parser):
    """Add CLI arguments for the create-dem-derivatives subcommand."""
    required = parser.add_argument_group("required arguments")
    optional = parser.add_argument_group("optional arguments")
    required.add_argument(
        "-i",
        "--input-file",
        required=True,
        help="Path to the input DEM, or flow direction with --input-type fdir (.nc, .asc, .tif).",
    )
    required.add_argument(
        "-o",
        "--output-path",
        required=True,
        help=(
            "Output file for netcdf, or output directory for one file per "
            "derivative (asc, tif or --one-file-per-variable). With --input-type "
            "fdir a directory receives facc.nc, facc.asc or facc.tif."
        ),
    )
    optional.add_argument(
        "-e",
        "--output-extension",
        default="nc",
        choices=list(DEM_DERIVATIVE_EXTENSIONS),
        help=(
            "nc writes one file holding every derivative plus grid and crs; "
            "asc and tif write one file per derivative. Default: nc."
        ),
    )
    optional.add_argument(
        "--one-file-per-variable",
        action="store_true",
        help=(
            "With nc, write one file per derivative named after it (dem.nc, "
            "dem_filled.nc, slope.nc, aspect.nc, facc.nc, fdir.nc) into the "
            "--output-path directory, each with lat, lon and crs."
        ),
    )
    optional.add_argument(
        "--input-type",
        default="dem",
        choices=["dem", "fdir"],
        help=(
            "dem derives every derivative from a DEM; fdir derives only facc from "
            "an existing flow direction grid. Default: dem."
        ),
    )
    optional.add_argument(
        "--ftype",
        default="d8",
        choices=list(FDIR_CODES),
        help="Flow direction type of the input with --input-type fdir. Default: d8.",
    )
    optional.add_argument(
        "--facc-unit",
        default="cell",
        choices=list(FACC_UNITS),
        help="Unit of facc: number of upstream cells or upstream area in km2. Default: cell.",
    )
    optional.add_argument(
        "--crs",
        default=None,
        help="CRS to use when the input has none (e.g. 'EPSG:4326').",
    )
    optional.add_argument(
        "-v",
        "--input-var",
        "--varname",
        dest="varname",
        default=None,
        help=(
            "Elevation or flow direction variable to read when the input holds "
            "several (create-catchment names its flow direction 'flwdir')."
        ),
    )
    add_netcdf_compression_args(parser)


def run(args):
    """Write the DEM derivatives for the given input file."""
    import logging

    from mhm_tools.pre.dem_derivatives import (
        create_dem_derivatives,
        create_facc_from_fdir,
    )

    logger = logging.getLogger(__name__)
    if args.input_type == "fdir":
        written = create_facc_from_fdir(
            args.input_file,
            args.output_path,
            output_extension=args.output_extension,
            crs=args.crs,
            var_name=args.varname,
            compression=get_netcdf_compression(args),
            ftype=args.ftype,
            facc_unit=args.facc_unit,
        )
    else:
        written = create_dem_derivatives(
            args.input_file,
            args.output_path,
            output_extension=args.output_extension,
            crs=args.crs,
            var_name=args.varname,
            compression=get_netcdf_compression(args),
            one_file_per_variable=args.one_file_per_variable,
            facc_unit=args.facc_unit,
        )
    for path in written:
        logger.info(f"Wrote {path}")
