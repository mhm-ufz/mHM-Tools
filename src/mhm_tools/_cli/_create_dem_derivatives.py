"""Create DEM derivatives (filled DEM, slope, aspect, facc, fdir) from a DEM."""

from mhm_tools.pre.dem_derivatives import OUTPUT_EXTENSIONS


def add_args(parser):
    """Add CLI arguments for the create-dem-derivatives subcommand."""
    required = parser.add_argument_group("required arguments")
    optional = parser.add_argument_group("optional arguments")
    required.add_argument(
        "-i",
        "--input-file",
        required=True,
        help="Path to the input DEM (.nc, .asc, .tif).",
    )
    required.add_argument(
        "-o",
        "--output-path",
        required=True,
        help=(
            "Output file for netcdf, or output directory for one file per "
            "derivative."
        ),
    )
    optional.add_argument(
        "-e",
        "--output-extension",
        default="nc",
        choices=list(OUTPUT_EXTENSIONS),
        help=(
            "nc writes one file holding every derivative plus grid and crs; "
            "asc and tif write one file per derivative. Default: nc."
        ),
    )
    optional.add_argument(
        "--crs",
        default=None,
        help="CRS to use when the input has none (e.g. 'EPSG:4326').",
    )
    optional.add_argument(
        "--varname",
        default=None,
        help="Elevation variable to read when the input holds several.",
    )


def run(args):
    """Write the DEM derivatives for the given input file."""
    import logging

    from mhm_tools.pre.dem_derivatives import create_dem_derivatives

    logger = logging.getLogger(__name__)
    written = create_dem_derivatives(
        args.input_file,
        args.output_path,
        output_extension=args.output_extension,
        crs=args.crs,
        var_name=args.varname,
    )
    for path in written:
        logger.info(f"Wrote {path}")
