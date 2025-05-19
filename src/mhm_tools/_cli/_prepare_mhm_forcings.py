"""mHM processing netCDF precipitation and temperature forcings"""

from mhm_tools.pre.prepare_mhm_forcings import prepare_forcings

def add_args(parser):
    """Add CLI arguments for prepare_mhm_forcings subcommand.

    Utility functions for mHM processing of netCDF precipitation and temperature forcings,
    including unit conversion, coordinate ordering, cropping, and setting correct metadata.

    Example:
      mhm-tools prepare_mhm_forcings \
        -i in/data -f input_*.nc -o out/data -v 2t --out-file processed.nc \
        --crop --lon-min 0 --lon-max 10 --lat-min -5 --lat-max 5
    """
    # Description and epilog
    parser.description = (
        "Utility functions for mHM processing netCDF precipitation and temperature forcings, "
        "including unit conversion, coordinate ordering, setting correct variable name and "
        "units as well as missing and fill values, and spatial cropping."
    )
    parser.epilog = (
        "See mhm-tools documentation for detailed examples and usage."
    )

    # Required arguments
    required = parser.add_argument_group("required arguments")
    required.add_argument(
        '-i', '--in-dir', required=True,
        help='Input directory containing forcing NetCDF files'
    )
    required.add_argument(
        '-f', '--in-file', required=True,
        help='Input filename or glob pattern (e.g. "data_*.nc")'
    )
    required.add_argument(
        '-o', '--out-dir', required=True,
        help='Output directory for processed files'
    )
    required.add_argument(
        '-v', '--var', required=True,
        help='Variable name to convert: 2t (temperature), tp (total precipitation), tprate (precipitation rate)'
    )

    # Optional arguments
    parser.add_argument(
        '-u', '--out-file', default='*',
        help=(
            "Output filename or pattern. Use '*' to retain input basename; "
            "otherwise literal name for each output file."
        )
    )
    parser.add_argument(
        '--crop', action='store_true',
        help='Enable spatial cropping of the dataset'
    )
    parser.add_argument('--lon-min', type=float, help='Minimum longitude for cropping')
    parser.add_argument('--lon-max', type=float, help='Maximum longitude for cropping')
    parser.add_argument('--lat-min', type=float, help='Minimum latitude for cropping')
    parser.add_argument('--lat-max', type=float, help='Maximum latitude for cropping')
    parser.add_argument(
        '--use-mfdataset',  type=bool, default=False,
        help='Use xarray.open_mfdataset for multi-file datasets'
    )


def run(args):
    """
    Run script to convert input forcings into the right mHM format.

    Parameters
    ----------
    args : argparse.Namespace
        parsed command line arguments

    """
    prepare_forcings(
        in_dir=args.in_dir,
        in_file=args.in_file,
        out_dir=args.out_dir,
        out_file=args.out_file,
        var=args.var,
        crop=args.crop,
        lon_min=args.lon_min,
        lon_max=args.lon_max,
        lat_min=args.lat_min,
        lat_max=args.lat_max,
        use_mfdataset=args.use_mfdataset,
    )