"""mHM processing netCDF precipitation and temperature forcings"""

from ..pre.prepare_mhm_forcings import prepare_forcings

def add_args(parser):
    """Add cli arguments for prepare_mhm_forcings subcommand.

    Parameters
    ----------
    parser : argparse.ArgumentParser
        the main argument parser
    """
    # Description and help for the subcommand
    parser.description = (
        "Utility functions for mHM processing netCDF precipitation and temperature forcings, "
        "including unit conversion, coordinate ordering, setting correct variable name and " 
        "units as well as missing and fill vaules, and it also includes spatial cropping."
    )
    parser.epilog = (
        "Example:"
        "  mhm-tools prepare_mhm_forcings -i in/data -f input_*.nc -o out/data -u processed.nc -v 2t --crop "
        "--lon-min 0 --lon-max 10 --lat-min -5 --lat-max 5"
    )
    # Positional and optional arguments
    parser.add_argument('-i', '--in-dir', required=True,
                        help='Input directory containing forcing NetCDF files')
    parser.add_argument('-f', '--in-file', required=True,
                        help='Input filename or glob pattern (e.g. "data_*.nc")')
    parser.add_argument('-o', '--out-dir', required=True,
                        help='Output directory for processed files')
    parser.add_argument('-u', '--out-file', default='*',
                        help=('Output filename or pattern. Use "*" to retain ' \
                              'input basename; otherwise literal name for single file.'))
    parser.add_argument('-v', '--var', required=True,
                        help='Variable name to convert (e.g. 2t, tp, tprate)')
    parser.add_argument('--crop', action='store_true',
                        help='Enable spatial cropping of the dataset')
    parser.add_argument('--lon-min', type=float,
                        help='Minimum longitude for cropping')
    parser.add_argument('--lon-max', type=float,
                        help='Maximum longitude for cropping')
    parser.add_argument('--lat-min', type=float,
                        help='Minimum latitude for cropping')
    parser.add_argument('--lat-max', type=float,
                        help='Maximum latitude for cropping')


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
        lat_max=args.lat_max
    )