

from mhm_tools.common.cli_utils import get_coords
from mhm_tools.post.seasonality_grid_validation import seasonality_grid_validation


def add_args(parser):
    """Add cli arguments for the seasonality validation.

    Parameters
    ----------
    parser : argparse.ArgumentParser
        the main argument parser
    """
    parser.add_argument("--input_file", help="Path to the input file.", required=True)
    parser.add_argument("--input_variable", help="Variable name in the input file.", required=False)
    parser.add_argument("--input_name", help="Name of the input dataset.", default=None, required=False)
    parser.add_argument("--input_factor", help="Unit Conversion factor. e.g. MJ/kg/day:  1 / 2.47 = 0.4", default=1, required=False)
    parser.add_argument("--output_dir", help="Path for the output dir.", required=True)
    parser.add_argument("--ref_file", help="Path to the first reference file.", default=None, required=False)
    parser.add_argument("--ref_name", help="Name of the reference dataset.", default=None, required=False)
    parser.add_argument("--ref_factor", help="Unit Conversion factor. e.g. MJ/kg/day:  1 / 2.47 = 0.4", default=1, required=False)
    parser.add_argument("--ref_variable", help="Variable name in the first reference file.", default=None, required=False)
    parser.add_argument("--only_plot", help="Set Flag if existing output file should be used to create plot", action="store_true", required=False)
    parser.add_argument(
        "--lonlatbox",
        required=False,
        default=None,
        help=(
            """coordinates in the form of 'lon_min,lon_max,lat_min,lat_max,resolution_l0'
            required unless --mask_file is provided"""
        ),
    )
    parser.add_argument(
        "--mask_file",
        required=False,
        default=None,
        help=(
            """path to the mask file, a .nc file with a variable 'mask' containing the grid mask at l0 resolution
            required unless --lonlatbox is provided"""
        ),
    )
    parser.add_argument(
        "--ncpus",
        default=1,
        type=int,
        help=("Number of CPUs to use"),
    )
def run(args):
    """Calculate the validation.

    Parameters
    ----------
    args : argparse.Namespace
        parsed command line arguments
    """
    lon_min, lon_max, lat_min, lat_max, mask = get_coords(args.lonlatbox, args.mask_file, raise_exception=False)
    coordinate_slice = None
    if lon_min is not None and lon_max is not None and lat_min is not None and lat_max is not None:
        coordinate_slice = {'lat': slice(lat_max, lat_min), 'lon': slice(lon_min, lon_max)}
    seasonality_grid_validation(args.input_file, args.input_variable, args.output_dir, args.ref_file, args.ref_variable, args.input_name, args.ref_name, float(args.input_factor), float(args.ref_factor), args.only_plot, coordinate_slice, args.ncpus)
