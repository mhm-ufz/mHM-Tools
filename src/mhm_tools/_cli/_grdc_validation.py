from mhm_tools.common.cli_utils import get_coords
from mhm_tools.post.GRDC_validation import evaludate_grdc_data

def add_args(parser):
    required_args = parser.add_argument_group("required arguments")
    required_args.add_argument(
        "--gauge_info_path",
        required=True,
        help=("Path to the gauge information file."),
    )
    required_args.add_argument(
        "--observed_data_path",
        required=True,
        help=("Path to the observation data file."),
    )
    required_args.add_argument(
        "--model_data_path",
        required=True,
        help=("Path to the model data."),
    )
    required_args.add_argument(
        "--observed_variable",
        required=True,
        help=(""),
    )
    required_args.add_argument(
        "--model_variable",
        required=True,
        help=(""),
    )
    parser.add_argument(
        "--ncpus",
        required=False,
        default=1,
        help=(""),
    )
    parser.add_argument(
        "--lonlatbox",
        required=False,
        default=None,
        help=(
            """coordinates in the form of 'lon_min,lon_max,lat_min,lat_max'"""
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
    # calcualte KGE and NSE or not 
    # sycronus and asyncronus
def run(args):
    lon_min, lon_max, lat_min, lat_max, mask = get_coords(args.lonlatbox, args.mask_file, raise_exception=False)
    coordinate_slice = None
    if lon_min is not None and lon_max is not None and lat_min is not None and lat_max is not None:
        coordinate_slice = {'lat': slice(lat_max, lat_min), 'lon': slice(lon_min, lon_max)}
    evaludate_grdc_data(
        args.model_data_path, args.observed_data_path, args.gauge_info_path, save_path=None, n_jobs=int(args.ncpus), sim_variable=args.model_variable, observed_variable=args.observed_variable, coordinate_slice=coordinate_slice
    )   
