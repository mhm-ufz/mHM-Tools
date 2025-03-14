"""Discharge Validation."""

from mhm_tools.common.cli_utils import get_coords
from mhm_tools.post.GRDC_validation import evaludate_grdc_data


def add_args(parser):
    required_args = parser.add_argument_group("required arguments")
    required_args.add_argument(
        "--gauge_info_path",
        required=False,
        default=None,
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
        help=("Variable name of the simulation data."),
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
        help=("""coordinates in the form of 'lon_min,lon_max,lat_min,lat_max'"""),
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
        "--n_boostrap_years",
        required=False,
        default=None,
        type=int,
        help=("""Number of years to draw for each boostrap experiment"""),
    )
    parser.add_argument(
        "--n_bootstrap_selections",
        required=False,
        default=None,
        type=int,
        help=("Number of boostrap experiments"),
    )
    parser.add_argument(
        "--start_date",
        required=False,
        default=None,
        type=str,
        help=("""First year allowed in the analysis."""),
    )
    parser.add_argument(
        "--end_date",
        required=False,
        default=None,
        type=str,
        help=("Lates year that is allowed in the analysis."),
    )
    parser.add_argument(
        "--direct_comparison",
        action="store_true",
        dest="direct_comparison",
        required=False,
        help=("Use no statistics but compare timeseries directly. Needs ref_path."),
    )
    parser.add_argument("--output_dir", help="Path for the output dir.", required=True)

def run(args):
    lon_min, lon_max, lat_min, lat_max, mask = get_coords(
        args.lonlatbox, args.mask_file, raise_exception=False
    )
    evaludate_grdc_data(
        args.model_data_path,
        args.observed_data_path,
        args.gauge_info_path,
        output_path=args.output_dir,
        n_jobs=int(args.ncpus),
        sim_variable=args.model_variable,
        observed_variable=args.observed_variable,
        lon_min=lon_min,
        lon_max=lon_max,
        lat_min=lat_min,
        lat_max=lat_max,
        n_boostrap_selections=args.n_bootstrap_selections,
        n_bootstrap_years=args.n_boostrap_years,
        direct_comparison=args.direct_comparison,
        start_data=args.start_date,
        end_date=args.end_date,
    )
