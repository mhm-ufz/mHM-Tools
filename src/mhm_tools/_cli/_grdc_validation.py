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
    # calcualte KGE and NSE or not 
    # sycronus and asyncronus
def run(args):
    evaludate_grdc_data(
        args.model_data_path, args.observed_data_path, args.gauge_info_path, save_path=None, n_jobs=int(args.ncpus), sim_variable=args.model_variable, obs_variable=args.observed_variable
    )   
