"""Discharge Validation."""


def add_args(parser):
    """Add CLI arguments for the grdc_validation subcommand."""
    required_args = parser.add_argument_group("required arguments")
    required_args.add_argument(
        "--output-dir", help="Path for the output dir.", required=True
    )
    optional = parser.add_argument_group("optional arguments")
    flags = parser.add_argument_group("flags")
    optional.add_argument(
        "--observed-data-path",
        required=False,
        help=("Path to the observation data file."),
    )
    optional.add_argument(
        "--model-data-path",
        required=False,
        help=("Path to the model data."),
    )
    optional.add_argument(
        "--model-file-name",
        required=False,
        default="mrm_node_output.nc",
        help=("File name pattern for model data files."),
    )
    optional.add_argument(
        "--observed-variable",
        required=False,
        default="runoff_mean",
        help=(""),
    )
    optional.add_argument(
        "--model-variable",
        required=False,
        default=None,
        help=("Variable name of the simulation data."),
    )
    optional.add_argument(
        "--facc-file",
        "--mrm-restart",
        dest="facc_file",
        required=False,
        default=None,
        help=("Path to flow-accumulation file used for gauge matching."),
    )
    optional.add_argument(
        "--facc-variable",
        required=False,
        default="L11_fAcc",
        help=("Variable name in --facc_file containing flow accumulation."),
    )
    optional.add_argument(
        "--scc-gauges-file",
        required=False,
        default=None,
        help=("Path to the scc gauges file."),
    )
    optional.add_argument(
        "--evaluation-gauges",
        required=False,
        default=None,
        help=(
            "Path to a file containing the gauge ids to be used for the evaluation. If not provided, all gauges will be used."
        ),
    )
    optional.add_argument(
        "--ncpus",
        required=False,
        default=1,
        help=(""),
    )
    optional.add_argument(
        "--lonlatbox",
        required=False,
        default=None,
        help=("""coordinates in the form of 'lon_min,lon_max,lat_min,lat_max'"""),
    )
    optional.add_argument(
        "--mask-file",
        required=False,
        default=None,
        help=(
            """path to the mask file, a .nc file with a variable 'mask' containing the grid mask at l0 resolution
            required unless --lonlatbox is provided"""
        ),
    )
    optional.add_argument(
        "--n-boostrap-years",
        required=False,
        default=None,
        type=int,
        help=("""Number of years to draw for each boostrap experiment"""),
    )
    optional.add_argument(
        "--n-bootstrap-selections",
        required=False,
        default=None,
        type=int,
        help=("Number of boostrap experiments"),
    )
    optional.add_argument(
        "--start-date",
        required=False,
        default=None,
        type=str,
        help=("""First year allowed in the analysis."""),
    )
    optional.add_argument(
        "--end-date",
        required=False,
        default=None,
        type=str,
        help=("Lates year that is allowed in the analysis."),
    )
    optional.add_argument(
        "--min-overlapping-years",
        required=False,
        default=None,
        type=int,
        help=("Minimum number of overlapping years for evaluation."),
    )
    optional.add_argument(
        "--gauge-location-method",
        required=False,
        default="basinex",
        choices=["basinex", "burek"],
        help=("Method used to optimize gauge location for mRM restart matching."),
    )
    optional.add_argument(
        "--gauge-max-distance-cells",
        required=False,
        default=3,
        type=int,
        help=("Maximum number of grid cells gauge location may be shifted."),
    )
    optional.add_argument(
        "--gauge-max-error",
        required=False,
        default=0.1,
        type=float,
        help=("Maximum allowed relative catchment-area error (fraction; 0.1 = 10%)."),
    )
    optional.add_argument(
        "--hydrograph-plots",
        default="tysc",
        required=False,
        help="specifies which graphics are generated."
        "t model timestep, y yearly, s seasonality, p flow duration, c scatter e.g. "
        "all with out seasonality (advised for performance) = typc",
    )
    flags.add_argument(
        "--use-cached-input-data",
        action="store_true",
        dest="use_cached_input_data",
        required=False,
        help=(
            "Use cached input data if available. Otherwise, process the input data from scratch."
        ),
    )
    flags.add_argument(
        "--save-hydrograph",
        help="Set flag if the calculated hydrographs should be saved and not just the metrics calculated.",
        action="store_true",
        required=False,
    )
    flags.add_argument(
        "--only-plot",
        help="Set Flag if existing output file should be used to create plot",
        action="store_true",
        required=False,
    )
    flags.add_argument(
        "--no-input-data-cache",
        help=(
            "Set flag to not write the processed input data to file. This avoids unnecessary file I/O but may slow down subsequent runs with the same input data. If file is written overwriting is controlled by the --overwrite flag."
        ),
        action="store_true",
        required=False,
    )


def run(args):
    """Evaluate GRDC discharge data against model output.

    Parameters
    ----------
    args : argparse.Namespace
        parsed command line arguments
    """
    from mhm_tools.common.cli_utils import get_coords
    from mhm_tools.post.discharge_evaluation import evaludate_discharge_data

    lon_min, lon_max, lat_min, lat_max, mask = get_coords(
        args.lonlatbox, args.mask_file, raise_exception=False
    )
    evaludate_discharge_data(
        args.model_data_path,
        args.observed_data_path,
        model_file_name=args.model_file_name,
        facc_file=args.facc_file,
        facc_variable=args.facc_variable,
        scc_gauges_file=args.scc_gauges_file,
        output_path=args.output_dir,
        evaluation_gauges=args.evaluation_gauges,
        n_jobs=int(args.ncpus),
        sim_variable=args.model_variable,
        observed_variable=args.observed_variable,
        lon_min=lon_min,
        lon_max=lon_max,
        lat_min=lat_min,
        lat_max=lat_max,
        n_boostrap_selections=args.n_bootstrap_selections,
        n_bootstrap_years=args.n_boostrap_years,
        direct_comparison=args.n_bootstrap_selections is None
        or args.n_boostrap_years is None,
        start_date=args.start_date,
        end_date=args.end_date,
        overwrite=not args.use_cached_input_data,
        only_plot=args.only_plot,
        save_hydrograph=args.save_hydrograph,
        min_overlapping_years=args.min_overlapping_years,
        write_input_data_cache=not args.no_input_data_cache,
        gauge_location_method=args.gauge_location_method,
        gauge_max_distance_cells=args.gauge_max_distance_cells,
        gauge_max_error=args.gauge_max_error,
        hydrograph_plots=args.hydrograph_plots,
    )
