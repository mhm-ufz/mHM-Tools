"""Evaluate a simulated total water storage anomaly against a gridded reference.

The tool sums the storage variables of an mHM fluxes-and-states record, or
reads a total water storage or an already computed anomaly, brings both
datasets onto the coarser grid and a monthly calendar, removes the mean of a
baseline period and writes the selected per cell metrics, the Kling-Gupta
efficiency and the root mean square error, as maps and a NetCDF file, plus a
per cell time series plot per evaluation region.

Authors
-------
- Simon Lüdke
"""

import logging

logger = logging.getLogger(__name__)


def add_args(parser):
    """Add cli arguments for the twsa evaluation.

    Parameters
    ----------
    parser : argparse.ArgumentParser
        the main argument parser
    """
    from mhm_tools.common.constants import WMO_REGION_BOUNDS
    from mhm_tools.post.twsa_evaluation import (
        AUTO_FORMAT,
        AVAILABLE_METRICS,
        DEFAULT_BASELINE_END_YEAR,
        DEFAULT_BASELINE_START_YEAR,
        DEFAULT_MAX_REGION_CELL_LINES,
        DEFAULT_MIN_VALID_MONTHS,
        INPUT_FORMATS,
        KGE_VMIN,
    )

    required = parser.add_argument_group("required arguments")
    optional = parser.add_argument_group("optional arguments")
    flags = parser.add_argument_group("flags")

    required.add_argument(
        "--input-path",
        required=True,
        help="Path to the model file, or a directory searched recursively.",
    )
    required.add_argument(
        "--ref-path",
        required=True,
        help="Path to the reference file, or a directory searched recursively.",
    )
    required.add_argument(
        "--output-dir",
        required=True,
        help="Path for the output dir.",
    )

    optional.add_argument(
        "--input-file-name",
        default="*.nc",
        help="Glob pattern for the recursive model search.",
    )
    optional.add_argument(
        "--ref-file-name",
        default="*.nc",
        help="Glob pattern for the recursive reference search.",
    )
    optional.add_argument(
        "--input-variable",
        default=None,
        help=(
            "Variable in the model file. Detected when the file holds one "
            "variable, ignored for the mhm_states format."
        ),
    )
    optional.add_argument(
        "--ref-variable",
        default=None,
        help="Variable in the reference file. Detected when the file holds one.",
    )
    optional.add_argument(
        "--input-name",
        default="mhm",
        help="Name of the model dataset, used in plots and file names.",
    )
    optional.add_argument(
        "--ref-name",
        default="ref",
        help="Name of the reference dataset, used in plots and file names.",
    )
    optional.add_argument(
        "--input-format",
        default=AUTO_FORMAT,
        choices=[AUTO_FORMAT, *INPUT_FORMATS],
        help=(
            "Model input format. 'auto' detects it from the variables present: "
            "mhm_states sums the mHM storages, tws reads a total water storage "
            "and twsa an already computed anomaly."
        ),
    )
    optional.add_argument(
        "--ref-format",
        default=AUTO_FORMAT,
        choices=[AUTO_FORMAT, "tws", "twsa"],
        help="Reference input format. 'auto' detects it from the variable.",
    )
    optional.add_argument(
        "--input-scale",
        type=float,
        default=None,
        help=(
            "Millimetres of water per model data unit. Overrides the 'units' "
            "attribute and is required when the file carries none."
        ),
    )
    optional.add_argument(
        "--ref-scale",
        type=float,
        default=None,
        help="Millimetres of water per reference data unit.",
    )
    optional.add_argument(
        "--baseline-start-year",
        type=int,
        default=DEFAULT_BASELINE_START_YEAR,
        help="First calendar year of the anomaly baseline.",
    )
    optional.add_argument(
        "--baseline-end-year",
        type=int,
        default=DEFAULT_BASELINE_END_YEAR,
        help="Last calendar year of the anomaly baseline, inclusive.",
    )
    optional.add_argument(
        "--min-valid-months",
        type=int,
        default=DEFAULT_MIN_VALID_MONTHS,
        help="Fewest compared months a cell needs for a metric, else it stays empty.",
    )
    optional.add_argument(
        "--metrics",
        default="all",
        help=(
            "Metrics every cell is scored with: 'all', 'none' or a comma "
            f"separated list of {', '.join(AVAILABLE_METRICS)}. A metric that "
            "is not selected is neither written nor plotted."
        ),
    )
    optional.add_argument(
        "--regions",
        default="all",
        help=(
            "Regions mapped separately: 'all', 'none' or a comma separated list "
            f"of {', '.join(WMO_REGION_BOUNDS)}."
        ),
    )
    optional.add_argument(
        "--max-region-cell-lines",
        type=int,
        default=DEFAULT_MAX_REGION_CELL_LINES,
        help=(
            "Most cell lines drawn per dataset in a region cell plot "
            "(--plot-region-cells), 0 for all cells."
        ),
    )
    optional.add_argument(
        "--kge-vmin",
        type=float,
        default=KGE_VMIN,
        help="Lower end of the KGE colour scale.",
    )
    optional.add_argument(
        "--ncpus",
        type=int,
        default=0,
        help=(
            "Cores the dask cluster computes on. "
            "0 uses every core the machine reports."
        ),
    )
    optional.add_argument(
        "--max-memory-gib",
        type=float,
        default=8.0,
        help="Memory budget the time chunks are sized against.",
    )

    flags.add_argument(
        "--no-write-twsa",
        action="store_true",
        help="Skip writing the normalized monthly anomaly fields.",
    )
    flags.add_argument(
        "--plot-kge-components",
        action="store_true",
        help="Also map the KGE components alpha, beta and gamma.",
    )
    flags.add_argument(
        "--plot-region-cells",
        action="store_true",
        help="Also plot every grid cell over time per region.",
    )
    flags.add_argument(
        "--write-region-stats",
        action="store_true",
        help="Also write the metrics table of the domain and the regions to CSV.",
    )


def select_metrics(requested_metrics):
    """Resolve the requested metric names against the available metrics.

    Args:
        requested_metrics: "all", "none" or a comma separated list of the
            metrics the twsa evaluation offers.

    Returns
    -------
        List of metric names, empty when no metric is wanted.
    """
    from mhm_tools.common.cli_utils import normalize_cli_sequence
    from mhm_tools.common.logger import ErrorLogger
    from mhm_tools.post.twsa_evaluation import AVAILABLE_METRICS

    requested = normalize_cli_sequence(requested_metrics) or []
    requested = [entry.lower() for entry in requested]
    if requested in ([], ["none"]):
        return []
    if requested == ["all"]:
        return list(AVAILABLE_METRICS)
    unknown = [entry for entry in requested if entry not in AVAILABLE_METRICS]
    if unknown:
        msg = (
            f"Unknown metric(s) {', '.join(unknown)}. "
            f"Available metrics: {', '.join(AVAILABLE_METRICS)}."
        )
        with ErrorLogger(logger):
            raise ValueError(msg)
    return [metric for metric in AVAILABLE_METRICS if metric in requested]


def run(args):
    """Run the total water storage anomaly evaluation."""
    from mhm_tools.common.logger import ErrorLogger
    from mhm_tools.common.parallel import create_dask_cluster
    from mhm_tools.post.twsa_evaluation import twsa_evaluation

    # refuse a run that would compute nothing before a cluster is even started
    selected_metrics = select_metrics(args.metrics)
    if not selected_metrics and args.no_write_twsa:
        msg = (
            "--metrics none and --no-write-twsa leave nothing to calculate. "
            "Select a metric or drop --no-write-twsa."
        )
        with ErrorLogger(logger):
            raise ValueError(msg)

    with create_dask_cluster(args.ncpus):
        written_files = twsa_evaluation(
            input_path=args.input_path,
            ref_path=args.ref_path,
            output_dir=args.output_dir,
            input_file_name=args.input_file_name,
            ref_file_name=args.ref_file_name,
            input_var=args.input_variable,
            ref_var=args.ref_variable,
            input_name=args.input_name,
            ref_name=args.ref_name,
            input_format=args.input_format,
            ref_format=args.ref_format,
            input_scale=args.input_scale,
            ref_scale=args.ref_scale,
            baseline_start_year=args.baseline_start_year,
            baseline_end_year=args.baseline_end_year,
            min_valid_months=args.min_valid_months,
            regions=args.regions,
            max_region_cell_lines=args.max_region_cell_lines,
            kge_vmin=args.kge_vmin,
            metrics=selected_metrics,
            write_twsa=not args.no_write_twsa,
            max_memory_gib=args.max_memory_gib,
            plot_kge_components=args.plot_kge_components,
            plot_region_cells=args.plot_region_cells,
            write_region_stats=args.write_region_stats,
        )
    for label, file_path in written_files.items():
        logger.info(f"{label}: {file_path}")
