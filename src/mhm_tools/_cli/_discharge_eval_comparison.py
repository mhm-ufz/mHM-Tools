"""Create comparison plots from one or more discharge-evaluation results CSVs.

The workflow is as follows:
1. Use discharge-eval to create results CSVs for two or more model runs.
2. Use discharge-eval-comparison to create comparison plots from the results CSVs.

These plots include:
- CDF plots
- Violin plots
- spatial difference maps (map-diff) between a reference run and one or more other runs.

Authors
-------
- Simon Lüdke
"""

from mhm_tools.post.discharge_eval_comparison import DISCHARGE_COMPARISON_PLOT_TYPES


def add_args(parser):
    """Add CLI arguments for the discharge-eval-comparison subcommand.

    Parameters
    ----------
    parser : argparse.ArgumentParser
        The subcommand parser to extend.
    """
    required = parser.add_argument_group("required arguments")
    optional = parser.add_argument_group("optional arguments")

    required.add_argument(
        "--input-path",
        "--input-paths",
        dest="input_paths",
        nargs="+",
        required=True,
        help="discharge-evaluation results.csv files or directories containing them.",
    )
    required.add_argument(
        "--output-dir",
        required=True,
        help="Directory for output PNG/PDF/CSV files.",
    )
    optional.add_argument(
        "--input-name",
        "--input-names",
        dest="input_names",
        nargs="+",
        default=None,
        help="Plot labels matching --input-paths. Defaults to path names.",
    )
    optional.add_argument(
        "--variable",
        "--variables",
        dest="variables",
        nargs="+",
        default=["kge"],
        help="Metric columns to plot. Defaults to kge.",
    )
    optional.add_argument(
        "--file-names",
        "--file-pattern",
        dest="file_names",
        default="results.csv",
        help="Glob pattern used when an input path is a directory.",
    )
    optional.add_argument(
        "--plot-type",
        "--plot-types",
        dest="plot_types",
        nargs="+",
        default=None,
        help=(
            "Comparison plot types to create: cdf, violin, catchment-map, "
            "map-diff. Defaults to cdf, map-diff, plus catchment-map "
            "whenever --shape-folder or --mask-folder is given."
        ),
    )
    optional.add_argument(
        "--dpi",
        default=400,
        type=int,
        help="Output image resolution for cdf/violin plots.",
    )
    optional.add_argument(
        "--shape-folder",
        default=None,
        help="Folder with shapefiles matched by gauge id, for catchment maps.",
    )
    optional.add_argument(
        "--mask-folder",
        default=None,
        help="Folder with NetCDF mask files matched by gauge id, for catchment maps.",
    )
    optional.add_argument(
        "--mask-var",
        default=None,
        help="Variable in NetCDF mask files used for catchment median maps.",
    )
    optional.add_argument(
        "--reference-name",
        default=None,
        help="Baseline run's label for map-diff. Defaults to the first --input-path.",
    )
    optional.add_argument(
        "--map-diff-cmap",
        default="coolwarm_r",
        help="Diverging colormap for map-diff, centered on zero.",
    )
    optional.add_argument(
        "--map-diff-dpi",
        default=400,
        type=int,
        help="Output image resolution for map-diff plots.",
    )


def _validate_discharge_comparison_plot_types(plot_types):
    """Validate discharge comparison plot type names.

    Parameters
    ----------
    plot_types : Sequence[str] or None
        Plot type names to validate.

    Returns
    -------
    list[str] or None
        Validated plot type names, or None to keep the post module's default.
    """
    if plot_types is None:
        return None
    invalid_plot_types = [
        plot_type
        for plot_type in plot_types
        if plot_type not in DISCHARGE_COMPARISON_PLOT_TYPES
    ]
    if invalid_plot_types:
        choices = ", ".join(DISCHARGE_COMPARISON_PLOT_TYPES)
        invalid = ", ".join(invalid_plot_types)
        msg = f"Invalid --plot-type value(s): {invalid}. Choose from: {choices}."
        raise ValueError(msg)
    return list(plot_types)


def run(args):
    """Create discharge-evaluation comparison plots.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed command line arguments.
    """
    from mhm_tools.common.cli_utils import normalize_cli_sequence
    from mhm_tools.post.discharge_eval_comparison import (
        write_discharge_eval_comparison_plots,
    )

    input_paths = normalize_cli_sequence(args.input_paths)
    input_names = normalize_cli_sequence(args.input_names, split_whitespace=False)
    variables = normalize_cli_sequence(args.variables)
    plot_types = _validate_discharge_comparison_plot_types(
        normalize_cli_sequence(args.plot_types)
    )
    write_discharge_eval_comparison_plots(
        input_paths=input_paths,
        input_names=input_names,
        variables=variables,
        output_dir=args.output_dir,
        file_names=args.file_names,
        plot_types=plot_types,
        dpi=args.dpi,
        shape_folder=args.shape_folder,
        mask_folder=args.mask_folder,
        mask_var=args.mask_var,
        reference_name=args.reference_name,
        map_diff_cmap=args.map_diff_cmap,
        map_diff_dpi=args.map_diff_dpi,
    )
