"""Create metric comparison plots from metric CSV files."""

METRIC_PLOT_TYPES = ("cdf", "violin", "catchment-map")


def add_args(parser):
    """Add CLI arguments for the metric-plots subcommand.

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
        help="CSV files or directories that are searched recursively for metric CSV files.",
    )
    required.add_argument(
        "--variable",
        "--variables",
        dest="variables",
        nargs="+",
        required=True,
        help="Exact CSV column names to plot.",
    )
    required.add_argument(
        "--output-dir",
        required=True,
        help="Directory for output PNG files.",
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
        "--file-names",
        "--file-pattern",
        dest="file_names",
        default="*.csv",
        help="Glob pattern used when an input path is a directory.",
    )
    optional.add_argument(
        "--output-prefix",
        default="cdf",
        help="Prefix for output PNG file names.",
    )
    optional.add_argument(
        "--plot-type",
        "--plot-types",
        dest="plot_types",
        nargs="+",
        default=["cdf", "violin"],
        help="Metric plot types to create: cdf, violin, catchment-map.",
    )
    optional.add_argument(
        "--dpi",
        default=450,
        type=int,
        help="Output image resolution.",
    )
    optional.add_argument(
        "--shape-path",
        "--shape-paths",
        dest="shape_paths",
        nargs="+",
        default=None,
        help="Shapefiles matched one-to-one with --input-paths.",
    )
    optional.add_argument(
        "--shape-folder",
        default=None,
        help="Folder with shapefiles matched by CSV stem, parent name, or id column.",
    )
    optional.add_argument(
        "--mask-path",
        "--mask-paths",
        dest="mask_paths",
        nargs="+",
        default=None,
        help="NetCDF mask files matched one-to-one with --input-paths.",
    )
    optional.add_argument(
        "--mask-folder",
        default=None,
        help="Folder with NetCDF mask files matched by CSV stem, parent name, or id column.",
    )
    optional.add_argument(
        "--mask-var",
        default=None,
        help="Variable in NetCDF mask files used for catchment median maps.",
    )
    optional.add_argument(
        "--geometry-match-mode",
        default="auto",
        choices=["auto"],
        help="Geometry matching mode.",
    )
    optional.add_argument(
        "--name-field",
        "--name-fields",
        dest="name_fields",
        nargs="+",
        default=None,
        help="Input-name metadata fields, e.g. experiment,set.",
    )
    optional.add_argument(
        "--name-separator",
        default="_",
        help="Separator used to split input names into metadata fields.",
    )
    optional.add_argument(
        "--group-by",
        nargs="+",
        default=None,
        help="Metadata fields used for additional grouped plots.",
    )
    optional.add_argument(
        "--color-by",
        default=None,
        help="Metadata field used for plot colors.",
    )
    optional.add_argument(
        "--style-by",
        default=None,
        help="Metadata field used for CDF line styles.",
    )


def _normalize_cli_sequence(values):
    """Normalize repeated and comma-separated CLI values.

    Parameters
    ----------
    values : str or Sequence[str] or None
        CLI value or values to normalize.

    Returns
    -------
    list[str] or None
        Normalized values, or None when no values were supplied.
    """
    if values is None:
        return None
    if isinstance(values, str):
        values = [values]
    normalized_values = []
    for value in values:
        for part in str(value).split(","):
            striped_part = part.strip()
            if striped_part:
                normalized_values.append(striped_part)
    if not normalized_values:
        return None
    return normalized_values


def _validate_metric_plot_types(plot_types):
    """Validate metric plot type names.

    Parameters
    ----------
    plot_types : Sequence[str]
        Plot type names to validate.

    Returns
    -------
    list[str]
        Validated plot type names.
    """
    if plot_types is None:
        return ["cdf", "violin"]
    invalid_plot_types = [
        plot_type for plot_type in plot_types if plot_type not in METRIC_PLOT_TYPES
    ]
    if invalid_plot_types:
        choices = ", ".join(METRIC_PLOT_TYPES)
        invalid = ", ".join(invalid_plot_types)
        msg = f"Invalid --plot-type value(s): {invalid}. Choose from: {choices}."
        raise ValueError(msg)
    return list(plot_types)


def run(args):
    """Create metric plots.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed command line arguments.
    """
    from mhm_tools.post.metric_plots import write_metric_plots

    input_paths = _normalize_cli_sequence(args.input_paths)
    input_names = _normalize_cli_sequence(args.input_names)
    variables = _normalize_cli_sequence(args.variables)
    plot_types = _validate_metric_plot_types(_normalize_cli_sequence(args.plot_types))
    shape_paths = _normalize_cli_sequence(args.shape_paths)
    mask_paths = _normalize_cli_sequence(args.mask_paths)
    name_fields = _normalize_cli_sequence(args.name_fields)
    group_by = _normalize_cli_sequence(args.group_by)
    write_metric_plots(
        input_paths=input_paths,
        input_names=input_names,
        variables=variables,
        output_dir=args.output_dir,
        file_names=args.file_names,
        output_prefix=args.output_prefix,
        plot_types=plot_types,
        dpi=args.dpi,
        shape_paths=shape_paths,
        shape_folder=args.shape_folder,
        mask_paths=mask_paths,
        mask_folder=args.mask_folder,
        mask_var=args.mask_var,
        geometry_match_mode=args.geometry_match_mode,
        name_fields=name_fields,
        name_separator=args.name_separator,
        group_by=group_by,
        color_by=args.color_by,
        style_by=args.style_by,
    )
