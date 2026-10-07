"""
Compute and plot a Taylor diagram comparing multiple model datasets against a reference dataset.

This script reads CF-compliant NetCDF files, computes normalized standard deviation,
correlation, and centered root mean square error (CRMSE) for each model variable
against a single reference field, and creates one or multiple Taylor diagrams.

Authors
-------
- Jeisson Leal
"""


def add_args(parser):
    """Add CLI arguments for the Taylor diagram subcommand."""
    parser.description = "Compute and plot a Taylor diagram comparing multiple model datasets against a single reference dataset."
    parser.epilog = (
        "Example:\n"
        "  mhm-tools taylor-diagram \\\n"
        "    --ref-dir /path/to/obs \\\n"
        '    --ref-file-name "obs.nc" \\\n'
        "    --ref-var pre \\\n"
        "    --input-dir /path/to/model1 /path/to/model2 \\\n"
        "    --input-file-name model1.nc model2.nc \\\n"
        "    --input-var mod1 mod2 \\\n"
        '    --title "Taylor Diagram for Precipitation" \\\n'
        "    -o /out/dir --output-file taylor.png"
    )

    # Required arguments, named like the evaluation tools with the former
    # reference/model names as hidden aliases
    req = parser.add_argument_group("required arguments")
    req.add_argument(
        "--ref-dir",
        "--ref-input-dir",
        dest="ref_input_dir",
        required=True,
        help="Directory with reference NetCDF file",
    )
    req.add_argument(
        "--ref-file-name",
        "--reference-pattern",
        dest="reference_pattern",
        required=True,
        help="Filename pattern for reference NetCDF file",
    )
    req.add_argument(
        "--ref-var", required=True, help="Variable name in reference dataset"
    )
    req.add_argument(
        "--input-dir",
        "--input-dirs",
        "--mod-input-dirs",
        dest="mod_input_dirs",
        nargs="+",
        required=True,
        help="List of directories containing model NetCDF files (one per model)",
    )
    req.add_argument(
        "--input-file-name",
        "--input-file-names",
        "--model-patterns",
        dest="model_patterns",
        nargs="+",
        required=True,
        help="List of filename patterns for model NetCDF files (one per model)",
    )
    req.add_argument(
        "--input-var",
        "--input-vars",
        "--mod-vars",
        dest="mod_vars",
        nargs="+",
        required=True,
        help="List of variable names in model datasets (one per model)",
    )
    req.add_argument(
        "-o", "--output-dir", required=True, help="Directory to save the output PNG."
    )
    req.add_argument(
        "--output-file", required=True, help="Filename for the output PNG."
    )

    # Optional arguments
    optional = parser.add_argument_group("optional arguments")
    flags = parser.add_argument_group("flags")
    optional.add_argument(
        "--title", default="Taylor Diagram", help="Title for the Taylor diagram."
    )
    optional.add_argument(
        "--ref-name",
        "--ref-label",
        dest="ref_label",
        default="Ref",
        help="Label to use for the reference data.",
    )
    optional.add_argument(
        "--input-name",
        "--input-names",
        "--mod-labels",
        dest="mod_labels",
        nargs="+",
        help="List of labels to use for the model data.",
    )
    flags.add_argument(
        "--normalize",
        action="store_true",
        help="If set, normalize standard deviations by the reference std.",
    )


def run(args):
    """Generate a Taylor diagram comparing model datasets to a reference.

    Parameters
    ----------
    args : argparse.Namespace
        parsed command line arguments
    """
    from mhm_tools.common.cli_utils import normalize_cli_sequence

    from ..post.taylor_diagram import generate_taylor_diagram

    mod_input_dirs = normalize_cli_sequence(args.mod_input_dirs)
    model_patterns = normalize_cli_sequence(args.model_patterns)
    mod_vars = normalize_cli_sequence(args.mod_vars)
    # a plot label may hold a space, so it is only split on commas
    mod_labels = normalize_cli_sequence(args.mod_labels, split_whitespace=False)

    # sanity check to ensure matched lists
    if not (len(mod_input_dirs) == len(model_patterns) == len(mod_vars)):
        msg = (
            "The number of --input-dir, --input-file-name "
            "and --input-var values must all match."
        )
        raise ValueError(msg)

    generate_taylor_diagram(
        ref_input_dir=args.ref_input_dir,
        reference_pattern=args.reference_pattern,
        ref_var=args.ref_var,
        ref_label=args.ref_label,
        mod_input_dirs=mod_input_dirs,
        model_patterns=model_patterns,
        mod_vars=mod_vars,
        mod_labels=mod_labels,
        title=args.title,
        output_dir=args.output_dir,
        output_file=args.output_file,
        normalize=args.normalize,
    )
