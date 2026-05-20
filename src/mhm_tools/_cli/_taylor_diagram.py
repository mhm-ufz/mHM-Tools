"""
Compute and plot a Taylor diagram comparing multiple model datasets against a reference dataset.

This script reads CF-compliant NetCDF files, computes normalized standard deviation,
correlation, and centered root mean square error (CRMSE) for each model variable
against a single reference field, and creates one or multiple Taylor diagrams.
"""


def add_args(parser):
    """Add CLI arguments for the Taylor diagram subcommand."""
    parser.description = "Compute and plot a Taylor diagram comparing multiple model datasets against a single reference dataset."
    parser.epilog = (
        "Example:\n"
        "  mhm-tools taylor_diagram \\\n"
        "    --ref_input_dir /path/to/obs \\\n"
        '    --reference_pattern "obs.nc" \\\n'
        "    --ref_var pre \\\n"
        "    --mod_input_dirs /path/to/model1 /path/to/model2 \\\n"
        "    --model_patterns model1.nc model2.nc \\\n"
        "    --mod_vars mod1 mod2 \\\n"
        '    --title "Taylor Diagram for Precipitation" \\\n'
        "    -o /out/dir --output_file taylor.png"
    )

    # Required arguments
    req = parser.add_argument_group("required arguments")
    req.add_argument(
        "--ref-input-dir", required=True, help="Directory with reference NetCDF file"
    )
    req.add_argument(
        "--reference-pattern",
        required=True,
        help="Filename pattern for reference NetCDF file",
    )
    req.add_argument(
        "--ref-var", required=True, help="Variable name in reference dataset"
    )
    req.add_argument(
        "--mod-input-dirs",
        nargs="+",
        required=True,
        help="List of directories containing model NetCDF files (one per model)",
    )
    req.add_argument(
        "--model-patterns",
        nargs="+",
        required=True,
        help="List of filename patterns for model NetCDF files (one per model)",
    )
    req.add_argument(
        "--mod-vars",
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
        "--ref-label", default="Ref", help="Label to use for the reference data."
    )
    optional.add_argument(
        "--mod-labels", nargs="+", help="List of labels to use for the model data."
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
    from ..post.taylor_diagram import generate_taylor_diagram

    # sanity check to ensure matched lists
    if not (len(args.mod_input_dirs) == len(args.model_patterns) == len(args.mod_vars)):
        msg = (
            "The number of --mod_input_dirs, --model_patterns, "
            "and --mod_vars must all match."
        )
        raise ValueError(msg)

    generate_taylor_diagram(
        ref_input_dir=args.ref_input_dir,
        reference_pattern=args.reference_pattern,
        ref_var=args.ref_var,
        ref_label=args.ref_label,
        mod_input_dirs=args.mod_input_dirs,
        model_patterns=args.model_patterns,
        mod_vars=args.mod_vars,
        mod_labels=args.mod_labels,
        title=args.title,
        output_dir=args.output_dir,
        output_file=args.output_file,
        normalize=args.normalize,
    )
