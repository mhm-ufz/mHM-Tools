"""
Computes and plots the spatial difference between a
model dataset and a reference dataset.
"""

import argparse

from ..post.long_term_mean_difference import long_term_mean_diff


def str2float(value):
    """Convert a string to float, but let None remain None."""
    # If argparse is giving us the default None (i.e. user didn't supply the flag) ...
    if value is None:
        return None

    # If the user literally typed "none" (case-insensitive), treat it as None too
    if isinstance(value, str) and value.lower() == "none":
        return None

    # Otherwise it must be a string we can cast
    if not isinstance(value, str):
        msg = f"Expected a string or None, but got {type(value).__name__}."
        raise argparse.ArgumentTypeError(msg)

    try:
        return float(value)
    except ValueError as err:
        msg = f"{value!r} is not a valid float."
        raise argparse.ArgumentTypeError(msg) from err


def add_args(parser):
    """Add CLI arguments for the long_term_mean_diff subcommand."""
    parser.description = (
        "Compute and plot the spatial difference between a model dataset and a reference dataset "
        "for a specified variable. Supports wildcard matching, custom variable names, colorbar labels, "
        "titles, output file naming, optional axis limits, custom colormap, and explicit colormap range."
    )
    parser.epilog = (
        "Example:\n"
        "  mhm-tools long_term_mean_validation \\\n"
        "    --ref_input_dir /path/to/ref \\\n"
        "    --mod_input_dir /path/to/mod \\\n"
        '    --reference_pattern "ref_*.nc" \\\n'
        '    --model_pattern "mod_*.nc" \\\n'
        "    --ref_var pre --mod_var pre \\\n"
        '    --colorbar_label "ΔP" \\\n'
        '    --title "Precip. Diff" \\\n'
        "    --x_min -10 --x_max 30 --y_min 40 --y_max 70 \\\n"
        "    --cmap viridis --vmin -5 --vmax 5 \\\n"
        "    -o /out/dir --output_file diff.png"
    )

    # required arguments
    req = parser.add_argument_group("required arguments")
    req.add_argument(
        "--ref_input_dir", required=True, help="Directory with reference NetCDF files"
    )
    req.add_argument(
        "--mod_input_dir", required=True, help="Directory with model NetCDF files"
    )
    req.add_argument(
        "--reference_pattern", required=True, help="Wildcard for reference file"
    )
    req.add_argument("--model_pattern", required=True, help="Wildcard for model file")
    req.add_argument(
        "--ref_var", required=True, help="Variable name in reference dataset"
    )
    req.add_argument("--mod_var", required=True, help="Variable name in model dataset")
    req.add_argument(
        "-o", "--output_dir", required=True, help="Directory to save the output PNG"
    )
    req.add_argument("--output_file", required=True, help="Filename for the output PNG")

    # optional arguments
    parser.add_argument(
        "--colorbar_label",
        default="Difference (model - reference)",
        help="Label for the plot colorbar",
    )
    parser.add_argument(
        "--title",
        default="Mean Difference (model - reference)",
        help="Title for the plot",
    )
    parser.add_argument(
        "--x_min", type=str2float, default=None, help="Minimum longitude to display"
    )
    parser.add_argument(
        "--x_max", type=str2float, default=None, help="Maximum longitude to display"
    )
    parser.add_argument(
        "--y_min", type=str2float, default=None, help="Minimum latitude to display"
    )
    parser.add_argument(
        "--y_max", type=str2float, default=None, help="Maximum latitude to display"
    )
    parser.add_argument(
        "--cmap",
        default="coolwarm",
        help="Matplotlib colormap name to use for the difference plot",
    )
    parser.add_argument(
        "--vmin",
        type=str2float,
        default=None,
        help="Minimum data value for colormap (optional)",
    )
    parser.add_argument(
        "--vmax",
        type=str2float,
        default=None,
        help="Maximum data value for colormap (optional)",
    )


def run(args):
    """
    Run script to plot long term means of a variable given (model - reference).

    Parameters
    ----------
    args : argparse.Namespace
        parsed command line arguments

    """
    long_term_mean_diff(
        ref_input_dir=args.ref_input_dir,
        mod_input_dir=args.mod_input_dir,
        reference_pattern=args.reference_pattern,
        model_pattern=args.model_pattern,
        ref_var=args.ref_var,
        mod_var=args.mod_var,
        colorbar_label=args.colorbar_label,
        title=args.title,
        output_dir=args.output_dir,
        output_file=args.output_file,
        x_min=args.x_min,
        x_max=args.x_max,
        y_min=args.y_min,
        y_max=args.y_max,
        cmap=args.cmap,
        vmin=args.vmin,
        vmax=args.vmax,
    )
