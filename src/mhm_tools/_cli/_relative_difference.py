"""
Compute and plot the spatial relative difference between a model dataset and a reference dataset.

This script reads CF-compliant NetCDF files for both model and reference datasets, computes
the spatial relative difference in percent as: diff = 100 * (da_ref - da_mod) / da_ref
for a specified variable,
applies any provided geographic or data range limits, and generates a high-resolution PNG
showing that difference with customizable title, colorbar label, and colormap.

Authors
-------
- Jeisson Leal
"""

import argparse

from mhm_tools.common.cli_utils import (
    add_netcdf_compression_args,
    get_netcdf_compression,
)


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
    """Add CLI arguments for the relative_difference subcommand."""
    parser.description = (
        "Compute and plot the spatial relative difference between a model dataset and a reference dataset "
        "for a specified variable in percent: diff = 100 * (da_ref - da_mod) / da_ref. "
        "Supports wildcard matching, custom variable names, colorbar labels, "
        "titles, output file naming, optional axis limits, custom colormap, and explicit colormap range."
    )
    parser.epilog = (
        "Example:\n"
        "  mhm-tools relative-difference \\\n"
        "    --ref-dir /path/to/ref \\\n"
        "    --input-dir /path/to/mod \\\n"
        '    --ref-file-name "ref_*.nc" \\\n'
        '    --input-file-name "mod_*.nc" \\\n'
        "    --ref-var pre --input-var pre --save-ncfile \\\n"
        '    --colorbar-label "ΔP" \\\n'
        '    --title "Precip. Diff" \\\n'
        "    --x-min -10 --x-max 30 --y-min 40 --y-max 70 \\\n"
        "    --cmap viridis --vmin -5 --vmax 5 \\\n"
        "    -o /out/dir --output-file-png diff.png"
    )

    # required arguments, named like the evaluation tools with the former
    # reference/model names as hidden aliases
    req = parser.add_argument_group("required arguments")
    req.add_argument(
        "--ref-dir",
        "--ref-input-dir",
        dest="ref_input_dir",
        required=True,
        help="Directory with reference NetCDF files",
    )
    req.add_argument(
        "--input-dir",
        "--mod-input-dir",
        dest="mod_input_dir",
        required=True,
        help="Directory with model NetCDF files",
    )
    req.add_argument(
        "--ref-file-name",
        "--reference-pattern",
        dest="reference_pattern",
        required=True,
        help="Wildcard for reference file",
    )
    req.add_argument(
        "--input-file-name",
        "--model-pattern",
        dest="model_pattern",
        required=True,
        help="Wildcard for model file",
    )
    req.add_argument(
        "--ref-var", required=True, help="Variable name in reference dataset"
    )
    req.add_argument(
        "--input-var",
        "--mod-var",
        dest="mod_var",
        required=True,
        help="Variable name in model dataset",
    )
    req.add_argument(
        "-o", "--output-dir", required=True, help="Directory to save the output PNG"
    )
    req.add_argument(
        "--output-file-png", required=True, help="Filename for the output PNG"
    )

    # optional arguments
    optional = parser.add_argument_group("optional arguments")
    flags = parser.add_argument_group("flags")
    flags.add_argument(
        "--save-ncfile",
        action="store_true",
        help="if set to True, stores a NetCDF file in --output_dir",
    )
    optional.add_argument(
        "--output-file-nc",
        default="relative_difference.nc",
        help="If --save_ncfile, gives the name of the file to be saved in --output_dir",
    )
    optional.add_argument(
        "--colorbar-label",
        default="Difference (model - reference)",
        help="Label for the plot colorbar",
    )
    optional.add_argument(
        "--title",
        default="Mean Difference (model - reference)",
        help="Title for the plot",
    )
    optional.add_argument(
        "--x-min", type=str2float, default=None, help="Minimum longitude to display"
    )
    optional.add_argument(
        "--x-max", type=str2float, default=None, help="Maximum longitude to display"
    )
    optional.add_argument(
        "--y-min", type=str2float, default=None, help="Minimum latitude to display"
    )
    optional.add_argument(
        "--y-max", type=str2float, default=None, help="Maximum latitude to display"
    )
    optional.add_argument(
        "--cmap",
        default="coolwarm",
        help="Matplotlib colormap name to use for the difference plot",
    )
    optional.add_argument(
        "--vmin",
        type=str2float,
        default=None,
        help=(
            "Minimum data value for colormap (optional). Without --vmin and "
            "--vmax the colours span +-25 %, +-50 %, +-75 % or +-100 %, the "
            "narrowest holding the data without its outliers."
        ),
    )
    optional.add_argument(
        "--vmax",
        type=str2float,
        default=None,
        help=(
            "Maximum data value for colormap (optional). Without --vmin and "
            "--vmax the colours span +-25 %, +-50 %, +-75 % or +-100 %, the "
            "narrowest holding the data without its outliers."
        ),
    )
    add_netcdf_compression_args(parser)


def run(args):
    """
    Compute and plot the spatial relative difference (ref - model) / ref.

    Parameters
    ----------
    args : argparse.Namespace
        parsed command line arguments

    """
    from ..post.relative_difference import calc_rel_diff

    calc_rel_diff(
        ref_input_dir=args.ref_input_dir,
        mod_input_dir=args.mod_input_dir,
        reference_pattern=args.reference_pattern,
        model_pattern=args.model_pattern,
        ref_var=args.ref_var,
        mod_var=args.mod_var,
        save_ncfile=args.save_ncfile,
        output_file_nc=args.output_file_nc,
        colorbar_label=args.colorbar_label,
        title=args.title,
        output_dir=args.output_dir,
        output_file_png=args.output_file_png,
        x_min=args.x_min,
        x_max=args.x_max,
        y_min=args.y_min,
        y_max=args.y_max,
        cmap=args.cmap,
        vmin=args.vmin,
        vmax=args.vmax,
        compression=get_netcdf_compression(args),
    )
