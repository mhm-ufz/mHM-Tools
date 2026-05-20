#!/usr/bin/env python
"""
Compute and plot a 2D map for a given NetCDF dataset.

This CLI reads a CF-compliant NetCDF file, extracts a specified variable, and
invokes mhm_tools.common.plotter.plot_map to generate and save a geo-aware
2D map plot with customizable labels, colormap, and spatial or data limits.
"""

import argparse
from pathlib import Path

import xarray as xr


def str2float(value):
    """Convert a string to float, but let None remain None."""
    if value is None or (isinstance(value, str) and value.lower() == "none"):
        return None
    try:
        return float(value)
    except (TypeError, ValueError) as err:
        msg = f"{value!r} is not a valid float."
        raise argparse.ArgumentTypeError(msg) from err


def add_args(parser: argparse.ArgumentParser):
    """Add CLI arguments for the 2d_map subcommand."""
    parser.description = (
        "Compute and plot a 2D spatial map for a specified variable from a NetCDF file."
    )
    parser.epilog = (
        "Example:\n"
        "mhm-tools 2d_map \\\n"
        "--input-file /path/to/data.nc \\\n"
        "--var temperature \\\n"
        "--colorbar-label 'Temp (°C)' \\\n"
        "--title 'Surface Temperature' \\\n"
        "--x-min -10 --x-max 30 --y-min 40 --y-max 70 \\\n"
        "--cmap viridis --vmin -5 --vmax 5 \\\n"
        "-o /out/dir --output-file_png map.png"
    )

    # required arguments
    req = parser.add_argument_group("required arguments")
    req.add_argument(
        "--input-file", required=True, help="Path to the input NetCDF file"
    )
    req.add_argument(
        "--var", required=True, help="Variable name in the NetCDF dataset to plot"
    )
    req.add_argument(
        "-o",
        "--output-dir",
        required=True,
        help="Directory where the output PNG will be saved",
    )
    req.add_argument(
        "--output-file-png", required=True, help="Filename for the output PNG"
    )

    # optional arguments
    optional = parser.add_argument_group("optional arguments")
    optional.add_argument("--colorbar-label", default="", help="Label for the colorbar")
    optional.add_argument("--title", default="", help="Title for the plot")
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
        "--cmap", default="RdBu_r", help="Matplotlib colormap name to use"
    )
    optional.add_argument(
        "--vmin", type=str2float, default=None, help="Minimum data value for colormap"
    )
    optional.add_argument(
        "--vmax", type=str2float, default=None, help="Maximum data value for colormap"
    )


def run(args: argparse.Namespace):
    """Entry point: read NetCDF, extract data array, and call plot_map."""
    # Load dataset and variable
    ds = xr.open_dataset(args.input_file)
    data = ds[args.var]

    # Build output path
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / args.output_file_png

    # Call the plotting utility
    from mhm_tools.common.plotter import plot_map

    plot_map(
        data=data,
        cb_label=args.colorbar_label,
        title=args.title,
        out_path=out_path,
        cmap=args.cmap,
        x_min=args.x_min,
        x_max=args.x_max,
        y_min=args.y_min,
        y_max=args.y_max,
        vmin=args.vmin,
        vmax=args.vmax,
    )
