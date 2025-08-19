"""Call function to plot a 2D map.

Authors
-------
- Jeisson Leal
"""

from pathlib import Path

import xarray as xr

from mhm_tools.common.plotter import plot_map


def run(args):
    """
    Read a NetCDF file, extract a variable, and generate a 2D map plot.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed arguments with attributes:
        - input_file: str or Path to NetCDF
        - var: variable name to plot
        - colorbar_label: str for the colorbar
        - title: str plot title
        - output_dir: directory to save plot
        - output_file_png: filename for the PNG
        - cmap, x_min, x_max, y_min, y_max, vmin, vmax
    """
    # Open dataset and select variable
    ds = xr.open_dataset(args.input_file)
    data = ds[args.var]

    # Ensure output directory exists
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Compose output path
    out_path = out_dir / args.output_file_png

    # Delegate to the common plotter
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
