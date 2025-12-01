"""
Calculate the spatial difference between reference and model datasets.

This script is intended for NetCDF files that represent *aggregated fields*
(such as long-term averages, climatologies, or single-time snapshots), rather
than full time series. It loads the reference and model datasets, interpolates
the model onto the reference grid, and then computes the difference
(reference - model). The result is visualized as a map and, optionally, written
to a NetCDF file.

Authors
-------
- Jeisson Leal
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from mhm_tools.common.file_handler import write_xarray_to_file
from mhm_tools.common.netcdf import read_dataset
from mhm_tools.common.plotter import plot_map
from mhm_tools.common.xarray_utils import get_coord_key, normalize_lat_lon


@dataclass
class PlotOptions:
    """Plotting options for the difference map."""

    colorbar_label: str
    title: str
    cmap: str = "RdBu"
    vmin: Optional[float] = None
    vmax: Optional[float] = None
    x_min: Optional[float] = None
    x_max: Optional[float] = None
    y_min: Optional[float] = None
    y_max: Optional[float] = None


@dataclass
class OutputOptions:
    """Output configuration for generated PNG/NetCDF files."""

    output_dir: str
    output_file_png: str
    save_ncfile: bool
    output_file_nc: str


def calc_diff(
    ref_input_dir: str,
    mod_input_dir: str,
    reference_pattern: str,
    model_pattern: str,
    ref_var: str,
    mod_var: str,
    plot: PlotOptions,
    output: OutputOptions,
) -> None:
    """Compute long-term mean difference between model and reference datasets and plot the result."""
    ds_ref = read_dataset(file_path=str(Path(ref_input_dir) / reference_pattern))
    ds_mod = read_dataset(file_path=str(Path(mod_input_dir) / model_pattern))

    # Get coordinate names
    ref_lat = get_coord_key(ds_ref, lat=True)
    ref_lon = get_coord_key(ds_ref, lon=True)
    mod_lat = get_coord_key(ds_mod, lat=True)
    mod_lon = get_coord_key(ds_mod, lon=True)

    # Normalize lat/lon names and squeeze time dimension
    da_ref = normalize_lat_lon(ds_ref, lat=ref_lat, lon=ref_lon)[ref_var].squeeze(
        "time"
    )
    da_mod = normalize_lat_lon(ds_mod, lat=mod_lat, lon=mod_lon)[mod_var].squeeze(
        "time"
    )

    # Interpolate model to reference grid to avoid alignment errors
    da_mod_interp = da_mod.interp_like(da_ref)

    # Compute difference
    diff = da_ref - da_mod_interp

    # Prepare output directory
    out_path_dir = Path(output.output_dir)
    out_path_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_path_dir / output.output_file_png

    # Plot difference map
    plot_map(
        data=diff,
        cb_label=plot.colorbar_label,
        title=plot.title,
        out_path=out_path,
        cmap=plot.cmap,
        x_min=plot.x_min,
        x_max=plot.x_max,
        y_min=plot.y_min,
        y_max=plot.y_max,
        vmin=plot.vmin,
        vmax=plot.vmax,
    )

    # Optionally save NetCDF
    if output.save_ncfile:
        write_xarray_to_file(ds=diff, file_path=out_path_dir / output.output_file_nc)
