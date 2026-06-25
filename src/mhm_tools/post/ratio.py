"""
Compute the spatial ratio between model and reference datasets.

This script is intended for NetCDF files that represent aggregated fields
(e.g., long-term averages, climatologies, or single-time snapshots), rather
than full time series. It interpolates the model dataset onto the reference
grid to ensure spatial alignment and then computes the ratio (model ÷ reference)
at each grid cell. Division by zero is handled safely by masking with NaN.

The result is plotted as a map and, if requested, saved as a NetCDF file.

Authors
-------
- Jeisson Leal
"""

from pathlib import Path
from typing import Optional

import numpy as np
import xarray as xr

from mhm_tools.common.file_handler import write_xarray_to_file
from mhm_tools.common.netcdf import read_dataset
from mhm_tools.common.plotter import plot_map
from mhm_tools.common.xarray_utils import get_coord_key, normalize_lat_lon


def calc_ratio(  # noqa: PLR0913
    ref_input_dir: str,
    mod_input_dir: str,
    reference_pattern: str,
    model_pattern: str,
    ref_var: str,
    mod_var: str,
    colorbar_label: str,
    title: str,
    output_dir: str,
    output_file_png: str,
    save_ncfile: bool,
    output_file_nc: str,
    x_min: Optional[float] = None,
    x_max: Optional[float] = None,
    y_min: Optional[float] = None,
    y_max: Optional[float] = None,
    cmap: str = "RdBu",
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
) -> None:
    """Compute long-term mean ratio between model and reference datasets and plot the result."""
    ds_ref = read_dataset(file_path=str(Path(ref_input_dir) / reference_pattern))
    ds_mod = read_dataset(file_path=str(Path(mod_input_dir) / model_pattern))

    # Gets coords names
    ref_lat = get_coord_key(ds_ref, lat=True)
    ref_lon = get_coord_key(ds_ref, lon=True)
    mod_lat = get_coord_key(ds_mod, lat=True)
    mod_lon = get_coord_key(ds_mod, lon=True)

    # Sets lon and lat names and lat and lon and returns the da
    da_ref = normalize_lat_lon(ds_ref, lat_key=ref_lat, lon_key=ref_lon)[
        ref_var
    ].squeeze("time")
    da_mod = normalize_lat_lon(ds_mod, lat_key=mod_lat, lon_key=mod_lon)[
        mod_var
    ].squeeze("time")

    # Interpolate model to reference grid to avoid alignment errors
    da_mod_interp = da_mod.interp_like(da_ref)

    # calculating ratio, if true prevents division by 0
    ratio = xr.where(da_ref != 0, da_mod_interp / da_ref, np.nan)

    # Sets output path to save plot
    out_path_dir = Path(output_dir)
    out_path_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_path_dir / output_file_png

    plot_map(
        data=ratio,
        cb_label=colorbar_label,
        title=title,
        out_path=out_path,
        cmap=cmap,
        x_min=x_min,
        x_max=x_max,
        y_min=y_min,
        y_max=y_max,
        vmin=vmin,
        vmax=vmax,
    )

    # If set, saves ratio file
    if save_ncfile:
        write_xarray_to_file(ds=ratio, file_path=out_path_dir / output_file_nc)
