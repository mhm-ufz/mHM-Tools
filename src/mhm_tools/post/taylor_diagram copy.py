from pathlib import Path
from typing import List
import os

import numpy as np
import matplotlib.pyplot as plt
import xarray as xr
from easy_mpl import taylor_plot

from mhm_tools.common.netcdf import read_dataset
from mhm_tools.common.xarray_utils import get_coord_key, normalize_lat_lon

def calc_tim_mean(da: xr.DataArray) -> xr.DataArray:
    if "time" not in da.dims:
        raise ValueError("Input DataArray must have a 'time' dimension")
    if da.sizes["time"] > 1:
        return da.mean(dim=["lat", "lon"])
    return da.isel(time=0).mean(dim=["lat", "lon"])

def prepare_da(input_dir, pattern, var_name):
    path = os.path.join(input_dir, pattern)
    ds = read_dataset(path)
    lat = get_coord_key(ds, lat=True)
    lon = get_coord_key(ds, lon=True)
    ds = normalize_lat_lon(ds, lat, lon)
    return ds[var_name]

def generate_taylor_diagram(
    ref_input_dir: str,
    reference_pattern: str,
    ref_var: str,
    ref_label: str,
    mod_input_dirs: List[str],
    model_patterns: List[str],
    mod_vars: List[str],
    mod_labels: List[str],
    title: str,
    output_dir: str,
    output_file: str,
) -> None:
    # 1) Ensure output directory exists
    outdir = Path(output_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    # 2) Load and process the reference time series
    da_ref = prepare_da(ref_input_dir, reference_pattern, ref_var)
    ref_values = calc_tim_mean(da_ref).values

    # 3) Build the observations dict for multi-plot mode
    observations = {
        ref_label: ref_values
    }

    # 4) Initialize the simulations dict with the same top-level key
    simulations = {
        ref_label: {}
    }

    # 5) Loop over each model directory/pattern/var/label and fill the nested dict
    for mod_dir, mod_pattern, mod_var, mod_label in zip(
        mod_input_dirs, model_patterns, mod_vars, mod_labels
    ):
        # a) read & normalize the DataArray
        da_model = prepare_da(mod_dir, mod_pattern, mod_var)
        # b) compute the spatially-averaged time series
        model_series = calc_tim_mean(da_model).values
        # c) insert into the nested dict under the ref_label key
        simulations[ref_label][mod_label] = model_series

    # 6) Generate the Taylor diagram (one subplot named by ref_label)
    fig = taylor_plot(
        observations=observations,
        simulations=simulations,
        cont_kws={'colors': 'blue', 'linewidths': 1.0, 'linestyles': 'dotted'},
        grid_kws={'axis': 'x', 'color': 'g', 'lw': 1.0},
        title=title or None,
    )

    # 7) Save and close
    fig.savefig(Path(output_dir) / output_file, dpi=300, bbox_inches="tight")
    plt.close(fig)
