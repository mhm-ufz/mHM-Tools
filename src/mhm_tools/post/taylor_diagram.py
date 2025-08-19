"""
Generate Taylor diagrams for evaluating model performance against a reference dataset.

This script is intended for NetCDF files containing time series of aggregated
variables. It computes spatially averaged time series (mean over lat/lon),
removes NaNs consistently across reference and model series, and optionally
normalizes values by the standard deviation of the reference. Multiple models
can be compared against a single reference within one Taylor diagram.

The workflow includes:
- loading reference and model NetCDF datasets,
- interpolating latitude/longitude naming and orientation,
- computing time-mean spatial series,
- aligning data and masking invalid values,
- plotting a Taylor diagram (via easy_mpl) to assess correlation, centered RMSD, and variance,
- saving the result as a PNG file.

Authors
-------
- Jeisson Leal
"""

import os
from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from easy_mpl import taylor_plot

from mhm_tools.common.netcdf import read_dataset
from mhm_tools.common.xarray_utils import get_coord_key, normalize_lat_lon


def calc_tim_mean(da: xr.DataArray) -> xr.DataArray:
    if "time" not in da.dims:
        raise ValueError("Input DataArray must have a 'time' dimension")
    if da.sizes["time"] <= 1:
        raise ValueError(
            "Input DataArray must have more than one time step for Taylor plot. "
            f"Found only {da.sizes['time']} time step(s)."
        )
    return da.mean(dim=["lat", "lon"], skipna=True)


def prepare_da(input_dir, pattern, var_name):
    path = os.path.join(input_dir, pattern)
    ds = read_dataset(path)
    lat = get_coord_key(ds, lat=True)
    lon = get_coord_key(ds, lon=True)
    ds = normalize_lat_lon(ds, lat, lon)
    return ds[var_name]


def mask_nan(obs: np.ndarray, sims: dict) -> tuple[np.ndarray, dict]:
    """
    Mask out NaNs from observation and simulation arrays, keeping only indices
    where all arrays have valid data.
    """
    mask = ~np.isnan(obs)
    for sim_vals in sims.values():
        mask &= ~np.isnan(sim_vals)

    obs_clean = obs[mask]
    sims_clean = {k: v[mask] for k, v in sims.items()}
    return obs_clean, sims_clean


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
    normalize: bool = False,
) -> None:
    # 1) Ensure output directory exists
    outdir = Path(output_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    # 2) Load and process the reference time series
    da_ref = prepare_da(ref_input_dir, reference_pattern, ref_var)
    ref_values = calc_tim_mean(da_ref).values

    # 3) Build the observations dict for multi-plot mode
    observations = {ref_label: ref_values}

    # 4) Initialize the simulations dict with the same top-level key
    simulations = {ref_label: {}}

    # 5) Loop over each model directory/pattern/var/label and fill the nested dict
    for mod_dir, mod_pattern, mod_var, mod_label in zip(
        mod_input_dirs, model_patterns, mod_vars, mod_labels
    ):
        da_model = prepare_da(mod_dir, mod_pattern, mod_var)
        model_series = calc_tim_mean(da_model).values
        simulations[ref_label][mod_label] = model_series

    # 6) Remove NaNs and align all time series
    obs_clean, sims_clean = mask_nan(ref_values, simulations[ref_label])

    # 7) Normalize if requested
    if normalize:
        obs_std = np.std(obs_clean)
        if obs_std == 0:
            raise ValueError(
                "Standard deviation of observations is zero, cannot normalize."
            )
        obs_clean = obs_clean / obs_std
        sims_clean = {k: v / obs_std for k, v in sims_clean.items()}

    observations_clean = {ref_label: obs_clean}
    simulations_clean = {ref_label: sims_clean}

    # 8) Generate the Taylor diagram (one subplot named by ref_label)
    fig = taylor_plot(
        observations=observations_clean,
        simulations=simulations_clean,
        cont_kws={"colors": "blue", "linewidths": 1.0, "linestyles": "dotted"},
        grid_kws={"axis": "x", "color": "g", "lw": 1.0},
        title=title or None,
    )

    # 9) Save and close
    fig.savefig(Path(output_dir) / output_file, dpi=300, bbox_inches="tight")
    plt.close(fig)
