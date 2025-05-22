"""
Compute and plot long-term mean differences between model and reference datasets.

This module reads one or more CF-compliant NetCDF datasets, computes the
long-term mean fields for both model outputs and reference data, calculates
their differences, and generates spatial and temporal plots of those differences.

Authors
-------
- Jeisson Leal
"""

from pathlib import Path
from typing import Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np

from mhm_tools.common.netcdf import read_dataset


def compute_difference(arr_model: np.ndarray, arr_reference: np.ndarray) -> np.ndarray:
    """Compute the element-wise difference between a model and reference array."""
    return arr_model - arr_reference


def plot_diff(
    diff: np.ndarray,
    lon: np.ndarray,
    lat: np.ndarray,
    cb_label: str,
    title: str,
    out_path: Path,
    cmap: str = "coolwarm",
    x_min: Optional[float] = None,
    x_max: Optional[float] = None,
    y_min: Optional[float] = None,
    y_max: Optional[float] = None,
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
) -> None:
    """Plot and save the difference field over longitude and latitude."""
    plt.figure(figsize=(12, 6))

    img = plt.imshow(
        diff,
        origin="upper",
        extent=[lon.min(), lon.max(), lat.min(), lat.max()],
        aspect="auto",
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
    )

    if x_min is not None or x_max is not None:
        plt.xlim(left=x_min, right=x_max)
    if y_min is not None or y_max is not None:
        plt.ylim(bottom=y_min, top=y_max)

    cb = plt.colorbar(img)
    cb.set_label(cb_label)

    plt.title(title)
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.show()
    plt.close()


def long_term_mean_diff(
    ref_input_dir: str,
    mod_input_dir: str,
    reference_pattern: str,
    model_pattern: str,
    ref_var: str,
    mod_var: str,
    colorbar_label: str,
    title: str,
    output_dir: str,
    output_file: str,
    extent: Optional[Tuple[float, float, float, float]] = None,
    cmap: str = "coolwarm",
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
) -> None:
    """
    Compute long-term mean difference between model and
    reference datasets and plot the result.
    """
    # Read the reference and model datasets
    ds_ref = read_dataset(file_path=str(Path(ref_input_dir) / reference_pattern))
    ds_mod = read_dataset(file_path=str(Path(mod_input_dir) / model_pattern))

    da_ref = ds_ref[ref_var]
    da_mod = ds_mod[mod_var]

    # Extract 2D arrays
    arr_ref = np.squeeze(da_ref.values)
    arr_mod = np.squeeze(da_mod.values)

    lon = da_mod["lon"].values
    lat = da_mod["lat"].values

    diff = compute_difference(arr_mod, arr_ref)

    # Ensure output directory exists
    out_path_dir = Path(output_dir)
    out_path_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_path_dir / output_file

    # Unpack extent limits
    x_min = x_max = y_min = y_max = None
    if extent is not None:
        x_min, x_max, y_min, y_max = extent

    plot_diff(
        diff=diff,
        lon=lon,
        lat=lat,
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
