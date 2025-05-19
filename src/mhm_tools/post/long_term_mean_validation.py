#!/usr/bin/env python3

import os
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np

from mhm_tools.common.netcdf import read_dataset


def compute_difference(arr_model: np.ndarray, arr_reference: np.ndarray) -> np.ndarray:
    return arr_model - arr_reference


def plot_diff(
    diff: np.ndarray,
    lon: np.ndarray,
    lat: np.ndarray,
    cb_label: str,
    title: str,
    out_path: str,
    cmap: str = "coolwarm",
    x_min: Optional[float] = None,
    x_max: Optional[float] = None,
    y_min: Optional[float] = None,
    y_max: Optional[float] = None,
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
) -> None:
    plt.figure(figsize=(12, 6))

    # create the image with optional fixed vmin/vmax
    img = plt.imshow(
        diff,
        origin="upper",
        extent=[lon.min(), lon.max(), lat.min(), lat.max()],
        aspect="auto",
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
    )

    # apply axis limits if provided
    if x_min is not None or x_max is not None:
        plt.xlim(left=x_min, right=x_max)
    if y_min is not None or y_max is not None:
        plt.ylim(bottom=y_min, top=y_max)

    # colorbar with optional range annotation
    cb = plt.colorbar(img)
    if vmin is not None and vmax is not None:
        cb.set_label(f"{cb_label}")
    else:
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
    x_min: Optional[float] = None,
    x_max: Optional[float] = None,
    y_min: Optional[float] = None,
    y_max: Optional[float] = None,
    cmap: str = "coolwarm",
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
) -> None:
    # read the two DataArrays
    ref_pattern = str(Path(ref_input_dir) / reference_pattern)
    ds_ref = read_dataset(
        file_path=ref_pattern,
    )
    mod_pattern = str(Path(mod_input_dir) / model_pattern)
    ds_mod = read_dataset(
        file_path=mod_pattern,
    )
    da_ref = da_ref[ref_var]
    da_mod = ds_mod[mod_var]

    # drop singleton time dim so we have 2D arrays
    arr_ref = np.squeeze(da_ref.values)
    arr_mod = np.squeeze(da_mod.values)

    lon = da_mod["lon"].values
    lat = da_mod["lat"].values

    diff = compute_difference(arr_mod, arr_ref)

    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, output_file)

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
