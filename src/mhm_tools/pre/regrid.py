#!/usr/bin/env python3

"""Regrid NetCDF data to an L2 grid aligned with an L0 mask.

The module derives a target grid from mask.nc and an L2 resolution, checks that
the requested L2 cells are integer multiples of the L0 grid, interpolates the
selected variable, and writes an aligned NetCDF output.

USAGE:
  python regrid_to_L2.py --input in.nc --mask mask.nc --output out.nc \
      --l2 0.05           --method nearest
  python regrid_to_L2.py --input in.nc --mask mask.nc --output out.nc \
      --l2 0.10x0.10      --method bilinear
  python regrid_to_L2.py --input in.nc --mask mask.nc --output out.nc \
      --l2 0.02,0.02      --method linear --var temp

Methods
-------
  nearest  -> xarray .interp(method="nearest")
  linear   -> xarray .interp(method="linear")

Notes
-----
- Assumes regular lon/lat grids.
- L2 must be an integer multiple of L0 in both x and y.

Authors
-------
- Simon Lüdke
"""

import logging
from pathlib import Path

import numpy as np
import xarray as xr

from mhm_tools.common.constants import NC_ENCODE_DEFAULTS
from mhm_tools.common.file_handler import get_xarray_ds_from_file, write_xarray_to_file
from mhm_tools.common.logger import ErrorLogger
from mhm_tools.common.xarray_utils import get_coord_key

logger = logging.getLogger(__name__)


def _delta_from_coords(vals: np.ndarray) -> float:
    # robust median step (handles ascending or descending)
    diffs = np.diff(vals)
    return float(np.median(np.abs(diffs)))


def _parse_res(s: str):
    s = s.strip().lower().replace(" ", "")
    if "x" in s or "," in s:
        sep = "x" if "x" in s else ","
        a, b = s.split(sep)
        return float(a), float(b)
    return float(s), float(s)


def _check_integer_multiple(l2, l0, tol=1e-9):
    k = l2 / l0
    return abs(k - round(k)) <= tol, round(k)


def _build_aligned_coords(vmin, vmax, step):
    # Build coordinates aligned to vmin with spacing `step`, stopping before overshoot.
    n = int(np.floor((vmax - vmin) / step + 0.5)) + 1
    return vmin + step * np.arange(n, dtype=float)


def regrid_xarray(ds, lon_name, lat_name, lon_target, lat_target, method, var=None):
    """Regrid an xarray Dataset using xarray interpolation."""
    # Select variables to regrid
    if var:
        dvs = [var]
    else:
        dvs = [
            v for v in ds.data_vars if lon_name in ds[v].dims and lat_name in ds[v].dims
        ]
    # Build target as dict for .interp
    target = {
        lon_name: xr.DataArray(lon_target, dims=(lon_name,)),
        lat_name: xr.DataArray(lat_target, dims=(lat_name,)),
    }

    # Create a DataArray for every variable, regridding those selected
    das = []
    interp_method = "nearest" if method == "nearest" else "linear"
    for v in ds.data_vars:
        da = ds[v].interp(target, method=interp_method) if v in dvs else ds[v]
        # ensure name consistency
        if da.name != v:
            da = da.rename(v)
        das.append(da)
        # Merge all DataArrays into a single Dataset and set target coords
    out = xr.merge(das)
    out = out.assign_coords({lon_name: target[lon_name], lat_name: target[lat_name]})

    # Copy attrs
    out.attrs.update(ds.attrs)
    return out


def regrid_file(input, mask, output, l2, method="nearest", var=None):
    """Regrid a single file to L2 grid using xarray."""
    # p.add_argument("--var", default=None, help="Single variable to regrid (default: all 2D/3D lon-lat vars)")

    # Load mask to infer L0 grid
    dsm = get_xarray_ds_from_file(mask)
    if "mask_l2" in dsm.data_vars:
        logger.info("Mask has L2 version")
        dam_l2 = dsm["mask_l2"]
        logger.info(dam_l2)
        lon_name = get_coord_key(dam_l2, lon=True)
        lat_name = get_coord_key(dam_l2, lat=True)
        lonL2 = dsm[lon_name].data
        latL2 = dsm[lat_name].data

    else:
        lon_name = get_coord_key(dsm, lon=True)
        lat_name = get_coord_key(dsm, lat=True)
        lon0 = dsm[lon_name].data
        lat0 = dsm[lat_name].data
        if lon0.ndim != 1 or lat0.ndim != 1:
            msg = "This script assumes 1D lon/lat coordinates."
            with ErrorLogger(logger):
                raise ValueError(msg)
        l0_dx = _delta_from_coords(lon0)
        l0_dy = _delta_from_coords(lat0)

        # l2_dx, l2_dy = _parse_res(l2)

        okx, kx = _check_integer_multiple(l2, l0_dx)
        oky, ky = _check_integer_multiple(l2, l0_dy)
        if not (okx and oky):
            msg = (
                "L2 "
                f"({l2},{l2}) must be integer multiples of L0 ({l0_dx:.12g},{l0_dy:.12g})."
            )
            with ErrorLogger(logger):
                raise ValueError(msg)

        # Build aligned L2 coords covering the same extent as mask grid
        lonL2 = _build_aligned_coords(lon0.min(), lon0.max(), l2)
        latL2 = _build_aligned_coords(lat0.min(), lat0.max(), l2)
    logger.info(f"{lon_name} {lonL2}")
    logger.info(f"{lat_name} {latL2}")
    # Load input
    dsi = get_xarray_ds_from_file(input)

    # xarray path (nearest/linear; bilinear falls back to linear)
    method = method if method != "bilinear" else "linear"
    # Align input lon/lat names for xarray
    try:
        in_lon = get_coord_key(dsi, lon=True)
        in_lat = get_coord_key(dsi, lat=True)
    except Exception:
        in_lon, in_lat = lon_name, lat_name
    logger.info(f"regrid with xarray {method} interpolation")
    out = regrid_xarray(dsi, in_lon, in_lat, lonL2, latL2, method, var=var)
    encoding = {
        v: {"zlib": True, "complevel": 4, **NC_ENCODE_DEFAULTS} for v in out.data_vars
    }
    logger.info(out)
    write_xarray_to_file(out, output, encoding=encoding)
    logger.info(f"Wrote {output}")


def regrid(input, mask, output, l2=None, method="nearest", var=None):
    """Regrid file(s) to an L2 grid."""
    input = Path(input)
    if input.is_dir():
        input_dir = input
        files = input.rglob("*.nc")
    elif input.is_file():
        input_dir = input.parent
        files = [input]
    else:
        msg = "Input is neither file nor dir."
        with ErrorLogger(logger):
            raise ValueError(msg)
    for file_input in files:
        output_name = file_input.name
        output_path = output
        if output.suffix:
            output_name = output.name
            output_path = output.parent
        file_output = (
            output_path / file_input.parent.relative_to(input_dir) / output_name
        )
        logger.info(f"{file_input} -> {file_output}")
        regrid_file(file_input, mask, file_output, l2, method, var)
