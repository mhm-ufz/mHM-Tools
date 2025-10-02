#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Regrid an input NetCDF to an L2 grid that is an integer multiple of the L0 grid in mask.nc.
Prefers keeping the L2 grid aligned to the L0 grid.

USAGE:
  python regrid_to_L2.py --input in.nc --mask mask.nc --output out.nc \
      --l2 0.05           --method nearest
  python regrid_to_L2.py --input in.nc --mask mask.nc --output out.nc \
      --l2 0.10x0.10      --method bilinear
  python regrid_to_L2.py --input in.nc --mask mask.nc --output out.nc \
      --l2 0.02,0.02      --method linear --var temp

METHODS:
  nearest  -> xarray .interp(method="nearest")
  linear   -> xarray .interp(method="linear")
  bilinear -> CDO remapbil if cdo is available, else fallback to xarray "linear"

Notes:
- Assumes regular lon/lat grids.
- L2 must be an integer multiple of L0 in both x and y.
"""

import logging
from pathlib import Path
import tempfile
from mhm_tools.common.file_handler import get_xarray_ds_from_file, write_xarray_to_file
from mhm_tools.common.xarray_utils import get_coord_key
import numpy as np
import xarray as xr
logger = logging.getLogger(__name__)

# Optional CDO support
try:
    from cdo import Cdo  # type: ignore
    _CDO = Cdo()
except Exception:
    _CDO = None


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
    return abs(k - round(k)) <= tol, int(round(k))

def _build_aligned_coords(vmin, vmax, step):
    # Build coordinates aligned to vmin with spacing `step`, stopping before overshoot.
    n = int(np.floor((vmax - vmin) / step + 0.5)) + 1
    return vmin + step * np.arange(n, dtype=float)

def _make_cdo_grid_file(lon, lat, path):
    # CDO grid description (lonlat)
    xfirst, xinc, xsize = float(lon[0]), float(lon[1]-lon[0]), int(len(lon))
    yfirst, yinc, ysize = float(lat[0]), float(lat[1]-lat[0]), int(len(lat))
    txt = (
        "gridtype = lonlat\n"
        f"xsize   = {xsize}\n"
        f"ysize   = {ysize}\n"
        f"xfirst  = {xfirst}\n"
        f"xinc    = {xinc}\n"
        f"yfirst  = {yfirst}\n"
        f"yinc    = {yinc}\n"
    )
    with open(path, "w") as f:
        f.write(txt)

def regrid_xarray(ds, lon_name, lat_name, lon_target, lat_target, method, var=None):
    # Select variables to regrid
    if var:
        dvs = [var]
    else:
        dvs = [v for v in ds.data_vars if lon_name in ds[v].dims and lat_name in ds[v].dims]
    # Build target as dict for .interp
    target = {lon_name: xr.DataArray(lon_target, dims=(lon_name,)),
              lat_name: xr.DataArray(lat_target, dims=(lat_name,))}
    # Interpolate each variable; keep others as-is
    out_vars = {}
    for v in ds.data_vars:
        if v in dvs:
            out_vars[v] = ds[v].interp(target, method="nearest" if method=="nearest" else "linear")
        else:
            out_vars[v] = ds[v]
    out = xr.Dataset(out_vars, coords={lon_name: target[lon_name], lat_name: target[lat_name]})
    # Copy attrs
    out.attrs.update(ds.attrs)
    return out

def regrid_cdo(in_file, out_file, lon_target, lat_target, method):
    assert _CDO is not None
    with tempfile.TemporaryDirectory() as tmpd:
        gridfile = Path(tmpd) / "grid.txt"
        _make_cdo_grid_file(lon_target, lat_target, gridfile)
        op = {"nearest": _CDO.remapnn, "bilinear": _CDO.remapbil, "linear": _CDO.remapbil}[method]
        op(gridfile, input=in_file, output=out_file, options="-O")

def regrid_file(input, mask, output, l2, method="nearest", var=None):

    # p.add_argument("--var", default=None, help="Single variable to regrid (default: all 2D/3D lon-lat vars)")

    # Load mask to infer L0 grid
    dsm = get_xarray_ds_from_file(mask)
    if 'mask_l2' in dsm.data_vars:
        logger.info('Mask has L2 version')
        dam_l2 = dsm['mask_l2']
        logger.info(dam_l2)
        lon_name = get_coord_key(dam_l2, lon=True)
        lat_name = get_coord_key(dam_l2, lat=True)
        lonL2 = dsm[lon_name].values
        latL2 = dsm[lat_name].values
        
    else: 
        lon_name, lat_name = get_coord_key(dsm, lon=True), get_coord_key(dsm, lat=True)
        lon0 = dsm[lon_name].values
        lat0 = dsm[lat_name].values
        if lon0.ndim != 1 or lat0.ndim != 1:
            raise ValueError("This script assumes 1D lon/lat coordinates.")
        l0_dx = _delta_from_coords(lon0)
        l0_dy = _delta_from_coords(lat0)

        # l2_dx, l2_dy = _parse_res(l2)

        okx, kx = _check_integer_multiple(l2, l0_dx)
        oky, ky = _check_integer_multiple(l2, l0_dy)
        if not (okx and oky):
            raise ValueError(f"L2 ({l2},{l2}) must be integer multiples of L0 ({l0_dx:.12g},{l0_dy:.12g}).")

        # Build aligned L2 coords covering the same extent as mask grid
        lonL2 = _build_aligned_coords(lon0.min(), lon0.max(), l2)
        latL2 = _build_aligned_coords(lat0.min(), lat0.max(), l2)

    # Load input
    dsi = get_xarray_ds_from_file(input)

    # If CDO available and method == bilinear, use it (fast & robust); else use xarray
    if _CDO is not None and method in ("bilinear",) and var is None:
        # Write temp input (ensure lon/lat names are lon/lat for CDO best behavior)
        # If names differ, rename temporarily
        rename_map = {}
        if lon_name not in dsi.coords:
            # try to find lon/lat names in input; fall back to mask names
            try:
                in_lon, in_lat = get_coord_key(dsi, lon=True), get_coord_key(dsi, lat=True)
            except Exception:
                in_lon, in_lat = lon_name, lat_name
        else:
            in_lon, in_lat = lon_name, lat_name

        tmp_in = input
        renamed = False
        if (in_lon, in_lat) != ("lon", "lat"):
            dsi_renamed = dsi.rename({in_lon: "lon", in_lat: "lat"})
            tmp_in = tempfile.mktemp(suffix=".nc")
            dsi_renamed.to_netcdf(tmp_in)
            renamed = True

        tmp_out = tempfile.mktemp(suffix=".nc")
        regrid_cdo(tmp_in, tmp_out, lonL2, latL2, "bilinear")
        out = get_xarray_ds_from_file(tmp_out)
        # Rename back if needed
        if renamed:
            out = out.rename({"lon": in_lon, "lat": in_lat})
        out.to_netcdf(output)
        logger.info(f"Wrote {output}")
    else:
        # xarray path (nearest/linear; bilinear falls back to linear)
        method = method if method != "bilinear" else "linear"
        # Align input lon/lat names for xarray
        try:
            in_lon = get_coord_key(dsi, lon=True)
            in_lat = get_coord_key(dsi, lat=True)
        except Exception:
            in_lon, in_lat = lon_name, lat_name
        out = regrid_xarray(dsi, in_lon, in_lat, lonL2, latL2, method, var=var)
        encoding = {v: {"zlib": True, "complevel": 4} for v in out.data_vars}
        write_xarray_to_file(out, output, encoding)
        print(f"Wrote {output}")

def regrid(input, mask, output, l2=None, method="nearest", var=None):
    input = Path(input)
    if input.is_dir():
        input_dir = input
        files = input.rglob('*.nc')
    elif input.is_file():
        input_dir = input.parent
        files = [input]
    else: 
        raise ValueError("Input is neither file nor dir.")
    for file_input in files: 
        output_name = file_input.name
        output_path = output
        if output.suffix:
            output_name = output.name
            output_path = output.parent
        file_output = output_path / file_input.parent.relative_to(input_dir) / output_name
        print(file_input, file_output)
        regrid_file(file_input, mask, file_output, l2, method, var)