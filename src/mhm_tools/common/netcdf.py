#!/usr/bin/env python3
"""
Common NetCDF/xarray routines and utilities for reading, encoding, and bounds generation.
"""
import logging
from pathlib import Path
import glob
from typing import Any, List, Optional, Union

import xarray as xr
from .constants import NC_ENCODE_DEFAULTS, WILDCARDS

logger = logging.getLogger(__name__)

# Preserve original dataset attributes throughout processing
xr.set_options(keep_attrs=True)

def _has_wildcards(path: Union[str, Path]) -> bool:
    """
    Determine if the given path string contains any wildcard characters.
    """
    return any(w in str(path) for w in WILDCARDS)

def _fallback_open(open_func: Any, *args: Any, **kwargs: Any) -> xr.Dataset:
    """
    Open a dataset with the provided xarray function,
    falling back from h5netcdf to netcdf4 engine if necessary.
    """
    try:
        return open_func(*args, **kwargs)
    except ValueError as exc:
        if "unrecognized engine 'h5netcdf'" in str(exc):
            kwargs["engine"] = "netcdf4"
            return open_func(*args, **kwargs)
        logger.error("Error opening dataset: %s", exc)
        raise

def _normalize_lat_lon(ds: xr.Dataset) -> xr.Dataset:
    """
    Ensure Latitude and Longitude coordinates exist and normalize names for latitude/longitude.
    """
    coords = ds.variables
    if "latitude" in coords:
        ds = ds.rename_vars({"latitude": "lat"})
    elif "lat" not in coords:
        msg = "Dataset must contain 'lat' or 'latitude' variable"
        logger.error(msg)
        raise ValueError(msg)
    if "longitude" in coords:
        ds = ds.rename_vars({"longitude": "lon"})
    elif "lon" not in coords:
        msg = "Dataset must contain 'lon' or 'longitude' variable"
        logger.error(msg)
        raise ValueError(msg)
    return ds

def _assert_var(ds: xr.Dataset, var: str) -> None:
    """
    Verify that the specified variable exists in the dataset.
    """
    if var not in ds.variables:
        msg = f"Variable '{var}' not found in dataset"
        logger.error(msg)
        raise ValueError(msg)

def read_dataset(
    input_dir: Union[str, Path],
    file_name: str,
    var: str,
    use_mfdataset: bool = False,
    engine: str = "h5netcdf",
) -> xr.DataArray:
    """
    Read specified variable from one or more NetCDF files within the given directory,
    using file_name pattern (including wildcards) and engine fallback.
    Searches both the top-level input_dir and any subdirectories for matching files.
    """
    dir_path = Path(input_dir)
    pattern_path = dir_path / file_name
    path_str = str(pattern_path)

    if _has_wildcards(path_str):
        logger.debug(f"Reading NetCDF file(s) with pattern: {pattern_path}")
        # Find matches in top-level and subdirectories
        top_paths = sorted(dir_path.glob(file_name))
        sub_paths = sorted(p for p in dir_path.rglob(file_name) if p.parent != dir_path)
        combined = top_paths + sub_paths
        paths = [str(p) for p in sorted(combined, key=lambda x: str(x))]
        if not paths:
            msg = f"No files match pattern: {pattern_path}"
            logger.error(msg)
            raise FileNotFoundError(msg)
        if sub_paths:
            logger.debug(f"Found {len(sub_paths)} files in subdirectories of {dir_path}")
        if use_mfdataset:
            logger.debug("Using xr.open_mfdataset for multiple files")
            ds = _fallback_open(xr.open_mfdataset, paths=paths, engine=engine)
            ds = _normalize_lat_lon(ds)
            _assert_var(ds, var)
            return ds[var]
        arrays: List[xr.DataArray] = []
        logger.debug("Opening files individually with xr.open_dataset")
        for p in paths:
            try:
                ds_tmp = _fallback_open(xr.open_dataset, filename_or_obj=p, engine=engine)
            except Exception as exc:
                logger.error("Failed opening %s: %s", p, exc)
                raise
            ds_tmp = _normalize_lat_lon(ds_tmp)
            _assert_var(ds_tmp, var)
            arrays.append(ds_tmp[var])
        return xr.combine_by_coords(arrays, combine_attrs="override")

    # Single file case
    logger.debug(f"Reading single NetCDF file: {pattern_path}")
    try:
        ds = _fallback_open(xr.open_dataset, filename_or_obj=path_str, engine=engine)
    except Exception as exc:
        logger.error(f"Failed opening {path_str}: {exc}")
        raise
    ds = _normalize_lat_lon(ds)
    _assert_var(ds, var)
    return ds[var]

def set_netcdf_encoding(
    ds: xr.Dataset,
    var_encoding: Optional[dict] = None,
) -> None:
    """
    Set default NetCDF encoding settings on an xarray Dataset.
    """
    encoding = var_encoding or NC_ENCODE_DEFAULTS
    dims = set(ds.dims)
    coords = set(ds.coords)
    dim_coords = coords & dims
    aux_coords = coords - dims
    bnds = {
        ds[c].attrs.get("bounds")
        for c in coords
        if "bounds" in ds[c].attrs
    }
    data_vars = set(ds.data_vars) - bnds
    for name in aux_coords | data_vars:
        ds[name].encoding = encoding
    for name in dim_coords | bnds:
        ds[name].encoding = {"_FillValue": None}

def generate_bounds(
    da: xr.DataArray,
    bounds_dim: str = "bnds",
) -> xr.DataArray:
    """
    Generate CF-compliant bounds for a coordinate DataArray.
    """
    (dim,) = da.dims
    diff = da.diff(dim)
    lower = da - diff / 2
    upper = da + diff / 2
    bounds = xr.concat([lower, upper], dim=bounds_dim)
    first_lower = bounds.isel({dim: 0}) - diff.isel({dim: 0})
    first = first_lower.assign_coords({dim: da[dim][0]})
    all_bounds = xr.concat([first, bounds], dim=dim)
    return all_bounds.transpose()
