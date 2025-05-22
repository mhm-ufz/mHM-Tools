"""Common NetCDF/xarray routines and utilities for reading, encoding, and bounds generation."""

import logging
from pathlib import Path
from typing import Any, List, Optional, Union

import xarray as xr

from .constants import NC_ENCODE_DEFAULTS, WILDCARDS

logger = logging.getLogger(__name__)

# Preserve original dataset attributes throughout processing
xr.set_options(keep_attrs=True)


def _has_wildcards(path: Union[str, Path]) -> bool:
    """Determine if the given path string contains any wildcard characters."""
    return any(w in str(path) for w in WILDCARDS)


def _fallback_open(open_func: Any, *args: Any, **kwargs: Any) -> xr.Dataset:
    """Open a dataset with the provided xarray function, falling back from
    h5netcdf to netcdf4 engine if necessary."""
    try:
        return open_func(*args, **kwargs)
    except ValueError as exc:
        if "unrecognized engine 'h5netcdf'" in str(exc):
            kwargs["engine"] = "netcdf4"
            return open_func(*args, **kwargs)
        logger.error("Error opening dataset: %s", exc)
        raise


def read_dataset(
    file_path: Union[str, Path, List[Union[str, Path]]],
    use_mfdataset: bool = False,
    engine: str = "h5netcdf",
) -> xr.Dataset:
    """
    Load one or more NetCDF files into a single xarray.Dataset.

    This function accepts either a single path (possibly containing
    shell-style wildcards), or a list of paths, and handles both:

    - Single-file case: opens it directly.
    - Multi-file case:
        * If `use_mfdataset=True`, uses `xarray.open_mfdataset` for
          contiguous datasets.
        * Otherwise, opens each file individually and combines them
          via coordinates (attributes overridden).

    Parameters
    ----------
    file_path : Union[str, Path, List[Union[str, Path]]]
        A file path (can include wildcards), or a list of explicit
        file paths.
    use_mfdataset : bool, default False
        If True and multiple files are found, open them with
        `xr.open_mfdataset`. Otherwise, open each with
        `xr.open_dataset` and combine.
    engine : str, default "h5netcdf"
        The backend engine to use for opening NetCDF files.

    Returns
    -------
    xr.Dataset
        An xarray Dataset containing the data from the specified file(s).

    Raises
    ------
    FileNotFoundError
        If a wildcard pattern is provided but no files match.
    Exception
        Any exception raised by xarray when opening files is propagated
        after being logged.

    Examples
    --------
    Read a single file:

    >>> ds = read_dataset("data/single_file.nc")

    Read all files in a directory (recursively):

    >>> ds = read_dataset("data/**/*.nc", use_mfdataset=True)

    Read a fixed list of files:

    >>> paths = ["data/part1.nc", "data/part2.nc"]
    >>> ds = read_dataset(paths)

    """
    # Normalize to list of string paths
    if isinstance(file_path, (list, tuple)):
        paths = [str(p) for p in file_path]
        logger.debug(f"Received explicit list of {len(paths)} paths")
    else:
        pattern = str(file_path)
        if _has_wildcards(pattern):
            logger.debug(f"Globbing for pattern: {pattern}")
            matches = sorted(Path().glob(pattern))
            paths = [str(p) for p in matches]
            if not paths:
                msg = f"No files match pattern: {pattern}"
                logger.error(msg)
                raise FileNotFoundError(msg)
            logger.debug(f"Found {len(paths)} files via glob")
        else:
            paths = [pattern]
            logger.debug(f"No wildcards: using single file path {pattern}")

    # Multi-file
    if len(paths) > 1:
        logger.debug(f"{len(paths)} files to open; use_mfdataset={use_mfdataset}")
        if use_mfdataset:
            try:
                ds = _fallback_open(xr.open_mfdataset, paths=paths, engine=engine)
            except Exception as exc:
                logger.error(f"open_mfdataset failed on {paths!r}: {exc}")
                raise
            return ds
        arrays = []
        for p in paths:
            logger.debug(f"Opening (single) {p}")
            try:
                ds_tmp = _fallback_open(
                    xr.open_dataset, filename_or_obj=p, engine=engine
                )
            except Exception as exc:
                logger.error(f"Failed opening {p}: {exc}")
                raise
            arrays.append(ds_tmp)
        return xr.combine_by_coords(arrays, combine_attrs="override")

    # Single-file case
    single = paths[0]
    logger.debug(f"Reading single NetCDF file: {single}")
    try:
        ds = _fallback_open(xr.open_dataset, filename_or_obj=single, engine=engine)
    except Exception as exc:
        logger.error(f"Failed opening {single}: {exc}")
        raise
    return ds


def set_netcdf_encoding(
    ds: xr.Dataset,
    var_encoding: Optional[dict] = None,
) -> None:
    """Set default NetCDF encoding settings on an xarray Dataset."""
    encoding = var_encoding or NC_ENCODE_DEFAULTS
    dims = set(ds.dims)
    coords = set(ds.coords)
    dim_coords = coords & dims
    aux_coords = coords - dims
    bnds = {ds[c].attrs.get("bounds") for c in coords if "bounds" in ds[c].attrs}
    data_vars = set(ds.data_vars) - bnds
    for name in aux_coords | data_vars:
        ds[name].encoding = encoding
    for name in dim_coords | bnds:
        ds[name].encoding = {"_FillValue": None}


def generate_bounds(
    da: xr.DataArray,
    bounds_dim: str = "bnds",
) -> xr.DataArray:
    """Generate CF-compliant bounds for a coordinate DataArray."""
    (dim,) = da.dims
    diff = da.diff(dim)
    lower = da - diff / 2
    upper = da + diff / 2
    bounds = xr.concat([lower, upper], dim=bounds_dim)
    first_lower = bounds.isel({dim: 0}) - diff.isel({dim: 0})
    first = first_lower.assign_coords({dim: da[dim][0]})
    all_bounds = xr.concat([first, bounds], dim=dim)
    return all_bounds.transpose()
