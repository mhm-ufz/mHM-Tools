"""Common NetCDF/xarray routines and utilities for reading, encoding, and bounds generation."""

import logging
from pathlib import Path
from typing import Any, List, Optional, Union

import numpy as np
import xarray as xr

from mhm_tools.common.logger import ErrorLogger

from .constants import NC_ENCODE_DEFAULTS, NO_DATA, WILDCARDS

logger = logging.getLogger(__name__)

# Preserve original dataset attributes throughout processing
xr.set_options(keep_attrs=True)


def _has_wildcards(path: Union[str, Path]) -> bool:
    """Determine if the given path string contains any wildcard characters."""
    return any(w in str(path) for w in WILDCARDS)


def _fallback_open(
    open_func: Any, *args: Any, _fallback_attempt: bool = False, **kwargs: Any
) -> xr.Dataset:
    """Open a dataset with the provided xarray function.

    Falls back between netcdf4, h5netcdf, and scipy engines if necessary.
    """
    try:
        return open_func(*args, **kwargs)
    except Exception as exc:
        if _fallback_attempt:
            logger.error("All fallback engines failed to open dataset.")
            raise exc
        engine = kwargs.get("engine")
        if engine == "h5netcdf":
            candidates = ["netcdf4", "scipy"]
        elif engine == "netcdf4":
            candidates = ["h5netcdf", "scipy"]
        elif engine == "scipy":
            candidates = ["h5netcdf", "netcdf4"]
        else:
            candidates = ["h5netcdf", "netcdf4", "scipy"]

        for cand in candidates:
            try:
                kwargs["engine"] = cand
                logger.debug(f"Retry opening dataset with fallback engine '{cand}'.")
                return _fallback_open(
                    open_func, *args, _fallback_attempt=True, **kwargs
                )
            except Exception:
                continue

        logger.error(f"Error opening dataset: {exc}")
        raise exc


def read_dataset(
    file_path: Union[str, Path, List[Union[str, Path]]],
    use_mfdataset: bool = False,
    engine: str = "netcdf4",
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
    engine : str, default "netcdf4"
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
        try:
            return xr.combine_by_coords(
                arrays,
                combine_attrs="override",
                compat="override",
                coords="minimal",
            )
        except Exception:
            return xr.combine_by_coords(arrays)

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
    # Ensure missing bounds variables are generated before encoding.
    _ensure_bounds_exist(ds)
    encoding = var_encoding or NC_ENCODE_DEFAULTS
    dims = set(ds.dims)
    coords = set(ds.coords)
    dim_coords = coords & dims
    aux_coords = coords - dims
    # collect bounds variables referenced by coords, but only keep ones actually present
    bnds_all = {ds[c].attrs.get("bounds") for c in coords if "bounds" in ds[c].attrs}
    bnds = {b for b in bnds_all if b in ds}
    missing_bnds = bnds_all - bnds
    if missing_bnds:
        logger.debug(
            f"Ignoring missing bounds variables referenced in attrs: {sorted(missing_bnds)}"
        )
    data_vars = set(ds.data_vars) - bnds
    for name in aux_coords | data_vars:
        if name in bnds:
            continue
        ds[name].encoding = encoding
    for name in dim_coords | bnds:
        enc = dict(ds[name].encoding) if ds[name].encoding else {}
        enc["_FillValue"] = None
        ds[name].encoding = enc


def _ensure_bounds_exist(ds: xr.Dataset, bounds_dim: str = "bnds") -> None:
    """
    Create bounds variables for 1D coordinates if missing.

    - If a coordinate already declares a 'bounds' attribute but the
      referenced variable is absent, the bounds variable is created.
    - If a coordinate lacks 'bounds', a new bounds variable named
      '{coord}_bnds' is generated (unless it already exists).

    Bounds generation uses `generate_bounds` and is skipped for
    coordinates with fewer than two points or with more than one
    dimension.
    """
    created: List[str] = []
    skipped: List[str] = []
    for coord in ds.coords:
        da = ds[coord]
        # only handle 1D coordinates with at least two points
        if da.ndim != 1 or da.sizes[da.dims[0]] < 2:
            skipped.append(coord)
            continue

        bounds_name = da.attrs.get("bounds", f"{coord}_bnds")
        has_bounds_var = bounds_name in ds

        # If bounds already exist, just ensure attr is set
        if has_bounds_var:
            if "bounds" not in da.attrs:
                da.attrs["bounds"] = bounds_name
            continue

        # Bounds missing: try to generate and attach
        try:
            if da.ndim != 1 or da.sizes[da.dims[0]] < 2:
                logger.debug(f"da: {da}")
                msg = "Cannot generate bounds for non-1D or too short data."
                with ErrorLogger(logger):
                    raise ValueError(msg)
            ds.coords[bounds_name] = generate_bounds(da, bounds_dim=bounds_dim)
            ds[coord].attrs["bounds"] = bounds_name
            created.append(bounds_name)
        except Exception:
            logger.debug(
                f"Could not generate bounds for coordinate '{coord}'", exc_info=True
            )
            skipped.append(coord)

    if created:
        logger.debug(f"Generated bounds for coordinates: {sorted(created)}")
    if skipped:
        logger.debug(
            f"Skipped bounds generation for coordinates (non-1D or too short): {sorted(skipped)}"
        )


def sanitize_nc_encoding(ds: "xr.Dataset", encoding: dict) -> dict:  # noqa: PLR0912
    """Return a safe encoding dict and clean ds attrs so netCDF4 won't error."""
    enc_out = {}
    for name in ds.data_vars:
        if name not in encoding:
            continue
        da = ds[name]
        dtype = np.dtype(da.dtype)
        e = dict(encoding[name])  # shallow copy

        # Always keep compression settings
        for k in list(e.keys()):
            if k not in {"zlib", "complevel", "shuffle", "_FillValue"}:
                # 'missing_value' and any other stray keys should not live in 'encoding'
                e.pop(k, None)

        # Clean any leftover attrs that might have been set earlier
        # (especially important for boolean vars) and avoid conflicts
        # between variable attributes and encoding entries (xarray will
        # raise if both provide _FillValue). If a fill/missing value is
        # present in attrs, prefer moving it into encoding (if not already
        # present) and then remove from attrs to avoid clashes.
        mv = da.attrs.pop("missing_value", None)
        fv = da.attrs.pop("_FillValue", None)
        # If encoding does not already specify _FillValue, prefer the
        # attribute value (if available) and cast it to the variable dtype.
        if (
            fv is not None
            and "_FillValue" not in e
            and not np.issubdtype(dtype, np.bool_)
        ):
            try:
                if (
                    np.issubdtype(dtype, np.unsignedinteger)
                    and np.asarray(fv).astype(float) < 0
                ):
                    e["_FillValue"] = np.iinfo(dtype).max
                else:
                    e["_FillValue"] = np.array(fv).astype(dtype).item()
            except Exception:
                if np.issubdtype(dtype, np.unsignedinteger):
                    e["_FillValue"] = np.iinfo(dtype).max
                else:
                    # if casting fails, drop the fill value
                    e.pop("_FillValue", None)
        # If mv (missing_value) exists and encoding doesn't include it,
        # move it to attrs (ensuring dtype match) — this keeps netCDF CF legacy
        # but avoids placing it into encoding where it can conflict.
        if mv is not None and "missing_value" not in e:
            try:
                da.attrs["missing_value"] = np.array(mv).astype(dtype).item()
            except Exception:
                # drop if cannot cast
                da.attrs.pop("missing_value", None)

        if np.issubdtype(dtype, np.bool_):
            # Booleans: do NOT set fill attributes at all
            e.pop("_FillValue", None)
        # Cast _FillValue to the variable dtype if present
        elif "_FillValue" in e:
            try:
                if (
                    np.issubdtype(dtype, np.unsignedinteger)
                    and np.asarray(e["_FillValue"]).astype(float) < 0
                ):
                    e["_FillValue"] = np.iinfo(dtype).max
                else:
                    e["_FillValue"] = np.array(e["_FillValue"]).astype(dtype).item()
            except Exception:
                if np.issubdtype(dtype, np.unsignedinteger):
                    e["_FillValue"] = np.iinfo(dtype).max
                else:
                    # If casting fails, drop it rather than erroring out
                    e.pop("_FillValue", None)
            else:
                if isinstance(e.get("_FillValue"), float) and np.isnan(
                    e.get("_FillValue")
                ):
                    e.pop("_FillValue", None)
                if np.issubdtype(dtype, np.integer):
                    info = np.iinfo(dtype)
                    if e["_FillValue"] < info.min or e["_FillValue"] > info.max:
                        if np.issubdtype(dtype, np.unsignedinteger):
                            e["_FillValue"] = info.max
                        elif info.min < NO_DATA and info.max > NO_DATA:
                            e["_FillValue"] = np.array(NO_DATA).astype(dtype).item()
                        else:
                            e.pop("_FillValue", None)

            # If you *must* also expose 'missing_value' (CF legacy), put it in attrs
            # with matching dtype (commented out by default):
            # mv = da.attrs.get("missing_value", None)
            # if mv is not None:
            #     da.attrs["missing_value"] = np.array(mv).astype(da.dtype).item()

        enc_out[name] = e

    return enc_out


# def generate_safe_nc_encoding(da, target_mb=320, fill=NO_DATA):
#     item = np.dtype(da.dtype).itemsize
#     ny = da.sizes.get("lat", 1)
#     nx = da.sizes.get("lon", 1)
#     per_t_bytes = ny * nx * item
#     t = max(1, int((target_mb * 1024**2) // per_t_bytes))
#     return {
#         da.name
#         or "var": {
#             "chunksizes": (min(t, da.sizes.get("time", t)), ny, nx),
#             "zlib": True,
#             "complevel": 4,
#             "dtype": "f4",
#             "_FillValue": fill,
#             "missing_value": fill,
#             "contiguous": False,
#         }
#     }


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


def generate_bounds_for_all_coords(
    ds: xr.Dataset, bounds_dim: str = "bnds"
) -> xr.Dataset:
    """Generate bounds for all 1D coordinate DataArrays in the dataset."""
    ds_out = ds.copy(deep=False)
    for coord in ds.coords:
        da = ds[coord]
        if da.ndim == 1 and da.sizes[da.dims[0]] >= 2:
            try:
                ds_out.coords[f"{coord}_bnds"] = generate_bounds(da, bounds_dim)
                if "bounds" not in da.attrs:
                    da.attrs["bounds"] = f"{coord}_bnds"
            except Exception:
                logger.debug(
                    f"Could not generate bounds for coordinate '{coord}'", exc_info=True
                )
    return ds_out
