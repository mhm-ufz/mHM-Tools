"""Common NetCDF/xarray routines and utilities for reading, encoding, and bounds generation."""

import logging
from pathlib import Path
from typing import Any, List, Optional, Sequence, Set, Union

import numpy as np
import xarray as xr

from mhm_tools.common.logger import ErrorLogger

from .constants import (
    LAT_KEYS,
    LON_KEYS,
    NC_ENCODE_DEFAULTS,
    NO_DATA,
    TIME_KEYS,
    WILDCARDS,
)

logger = logging.getLogger(__name__)

CF_DEFAULT_CONVENTIONS = "CF-1.12"
RESERVED_FOR_ENCODING = {
    "_FillValue",
    "scale_factor",
    "add_offset",
    "dtype",
    "zlib",
    "complevel",
    "shuffle",
    "fletcher32",
    "contiguous",
    "chunksizes",
    "endian",
    "least_significant_digit",
}
COORD_ALLOWED_ENCODING = {"dtype", "_FillValue", "units", "calendar"}
STALE_COORD_ENCODING_KEYS = {
    "zlib",
    "szip",
    "zstd",
    "bzip2",
    "blosc",
    "shuffle",
    "complevel",
    "fletcher32",
    "contiguous",
    "chunksizes",
    "preferred_chunks",
    "source",
    "original_shape",
    "_ChunkSizes",
    "chunks",
    "compression",
    "chunking",
}

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
    encoding = NC_ENCODE_DEFAULTS if var_encoding is None else var_encoding
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

        enc_out[name] = e

    return enc_out


def get_netcdf_metadata_data_vars(dataset: xr.Dataset) -> Set[str]:
    """Return data variables that describe metadata rather than payload data.

    Parameters
    ----------
    dataset : xr.Dataset
        Dataset whose coordinate bounds and grid mappings should be detected.

    Returns
    -------
    set[str]
        Data variable names referenced as coordinate bounds or grid mappings.
    """
    metadata_vars = set()
    for coord in dataset.coords.values():
        bounds = coord.attrs.get("bounds")
        if bounds in dataset:
            metadata_vars.add(bounds)
    for var in dataset.data_vars.values():
        grid_mapping = var.attrs.get("grid_mapping")
        if grid_mapping in dataset:
            metadata_vars.add(grid_mapping)
    return metadata_vars


def apply_cf_baseline_metadata(ds: xr.Dataset, data_vars: Sequence[str]) -> None:
    """Add best-effort CF baseline metadata before NetCDF output.

    Parameters
    ----------
    ds : xr.Dataset
        Dataset to update in place.
    data_vars : Sequence[str]
        Payload data variables to check for CF-recommended metadata.

    Returns
    -------
    None
    """
    if "Conventions" not in ds.attrs:
        ds.attrs["Conventions"] = CF_DEFAULT_CONVENTIONS
    elif not str(ds.attrs["Conventions"]).startswith("CF-"):
        logger.warning(
            f"Global attribute 'Conventions' is not CF-like "
            f"({ds.attrs['Conventions']!r}). Leaving it unchanged."
        )

    lat_key = _get_axis_coord_key(ds, axis="Y", candidate_names=LAT_KEYS)
    lon_key = _get_axis_coord_key(ds, axis="X", candidate_names=LON_KEYS)
    time_key = _get_time_coord_key(ds)

    if lat_key is None:
        logger.warning(
            "Could not infer latitude coordinate; skipping CF latitude attrs."
        )
    else:
        if _lacks_explicit_cf_axis_metadata(ds[lat_key]):
            logger.warning(
                "Could not infer latitude coordinate from explicit metadata; "
                f"using inferred coordinate {lat_key!r}."
            )
        lat = ds[lat_key]
        lat.attrs.setdefault("standard_name", "latitude")
        lat.attrs.setdefault("units", "degrees_north")
        lat.attrs.setdefault("axis", "Y")

    if lon_key is None:
        logger.warning(
            "Could not infer longitude coordinate; skipping CF longitude attrs."
        )
    else:
        if _lacks_explicit_cf_axis_metadata(ds[lon_key]):
            logger.warning(
                "Could not infer longitude coordinate from explicit metadata; "
                f"using inferred coordinate {lon_key!r}."
            )
        lon = ds[lon_key]
        lon.attrs.setdefault("standard_name", "longitude")
        lon.attrs.setdefault("units", "degrees_east")
        lon.attrs.setdefault("axis", "X")

    if time_key is None:
        logger.warning("Could not infer time coordinate; skipping CF time attrs.")
    else:
        time = ds[time_key]
        time.attrs.setdefault("standard_name", "time")
        time.attrs.setdefault("axis", "T")

    for name in data_vars:
        if name not in ds.data_vars:
            logger.warning(
                f"Requested data variable {name!r} not in dataset; skipping CF "
                "checks for it."
            )
            continue
        attrs = ds[name].attrs
        if "units" not in attrs:
            logger.warning(
                f"Data variable {name!r} has no 'units' attribute (CF-recommended)."
            )
        if "standard_name" not in attrs and "long_name" not in attrs:
            logger.warning(
                f"Data variable {name!r} has neither 'standard_name' nor "
                "'long_name' (CF-recommended)."
            )


def prepare_time_bounds_encoding(
    ds: xr.Dataset, strip_time_attrs: bool = False
) -> xr.Dataset:
    """Synchronize time and time-bounds encoding for NetCDF output.

    Parameters
    ----------
    ds : xr.Dataset
        Dataset to update.
    strip_time_attrs : bool, default False
        If True, remove units/calendar attrs from the time coordinate as well
        as its bounds variable.

    Returns
    -------
    xr.Dataset
        The original dataset or a shallow copy if time bounds were regenerated.
    """
    time_key = _get_time_coord_key(ds)
    if time_key is None:
        return ds

    time_da = ds[time_key]
    bounds_name = time_da.attrs.get("bounds")
    if "units" not in time_da.encoding:
        if "units" in time_da.attrs:
            time_da.encoding["units"] = time_da.attrs["units"]
        else:
            time_da.encoding["units"] = "days since 1970-01-01 00:00:00"
    if "calendar" not in time_da.encoding and "calendar" in time_da.attrs:
        time_da.encoding["calendar"] = time_da.attrs["calendar"]
    if strip_time_attrs:
        for key in ("units", "calendar"):
            time_da.attrs.pop(key, None)

    if bounds_name not in ds:
        return ds

    bnds_da = ds[bounds_name]
    if np.issubdtype(time_da.dtype, np.datetime64) and not np.issubdtype(
        bnds_da.dtype, np.datetime64
    ):
        try:
            ds = ds.copy(deep=False)
            ds.coords[bounds_name] = generate_bounds(time_da)
            time_da = ds[time_key]
            bnds_da = ds[bounds_name]
        except Exception:
            logger.debug("Could not regenerate time bounds", exc_info=True)
    if np.issubdtype(time_da.dtype, np.number) and np.issubdtype(
        bnds_da.dtype, np.number
    ):
        try:
            time_max = float(np.nanmax(np.abs(time_da.values)))
            bounds_max = float(np.nanmax(np.abs(bnds_da.values)))
            if time_max > 0 and bounds_max / time_max > 1e6:
                logger.warning(
                    f"Time bounds look out of scale (max={bounds_max} vs time "
                    f"max={time_max}); regenerating bounds from time."
                )
                ds = ds.copy(deep=False)
                ds.coords[bounds_name] = generate_bounds(time_da)
                time_da = ds[time_key]
                bnds_da = ds[bounds_name]
        except Exception:
            logger.debug(
                "Could not validate/regenerate numeric time bounds",
                exc_info=True,
            )

    if "units" not in bnds_da.encoding:
        bnds_da.encoding["units"] = time_da.encoding.get("units")
    if "calendar" in time_da.encoding and "calendar" not in bnds_da.encoding:
        bnds_da.encoding["calendar"] = time_da.encoding["calendar"]
    for key in ("units", "calendar"):
        bnds_da.attrs.pop(key, None)
    return ds


def prepare_dataset_for_netcdf_write(
    ds: xr.Dataset,
    data_vars: Sequence[str],
    encoding: Optional[dict] = None,
) -> tuple:
    """Return a cleaned dataset and encoding for NetCDF output.

    Parameters
    ----------
    ds : xr.Dataset
        Dataset to clean.
    data_vars : Sequence[str]
        Payload data variables that should receive data encoding.
    encoding : dict, optional
        Initial per-variable encoding.

    Returns
    -------
    tuple
        ``(cleaned_dataset, safe_encoding)`` ready for ``to_netcdf``.
    """
    _remove_reserved_attrs_and_normalize_encoding(ds)
    try:
        ds_clean, moved_encoding = move_reserved_attrs_to_encoding(
            ds, include_coords=True, encoding_in=encoding or {}
        )
        logger.debug(
            f"Moved reserved attrs into encoding for variables: "
            f"{list(moved_encoding.keys())}"
        )
    except Exception:
        logger.debug(
            "move_reserved_attrs_to_encoding failed; falling back",
            exc_info=True,
        )
        ds_clean = ds
        moved_encoding = encoding or {}

    merged_encoding = _merge_variable_encodings(encoding or {}, moved_encoding)
    merged_encoding = {k: v for k, v in merged_encoding.items() if k in data_vars}
    _normalize_data_vars_for_netcdf(ds_clean, data_vars, merged_encoding)

    safe_encoding = sanitize_nc_encoding(ds_clean, merged_encoding)
    logger.info(f"Using encoding: {safe_encoding}")
    set_netcdf_encoding(ds_clean, safe_encoding)
    ds_clean = prepare_time_bounds_encoding(ds_clean, strip_time_attrs=True)
    sanitize_coordinate_encoding(ds_clean)
    return ds_clean, safe_encoding


def sanitize_coordinate_encoding(ds: xr.Dataset) -> None:
    """Remove stale backend encoding from coordinates and metadata variables.

    Parameters
    ----------
    ds : xr.Dataset
        Dataset to update in place.

    Returns
    -------
    None
    """
    for coord in ds.coords:
        enc = _safe_coordinate_encoding(ds[coord].encoding)
        if coord in ds.dims:
            enc["_FillValue"] = None
        ds[coord].encoding = enc
    for name in get_netcdf_metadata_data_vars(ds):
        if name not in ds or name in ds.coords:
            continue
        enc = _safe_coordinate_encoding(ds[name].encoding)
        enc["_FillValue"] = None
        ds[name].encoding = enc
        ds[name].attrs.pop("_FillValue", None)
        ds[name].attrs.pop("missing_value", None)


def move_reserved_attrs_to_encoding(
    obj: Union[xr.Dataset, xr.DataArray],
    include_coords: bool = True,
    extra_reserved: Optional[Set[str]] = None,
    encoding_in: Optional[dict] = None,
):
    """Move reserved serialization attrs into xarray encoding.

    Parameters
    ----------
    obj : xr.Dataset or xr.DataArray
        Object to clean.
    include_coords : bool, default True
        If True, also process coordinate variables.
    extra_reserved : set[str], optional
        Additional keys to treat as reserved.
    encoding_in : dict, optional
        Initial per-variable encoding to merge into the result.

    Returns
    -------
    tuple
        ``(cleaned_object, encoding)`` with reserved attrs moved.
    """
    reserved = set(RESERVED_FOR_ENCODING)
    if extra_reserved:
        reserved |= set(extra_reserved)
    encoding_in = encoding_in or {}

    def _process_var(var):
        for key in list(var.attrs.keys()):
            if key in reserved:
                var.encoding[key] = var.attrs.pop(key)

    if isinstance(obj, xr.Dataset):
        ds = obj.copy(deep=False)
        names = list(ds.variables) if include_coords else list(ds.data_vars)
        for name in names:
            _process_var(ds.variables[name])

        encoding = {
            name: {
                key: value
                for key, value in ds.variables[name].encoding.items()
                if key in reserved
            }
            for name in names
            if ds.variables[name].encoding
        }
        return ds, _merge_variable_encodings(encoding_in, encoding)

    if not isinstance(obj, xr.DataArray):
        msg = f"wrong type {obj}"
        raise ValueError(msg)

    da = obj.copy(deep=False)
    _process_var(da)

    coord_encoding = {}
    if include_coords:
        for coord_name, coord_var in da.coords.variables.items():
            _process_var(coord_var)
            if coord_var.encoding:
                coord_encoding[coord_name] = {
                    key: value
                    for key, value in coord_var.encoding.items()
                    if key in reserved
                }

    data_encoding = {}
    if da.name is not None and da.encoding:
        data_encoding[da.name] = {
            key: value for key, value in da.encoding.items() if key in reserved
        }

    return da, _merge_variable_encodings(
        encoding_in, {**coord_encoding, **data_encoding}
    )


def _get_axis_coord_key(
    ds: xr.Dataset, axis: str, candidate_names: Sequence[str]
) -> Optional[str]:
    """Return a coordinate key matching an axis or known coordinate name."""
    for name in list(ds.coords) + list(ds.dims):
        if name in ds and ds[name].attrs.get("axis") == axis:
            return name
    for name in candidate_names:
        if name in ds and len(ds[name].shape) == 1:
            return name
    for name in candidate_names:
        if name in ds:
            logger.warning(
                f"{type(ds)} contains key: {name} but ds[name] has shape {ds[name].shape}."
            )
            return name
    return None


def _get_time_coord_key(ds: xr.Dataset) -> Optional[str]:
    """Return the time coordinate key without importing xarray_utils."""
    return _get_axis_coord_key(ds, axis="T", candidate_names=TIME_KEYS)


def _lacks_explicit_cf_axis_metadata(coord: xr.DataArray) -> bool:
    """Return True if a coordinate lacks explicit CF axis metadata."""
    attrs = coord.attrs
    return not any(key in attrs for key in ("axis", "standard_name", "units"))


def _remove_reserved_attrs_and_normalize_encoding(ds: xr.Dataset) -> None:
    """Remove reserved attrs and cast reserved encoding values in place."""
    for name in list(ds.variables):
        try:
            attrs = ds.variables[name].attrs
            attrs.pop("_FillValue", None)
            attrs.pop("missing_value", None)
        except Exception:
            logger.debug(
                f"Could not pop reserved attrs for variable {name}",
                exc_info=True,
            )

        try:
            var = ds.variables[name]
            dtype = np.dtype(var.dtype)
            for key in ("_FillValue", "missing_value"):
                if key not in var.encoding:
                    continue
                try:
                    var.encoding[key] = np.array(var.encoding[key]).astype(dtype).item()
                except Exception:
                    var.encoding.pop(key, None)
        except Exception:
            logger.debug(
                f"Could not normalize encoding for variable {name}",
                exc_info=True,
            )


def _merge_variable_encodings(base_encoding: dict, override_encoding: dict) -> dict:
    """Merge two per-variable encoding dictionaries."""
    merged = {}
    for name in set(list(base_encoding.keys()) + list(override_encoding.keys())):
        merged[name] = {}
        if name in base_encoding:
            merged[name].update(base_encoding[name])
        if name in override_encoding:
            merged[name].update(override_encoding[name])
    return merged


def _normalize_data_vars_for_netcdf(
    ds: xr.Dataset, data_vars: Sequence[str], encoding: dict
) -> None:
    """Cast data variables that need NetCDF-compatible representation."""
    for name in data_vars:
        data_array = ds[name]
        dtype = np.dtype(data_array.dtype)
        if np.issubdtype(dtype, np.bool_):
            ds[name] = data_array.astype("uint8")
            encoding.pop(name, None)
            continue
        if np.issubdtype(dtype, np.integer):
            has_nan = bool(np.any(np.isnan(data_array.values)))
            if has_nan:
                fill_value = data_array.attrs.get("_FillValue", NO_DATA)
                ds[name] = data_array.fillna(fill_value).astype(dtype)


def _safe_coordinate_encoding(encoding: dict) -> dict:
    """Return coordinate encoding without stale backend-specific keys."""
    enc = dict(encoding) if encoding else {}
    for key in STALE_COORD_ENCODING_KEYS:
        enc.pop(key, None)
    return {key: value for key, value in enc.items() if key in COORD_ALLOWED_ENCODING}


def generate_bounds(
    da: xr.DataArray, bounds_dim: str = "bnds", res=None
) -> xr.DataArray:
    """Generate CF-compliant bounds for a coordinate DataArray."""
    (dim,) = da.dims
    if res is None:
        res = da.diff(dim)
    lower = da - res / 2
    upper = da + res / 2
    bounds = xr.concat([lower, upper], dim=bounds_dim)
    first_lower = bounds.isel({dim: 0}) - res.isel({dim: 0})
    first = first_lower.assign_coords({dim: da[dim][0]})
    all_bounds = xr.concat([first, bounds], dim=dim)
    return all_bounds.transpose()


def generate_bounds_for_all_coords(
    ds: xr.Dataset, bounds_dim: str = "bnds", res: Optional[float] = None
) -> xr.Dataset:
    """Generate bounds for all 1D coordinate DataArrays in the dataset."""
    ds_out = ds.copy(deep=False)
    for coord in ds.coords:
        da = ds[coord]
        if da.ndim == 1 and da.sizes[da.dims[0]] >= 2:
            try:
                ds_out.coords[f"{coord}_bnds"] = generate_bounds(da, bounds_dim, res)
                if "bounds" not in da.attrs:
                    da.attrs["bounds"] = f"{coord}_bnds"
            except Exception:
                logger.debug(
                    f"Could not generate bounds for coordinate '{coord}'", exc_info=True
                )
    return ds_out
