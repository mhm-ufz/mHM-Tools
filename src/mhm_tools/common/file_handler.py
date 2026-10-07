"""Read, write, and convert gridded setup files.

The module provides shared helpers for NetCDF, ESRI ASCII, and GeoTIFF-style
workflows, including coordinate normalization, header creation, encoding
cleanup, grid metadata handling, and output provenance.

Authors
-------
- Simon Lüdke
"""

import contextlib
import logging
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple, Union

import numpy as np
import xarray as xr
from rasterio.crs import CRS
from rasterio.enums import Resampling
from rasterio.transform import from_origin
from rasterio.warp import calculate_default_transform

from mhm_tools.common.constants import NC_ENCODE_DEFAULTS, NC_ENCODE_MASK, NO_DATA
from mhm_tools.common.crs_handler import (
    MissingCRSError,
    resolve_crs,
    set_spatial_dims,
    write_object_crs,
)
from mhm_tools.common.esri_grid import standardize_header, write_header
from mhm_tools.common.logger import ErrorLogger, log_arguments
from mhm_tools.common.netcdf import (
    QUANTIZATION_ATTRS,
    QUANTIZATION_ENCODING_KEYS,
    NetcdfCompression,
    add_variable_hard_link,
    apply_cf_baseline_metadata,
    generate_bounds,
    generate_bounds_for_all_coords,
    get_netcdf_metadata_data_vars,
    normalize_variable_selection,
    prepare_dataset_for_netcdf_write,
    prepare_time_bounds_encoding,
    read_dataset,
)
from mhm_tools.common.provenance import apply_output_provenance
from mhm_tools.common.xarray_utils import (
    get_coord_key,
    get_dtype,
    get_single_data_var,
    normalize_lat_lon,
)

logger = logging.getLogger(__name__)
# Re-exported so raster callers need only this module.
__all__ = ["MissingCRSError", "get_nodata"]


@dataclass
class GridDefinition:
    """Container to preserve a dataset's spatial grid metadata."""

    template: xr.Dataset
    dims: Tuple[str, ...]


def get_raster_data(file_path, var_name=None, crs=None) -> xr.DataArray:
    """Read one two-dimensional, georeferenced raster payload."""
    ds = get_xarray_ds_from_file(
        file_path,
        var_name=var_name,
        load_crs=Path(file_path).suffix.lower() == ".nc",
    )
    try:
        name = var_name or get_single_data_var(ds)
        if name is None or name not in ds.data_vars:
            msg = "Raster input must contain exactly one payload data variable."
            raise ValueError(msg)
        data = ds[name]
        if data.ndim != 2:
            msg = f"Raster payload {name!r} must be two-dimensional, got {data.ndim}."
            raise ValueError(msg)
        data = set_spatial_dims(data)
        dataset_crs = resolve_crs(ds, crs)
        resolved = resolve_crs(data, dataset_crs, required=True)
        result = data.rio.write_crs(resolved, inplace=False)
        result.set_close(ds.close)
        return result
    except Exception:
        ds.close()
        raise


def align_raster_to_reference(
    data: xr.DataArray,
    reference: xr.DataArray,
    *,
    nodata=NO_DATA,
    resampling="nearest",
    data_kind=None,
    mask_reference=False,
) -> xr.DataArray:
    """Match a raster to a reference grid with explicit or automatic resampling."""
    data = _as_yx(data)
    reference = _as_yx(reference)
    resolve_crs(data, required=True)
    reference_crs = resolve_crs(reference, required=True)
    source_nodata = get_nodata(data)
    data = _ensure_nodata_dtype(data, nodata)
    resampling = _resolve_raster_resampling(
        data,
        reference,
        resampling=resampling,
        data_kind=data_kind,
    )

    if _same_raster_grid(data, reference):
        values = _restated_nodata(data, source_nodata, nodata)
    else:
        values = data.rio.reproject_match(
            reference,
            resampling=resampling,
            nodata=nodata,
        ).data
    if mask_reference:
        values = _blank_outside_reference(values, reference, nodata)

    coords = {
        name: coord
        for name, coord in reference.coords.items()
        if set(coord.dims).issubset(set(reference.dims))
    }
    attrs = dict(data.attrs)
    for key in ("_FillValue", "missing_value"):
        attrs.pop(key, None)
    attrs["nodata_value"] = nodata
    result = xr.DataArray(
        values,
        dims=reference.dims,
        coords=coords,
        name=data.name,
        attrs=attrs,
    )
    result = set_spatial_dims(result).rio.write_crs(reference_crs, inplace=False)
    result = result.rio.write_transform(reference.rio.transform(), inplace=False)
    result.attrs["nodata_value"] = nodata
    return result.rio.write_nodata(nodata, inplace=False)


def _as_yx(data: xr.DataArray) -> xr.DataArray:
    """Return a raster tagged as spatial and transposed to ``(y, x)`` order."""
    data = set_spatial_dims(data)
    y_dim, x_dim = data.rio.y_dim, data.rio.x_dim
    return data.transpose(y_dim, x_dim).rio.set_spatial_dims(x_dim=x_dim, y_dim=y_dim)


def _restated_nodata(data, source_nodata, nodata):
    """Return the values of an already aligned raster on the requested nodata."""
    normalized = data
    if source_nodata is not None and not _is_nan(source_nodata):
        normalized = normalized.where(normalized != source_nodata, nodata)
    if np.issubdtype(data.dtype, np.integer):
        return normalized.astype(data.dtype).data
    return normalized.where(normalized.notnull(), nodata).data


def _blank_outside_reference(values, reference, nodata):
    """Set the cells the reference itself marks as nodata back to nodata."""
    reference_values = np.asarray(reference.values)
    valid = np.ones(reference.shape, dtype=bool)
    if np.issubdtype(reference_values.dtype, np.floating):
        valid &= np.isfinite(reference_values)
    for reference_nodata in _raster_nodata_values(reference):
        valid &= reference_values != reference_nodata
    return np.where(valid, values, nodata)


def _resolve_raster_resampling(data, reference, *, resampling, data_kind):
    """Resolve a Rasterio resampling enum, including data-aware ``auto`` mode."""
    if isinstance(resampling, Resampling):
        return resampling

    method = str(resampling).strip().lower()
    if method in {"categorical", "continuous"}:
        data_kind = method
        method = "auto"
    if method != "auto":
        try:
            return Resampling[method]
        except KeyError as exc:
            choices = ", ".join(member.name for member in Resampling)
            msg = (
                f"Unsupported resampling method {resampling!r}. "
                f"Choose from: {choices}."
            )
            raise ValueError(msg) from exc

    if data_kind is None:
        data_kind = (
            "categorical" if np.issubdtype(data.dtype, np.integer) else "continuous"
        )
    data_kind = str(data_kind).strip().lower()
    if data_kind not in {"categorical", "continuous"}:
        msg = "data_kind must be 'categorical' or 'continuous'."
        raise ValueError(msg)

    downsampling = _is_raster_downsampling(data, reference)
    if data_kind == "categorical":
        return Resampling.mode if downsampling else Resampling.nearest
    return Resampling.average if downsampling else Resampling.bilinear


def _is_raster_downsampling(data, reference) -> bool:
    """Return whether a reference pixel covers more area than a source pixel."""
    source_transform = data.rio.transform()
    source_crs = resolve_crs(data, required=True)
    reference_crs = resolve_crs(reference, required=True)
    if source_crs != reference_crs:
        source_transform, _, _ = calculate_default_transform(
            source_crs,
            reference_crs,
            data.sizes[data.rio.x_dim],
            data.sizes[data.rio.y_dim],
            *data.rio.bounds(),
        )
    reference_transform = reference.rio.transform()
    source_area = abs(
        source_transform.a * source_transform.e
        - source_transform.b * source_transform.d
    )
    reference_area = abs(
        reference_transform.a * reference_transform.e
        - reference_transform.b * reference_transform.d
    )
    return bool(
        reference_area > source_area
        and not np.isclose(reference_area, source_area, rtol=1e-9, atol=0.0)
    )


def _declared_nodata(data):
    """Return every nodata sentinel a raster declares, most authoritative first."""
    values = []
    with contextlib.suppress(Exception):
        values.extend([data.rio.encoded_nodata, data.rio.nodata])
    values.extend(
        data.attrs.get(key) for key in ("nodata_value", "_FillValue", "missing_value")
    )
    values.extend(data.encoding.get(key) for key in ("_FillValue", "missing_value"))
    return values


def get_nodata(data):
    """Return the first nodata value a raster declares, preserving zero."""
    return next((value for value in _declared_nodata(data) if value is not None), None)


def _raster_nodata_values(data):
    """Yield the distinct finite nodata sentinels a raster declares."""
    seen = set()
    for value in _declared_nodata(data):
        if value is None or _is_nan(value):
            continue
        try:
            number = float(value)
        except (TypeError, ValueError):
            continue
        if np.isfinite(number) and number not in seen:
            seen.add(number)
            yield number


def _same_raster_grid(data: xr.DataArray, reference: xr.DataArray) -> bool:
    if data.dims != (data.rio.y_dim, data.rio.x_dim):
        return False
    if reference.dims != (reference.rio.y_dim, reference.rio.x_dim):
        return False
    if data.shape != reference.shape or data.rio.crs != reference.rio.crs:
        return False
    return data.rio.transform().almost_equals(reference.rio.transform())


def _dtype_holding_nodata(dtype, nodata):
    """Return the narrowest dtype holding both the range of ``dtype`` and ``nodata``.

    Falls back to float64 for a non-integral nodata, and raises when no integer
    dtype can represent both.
    """
    dtype = np.dtype(dtype)
    info = np.iinfo(dtype)
    if info.min <= nodata <= info.max:
        return dtype
    if not np.isfinite(nodata) or float(nodata) != int(nodata):
        return np.dtype("float64")
    minimum = min(info.min, int(nodata))
    maximum = max(info.max, int(nodata))
    for candidate in (np.int8, np.int16, np.int32, np.int64):
        candidate_info = np.iinfo(candidate)
        if candidate_info.min <= minimum and maximum <= candidate_info.max:
            return np.dtype(candidate)
    msg = f"Raster dtype {dtype} and nodata {nodata} have no safe integer dtype."
    raise ValueError(msg)


def _ensure_nodata_dtype(data: xr.DataArray, nodata) -> xr.DataArray:
    """Promote integer data when needed to represent the requested nodata."""
    if nodata is None or not np.issubdtype(data.dtype, np.integer):
        return data
    dtype = _dtype_holding_nodata(data.dtype, nodata)
    return data if dtype == data.dtype else data.astype(dtype)


def _is_nan(value):
    """Return whether a scalar value is NaN."""
    with contextlib.suppress(TypeError):
        return bool(np.isnan(value))
    return False


def get_grid(
    ds: xr.Dataset, data_vars: Optional[Union[str, Sequence[str]]] = None
) -> GridDefinition:
    """Extract a grid definition from ``ds``.

    Parameters
    ----------
    ds : xr.Dataset
        Dataset that defines the spatial/temporal grid.
    data_vars : str | sequence[str], optional
        Data variables to drop from the returned template. When omitted the
        first data variable is used.
    """
    if data_vars is None:
        var_name = get_single_data_var(ds)
        if var_name is None:
            msg = "Cannot determine data_var to describe grid."
            with ErrorLogger(logger):
                raise ValueError(msg)
        drop_vars = [var_name]
    elif isinstance(data_vars, str):
        drop_vars = [data_vars]
        var_name = data_vars
    else:
        drop_vars = list(data_vars)
        var_name = drop_vars[0]

    if var_name not in ds.data_vars:
        msg = f"Grid descriptor variable {var_name} not present in dataset."
        with ErrorLogger(logger):
            raise ValueError(msg)

    template = ds.drop_vars(drop_vars, errors="ignore")
    dims = tuple(ds[var_name].dims)
    return GridDefinition(template=template, dims=dims)


def set_grid(
    data: np.ndarray,
    grid: GridDefinition,
    var_name: str,
    data_attrs: Optional[Dict[str, Union[str, float, int]]] = None,
) -> xr.Dataset:
    """Attach ``data`` to a preserved grid definition."""
    coords = {}
    for name, coord in grid.template.coords.items():
        # Only include coords compatible with the target variable dims.
        if set(coord.dims).issubset(set(grid.dims)):
            coords[name] = coord
    da = xr.DataArray(
        data,
        coords=coords,
        dims=grid.dims,
        attrs=data_attrs or {},
        name=var_name,
    )
    ds = grid.template.copy(deep=False)
    ds[var_name] = da
    return ds


def create_header(
    ds,
    output_path=None,
    no_data_value=None,
    cellsize=None,
    xllcorner=None,
    yllcorner=None,
) -> dict:
    """Write a header file from a dataset.

    Takes an xarray Dataset and writes the ASCII header needed for GIS tools.
    """
    if no_data_value is None:
        no_data_value = NO_DATA
    lat_key = get_coord_key(ds, lat=True)
    lon_key = get_coord_key(ds, lon=True)
    x = ds[lon_key].data
    y = ds[lat_key].data
    if cellsize is None:
        if len(x) > 1:
            cellsize = abs(x[1] - x[0])
        elif len(y) > 1:
            cellsize = abs(y[1] - y[0])
        else:
            with contextlib.suppress(Exception):
                transform = ds.rio.transform()
                x_size, y_size = abs(transform.a), abs(transform.e)
                if x_size > 0 and np.isclose(x_size, y_size):
                    cellsize = x_size
            if cellsize is None:
                msg = (
                    "Cannot determine cellsize from dataset with only one x "
                    "and one y value. Please provide cellsize as an argument."
                )
                with ErrorLogger(logger):
                    raise ValueError(msg)
    if xllcorner is None:
        xllcorner = np.nanmin(x) - 0.5 * cellsize
    if yllcorner is None:
        yllcorner = np.nanmin(y) - 0.5 * cellsize

    ncols = len(x)
    nrows = len(y)
    dtype = get_dtype(ds)
    typ = int if issubclass(np.dtype(dtype).type, np.integer) else float
    header_dict = {
        "ncols": ncols,
        "nrows": nrows,
        "xllcorner": xllcorner,
        "yllcorner": yllcorner,
        "cellsize": cellsize,
        "nodata_value": typ(no_data_value),
    }
    header_dict = standardize_header(header_dict)

    if output_path is not None:
        output_path = Path(output_path)
        if output_path.is_dir() or not output_path.suffix:
            header_out_path = output_path / "header.txt"
        else:
            header_out_path = output_path
        header_out_path.parent.mkdir(parents=True, exist_ok=True)
        header_str = write_header(header_out_path, header_dict, dtype)
        logger.info(
            f"Writing header file to {header_out_path} with header str: \n{header_str}"
        )
    return header_dict


def crop_file_by_mask(ds, mask_file):
    """Crop file by mask."""
    if isinstance(mask_file, xr.Dataset):
        mask = mask_file
    else:
        mask = get_xarray_ds_from_file(mask_file)
    lat_key_mask = get_coord_key(mask, lat=True)
    lon_key_mask = get_coord_key(mask, lon=True)
    lat_key = get_coord_key(ds, lat=True)
    lon_key = get_coord_key(ds, lon=True)
    return ds.sel(
        {
            lat_key: slice(mask[lat_key_mask].max(), mask[lat_key_mask].min()),
            lon_key: slice(mask[lon_key_mask].min(), mask[lon_key_mask].max()),
        }
    )


def chunk_dataset_space_only(
    ds: xr.Dataset, available_mem_gib: float, var_name: Optional[str] = None
) -> Dict[str, int]:
    """Chunk only in space (lat/lon), leaving time whole, sized to available memory.

    - Uses 10% of available_mem_gib for a single chunk.
    - Computes how many total cells (t * y * x) fit, then allocates all t,
      and splits y/x so that t·y·x·bytes_per_cell ≤ work_bytes.
    - If no time dimension, behaves similarly with t=1.
    """
    logger.info(
        f"Chunking spatial dims to fit ≈{available_mem_gib} GiB (time unchunked)"
    )
    # --- pick one variable to get dtype size ---
    if var_name is None:
        var_name = get_single_data_var(ds)

    dtype_sz = ds[var_name].dtype.itemsize  # bytes per element

    # --- find coordinate names ---
    lat_key = get_coord_key(ds, lat=True)
    lon_key = get_coord_key(ds, lon=True)
    time_key = None
    # with contextlib.suppress(ValueError):
    time_key = get_coord_key(ds, time=True, raise_exception=False)

    ny = ds.sizes[lat_key]
    nx = ds.sizes[lon_key]
    nt = ds.sizes.get(time_key, 1)

    # --- memory budget in bytes (10%) ---
    work_bytes = max(int(0.1 * available_mem_gib * 1024**3), 4 * 1024**2)
    # how many total cells fit
    max_cells = work_bytes // dtype_sz
    # allocate all time steps
    cells_per_slice = max_cells // nt

    # square-ish spatial block
    side = max(1, int(np.sqrt(cells_per_slice)))
    y_chunk = min(ny, side)
    x_chunk = min(nx, side)

    chunks = {lat_key: int(y_chunk), lon_key: int(x_chunk)}
    if time_key:
        # -1 means “take all” for that dim
        chunks[time_key] = -1

    logger.debug(
        f"Chunk sizes → time: {chunks.get(time_key, '—')}, "
        f"{lat_key}: {y_chunk}, {lon_key}: {x_chunk}"
    )
    return chunks


def chunk_dataset_space_and_time(
    ds, available_mem_gib, var_name: Optional[str] = None
) -> Dict[str, int]:
    """Chunk dataset adjusting chunk size to avaiable memory.

    Simple heuristic:
      - try to keep time chunks small (1…4)vi
      - make y/x chunks as square as possible
    """
    logger.info(
        f"Chunking dataset with a max amount of mem of {available_mem_gib:.1f}Gb"
    )
    # ---------------- metadata only (cheap) --------------------------------
    if isinstance(ds, xr.Dataset):
        if var_name is None:
            var_name = next(iter(ds.data_vars))  # first data variable
        var = ds[var_name]  # an xarray.Variable wrapper
    else:
        var = ds
    dtype_sz = var.dtype.itemsize  # bytes per element

    lat_key = get_coord_key(ds, lat=True)
    lon_key = get_coord_key(ds, lon=True)
    time_key = get_coord_key(ds, time=True, raise_exception=False)
    if time_key is None:
        return chunk_dataset_space_only(ds, available_mem_gib, var_name)

    ny = ds.sizes[lat_key]
    nx = ds.sizes[lon_key]
    nt = ds.sizes.get(time_key, 1)

    # ---------------- convert GiB → bytes and keep 80 % ---------------------
    _MIN_BYTES_PER_CHUNK = 4 * 1024**2  # 4MB
    work_bytes = max(int(0.8 * available_mem_gib * 1024**3), _MIN_BYTES_PER_CHUNK)
    max_cells = work_bytes // dtype_sz  # how many array elements fit

    # ---------------- choose chunk sizes -----------------------------------
    t_chunk = min(nt, 4) if time_key else None  # ≤4 along time
    cells_per_t = max_cells // (t_chunk or 1)

    side = max(1, int(np.sqrt(cells_per_t)))  # square-ish y/x chunk
    y_chunk = min(ny, side)
    x_chunk = min(nx, side)

    chunks = {lat_key: int(y_chunk), lon_key: int(x_chunk)}
    if time_key:
        t_chunk = max(1, max_cells // max(1, y_chunk * x_chunk))
        chunks[time_key] = int(t_chunk)
    logger.debug(f"   The chunks used are {chunks}")

    return chunks


class ChunkType(Enum):
    """Define Types of chunking.

    SPACE: Only chunking in space. Time dimension is conserved.
    TIME: Chunking predominately in time. If necessary also in space.
    """

    SPACE = 1
    TIME = 2


@log_arguments()
def chunk_dataset(ds, chunk_type, available_mem_gib, var_name=None):
    """Chunk xarray.DataSet depending on chunk_type and available memory."""
    if chunk_type == ChunkType.TIME:
        chunks = chunk_dataset_space_and_time(ds, available_mem_gib, var_name)
    if chunk_type == ChunkType.SPACE:
        chunks = chunk_dataset_space_only(ds, available_mem_gib, var_name)
    try:
        return ds.chunk(chunks)
    except Exception as e:
        logger.error(chunks)
        logger.error(ds)
        with ErrorLogger(logger):
            raise e


def write_xarray_to_netcdf(
    ds,
    file_path,
    var_name=None,
    encoding=None,
    engine="netcdf4",
    compression=None,
):
    """Write an xarray Dataset or DataArray to a NetCDF file.

    Parameters
    ----------
    ds : xr.Dataset or xr.DataArray
        Dataset or data array to write.
    file_path : str or pathlib.Path
        Target NetCDF file path.
    var_name : str, optional
        Data variable to write or DataArray name override.
    encoding : dict, optional
        Per-variable NetCDF encoding. Its compression keys are replaced by
        `compression`, every other key it holds is kept.
    engine : str, default "netcdf4"
        Xarray NetCDF backend engine. Quantization needs "netcdf4".
    compression : NetcdfCompression, optional
        Lossless and lossy compression settings. None keeps the compression a
        dataset read from a NetCDF file carries, falling back to the defaults
        when it carries none.

    Returns
    -------
    None
    """
    ds, data_vars = _get_netcdf_write_dataset_and_data_vars(ds, var_name)
    ds = apply_output_provenance(ds)
    apply_cf_baseline_metadata(ds, data_vars)
    compression = (
        NetcdfCompression(complevel=None, shuffle=None)
        if compression is None
        else compression
    )
    compression_encoding = compression.create_variable_encoding(ds, data_vars)
    drop_quantization = engine != "netcdf4"
    if drop_quantization and any(
        "significant_digits" in settings for settings in compression_encoding.values()
    ):
        logger.warning(
            f"Ignoring quantization for {file_path} because the {engine} engine "
            "does not support it. Use engine='netcdf4' to quantize."
        )
    for name in set(data_vars) | set(ds.coords) | get_netcdf_metadata_data_vars(ds):
        for attr in QUANTIZATION_ATTRS:
            ds[name].attrs.pop(attr, None)
        for key in QUANTIZATION_ENCODING_KEYS:
            ds[name].encoding.pop(key, None)
            ds[name].attrs.pop(key, None)
    if drop_quantization:
        for settings in compression_encoding.values():
            for key in QUANTIZATION_ENCODING_KEYS:
                settings.pop(key, None)
    if encoding is None:
        encoding = {
            var: {**NC_ENCODE_DEFAULTS, **compression_encoding[var]}
            for var in data_vars
        }
    else:
        # the caller owns dtype and the fill values, the compression settings
        # own the rest, so one option reaches every write that names variables
        encoding = {
            key: {**value, **compression_encoding.get(key, {})}
            for key, value in encoding.items()
            if key in data_vars
        }
        for var, settings in compression_encoding.items():
            if var not in encoding:
                encoding[var] = dict(settings)

    numeric_coord_attrs = {
        name: {
            key: value
            for key, value in {**ds[name].encoding, **ds[name].attrs}.items()
            if key in {"units", "calendar"}
        }
        for name in set(ds.coords) | get_netcdf_metadata_data_vars(ds)
        if np.issubdtype(ds[name].dtype, np.number)
    }
    ds = prepare_time_bounds_encoding(ds)
    try:
        ds_clean, safe_encoding = prepare_dataset_for_netcdf_write(
            ds, data_vars, encoding, compression
        )
        # Restate resolved settings after inherited encoding has been merged.
        for var, settings in compression_encoding.items():
            if var in safe_encoding:
                for key in QUANTIZATION_ENCODING_KEYS:
                    safe_encoding[var].pop(key, None)
                safe_encoding[var].update(settings)
        # Numeric time axes (including LAI months) store units as attributes.
        for name, attrs in numeric_coord_attrs.items():
            if np.issubdtype(ds_clean[name].dtype, np.number):
                ds_clean[name].attrs.update(attrs)
                for key in ("units", "calendar"):
                    ds_clean[name].encoding.pop(key, None)
        ds_clean.to_netcdf(
            file_path, engine=engine, format="NETCDF4", encoding=safe_encoding
        )
    except ValueError:
        logger.error(f"Error while writing to {file_path}")
        logger.error(ds)
        logger.info(
            f"Trying to write without encoding {encoding}. This drops "
            "compression, fill values and any quantization."
        )
        ds = prepare_time_bounds_encoding(ds, strip_time_attrs=True)
        ds.to_netcdf(file_path, engine=engine, format="NETCDF4")
    except Exception as e:
        with ErrorLogger(logger):
            raise e


def _get_netcdf_write_dataset_and_data_vars(ds, var_name=None):
    """Return a Dataset and payload variable names for NetCDF writing.

    Parameters
    ----------
    ds : xr.Dataset or xr.DataArray
        Object to prepare for NetCDF writing.
    var_name : str, optional
        Requested data variable name.

    Returns
    -------
    tuple
        ``(dataset, data_vars)`` where data_vars excludes bounds/grid mappings.
    """
    if isinstance(ds, xr.DataArray):
        var_name = ds.name if var_name is None else var_name
        logger.debug(f"var {var_name}")
        ds = ds.to_dataset(name=var_name)
        data_vars = [var_name]
        logger.debug(f"Creating data vars from dataarray name: {data_vars}")
    elif var_name is None or var_name not in ds.data_vars:
        if var_name is not None and var_name not in ds.data_vars:
            logger.warning(
                f"Requested var_name {var_name!r} not found in dataset. "
                "Using all data variables instead."
            )
        logger.info(f"Taking data vars from list of ds.data_vars {ds.data_vars}")
        metadata_vars = get_netcdf_metadata_data_vars(ds)
        data_vars = [var for var in ds.data_vars if var not in metadata_vars]
    else:
        data_vars = [var_name]
        logger.info(f"Setting data vars as input varname [var_name]: {data_vars}")

    if not data_vars:
        logger.warning(
            "Dataset has no data variables. Writing coordinate-only dataset."
        )
    logger.debug(f"data vars: {data_vars}")
    return ds, data_vars


@log_arguments()
def get_xarray_ds_from_file(  # noqa: PLR0912
    file_path,
    var_name=None,
    chunking=False,
    available_mem_gib=None,
    chunk_type=ChunkType.SPACE,
    use_mfdataset=False,
    engine="netcdf4",
    normalize_latlon_coords=False,
    force_decending_y=False,
    force_ascending_y=False,
    landcover=False,
    landcover_year_start=None,
    create_bounds=False,
    load_crs=False,
    drop_auxiliary_coords=False,
):
    """Read file and return xarray dataset.

    Set ``load_crs`` to also decode the grid-mapping variable that carries the
    CRS; it is left out by default because it is not payload data. Set
    ``drop_auxiliary_coords`` to drop multi-dimensional coordinates, such as the
    2D lat/lon of a projected grid, which otherwise keep its 1D x/y axes from
    being normalized to lat/lon.
    """
    file_path = Path(file_path)
    logger.debug(f"Reading {file_path} to xarray with chunking = {chunking}")
    ds_out = None
    if not file_path.is_file():
        msg = f"File path does not point to an existing file: {file_path}"
        with ErrorLogger(logger):
            raise ValueError(msg)
    suffix = file_path.suffix.lower()
    if suffix == ".asc":
        if landcover:
            logger.info("Reading ascii landcover file.")
            ds_out = read_ascii_to_xarray(
                filepath=file_path,
                var_name=var_name,
                landcover=landcover,
                landcover_year_start=landcover_year_start,
            )
        else:
            ds_out = read_ascii_to_xarray(
                filepath=file_path,
                var_name=var_name,
            )
        chunk_type = ChunkType.SPACE
    elif suffix == ".nc":
        ds_out = read_dataset(
            file_path=file_path,
            use_mfdataset=use_mfdataset,
            engine=engine,
            # "all" also decodes the grid mapping variable holding the CRS
            decode_coords="all" if load_crs else "coordinates",
            variables=normalize_variable_selection(var_name),
        )
    elif suffix in {".tif", ".tiff"}:
        import rioxarray as rxr

        da = rxr.open_rasterio(file_path)
        if var_name is not None:
            da = da.rename(var_name)
        else:
            da.name = "data"
        if "band" in da.dims and da.sizes.get("band", 0) == 1:
            da = da.squeeze("band", drop=True)
        nodata = None
        with contextlib.suppress(Exception):
            nodata = da.rio.nodata
        if nodata is not None:
            da.attrs.setdefault("nodata_value", nodata)
            da.attrs.setdefault("_FillValue", nodata)
        if "x" in da.coords:
            da["x"].attrs.setdefault("axis", "X")
        if "y" in da.coords:
            da["y"].attrs.setdefault("axis", "Y")
        ds_out = da.to_dataset()
    else:
        msg = (
            "Reading file types other than asci, netcdf, and geotiff is not "
            f"implemented. The suffix of the file was: {file_path.suffix}"
        )
        with ErrorLogger(logger):
            raise NotImplementedError(msg)
    if drop_auxiliary_coords:
        # the 2D lat/lon of a projected grid take the names its x/y axes are
        # normalized to
        ds_out = ds_out.drop_vars(
            [name for name, coord in ds_out.coords.items() if coord.ndim > 1]
        )
    lat_key = get_coord_key(ds_out, lat=True, raise_exception=False)
    lon_key = get_coord_key(ds_out, lon=True, raise_exception=False)
    # normalize before the axis order is read, because a coordinate that is not
    # an index yet makes the comparison below positional and therefore always
    # ascending, which would silently keep a flipped grid
    if normalize_latlon_coords:
        # re-name input coords to lat and lon
        ds_out = normalize_lat_lon(
            ds_out, lat_key=lat_key, lon_key=lon_key, raise_exceptions=False
        )
        lat_key = get_coord_key(ds_out, lat=True, raise_exception=False)
        lon_key = get_coord_key(ds_out, lon=True, raise_exception=False)
    # force correct order of y coordinate
    if lat_key is not None and (
        (force_decending_y and ds_out[lat_key].data[0] < ds_out[lat_key].data[-1])
        or (force_ascending_y and ds_out[lat_key].data[0] > ds_out[lat_key].data[-1])
    ):
        ds_out = ds_out.sel({lat_key: slice(None, None, -1)})
    logger.debug(ds_out)
    logger.debug(lat_key)
    logger.debug(lon_key)
    if create_bounds:
        ds_out = generate_bounds_for_all_coords(ds_out)
    if lon_key is None and lat_key is None:
        logger.warning("Dataset does not have lon and lat key.")
    elif lon_key is None or lat_key is None:
        logger.error("Dataset has only one of lon at lat keys.")

    if chunking and available_mem_gib is not None:
        # several variables share one dtype and shape, so sizing the chunks on
        # the first of them sizes them for each
        selected = normalize_variable_selection(var_name)
        ds_out = chunk_dataset(
            ds_out, chunk_type, available_mem_gib, selected[0] if selected else None
        )
    else:
        # if no chunking remove chunking encoding because this might cause errors while writing
        for name in list(ds_out.variables):
            enc = ds_out.variables[name].encoding
            enc.pop("chunksizes", None)
            enc.pop("_ChunkSizes", None)
            enc.pop("chunks", None)
            enc["contiguous"] = True
    if ds_out is None:
        msg = f"The dataset read from {file_path} is empty."
        with ErrorLogger(logger):
            raise NotImplementedError()
    logger.debug(f"ds_out: {ds_out}")
    return ds_out


def write_xarray_to_file(
    ds,
    file_path,
    var_name=None,
    create_folder=True,
    encoding=None,
    engine="netcdf4",
    resolution=None,
    crs=None,
    compression=None,
    geotiff_compression=None,
):
    """Write xarray Datasets to file with file type depending on the file suffix.

    Parameters
    ----------
    ds : xr.Dataset or xr.DataArray
        Dataset or data array to write.
    file_path : str or pathlib.Path
        Target file path. The suffix selects ascii, NetCDF or GeoTIFF.
    var_name : str, optional
        Data variable to write or DataArray name override.
    create_folder : bool, default True
        Create the parent directory when it is missing.
    encoding : dict, optional
        Per-variable NetCDF encoding. Its compression keys are replaced by
        `compression`, every other key it holds is kept.
    engine : str, default "netcdf4"
        Xarray NetCDF backend engine. Quantization needs "netcdf4".
    resolution : float, optional
        Cell size used by the ascii writer.
    crs : Any, optional
        Coordinate reference system written to the file.
    compression : NetcdfCompression, optional
        Lossless and lossy compression settings for a NetCDF file. None uses
        inherited settings or defaults. Ignored by ASCII and GeoTIFF writers.
    geotiff_compression : str, optional
        GeoTIFF codec, such as "deflate". None keeps the raster writer default.

    Returns
    -------
    None
    """
    file_path = Path(file_path)
    suffix = file_path.suffix.lower()
    if suffix not in {".asc", ".nc", ".tif", ".tiff"}:
        msg = (
            "Writing to file types other than asci, netcdf, and geotiff is not "
            "implemented. "
            f"The suffix of the file was: {file_path.suffix}"
        )
        with ErrorLogger(logger):
            raise NotImplementedError(msg)
    if create_folder and not file_path.parent.is_dir():
        file_path.parent.mkdir(parents=True, exist_ok=True)
    logger.info(f"Writing file to {file_path}")
    # ds = chunk_if_too_big(ds)
    # ds = ds.chunk({'time': 512, 'lat': 121, 'lon': 131})
    if suffix == ".asc":
        write_xarray_to_ascii(
            ds,
            file_path,
            var_name,
            resolution=resolution,
            crs=crs,
        )
    elif suffix == ".nc":
        if crs is not None:
            resolved = resolve_crs(ds, crs)
            if resolved is not None:
                ds = write_object_crs(ds, resolved)
        write_xarray_to_netcdf(
            ds, file_path, var_name, encoding, engine, compression=compression
        )
    else:
        write_xarray_to_geotiff(
            ds,
            file_path,
            var_name,
            crs=crs,
            geotiff_compression=geotiff_compression,
        )


def write_mask_to_file(
    mask_array, lat, lon, file_path, long_name="mask", var_name="mask"
):
    """Write a boolean mask array to a NetCDF file with CF lat/lon metadata.

    Adds a ``land_mask`` alias for `var_name` via an HDF5 hard link (no data
    duplication), unless `var_name` is already ``"land_mask"``.

    Parameters
    ----------
    mask_array : numpy.ndarray or xarray.DataArray
        2D boolean/int mask, `(lat, lon)` ordered.
    lat, lon : array-like
        Coordinate values for the mask grid.
    file_path : str or pathlib.Path
        NetCDF file to write.
    long_name : str, optional
        CF `long_name` attribute for the mask variable.
    var_name : str, optional
        Name of the mask data variable, by default ``"mask"``.

    Returns
    -------
    pathlib.Path
        The written file path.
    """
    file_path = Path(file_path)
    mask_values = np.where(np.asarray(mask_array), 1, 0)
    mask_da = xr.DataArray(
        mask_values, coords={"lat": lat, "lon": lon}, dims=["lat", "lon"]
    )
    mask_da["lat"].attrs.update(
        {
            "units": "degrees_north",
            "long_name": "latitude",
            "standard_name": "latitude",
            "axis": "Y",
        }
    )
    mask_da["lon"].attrs.update(
        {
            "units": "degrees_east",
            "long_name": "longitude",
            "standard_name": "longitude",
            "axis": "X",
        }
    )
    mask_da.attrs.update(
        {
            "units": "1",
            "long_name": long_name,
            "flag_values": np.array([0, 1], dtype=mask_da.dtype),
            "flag_meanings": "outside_mask inside_mask",
        }
    )
    mask_ds = xr.Dataset({var_name: mask_da})
    for coord in ("lat", "lon"):
        bounds_name = f"{coord}_bnds"
        try:
            mask_ds.coords[bounds_name] = generate_bounds(mask_ds[coord])
            mask_ds[coord].attrs["bounds"] = bounds_name
        except IndexError:
            logger.info(f"Could not generate bounds for coord {coord}")
    encoding = {var_name: dict(NC_ENCODE_MASK)}
    write_xarray_to_file(mask_ds, file_path, encoding=encoding)
    if var_name != "land_mask":
        add_variable_hard_link(file_path, existing_var=var_name, alias_var="land_mask")
    logger.info(f"Mask file has been written to {file_path}")
    return file_path


def write_xarray_to_ascii(
    dataset,
    filepath,
    data_var=None,
    nodata_value=None,
    resolution=None,
    crs=None,
):
    """Write xarray Dataset to an ASCII file that can be read by mHM."""
    # check if a data_var can be optained for writing the data
    if data_var is None and isinstance(dataset, xr.Dataset):
        data_var = get_single_data_var(dataset)
        if data_var is None:
            logger.error(
                f"Data can not be written to {filepath} as the dataset has multiple data_vars, which is incompatible with asci or no datavar exists."
            )
            return
    # get the data from the dataset
    data = dataset[data_var] if isinstance(dataset, xr.Dataset) else dataset
    data = _as_yx(data)
    x_dim, y_dim = data.rio.x_dim, data.rio.y_dim
    _validate_ascii_grid(data)
    if data.sizes[x_dim] > 1 and data[x_dim].values[0] > data[x_dim].values[-1]:
        data = data.isel({x_dim: slice(None, None, -1)})
    if data.sizes[y_dim] > 1 and data[y_dim].values[0] < data[y_dim].values[-1]:
        data = data.isel({y_dim: slice(None, None, -1)})
    resolved = resolve_crs(dataset, crs)

    # set the nodata value
    dtype = get_dtype(data)
    if nodata_value is None:
        nodata_value = get_nodata(data)
        if nodata_value is None:
            is_int = issubclass(np.dtype(dtype).type, (np.integer, np.unsignedinteger))
            typ = int if is_int else float
            nodata_value = typ(NO_DATA)

    header = create_header(data, no_data_value=nodata_value)
    if resolution is not None:
        header["cellsize"] = resolution

    out_header_str = write_header(filepath, header, dtype=dtype)
    rows_per_block = max(
        1, (8 * 1024**2) // (data.sizes[x_dim] * np.dtype(dtype).itemsize)
    )
    is_int = np.issubdtype(np.dtype(dtype), np.integer)
    with Path(filepath).open("a") as output:
        for start in range(0, data.sizes[y_dim], rows_per_block):
            values = data.isel({y_dim: slice(start, start + rows_per_block)}).values
            values = np.where(np.isnan(values), nodata_value, values).astype(dtype)
            np.savetxt(output, values, fmt="%i" if is_int else "%f")
    logger.info(f"Writting file to {filepath}")
    logger.debug(f"Header written:\n{out_header_str}")
    prj_path = Path(filepath).with_suffix(".prj")
    upper_prj_path = prj_path.with_suffix(".PRJ")
    if resolved is None:
        prj_path.unlink(missing_ok=True)
        upper_prj_path.unlink(missing_ok=True)
    else:
        upper_prj_path.unlink(missing_ok=True)
        prj_path.write_text(resolved.to_wkt(), encoding="utf-8")


def _validate_ascii_grid(data):
    """Reject grids that the single-cellsize ASCII format cannot represent."""
    if data.ndim != 2:
        msg = f"ASCII output requires two-dimensional data, got {data.ndim}."
        raise ValueError(msg)
    spatial = set_spatial_dims(data)
    transform = spatial.rio.transform()
    scale = max(abs(transform.a), abs(transform.e), 1.0)
    tolerance = scale * 1e-10
    if not np.isclose(transform.b, 0.0, atol=tolerance) or not np.isclose(
        transform.d, 0.0, atol=tolerance
    ):
        msg = "ASCII output does not support rotated or sheared grids."
        raise ValueError(msg)
    if not np.isclose(abs(transform.a), abs(transform.e)):
        msg = "ASCII output requires equal X and Y cell sizes."
        raise ValueError(msg)


def write_xarray_to_geotiff(
    dataset,
    filepath,
    data_var=None,
    nodata_value=None,
    crs=None,
    geotiff_compression=None,
):
    """Write a 2D payload to filepath, optionally using geotiff_compression."""
    if isinstance(dataset, xr.Dataset):
        data_var = data_var or get_single_data_var(dataset)
        if data_var is None or data_var not in dataset.data_vars:
            msg = "GeoTIFF output requires exactly one payload data variable."
            raise ValueError(msg)
        data = dataset[data_var]
    else:
        data = dataset.rename(data_var) if data_var is not None else dataset
    if data.ndim != 2:
        msg = f"GeoTIFF output requires two-dimensional data, got {data.ndim}."
        raise ValueError(msg)

    data = _as_yx(data)
    resolved = resolve_crs(data, crs, required=True)
    data = data.rio.write_crs(resolved, inplace=False)
    nodata_value = get_nodata(data) if nodata_value is None else nodata_value
    has_missing = bool(data.isnull().any().compute().item())
    if nodata_value is None and has_missing:
        nodata_value = NO_DATA

    output_dtype = _geotiff_dtype(data, nodata_value)
    if nodata_value is not None:
        data = data.where(data.notnull(), nodata_value)
    data = data.astype(output_dtype)
    data.encoding = {}
    for key in ("_FillValue", "missing_value", "scale_factor", "add_offset"):
        data.attrs.pop(key, None)
    if nodata_value is not None:
        data = data.rio.write_nodata(nodata_value, encoded=True, inplace=False)
    Path(filepath).parent.mkdir(parents=True, exist_ok=True)
    options = {} if geotiff_compression is None else {"compress": geotiff_compression}
    data.rio.to_raster(filepath, dtype=output_dtype.name, **options)


def _geotiff_dtype(data, nodata_value):
    """Choose a numeric output dtype compatible with decoded raster values."""
    encoded_dtype = data.encoding.get("rasterio_dtype") or data.encoding.get("dtype")
    scaled = any(key in data.encoding for key in ("scale_factor", "add_offset"))
    try:
        dtype = np.dtype(encoded_dtype) if encoded_dtype is not None else data.dtype
    except TypeError:
        dtype = data.dtype
    if scaled or dtype.kind not in "iuf":
        dtype = data.dtype if data.dtype.kind in "iuf" else np.dtype("float64")
    if nodata_value is None or dtype.kind == "f":
        return np.dtype(dtype)
    return _dtype_holding_nodata(dtype, nodata_value)


def read_ascii_to_xarray(
    filepath,
    var_name=None,
    landcover=False,
    landcover_year_start=None,
):
    """Read an mHM readable asci file to an xarray dataset."""
    # Read the header from the file
    with filepath.open("r") as f:
        header = {}

        for i, line in enumerate(f.readlines()):
            line_striped = line.strip()
            if not line_striped:
                continue
            logger.debug(f"File {filepath.name} {i}: {line_striped}")
            key, value = line_striped.split()
            header[key.lower()] = float(value) if "." in value else int(value)
            if len(header) == 6:
                break
        # Extract header information
        ncols = header["ncols"]
        nrows = header["nrows"]
        xllcorner = header["xllcorner"]
        yllcorner = header["yllcorner"]
        cellsize = header["cellsize"]
        nodata_value = header["nodata_value"]

    # Load the data values
    data_values = np.asarray(np.loadtxt(filepath, skiprows=i + 1)).reshape(nrows, ncols)
    if isinstance(nodata_value, int):
        data_values = data_values.astype(np.int32)

    # Calculate latitude and longitude coordinates
    lon = np.arange(
        xllcorner + cellsize / 2, xllcorner + (ncols + 0.5) * cellsize, cellsize
    )
    lat = np.arange(
        yllcorner + (nrows - 0.5) * cellsize, yllcorner - cellsize / 2, -cellsize
    )
    logger.debug(lon)
    logger.debug(lat)

    # Create DataArray with lat/lon dimensions and nodata value
    name = "data" if var_name is None else var_name
    coords = {"lon": ("lon", lon, {"axis": "X"}), "lat": ("lat", lat, {"axis": "Y"})}

    # If this is a landcover file add a 1-element time coordinate
    if landcover and landcover_year_start is not None:
        start_ts = np.datetime64(f"{landcover_year_start}-01-01", "ns")
        time = np.array([start_ts], dtype="datetime64[ns]")
        coords["time"] = ("time", time)

    # If we added a time coordinate, expand the data to have a leading time dim
    if "time" in coords:
        data_arr = np.expand_dims(data_values, axis=0)  # shape (1, nrows, ncols)
        dims = ["time", "lat", "lon"]
    else:
        data_arr = data_values
        dims = ["lat", "lon"]

    da = xr.DataArray(
        data=data_arr,
        dims=dims,
        coords=coords,
        name=name,
        attrs={"nodata_value": nodata_value, "_FillValue": nodata_value},
    )
    da = set_spatial_dims(da).rio.write_transform(
        from_origin(
            xllcorner,
            yllcorner + nrows * cellsize,
            cellsize,
            cellsize,
        ),
        inplace=False,
    )

    prj_path = Path(filepath).with_suffix(".prj")
    if not prj_path.is_file():
        upper_prj = prj_path.with_suffix(".PRJ")
        prj_path = upper_prj if upper_prj.is_file() else prj_path
    if prj_path.is_file():
        da = da.rio.write_crs(
            CRS.from_wkt(prj_path.read_text(encoding="utf-8")),
            inplace=False,
        )
    return da.to_dataset()


def get_coord_values(ds, lat=False, lon=False):
    """Get latitude or longitude values from DataSet."""
    key = get_coord_key(ds, lat=lat, lon=lon)
    return ds[key].values


def get_dataset_from_path(
    path,
    var_name=None,
    chunking=None,
    available_mem_gib=None,
    chunk_type=ChunkType.SPACE,
    use_mfdataset=False,
    engine="netcdf4",
    normalize_latlon_coords=False,
    force_decending_y=False,
    force_ascending_y=False,
    landcover=False,
    landcover_year_start=None,
    available_mem=None,
    file_name="*.*",
    create_bounds=False,
):
    """Load a dataset from a file, directory, or pattern.

    This mirrors ``get_xarray_ds_from_file`` for single-file inputs and
    extends it to directories (multi-file datasets).
    """
    if path is None:
        path_is_none_msg = "Input path is None. Please provide a valid file path, directory path, or glob pattern."
        with ErrorLogger(logger):
            raise ValueError(path_is_none_msg)

    if available_mem_gib is None and available_mem is not None:
        available_mem_gib = available_mem
        if chunking is None:
            chunking = True
    if chunking is None:
        chunking = False

    def _postprocess(ds_out):
        lat_key = get_coord_key(ds_out, lat=True, raise_exception=False)
        lon_key = get_coord_key(ds_out, lon=True, raise_exception=False)
        # normalize before the axis order is read, because a coordinate that is
        # not an index yet makes the comparison below positional and therefore
        # always ascending, which would silently keep a flipped grid
        if normalize_latlon_coords:
            ds_out = normalize_lat_lon(
                ds_out, lat_key=lat_key, lon_key=lon_key, raise_exceptions=False
            )
            lat_key = get_coord_key(ds_out, lat=True, raise_exception=False)
            lon_key = get_coord_key(ds_out, lon=True, raise_exception=False)

        if lat_key is not None and (
            (force_decending_y and ds_out[lat_key].data[0] < ds_out[lat_key].data[-1])
            or (
                force_ascending_y and ds_out[lat_key].data[0] > ds_out[lat_key].data[-1]
            )
        ):
            ds_out = ds_out.sel({lat_key: slice(None, None, -1)})
        if create_bounds:
            ds_out = generate_bounds_for_all_coords(ds_out)

        logger.debug(ds_out)
        logger.debug(lat_key)
        logger.debug(lon_key)

        if lon_key is None and lat_key is None:
            logger.warning("Dataset does not have lon and lat key.")
        elif lon_key is None or lat_key is None:
            logger.error("Dataset has only one of lon at lat keys.")

        if chunking and available_mem_gib is not None:
            # several variables share one dtype and shape, so sizing the chunks
            # on the first of them sizes them for each
            selected = normalize_variable_selection(var_name)
            chunk_var = selected[0] if selected else None
            ds_out = chunk_dataset(ds_out, chunk_type, available_mem_gib, chunk_var)
        else:
            for name in list(ds_out.variables):
                enc = ds_out.variables[name].encoding
                enc.pop("chunksizes", None)
                enc.pop("_ChunkSizes", None)
                enc.pop("chunks", None)
                enc["contiguous"] = True

        return ds_out

    def _is_dir_or_list(p):
        if isinstance(p, list):
            return True
        if isinstance(p, str):
            p = Path(p)
        return p.is_dir()

    if _is_dir_or_list(path):
        file_list = []
        if not isinstance(path, list):
            path = Path(path)
            file_list = list(path.rglob(file_name))
        else:
            path = [Path(p) for p in path]
            for p in path:
                if p.is_file():
                    file_list.append(p)
                elif p.is_dir():
                    file_list.extend(list(p.rglob(file_name)))
        if not file_list:
            with ErrorLogger(logger):
                msg = f"No files found in {path}."
                raise ValueError(msg)
        if len(file_list) == 1:
            return get_xarray_ds_from_file(
                file_list[0],
                var_name=var_name,
                chunking=chunking,
                available_mem_gib=available_mem_gib,
                chunk_type=chunk_type,
                use_mfdataset=use_mfdataset,
                engine=engine,
                normalize_latlon_coords=normalize_latlon_coords,
                force_decending_y=force_decending_y,
                force_ascending_y=force_ascending_y,
                landcover=landcover,
                landcover_year_start=landcover_year_start,
                create_bounds=create_bounds,
            )
        non_nc = [p for p in file_list if Path(p).suffix != ".nc"]
        if non_nc:
            with ErrorLogger(logger):
                msg = "Multi-file loading supports NetCDF files only."
                raise ValueError(msg)
        ds_out = read_dataset(
            file_list,
            use_mfdataset=use_mfdataset,
            engine=engine,
            variables=normalize_variable_selection(var_name),
        )
        return _postprocess(ds_out)

    path_in = path
    path = Path(path_in)

    if path.is_file():
        return get_xarray_ds_from_file(
            path,
            var_name=var_name,
            chunking=chunking,
            available_mem_gib=available_mem_gib,
            chunk_type=chunk_type,
            use_mfdataset=use_mfdataset,
            engine=engine,
            normalize_latlon_coords=normalize_latlon_coords,
            force_decending_y=force_decending_y,
            force_ascending_y=force_ascending_y,
            landcover=landcover,
            landcover_year_start=landcover_year_start,
            create_bounds=create_bounds,
        )

    path_str = str(path_in)
    if any(w in path_str for w in ("*", "?", "[", "]")) and path_str.endswith(".nc"):
        ds_out = read_dataset(
            path_str,
            use_mfdataset=use_mfdataset,
            engine=engine,
            variables=normalize_variable_selection(var_name),
        )
        return _postprocess(ds_out)

    with ErrorLogger(logger):
        msg = f"Path {path} does not exist."
        raise ValueError(msg)
