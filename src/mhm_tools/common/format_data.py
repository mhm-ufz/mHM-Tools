"""
Shared categorical raster formatting for mHM.

Authors
-------
- Sanjeev Bashyal
"""

from __future__ import annotations

import contextlib
import logging
from pathlib import Path
from typing import Union

import numpy as np
import xarray as xr

from mhm_tools.common.constants import NO_DATA
from mhm_tools.common.crs_handler import set_spatial_dims
from mhm_tools.common.file_handler import (
    _raster_nodata_values,
    align_raster_to_reference,
    get_grid,
    get_raster_data,
    set_grid,
    write_xarray_to_file,
)
from mhm_tools.common.lookup_handler import _lookup_mapping
from mhm_tools.common.netcdf import COMPRESSION_ENCODING_KEYS

logger = logging.getLogger(__name__)

PathLike = Union[str, Path]
_INPUT_SUFFIXES = {".asc", ".nc", ".tif", ".tiff"}
_OUTPUT_SUFFIXES = {".asc", ".nc", ".tif"}
_NODATA = np.int32(NO_DATA)


def get_categorical_output_path(
    output_path: PathLike, output_name: str, extension: str
) -> Path:
    """Return the validated output path for a categorical class raster."""
    extension = str(extension).lower().lstrip(".")
    if f".{extension}" not in _OUTPUT_SUFFIXES:
        msg = "Output extension must be 'nc', 'asc', or 'tif'."
        raise ValueError(msg)
    return Path(output_path) / f"{output_name.lower()}.{extension}"


def _nodata_values(data: xr.DataArray):
    candidates = []
    for source in (data.attrs, data.encoding):
        for key in ("_FillValue", "missing_value", "nodata_value"):
            if key in source:
                candidates.extend(np.asarray(source[key]).reshape(-1).tolist())
    with contextlib.suppress(AttributeError, ImportError):
        candidates.append(data.rio.nodata)

    seen = set()
    for candidate in candidates:
        if candidate is None:
            continue
        try:
            number = float(candidate)
        except (TypeError, ValueError):
            continue
        if np.isfinite(number) and number not in seen:
            seen.add(number)
            yield number


def _format_categories(categories: np.ndarray, limit: int = 10) -> str:
    values = []
    for value in categories[:limit]:
        number = float(value)
        values.append(str(int(number)) if number.is_integer() else str(number))
    suffix = ", ..." if categories.size > limit else ""
    return ", ".join(values) + suffix


def _reclassify(data: xr.DataArray, mapping: dict) -> np.ndarray:
    values = np.asarray(data.values)
    if not np.issubdtype(values.dtype, np.number) or np.iscomplexobj(values):
        msg = (
            f"Raster variable {data.name!r} must contain real numeric categories; "
            f"found dtype {values.dtype}."
        )
        raise TypeError(msg)

    valid = np.ones(values.shape, dtype=bool)
    if np.issubdtype(values.dtype, np.floating):
        valid &= np.isfinite(values)
    for nodata in _nodata_values(data):
        valid &= values != nodata

    output = np.full(values.shape, _NODATA, dtype=np.int32)
    mapped = np.zeros(values.shape, dtype=bool)
    for key, target in mapping.items():
        matches = valid & (values == key)
        if np.any(matches):
            output[matches] = target
            mapped |= matches
    if not np.any(mapped):
        msg = "No valid raster category matched the selected lookup mapping field."
        raise ValueError(msg)

    unmatched = valid & ~mapped
    if np.any(unmatched):
        categories = np.unique(values[unmatched])
        logger.warning(
            "%d valid raster cells in %d categor%s were not present in the "
            "lookup table and were written as %d: %s",
            int(np.count_nonzero(unmatched)),
            categories.size,
            "y" if categories.size == 1 else "ies",
            int(_NODATA),
            _format_categories(categories),
        )
    return output


def reclassify_categorical_raster(
    data: xr.DataArray,
    table,
    mapping_field: str,
    class_field: str,
    *,
    variable_name: str,
) -> xr.DataArray:
    """Map categories on their source grid before spatial resampling."""
    values = _reclassify(data, _lookup_mapping(table, mapping_field, class_field))
    attrs = {
        "long_name": f"mHM {variable_name.replace('_', ' ')}",
        "units": "1",
        "nodata_value": int(_NODATA),
    }
    result = xr.DataArray(
        values,
        dims=data.dims,
        coords=data.coords,
        name=variable_name,
        attrs=attrs,
    )
    result = result.rio.set_spatial_dims(
        x_dim=data.rio.x_dim,
        y_dim=data.rio.y_dim,
    )
    result = result.rio.write_crs(data.rio.crs, inplace=False)
    result = result.rio.write_transform(data.rio.transform(), inplace=False)
    return result.rio.write_nodata(int(_NODATA), inplace=False)


def fill_grid_nodata(
    values: np.ndarray,
    *,
    x=None,
    y=None,
    mask=None,
    missing_value=None,
    fill_value=np.nan,
    name: str = "data",
    source_file: PathLike | None = None,
) -> int:
    """Fill the nodata cells of one raster block from nearest valid neighbours.

    The block is wrapped in a labelled data array so that the shared
    :func:`mhm_tools.pre.fill_nearest.fill_dataarray_with_nearest`
    interpolator can be reused as is.

    Parameters
    ----------
    values : numpy.ndarray
        Two-dimensional ``(y, x)`` array modified in place.
    x, y : numpy.ndarray, optional
        Cell-centre coordinates along each axis. Cell indices are used when a
        coordinate vector is omitted, which is only correct for square cells.
    mask : numpy.ndarray, optional
        Boolean array where true cells are excluded from filling and set to
        ``fill_value``. Cells inside the mask still act as fill sources.
    missing_value : float, optional
        Missing-value marker to fill. NaNs are always treated as missing.
    fill_value : float, default numpy.nan
        Value written to masked cells and, when no valid source cell exists,
        to the target cells.
    name : str, default "data"
        Variable name used in the log messages of the interpolator.
    source_file : path-like, optional
        File path used only for diagnostic log messages.

    Returns
    -------
    int
        Number of unmasked cells filled from a nearest valid neighbour.
    """
    from mhm_tools.pre.fill_nearest import fill_dataarray_with_nearest

    values = np.asarray(values)
    if values.ndim != 2:
        msg = (
            "Nearest-neighbour grid filling requires a two-dimensional array; "
            f"got shape {values.shape}."
        )
        raise ValueError(msg)

    rows, columns = values.shape
    array = xr.DataArray(
        values,
        dims=("y", "x"),
        coords={
            "y": (
                np.arange(rows, dtype=np.float64)
                if y is None
                else np.asarray(y, dtype=np.float64)
            ),
            "x": (
                np.arange(columns, dtype=np.float64)
                if x is None
                else np.asarray(x, dtype=np.float64)
            ),
        },
        name=name,
    )
    return fill_dataarray_with_nearest(
        array,
        along_time=False,
        missing_value=missing_value,
        mask=mask,
        fill_value=fill_value,
        source_file=source_file,
    )


def _reference_valid(reference: xr.DataArray) -> np.ndarray:
    """Return the boolean mask of reference cells that carry data."""
    values = np.asarray(reference.values)
    valid = np.ones(values.shape, dtype=bool)
    if np.issubdtype(values.dtype, np.floating):
        valid &= np.isfinite(values)
    for nodata in _raster_nodata_values(reference):
        valid &= values != nodata
    return valid


def _fill_aligned_nodata(
    values: np.ndarray,
    reference: xr.DataArray,
    *,
    variable_name: str,
    input_file: PathLike,
) -> np.ndarray:
    """Restrict classes to the reference domain and close their nodata gaps."""
    # Aligned values follow the (y, x) order that align_raster_to_reference uses.
    reference = set_spatial_dims(reference)
    y_dim, x_dim = reference.rio.y_dim, reference.rio.x_dim
    reference = reference.transpose(y_dim, x_dim)
    valid = _reference_valid(reference)
    if not np.any(valid):
        msg = "The reference raster has no valid cell to define the domain."
        raise ValueError(msg)
    y_values = np.asarray(reference[y_dim].values, dtype=np.float64)
    x_values = np.asarray(reference[x_dim].values, dtype=np.float64)
    missing = int(np.count_nonzero((values == _NODATA) & valid))
    filled = fill_grid_nodata(
        values,
        x=x_values,
        y=y_values,
        mask=~valid,
        missing_value=float(_NODATA),
        fill_value=int(_NODATA),
        name=variable_name,
        source_file=input_file,
    )
    if missing:
        logger.info(
            "Filled %d of %d nodata cells of %s from nearest valid neighbours.",
            filled,
            missing,
            variable_name,
        )
    return values


def prepare_categorical_data(
    input_file: PathLike,
    reference: xr.DataArray,
    table,
    mapping_field: str,
    class_field: str,
    *,
    variable_name: str,
    input_crs: str | None = None,
    resampling="nearest",
    mask_reference: bool = False,
    fill_nodata: bool = False,
) -> xr.Dataset:
    """Classify one source raster and align it to an open reference raster.

    When ``fill_nodata`` is true the aligned classes are cut to the reference
    domain and their remaining nodata cells are taken from the nearest
    classified neighbour, which may lie outside that domain.
    """
    input_file = Path(input_file)
    if not input_file.is_file():
        msg = f"Input raster does not exist: {input_file}"
        raise ValueError(msg)
    if input_file.suffix.lower() not in _INPUT_SUFFIXES:
        suffixes = ", ".join(sorted(_INPUT_SUFFIXES))
        msg = f"Input must be a raster with one of these suffixes: {suffixes}."
        raise ValueError(msg)

    source = get_raster_data(input_file, crs=input_crs)
    try:
        classified = reclassify_categorical_raster(
            source,
            table,
            mapping_field,
            class_field,
            variable_name=variable_name,
        )
        aligned = align_raster_to_reference(
            classified,
            reference,
            nodata=int(_NODATA),
            resampling=resampling,
            data_kind="categorical",
            mask_reference=mask_reference,
        )
        values = np.asarray(aligned.values, dtype=np.int32)
        if fill_nodata:
            values = _fill_aligned_nodata(
                values,
                reference,
                variable_name=variable_name,
                input_file=input_file,
            )
        output = set_grid(
            values,
            get_grid(reference.to_dataset(name="_dem"), "_dem"),
            variable_name,
            data_attrs={
                "long_name": f"mHM {variable_name.replace('_', ' ')}",
                "units": "1",
                "nodata_value": int(_NODATA),
            },
        )
        if input_file.suffix.lower() == ".nc":
            output[variable_name].encoding.update(
                {
                    key: value
                    for key, value in source.encoding.items()
                    if key in COMPRESSION_ENCODING_KEYS
                }
            )
        return output
    finally:
        source.close()


def format_categorical_data(
    input_file: PathLike,
    dem_file: PathLike,
    output_file: PathLike,
    table,
    mapping_field: str,
    class_field: str,
    *,
    variable_name: str,
    input_crs: str | None = None,
    dem_crs: str | None = None,
    resampling="nearest",
    fill_nodata: bool = False,
    compression=None,
) -> Path:
    """Map a categorical raster to classes on the exact DEM grid."""
    input_file = Path(input_file)
    dem_file = Path(dem_file)
    output_file = Path(output_file)
    if not dem_file.is_file():
        msg = f"DEM raster does not exist: {dem_file}"
        raise ValueError(msg)
    if dem_file.suffix.lower() not in _INPUT_SUFFIXES:
        suffixes = ", ".join(sorted(_INPUT_SUFFIXES))
        msg = f"DEM must be a raster with one of these suffixes: {suffixes}."
        raise ValueError(msg)
    if output_file.suffix.lower() not in _OUTPUT_SUFFIXES:
        msg = "Output extension must be 'nc', 'asc', or 'tif'."
        raise ValueError(msg)
    if output_file.parent.exists() and not output_file.parent.is_dir():
        msg = f"Output path must be a directory: {output_file.parent}"
        raise ValueError(msg)
    if output_file.resolve() in {input_file.resolve(), dem_file.resolve()}:
        msg = f"Raster output must differ from input files: {output_file}"
        raise ValueError(msg)

    if not input_file.is_file():
        msg = f"Input raster does not exist: {input_file}"
        raise ValueError(msg)
    if input_file.suffix.lower() not in _INPUT_SUFFIXES:
        suffixes = ", ".join(sorted(_INPUT_SUFFIXES))
        msg = f"Input must be a raster with one of these suffixes: {suffixes}."
        raise ValueError(msg)

    reference = get_raster_data(dem_file, crs=dem_crs)
    try:
        output = prepare_categorical_data(
            input_file,
            reference,
            table,
            mapping_field,
            class_field,
            variable_name=variable_name,
            input_crs=input_crs,
            resampling=resampling,
            fill_nodata=fill_nodata,
        )
        encoding = {
            variable_name: {
                "_FillValue": int(_NODATA),
                "dtype": "int32",
            }
        }
        write_xarray_to_file(
            output,
            output_file,
            var_name=variable_name,
            encoding=encoding if output_file.suffix.lower() == ".nc" else None,
            crs=reference.rio.crs,
            compression=compression,
        )
    finally:
        reference.close()

    if not output_file.is_file():
        msg = f"Formatted raster was not created: {output_file}"
        raise RuntimeError(msg)
    logger.info("Wrote formatted categorical data to %s", output_file)
    return output_file
