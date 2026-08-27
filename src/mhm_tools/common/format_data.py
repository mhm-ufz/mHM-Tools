"""Shared categorical raster formatting for mHM."""

from __future__ import annotations

import contextlib
import logging
from pathlib import Path
from typing import Union

import geopandas as gpd
import numpy as np
import pandas as pd
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

logger = logging.getLogger(__name__)

PathLike = Union[str, Path]
_INPUT_SUFFIXES = {".asc", ".nc", ".tif", ".tiff"}
_OUTPUT_SUFFIXES = {".asc", ".nc", ".tif"}
_NODATA = np.int32(NO_DATA)
_MANIFEST_NAMES = ("format-data.csv", "format-data.txt")


def _normalise_field_name(field_name: object) -> str:
    field_text = str(field_name).strip().lstrip("*").strip()
    if "[" in field_text:
        field_text = field_text.split("[", 1)[0].strip()
    return "".join(char.lower() for char in field_text if char.isalnum())


def _resolve_field(columns, requested: str):
    normalized = _normalise_field_name(requested)
    if not normalized:
        msg = "The lookup mapping field must not be empty."
        raise ValueError(msg)
    exact = [column for column in columns if str(column) == requested]
    if exact:
        return exact[0]
    matches = [
        column for column in columns if _normalise_field_name(column) == normalized
    ]
    if not matches:
        available = ", ".join(str(column) for column in columns)
        msg = (
            f"Lookup table field {requested!r} was not found. "
            f"Available fields: {available or '<none>'}."
        )
        raise ValueError(msg)
    if len(matches) > 1:
        names = ", ".join(str(column) for column in matches)
        msg = (
            f"Lookup table field {requested!r} is ambiguous after "
            f"normalization: {names}."
        )
        raise ValueError(msg)
    return matches[0]


def _finite_number(value: object, field: object, row_number: int) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        msg = f"Lookup row {row_number} has a non-numeric {field!r} value: {value!r}."
        raise ValueError(msg) from exc
    if not np.isfinite(number):
        msg = f"Lookup row {row_number} has a non-finite {field!r} value: {value!r}."
        raise ValueError(msg)
    return number


def _is_blank(value: object) -> bool:
    if value is None:
        return True
    if isinstance(value, str):
        return not value.strip()
    try:
        missing = pd.isna(value)
    except (TypeError, ValueError):
        return False
    return bool(missing) if np.isscalar(missing) else False


def read_lookup_table(lookup_table: PathLike):
    """Read a non-empty OGR-compatible lookup table without geometry."""
    lookup_table = Path(lookup_table)
    if not lookup_table.is_file():
        msg = f"Lookup table does not exist: {lookup_table}"
        raise ValueError(msg)
    try:
        table = gpd.read_file(lookup_table, ignore_geometry=True)
    except Exception as exc:
        msg = f"Could not read lookup table {lookup_table}: {exc}"
        raise ValueError(msg) from exc
    if table.empty:
        msg = f"Lookup table {lookup_table} is empty."
        raise ValueError(msg)
    return table


def get_format_manifest_path(input_path: PathLike) -> Path:
    """Return the single format-data manifest in an input directory."""
    input_path = Path(input_path)
    if not input_path.is_dir():
        msg = f"Input path must be a directory: {input_path}"
        raise ValueError(msg)
    manifests = [input_path / name for name in _MANIFEST_NAMES]
    manifests = [path for path in manifests if path.is_file()]
    if not manifests:
        names = " or ".join(_MANIFEST_NAMES)
        msg = f"Input directory must contain {names}: {input_path}"
        raise ValueError(msg)
    if len(manifests) > 1:
        names = ", ".join(str(path) for path in manifests)
        msg = f"Input directory contains multiple format-data manifests: {names}"
        raise ValueError(msg)
    return manifests[0]


def read_format_manifest(
    input_path: PathLike,
    required_columns,
    *,
    skiprows: int = 0,
):
    """Read and normalize a comma-separated ``format-data`` manifest."""
    manifest = get_format_manifest_path(input_path)

    try:
        table = pd.read_csv(manifest, skiprows=skiprows)
    except Exception as exc:
        msg = f"Could not read format-data manifest {manifest}: {exc}"
        raise ValueError(msg) from exc
    if table.empty:
        msg = f"Format-data manifest {manifest} is empty."
        raise ValueError(msg)

    columns = {}
    for column in table.columns:
        normalized = _normalise_field_name(column)
        if normalized in columns:
            msg = (
                f"Format-data manifest {manifest} has ambiguous columns "
                f"{columns[normalized]!r} and {column!r}."
            )
            raise ValueError(msg)
        columns[normalized] = column

    rename = {}
    for required in required_columns:
        column = columns.get(_normalise_field_name(required))
        if column is None:
            available = ", ".join(str(name) for name in table.columns)
            msg = (
                f"Format-data manifest {manifest} is missing required column "
                f"{required!r}. Available columns: {available or '<none>'}."
            )
            raise ValueError(msg)
        rename[column] = required
    return manifest, table.rename(columns=rename)[list(required_columns)].copy()


def get_categorical_output_path(
    output_path: PathLike, output_name: str, extension: str
) -> Path:
    """Return the validated output path for a categorical class raster."""
    extension = str(extension).lower().lstrip(".")
    if f".{extension}" not in _OUTPUT_SUFFIXES:
        msg = "Output extension must be 'nc', 'asc', or 'tif'."
        raise ValueError(msg)
    return Path(output_path) / f"{output_name.lower()}.{extension}"


def _lookup_mapping(table, mapping_field: str, class_field: str) -> dict:
    key_field = _resolve_field(table.columns, mapping_field)
    value_field = _resolve_field(table.columns, class_field)
    mapping = {}
    int32_max = np.iinfo(np.int32).max
    for row_number, (key_value, class_value) in enumerate(
        zip(table[key_field], table[value_field]), start=1
    ):
        if _is_blank(key_value):
            logger.warning(
                "Skipping lookup row %d because %s is empty.",
                row_number,
                key_field,
            )
            continue
        key = _finite_number(key_value, key_field, row_number)
        class_number = _finite_number(class_value, value_field, row_number)
        if not class_number.is_integer() or not 0 < class_number <= int32_max:
            msg = (
                f"Lookup row {row_number} has invalid {class_field} "
                f"{class_value!r}; expected a positive int32 value."
            )
            raise ValueError(msg)
        target = int(class_number)
        previous = mapping.get(key)
        if previous is not None and previous != target:
            msg = (
                f"Lookup key {key_value!r} maps to conflicting {class_field} "
                f"values {previous} and {target}."
            )
            raise ValueError(msg)
        mapping[key] = target
    if not mapping:
        msg = "Lookup table contains no usable category mappings."
        raise ValueError(msg)
    return mapping


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
            "y": np.arange(rows, dtype=np.float64)
            if y is None
            else np.asarray(y, dtype=np.float64),
            "x": np.arange(columns, dtype=np.float64)
            if x is None
            else np.asarray(x, dtype=np.float64),
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
        return set_grid(
            values,
            get_grid(reference.to_dataset(name="_dem"), "_dem"),
            variable_name,
            data_attrs={
                "long_name": f"mHM {variable_name.replace('_', ' ')}",
                "units": "1",
                "nodata_value": int(_NODATA),
            },
        )
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
                "zlib": True,
                "complevel": 4,
                "shuffle": True,
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
        )
    finally:
        reference.close()

    if not output_file.is_file():
        msg = f"Formatted raster was not created: {output_file}"
        raise RuntimeError(msg)
    logger.info("Wrote formatted categorical data to %s", output_file)
    return output_file
