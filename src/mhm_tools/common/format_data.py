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
from mhm_tools.common.file_handler import (
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


def get_categorical_output_path(
    output_path: PathLike, class_field: str, extension: str
) -> Path:
    """Return the validated output path for a categorical class raster."""
    extension = str(extension).lower().lstrip(".")
    if f".{extension}" not in _OUTPUT_SUFFIXES:
        msg = "Output extension must be 'nc', 'asc', or 'tif'."
        raise ValueError(msg)
    return Path(output_path) / f"{class_field.lower()}.{extension}"


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


def format_categorical_data(
    input_file: PathLike,
    dem_file: PathLike,
    output_file: PathLike,
    table,
    mapping_field: str,
    class_field: str,
    *,
    input_crs: str | None = None,
    dem_crs: str | None = None,
) -> Path:
    """Map a categorical raster to classes on the exact DEM grid."""
    input_file = Path(input_file)
    dem_file = Path(dem_file)
    output_file = Path(output_file)
    if not input_file.is_file():
        msg = f"Input raster does not exist: {input_file}"
        raise ValueError(msg)
    if input_file.suffix.lower() not in _INPUT_SUFFIXES:
        suffixes = ", ".join(sorted(_INPUT_SUFFIXES))
        msg = f"Input must be a raster with one of these suffixes: {suffixes}."
        raise ValueError(msg)
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

    mapping = _lookup_mapping(table, mapping_field, class_field)
    source = get_raster_data(input_file, crs=input_crs)
    try:
        reference = get_raster_data(dem_file, crs=dem_crs)
        try:
            aligned = align_raster_to_reference(source, reference, nodata=int(_NODATA))
            classes = _reclassify(aligned, mapping)
            variable_name = class_field.lower()
            output = set_grid(
                classes,
                get_grid(reference.to_dataset(name="_dem"), "_dem"),
                variable_name,
                data_attrs={
                    "long_name": f"mHM {variable_name.replace('_', ' ')}",
                    "nodata_value": int(_NODATA),
                },
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
    finally:
        source.close()

    if not output_file.is_file():
        msg = f"Formatted raster was not created: {output_file}"
        raise RuntimeError(msg)
    logger.info("Wrote formatted categorical data to %s", output_file)
    return output_file
