"""
Lookup-table and CSV-manifest reading and value mapping.

Authors
-------
- Sanjeev Bashyal
"""

import logging
from pathlib import Path
from typing import Union

import geopandas as gpd
import numpy as np
import pandas as pd

from mhm_tools.common.constants import NO_DATA

logger = logging.getLogger(__name__)

PathLike = Union[str, Path]


def _normalise_field_name(field_name: object) -> str:
    """Normalize a vector or lookup-table field name for matching."""
    field_text = str(field_name).strip().lstrip("*").strip()
    if "[" in field_text:
        field_text = field_text.split("[", 1)[0].strip()
    return "".join(char.lower() for char in field_text if char.isalnum())


def _resolve_field(columns, requested: str, source: str = "lookup table"):
    """Resolve an exact or normalized field name."""
    normalized = _normalise_field_name(requested)
    if not normalized:
        msg = f"The {source} field must be a non-empty column name."
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
            f"Field {requested!r} was not found in the {source}. "
            f"Available fields: {available or '<none>'}."
        )
        raise ValueError(msg)
    if len(matches) > 1:
        names = ", ".join(str(column) for column in matches)
        msg = f"Field {requested!r} is ambiguous in the {source}: {names}."
        raise ValueError(msg)
    return matches[0]


def _required_number(
    value: object,
    field: object,
    row_number: int,
    source: str = "Lookup",
) -> float:
    """Return one required finite number from a lookup or manifest row."""
    if _is_blank(value):
        msg = f"{source} row {row_number} has an empty {field!r} value."
        raise ValueError(msg)
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        msg = (
            f"{source} row {row_number} has a non-numeric "
            f"{field!r} value: {value!r}."
        )
        raise ValueError(msg) from exc
    if not np.isfinite(number):
        msg = (
            f"{source} row {row_number} has a non-finite "
            f"{field!r} value: {value!r}."
        )
        raise ValueError(msg)
    return number


def _required_integer(
    value: object,
    field: object,
    row_number: int,
    source: str = "Lookup",
) -> int:
    """Return one required integer from a lookup or manifest row."""
    number = _required_number(value, field, row_number, source)
    if not number.is_integer():
        msg = (
            f"{source} row {row_number} has a non-integer "
            f"{field!r} value: {value!r}."
        )
        raise ValueError(msg)
    return int(number)


def _is_blank(value: object) -> bool:
    """Return whether a lookup or manifest value is empty."""
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
    """Read a non-empty spatial or comma-separated lookup table."""
    lookup_table = Path(lookup_table)
    if not lookup_table.is_file():
        msg = f"Lookup table does not exist: {lookup_table}"
        raise ValueError(msg)
    try:
        if lookup_table.suffix.lower() in {".csv", ".txt"}:
            table = pd.read_csv(lookup_table)
        else:
            table = gpd.read_file(lookup_table, ignore_geometry=True)
    except Exception as exc:
        msg = f"Could not read lookup table {lookup_table}: {exc}"
        raise ValueError(msg) from exc
    if table.empty:
        msg = f"Lookup table {lookup_table} is empty."
        raise ValueError(msg)
    return table


def read_format_manifest(input_file: PathLike, required_columns):
    """Read and normalize a comma-separated manifest."""
    manifest = Path(input_file)
    if not manifest.is_file():
        msg = f"Manifest file does not exist: {manifest}"
        raise ValueError(msg)

    try:
        table = pd.read_csv(manifest)
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
    return manifest, table.rename(columns=rename).copy()


def _lookup_mapping(table, mapping_field: str, class_field: str) -> dict:
    """Return validated numeric category-to-class mappings."""
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
        key = _required_number(key_value, key_field, row_number)
        target = _required_integer(class_value, value_field, row_number)
        if not 0 < target <= int32_max:
            msg = (
                f"Lookup row {row_number} has invalid {class_field} "
                f"{class_value!r}; expected a positive int32 value."
            )
            raise ValueError(msg)
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


def _category_key(value: object, field: str, row: int):
    """Return the category normalization used by the pymhm lookup UI."""
    missing = pd.isna(value)
    if isinstance(missing, (bool, np.bool_)) and missing:
        msg = f"{field} contains a missing value at row {row}."
        raise ValueError(msg)

    text = str(value).strip()
    if not text:
        msg = f"{field} contains an empty value at row {row}."
        raise ValueError(msg)
    try:
        number = float(text)
    except ValueError:
        return text
    if not np.isfinite(number):
        msg = f"{field} contains a non-finite value at row {row}."
        raise ValueError(msg)
    return str(int(number)) if number.is_integer() else text


def _lookup_target(value: object, field: str, row: int) -> int:
    """Validate and return one finite integral int32 target value."""
    target = _required_integer(value, field, row)
    int32 = np.iinfo(np.int32)
    if not int32.min <= target <= int32.max:
        msg = f"{field} must contain finite integral int32 targets; found {value!r}."
        raise ValueError(msg)
    if target == int(NO_DATA):
        msg = f"{field} contains the reserved nodata value {target}."
        raise ValueError(msg)
    return target


def _read_lookup_mapping(
    lookup_table: Path, mapping_field: str, burn_field: str
) -> dict:
    """Read normalized category-to-raster mappings from an OGR table."""
    table = read_lookup_table(lookup_table)

    key_field = _resolve_field(table.columns, mapping_field, "lookup table")
    value_field = _resolve_field(table.columns, burn_field, "lookup table")
    mapping = {}
    for row, (key_value, target_value) in enumerate(
        zip(table[key_field], table[value_field]), start=1
    ):
        if _is_blank(key_value):
            continue
        key = _category_key(key_value, str(key_field), row)
        target = _lookup_target(target_value, str(value_field), row)
        previous = mapping.get(key)
        if previous is not None and previous != target:
            msg = (
                f"Lookup key {key_value!r} maps to conflicting target values "
                f"{previous} and {target}."
            )
            raise ValueError(msg)
        mapping[key] = target
    if not mapping:
        msg = f"Lookup table contains no usable values in {str(key_field)!r}."
        raise ValueError(msg)
    return mapping


def _map_vector_values(frame, mapping_field, lookup):
    """Map every vector feature to its validated lookup target."""
    targets = []
    for row, value in enumerate(frame[mapping_field], start=1):
        key = _category_key(value, str(mapping_field), row)
        if key not in lookup:
            msg = (
                f"Vector value {value!r} in field {mapping_field!r} at feature "
                f"{row} is not present in the lookup table."
            )
            raise ValueError(msg)
        targets.append(lookup[key])
    return np.asarray(targets, dtype=np.int32)
