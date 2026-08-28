"""Lookup-table field resolution and value mapping."""

from pathlib import Path

import numpy as np
import pandas as pd

from mhm_tools.common.constants import NO_DATA


def _normalise_field_name(field_name: object) -> str:
    """Normalize a vector or lookup-table field name for matching."""
    field_text = str(field_name).strip().lstrip("*").strip()
    if "[" in field_text:
        field_text = field_text.split("[", 1)[0].strip()
    return "".join(char.lower() for char in field_text if char.isalnum())


def _resolve_field(columns, requested: str, source: str):
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


def _blank_category(value: object) -> bool:
    """Return whether a lookup category is empty."""
    missing = pd.isna(value)
    if isinstance(missing, (bool, np.bool_)) and missing:
        return True
    return isinstance(value, str) and not value.strip()


def _soil_class(value: object, field: str, row: int) -> int:
    """Validate and return one finite integral int32 target value."""
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        msg = f"{field} contains a non-numeric target {value!r} at row {row}."
        raise ValueError(msg) from exc
    int32 = np.iinfo(np.int32)
    if (
        not np.isfinite(number)
        or not number.is_integer()
        or not int32.min <= number <= int32.max
    ):
        msg = f"{field} must contain finite integral int32 targets; found {value!r}."
        raise ValueError(msg)
    target = int(number)
    if target == int(NO_DATA):
        msg = f"{field} contains the reserved nodata value {target}."
        raise ValueError(msg)
    return target


def _read_lookup_mapping(
    lookup_table: Path, mapping_field: str, burn_field: str
) -> dict:
    """Read normalized category-to-raster mappings from an OGR table."""
    import geopandas as gpd

    try:
        table = gpd.read_file(lookup_table, ignore_geometry=True)
    except Exception as exc:
        msg = f"Could not read lookup table {lookup_table}: {exc}"
        raise ValueError(msg) from exc
    if table.empty:
        msg = f"Lookup table is empty: {lookup_table}"
        raise ValueError(msg)

    key_field = _resolve_field(table.columns, mapping_field, "lookup table")
    value_field = _resolve_field(table.columns, burn_field, "lookup table")
    mapping = {}
    for row, (key_value, target_value) in enumerate(
        zip(table[key_field], table[value_field]), start=1
    ):
        if _blank_category(key_value):
            continue
        key = _category_key(key_value, str(key_field), row)
        target = _soil_class(target_value, str(value_field), row)
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
