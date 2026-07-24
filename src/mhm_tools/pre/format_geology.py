"""
Prepare an mHM geology-class raster from categorical raster data.

Authors
-------
- Sanjeev Bashyal
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Union

import numpy as np
import pandas as pd

from mhm_tools.common.format_data import (
    format_categorical_data,
    get_categorical_output_path,
    read_lookup_table,
)

logger = logging.getLogger(__name__)

PathLike = Union[str, Path]


def _normalise_field_name(field_name: object) -> str:
    """Return the field-name normalization used by pymhm lookup tables."""
    field_text = str(field_name).strip().lstrip("*").strip()
    if "[" in field_text:
        field_text = field_text.split("[", 1)[0].strip()
    return "".join(char.lower() for char in field_text if char.isalnum())


def _is_blank(value: object) -> bool:
    """Return whether a lookup value is empty."""
    if value is None:
        return True
    if isinstance(value, str):
        return not value.strip()
    try:
        missing = pd.isna(value)
    except (TypeError, ValueError):
        return False
    return bool(missing) if np.isscalar(missing) else False


def _finite_number(value: object, field: object, row_number: int) -> float:
    """Convert a lookup value to a finite number."""
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        msg = f"Lookup row {row_number} has a non-numeric {field!r} value: {value!r}."
        raise ValueError(msg) from exc
    if not np.isfinite(number):
        msg = f"Lookup row {row_number} has a non-finite {field!r} value: {value!r}."
        raise ValueError(msg)
    return number


def _required_field(field_lookup: dict, field_name: str):
    """Return a required normalized geology field."""
    field = field_lookup.get(_normalise_field_name(field_name))
    if field is None:
        msg = f"Geology lookup table is missing required field {field_name!r}."
        raise ValueError(msg)
    return field


def _required_int(value: object, row_number: int, field: object) -> int:
    """Read one required finite integer lookup value."""
    if _is_blank(value):
        msg = f"Row {row_number} has an empty value for required field {field!r}."
        raise ValueError(msg)
    number = _finite_number(value, field, row_number)
    if not number.is_integer():
        msg = (
            f"Row {row_number} has non-integer value {value!r} "
            f"for required field {field!r}."
        )
        raise ValueError(msg)
    return int(number)


def _required_bool_int(value: object, row_number: int, field: object) -> int:
    """Read a required boolean lookup value as zero or one."""
    if _is_blank(value):
        msg = f"Row {row_number} has an empty value for required field {field!r}."
        raise ValueError(msg)
    if isinstance(value, (bool, np.bool_)):
        return int(value)

    value_text = str(value).strip().lower()
    if value_text in {"1", "true", "t", "yes", "y"}:
        return 1
    if value_text in {"0", "false", "f", "no", "n"}:
        return 0
    msg = (
        f"Row {row_number} has invalid boolean value {value!r} "
        f"for required field {field!r}."
    )
    raise ValueError(msg)


def _classdefinition_rows(table, class_field: str) -> list:
    """Validate and sort geology class-definition rows."""
    field_lookup = {}
    for field_name in table.columns:
        field_lookup.setdefault(_normalise_field_name(field_name), field_name)
    geology_class_field = _required_field(field_lookup, class_field)
    geo_class_field = _required_field(field_lookup, "GEO_CLASS")
    karstic_field = _required_field(field_lookup, "KARSTIC")

    rows = []
    for row_number, (_, row) in enumerate(table.iterrows(), start=2):
        rows.append(
            {
                "geo_param": _required_int(
                    row[geo_class_field], row_number, geo_class_field
                ),
                "class_unit": _required_int(
                    row[geology_class_field], row_number, geology_class_field
                ),
                "karstic": _required_bool_int(
                    row[karstic_field], row_number, karstic_field
                ),
            }
        )
    return sorted(rows, key=lambda item: (item["geo_param"], item["class_unit"]))


def _classdefinition_text(table, class_field: str) -> str:
    """Render the geology class-definition text used by pymhm."""
    rows = _classdefinition_rows(table, class_field)
    if not rows:
        msg = "No valid geology classdefinition rows were found."
        raise ValueError(msg)

    lines = [
        f"nGeo_Formations  {len(rows)}\n",
        "GeoParam(i)   ClassUnit     Karstic      Description\n",
    ]
    for row in rows:
        lines.append(
            f"{row['geo_param']:10d}\t"
            f"{row['class_unit']:10d}     "
            f"{row['karstic']:10d}      "
            f"GeoUnit-{row['class_unit']}\n"
        )
    lines.extend(
        [
            "!<-END\n",
            "\n",
            "\n",
            "!***********************************\n",
            "! NOTES\n",
            "!***********************************\n",
            "1 = Karstic\n",
            "0 = Non-karstic\n",
            "\n",
            "IMPORTANT ::\n",
            "   Ordering has to be according to the ordering in mhm_parameter.nml\n",
            "   (namelist: geoparameter)\n",
        ]
    )
    return "".join(lines)


def _write_classdefinition_text(text: str, output_file: Path) -> Path:
    """Write prevalidated geology class-definition text."""
    output_file.parent.mkdir(parents=True, exist_ok=True)
    output_file.write_text(text, encoding="utf-8")
    logger.info("Wrote geology class definition to %s", output_file)
    return output_file


def write_geology_classdefinition(
    lookup_table: PathLike, output_file: PathLike, class_field: str
) -> Path:
    """Write ``geology_classdefinition.txt`` from a geology lookup table."""
    lookup_table = Path(lookup_table)
    output_file = Path(output_file)
    if not lookup_table.is_file():
        msg = f"Lookup table does not exist: {lookup_table}"
        raise ValueError(msg)
    table = read_lookup_table(lookup_table)
    return _write_classdefinition_text(
        _classdefinition_text(table, class_field), output_file
    )


def format_geology_data(
    input_file: PathLike,
    dem_file: PathLike,
    output_path: PathLike,
    lookup_table: PathLike,
    mapping_field: str,
    class_field: str,
    output_type: str = "nc",
    *,
    input_crs: str | None = None,
    dem_crs: str | None = None,
) -> Path:
    """Map a categorical raster and write its mHM geology definition."""
    input_file = Path(input_file)
    dem_file = Path(dem_file)
    output_path = Path(output_path)
    lookup_table = Path(lookup_table)
    raster_output = get_categorical_output_path(
        output_path, "geology_class", output_type
    )
    definition_output = output_path / "geology_classdefinition.txt"

    protected_inputs = {
        input_file.resolve(),
        dem_file.resolve(),
        lookup_table.resolve(),
    }
    if raster_output.resolve() in protected_inputs:
        msg = f"Raster output must differ from all input files: {raster_output}"
        raise ValueError(msg)
    if definition_output.resolve() in protected_inputs:
        msg = (
            "Classdefinition output must differ from all input files: "
            f"{definition_output}"
        )
        raise ValueError(msg)

    table = read_lookup_table(lookup_table)
    definition_text = _classdefinition_text(table, class_field)
    format_categorical_data(
        input_file,
        dem_file,
        raster_output,
        table,
        mapping_field,
        class_field,
        variable_name="geology_class",
        input_crs=input_crs,
        dem_crs=dem_crs,
    )
    _write_classdefinition_text(definition_text, definition_output)
    return raster_output
