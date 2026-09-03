"""
Prepare an mHM geology-class raster from categorical raster data.

The lookup table drives both outputs: it maps the input raster categories onto
geology classes, and it supplies the karstic flag and ordering that the
companion ``geology_classdefinition.txt`` records. Columns whose name starts
with ``*`` are treated as comments and ignored throughout.

Authors
-------
- Sanjeev Bashyal
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Union

import numpy as np

from mhm_tools.common.format_data import (
    format_categorical_data,
    get_categorical_output_path,
)
from mhm_tools.common.lookup_handler import (
    _is_blank,
    _normalise_field_name,
    _required_integer,
    read_lookup_table,
)

logger = logging.getLogger(__name__)

PathLike = Union[str, Path]


# Lookup table fields


def _required_field(field_lookup: dict, field_name: str, *aliases: str):
    """Return a required normalized geology field."""
    for name in (field_name, *aliases):
        field = field_lookup.get(_normalise_field_name(name))
        if field is not None:
            return field
    expected = ", ".join(repr(name) for name in (field_name, *aliases))
    msg = f"Geology lookup table is missing required field {expected}."
    raise ValueError(msg)


def _optional_field(field_lookup: dict, *field_names: str):
    """Return the first of several optional fields, or None when absent."""
    for name in field_names:
        field = field_lookup.get(_normalise_field_name(name))
        if field is not None:
            return field
    return None


def _visible_table(table):
    """Drop lookup columns explicitly marked as ignored with ``*``."""
    return table[
        [column for column in table.columns if not str(column).strip().startswith("*")]
    ]


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


# Class definition output


_CLASSDEFINITION_HEADER = "GeoParam(i)   ClassUnit     Karstic      Description\n"
_CLASSDEFINITION_FOOTER = (
    "!<-END\n"
    "\n"
    "\n"
    "!***********************************\n"
    "! NOTES\n"
    "!***********************************\n"
    "1 = Karstic\n"
    "0 = Non-karstic\n"
    "\n"
    "IMPORTANT ::\n"
    "   Ordering has to be according to the ordering in mhm_parameter.nml\n"
    "   (namelist: geoparameter)\n"
)


def _classdefinition_rows(table, class_field: str) -> list:
    """Validate and sort geology class-definition rows."""
    field_lookup = {}
    for field_name in table.columns:
        if str(field_name).strip().startswith("*"):
            continue
        field_lookup.setdefault(_normalise_field_name(field_name), field_name)
    geology_class_field = _required_field(field_lookup, class_field)
    geo_class_field = _optional_field(field_lookup, "GEO_CLASS", "GEO_ID")
    geo_class_field = geo_class_field or geology_class_field
    karstic_field = _required_field(field_lookup, "KARSTIC")

    rows = []
    for row_number, (_, row) in enumerate(table.iterrows(), start=2):
        rows.append(
            {
                "geo_param": _required_integer(
                    row[geo_class_field], geo_class_field, row_number
                ),
                "class_unit": _required_integer(
                    row[geology_class_field], geology_class_field, row_number
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

    lines = [f"nGeo_Formations  {len(rows)}\n", _CLASSDEFINITION_HEADER]
    for row in rows:
        lines.append(
            f"{row['geo_param']:10d}\t"
            f"{row['class_unit']:10d}     "
            f"{row['karstic']:10d}      "
            f"GeoUnit-{row['class_unit']}\n"
        )
    lines.append(_CLASSDEFINITION_FOOTER)
    return "".join(lines)


def _write_classdefinition_text(text: str, output_file: Path) -> Path:
    """Write prevalidated geology class-definition text."""
    output_file.parent.mkdir(parents=True, exist_ok=True)
    output_file.write_text(text, encoding="utf-8")
    logger.info("Wrote geology class definition to %s", output_file)
    return output_file


# Public entry points


def write_geology_classdefinition(
    lookup_table: PathLike, output_file: PathLike, class_field: str
) -> Path:
    """Write ``geology_classdefinition.txt`` from a geology lookup table."""
    lookup_table = Path(lookup_table)
    output_file = Path(output_file)
    if not lookup_table.is_file():
        msg = f"Lookup table does not exist: {lookup_table}"
        raise ValueError(msg)
    table = _visible_table(read_lookup_table(lookup_table))
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
    resampling="nearest",
    fill_nodata: bool = True,
) -> Path:
    """Map a categorical raster and write its mHM geology definition.

    Parameters
    ----------
    fill_nodata : bool, default True
        Restrict the output to the DEM domain and take its remaining nodata
        cells from the nearest classified neighbour.
    """
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
    for label, path in (
        ("Raster output", raster_output),
        ("Classdefinition output", definition_output),
    ):
        if path.resolve() in protected_inputs:
            msg = f"{label} must differ from all input files: {path}"
            raise ValueError(msg)

    table = _visible_table(read_lookup_table(lookup_table))
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
        resampling=resampling,
        fill_nodata=fill_nodata,
    )
    _write_classdefinition_text(definition_text, definition_output)
    return raster_output
