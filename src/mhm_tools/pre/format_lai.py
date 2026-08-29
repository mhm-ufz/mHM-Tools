"""Prepare an mHM LAI-class raster and class definition."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Union

from mhm_tools.common.format_data import (
    format_categorical_data,
    get_categorical_output_path,
)
from mhm_tools.common.lookup_handler import (
    _is_blank,
    _required_integer,
    _required_number,
    _resolve_field,
    read_lookup_table,
)

logger = logging.getLogger(__name__)

PathLike = Union[str, Path]
_MONTHS = (
    "Jan",
    "Feb",
    "Mar",
    "Apr",
    "May",
    "Jun",
    "Jul",
    "Aug",
    "Sep",
    "Oct",
    "Nov",
    "Dec",
)
_HEADER = (
    "ID   LAND-USE                  Jan.   Feb.    Mar.    Apr.    May    "
    "Jun.    Jul.    Aug.    Sep.    Oct.    Nov.    Dec.\n"
)


def _classdefinition_rows(table, class_field: str) -> list:
    """Validate and sort the LAI definitions in a lookup table."""
    class_column = _resolve_field(table.columns, class_field)
    land_use_column = _resolve_field(table.columns, "LAND-USE")
    month_columns = [_resolve_field(table.columns, month) for month in _MONTHS]
    definitions = {}

    for row_number, (_, row) in enumerate(table.iterrows(), start=2):
        class_id = _required_integer(
            row[class_column], class_column, row_number, "LAI lookup"
        )
        if class_id <= 0:
            msg = f"LAI lookup row {row_number} has non-positive class {class_id}."
            raise ValueError(msg)
        if _is_blank(row[land_use_column]):
            msg = (
                f"LAI lookup row {row_number} has an empty "
                f"{land_use_column!r} value."
            )
            raise ValueError(msg)
        land_use = str(row[land_use_column]).strip()
        values = tuple(
            _required_number(row[column], column, row_number, "LAI lookup")
            for column in month_columns
        )
        if any(value < 0 for value in values):
            msg = f"LAI lookup row {row_number} contains a negative monthly value."
            raise ValueError(msg)

        definition = (land_use, values)
        previous = definitions.get(class_id)
        if previous is not None and previous != definition:
            msg = f"LAI class {class_id} has conflicting lookup definitions."
            raise ValueError(msg)
        definitions[class_id] = definition

    class_ids = sorted(definitions)
    if class_ids != list(range(1, len(class_ids) + 1)):
        msg = "LAI classes must be consecutive positive integers starting at 1."
        raise ValueError(msg)
    return [
        {
            "class_id": class_id,
            "land_use": definitions[class_id][0],
            "values": definitions[class_id][1],
        }
        for class_id in class_ids
    ]


def _format_lai_value(value: float) -> str:
    return f"{value:.6g}"


def _classdefinition_text(table, class_field: str) -> str:
    """Render an mHM ``LAI_classdefinition.txt`` lookup."""
    rows = _classdefinition_rows(table, class_field)
    if not rows:
        msg = "No valid LAI classdefinition rows were found."
        raise ValueError(msg)

    lines = [f"NoLAIclasses           {len(rows)}\n", _HEADER]
    for row in rows:
        values = "".join(f"{_format_lai_value(value):<8}" for value in row["values"])
        lines.append(
            f"{row['class_id']:2d}    {row['land_use']:<25} {values.rstrip()}\n"
        )
    return "".join(lines)


def _write_classdefinition_text(text: str, output_file: Path) -> Path:
    output_file.parent.mkdir(parents=True, exist_ok=True)
    output_file.write_text(text, encoding="utf-8")
    logger.info("Wrote LAI class definition to %s", output_file)
    return output_file


def write_lai_classdefinition(
    lookup_table: PathLike, output_file: PathLike, class_field: str
) -> Path:
    """Write ``LAI_classdefinition.txt`` from an LAI lookup table."""
    lookup_table = Path(lookup_table)
    output_file = Path(output_file)
    table = read_lookup_table(lookup_table)
    return _write_classdefinition_text(
        _classdefinition_text(table, class_field), output_file
    )


def format_lai_data(
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
    """Map a categorical raster and write its mHM LAI definition."""
    input_file = Path(input_file)
    dem_file = Path(dem_file)
    output_path = Path(output_path)
    lookup_table = Path(lookup_table)
    raster_output = get_categorical_output_path(output_path, "lai_class", output_type)
    definition_output = output_path / "LAI_classdefinition.txt"

    protected_inputs = {
        input_file.resolve(),
        dem_file.resolve(),
        lookup_table.resolve(),
    }
    for output in (raster_output, definition_output):
        if output.resolve() in protected_inputs:
            msg = f"Output must differ from all input files: {output}"
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
        variable_name="lai_class",
        input_crs=input_crs,
        dem_crs=dem_crs,
        resampling=resampling,
        fill_nodata=fill_nodata,
    )
    _write_classdefinition_text(definition_text, definition_output)
    return raster_output
