"""Prepare an mHM soil-class raster from categorical raster data."""

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
_CLASSDEFINITION_HEADER = (
    "SOIL_NR\tHORIZON\tUD[mm]\tLD[mm]\tClay[%]\tSAND[%]\tBd[gcm-3]\tSilt[%]\n"
)


def _normalise_field_name(field_name: object) -> str:
    """Return the field-name normalization used by pymhm lookup tables."""
    field_text = str(field_name).strip().lstrip("*").strip()
    if "[" in field_text:
        field_text = field_text.split("[", 1)[0].strip()
    return "".join(char.lower() for char in field_text if char.isalnum())


def _is_blank(value: object) -> bool:
    """Return whether a lookup-table value is empty."""
    if value is None:
        return True
    if isinstance(value, str):
        return not value.strip()
    try:
        missing = pd.isna(value)
    except (TypeError, ValueError):
        return False
    return bool(missing) if np.isscalar(missing) else False


def _required_float(value: object, row_number: int, field_name: object) -> float:
    """Read a required finite numeric value from a lookup row."""
    if _is_blank(value):
        msg = (
            f"Row {row_number} has an empty value for required field "
            f"{str(field_name)!r}."
        )
        raise ValueError(msg)
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        msg = (
            f"Row {row_number} has invalid numeric value {value!r} "
            f"for required field {str(field_name)!r}."
        )
        raise ValueError(msg) from exc
    if not np.isfinite(number):
        msg = (
            f"Row {row_number} has invalid numeric value {value!r} "
            f"for required field {str(field_name)!r}."
        )
        raise ValueError(msg)
    return number


def _required_int(value: object, row_number: int, field_name: object) -> int:
    """Read a required integer value from a lookup row."""
    number = _required_float(value, row_number, field_name)
    if not number.is_integer():
        msg = (
            f"Row {row_number} has non-integer value {value!r} "
            f"for required field {str(field_name)!r}."
        )
        raise ValueError(msg)
    return int(number)


def _required_soil_field(field_lookup: dict, field_name: str):
    """Resolve a required normalized soil-definition field."""
    field = field_lookup.get(_normalise_field_name(field_name))
    if field is None:
        msg = f"Soil lookup table is missing required field {field_name!r}."
        raise ValueError(msg)
    return field


def _rowwise_soil_rows(table, soil_class_field, fields: dict) -> list:
    """Parse a lookup table containing one row per horizon."""
    output_rows = []
    for row_number, (_, row) in enumerate(table.iterrows(), start=2):
        output_rows.append(
            {
                "soil_class": _required_int(
                    row[soil_class_field], row_number, soil_class_field
                ),
                "horizon": _required_int(
                    row[fields["horizon"]], row_number, fields["horizon"]
                ),
                "upper_depth": _required_float(
                    row[fields["upper_depth"]], row_number, fields["upper_depth"]
                ),
                "lower_depth": _required_float(
                    row[fields["lower_depth"]], row_number, fields["lower_depth"]
                ),
                "clay": _required_float(
                    row[fields["clay"]], row_number, fields["clay"]
                ),
                "sand": _required_float(
                    row[fields["sand"]], row_number, fields["sand"]
                ),
                "silt": _required_float(
                    row[fields["silt"]], row_number, fields["silt"]
                ),
                "bulk_density": _required_float(
                    row[fields["bulk_density"]],
                    row_number,
                    fields["bulk_density"],
                ),
            }
        )
    return sorted(output_rows, key=lambda item: (item["soil_class"], item["horizon"]))


def _wide_soil_rows(
    table, field_lookup: dict, soil_class_field, horizons_field
) -> list:
    """Parse a lookup table containing suffixed fields for each horizon."""
    output_rows = []
    for row_number, (_, row) in enumerate(table.iterrows(), start=2):
        soil_class = _required_int(row[soil_class_field], row_number, soil_class_field)
        horizons = _required_int(row[horizons_field], row_number, horizons_field)
        upper_depth = 0.0

        for horizon in range(1, horizons + 1):
            depth_field = _required_soil_field(field_lookup, f"DEPTH{horizon}")
            bulk_density_field = _required_soil_field(
                field_lookup, f"BULK_DENSITY{horizon}"
            )
            clay_field = _required_soil_field(field_lookup, f"CLAY{horizon}")
            silt_field = _required_soil_field(field_lookup, f"SILT{horizon}")
            sand_field = _required_soil_field(field_lookup, f"SAND{horizon}")
            lower_depth = _required_float(row[depth_field], row_number, depth_field)
            output_rows.append(
                {
                    "soil_class": soil_class,
                    "horizon": horizon,
                    "upper_depth": upper_depth,
                    "lower_depth": lower_depth,
                    "clay": _required_float(row[clay_field], row_number, clay_field),
                    "sand": _required_float(row[sand_field], row_number, sand_field),
                    "silt": _required_float(row[silt_field], row_number, silt_field),
                    "bulk_density": _required_float(
                        row[bulk_density_field], row_number, bulk_density_field
                    ),
                }
            )
            upper_depth = lower_depth

    return sorted(output_rows, key=lambda item: (item["soil_class"], item["horizon"]))


def _soil_classdefinition_rows(table) -> list:
    """Build definition rows from the rowwise or wide lookup layout."""
    field_lookup = {}
    for field_name in table.columns:
        field_lookup.setdefault(_normalise_field_name(field_name), field_name)
    soil_class_field = _required_soil_field(field_lookup, "SOIL_CLASS")

    rowwise_fields = (
        "HORIZON",
        "UPPER_DEPTH",
        "LOWER_DEPTH",
        "CLAY",
        "SAND",
        "SILT",
        "BULK_DENSITY",
    )
    if all(
        _normalise_field_name(field_name) in field_lookup
        for field_name in rowwise_fields
    ):
        fields = {
            field_name.lower(): _required_soil_field(field_lookup, field_name)
            for field_name in rowwise_fields
        }
        return _rowwise_soil_rows(table, soil_class_field, fields)

    horizons_field = field_lookup.get(_normalise_field_name("HORIZONS"))
    if horizons_field is not None:
        return _wide_soil_rows(table, field_lookup, soil_class_field, horizons_field)

    msg = (
        "Soil lookup table must use either the row-per-horizon layout "
        "(HORIZON, UPPER_DEPTH, LOWER_DEPTH, CLAY, SAND, SILT, BULK_DENSITY) "
        "or the wide layout "
        "(HORIZONS, DEPTH1, BULK_DENSITY1, CLAY1, SILT1, SAND1, ...)."
    )
    raise ValueError(msg)


def _format_soil_value(value: object) -> str:
    """Format a numeric value like the pymhm classdefinition writer."""
    if value is None:
        return ""
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    if number.is_integer():
        return str(int(number))
    return f"{number:.6g}"


def _soil_classdefinition_text(table) -> str:
    """Validate a lookup table and render its classdefinition text."""
    output_rows = _soil_classdefinition_rows(table)
    if not output_rows:
        msg = "No valid soil horizon rows were found."
        raise ValueError(msg)

    unique_soil_classes = {row["soil_class"] for row in output_rows}
    lines = [f"nSoil_Types {len(unique_soil_classes)}\n", _CLASSDEFINITION_HEADER]
    for row in output_rows:
        lines.append(
            f"{row['soil_class']}\t{row['horizon']}\t"
            f"{_format_soil_value(row['upper_depth'])}\t"
            f"{_format_soil_value(row['lower_depth'])}\t"
            f"{_format_soil_value(row['clay'])}\t"
            f"{_format_soil_value(row['sand'])}\t"
            f"{_format_soil_value(row['bulk_density'])}\t"
            f"{_format_soil_value(row['silt'])}\n"
        )
    return "".join(lines)


def _write_soil_classdefinition_text(text: str, output_file: Path) -> Path:
    """Write prevalidated classdefinition text."""
    output_file.parent.mkdir(parents=True, exist_ok=True)
    output_file.write_text(text, encoding="utf-8")
    logger.info("Wrote soil class definition to %s", output_file)
    return output_file


def write_soil_classdefinition(lookup_table: PathLike, output_file: PathLike) -> Path:
    """Write an mHM ``soil_classdefinition.txt`` from a lookup table.

    The lookup may contain either one row per soil horizon or one row per soil
    class with horizon-specific fields suffixed by ``1``, ``2``, and so on.
    """
    lookup_table = Path(lookup_table)
    output_file = Path(output_file)
    if not lookup_table.is_file():
        msg = f"Lookup table does not exist: {lookup_table}"
        raise ValueError(msg)
    table = read_lookup_table(lookup_table)
    text = _soil_classdefinition_text(table)
    return _write_soil_classdefinition_text(text, output_file)


def format_soil_data(
    input_file: PathLike,
    dem_file: PathLike,
    output_path: PathLike,
    lookup_table: PathLike,
    mapping_field: str,
    output_type: str = "nc",
    *,
    input_crs: str | None = None,
    dem_crs: str | None = None,
) -> Path:
    """Map a categorical raster and write its mHM soil definition.

    Parameters
    ----------
    input_file : path-like
        Single-variable, two-dimensional ASCII, NetCDF, or GeoTIFF raster.
    dem_file : path-like
        ASCII, NetCDF, or GeoTIFF DEM providing the exact output grid.
    output_path : path-like
        Directory containing the soil-class raster and classdefinition.
    lookup_table : path-like
        OGR-readable table containing ``mapping_field`` and ``SOIL_CLASS``.
    mapping_field : str
        Numeric lookup-table column corresponding to the input raster values.
    output_type : {"nc", "asc", "tif"}, default "nc"
        Output raster format.
    input_crs, dem_crs : str, optional
        CRS to assign only when the corresponding raster has no CRS metadata.

    Returns
    -------
    pathlib.Path
        Path to the created soil-class raster.
    """
    input_file = Path(input_file)
    dem_file = Path(dem_file)
    lookup_table = Path(lookup_table)
    output_path = Path(output_path)
    raster_output = get_categorical_output_path(output_path, "soil_class", output_type)
    definition_output = output_path / "soil_classdefinition.txt"

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
    table = read_lookup_table(lookup_table)
    definition_text = _soil_classdefinition_text(table)
    format_categorical_data(
        input_file,
        dem_file,
        raster_output,
        table,
        mapping_field,
        "SOIL_CLASS",
        input_crs=input_crs,
        dem_crs=dem_crs,
    )
    _write_soil_classdefinition_text(definition_text, definition_output)
    return raster_output
