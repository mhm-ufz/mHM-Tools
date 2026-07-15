"""Prepare an mHM soil-class raster from categorical raster data."""

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
_RASTER_SUFFIXES = {".asc", ".nc", ".tif", ".tiff"}
_OUTPUT_TYPES = {"asc", "nc"}
_NODATA = np.int32(NO_DATA)
_CLASSDEFINITION_HEADER = (
    "SOIL_NR\tHORIZON\tUD[mm]\tLD[mm]\tClay[%]\tSAND[%]\tBd[gcm-3]\tSilt[%]\n"
)


def _normalise_field_name(field_name: object) -> str:
    """Return the field-name normalization used by pymhm lookup tables."""
    field_text = str(field_name).strip().lstrip("*").strip()
    if "[" in field_text:
        field_text = field_text.split("[", 1)[0].strip()
    return "".join(char.lower() for char in field_text if char.isalnum())


def _resolve_field(columns, requested: str) -> str:
    """Resolve a requested field using normalized, case-insensitive matching."""
    requested_normalised = _normalise_field_name(requested)
    if not requested_normalised:
        msg = "The lookup mapping field must not be empty."
        raise ValueError(msg)

    exact_matches = [column for column in columns if str(column) == requested]
    if exact_matches:
        return exact_matches[0]

    matches = [
        column
        for column in columns
        if _normalise_field_name(column) == requested_normalised
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


def _finite_number(value: object, field: str, row_number: int) -> float:
    """Convert a lookup value to a finite number with a readable error."""
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        msg = f"Lookup row {row_number} has a non-numeric {field!r} value: {value!r}."
        raise ValueError(msg) from exc
    if not np.isfinite(number):
        msg = f"Lookup row {row_number} has a non-finite {field!r} value: {value!r}."
        raise ValueError(msg)
    return number


def _read_lookup_table(lookup_table: Path):
    """Read an OGR-compatible lookup table without its geometry."""
    try:
        table = gpd.read_file(lookup_table, ignore_geometry=True)
    except Exception as exc:
        msg = f"Could not read lookup table {lookup_table}: {exc}"
        raise ValueError(msg) from exc

    if table.empty:
        msg = f"Lookup table {lookup_table} is empty."
        raise ValueError(msg)

    return table


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


def _lookup_mapping(table, lookup_table: Path, mapping_field: str) -> dict:
    """Return numeric-category to soil-class mappings from a lookup table."""
    key_field = _resolve_field(table.columns, mapping_field)
    class_field = _resolve_field(table.columns, "SOIL_CLASS")
    mapping = {}
    int32_max = np.iinfo(np.int32).max

    for row_number, (key_value, class_value) in enumerate(
        zip(table[key_field], table[class_field]), start=1
    ):
        if _is_blank(key_value):
            logger.warning(
                "Skipping lookup row %d because %s is empty.",
                row_number,
                key_field,
            )
            continue
        key = _finite_number(key_value, str(key_field), row_number)
        class_number = _finite_number(class_value, str(class_field), row_number)
        if not class_number.is_integer() or not 0 < class_number <= int32_max:
            msg = (
                f"Lookup row {row_number} has invalid SOIL_CLASS "
                f"{class_value!r}; expected a positive int32 value."
            )
            raise ValueError(msg)
        soil_class = int(class_number)
        previous = mapping.get(key)
        if previous is not None and previous != soil_class:
            msg = (
                f"Lookup key {key_value!r} maps to conflicting SOIL_CLASS "
                f"values {previous} and {soil_class}."
            )
            raise ValueError(msg)
        mapping[key] = soil_class

    if not mapping:
        msg = f"Lookup table {lookup_table} contains no usable category mappings."
        raise ValueError(msg)
    logger.info(
        "Loaded %d soil-category mappings from %s (%s -> %s).",
        len(mapping),
        lookup_table,
        key_field,
        class_field,
    )
    return mapping


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
    table = _read_lookup_table(lookup_table)
    text = _soil_classdefinition_text(table)
    return _write_soil_classdefinition_text(text, output_file)


def _nodata_values(data: xr.DataArray):
    """Yield finite nodata sentinels recorded on an input raster variable."""
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
    """Format a short category sample for warning messages."""
    values = []
    for value in categories[:limit]:
        number = float(value)
        values.append(str(int(number)) if number.is_integer() else str(number))
    suffix = ", ..." if categories.size > limit else ""
    return ", ".join(values) + suffix


def _reclassify(data: xr.DataArray, mapping: dict) -> np.ndarray:
    """Apply a soil lookup while retaining input nodata positions."""
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
    for key, soil_class in mapping.items():
        matches = valid & (values == key)
        if np.any(matches):
            output[matches] = soil_class
            mapped |= matches

    mapped_count = int(np.count_nonzero(mapped))
    if mapped_count == 0:
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
        Directory in which ``soil_class.nc`` or ``soil_class.asc`` is written.
    lookup_table : path-like
        OGR-readable table containing ``mapping_field`` and ``SOIL_CLASS``.
    mapping_field : str
        Numeric lookup-table column corresponding to the input raster values.
    output_type : {"nc", "asc"}, default "nc"
        Default output raster format. The CLI restricts this to NetCDF or
        ASCII.
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
    output_type = str(output_type).lower().lstrip(".")
    raster_output = output_path / f"soil_class.{output_type}"
    definition_output = output_path / "soil_classdefinition.txt"

    if not input_file.is_file():
        msg = f"Input raster does not exist: {input_file}"
        raise ValueError(msg)
    if input_file.suffix.lower() not in _RASTER_SUFFIXES:
        msg = (
            "Input must be a raster with one of these suffixes: "
            f"{', '.join(sorted(_RASTER_SUFFIXES))}."
        )
        raise ValueError(msg)
    if not dem_file.is_file():
        msg = f"DEM raster does not exist: {dem_file}"
        raise ValueError(msg)
    if dem_file.suffix.lower() not in _RASTER_SUFFIXES:
        msg = (
            "DEM must be a raster with one of these suffixes: "
            f"{', '.join(sorted(_RASTER_SUFFIXES))}."
        )
        raise ValueError(msg)
    if not lookup_table.is_file():
        msg = f"Lookup table does not exist: {lookup_table}"
        raise ValueError(msg)
    if output_path.exists() and not output_path.is_dir():
        msg = f"Output path must be a directory: {output_path}"
        raise ValueError(msg)
    if output_type not in _OUTPUT_TYPES:
        msg = "output_type must be either 'nc' or 'asc'."
        raise ValueError(msg)

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
    if raster_output.resolve() == definition_output.resolve():
        msg = "Raster and classdefinition outputs must be different files."
        raise ValueError(msg)

    table = _read_lookup_table(lookup_table)
    mapping = _lookup_mapping(table, lookup_table, mapping_field)
    definition_text = _soil_classdefinition_text(table)

    source = get_raster_data(input_file, crs=input_crs)
    try:
        reference = get_raster_data(dem_file, crs=dem_crs)
        try:
            aligned = align_raster_to_reference(source, reference, nodata=int(_NODATA))
            soil_classes = _reclassify(aligned, mapping)
            attrs = {
                "long_name": "mHM soil class",
                "nodata_value": int(_NODATA),
            }
            reference_dataset = reference.to_dataset(name="_dem")
            output_dataset = set_grid(
                soil_classes,
                get_grid(reference_dataset, "_dem"),
                "soil_class",
                data_attrs=attrs,
            )
            encoding = {
                "soil_class": {
                    "zlib": True,
                    "complevel": 4,
                    "shuffle": True,
                    "_FillValue": int(_NODATA),
                    "dtype": "int32",
                }
            }
            write_xarray_to_file(
                output_dataset,
                raster_output,
                var_name="soil_class",
                encoding=encoding if output_type == "nc" else None,
                crs=reference.rio.crs,
            )
        finally:
            reference.close()
    finally:
        source.close()

    if not raster_output.is_file():
        msg = f"Soil-class raster was not created: {raster_output}"
        raise RuntimeError(msg)
    _write_soil_classdefinition_text(definition_text, definition_output)
    logger.info("Wrote formatted soil data to %s", raster_output)
    return raster_output
