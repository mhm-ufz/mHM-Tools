"""Prepare an mHM geology-class raster from categorical raster data."""

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


def _normalise_field_name(field_name: object) -> str:
    """Return the field-name normalization used by pymhm lookup tables."""
    field_text = str(field_name).strip().lstrip("*").strip()
    if "[" in field_text:
        field_text = field_text.split("[", 1)[0].strip()
    return "".join(char.lower() for char in field_text if char.isalnum())


def _resolve_field(columns, requested: str):
    """Resolve an exact or normalized lookup-table field name."""
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


def _read_lookup_table(lookup_table: Path):
    """Read an OGR-compatible lookup table without geometry."""
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


def _lookup_mapping(table, lookup_table: Path, mapping_field: str) -> dict:
    """Return numeric-category to geology-class mappings."""
    key_field = _resolve_field(table.columns, mapping_field)
    class_field = _resolve_field(table.columns, "GEOLOGY_CLASS")
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
        key = _finite_number(key_value, key_field, row_number)
        class_number = _finite_number(class_value, class_field, row_number)
        if not class_number.is_integer() or not 0 < class_number <= int32_max:
            msg = (
                f"Lookup row {row_number} has invalid GEOLOGY_CLASS "
                f"{class_value!r}; expected a positive int32 value."
            )
            raise ValueError(msg)
        geology_class = int(class_number)
        previous = mapping.get(key)
        if previous is not None and previous != geology_class:
            msg = (
                f"Lookup key {key_value!r} maps to conflicting GEOLOGY_CLASS "
                f"values {previous} and {geology_class}."
            )
            raise ValueError(msg)
        mapping[key] = geology_class

    if not mapping:
        msg = f"Lookup table {lookup_table} contains no usable category mappings."
        raise ValueError(msg)
    return mapping


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


def _classdefinition_rows(table) -> list:
    """Validate and sort geology class-definition rows."""
    field_lookup = {}
    for field_name in table.columns:
        field_lookup.setdefault(_normalise_field_name(field_name), field_name)
    geology_class_field = _required_field(field_lookup, "GEOLOGY_CLASS")
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


def _classdefinition_text(table) -> str:
    """Render the geology class-definition text used by pymhm."""
    rows = _classdefinition_rows(table)
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
    lookup_table: PathLike, output_file: PathLike
) -> Path:
    """Write ``geology_classdefinition.txt`` from a geology lookup table."""
    lookup_table = Path(lookup_table)
    output_file = Path(output_file)
    if not lookup_table.is_file():
        msg = f"Lookup table does not exist: {lookup_table}"
        raise ValueError(msg)
    table = _read_lookup_table(lookup_table)
    return _write_classdefinition_text(_classdefinition_text(table), output_file)


def _nodata_values(data: xr.DataArray):
    """Yield finite nodata values recorded on an input raster."""
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
    """Format a short category sample for warnings."""
    values = []
    for value in categories[:limit]:
        number = float(value)
        values.append(str(int(number)) if number.is_integer() else str(number))
    suffix = ", ..." if categories.size > limit else ""
    return ", ".join(values) + suffix


def _reclassify(data: xr.DataArray, mapping: dict) -> np.ndarray:
    """Apply a geology lookup while retaining input nodata positions."""
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
    for key, geology_class in mapping.items():
        matches = valid & (values == key)
        if np.any(matches):
            output[matches] = geology_class
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


def format_geology_data(
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
    """Map a categorical raster and write its mHM geology definition."""
    input_file = Path(input_file)
    dem_file = Path(dem_file)
    output_path = Path(output_path)
    lookup_table = Path(lookup_table)
    output_type = str(output_type).lower().lstrip(".")
    raster_output = output_path / f"geology_class.{output_type}"
    definition_output = output_path / "geology_classdefinition.txt"

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
    if raster_output.resolve() in protected_inputs:
        msg = f"Raster output must differ from all input files: {raster_output}"
        raise ValueError(msg)
    if definition_output.resolve() in protected_inputs:
        msg = (
            "Classdefinition output must differ from all input files: "
            f"{definition_output}"
        )
        raise ValueError(msg)

    table = _read_lookup_table(lookup_table)
    mapping = _lookup_mapping(table, lookup_table, mapping_field)
    definition_text = _classdefinition_text(table)

    source = get_raster_data(input_file, crs=input_crs)
    try:
        reference = get_raster_data(dem_file, crs=dem_crs)
        try:
            aligned = align_raster_to_reference(source, reference, nodata=int(_NODATA))
            geology_classes = _reclassify(aligned, mapping)
            attrs = {
                "long_name": "mHM geology class",
                "nodata_value": int(_NODATA),
            }
            reference_dataset = reference.to_dataset(name="_dem")
            output_dataset = set_grid(
                geology_classes,
                get_grid(reference_dataset, "_dem"),
                "geology_class",
                data_attrs=attrs,
            )
            encoding = {
                "geology_class": {
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
                var_name="geology_class",
                encoding=encoding if output_type == "nc" else None,
                crs=reference.rio.crs,
            )
        finally:
            reference.close()
    finally:
        source.close()

    if not raster_output.is_file():
        msg = f"Geology-class raster was not created: {raster_output}"
        raise RuntimeError(msg)
    _write_classdefinition_text(definition_text, definition_output)
    logger.info("Wrote formatted geology data to %s", raster_output)
    return raster_output
