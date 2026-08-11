"""
Prepare an mHM soil-class raster from categorical raster data.

Authors
-------
- Sanjeev Bashyal
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Union

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
from mhm_tools.common.format_data import (
    format_categorical_data,
    get_categorical_output_path,
    get_format_manifest_path,
    read_format_manifest,
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


def _soil_classdefinition_rows(table, class_field: str) -> list:
    """Build definition rows from the rowwise or wide lookup layout."""
    field_lookup = {}
    for field_name in table.columns:
        field_lookup.setdefault(_normalise_field_name(field_name), field_name)
    soil_class_field = _required_soil_field(field_lookup, class_field)

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


def _soil_classdefinition_text(table, class_field: str) -> str:
    """Validate a lookup table and render its classdefinition text."""
    output_rows = _soil_classdefinition_rows(table, class_field)
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


def write_soil_classdefinition(
    lookup_table: PathLike, output_file: PathLike, class_field: str
) -> Path:
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
    text = _soil_classdefinition_text(table, class_field)
    return _write_soil_classdefinition_text(text, output_file)


def format_soil_data(
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
        OGR-readable table containing ``mapping_field`` and ``class_field``.
    mapping_field : str
        Numeric lookup-table column corresponding to the input raster values.
    class_field : str
        Numeric lookup-table column containing the output soil classes.
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
    definition_text = _soil_classdefinition_text(table, class_field)
    format_categorical_data(
        input_file,
        dem_file,
        raster_output,
        table,
        mapping_field,
        class_field,
        variable_name="soil_class",
        input_crs=input_crs,
        dem_crs=dem_crs,
        resampling=resampling,
    )
    _write_soil_classdefinition_text(definition_text, definition_output)
    return raster_output


_SOIL_MANIFEST_COLUMNS = (
    "Horizon",
    "Upper Depth",
    "Lower Depth",
    "Clay Layer",
    "Sand Layer",
    "Silt Layer",
    "Bulk Density Layer",
)
_BULK_DENSITY_UNIT_PATTERN = re.compile(
    r"^\s*Bulk\s+Density\s+Unit\s*=\s*(.*?)\s*$",
    re.IGNORECASE,
)


def _bulk_density_unit(unit: str) -> tuple[str, float]:
    """Return a canonical bulk-density unit and conversion to g/cm3."""
    normalized = (
        str(unit)
        .strip()
        .lower()
        .replace("³", "3")
        .replace("⁻", "-")
        .replace("−", "-")
        .replace("^", "")
        .replace("·", "")
        .replace(" ", "")
    )
    aliases = {
        "g/cm3": ("g/cm3", 1.0),
        "gcm-3": ("g/cm3", 1.0),
        "gcm3": ("g/cm3", 1.0),
        "kg/m3": ("kg/m3", 1.0e-3),
        "kgm-3": ("kg/m3", 1.0e-3),
        "kgm3": ("kg/m3", 1.0e-3),
        "cg/cm3": ("cg/cm3", 1.0e-2),
        "cgcm-3": ("cg/cm3", 1.0e-2),
        "cgcm3": ("cg/cm3", 1.0e-2),
        "mg/cm3": ("mg/cm3", 1.0e-3),
        "mgcm-3": ("mg/cm3", 1.0e-3),
        "mgcm3": ("mg/cm3", 1.0e-3),
        "g/dm3": ("g/dm3", 1.0e-3),
        "gdm-3": ("g/dm3", 1.0e-3),
        "gdm3": ("g/dm3", 1.0e-3),
        "kg/dm3": ("kg/dm3", 1.0),
        "kgdm-3": ("kg/dm3", 1.0),
        "kgdm3": ("kg/dm3", 1.0),
    }
    try:
        return aliases[normalized]
    except KeyError as exc:
        supported = "g/cm3, kg/m3, cg/cm3, mg/cm3, g/dm3, or kg/dm3"
        msg = f"Unsupported bulk density unit {unit!r}; expected {supported}."
        raise ValueError(msg) from exc


def _manifest_number(value, column: str, row_number: int) -> float:
    """Return one finite numeric soil-manifest value."""
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        msg = f"Manifest row {row_number} has invalid {column} value {value!r}."
        raise ValueError(msg) from exc
    if not np.isfinite(number):
        msg = f"Manifest row {row_number} has invalid {column} value {value!r}."
        raise ValueError(msg)
    return number


def _soil_manifest_path(manifest: Path, value, column: str, row_number: int) -> Path:
    """Resolve one soil raster path relative to its manifest."""
    if pd.isna(value) or not str(value).strip():
        msg = f"Manifest row {row_number} has an empty {column}."
        raise ValueError(msg)
    path = Path(str(value).strip()).expanduser()
    if not path.is_absolute():
        path = manifest.parent / path
    path = path.resolve()
    if not path.is_file():
        msg = f"Manifest raster does not exist: {path}"
        raise ValueError(msg)
    return path


def _read_soil_manifest(input_path: PathLike):
    """Read bulk-density metadata and validated soil horizon records."""
    manifest = get_format_manifest_path(input_path)
    lines = manifest.read_text(encoding="utf-8-sig").splitlines()
    if not lines:
        msg = (
            f"Soil manifest {manifest} must begin with "
            "'Bulk Density Unit = <unit>'."
        )
        raise ValueError(msg)
    first_line = lines[0]
    match = _BULK_DENSITY_UNIT_PATTERN.match(first_line)
    if match is None or not match.group(1):
        msg = (
            f"Soil manifest {manifest} must begin with "
            "'Bulk Density Unit = <unit>'."
        )
        raise ValueError(msg)
    unit, factor = _bulk_density_unit(match.group(1))
    _, table = read_format_manifest(
        input_path,
        _SOIL_MANIFEST_COLUMNS,
        skiprows=1,
    )

    horizons = []
    for row_number, (_, row) in enumerate(table.iterrows(), start=3):
        horizon_number = _manifest_number(row["Horizon"], "Horizon", row_number)
        if not horizon_number.is_integer() or horizon_number < 1:
            msg = (
                f"Manifest row {row_number} has invalid Horizon value "
                f"{row['Horizon']!r}."
            )
            raise ValueError(msg)
        upper = _manifest_number(row["Upper Depth"], "Upper Depth", row_number)
        lower = _manifest_number(row["Lower Depth"], "Lower Depth", row_number)
        if upper < 0 or lower <= upper:
            msg = (
                f"Manifest row {row_number} requires 0 <= Upper Depth < Lower Depth."
            )
            raise ValueError(msg)
        horizons.append(
            {
                "horizon": int(horizon_number),
                "upper": upper,
                "lower": lower,
                "clay": _soil_manifest_path(
                    manifest, row["Clay Layer"], "Clay Layer", row_number
                ),
                "sand": _soil_manifest_path(
                    manifest, row["Sand Layer"], "Sand Layer", row_number
                ),
                "silt": _soil_manifest_path(
                    manifest, row["Silt Layer"], "Silt Layer", row_number
                ),
                "bulk_density": _soil_manifest_path(
                    manifest,
                    row["Bulk Density Layer"],
                    "Bulk Density Layer",
                    row_number,
                ),
            }
        )

    horizons.sort(key=lambda item: item["horizon"])
    numbers = [item["horizon"] for item in horizons]
    if numbers != list(range(1, len(horizons) + 1)):
        msg = "Soil manifest Horizon values must be unique and consecutive from 1."
        raise ValueError(msg)
    if not np.isclose(horizons[0]["upper"], 0.0):
        msg = "The first soil horizon must start at depth 0."
        raise ValueError(msg)
    for previous, current in zip(horizons, horizons[1:]):
        if not np.isclose(previous["lower"], current["upper"]):
            msg = (
                "Soil horizons must be contiguous: horizon "
                f"{previous['horizon']} ends at {previous['lower']} but horizon "
                f"{current['horizon']} starts at {current['upper']}."
            )
            raise ValueError(msg)
    return manifest, unit, factor, horizons


def _aligned_continuous_layer(path, reference, *, input_crs, resampling):
    """Read one physical-property raster on the exact DEM grid."""
    source = get_raster_data(path, crs=input_crs)
    try:
        aligned = align_raster_to_reference(
            source,
            reference,
            nodata=np.nan,
            resampling=resampling,
            data_kind="continuous",
            mask_reference=True,
        )
        return np.asarray(aligned.values, dtype=np.float64)
    finally:
        source.close()


def _quantized_bins(values, valid, step: float) -> np.ndarray:
    """Quantize valid values with deterministic half-up rounding."""
    bins = np.full(values.shape, -1, dtype=np.int32)
    rounded = np.floor(values[valid] / step + 0.5)
    if rounded.size and rounded.max() > np.iinfo(np.int32).max:
        msg = "Soil quantization exceeds the supported int32 range."
        raise ValueError(msg)
    bins[valid] = rounded.astype(np.int32)
    return bins


def _prepare_soil_layers(
    horizons,
    reference,
    *,
    bulk_density_factor,
    composition_step,
    bulk_density_step,
    input_crs,
    resampling,
):
    """Align, normalize, validate, and quantize all soil properties."""
    layers = []
    for horizon in horizons:
        clay = _aligned_continuous_layer(
            horizon["clay"], reference, input_crs=input_crs, resampling=resampling
        )
        sand = _aligned_continuous_layer(
            horizon["sand"], reference, input_crs=input_crs, resampling=resampling
        )
        silt = _aligned_continuous_layer(
            horizon["silt"], reference, input_crs=input_crs, resampling=resampling
        )
        bulk_density = _aligned_continuous_layer(
            horizon["bulk_density"],
            reference,
            input_crs=input_crs,
            resampling=resampling,
        )
        bulk_density *= bulk_density_factor

        valid = (
            np.isfinite(clay)
            & np.isfinite(sand)
            & np.isfinite(silt)
            & (clay >= 0)
            & (sand >= 0)
            & (silt >= 0)
        )
        total = clay + sand + silt
        valid &= np.isfinite(total) & (total > 0)
        valid &= (
            np.isfinite(bulk_density)
            & (bulk_density > 0)
            & (bulk_density <= 5)
        )

        clay_percent = np.full(clay.shape, np.nan, dtype=np.float64)
        sand_percent = np.full(sand.shape, np.nan, dtype=np.float64)
        silt_percent = np.full(silt.shape, np.nan, dtype=np.float64)
        clay_percent[valid] = clay[valid] / total[valid] * 100.0
        sand_percent[valid] = sand[valid] / total[valid] * 100.0
        silt_percent[valid] = silt[valid] / total[valid] * 100.0
        layers.append(
            {
                **horizon,
                "valid": valid,
                "clay_bins": _quantized_bins(
                    clay_percent, valid, composition_step
                ),
                "sand_bins": _quantized_bins(
                    sand_percent, valid, composition_step
                ),
                "silt_bins": _quantized_bins(
                    silt_percent, valid, composition_step
                ),
                "bulk_density_bins": _quantized_bins(
                    bulk_density, valid, bulk_density_step
                ),
            }
        )
    return layers


def _soil_encoding():
    return {
        "soil_class": {
            "zlib": True,
            "complevel": 4,
            "shuffle": True,
            "_FillValue": int(NO_DATA),
            "dtype": "int32",
        }
    }


def _soil_spatial_dataset(classes, reference):
    return set_grid(
        classes,
        get_grid(reference.to_dataset(name="_dem"), "_dem"),
        "soil_class",
        data_attrs={
            "long_name": "mHM soil class",
            "units": "1",
            "nodata_value": int(NO_DATA),
        },
    )


def _classic_soil_outputs(
    layers,
    reference,
    output_path,
    *,
    composition_step,
    bulk_density_step,
):
    """Write one v5 profile-class raster and its classic soil LUT."""
    valid = np.logical_and.reduce([layer["valid"] for layer in layers])
    if not np.any(valid):
        msg = "No cell has valid soil data in every horizon."
        raise ValueError(msg)
    columns = []
    for layer in layers:
        columns.extend(
            [
                layer["clay_bins"].ravel(),
                layer["sand_bins"].ravel(),
                layer["bulk_density_bins"].ravel(),
            ]
        )
    profiles = np.column_stack(columns)
    unique_profiles, inverse = np.unique(
        profiles[valid.ravel()], axis=0, return_inverse=True
    )
    if unique_profiles.shape[0] > np.iinfo(np.int32).max:
        msg = "Too many soil profile classes for int32 output."
        raise ValueError(msg)

    classes = np.full(valid.size, int(NO_DATA), dtype=np.int32)
    classes[valid.ravel()] = inverse.astype(np.int32) + 1
    classes = classes.reshape(valid.shape)
    raster_output = output_path / "soil_class.asc"
    definition_output = output_path / "soil_classdefinition.txt"
    write_xarray_to_file(
        _soil_spatial_dataset(classes, reference),
        raster_output,
        var_name="soil_class",
        crs=reference.rio.crs,
    )

    lines = [
        f"nSoil_Types {unique_profiles.shape[0]}\n",
        "MU_GLOBAL\tHORIZON\tUD[mm]\tLD[mm]\tCLAY[%]\tSAND[%]\tBD[gcm-3]\n",
    ]
    for class_index, profile in enumerate(unique_profiles, start=1):
        for layer_index, layer in enumerate(layers):
            offset = layer_index * 3
            lines.append(
                f"{class_index}\t{layer['horizon']}\t"
                f"{_format_soil_value(layer['upper'])}\t"
                f"{_format_soil_value(layer['lower'])}\t"
                f"{_format_soil_value(profile[offset] * composition_step)}\t"
                f"{_format_soil_value(profile[offset + 1] * composition_step)}\t"
                f"{_format_soil_value(profile[offset + 2] * bulk_density_step)}\n"
            )
    _write_soil_classdefinition_text("".join(lines), definition_output)
    return raster_output, definition_output


def _horizon_soil_dataset(classes, layers, reference):
    """Create a CF-style layered soil-class dataset."""
    grid = get_grid(reference.to_dataset(name="_dem"), "_dem")
    coordinates = {
        name: coordinate
        for name, coordinate in grid.template.coords.items()
        if set(coordinate.dims).issubset(set(grid.dims))
    }
    lower_depths = np.asarray([layer["lower"] for layer in layers], dtype=np.float64)
    coordinates["z"] = xr.DataArray(lower_depths, dims=("z",))
    dataset = grid.template.copy(deep=False)
    dataset["soil_class"] = xr.DataArray(
        classes,
        dims=("z", *grid.dims),
        coords=coordinates,
        attrs={
            "long_name": "mHM soil class by horizon",
            "units": "1",
            "nodata_value": int(NO_DATA),
        },
    )
    dataset["z_bnds"] = xr.DataArray(
        np.asarray(
            [[layer["upper"], layer["lower"]] for layer in layers],
            dtype=np.float64,
        ),
        dims=("z", "bnds"),
    )
    dataset["z"].attrs.update(
        {
            "long_name": "soil horizon lower boundary depth",
            "standard_name": "depth",
            "units": "mm",
            "positive": "down",
            "axis": "Z",
            "bounds": "z_bnds",
        }
    )
    dataset["z_bnds"].attrs["long_name"] = "soil horizon depth bounds"
    dataset.attrs.update(
        {
            "title": "Horizon-specific soil classes for mHM",
            "source": "mhm-tools format-data soil manifest",
        }
    )
    return dataset


def _horizon_soil_outputs(
    layers,
    reference,
    output_path,
    *,
    composition_step,
    bulk_density_step,
):
    """Write one v6 horizon-class NetCDF and its mode-1 soil LUT."""
    observed = []
    packed_layers = []
    for layer in layers:
        valid = layer["valid"]
        packed = (
            layer["clay_bins"].astype(np.uint64) << np.uint64(32)
        ) | layer["sand_bins"].astype(np.uint32)
        packed_layers.append(packed)
        observed.append(packed[valid])
    observed = [values for values in observed if values.size]
    if not observed:
        msg = "No valid soil cells were found in any horizon."
        raise ValueError(msg)
    keys = np.unique(np.concatenate(observed))
    if keys.size > np.iinfo(np.int32).max:
        msg = "Too many horizon soil classes for int32 output."
        raise ValueError(msg)

    classes = np.full(
        (len(layers), *reference.shape), int(NO_DATA), dtype=np.int32
    )
    density_sum = np.zeros(keys.size, dtype=np.float64)
    counts = np.zeros(keys.size, dtype=np.int64)
    for index, (layer, packed) in enumerate(zip(layers, packed_layers)):
        valid = layer["valid"]
        positions = np.searchsorted(keys, packed[valid])
        classes[index][valid] = positions.astype(np.int32) + 1
        density_sum += np.bincount(
            positions,
            weights=layer["bulk_density_bins"][valid].astype(np.float64),
            minlength=keys.size,
        )
        counts += np.bincount(positions, minlength=keys.size)
    density_bins = np.floor(density_sum / counts + 0.5).astype(np.int32)

    raster_output = output_path / "soil_horizon_class.nc"
    definition_output = output_path / "soil_classdefinition_iFlag_soilDB_1.txt"
    dataset = _horizon_soil_dataset(classes, layers, reference)
    write_xarray_to_file(
        dataset,
        raster_output,
        var_name="soil_class",
        encoding=_soil_encoding(),
        crs=reference.rio.crs,
    )

    clay_bins = (keys >> np.uint64(32)).astype(np.int32)
    sand_bins = (keys & np.uint64(0xFFFFFFFF)).astype(np.int32)
    lines = [
        f"nSoil_Types {keys.size}\n",
        "ID\tCLAY[%]\tSAND[%]\tBD[gcm-3]\n",
    ]
    for class_index, (clay, sand, density) in enumerate(
        zip(clay_bins, sand_bins, density_bins), start=1
    ):
        lines.append(
            f"{class_index}\t"
            f"{_format_soil_value(clay * composition_step)}\t"
            f"{_format_soil_value(sand * composition_step)}\t"
            f"{_format_soil_value(density * bulk_density_step)}\n"
        )
    _write_soil_classdefinition_text("".join(lines), definition_output)
    return raster_output, definition_output


def format_soil_horizons(
    input_path: PathLike,
    dem_file: PathLike,
    output_path: PathLike,
    output_type: str = "nc",
    *,
    input_crs: str | None = None,
    dem_crs: str | None = None,
    resampling="auto",
    composition_step: float = 5.0,
    bulk_density_step: float = 0.1,
) -> tuple[Path, Path]:
    """Classify multi-horizon physical soil rasters from a directory manifest."""
    output_type = str(output_type).lower().lstrip(".")
    if output_type not in {"asc", "nc"}:
        msg = "Multi-horizon soil output extension must be 'asc' or 'nc'."
        raise ValueError(msg)
    steps = {}
    for name, raw_value in (
        ("composition_step", composition_step),
        ("bulk_density_step", bulk_density_step),
    ):
        try:
            value = float(raw_value)
        except (TypeError, ValueError) as exc:
            msg = f"{name} must be a positive finite number."
            raise ValueError(msg) from exc
        if not np.isfinite(value) or value <= 0:
            msg = f"{name} must be a positive finite number."
            raise ValueError(msg)
        steps[name] = value

    manifest, unit, density_factor, horizons = _read_soil_manifest(input_path)
    dem_file = Path(dem_file)
    output_path = Path(output_path)
    if not dem_file.is_file():
        msg = f"DEM raster does not exist: {dem_file}"
        raise ValueError(msg)
    if output_path.exists() and not output_path.is_dir():
        msg = f"Output path must be a directory: {output_path}"
        raise ValueError(msg)

    reference = get_raster_data(dem_file, crs=dem_crs)
    try:
        layers = _prepare_soil_layers(
            horizons,
            reference,
            bulk_density_factor=density_factor,
            composition_step=steps["composition_step"],
            bulk_density_step=steps["bulk_density_step"],
            input_crs=input_crs,
            resampling=resampling,
        )
        protected = {manifest.resolve(), dem_file.resolve()}
        for horizon in horizons:
            protected.update(
                horizon[name].resolve()
                for name in ("clay", "sand", "silt", "bulk_density")
            )
        expected = (
            (output_path / "soil_class.asc", output_path / "soil_classdefinition.txt")
            if output_type == "asc"
            else (
                output_path / "soil_horizon_class.nc",
                output_path / "soil_classdefinition_iFlag_soilDB_1.txt",
            )
        )
        if any(path.resolve() in protected for path in expected):
            msg = "Soil outputs must differ from all manifest input files."
            raise ValueError(msg)
        output_path.mkdir(parents=True, exist_ok=True)

        if output_type == "asc":
            outputs = _classic_soil_outputs(
                layers,
                reference,
                output_path,
                composition_step=steps["composition_step"],
                bulk_density_step=steps["bulk_density_step"],
            )
        else:
            outputs = _horizon_soil_outputs(
                layers,
                reference,
                output_path,
                composition_step=steps["composition_step"],
                bulk_density_step=steps["bulk_density_step"],
            )
        logger.info(
            "Formatted %d soil horizons with bulk density converted from %s.",
            len(horizons),
            unit,
        )
        return outputs
    finally:
        reference.close()
