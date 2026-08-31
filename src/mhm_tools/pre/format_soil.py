"""
Prepare mHM soil inputs from categorical rasters or physical horizon layers.

Two independent pipelines live here. The categorical one maps a soil raster
through a lookup table and writes the companion ``soil_classdefinition.txt``.
The horizon one reads a manifest of clay, sand, silt, and bulk-density rasters
for each horizon, warps them onto the DEM grid, and classifies them one raster
window at a time so that peak memory does not grow with the grid size; it
writes either v5 profile classes as ASCII or v6 per-horizon classes as NetCDF.

The sections below follow the horizon data flow: read the manifest, open and
warp the rasters onto the DEM grid, quantize each window into soil profiles,
then stream the classes out. The public entry points of both pipelines come
last.

Authors
-------
- Sanjeev Bashyal
"""

from __future__ import annotations

import logging
import sqlite3
import tempfile
from contextlib import ExitStack
from pathlib import Path
from typing import Union

import numpy as np
import pandas as pd
import rasterio
from netCDF4 import Dataset
from rasterio.crs import CRS
from rasterio.enums import Resampling
from rasterio.shutil import copy as copy_raster
from rasterio.vrt import WarpedVRT
from rasterio.warp import transform_bounds
from rasterio.windows import Window

from mhm_tools.common.constants import NO_DATA
from mhm_tools.common.format_data import (
    fill_grid_nodata,
    format_categorical_data,
    get_categorical_output_path,
)
from mhm_tools.common.lookup_handler import (
    _is_blank,
    _normalise_field_name,
    _required_integer,
    _required_number,
    read_format_manifest,
    read_lookup_table,
)

logger = logging.getLogger(__name__)

PathLike = Union[str, Path]


# Shared value formatting and class-definition output


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


def _write_soil_classdefinition_text(text: str, output_file: Path) -> Path:
    """Write prevalidated classdefinition text."""
    output_file.parent.mkdir(parents=True, exist_ok=True)
    output_file.write_text(text, encoding="utf-8")
    logger.info("Wrote soil class definition to %s", output_file)
    return output_file


# Soil class lookup tables


_CLASSDEFINITION_HEADER = (
    "SOIL_NR\tHORIZON\tUD[mm]\tLD[mm]\tClay[%]\tSAND[%]\tBd[gcm-3]\tSilt[%]\n"
)


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
                "soil_class": _required_integer(
                    row[soil_class_field], soil_class_field, row_number
                ),
                "horizon": _required_integer(
                    row[fields["horizon"]], fields["horizon"], row_number
                ),
                "upper_depth": _required_number(
                    row[fields["upper_depth"]], fields["upper_depth"], row_number
                ),
                "lower_depth": _required_number(
                    row[fields["lower_depth"]], fields["lower_depth"], row_number
                ),
                "clay": _required_number(
                    row[fields["clay"]], fields["clay"], row_number
                ),
                "sand": _required_number(
                    row[fields["sand"]], fields["sand"], row_number
                ),
                "silt": _required_number(
                    row[fields["silt"]], fields["silt"], row_number
                ),
                "bulk_density": _required_number(
                    row[fields["bulk_density"]],
                    fields["bulk_density"],
                    row_number,
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
        soil_class = _required_integer(
            row[soil_class_field], soil_class_field, row_number
        )
        horizons = _required_integer(row[horizons_field], horizons_field, row_number)
        upper_depth = 0.0

        for horizon in range(1, horizons + 1):
            depth_field = _required_soil_field(field_lookup, f"DEPTH{horizon}")
            bulk_density_field = _required_soil_field(
                field_lookup, f"BULK_DENSITY{horizon}"
            )
            clay_field = _required_soil_field(field_lookup, f"CLAY{horizon}")
            silt_field = _required_soil_field(field_lookup, f"SILT{horizon}")
            sand_field = _required_soil_field(field_lookup, f"SAND{horizon}")
            lower_depth = _required_number(row[depth_field], depth_field, row_number)
            output_rows.append(
                {
                    "soil_class": soil_class,
                    "horizon": horizon,
                    "upper_depth": upper_depth,
                    "lower_depth": lower_depth,
                    "clay": _required_number(row[clay_field], clay_field, row_number),
                    "sand": _required_number(row[sand_field], sand_field, row_number),
                    "silt": _required_number(row[silt_field], silt_field, row_number),
                    "bulk_density": _required_number(
                        row[bulk_density_field], bulk_density_field, row_number
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


# Horizon manifest


_SOIL_MANIFEST_COLUMNS = (
    "Horizon",
    "Upper Depth",
    "Lower Depth",
    "Clay Layer",
    "Sand Layer",
    "Silt Layer",
    "Bulk Density Layer",
    "Bulk Density Unit",
)


def _bulk_density_unit(unit: str) -> tuple[str, float]:
    """Return a canonical bulk-density unit and conversion to g/cm3."""
    normalized = (
        str(unit)
        .strip()
        .lower()
        .replace("³", "3")
        .replace("⁻", "-")
        .replace("\N{MINUS SIGN}", "-")
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


def _read_soil_manifest(input_file: PathLike):
    """Read bulk-density metadata and validated soil horizon records."""
    manifest, table = read_format_manifest(
        input_file,
        _SOIL_MANIFEST_COLUMNS,
    )

    horizons = []
    unit = None
    factor = None
    for row_number, (_, row) in enumerate(table.iterrows(), start=2):
        raw_unit = row["Bulk Density Unit"]
        if _is_blank(raw_unit):
            msg = f"Manifest row {row_number} has an empty Bulk Density Unit."
            raise ValueError(msg)
        row_unit, row_factor = _bulk_density_unit(raw_unit)
        if unit is None:
            unit, factor = row_unit, row_factor
        elif row_unit != unit:
            msg = "Soil manifest Bulk Density Unit must match in every row."
            raise ValueError(msg)
        horizon_number = _required_integer(
            row["Horizon"], "Horizon", row_number, "Manifest"
        )
        if horizon_number < 1:
            msg = (
                f"Manifest row {row_number} has invalid Horizon value "
                f"{row['Horizon']!r}."
            )
            raise ValueError(msg)
        upper = _required_number(
            row["Upper Depth"], "Upper Depth", row_number, "Manifest"
        )
        lower = _required_number(
            row["Lower Depth"], "Lower Depth", row_number, "Manifest"
        )
        if upper < 0 or lower <= upper:
            msg = (
                f"Manifest row {row_number} requires 0 <= Upper Depth < Lower Depth."
            )
            raise ValueError(msg)
        horizons.append(
            {
                "horizon": horizon_number,
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


# Raster grid access


_SOIL_BLOCK_ROWS = 64


def _open_raster(stack: ExitStack, path: Path):
    """Open one single-band raster, including a single NetCDF subdataset."""
    dataset = stack.enter_context(rasterio.open(path))
    if dataset.count == 0 and len(dataset.subdatasets) == 1:
        dataset = stack.enter_context(rasterio.open(dataset.subdatasets[0]))
    if dataset.count != 1:
        msg = f"Soil raster must contain exactly one data band: {path}"
        raise ValueError(msg)
    return dataset


def _soil_resampling(source, reference, source_crs, reference_crs, requested):
    """Resolve continuous-data resampling without loading either raster."""
    if isinstance(requested, Resampling):
        return requested
    method = str(requested).strip().lower()
    if method != "auto":
        try:
            return Resampling[method]
        except KeyError as exc:
            choices = ", ".join(item.name for item in Resampling)
            msg = f"Unsupported resampling method {requested!r}; expected {choices}."
            raise ValueError(msg) from exc
    bounds = transform_bounds(source_crs, reference_crs, *source.bounds)
    source_area = abs((bounds[2] - bounds[0]) * (bounds[3] - bounds[1]))
    source_pixel_area = source_area / (source.width * source.height)
    reference_pixel_area = abs(
        reference.transform.a * reference.transform.e
        - reference.transform.b * reference.transform.d
    )
    return (
        Resampling.average
        if source_pixel_area < reference_pixel_area
        else Resampling.bilinear
    )


def _open_soil_grid(
    stack: ExitStack,
    dem_file: Path,
    horizons,
    *,
    input_crs,
    dem_crs,
    resampling,
):
    """Open the DEM and lightweight warped views of all soil inputs."""
    reference = _open_raster(stack, dem_file)
    reference_crs = reference.crs or (CRS.from_user_input(dem_crs) if dem_crs else None)
    if reference_crs is None:
        msg = f"DEM raster has no CRS metadata: {dem_file}"
        raise ValueError(msg)

    warped = []
    for horizon in horizons:
        properties = {}
        for name in ("clay", "sand", "silt", "bulk_density"):
            source = _open_raster(stack, horizon[name])
            source_crs = source.crs or (
                CRS.from_user_input(input_crs) if input_crs else None
            )
            if source_crs is None:
                msg = f"Soil raster has no CRS metadata: {horizon[name]}"
                raise ValueError(msg)
            method = _soil_resampling(
                source, reference, source_crs, reference_crs, resampling
            )
            options = {
                "crs": reference_crs,
                "transform": reference.transform,
                "width": reference.width,
                "height": reference.height,
                "resampling": method,
                "nodata": np.nan,
                "dtype": "float32",
            }
            if source.crs is None:
                options["src_crs"] = source_crs
            properties[name] = stack.enter_context(WarpedVRT(source, **options))
        warped.append(properties)
    return reference, reference_crs, warped


def _dem_valid(reference, window=None):
    """Return the boolean mask of DEM cells that carry data."""
    dem = reference.read(1, window=window, masked=True)
    values = np.ma.getdata(dem)
    valid = ~np.ma.getmaskarray(dem)
    if np.issubdtype(values.dtype, np.floating):
        valid = valid & np.isfinite(values)
    return valid


def _grid_coordinates(reference):
    """Return the cell-centre coordinates of a north-up reference grid."""
    transform = reference.transform
    x = transform.c + (np.arange(reference.width) + 0.5) * transform.a
    y = transform.f + (np.arange(reference.height) + 0.5) * transform.e
    return x, y


def _soil_windows(reference):
    for row in range(0, reference.height, _SOIL_BLOCK_ROWS):
        height = min(_SOIL_BLOCK_ROWS, reference.height - row)
        yield Window(0, row, reference.width, height)


def _read_float(dataset, window):
    return dataset.read(1, window=window, masked=True, out_dtype="float32").filled(
        np.nan
    )


def _fill_soil_layers(stack, reference, reference_crs, warped, horizons, temp_path):
    """Replace the warped soil views with nodata-filled copies of themselves.

    Every physical property of every horizon is materialized on the DEM grid
    and its remaining nodata cells inside the DEM domain are taken from the
    nearest valid cell of the same layer. Valid cells outside the DEM domain
    stay available as fill sources, but are written as nodata so that the
    filled layer covers exactly the modelled domain.
    """
    dem_valid = _dem_valid(reference)
    if not np.any(dem_valid):
        msg = "The DEM has no valid cell to define the soil domain."
        raise ValueError(msg)
    outside = ~dem_valid
    x, y = _grid_coordinates(reference)
    profile = {
        "driver": "GTiff",
        "height": reference.height,
        "width": reference.width,
        "count": 1,
        "dtype": "float32",
        "crs": reference_crs,
        "transform": reference.transform,
        "nodata": np.nan,
        "compress": "deflate",
        "tiled": True,
        "blockxsize": 256,
        "blockysize": _SOIL_BLOCK_ROWS,
        "BIGTIFF": "IF_SAFER",
    }

    filled = []
    for horizon, properties in zip(horizons, warped):
        replacements = {}
        for name, source in properties.items():
            values = _read_float(source, None)
            missing = int(np.count_nonzero(~np.isfinite(values) & dem_valid))
            count = fill_grid_nodata(
                values,
                x=x,
                y=y,
                mask=outside,
                fill_value=np.nan,
                name=f"{name}_horizon_{horizon['horizon']}",
                source_file=horizon[name],
            )
            if missing:
                logger.info(
                    "Filled %d of %d nodata cells of the %s layer of horizon "
                    "%d from nearest valid neighbours (%s).",
                    count,
                    missing,
                    name,
                    horizon["horizon"],
                    horizon[name],
                )
            layer_path = temp_path / f"filled_h{horizon['horizon']}_{name}.tif"
            with rasterio.open(layer_path, "w", **profile) as target:
                target.write(values, 1)
            replacements[name] = stack.enter_context(rasterio.open(layer_path))
        filled.append(replacements)
    return filled


# Quantized soil profiles


def _quantized_bins(values, valid, step: float) -> np.ndarray:
    """Quantize valid values with deterministic half-up rounding."""
    bins = np.full(values.shape, -1, dtype=np.int32)
    rounded = np.floor(values[valid] / step + 0.5)
    if rounded.size and rounded.max() > np.iinfo(np.int32).max:
        msg = "Soil quantization exceeds the supported int32 range."
        raise ValueError(msg)
    bins[valid] = rounded.astype(np.int32)
    return bins


def _quantized_soil_block(
    properties,
    window,
    dem_valid,
    *,
    density_factor,
    composition_step,
    density_step,
):
    """Normalize and quantize one horizon in one target-grid window."""
    clay = _read_float(properties["clay"], window)
    sand = _read_float(properties["sand"], window)
    silt = _read_float(properties["silt"], window)
    density = _read_float(properties["bulk_density"], window)
    density *= density_factor
    total = clay + sand + silt
    valid = (
        dem_valid
        & np.isfinite(total)
        & (total > 0)
        & np.isfinite(clay)
        & (clay >= 0)
        & np.isfinite(sand)
        & (sand >= 0)
        & np.isfinite(silt)
        & (silt >= 0)
        & np.isfinite(density)
        & (density > 0)
        & (density <= 5)
    )
    clay_percent = np.full(clay.shape, np.nan, dtype=np.float32)
    sand_percent = np.full(sand.shape, np.nan, dtype=np.float32)
    np.divide(clay, total, out=clay_percent, where=valid)
    np.divide(sand, total, out=sand_percent, where=valid)
    clay = _quantized_bins(clay_percent * 100.0, valid, composition_step)
    sand = _quantized_bins(sand_percent * 100.0, valid, composition_step)
    density = _quantized_bins(density, valid, density_step)
    return valid, clay, sand, density


def _packed_texture_keys(clay, sand, valid) -> np.ndarray:
    """Pack the quantized clay and sand bins of valid cells into one key each.

    The v6 classes are identified by their texture alone, so a single unsigned
    key per cell lets both passes look a class up with ``np.searchsorted``.
    """
    return (clay[valid].astype(np.uint64) << np.uint64(32)) | sand[valid].astype(
        np.uint32
    )


def _profile_block(
    warped,
    reference,
    window,
    *,
    density_factor,
    composition_step,
    density_step,
):
    dem_valid = _dem_valid(reference, window)
    layers = []
    valid = dem_valid
    for properties in warped:
        layer = _quantized_soil_block(
            properties,
            window,
            dem_valid,
            density_factor=density_factor,
            composition_step=composition_step,
            density_step=density_step,
        )
        layers.append(layer)
        valid = valid & layer[0]
    profiles = np.empty((int(valid.sum()), len(layers) * 3), dtype=np.int32)
    for index, (_, clay, sand, density) in enumerate(layers):
        profiles[:, index * 3 : index * 3 + 3] = np.column_stack(
            (clay[valid], sand[valid], density[valid])
        )
    return valid, profiles


# Horizon soil writers


def _classic_definition_text(
    rows, horizons, composition_step: float, density_step: float
) -> str:
    """Render the v5 ``soil_classdefinition.txt`` for the profile classes.

    ``rows`` holds one quantized profile per class, as consecutive triples of
    clay, sand, and bulk-density bins for each horizon in turn.
    """
    lines = [
        f"nSoil_Types {len(rows)}\n",
        "MU_GLOBAL\tHORIZON\tUD[mm]\tLD[mm]\tCLAY[%]\tSAND[%]\tBD[gcm-3]\n",
    ]
    for class_index, row in enumerate(rows, start=1):
        for layer_index, horizon in enumerate(horizons):
            offset = layer_index * 3
            lines.append(
                f"{class_index}\t{horizon['horizon']}\t"
                f"{_format_soil_value(horizon['upper'])}\t"
                f"{_format_soil_value(horizon['lower'])}\t"
                f"{_format_soil_value(row[offset] * composition_step)}\t"
                f"{_format_soil_value(row[offset + 1] * composition_step)}\t"
                f"{_format_soil_value(row[offset + 2] * density_step)}\n"
            )
    return "".join(lines)


def _stream_classic_soil(
    reference,
    reference_crs,
    warped,
    horizons,
    output_path,
    temp_path,
    *,
    density_factor,
    composition_step,
    density_step,
):
    """Write v5 profile classes with memory bounded by one raster window."""
    columns = [f"v{index} INTEGER" for index in range(len(horizons) * 3)]
    names = [f"v{index}" for index in range(len(columns))]
    database = sqlite3.connect(temp_path / "profiles.sqlite")
    try:
        database.execute(
            f"CREATE TABLE profiles ({', '.join(columns)}, "
            f"PRIMARY KEY ({', '.join(names)})) WITHOUT ROWID"
        )
        placeholders = ", ".join("?" for _ in names)
        insert = f"INSERT OR IGNORE INTO profiles VALUES ({placeholders})"
        found = False
        for window in _soil_windows(reference):
            _, profiles = _profile_block(
                warped,
                reference,
                window,
                density_factor=density_factor,
                composition_step=composition_step,
                density_step=density_step,
            )
            if profiles.size:
                found = True
                unique = np.unique(profiles, axis=0)
                database.executemany(insert, map(tuple, unique.tolist()))
            database.commit()
        if not found:
            msg = "No cell has valid soil data in every horizon."
            raise ValueError(msg)
        rows = database.execute(
            f"SELECT {', '.join(names)} FROM profiles ORDER BY {', '.join(names)}"
        ).fetchall()
    finally:
        database.close()
    if len(rows) > np.iinfo(np.int32).max:
        msg = "Too many soil profile classes for int32 output."
        raise ValueError(msg)
    class_ids = {tuple(row): index for index, row in enumerate(rows, start=1)}

    temporary_raster = temp_path / "soil_class.tif"
    profile = reference.profile.copy()
    profile.update(
        driver="GTiff",
        count=1,
        dtype="int32",
        nodata=int(NO_DATA),
        crs=reference_crs,
        compress="deflate",
        tiled=True,
        blockxsize=256,
        blockysize=_SOIL_BLOCK_ROWS,
    )
    with rasterio.open(temporary_raster, "w", **profile) as target:
        for window in _soil_windows(reference):
            valid, profiles = _profile_block(
                warped,
                reference,
                window,
                density_factor=density_factor,
                composition_step=composition_step,
                density_step=density_step,
            )
            classes = np.full(valid.shape, int(NO_DATA), dtype=np.int32)
            if profiles.size:
                unique, inverse = np.unique(profiles, axis=0, return_inverse=True)
                ids = np.asarray(
                    [class_ids[tuple(row)] for row in unique.tolist()], dtype=np.int32
                )
                classes[valid] = ids[inverse]
            target.write(classes, 1, window=window)

    raster_output = output_path / "soil_class.asc"
    definition_output = output_path / "soil_classdefinition.txt"
    copy_raster(temporary_raster, raster_output, driver="AAIGrid")
    _write_soil_classdefinition_text(
        _classic_definition_text(rows, horizons, composition_step, density_step),
        definition_output,
    )
    return raster_output, definition_output


def _create_soil_netcdf(path, reference, reference_crs, horizons):
    """Create the CF structure needed for block-wise v6 output."""
    dataset = Dataset(path, "w", format="NETCDF4")
    dataset.createDimension("z", len(horizons))
    dataset.createDimension("y", reference.height)
    dataset.createDimension("x", reference.width)
    dataset.createDimension("bnds", 2)
    transform = reference.transform
    x = dataset.createVariable("x", "f8", ("x",))
    y = dataset.createVariable("y", "f8", ("y",))
    z = dataset.createVariable("z", "f8", ("z",))
    z_bnds = dataset.createVariable("z_bnds", "f8", ("z", "bnds"))
    x[:], y[:] = _grid_coordinates(reference)
    z[:] = [item["lower"] for item in horizons]
    z_bnds[:] = [[item["upper"], item["lower"]] for item in horizons]
    x.setncatts({"standard_name": "projection_x_coordinate", "axis": "X"})
    y.setncatts({"standard_name": "projection_y_coordinate", "axis": "Y"})
    z.setncatts(
        {
            "long_name": "soil horizon lower boundary depth",
            "standard_name": "depth",
            "units": "mm",
            "positive": "down",
            "axis": "Z",
            "bounds": "z_bnds",
        }
    )
    z_bnds.long_name = "soil horizon depth bounds"
    crs = dataset.createVariable("crs", "i4")
    crs.setncatts(
        {
            "spatial_ref": reference_crs.to_wkt(),
            "crs_wkt": reference_crs.to_wkt(),
            "GeoTransform": " ".join(str(value) for value in transform.to_gdal()),
        }
    )
    classes = dataset.createVariable(
        "soil_class",
        "i4",
        ("z", "y", "x"),
        fill_value=int(NO_DATA),
        zlib=True,
        complevel=4,
        shuffle=True,
        chunksizes=(
            1,
            min(_SOIL_BLOCK_ROWS, reference.height),
            min(512, reference.width),
        ),
    )
    classes.setncatts(
        {
            "long_name": "mHM soil class by horizon",
            "units": "1",
            "nodata_value": int(NO_DATA),
            "grid_mapping": "crs",
        }
    )
    dataset.setncatts(
        {
            "Conventions": "CF-1.8",
            "title": "Horizon-specific soil classes for mHM",
            "source": "mhm-tools format-data soil manifest",
        }
    )
    return dataset, classes


def _horizon_definition_text(
    keys, density_bins, composition_step: float, density_step: float
) -> str:
    """Render the v6 ``soil_classdefinition_iFlag_soilDB_1.txt`` lookup.

    ``keys`` are the packed texture keys of :func:`_packed_texture_keys`, one
    per class, and ``density_bins`` their mean quantized bulk density.
    """
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
            f"{class_index}\t{_format_soil_value(clay * composition_step)}\t"
            f"{_format_soil_value(sand * composition_step)}\t"
            f"{_format_soil_value(density * density_step)}\n"
        )
    return "".join(lines)


def _stream_horizon_soil(
    reference,
    reference_crs,
    warped,
    horizons,
    output_path,
    *,
    density_factor,
    composition_step,
    density_step,
):
    """Write v6 horizon classes with memory bounded by one raster window."""
    observed = set()
    for window in _soil_windows(reference):
        dem_valid = _dem_valid(reference, window)
        for properties in warped:
            valid, clay, sand, _ = _quantized_soil_block(
                properties,
                window,
                dem_valid,
                density_factor=density_factor,
                composition_step=composition_step,
                density_step=density_step,
            )
            packed = _packed_texture_keys(clay, sand, valid)
            observed.update(np.unique(packed).tolist())
    if not observed:
        msg = "No valid soil cells were found in any horizon."
        raise ValueError(msg)
    keys = np.asarray(sorted(observed), dtype=np.uint64)
    density_sum = np.zeros(keys.size, dtype=np.float64)
    counts = np.zeros(keys.size, dtype=np.int64)
    raster_output = output_path / "soil_horizon_class.nc"
    definition_output = output_path / "soil_classdefinition_iFlag_soilDB_1.txt"
    dataset, output = _create_soil_netcdf(
        raster_output, reference, reference_crs, horizons
    )
    try:
        for window in _soil_windows(reference):
            row = int(window.row_off)
            height = int(window.height)
            dem_valid = _dem_valid(reference, window)
            for horizon_index, properties in enumerate(warped):
                valid, clay, sand, density = _quantized_soil_block(
                    properties,
                    window,
                    dem_valid,
                    density_factor=density_factor,
                    composition_step=composition_step,
                    density_step=density_step,
                )
                packed = _packed_texture_keys(clay, sand, valid)
                positions = np.searchsorted(keys, packed)
                classes = np.full(valid.shape, int(NO_DATA), dtype=np.int32)
                classes[valid] = positions.astype(np.int32) + 1
                output[horizon_index, row : row + height, :] = classes
                density_sum += np.bincount(
                    positions,
                    weights=density[valid].astype(np.float64),
                    minlength=keys.size,
                )
                counts += np.bincount(positions, minlength=keys.size)
    finally:
        dataset.close()
    density_bins = np.floor(density_sum / counts + 0.5).astype(np.int32)
    _write_soil_classdefinition_text(
        _horizon_definition_text(keys, density_bins, composition_step, density_step),
        definition_output,
    )
    return raster_output, definition_output


# Public entry points


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
    fill_nodata: bool = True,
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
    fill_nodata : bool, default True
        Restrict the output to the DEM domain and take its remaining nodata
        cells from the nearest classified neighbour.

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
        fill_nodata=fill_nodata,
    )
    _write_soil_classdefinition_text(definition_text, definition_output)
    return raster_output


def format_soil_horizons(
    input_file: PathLike,
    dem_file: PathLike,
    output_path: PathLike,
    output_type: str = "nc",
    *,
    input_crs: str | None = None,
    dem_crs: str | None = None,
    resampling="auto",
    composition_step: float = 5.0,
    bulk_density_step: float = 0.1,
    fill_nodata: bool = True,
) -> tuple[Path, Path]:
    """Classify multi-horizon physical soil rasters from a manifest.

    Parameters
    ----------
    fill_nodata : bool, default True
        Take the nodata cells of every clay, sand, silt, and bulk-density
        layer from their nearest valid neighbour before classification, so
        that gaps in one input layer do not drop cells from the output.
    """
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

    manifest, unit, density_factor, horizons = _read_soil_manifest(input_file)
    dem_file = Path(dem_file)
    output_path = Path(output_path)
    if not dem_file.is_file():
        msg = f"DEM raster does not exist: {dem_file}"
        raise ValueError(msg)
    if output_path.exists() and not output_path.is_dir():
        msg = f"Output path must be a directory: {output_path}"
        raise ValueError(msg)

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

    with tempfile.TemporaryDirectory(prefix="mhm-soil-", dir=output_path) as temporary:
        with ExitStack() as stack:
            stack.enter_context(rasterio.Env(GDAL_CACHEMAX=128 * 1024 * 1024))
            reference, reference_crs, warped = _open_soil_grid(
                stack,
                dem_file,
                horizons,
                input_crs=input_crs,
                dem_crs=dem_crs,
                resampling=resampling,
            )
            if fill_nodata:
                warped = _fill_soil_layers(
                    stack,
                    reference,
                    reference_crs,
                    warped,
                    horizons,
                    Path(temporary),
                )
            if output_type == "asc":
                outputs = _stream_classic_soil(
                    reference,
                    reference_crs,
                    warped,
                    horizons,
                    output_path,
                    Path(temporary),
                    density_factor=density_factor,
                    composition_step=steps["composition_step"],
                    density_step=steps["bulk_density_step"],
                )
            else:
                outputs = _stream_horizon_soil(
                    reference,
                    reference_crs,
                    warped,
                    horizons,
                    output_path,
                    density_factor=density_factor,
                    composition_step=steps["composition_step"],
                    density_step=steps["bulk_density_step"],
                )
        logger.info(
            "Formatted %d soil horizons with bulk density converted from %s.",
            len(horizons),
            unit,
        )
        return outputs
