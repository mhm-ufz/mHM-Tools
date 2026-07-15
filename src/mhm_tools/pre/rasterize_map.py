"""Rasterize vector map data on a DEM grid."""

from pathlib import Path

import numpy as np
import pandas as pd

from mhm_tools.common.constants import NO_DATA
from mhm_tools.common.file_handler import (
    get_grid,
    get_raster_data,
    set_grid,
    write_xarray_to_file,
)
from mhm_tools.common.rasterize import rasterize_vector

_OUTPUT_SUFFIXES = {".asc", ".nc", ".tif", ".tiff"}
_OUTPUT_VARIABLE = "rasterized_map"


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
    lookup_table: Path, lookup_mapping_field: str, lookup_value_field: str
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

    key_field = _resolve_field(table.columns, lookup_mapping_field, "lookup table")
    value_field = _resolve_field(table.columns, lookup_value_field, "lookup table")
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


def _assign_vector_crs(frame, input_crs, input_path):
    """Assign a missing vector CRS or reject an explicit conflict."""
    from pyproj import CRS

    explicit = CRS.from_user_input(input_crs) if input_crs is not None else None
    if frame.crs is None:
        if explicit is None:
            msg = f"Input vector has no coordinate reference system: {input_path}"
            raise ValueError(msg)
        return frame.set_crs(explicit)
    if explicit is not None and frame.crs != explicit:
        msg = f"Explicit CRS {explicit} conflicts with vector CRS {frame.crs}."
        raise ValueError(msg)
    return frame


def _validate_output_collisions(output_path: Path, sources) -> None:
    """Prevent raster or ASCII sidecar outputs from replacing inputs."""
    protected = {path.resolve() for path in sources}
    output_targets = {output_path.resolve()}
    if output_path.suffix.lower() == ".asc":
        output_targets.update(
            output_path.with_suffix(suffix).resolve() for suffix in (".prj", ".PRJ")
        )
        for source in sources:
            if source.suffix.lower() in {".asc", ".shp"}:
                protected.update(
                    source.with_suffix(suffix).resolve() for suffix in (".prj", ".PRJ")
                )
    collisions = output_targets & protected
    if collisions:
        paths = ", ".join(str(path) for path in sorted(collisions))
        msg = f"Output file or sidecar must differ from all input files: {paths}"
        raise ValueError(msg)


def rasterize_map_data(
    input_file,
    dem_file,
    output_file,
    mapping_field,
    *,
    lookup_table=None,
    lookup_mapping_field=None,
    lookup_value_field="SOIL_CLASS",
    input_crs=None,
    dem_crs=None,
) -> Path:
    """Rasterize a vector attribute using a DEM as the exact target grid.

    Parameters
    ----------
    input_file : path-like
        Input vector file.
    dem_file : path-like
        DEM providing the target CRS, transform, extent, and dimensions.
    output_file : path-like
        Destination ASCII, NetCDF, or GeoTIFF file. The suffix selects the
        output format.
    mapping_field : str
        Vector attribute to burn directly, or to map through ``lookup_table``.
    lookup_table : path-like, optional
        OGR-readable lookup table for textual or numeric vector categories.
    lookup_mapping_field : str, optional
        Lookup category column. Defaults to ``mapping_field``.
    lookup_value_field : str, default "SOIL_CLASS"
        Lookup column containing finite integral int32 burn values.
    input_crs : str or CRS, optional
        CRS to assign when the vector has no embedded CRS metadata.
    dem_crs : str or CRS, optional
        CRS to assign when the DEM has no embedded or sidecar CRS metadata.

    Returns
    -------
    pathlib.Path
        Path to the created raster.
    """
    import geopandas as gpd

    input_path = Path(input_file)
    dem_path = Path(dem_file)
    output_path = Path(output_file)
    output_suffix = output_path.suffix.lower()

    if not input_path.is_file():
        msg = f"Input vector file does not exist: {input_path}"
        raise FileNotFoundError(msg)
    if not dem_path.is_file():
        msg = f"DEM file does not exist: {dem_path}"
        raise FileNotFoundError(msg)
    if output_suffix not in _OUTPUT_SUFFIXES:
        msg = "Output file must use an .asc, .nc, .tif, or .tiff suffix."
        raise ValueError(msg)
    if not isinstance(mapping_field, str) or not mapping_field.strip():
        msg = "Mapping field must be a non-empty column name."
        raise ValueError(msg)
    lookup_path = None
    if lookup_table is not None:
        lookup_path = Path(lookup_table)
        if not lookup_path.is_file():
            msg = f"Lookup table does not exist: {lookup_path}"
            raise FileNotFoundError(msg)
    elif lookup_mapping_field is not None or lookup_value_field != "SOIL_CLASS":
        msg = "Lookup field options require lookup_table."
        raise ValueError(msg)

    sources = [input_path, dem_path]
    if lookup_path is not None:
        sources.append(lookup_path)
    _validate_output_collisions(output_path, sources)

    reference = get_raster_data(dem_path, crs=dem_crs)
    try:
        y_dim, x_dim = reference.rio.y_dim, reference.rio.x_dim
        dem = reference.transpose(y_dim, x_dim).rio.set_spatial_dims(
            x_dim=x_dim, y_dim=y_dim
        )
        if dem.rio.width < 1 or dem.rio.height < 1:
            msg = f"DEM does not define a non-empty raster grid: {dem_path}"
            raise ValueError(msg)
        if dem.rio.crs is None:
            msg = f"DEM has no coordinate reference system: {dem_path}"
            raise ValueError(msg)
        transform = dem.rio.transform()
        transform_values = np.asarray(tuple(transform)[:6], dtype=float)
        if not np.all(np.isfinite(transform_values)) or transform.is_degenerate:
            msg = f"DEM has an invalid affine transform: {dem_path}"
            raise ValueError(msg)
        if not np.all(np.isfinite(dem.rio.bounds())):
            msg = f"DEM has invalid spatial bounds: {dem_path}"
            raise ValueError(msg)

        out_shape = (dem.rio.height, dem.rio.width)
        target_crs = dem.rio.crs
        grid = get_grid(dem.to_dataset(name="_dem"), "_dem")

        frame = _assign_vector_crs(gpd.read_file(input_path), input_crs, input_path)
        if frame.crs != target_crs:
            frame = frame.to_crs(target_crs)

        vector_field = _resolve_field(frame.columns, mapping_field, "input vector")
        output_field = vector_field
        if lookup_path is not None:
            lookup_mapping_field = lookup_mapping_field or mapping_field
            lookup = _read_lookup_mapping(
                lookup_path, lookup_mapping_field, lookup_value_field
            )
            burn_field = "__mhm_tools_raster_value__"
            while burn_field in frame.columns:
                burn_field = f"_{burn_field}"
            frame = frame.copy()
            frame[burn_field] = _map_vector_values(frame, vector_field, lookup)
            output_field = burn_field

        data = rasterize_vector(frame, output_field, out_shape, transform)
        description = (
            lookup_value_field if lookup_path is not None else str(vector_field)
        )
        output = set_grid(
            data,
            grid,
            _OUTPUT_VARIABLE,
            data_attrs={
                "long_name": description,
                "nodata_value": int(NO_DATA),
            },
        )
        encoding = {
            _OUTPUT_VARIABLE: {
                "zlib": True,
                "complevel": 4,
                "shuffle": True,
                "_FillValue": int(NO_DATA),
                "dtype": "int32",
            }
        }
        if output_suffix in {".tif", ".tiff"}:
            raster = output[_OUTPUT_VARIABLE].rio.write_crs(target_crs)
            raster = raster.rio.write_nodata(int(NO_DATA), encoded=True)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            raster.rio.to_raster(output_path, dtype="int32", compress="deflate")
        else:
            write_xarray_to_file(
                output,
                output_path,
                var_name=_OUTPUT_VARIABLE,
                encoding=encoding if output_suffix == ".nc" else None,
                crs=target_crs,
            )
    finally:
        reference.close()

    return output_path
