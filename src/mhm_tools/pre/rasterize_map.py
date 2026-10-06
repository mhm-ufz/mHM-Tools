"""Rasterize vector map data on a DEM grid."""

from pathlib import Path

import numpy as np

from mhm_tools.common.constants import NO_DATA
from mhm_tools.common.file_handler import (
    get_grid,
    get_raster_data,
    set_grid,
    write_xarray_to_file,
)
from mhm_tools.common.lookup_handler import (
    _map_vector_values,
    _read_lookup_mapping,
    _resolve_field,
)
from mhm_tools.common.rasterize import rasterize_vector

_OUTPUT_SUFFIXES = {".asc", ".nc", ".tif", ".tiff"}
_OUTPUT_VARIABLE = "rasterized_map"


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
    burn_field,
    *,
    lookup_table=None,
    mapping_field=None,
    input_crs=None,
    dem_crs=None,
    compression=None,
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
    burn_field : str
        Column containing the values to burn. This column is read from the
        input vector directly, or from ``lookup_table`` when one is provided.
    lookup_table : path-like, optional
        OGR-readable lookup table for textual or numeric vector categories.
    mapping_field : str, optional
        Category column shared by the input vector and lookup table. Required
        with ``lookup_table``.
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
    if not isinstance(burn_field, str) or not burn_field.strip():
        msg = "Burn field must be a non-empty column name."
        raise ValueError(msg)
    if (lookup_table is None) != (mapping_field is None):
        msg = "lookup_table and mapping_field must be provided together."
        raise ValueError(msg)
    if lookup_table is not None and (
        not isinstance(mapping_field, str) or not mapping_field.strip()
    ):
        msg = "Mapping field must be a non-empty column name."
        raise ValueError(msg)
    lookup_path = None
    if lookup_table is not None:
        lookup_path = Path(lookup_table)
        if not lookup_path.is_file():
            msg = f"Lookup table does not exist: {lookup_path}"
            raise FileNotFoundError(msg)

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

        vector_field = _resolve_field(
            frame.columns,
            mapping_field if lookup_path is not None else burn_field,
            "input vector",
        )
        output_field = vector_field
        if lookup_path is not None:
            lookup = _read_lookup_mapping(lookup_path, mapping_field, burn_field)
            output_field = burn_field
            frame = frame.copy()
            frame[output_field] = _map_vector_values(frame, vector_field, lookup)

        data = rasterize_vector(frame, output_field, out_shape, transform)
        output = set_grid(
            data,
            grid,
            _OUTPUT_VARIABLE,
            data_attrs={
                "long_name": burn_field,
                "nodata_value": int(NO_DATA),
            },
        )
        encoding = {
            _OUTPUT_VARIABLE: {
                "_FillValue": int(NO_DATA),
                "dtype": "int32",
            }
        }
        write_xarray_to_file(
            output,
            output_path,
            var_name=_OUTPUT_VARIABLE,
            encoding=encoding if output_suffix == ".nc" else None,
            crs=target_crs,
            compression=compression,
            geotiff_compression="deflate",
        )
    finally:
        reference.close()

    return output_path
