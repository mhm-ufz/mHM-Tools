"""
Prepare an mHM land-cover raster from categorical raster data.

Authors
-------
- Sanjeev Bashyal
"""

from __future__ import annotations

from pathlib import Path
from typing import Union

PathLike = Union[str, Path]


def format_lc_data(
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
    """Map a categorical raster to mHM land-cover classes on the DEM grid."""
    from mhm_tools.common.format_data import (
        format_categorical_data,
        get_categorical_output_path,
        read_lookup_table,
    )

    input_file = Path(input_file)
    dem_file = Path(dem_file)
    lookup_table = Path(lookup_table)
    raster_output = get_categorical_output_path(output_path, "lc", output_type)

    if raster_output.resolve() in {
        input_file.resolve(),
        dem_file.resolve(),
        lookup_table.resolve(),
    }:
        msg = f"Raster output must differ from all input files: {raster_output}"
        raise ValueError(msg)

    return format_categorical_data(
        input_file,
        dem_file,
        raster_output,
        read_lookup_table(lookup_table),
        mapping_field,
        class_field,
        variable_name="land_cover",
        input_crs=input_crs,
        dem_crs=dem_crs,
    )
