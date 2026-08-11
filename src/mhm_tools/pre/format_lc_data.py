"""
Prepare an mHM land-cover raster from categorical raster data.

Authors
-------
- Sanjeev Bashyal
"""

from __future__ import annotations

from pathlib import Path
from typing import Union

import numpy as np
import pandas as pd
import xarray as xr

from mhm_tools.common.constants import NO_DATA
from mhm_tools.common.file_handler import get_raster_data, write_xarray_to_file
from mhm_tools.common.format_data import (
    format_categorical_data,
    get_categorical_output_path,
    prepare_categorical_data,
    read_format_manifest,
    read_lookup_table,
)

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
    resampling="auto",
) -> Path:
    """Map a categorical raster to mHM land-cover classes on the DEM grid."""
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
        resampling=resampling,
    )


def _manifest_year(value, column: str, row_number: int) -> int:
    """Return one required integer year from a manifest row."""
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        msg = f"Manifest row {row_number} has invalid {column} value {value!r}."
        raise ValueError(msg) from exc
    if not np.isfinite(number) or not number.is_integer():
        msg = f"Manifest row {row_number} has invalid {column} value {value!r}."
        raise ValueError(msg)
    return int(number)


def _manifest_raster_path(manifest: Path, value, row_number: int) -> Path:
    """Resolve one manifest raster path relative to its manifest."""
    if pd.isna(value) or not str(value).strip():
        msg = f"Manifest row {row_number} has an empty FilePath."
        raise ValueError(msg)
    path = Path(str(value).strip()).expanduser()
    if not path.is_absolute():
        path = manifest.parent / path
    path = path.resolve()
    if not path.is_file():
        msg = f"Manifest raster does not exist: {path}"
        raise ValueError(msg)
    return path


def _read_lc_manifest(input_path: PathLike):
    """Read validated, chronological land-cover period records."""
    manifest, table = read_format_manifest(
        input_path,
        ("StartYear", "EndYear", "FilePath"),
    )
    periods = []
    for row_number, row in enumerate(table.itertuples(index=False), start=2):
        start = _manifest_year(row.StartYear, "StartYear", row_number)
        end = _manifest_year(row.EndYear, "EndYear", row_number)
        if start > end:
            msg = (
                f"Manifest row {row_number} has StartYear {start} after EndYear {end}."
            )
            raise ValueError(msg)
        periods.append(
            {
                "start": start,
                "end": end,
                "path": _manifest_raster_path(manifest, row.FilePath, row_number),
            }
        )
    periods.sort(key=lambda period: (period["start"], period["end"]))
    for previous, current in zip(periods, periods[1:]):
        if current["start"] <= previous["end"]:
            msg = (
                "Land-cover periods overlap: "
                f"{previous['start']}-{previous['end']} and "
                f"{current['start']}-{current['end']}."
            )
            raise ValueError(msg)
        if current["start"] != previous["end"] + 1:
            msg = (
                "Land-cover periods have a gap between "
                f"{previous['end']} and {current['start']}."
            )
            raise ValueError(msg)
    return manifest, periods


def _land_cover_encoding():
    return {
        "land_cover": {
            "zlib": True,
            "complevel": 4,
            "shuffle": True,
            "_FillValue": int(NO_DATA),
            "dtype": "int32",
        }
    }


def _period_dataset(datasets, periods):
    """Combine aligned period datasets with an explicit CF time axis."""
    times = np.array(
        [np.datetime64(f"{period['start']:04d}-01-01", "ns") for period in periods]
    )
    merged = xr.concat(
        datasets,
        dim=xr.IndexVariable("time", times),
        data_vars=["land_cover"],
        coords="minimal",
        compat="override",
    )
    bounds = np.array(
        [
            [
                np.datetime64(f"{period['start']:04d}-01-01", "ns"),
                np.datetime64(f"{period['end'] + 1:04d}-01-01", "ns"),
            ]
            for period in periods
        ]
    )
    merged["time_bnds"] = xr.DataArray(bounds, dims=("time", "bnds"))
    merged["time"].attrs.update(
        {
            "standard_name": "time",
            "long_name": "land-cover period start",
            "axis": "T",
            "bounds": "time_bnds",
        }
    )
    merged["time_bnds"].attrs["long_name"] = "land-cover period bounds"
    merged["land_cover"].attrs.update(
        {"long_name": "mHM land cover", "units": "1", "nodata_value": int(NO_DATA)}
    )
    merged.attrs.update(
        {
            "title": "Temporal land-cover classes for mHM",
            "source": "mhm-tools format-data land-cover manifest",
        }
    )
    return merged


def format_lc_periods(
    input_path: PathLike,
    dem_file: PathLike,
    output_path: PathLike,
    lookup_table: PathLike,
    mapping_field: str,
    class_field: str,
    output_type: str = "nc",
    *,
    input_crs: str | None = None,
    dem_crs: str | None = None,
    resampling="auto",
) -> tuple[Path, ...]:
    """Format historical land-cover rasters listed by a directory manifest."""
    output_type = str(output_type).lower().lstrip(".")
    if output_type not in {"asc", "nc"}:
        msg = "Historical land-cover output extension must be 'asc' or 'nc'."
        raise ValueError(msg)
    manifest, periods = _read_lc_manifest(input_path)
    dem_file = Path(dem_file)
    lookup_table = Path(lookup_table)
    output_path = Path(output_path)
    if not dem_file.is_file():
        msg = f"DEM raster does not exist: {dem_file}"
        raise ValueError(msg)
    if output_path.exists() and not output_path.is_dir():
        msg = f"Output path must be a directory: {output_path}"
        raise ValueError(msg)
    table = read_lookup_table(lookup_table)

    reference = get_raster_data(dem_file, crs=dem_crs)
    try:
        datasets = [
            prepare_categorical_data(
                period["path"],
                reference,
                table,
                mapping_field,
                class_field,
                variable_name="land_cover",
                input_crs=input_crs,
                resampling=resampling,
                mask_reference=True,
            )
            for period in periods
        ]
        protected = {
            manifest.resolve(),
            dem_file.resolve(),
            lookup_table.resolve(),
            *(period["path"].resolve() for period in periods),
        }
        if output_type == "asc":
            outputs = tuple(
                output_path / f"lc_{period['start']}_{period['end']}.asc"
                for period in periods
            )
            collisions = [output for output in outputs if output.resolve() in protected]
            if collisions:
                msg = (
                    "Land-cover outputs must differ from all input files: "
                    + ", ".join(str(path) for path in collisions)
                )
                raise ValueError(msg)
            output_path.mkdir(parents=True, exist_ok=True)
            for dataset, output in zip(datasets, outputs):
                write_xarray_to_file(
                    dataset,
                    output,
                    var_name="land_cover",
                    crs=reference.rio.crs,
                )
            return outputs

        output = output_path / "lc_periods.nc"
        if output.resolve() in protected:
            msg = f"Land-cover output must differ from all input files: {output}"
            raise ValueError(msg)
        output_path.mkdir(parents=True, exist_ok=True)
        merged = _period_dataset(datasets, periods)
        write_xarray_to_file(
            merged,
            output,
            var_name="land_cover",
            encoding=_land_cover_encoding(),
            crs=reference.rio.crs,
        )
        return (output,)
    finally:
        reference.close()
