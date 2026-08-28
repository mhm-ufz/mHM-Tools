"""
Prepare an mHM land-cover raster from categorical raster data.

Authors
-------
- Sanjeev Bashyal
"""

from __future__ import annotations

import logging
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Union

import numpy as np
import pandas as pd
import rasterio
from rasterio.crs import CRS
from rasterio.enums import Resampling
from rasterio.shutil import copy as copy_raster
from rasterio.vrt import WarpedVRT
from rasterio.windows import Window

from mhm_tools.common.constants import NO_DATA
from mhm_tools.common.format_data import (
    fill_grid_nodata,
    format_categorical_data,
    get_categorical_output_path,
)
from mhm_tools.common.lookup_handler import (
    _lookup_mapping,
    _required_integer,
    read_format_manifest,
    read_lookup_table,
)

logger = logging.getLogger(__name__)

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
    fill_nodata: bool = True,
) -> Path:
    """Map a categorical raster to mHM land-cover classes on the DEM grid.

    Parameters
    ----------
    fill_nodata : bool, default True
        Restrict the output to the DEM domain and take its remaining nodata
        cells from the nearest classified neighbour.
    """
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
        fill_nodata=fill_nodata,
    )


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


def _read_lc_manifest(input_file: PathLike):
    """Read validated, chronological land-cover period records."""
    manifest, table = read_format_manifest(
        input_file,
        ("StartYear", "EndYear", "FilePath"),
    )
    periods = []
    for row_number, row in enumerate(table.itertuples(index=False), start=2):
        start = _required_integer(
            row.StartYear, "StartYear", row_number, "Manifest"
        )
        end = _required_integer(row.EndYear, "EndYear", row_number, "Manifest")
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


_LC_BLOCK_ROWS = 256


def _raster_crs(dataset, override, label: str) -> CRS:
    """Return raster CRS metadata or its explicit fallback."""
    crs = dataset.crs or (CRS.from_user_input(override) if override else None)
    if crs is None:
        msg = f"{label} raster has no CRS metadata: {dataset.name}"
        raise ValueError(msg)
    return crs


def _lc_resampling(requested) -> Resampling:
    """Return categorical resampling for a Rasterio virtual warp."""
    if isinstance(requested, Resampling):
        return requested
    method = str(getattr(requested, "name", requested)).strip().lower()
    method = {"auto": "mode", "near": "nearest"}.get(method, method)
    try:
        return Resampling[method]
    except KeyError as exc:
        choices = "nearest, bilinear, cubic, average, or mode"
        msg = f"Unsupported raster resampling method {requested!r}; expected {choices}."
        raise ValueError(msg) from exc


def _lc_windows(dataset):
    for row in range(0, dataset.height, _LC_BLOCK_ROWS):
        height = min(_LC_BLOCK_ROWS, dataset.height - row)
        yield Window(0, row, dataset.width, height)


def _reference_valid(reference, window=None):
    values = reference.read(1, window=window, masked=True)
    valid = ~np.ma.getmaskarray(values)
    data = np.ma.getdata(values)
    if np.issubdtype(data.dtype, np.floating):
        valid &= np.isfinite(data)
    return valid


def _write_mapped_lc(source, source_crs, mapped_path, mapping):
    """Map source categories to int32 classes before spatial resampling."""
    profile = {
        "driver": "GTiff",
        "height": source.height,
        "width": source.width,
        "count": 1,
        "dtype": "int32",
        "crs": source_crs,
        "transform": source.transform,
        "nodata": int(NO_DATA),
        "compress": "lzw",
        "tiled": True,
        "blockxsize": 256,
        "blockysize": 256,
        "BIGTIFF": "IF_SAFER",
    }
    matched = False
    with rasterio.open(mapped_path, "w", **profile) as target:
        for _, window in source.block_windows(1):
            source_values = source.read(1, window=window, masked=True)
            values = np.ma.getdata(source_values)
            valid = ~np.ma.getmaskarray(source_values) & np.isfinite(values)
            result = np.full(values.shape, int(NO_DATA), dtype=np.int32)
            for source_value, class_value in mapping.items():
                selected = valid & (values == source_value)
                if np.any(selected):
                    result[selected] = int(class_value)
                    matched = True
            target.write(result, 1, window=window)
    if not matched:
        msg = "No valid raster category matched the selected lookup mapping field."
        raise ValueError(msg)


def _write_aligned_lc_period(
    source_path,
    reference,
    reference_crs,
    aligned_path,
    mapping,
    *,
    input_crs,
    resampling,
    fill_nodata,
):
    """Map and align one land-cover period to the exact DEM grid."""
    mapped_path = aligned_path.with_name(f"{aligned_path.stem}_mapped.tif")
    with rasterio.open(source_path) as source:
        if source.count != 1:
            msg = "Land-cover inputs must contain exactly one raster band."
            raise ValueError(msg)
        source_crs = _raster_crs(source, input_crs, "Land-cover")
        _write_mapped_lc(source, source_crs, mapped_path, mapping)

    profile = {
        "driver": "GTiff",
        "height": reference.height,
        "width": reference.width,
        "count": 1,
        "dtype": "int32",
        "crs": reference_crs,
        "transform": reference.transform,
        "nodata": int(NO_DATA),
        "compress": "lzw",
        "tiled": True,
        "blockxsize": 256,
        "blockysize": 256,
        "BIGTIFF": "IF_SAFER",
    }
    with rasterio.open(mapped_path) as mapped:  # noqa: SIM117
        with WarpedVRT(
            mapped,
            crs=reference_crs,
            transform=reference.transform,
            width=reference.width,
            height=reference.height,
            src_nodata=int(NO_DATA),
            nodata=int(NO_DATA),
            dtype="int32",
            resampling=_lc_resampling(resampling),
        ) as warped:
            with rasterio.open(aligned_path, "w", **profile) as target:
                if fill_nodata:
                    valid = _reference_valid(reference)
                    if not np.any(valid):
                        msg = "The DEM has no valid cell to define the land-cover domain."
                        raise ValueError(msg)
                    values = warped.read(1, out_dtype="int32")
                    missing = int(
                        np.count_nonzero((values == int(NO_DATA)) & valid)
                    )
                    transform = reference.transform
                    filled = fill_grid_nodata(
                        values,
                        x=transform.c
                        + (np.arange(reference.width) + 0.5) * transform.a,
                        y=transform.f
                        + (np.arange(reference.height) + 0.5) * transform.e,
                        mask=~valid,
                        missing_value=float(NO_DATA),
                        fill_value=int(NO_DATA),
                        name="land_cover",
                        source_file=source_path,
                    )
                    if missing:
                        logger.info(
                            "Filled %d of %d nodata cells of %s from nearest "
                            "valid neighbours.",
                            filled,
                            missing,
                            source_path,
                        )
                    target.write(values, 1)
                else:
                    for window in _lc_windows(reference):
                        values = warped.read(1, window=window, out_dtype="int32")
                        values[~_reference_valid(reference, window)] = int(NO_DATA)
                        target.write(values, 1, window=window)
    return aligned_path


def _validate_reference_grid(reference):
    if reference.transform.b or reference.transform.d:
        msg = "Rotated DEM grids are not supported for historical land cover."
        raise ValueError(msg)


def _format_lc_period_asc_streaming(
    source_path,
    dem_path,
    output_path,
    mapping,
    *,
    input_crs=None,
    dem_crs=None,
    resampling="auto",
    fill_nodata=True,
):
    """Reclassify and align one period with Rasterio and bounded memory."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(dem_path) as reference:
        reference_crs = _raster_crs(reference, dem_crs, "DEM")
        _validate_reference_grid(reference)
        with TemporaryDirectory(
            prefix="mhm_tools_lc_", dir=output_path.parent
        ) as temp_name:
            aligned_path = Path(temp_name) / "aligned.tif"
            _write_aligned_lc_period(
                source_path,
                reference,
                reference_crs,
                aligned_path,
                mapping,
                input_crs=input_crs,
                resampling=resampling,
                fill_nodata=fill_nodata,
            )
            copy_raster(
                aligned_path,
                output_path,
                driver="AAIGrid",
                DECIMAL_PRECISION=0,
            )
    return output_path


def _format_lc_periods_netcdf_streaming(  # noqa: PLR0915
    periods,
    dem_file,
    output,
    mapping,
    *,
    input_crs=None,
    dem_crs=None,
    resampling="auto",
    fill_nodata=True,
):
    """Write a CF time stack without retaining full period arrays."""
    import datetime as dt

    import netCDF4

    reference = rasterio.open(dem_file)
    reference_crs = _raster_crs(reference, dem_crs, "DEM")
    _validate_reference_grid(reference)
    transform = reference.transform
    rows, cols = reference.height, reference.width
    projection = reference_crs.to_wkt()
    geographic = reference_crs.is_geographic
    x_values = transform.c + (np.arange(cols) + 0.5) * transform.a
    y_values = transform.f + (np.arange(rows) + 0.5) * transform.e
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(
        prefix="mhm_tools_lc_stack_", dir=output.parent
    ) as temp_name:
        temp = Path(temp_name)
        dataset = netCDF4.Dataset(output, "w", format="NETCDF4")
        try:
            dataset.createDimension("time", len(periods))
            dataset.createDimension("y", rows)
            dataset.createDimension("x", cols)
            dataset.createDimension("bnds", 2)
            time = dataset.createVariable("time", "f8", ("time",))
            time_bounds = dataset.createVariable("time_bnds", "f8", ("time", "bnds"))
            x = dataset.createVariable("x", "f8", ("x",))
            y = dataset.createVariable("y", "f8", ("y",))
            x_bounds = dataset.createVariable("x_bnds", "f8", ("x", "bnds"))
            y_bounds = dataset.createVariable("y_bnds", "f8", ("y", "bnds"))
            crs = dataset.createVariable("crs", "i4")
            land_cover = dataset.createVariable(
                "land_cover",
                "i4",
                ("time", "y", "x"),
                fill_value=int(NO_DATA),
                zlib=True,
                complevel=4,
                shuffle=True,
                chunksizes=(1, min(256, rows), min(256, cols)),
            )
            units = "days since 1970-01-01 00:00:00"
            calendar = "proleptic_gregorian"
            starts = [dt.datetime(period["start"], 1, 1) for period in periods]
            ends = [dt.datetime(period["end"] + 1, 1, 1) for period in periods]
            time[:] = netCDF4.date2num(starts, units, calendar=calendar)
            time_bounds[:, 0] = netCDF4.date2num(starts, units, calendar=calendar)
            time_bounds[:, 1] = netCDF4.date2num(ends, units, calendar=calendar)
            time.units = units
            time.calendar = calendar
            time.standard_name = "time"
            time.long_name = "land-cover period start"
            time.axis = "T"
            time.bounds = "time_bnds"
            time_bounds.units = units
            time_bounds.calendar = calendar
            time_bounds.long_name = "land-cover period bounds"
            x[:] = x_values
            y[:] = y_values
            x_bounds[:, 0] = x_values - abs(transform.a) / 2.0
            x_bounds[:, 1] = x_values + abs(transform.a) / 2.0
            y_bounds[:, 0] = y_values - abs(transform.e) / 2.0
            y_bounds[:, 1] = y_values + abs(transform.e) / 2.0
            x.standard_name = "longitude" if geographic else "projection_x_coordinate"
            y.standard_name = "latitude" if geographic else "projection_y_coordinate"
            x.units = "degrees_east" if geographic else "m"
            y.units = "degrees_north" if geographic else "m"
            x.axis = "X"
            y.axis = "Y"
            x.bounds = "x_bnds"
            y.bounds = "y_bnds"
            if projection:
                crs.spatial_ref = projection
                crs.crs_wkt = projection
            crs.GeoTransform = " ".join(
                str(value) for value in transform.to_gdal()
            )
            land_cover.long_name = "mHM land cover"
            land_cover.units = "1"
            land_cover.nodata_value = int(NO_DATA)
            land_cover.grid_mapping = "crs"
            dataset.Conventions = "CF-1.8"
            dataset.title = "Temporal land-cover classes for mHM"
            dataset.source = "mhm-tools format-data land-cover manifest"

            for index, period in enumerate(periods):
                aligned = temp / f"period_{index}.tif"
                _write_aligned_lc_period(
                    period["path"],
                    reference,
                    reference_crs,
                    aligned,
                    mapping,
                    input_crs=input_crs,
                    resampling=resampling,
                    fill_nodata=fill_nodata,
                )
                with rasterio.open(aligned) as raster:
                    for window in _lc_windows(raster):
                        row = int(window.row_off)
                        height = int(window.height)
                        land_cover[index, row : row + height, :] = raster.read(
                            1, window=window
                        )
        finally:
            dataset.close()
            reference.close()
    return output


def format_lc_periods(
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
    fill_nodata: bool = True,
) -> tuple[Path, ...]:
    """Format historical land-cover rasters listed by a manifest.

    Parameters
    ----------
    fill_nodata : bool, default True
        Take the nodata cells left inside the DEM domain from their nearest
        classified neighbour, so that gaps in one period do not create holes
        in the formatted land cover.
    """
    output_type = str(output_type).lower().lstrip(".")
    if output_type not in {"asc", "nc"}:
        msg = "Historical land-cover output extension must be 'asc' or 'nc'."
        raise ValueError(msg)
    manifest, periods = _read_lc_manifest(input_file)
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

    if output_type == "asc":
        outputs = tuple(
            output_path / f"lc_{period['start']}_{period['end']}.asc"
            for period in periods
        )
        protected = {
            manifest.resolve(),
            dem_file.resolve(),
            lookup_table.resolve(),
            *(period["path"].resolve() for period in periods),
        }
        collisions = [output for output in outputs if output.resolve() in protected]
        if collisions:
            msg = "Land-cover outputs must differ from all input files: " + ", ".join(
                str(path) for path in collisions
            )
            raise ValueError(msg)
        output_path.mkdir(parents=True, exist_ok=True)
        mapping = _lookup_mapping(table, mapping_field, class_field)
        for period, output in zip(periods, outputs):
            _format_lc_period_asc_streaming(
                period["path"],
                dem_file,
                output,
                mapping,
                input_crs=input_crs,
                dem_crs=dem_crs,
                resampling=resampling,
                fill_nodata=fill_nodata,
            )
        return outputs

    output = output_path / "lc_periods.nc"
    protected = {
        manifest.resolve(),
        dem_file.resolve(),
        lookup_table.resolve(),
        *(period["path"].resolve() for period in periods),
    }
    if output.resolve() in protected:
        msg = f"Land-cover output must differ from all input files: {output}"
        raise ValueError(msg)
    mapping = _lookup_mapping(table, mapping_field, class_field)
    return (
        _format_lc_periods_netcdf_streaming(
            periods,
            dem_file,
            output,
            mapping,
            input_crs=input_crs,
            dem_crs=dem_crs,
            resampling=resampling,
            fill_nodata=fill_nodata,
        ),
    )
