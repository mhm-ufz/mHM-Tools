"""
Prepare mHM land-cover rasters from a single map or a manifest of periods.

Two pipelines live here. The single-raster one maps one categorical raster
through a lookup table onto the DEM grid. The historical one reads a manifest
of non-overlapping, gapless year ranges and formats each period, writing
either one ASCII raster per period or a single CF time stack as NetCDF.

Both share the same two-step placement: categories are reclassified to mHM
land-cover classes at the source resolution first, then that integer raster is
warped onto the DEM grid, so no interpolation ever mixes class numbers. The
sections below follow that flow, and the public entry points come last.

Authors
-------
- Sanjeev Bashyal
"""

from __future__ import annotations

import datetime as dt
import logging
from contextlib import ExitStack
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Union

import dask
import dask.array as da
import numpy as np
import pandas as pd
import rasterio
import rioxarray
import xarray as xr
from netCDF4 import date2num
from rasterio.crs import CRS
from rasterio.enums import Resampling
from rasterio.vrt import WarpedVRT
from rasterio.windows import Window

from mhm_tools.common.constants import NO_DATA
from mhm_tools.common.file_handler import write_xarray_to_file
from mhm_tools.common.format_data import (
    fill_grid_nodata,
    format_categorical_data,
    get_categorical_output_path,
)
from mhm_tools.common.lookup_handler import (
    _lookup_mapping,
    _normalise_field_name,
    read_format_manifest,
    read_lookup_table,
)

logger = logging.getLogger(__name__)

PathLike = Union[str, Path]


# Land-cover period manifest


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


def _period_datetime(value, label: str) -> dt.datetime:
    """Parse a year or ISO datetime value into a naive period boundary."""
    text = str(value).strip()
    if len(text) == 4 and text.isdigit():
        text += "-01-01"
    try:
        result = dt.datetime.fromisoformat(text)
    except ValueError as error:
        msg = f"{label} must be a year or ISO datetime: {value!r}"
        raise ValueError(msg) from error
    if result.tzinfo is not None:
        msg = f"{label} must use a timezone-free UTC datetime."
        raise ValueError(msg)
    return result


def _read_lc_manifest(input_file: PathLike):
    """Read continuous [start, end) dates, retaining inclusive legacy year CSVs."""
    manifest, table = read_format_manifest(input_file, ("FilePath",))
    normalized_columns = {
        _normalise_field_name(column): column for column in table.columns
    }
    legacy = {"startyear", "endyear"}.issubset(normalized_columns)
    labels = ("StartYear", "EndYear") if legacy else ("StartDateTime", "EndDateTime")
    columns = tuple(
        normalized_columns.get(_normalise_field_name(label)) for label in labels
    )
    if any(column is None for column in columns):
        msg = "Manifest requires StartDateTime/EndDateTime or StartYear/EndYear."
        raise ValueError(msg)
    periods = []
    for row_number, row in enumerate(table.to_dict("records"), start=2):
        start = _period_datetime(row[columns[0]], labels[0])
        end = _period_datetime(row[columns[1]], labels[1])
        if legacy:
            end = dt.datetime(end.year + 1, 1, 1)
        if start >= end:
            msg = f"Manifest row {row_number} must end after it starts."
            raise ValueError(msg)
        periods.append(
            {
                "start": start,
                "end": end,
                "path": _manifest_raster_path(manifest, row["FilePath"], row_number),
            }
        )
    for previous, current in zip(periods, periods[1:]):
        if current["start"] != previous["end"]:
            msg = "Land-cover periods must be ordered, continuous and non-overlapping."
            raise ValueError(msg)
    return manifest, periods


# Raster grid access


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
    """Yield full-width row blocks covering a raster."""
    for row in range(0, dataset.height, _LC_BLOCK_ROWS):
        height = min(_LC_BLOCK_ROWS, dataset.height - row)
        yield Window(0, row, dataset.width, height)


def _grid_coordinates(dataset):
    """Return the cell-centre coordinates of a north-up raster grid."""
    transform = dataset.transform
    x = transform.c + (np.arange(dataset.width) + 0.5) * transform.a
    y = transform.f + (np.arange(dataset.height) + 0.5) * transform.e
    return x, y


def _reference_valid(reference, window=None):
    """Return the boolean mask of DEM cells that carry data."""
    values = reference.read(1, window=window, masked=True)
    valid = ~np.ma.getmaskarray(values)
    data = np.ma.getdata(values)
    if np.issubdtype(data.dtype, np.floating):
        valid &= np.isfinite(data)
    return valid


def _validate_reference_grid(reference):
    """Reject a rotated DEM, whose cells do not align to the output axes."""
    if reference.transform.b or reference.transform.d:
        msg = "Rotated DEM grids are not supported for historical land cover."
        raise ValueError(msg)


# Reclassification and alignment


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
                        msg = (
                            "The DEM has no valid cell to define the land-cover domain."
                        )
                        raise ValueError(msg)
                    values = warped.read(1, out_dtype="int32")
                    missing = int(np.count_nonzero((values == int(NO_DATA)) & valid))
                    x, y = _grid_coordinates(reference)
                    filled = fill_grid_nodata(
                        values,
                        x=x,
                        y=y,
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


# Period writers


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
            with rioxarray.open_rasterio(
                aligned_path, chunks={"band": 1, "y": 256, "x": 256}
            ) as raster:
                write_xarray_to_file(
                    raster.squeeze("band", drop=True), output_path, crs=reference_crs
                )
    return output_path


def _create_lc_dataset(reference, reference_crs, periods, values):
    """Return a CF Dataset from reference, reference_crs, periods and lazy values."""
    transform = reference.transform
    x_values, y_values = _grid_coordinates(reference)
    units = "days since 1970-01-01 00:00:00"
    calendar = "proleptic_gregorian"
    starts = np.asarray(
        date2num([period["start"] for period in periods], units, calendar=calendar),
        dtype="float64",
    )
    ends = np.asarray(
        date2num([period["end"] for period in periods], units, calendar=calendar),
        dtype="float64",
    )
    geographic = reference_crs.is_geographic
    return xr.Dataset(
        data_vars={
            "land_cover": (
                ("time", "y", "x"),
                values,
                {
                    "long_name": "mHM land cover",
                    "units": "1",
                    "nodata_value": int(NO_DATA),
                    "grid_mapping": "crs",
                },
            ),
            "time_bnds": (
                ("time", "bnds"),
                np.column_stack((starts, ends)),
                {
                    "long_name": "land-cover period bounds",
                    "units": units,
                    "calendar": calendar,
                },
            ),
            "x_bnds": (
                ("x", "bnds"),
                np.column_stack(
                    (
                        x_values - abs(transform.a) / 2,
                        x_values + abs(transform.a) / 2,
                    )
                ),
            ),
            "y_bnds": (
                ("y", "bnds"),
                np.column_stack(
                    (
                        y_values - abs(transform.e) / 2,
                        y_values + abs(transform.e) / 2,
                    )
                ),
            ),
            "crs": (
                (),
                np.int32(0),
                {
                    "spatial_ref": reference_crs.to_wkt(),
                    "crs_wkt": reference_crs.to_wkt(),
                    "GeoTransform": " ".join(
                        str(value) for value in transform.to_gdal()
                    ),
                },
            ),
        },
        coords={
            "time": (
                "time",
                starts,
                {
                    "standard_name": "time",
                    "long_name": "land-cover period start",
                    "axis": "T",
                    "bounds": "time_bnds",
                    "units": units,
                    "calendar": calendar,
                },
            ),
            "x": (
                "x",
                x_values,
                {
                    "standard_name": (
                        "longitude" if geographic else "projection_x_coordinate"
                    ),
                    "units": "degrees_east" if geographic else "m",
                    "axis": "X",
                    "bounds": "x_bnds",
                },
            ),
            "y": (
                "y",
                y_values,
                {
                    "standard_name": (
                        "latitude" if geographic else "projection_y_coordinate"
                    ),
                    "units": "degrees_north" if geographic else "m",
                    "axis": "Y",
                    "bounds": "y_bnds",
                },
            ),
        },
        attrs={
            "Conventions": "CF-1.8",
            "title": "Temporal land-cover classes for mHM",
            "source": "mhm-tools format-data land-cover manifest",
        },
    )


def _format_lc_periods_netcdf_streaming(
    periods,
    dem_file,
    output,
    mapping,
    *,
    input_crs=None,
    dem_crs=None,
    resampling="auto",
    fill_nodata=True,
    compression=None,
):
    """Write a CF time stack without retaining full period arrays."""
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(
        prefix="mhm_tools_lc_stack_", dir=output.parent
    ) as temp_name, ExitStack() as stack:
        reference = stack.enter_context(rasterio.open(dem_file))
        reference_crs = _raster_crs(reference, dem_crs, "DEM")
        _validate_reference_grid(reference)
        period_arrays = []
        for index, period in enumerate(periods):
            aligned = Path(temp_name) / f"period_{index}.tif"
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
            raster = stack.enter_context(
                rioxarray.open_rasterio(aligned, chunks={"band": 1, "y": 256, "x": 256})
            )
            period_arrays.append(raster.squeeze("band", drop=True).data)
        dataset = _create_lc_dataset(
            reference, reference_crs, periods, da.stack(period_arrays)
        )
        temporary_file = Path(temp_name) / output.name
        with dask.config.set(scheduler="synchronous"):
            write_xarray_to_file(
                dataset,
                temporary_file,
                compression=compression,
                encoding={"land_cover": {"_FillValue": int(NO_DATA)}},
            )
        temporary_file.replace(output)
    return output


# Public entry points


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
    compression=None,
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
        compression=compression,
    )


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
    compression=None,
) -> tuple[Path, ...]:
    """Format historical land-cover rasters listed by a manifest.

    StartDateTime/EndDateTime accept ISO datetimes or years with matching
    adjacent boundaries. Legacy StartYear/EndYear remain inclusive whole years.

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
    protected = {
        manifest.resolve(),
        dem_file.resolve(),
        lookup_table.resolve(),
        *(period["path"].resolve() for period in periods),
    }

    if output_type == "asc":
        outputs = tuple(
            output_path
            / (
                f"lc_{period['start'].year}_{period['end'].year - 1}.asc"
                if period["start"] == dt.datetime(period["start"].year, 1, 1)
                and period["end"] == dt.datetime(period["end"].year, 1, 1)
                else f"lc_{period['start']:%Y%m%dT%H%M%S}_{period['end']:%Y%m%dT%H%M%S}.asc"
            )
            for period in periods
        )
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
            compression=compression,
        ),
    )
