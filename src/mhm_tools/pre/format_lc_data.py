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
    """Reclassify and align one large period with bounded memory.

    Cells outside the DEM domain are always written as nodata. When
    ``fill_nodata`` is true the nodata cells left inside the DEM domain are
    taken from their nearest classified neighbour, which requires the aligned
    period to be held in memory once.
    """
    from osgeo import gdal, osr

    source = gdal.Open(str(source_path))
    reference = gdal.Open(str(dem_path))
    if source is None or reference is None:
        raise ValueError("Land-cover source and DEM must be GDAL-readable rasters.")
    if source.RasterCount != 1:
        raise ValueError("Land-cover inputs must contain exactly one raster band.")
    transform = reference.GetGeoTransform()
    if transform[2] or transform[4]:
        raise ValueError("Rotated DEM grids cannot be written as mHM ASCII files.")
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix="mhm_tools_lc_", dir=output_path.parent) as temp:
        temp = Path(temp)
        mapped_path = temp / "mapped.tif"
        aligned_path = temp / "aligned.tif"
        driver = gdal.GetDriverByName("GTiff")
        mapped = driver.Create(
            str(mapped_path),
            source.RasterXSize,
            source.RasterYSize,
            1,
            gdal.GDT_Int32,
            options=["TILED=YES", "COMPRESS=LZW", "BIGTIFF=IF_SAFER"],
        )
        mapped.SetGeoTransform(source.GetGeoTransform())
        projection = source.GetProjection()
        if not projection and input_crs:
            spatial_ref = osr.SpatialReference()
            if spatial_ref.SetFromUserInput(str(input_crs)) == 0:
                projection = spatial_ref.ExportToWkt()
        if projection:
            mapped.SetProjection(projection)
        source_band = source.GetRasterBand(1)
        mapped_band = mapped.GetRasterBand(1)
        mapped_band.SetNoDataValue(int(NO_DATA))
        nodata = source_band.GetNoDataValue()
        block_x, block_y = source_band.GetBlockSize()
        block_x = max(1, min(source.RasterXSize, block_x or 1024))
        block_y = max(1, min(source.RasterYSize, block_y or 256))
        matched = False
        for yoff in range(0, source.RasterYSize, block_y):
            height = min(block_y, source.RasterYSize - yoff)
            for xoff in range(0, source.RasterXSize, block_x):
                width = min(block_x, source.RasterXSize - xoff)
                values = source_band.ReadAsArray(xoff, yoff, width, height)
                valid = np.isfinite(values)
                if nodata is not None and np.isfinite(nodata):
                    valid &= values != nodata
                result = np.full(values.shape, int(NO_DATA), dtype=np.int32)
                for source_value, class_value in mapping.items():
                    selected = valid & (values == source_value)
                    if np.any(selected):
                        result[selected] = int(class_value)
                        matched = True
                mapped_band.WriteArray(result, xoff, yoff)
        mapped_band.FlushCache()
        mapped_band = None
        mapped = None
        source_band = None
        source = None
        if not matched:
            raise ValueError(
                "No valid raster category matched the selected lookup mapping field."
            )

        min_x = transform[0]
        max_y = transform[3]
        max_x = min_x + transform[1] * reference.RasterXSize
        min_y = max_y + transform[5] * reference.RasterYSize
        target_projection = reference.GetProjection()
        if not target_projection and dem_crs:
            spatial_ref = osr.SpatialReference()
            if spatial_ref.SetFromUserInput(str(dem_crs)) == 0:
                target_projection = spatial_ref.ExportToWkt()
        method = getattr(resampling, "name", resampling)
        method = str(method).lower()
        method = "mode" if method == "auto" else method
        if method not in {"near", "nearest", "bilinear", "cubic", "average", "mode"}:
            raise ValueError(f"Unsupported raster resampling method: {resampling}")
        if method == "nearest":
            method = "near"
        warped = gdal.Warp(
            str(aligned_path),
            str(mapped_path),
            format="GTiff",
            outputBounds=(min_x, min_y, max_x, max_y),
            width=reference.RasterXSize,
            height=reference.RasterYSize,
            srcSRS=projection or None,
            dstSRS=target_projection or None,
            srcNodata=int(NO_DATA),
            dstNodata=int(NO_DATA),
            resampleAlg=method,
            multithread=False,
            creationOptions=["TILED=YES", "COMPRESS=LZW", "BIGTIFF=IF_SAFER"],
        )
        if warped is None:
            raise RuntimeError("Could not align the land-cover period to the DEM grid.")
        warped = None

        aligned = gdal.Open(str(aligned_path), gdal.GA_Update)
        aligned_band = aligned.GetRasterBand(1)
        dem_band = reference.GetRasterBand(1)
        dem_nodata = dem_band.GetNoDataValue()
        if fill_nodata:
            dem_values = dem_band.ReadAsArray()
            valid = np.isfinite(dem_values)
            if dem_nodata is not None and np.isfinite(dem_nodata):
                valid &= dem_values != dem_nodata
            if not np.any(valid):
                raise ValueError(
                    "The DEM has no valid cell to define the land-cover domain."
                )
            values = aligned_band.ReadAsArray()
            missing = int(np.count_nonzero((values == int(NO_DATA)) & valid))
            filled = fill_grid_nodata(
                values,
                x=transform[0] + (np.arange(reference.RasterXSize) + 0.5) * transform[1],
                y=transform[3] + (np.arange(reference.RasterYSize) + 0.5) * transform[5],
                mask=~valid,
                missing_value=float(NO_DATA),
                fill_value=int(NO_DATA),
                name="land_cover",
                source_file=source_path,
            )
            if missing:
                logger.info(
                    "Filled %d of %d nodata cells of %s from nearest valid "
                    "neighbours.",
                    filled,
                    missing,
                    source_path,
                )
            aligned_band.WriteArray(values, 0, 0)
        else:
            _, block_y = dem_band.GetBlockSize()
            block_y = max(1, min(reference.RasterYSize, block_y or 256))
            for yoff in range(0, reference.RasterYSize, block_y):
                height = min(block_y, reference.RasterYSize - yoff)
                dem_values = dem_band.ReadAsArray(
                    0, yoff, reference.RasterXSize, height
                )
                valid = np.isfinite(dem_values)
                if dem_nodata is not None and np.isfinite(dem_nodata):
                    valid &= dem_values != dem_nodata
                values = aligned_band.ReadAsArray(
                    0, yoff, reference.RasterXSize, height
                )
                values[~valid] = int(NO_DATA)
                aligned_band.WriteArray(values, 0, yoff)
        aligned_band.FlushCache()
        aligned_band = None
        aligned = None
        dem_band = None
        reference = None
        translated = gdal.Translate(
            str(output_path),
            str(aligned_path),
            format="AAIGrid",
            creationOptions=["DECIMAL_PRECISION=0"],
        )
        if translated is None:
            raise RuntimeError(f"Could not write land-cover ASCII file: {output_path}")
        translated = None
    return output_path


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
):
    """Write a CF time stack without retaining full period arrays."""
    import datetime as dt

    import netCDF4
    from osgeo import gdal, osr

    reference = gdal.Open(str(dem_file))
    if reference is None:
        raise ValueError(f"DEM raster does not exist: {dem_file}")
    transform = reference.GetGeoTransform()
    if transform[2] or transform[4]:
        raise ValueError("Rotated DEM grids cannot be written as a CF rectilinear grid.")
    rows, cols = reference.RasterYSize, reference.RasterXSize
    projection = reference.GetProjection()
    spatial_ref = osr.SpatialReference()
    geographic = bool(
        projection
        and spatial_ref.ImportFromWkt(projection) == 0
        and spatial_ref.IsGeographic()
    )
    x_values = transform[0] + (np.arange(cols) + 0.5) * transform[1]
    y_values = transform[3] + (np.arange(rows) + 0.5) * transform[5]
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix="mhm_tools_lc_stack_", dir=output.parent) as temp:
        temp = Path(temp)
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
            x_bounds[:, 0] = x_values - abs(transform[1]) / 2.0
            x_bounds[:, 1] = x_values + abs(transform[1]) / 2.0
            y_bounds[:, 0] = y_values - abs(transform[5]) / 2.0
            y_bounds[:, 1] = y_values + abs(transform[5]) / 2.0
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
            crs.GeoTransform = " ".join(str(value) for value in transform)
            land_cover.long_name = "mHM land cover"
            land_cover.units = "1"
            land_cover.nodata_value = int(NO_DATA)
            land_cover.grid_mapping = "crs"
            dataset.Conventions = "CF-1.8"
            dataset.title = "Temporal land-cover classes for mHM"
            dataset.source = "mhm-tools format-data land-cover manifest"

            for index, period in enumerate(periods):
                aligned = temp / f"period_{index}.asc"
                _format_lc_period_asc_streaming(
                    period["path"],
                    dem_file,
                    aligned,
                    mapping,
                    input_crs=input_crs,
                    dem_crs=dem_crs,
                    resampling=resampling,
                    fill_nodata=fill_nodata,
                )
                raster = gdal.Open(str(aligned))
                band = raster.GetRasterBand(1)
                block_rows = min(256, rows)
                for yoff in range(0, rows, block_rows):
                    height = min(block_rows, rows - yoff)
                    land_cover[index, yoff : yoff + height, :] = band.ReadAsArray(
                        0, yoff, cols, height
                    )
                band = None
                raster = None
        finally:
            dataset.close()
            reference = None
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
