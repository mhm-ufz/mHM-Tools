"""Prepare categorical and gridded LAI data for mHM.

Two independent pipelines live here. The categorical one maps a land-cover
raster to LAI classes and writes the companion ``LAI_classdefinition.txt``.
The gridded one aggregates a LAI NetCDF in time and places it on the model
grid, streaming one block of target rows and one time step at a time so that
peak memory never grows with the record length or the grid size.

The sections below follow the gridded data flow: aggregate in time, find the
source variable, build the target grid, refuse a request that cannot fit on
disk, sample the source onto row blocks, and write. The two gridded pipelines,
the categorical class definitions, and the public entry points follow.

Authors
-------
- Sanjeev Bashyal
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import NamedTuple, Union

from mhm_tools.common.format_data import (
    format_categorical_data,
    get_categorical_output_path,
)
from mhm_tools.common.lookup_handler import (
    _is_blank,
    _required_integer,
    _required_number,
    _resolve_field,
    read_lookup_table,
)

logger = logging.getLogger(__name__)

PathLike = Union[str, Path]


# Writer tuning and nodata conventions


NODATA = -9999.0
# Cells the source cube does not reach are padded with a valid LAI instead of
# nodata, so widening the grid to the model extent leaves no holes for mHM.
PAD_VALUE = 0.0
DEFAULT_BLOCK_BYTES = 32 * 1024**2
CHUNK_ROWS = 128
CHUNK_COLS = 1024
# Level 1 compresses an upsampled LAI grid about as well as level 4 (76:1 vs
# 78:1 on a 100x upsample) in roughly half the time, and the time dominates.
LAI_COMPRESS_LEVEL = 1
COORD_COMPRESS_LEVEL = 4
# Measured on a 100x upsample of real GIMMS LAI: bilinear output compresses only
# 1.80:1 on the DEM grid and 1.85:1 on the L0 grid, because every cell differs.
# Nearest-neighbour reached 70:1 on its long runs of repeated values, so do not
# reuse that figure here. The guard re-reads free space at each stage, so an
# accurate ratio lets stage 1 through and stops stage 2 once the disk is short.
LAI_ASSUMED_COMPRESSION = 1.8


# Temporal aggregation


TARGET_TIMESTEPS = {
    "daily": (-1, "D"),
    "monthly": (-2, "MS"),
    "annual": (-3, "YS"),
    "long-term-mean-monthly": (1, None),
}
_TARGET_ALIASES = {
    "daily gridded data": "daily",
    "monthly gridded data": "monthly",
    "yearly gridded data": "annual",
    "annual gridded data": "annual",
    "long term mean monthly gridded data": "long-term-mean-monthly",
}
_INPUT_DAYS = {
    "daily": 1.0,
    "semi-weekly": 3.5,
    "weekly": 7.0,
    "biweekly": 14.0,
    "monthly": 30.4375,
    "annual": 365.25,
}
_TARGET_DAYS = {-1: 1.0, -2: 30.4375, -3: 365.25}


@dataclass(frozen=True)
class _LaiTemporalResult:
    data: object
    time_bounds: object
    time_step: int
    output_temporal_resolution: str


def _target_name(output_temporal_resolution: str) -> str:
    normalized = str(output_temporal_resolution or "").strip().lower()
    normalized = _TARGET_ALIASES.get(normalized, normalized)
    if normalized not in TARGET_TIMESTEPS:
        choices = ", ".join(TARGET_TIMESTEPS)
        msg = (
            f"Unsupported LAI output temporal resolution "
            f"{output_temporal_resolution!r}; expected {choices}."
        )
        raise ValueError(msg)
    return normalized


def lai_time_step(output_temporal_resolution: str) -> int:
    """Return the mHM time-step flag for an LAI output resolution."""
    return TARGET_TIMESTEPS[_target_name(output_temporal_resolution)][0]


def _source_offset(name):
    import pandas as pd

    return {
        "daily": pd.offsets.Day(1),
        "semi-weekly": pd.Timedelta(days=3, hours=12),
        "weekly": pd.offsets.Week(1),
        "biweekly": pd.offsets.Week(2),
        "monthly": pd.offsets.MonthBegin(1),
        "annual": pd.offsets.YearBegin(1),
    }[name]


def _target_offset(step):
    import pandas as pd

    return {
        -1: pd.offsets.Day(1),
        -2: pd.offsets.MonthBegin(1),
        -3: pd.offsets.YearBegin(1),
    }[step]


def _inferred_input_resolution(data, time_dim, time_bounds=None) -> str:
    """Infer the source cadence from decoded coordinates or CF time bounds."""
    import numpy as np

    values = np.asarray(data[time_dim].values)
    if not np.issubdtype(values.dtype, np.datetime64):
        msg = "LAI time coordinates must be decoded calendar dates."
        raise ValueError(msg)

    if values.size > 1:
        differences = np.diff(values.astype("datetime64[ns]"))
        days = np.median(differences / np.timedelta64(1, "D"))
    elif time_bounds is not None:
        bounds = np.asarray(time_bounds)
        if bounds.shape != (1, 2):
            msg = "LAI time bounds must have shape (time, 2)."
            raise ValueError(msg)
        days = float((bounds[0, 1] - bounds[0, 0]) / np.timedelta64(1, "D"))
    else:
        msg = "Cannot infer LAI input cadence from a single time value without bounds."
        raise ValueError(msg)

    if not np.isfinite(days) or days <= 0:
        msg = "LAI time coordinates must be strictly increasing."
        raise ValueError(msg)
    if days <= 2:
        return "daily"
    if days <= 5.25:
        return "semi-weekly"
    if days <= 10.5:
        return "weekly"
    if days <= 22:
        return "biweekly"
    if days <= 180:
        return "monthly"
    return "annual"


def prepare_lai_temporal(
    data,
    output_temporal_resolution,
    time_dim="time",
    time_bounds=None,
):
    """Convert one LAI array to the requested mHM time convention."""
    import numpy as np
    import pandas as pd

    target_name = _target_name(output_temporal_resolution)
    if time_dim not in data.dims:
        msg = "The selected LAI variable has no time dimension."
        raise ValueError(msg)
    if data.sizes.get(time_dim, 0) == 0:
        msg = "The selected LAI variable has an empty time dimension."
        raise ValueError(msg)

    values = data[time_dim].values
    if not np.issubdtype(np.asarray(values).dtype, np.datetime64):
        if target_name == "long-term-mean-monthly" and data.sizes[time_dim] == 12:
            monthly = data.rename({time_dim: "time"}).assign_coords(
                time=np.arange(1, 13, dtype=np.int32)
            )
            bounds = np.column_stack(
                (np.arange(0, 12, dtype=float), np.arange(1, 13, dtype=float))
            )
            return _LaiTemporalResult(monthly, bounds, 1, output_temporal_resolution)
        msg = "LAI time coordinates must be decoded calendar dates."
        raise ValueError(msg)

    data = data.sortby(time_dim)
    if target_name == "long-term-mean-monthly":
        monthly = data.groupby(f"{time_dim}.month").mean(time_dim, skipna=True)
        if monthly.sizes.get("month") != 12:
            msg = "Long-term monthly LAI requires observations for all 12 months."
            raise ValueError(msg)
        monthly = monthly.rename({"month": "time"}).assign_coords(
            time=np.arange(1, 13, dtype=np.int32)
        )
        bounds = np.column_stack(
            (np.arange(0, 12, dtype=float), np.arange(1, 13, dtype=float))
        )
        return _LaiTemporalResult(monthly, bounds, 1, output_temporal_resolution)

    input_name = _inferred_input_resolution(data, time_dim, time_bounds)
    target_step, frequency = TARGET_TIMESTEPS[target_name]
    source_days = _INPUT_DAYS[input_name]
    target_days = _TARGET_DAYS[target_step]
    if source_days < target_days:
        converted = data.resample({time_dim: frequency}).mean(skipna=True)
    elif source_days > target_days:
        times = pd.DatetimeIndex(data[time_dim].values)
        end = times[-1] + _source_offset(input_name)
        target_index = pd.date_range(
            start=times[0], end=end, freq=frequency, inclusive="left"
        )
        converted = data.reindex({time_dim: target_index}, method="ffill")
    else:
        converted = data

    if time_dim != "time":
        converted = converted.rename({time_dim: "time"})
    converted = converted.astype("float64")
    starts = pd.DatetimeIndex(converted["time"].values)
    ends = starts + _target_offset(target_step)
    bounds = np.column_stack(
        (starts.values.astype("datetime64[ns]"), ends.values.astype("datetime64[ns]"))
    )
    return _LaiTemporalResult(
        converted, bounds, target_step, output_temporal_resolution
    )


# Source NetCDF discovery


class _LaiSource(NamedTuple):
    """A located LAI cube and the coordinate names it is indexed by."""

    dataset: object
    data: object
    lat_coord: str
    lon_coord: str
    time_dim: str


def locate_lai_cube(dataset, source_variable=None, log=None) -> _LaiSource:
    """Find the LAI variable in an open dataset and name its three axes.

    Latitude and longitude are promoted to the cube's own dimensions, so the
    caller can index it by coordinate name instead of by whatever dimension
    names the source file happened to use.
    """
    lat_coord = find_coordinate(dataset, "lat")
    lon_coord = find_coordinate(dataset, "lon")
    if lat_coord is None or lon_coord is None:
        msg = "LAI NetCDF must contain 1D latitude and longitude coordinates."
        raise ValueError(msg)

    promote = [name for name in (lat_coord, lon_coord) if name in dataset.data_vars]
    if promote:
        dataset = dataset.set_coords(promote)

    lat_dim = dataset[lat_coord].dims[0]
    lon_dim = dataset[lon_coord].dims[0]
    lai_data = find_variable(dataset, source_variable, lat_dim, lon_dim, log)
    lai_data, lat_dim, lon_dim = use_coordinate_dimensions(
        lai_data, lat_coord, lon_coord, lat_dim, lon_dim
    )
    time_dim = find_time_dimension(lai_data, lat_dim, lon_dim)
    if time_dim is None:
        msg = "LAI variable must contain exactly one temporal dimension."
        raise ValueError(msg)
    extra = [dim for dim in lai_data.dims if dim not in (time_dim, lat_dim, lon_dim)]
    if extra:
        msg = f"LAI variable has unsupported extra dimension(s): {', '.join(extra)}."
        raise ValueError(msg)
    return _LaiSource(dataset, lai_data, lat_coord, lon_coord, time_dim)


def source_time_bounds(dataset, time_dim):
    """Return the CF time-bounds array of a time axis, or None when absent."""
    bounds_name = dataset[time_dim].attrs.get("bounds")
    if bounds_name and bounds_name in dataset.variables:
        return dataset[bounds_name].values
    return None


def find_coordinate(dataset, coordinate_type: str) -> str | None:
    """Find a 1D latitude or longitude coordinate in a NetCDF dataset."""
    if coordinate_type == "lat":
        candidates = ("lat", "latitude", "y")
        standard_name = "latitude"
        axis = "Y"
    else:
        candidates = ("lon", "longitude", "x")
        standard_name = "longitude"
        axis = "X"

    for name in candidates:
        if name in dataset.variables and dataset[name].ndim == 1:
            return name

    for name in dataset.variables:
        variable = dataset[name]
        if variable.ndim != 1:
            continue
        attrs = variable.attrs
        if str(attrs.get("standard_name", "")).lower() == standard_name:
            return name
        if str(attrs.get("axis", "")).upper() == axis:
            return name
    return None


def find_time_dimension(data_array, lat_dim, lon_dim):
    """Return the single non-spatial dimension, or None when it is ambiguous."""
    dimensions = [dim for dim in data_array.dims if dim not in (lat_dim, lon_dim)]
    if len(dimensions) != 1:
        return None
    return dimensions[0]


def find_variable(dataset, source_variable, lat_dim: str, lon_dim: str, log=None):
    """Find the LAI data variable to process."""
    candidates = []
    if source_variable:
        candidates.append(source_variable)
        basename = Path(source_variable).name
        if basename not in candidates:
            candidates.append(basename)

    for candidate in candidates:
        if candidate in dataset.data_vars:
            data_array = dataset[candidate]
            if lat_dim in data_array.dims and lon_dim in data_array.dims:
                return data_array
            msg = (
                f"Selected NetCDF variable '{candidate}' does not use the "
                "detected latitude/longitude dimensions."
            )
            raise ValueError(msg)

    for candidate in ("lai", "LAI", "leaf_area_index", "Leaf_Area_Index"):
        if candidate in dataset.data_vars:
            data_array = dataset[candidate]
            if lat_dim in data_array.dims and lon_dim in data_array.dims:
                return data_array

    for name, data_array in dataset.data_vars.items():
        if lat_dim not in data_array.dims or lon_dim not in data_array.dims:
            continue
        if find_time_dimension(data_array, lat_dim, lon_dim):
            if log:
                log(f"Using LAI variable '{name}'.")
            return data_array

    msg = (
        "Could not find a LAI variable with latitude, longitude, and time "
        "dimensions."
    )
    raise ValueError(msg)


def use_coordinate_dimensions(data_array, lat_coord, lon_coord, lat_dim, lon_dim):
    """Use latitude and longitude coordinates as the interpolation dimensions."""
    swap = {}
    if lat_coord != lat_dim:
        swap[lat_dim] = lat_coord
    if lon_coord != lon_dim:
        swap[lon_dim] = lon_coord
    if swap:
        data_array = data_array.swap_dims(swap)
        lat_dim, lon_dim = lat_coord, lon_coord
    return data_array, lat_dim, lon_dim


def normalize_longitudes(data_array, lon_dim):
    """Convert a 0..360 longitude axis to the conventional -180..180 range."""
    import numpy as np

    longitude = np.asarray(data_array[lon_dim].values, dtype="float64")
    if longitude.size and np.nanmin(longitude) >= 0 and np.nanmax(longitude) > 180:
        longitude = (longitude + 180) % 360 - 180
        data_array = data_array.assign_coords({lon_dim: longitude}).sortby(lon_dim)
        longitude = np.asarray(data_array[lon_dim].values, dtype="float64")
        keep = np.concatenate(([True], np.diff(longitude) > 1e-10))
        data_array = data_array.isel({lon_dim: keep})
    return data_array


# Target grid geometry


class _LaiTargetGrid(NamedTuple):
    """A target raster grid and lazy per-row-block WGS84 coordinates."""

    x_centers: object
    y_centers: object
    row_lonlat: object
    transform: object
    crs: str


def is_geographic_crs_string(crs_string) -> bool:
    """Return True when a CRS string describes a lon/lat system."""
    text = str(crs_string or "").strip()
    if not text:
        return False
    if text.upper() in {"EPSG:4326", "OGC:CRS84", "CRS84"}:
        return True
    try:
        from pyproj import CRS

        return bool(CRS.from_user_input(text).is_geographic)
    except Exception:
        return False


def lazy_target_grid(x_centers, y_centers, transform, crs_string) -> _LaiTargetGrid:
    """Return target cell centres plus a per-row-block WGS84 mesh generator.

    The full mesh is never built. On a 13320 x 6120 grid it would be two
    652 MiB arrays before any LAI value has been read.
    """
    import numpy as np

    x_centers = np.asarray(x_centers, dtype=np.float64)
    y_centers = np.asarray(y_centers, dtype=np.float64)
    geographic = is_geographic_crs_string(crs_string)
    transformer = None
    if not geographic:
        from pyproj import Transformer

        transformer = Transformer.from_crs(crs_string, "EPSG:4326", always_xy=True)

    def row_lonlat(start, stop):
        rows = y_centers[start:stop]
        if geographic:
            lon = np.broadcast_to(x_centers, (rows.size, x_centers.size))
            lat = np.repeat(rows[:, None], x_centers.size, axis=1)
            return lon, lat
        x_grid, y_grid = np.meshgrid(x_centers, rows)
        lon, lat = transformer.transform(x_grid, y_grid)
        return np.asarray(lon, dtype=np.float64), np.asarray(lat, dtype=np.float64)

    return _LaiTargetGrid(x_centers, y_centers, row_lonlat, transform, crs_string)


def target_grid_from_header(header, crs_string) -> _LaiTargetGrid:
    """Return the target grid described by an mHM-style header."""
    import numpy as np
    from rasterio.transform import from_origin

    ncols = int(header["ncols"])
    nrows = int(header["nrows"])
    cellsize = float(header["cellsize"])
    if ncols <= 0 or nrows <= 0 or cellsize <= 0:
        msg = "Grid header has invalid dimensions or cell size."
        raise ValueError(msg)
    xmin = float(header["xllcorner"])
    ymax = float(header["yllcorner"]) + nrows * cellsize
    x_centers = xmin + (np.arange(ncols, dtype=np.float64) + 0.5) * cellsize
    y_centers = ymax - (np.arange(nrows, dtype=np.float64) + 0.5) * cellsize
    return lazy_target_grid(
        x_centers,
        y_centers,
        from_origin(xmin, ymax, cellsize, cellsize),
        crs_string,
    )


def target_grid_from_raster(raster_path, crs_string=None) -> tuple[_LaiTargetGrid, str]:
    """Return the target grid of a raster file, read without QGIS."""
    import numpy as np
    import rasterio

    with rasterio.open(raster_path) as dataset:
        transform = dataset.transform
        width, height = dataset.width, dataset.height
        if width <= 0 or height <= 0:
            msg = f"Raster has invalid dimensions: {raster_path}"
            raise ValueError(msg)
        if transform.b or transform.d:
            msg = f"Rotated grids are not supported: {raster_path}"
            raise ValueError(msg)
        resolved = dataset.crs.to_string() if dataset.crs else str(crs_string or "")

    x_centers = transform.c + (np.arange(width, dtype=np.float64) + 0.5) * transform.a
    y_centers = transform.f + (np.arange(height, dtype=np.float64) + 0.5) * transform.e
    if not resolved:
        msg = f"Could not determine the CRS of {raster_path}"
        raise ValueError(msg)
    return lazy_target_grid(x_centers, y_centers, transform, resolved), resolved


def _axis_step(values, name):
    """Return the positive spacing of a regular source coordinate axis."""
    import numpy as np

    values = np.asarray(values, dtype="float64")
    if values.size < 2:
        msg = f"LAI {name} coordinates need at least two values."
        raise ValueError(msg)
    differences = np.diff(values)
    step = float(np.median(np.abs(differences)))
    if step <= 0 or not np.allclose(np.abs(differences), step, rtol=1e-5):
        msg = f"LAI {name} coordinates must form a regular grid."
        raise ValueError(msg)
    return step


def lai_window_offsets(source_x, source_y, target_header):
    """Return the integer (row, column) offset of the target grid into a source.

    ``source index = target index + offset`` on both axes. Raises when the two
    grids do not share a cell size and cell boundaries, because the copy would
    then silently shift the data.
    """
    import numpy as np

    source_x = np.asarray(source_x, dtype="float64")
    source_y = np.asarray(source_y, dtype="float64")
    if source_x.size < 2 or source_y.size < 2:
        msg = "The staged LAI grid needs at least two cells per axis."
        raise ValueError(msg)
    cellsize = float(target_header["cellsize"])
    tolerance = abs(cellsize) * 1e-6

    source_cell_x = float(abs(source_x[1] - source_x[0]))
    source_cell_y = float(abs(source_y[1] - source_y[0]))
    for name, value in (("x", source_cell_x), ("y", source_cell_y)):
        if abs(value - cellsize) > tolerance:
            msg = (
                f"Staged LAI {name} cell size {value} does not match the target "
                f"cell size {cellsize}; it cannot be cropped without resampling."
            )
            raise ValueError(msg)

    # yc runs from the top down, so its first centre is half a cell below ymax.
    source_ymax = float(source_y[0]) + cellsize / 2.0
    source_xmin = float(source_x[0]) - cellsize / 2.0
    target_ymax = (
        float(target_header["yllcorner"]) + int(target_header["nrows"]) * cellsize
    )
    target_xmin = float(target_header["xllcorner"])

    row_offset = (source_ymax - target_ymax) / cellsize
    column_offset = (target_xmin - source_xmin) / cellsize
    for name, value in (("row", row_offset), ("column", column_offset)):
        if abs(value - round(value)) > 1e-6:
            msg = f"The staged LAI grid is not aligned to the target {name} grid."
            raise ValueError(msg)
    return round(row_offset), round(column_offset)


# Output size guard


def lai_grid_byte_size(steps: int, nrows: int, ncols: int) -> int:
    """Return the double-precision byte size of the placed LAI array."""
    return int(steps) * int(nrows) * int(ncols) * 8


def assert_lai_output_fits(steps: int, nrows: int, ncols: int, folder) -> int:
    """Reject an LAI request that cannot fit on disk, and return its size.

    Memory is bounded by the streaming writer, so the remaining limit is the
    output file. It is written compressed, but refuse outright when even a
    generous compression estimate cannot fit in the free space.
    """
    import shutil

    required = lai_grid_byte_size(steps, nrows, ncols)
    try:
        free = shutil.disk_usage(str(folder)).free
    except OSError:
        return required
    if required / LAI_ASSUMED_COMPRESSION > free:
        msg = (
            f"LAI needs about {required / 1024 ** 3:.1f} GiB uncompressed "
            f"({int(steps)} time step(s) on a {int(ncols)} x {int(nrows)} "
            f"grid) and only {free / 1024 ** 3:.1f} GiB is free on the output "
            "volume. Choose 'long-term-mean-monthly' or use a smaller "
            "model extent."
        )
        raise MemoryError(msg)
    return required


# Row-block samplers


class _RasterioSampler:
    """Warp an in-memory latitude/longitude cube onto target row blocks."""

    def __init__(
        self,
        source_values,
        source_lat,
        source_lon,
        target_grid,
        method="bilinear",
        blank_fill=None,
    ):
        import numpy as np
        from rasterio.enums import Resampling
        from rasterio.transform import from_origin

        method = str(method or "bilinear").strip().lower()
        methods = {"bilinear": Resampling.bilinear, "nearest": Resampling.nearest}
        if method not in methods:
            msg = f"Unsupported LAI resampling method: {method}"
            raise ValueError(msg)

        values = np.asarray(source_values, dtype="float64")
        lat = np.asarray(source_lat, dtype="float64")
        lon = np.asarray(source_lon, dtype="float64")
        if lat[0] < lat[-1]:
            lat = lat[::-1]
            values = values[:, ::-1, :]
        if lon[0] > lon[-1]:
            lon = lon[::-1]
            values = values[:, :, ::-1]

        x_step = _axis_step(lon, "longitude")
        y_step = _axis_step(lat, "latitude")
        self.values = values
        self.source_transform = from_origin(
            lon[0] - x_step / 2,
            lat[0] + y_step / 2,
            x_step,
            y_step,
        )
        self.target_grid = target_grid
        self.resampling = methods[method]
        self.blank_fill = None if blank_fill is None else float(blank_fill)
        self._window = None

    @property
    def steps(self) -> int:
        return int(self.values.shape[0])

    def prepare_block(self, start, stop, _lon_block, _lat_block):
        from rasterio.windows import Window

        self._window = Window(
            0,
            start,
            len(self.target_grid.x_centers),
            stop - start,
        )

    def sample(self, step):
        import numpy as np
        from rasterio.warp import reproject
        from rasterio.windows import transform

        destination = np.full(
            (int(self._window.height), int(self._window.width)),
            np.nan,
            dtype="float64",
        )
        reproject(
            source=self.values[step],
            destination=destination,
            src_transform=self.source_transform,
            src_crs="EPSG:4326",
            src_nodata=np.nan,
            dst_transform=transform(self._window, self.target_grid.transform),
            dst_crs=self.target_grid.crs,
            dst_nodata=np.nan,
            resampling=self.resampling,
        )
        if self.blank_fill is not None:
            destination[~np.isfinite(destination)] = self.blank_fill
        return destination


class _WindowSampler:
    """Copy an aligned source variable by integer window, padding the rest.

    ``pad_value`` fills the cells the source window does not cover; it is a
    valid LAI rather than nodata so that expanding the extent never creates
    holes inside the model domain.
    """

    def __init__(self, variable, row_offset, column_offset, pad_value=PAD_VALUE):
        self.variable = variable
        self.row_offset = int(row_offset)
        self.column_offset = int(column_offset)
        self.pad_value = float(pad_value)
        self._source_rows = int(variable.shape[1])
        self._source_columns = int(variable.shape[2])
        self._window = None

    @property
    def steps(self) -> int:
        return int(self.variable.shape[0])

    def prepare_block(self, start, stop, _lon_block, _lat_block):
        rows = stop - start
        source_start = start + self.row_offset
        read_start = max(source_start, 0)
        read_stop = min(source_start + rows, self._source_rows)
        column_start = max(self.column_offset, 0)
        column_stop = min(self.column_offset + self._columns, self._source_columns)
        self._window = (
            rows,
            read_start,
            read_stop,
            read_start - source_start,
            column_start,
            column_stop,
            column_start - self.column_offset,
        )

    def bind(self, columns):
        """Record the target column count before the first block is prepared."""
        self._columns = int(columns)

    def sample(self, step):
        import numpy as np

        (
            rows,
            read_start,
            read_stop,
            target_row,
            column_start,
            column_stop,
            target_column,
        ) = self._window
        block = np.full((rows, self._columns), self.pad_value, dtype="float64")
        if read_stop > read_start and column_stop > column_start:
            values = np.asarray(
                self.variable[step, read_start:read_stop, column_start:column_stop],
                dtype="float64",
            )
            block[
                target_row : target_row + (read_stop - read_start),
                target_column : target_column + (column_stop - column_start),
            ] = values
        return block


# Streaming NetCDF writer


def block_row_count(ncols: int, block_bytes: int = DEFAULT_BLOCK_BYTES) -> int:
    """Return how many target rows one working block should cover."""
    per_row = max(1, int(ncols)) * 8
    rows = max(1, int(block_bytes) // per_row)
    # Keep blocks a whole number of chunk rows so no write is a partial chunk.
    rows = max(CHUNK_ROWS, (rows // CHUNK_ROWS) * CHUNK_ROWS)
    return min(rows, 4096)


# Attributes describing how the source file stored its values. The writer sets
# its own, so carrying these over would misdescribe the output.
_SOURCE_ENCODING_ATTRS = (
    "_FillValue",
    "missing_value",
    "scale_factor",
    "add_offset",
    "coordinates",
    "grid_mapping",
)


def lai_output_attrs(source_attrs) -> dict:
    """Return the ``lai`` variable attributes to write, from the source ones."""
    attrs = {
        key: value
        for key, value in dict(source_attrs).items()
        if key not in _SOURCE_ENCODING_ATTRS
    }
    attrs.setdefault("long_name", "leaf area index")
    attrs.setdefault("units", "1")
    attrs["nodata_value"] = NODATA
    return attrs


def time_attrs_for_step(time_step: int) -> dict:
    """Return the time-axis attributes for a prepared LAI time step."""
    if time_step == 1:
        return {"long_name": "month", "units": "month"}
    return {"standard_name": "time", "axis": "T"}


def coordinate_dataset(
    x_centers, y_centers, crs_string, description, time_values, time_bounds, time_attrs
):
    """Build the small coordinate-only dataset written before the LAI cube."""
    import numpy as np
    import xarray as xr

    dataset = xr.Dataset(
        data_vars={"time_bnds": (("time", "bnds"), time_bounds)},
        coords={
            "time": time_values,
            "yc": np.asarray(y_centers, dtype=np.float64),
            "xc": np.asarray(x_centers, dtype=np.float64),
            "bnds": np.arange(2, dtype=np.int8),
        },
        attrs={
            "description": description,
            "projection": str(crs_string or "").lower(),
        },
    )
    dataset["time"].attrs.update(dict(time_attrs or {}))
    dataset["time"].attrs["bounds"] = "time_bnds"
    units = "degrees" if is_geographic_crs_string(crs_string) else "m"
    dataset["yc"].attrs.update({"axis": "Y", "units": units})
    dataset["xc"].attrs.update({"axis": "X", "units": units})
    return dataset


def stream_lai_grid(
    output_path,
    *,
    coordinate_dataset,
    sampler,
    x_centers,
    y_centers,
    row_lonlat,
    lai_attrs=None,
    mask=None,
    block_bytes: int = DEFAULT_BLOCK_BYTES,
    is_cancelled=None,
    progress=None,
    log=None,
) -> str:
    """Write a LAI NetCDF one block of target rows and one time step at a time.

    ``coordinate_dataset`` is a small xarray dataset carrying time, time_bnds,
    yc, xc and the global attributes; it is written first so xarray handles the
    CF time encoding. ``row_lonlat(start, stop)`` returns the WGS84 longitude
    and latitude meshes for target rows ``[start, stop)``.
    """
    import numpy as np
    from netCDF4 import Dataset

    nrows = len(y_centers)
    ncols = len(x_centers)
    steps = int(sampler.steps)
    rows_per_block = block_row_count(ncols, block_bytes)
    if hasattr(sampler, "bind"):
        sampler.bind(ncols)

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.tmp")
    temporary.unlink(missing_ok=True)
    coordinate_dataset.to_netcdf(temporary)

    chunks = (1, min(nrows, CHUNK_ROWS), min(ncols, CHUNK_COLS))
    try:
        with Dataset(temporary, "a") as handle:
            lai = handle.createVariable(
                "lai",
                "f8",
                ("time", "yc", "xc"),
                zlib=True,
                complevel=LAI_COMPRESS_LEVEL,
                chunksizes=chunks,
                fill_value=NODATA,
            )
            lai.setncatts(dict(lai_attrs or {}))
            latitude = handle.createVariable(
                "lat",
                "f8",
                ("yc", "xc"),
                zlib=True,
                complevel=COORD_COMPRESS_LEVEL,
                chunksizes=chunks[1:],
            )
            latitude.setncatts({"units": "degrees_north", "long_name": "latitude"})
            longitude = handle.createVariable(
                "lon",
                "f8",
                ("yc", "xc"),
                zlib=True,
                complevel=COORD_COMPRESS_LEVEL,
                chunksizes=chunks[1:],
            )
            longitude.setncatts({"units": "degrees_east", "long_name": "longitude"})

            for start in range(0, nrows, rows_per_block):
                if is_cancelled is not None and is_cancelled():
                    msg = "Task cancelled."
                    raise RuntimeError(msg)
                stop = min(start + rows_per_block, nrows)
                lon_block, lat_block = row_lonlat(start, stop)
                lon_block = np.asarray(lon_block, dtype="float64")
                lat_block = np.asarray(lat_block, dtype="float64")
                longitude[start:stop, :] = lon_block
                latitude[start:stop, :] = lat_block

                sampler.prepare_block(start, stop, lon_block, lat_block)
                block_mask = (
                    None
                    if mask is None
                    else np.asarray(mask[start:stop, :], dtype=bool)
                )

                for step in range(steps):
                    if is_cancelled is not None and is_cancelled():
                        msg = "Task cancelled."
                        raise RuntimeError(msg)
                    placed = sampler.sample(step)
                    if block_mask is not None:
                        placed = np.where(block_mask, placed, NODATA)
                    lai[step, start:stop, :] = placed

                if progress is not None:
                    progress(100.0 * stop / nrows)
                if log:
                    log(f"LAI rows {stop}/{nrows} written ({steps} time step(s)).")
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise

    if output.exists():
        output.unlink()
    temporary.replace(output)
    return str(output)


# Gridded LAI pipelines


def resample_lai_file_to_grid(
    source_path,
    output_path,
    *,
    target_grid,
    crs_string,
    source_variable=None,
    output_temporal_resolution="long-term-mean-monthly",
    description="Monthly LAI resampled to the target grid",
    method="bilinear",
    blank_fill=None,
    is_cancelled=None,
    progress=None,
    log=None,
) -> str:
    """Aggregate LAI in time, then stream it onto ``target_grid``.

    Only the small source cube is held; the target is filled and written one
    block of rows at a time for one time step at a time, so peak memory does
    not grow with the record length or the grid size.
    """
    import numpy as np
    import xarray as xr

    dataset = xr.open_dataset(source_path)
    try:
        source = locate_lai_cube(dataset, source_variable, log)
        dataset = source.dataset
        lat_coord, lon_coord = source.lat_coord, source.lon_coord

        temporal = prepare_lai_temporal(
            source.data,
            output_temporal_resolution,
            time_dim=source.time_dim,
            time_bounds=source_time_bounds(dataset, source.time_dim),
        )
        lai_data = temporal.data.sortby(lat_coord).sortby(lon_coord)
        lai_data = normalize_longitudes(lai_data, lon_coord)

        x_centers, y_centers = target_grid.x_centers, target_grid.y_centers
        steps = int(lai_data.sizes.get("time", 1))
        required = assert_lai_output_fits(
            steps,
            len(y_centers),
            len(x_centers),
            Path(output_path).parent,
        )
        if log:
            log(
                f"LAI: placing {steps} time step(s) on a "
                f"{len(x_centers)} x {len(y_centers)} grid "
                f"({required / 1024 ** 3:.1f} GiB uncompressed), streamed in "
                f"row blocks using {method} interpolation."
            )

        # The source stays small enough to hold; only the target is huge.
        source_values = np.asarray(
            lai_data.transpose("time", lat_coord, lon_coord).values
        )

        return stream_lai_grid(
            output_path,
            coordinate_dataset=coordinate_dataset(
                x_centers,
                y_centers,
                crs_string,
                description,
                lai_data["time"].values,
                temporal.time_bounds,
                time_attrs_for_step(temporal.time_step),
            ),
            sampler=_RasterioSampler(
                source_values,
                np.asarray(lai_data[lat_coord].values, dtype=np.float64),
                np.asarray(lai_data[lon_coord].values, dtype=np.float64),
                target_grid,
                method=method,
                blank_fill=blank_fill,
            ),
            x_centers=x_centers,
            y_centers=y_centers,
            row_lonlat=target_grid.row_lonlat,
            lai_attrs=lai_output_attrs(lai_data.attrs),
            is_cancelled=is_cancelled,
            progress=progress,
            log=log,
        )
    finally:
        dataset.close()


def window_copy_lai_file(
    source_path,
    output_path,
    target_header,
    crs_string,
    description,
    mask=None,
    pad_value=PAD_VALUE,
    is_cancelled=None,
    progress=None,
    log=None,
) -> str:
    """Copy an aligned LAI file onto ``target_header`` without resampling.

    The staged LAI already sits on the filled DEM cell grid, so the common
    extent is reached by an integer window copy, exactly like the raster
    layers. Cells beyond the staged extent take ``pad_value`` rather than
    nodata, so widening the grid leaves no holes inside the model domain. One
    time step of one row block is held at a time.
    """
    import numpy as np
    import xarray as xr
    from netCDF4 import Dataset

    target_grid = target_grid_from_header(target_header, crs_string)
    if mask is not None and tuple(np.shape(mask)) != (
        int(target_header["nrows"]),
        int(target_header["ncols"]),
    ):
        msg = "The watershed mask does not match the L0 grid."
        raise ValueError(msg)

    with xr.open_dataset(source_path) as staged:
        source_x = np.asarray(staged["xc"].values, dtype=np.float64)
        source_y = np.asarray(staged["yc"].values, dtype=np.float64)
        time_values = staged["time"].values
        time_bounds = np.asarray(staged["time_bnds"].values)
        time_attrs = dict(staged["time"].attrs)
        lai_attrs = dict(staged["lai"].attrs)
    row_offset, column_offset = lai_window_offsets(source_x, source_y, target_header)

    handle = Dataset(str(source_path), "r")
    try:
        sampler = _WindowSampler(
            handle.variables["lai"],
            row_offset,
            column_offset,
            pad_value=pad_value,
        )
        # The expanded extent makes this output no smaller than the staged one,
        # so the disk guard applies here too.
        required = assert_lai_output_fits(
            sampler.steps,
            int(target_header["nrows"]),
            int(target_header["ncols"]),
            Path(output_path).parent,
        )
        if log:
            log(
                f"LAI: copying {sampler.steps} time step(s) onto a "
                f"{int(target_header['ncols'])} x {int(target_header['nrows'])} "
                f"grid at row offset {row_offset}, column offset "
                f"{column_offset} ({required / 1024 ** 3:.1f} GiB uncompressed)."
            )
        return stream_lai_grid(
            output_path,
            coordinate_dataset=coordinate_dataset(
                target_grid.x_centers,
                target_grid.y_centers,
                crs_string,
                description,
                time_values,
                time_bounds,
                time_attrs,
            ),
            sampler=sampler,
            x_centers=target_grid.x_centers,
            y_centers=target_grid.y_centers,
            row_lonlat=target_grid.row_lonlat,
            lai_attrs=lai_attrs,
            mask=mask,
            is_cancelled=is_cancelled,
            progress=progress,
            log=log,
        )
    finally:
        handle.close()


# Categorical class definitions


_MONTHS = (
    "Jan",
    "Feb",
    "Mar",
    "Apr",
    "May",
    "Jun",
    "Jul",
    "Aug",
    "Sep",
    "Oct",
    "Nov",
    "Dec",
)
_CLASSDEFINITION_HEADER = (
    "ID   LAND-USE                  Jan.   Feb.    Mar.    Apr.    May    "
    "Jun.    Jul.    Aug.    Sep.    Oct.    Nov.    Dec.\n"
)


def _classdefinition_rows(table, class_field: str) -> list:
    """Validate and sort LAI definitions from a lookup table."""
    class_column = _resolve_field(table.columns, class_field)
    land_use_column = _resolve_field(table.columns, "LAND-USE")
    month_columns = [_resolve_field(table.columns, month) for month in _MONTHS]
    definitions = {}

    for row_number, (_, row) in enumerate(table.iterrows(), start=2):
        class_id = _required_integer(
            row[class_column], class_column, row_number, "LAI lookup"
        )
        if class_id <= 0:
            msg = f"LAI lookup row {row_number} has non-positive class {class_id}."
            raise ValueError(msg)
        if _is_blank(row[land_use_column]):
            msg = (
                f"LAI lookup row {row_number} has an empty "
                f"{land_use_column!r} value."
            )
            raise ValueError(msg)
        values = tuple(
            _required_number(row[column], column, row_number, "LAI lookup")
            for column in month_columns
        )
        if any(value < 0 for value in values):
            msg = f"LAI lookup row {row_number} contains a negative monthly value."
            raise ValueError(msg)

        definition = (str(row[land_use_column]).strip(), values)
        previous = definitions.get(class_id)
        if previous is not None and previous != definition:
            msg = f"LAI class {class_id} has conflicting lookup definitions."
            raise ValueError(msg)
        definitions[class_id] = definition

    class_ids = sorted(definitions)
    if class_ids != list(range(1, len(class_ids) + 1)):
        msg = "LAI classes must be consecutive positive integers starting at 1."
        raise ValueError(msg)
    return [
        {
            "class_id": class_id,
            "land_use": definitions[class_id][0],
            "values": definitions[class_id][1],
        }
        for class_id in class_ids
    ]


def _classdefinition_text(table, class_field: str) -> str:
    """Render an mHM ``LAI_classdefinition.txt`` lookup."""
    rows = _classdefinition_rows(table, class_field)
    if not rows:
        msg = "No valid LAI classdefinition rows were found."
        raise ValueError(msg)

    lines = [f"NoLAIclasses           {len(rows)}\n", _CLASSDEFINITION_HEADER]
    for row in rows:
        values = "".join(f"{value:.6g}".ljust(8) for value in row["values"])
        lines.append(
            f"{row['class_id']:2d}    {row['land_use']:<25} {values.rstrip()}\n"
        )
    return "".join(lines)


def _write_classdefinition_text(text: str, output_file: Path) -> Path:
    output_file.parent.mkdir(parents=True, exist_ok=True)
    output_file.write_text(text, encoding="utf-8")
    logger.info("Wrote LAI class definition to %s", output_file)
    return output_file


def write_lai_classdefinition(
    lookup_table: PathLike, output_file: PathLike, class_field: str
) -> Path:
    """Write ``LAI_classdefinition.txt`` from an LAI lookup table."""
    table = read_lookup_table(Path(lookup_table))
    return _write_classdefinition_text(
        _classdefinition_text(table, class_field), Path(output_file)
    )


# Public entry points


def format_lai_data(
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
    """Map a categorical raster and write its mHM LAI definition."""
    input_file = Path(input_file)
    dem_file = Path(dem_file)
    output_path = Path(output_path)
    lookup_table = Path(lookup_table)
    raster_output = get_categorical_output_path(output_path, "lai_class", output_type)
    definition_output = output_path / "LAI_classdefinition.txt"

    protected_inputs = {
        input_file.resolve(),
        dem_file.resolve(),
        lookup_table.resolve(),
    }
    for output in (raster_output, definition_output):
        if output.resolve() in protected_inputs:
            msg = f"Output must differ from all input files: {output}"
            raise ValueError(msg)

    table = read_lookup_table(lookup_table)
    definition_text = _classdefinition_text(table, class_field)
    format_categorical_data(
        input_file,
        dem_file,
        raster_output,
        table,
        mapping_field,
        class_field,
        variable_name="lai_class",
        input_crs=input_crs,
        dem_crs=dem_crs,
        resampling=resampling,
        fill_nodata=fill_nodata,
    )
    _write_classdefinition_text(definition_text, definition_output)
    return raster_output


def format_lai_netcdf_file(
    input_file: PathLike,
    dem_file: PathLike,
    output_file: PathLike,
    *,
    output_temporal_resolution: str = "long-term-mean-monthly",
    source_variable: str | None = None,
    dem_crs: str | None = None,
    resampling="bilinear",
    is_cancelled=None,
    progress=None,
    log=None,
) -> Path:
    """Temporally aggregate LAI and stream it onto the exact DEM grid."""
    input_file = Path(input_file)
    dem_file = Path(dem_file)
    output_file = Path(output_file)
    for label, path in (("LAI NetCDF", input_file), ("DEM raster", dem_file)):
        if not path.is_file():
            msg = f"{label} does not exist: {path}"
            raise ValueError(msg)
    if output_file.suffix.lower() != ".nc":
        msg = "Gridded LAI output must be a NetCDF file."
        raise ValueError(msg)
    if output_file.resolve() in {input_file.resolve(), dem_file.resolve()}:
        msg = f"LAI output must differ from input files: {output_file}"
        raise ValueError(msg)

    output_file.parent.mkdir(parents=True, exist_ok=True)
    target_grid, target_crs = target_grid_from_raster(dem_file, dem_crs)
    result = resample_lai_file_to_grid(
        input_file,
        output_file,
        target_grid=target_grid,
        crs_string=target_crs,
        source_variable=source_variable,
        output_temporal_resolution=output_temporal_resolution,
        description="LAI resampled to the DEM grid",
        method=resampling,
        blank_fill=0.0,
        is_cancelled=is_cancelled,
        progress=progress,
        log=log,
    )
    logger.info("Wrote formatted gridded LAI to %s", result)
    return Path(result)


def format_lai_netcdf_data(
    input_file: PathLike,
    dem_file: PathLike,
    output_path: PathLike,
    *,
    output_temporal_resolution: str = "long-term-mean-monthly",
    source_variable: str | None = None,
    dem_crs: str | None = None,
    resampling="bilinear",
) -> Path:
    """Write gridded LAI as ``lai.nc`` in an output directory."""
    return format_lai_netcdf_file(
        input_file,
        dem_file,
        Path(output_path) / "lai.nc",
        output_temporal_resolution=output_temporal_resolution,
        source_variable=source_variable,
        dem_crs=dem_crs,
        resampling=resampling,
    )


def copy_lai_netcdf_to_grid(
    input_file: PathLike,
    output_file: PathLike,
    target_header,
    crs: str,
    description: str,
    *,
    is_cancelled=None,
    progress=None,
    log=None,
) -> Path:
    """Window-copy aligned LAI onto another grid, padding with zero."""
    return Path(
        window_copy_lai_file(
            input_file,
            output_file,
            target_header,
            crs,
            description,
            is_cancelled=is_cancelled,
            progress=progress,
            log=log,
        )
    )
