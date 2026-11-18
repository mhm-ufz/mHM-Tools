"""
Format meteorological and observation NetCDF forcing onto an mHM grid.

Every NetCDF file a glob matches is read in bounded blocks, de-accumulated,
aggregated to the output temporal resolution, converted to the mHM unit and
sampled onto the target grid (nearest neighbour), then appended to one NetCDF.

Authors
-------
- Sanjeev Bashyal
"""

from __future__ import annotations

import itertools
import logging
import math
from contextlib import contextmanager
from dataclasses import dataclass, replace
from functools import lru_cache
from pathlib import Path

from mhm_tools.common.esri_grid import get_header_cell_centers

logger = logging.getLogger(__name__)

MHM_READY = "mHM ready"
AUTO = "auto"
_KINDS = {
    "pre": "pre",
    "precipitation": "pre",
    "temp": "tavg",
    "temperature": "tavg",
    **{name: name for name in ("tavg", "tmin", "tmax", "tann", "ssrd", "strd", "netrad", "eabs", "windspeed")},
    "pet": "pet",
    **{name: name for name in ("et", "sm", "neutrons", "twsa")},
}


def normalize_kind(kind: str) -> str:
    """Return the canonical meteorology input kind."""
    normalized = _KINDS.get(str(kind).strip().lower())
    if not normalized:
        msg = f"Unsupported meteorology data type: {kind}"
        raise ValueError(msg)
    return normalized


def normalize_source(source: str) -> str:
    """Normalize UI and legacy spellings of a meteorology data source."""
    compact = str(source).strip().lower().replace("_", "").replace("-", "")
    compact = compact.replace(" ", "")
    if compact == AUTO:
        return AUTO
    if compact == "mhmready":
        return MHM_READY
    if compact == "era5land":  # an older saved label; processing never depended on it
        return AUTO
    msg = f"Unsupported meteorology data source: {source}"
    raise ValueError(msg)


@dataclass(frozen=True)
class MeteoFolderSpec:
    """One user-selected meteorology input folder."""

    kind: str
    folder: Path | str
    source: str = AUTO
    crs: str | None = None
    pattern: str = "*.nc"
    temporal_resolution: str = "Daily"
    output_temporal_resolution: str = ""
    aggregation: str = ""
    unit: str = ""
    cumulative: bool = False
    cumulative_resolution: str = ""
    cumulative_reset: int | None = None  # the last hour, day or month a cycle covers; None: the period's last
    variable: str = ""  # the source variable; empty: the one `kind`'s aliases name
    clip_negative: bool = False  # the user accepted clipping the input's small negative amounts

    def normalized(self) -> MeteoFolderSpec:
        """Return the spec with canonical kind and source, and defaults filled in."""
        return MeteoFolderSpec(
            kind=normalize_kind(self.kind),
            folder=Path(self.folder),
            source=normalize_source(self.source),
            crs=self.crs,
            pattern=self.pattern,
            temporal_resolution=self.temporal_resolution,
            output_temporal_resolution=(
                self.output_temporal_resolution or self.temporal_resolution
            ),
            aggregation=(
                self.aggregation
                if (self.output_temporal_resolution or self.temporal_resolution)
                != self.temporal_resolution
                else ""
            ),
            unit=self.unit,
            cumulative=self.cumulative,
            cumulative_resolution=self.cumulative_resolution if self.cumulative else "",
            cumulative_reset=self.cumulative_reset if self.cumulative else None,
            variable=self.variable,
            clip_negative=self.clip_negative,
        )


@dataclass(frozen=True)
class SpatialMetadata:
    """Spatial metadata shared by all NetCDF files in an input folder."""

    kind: str
    source: str
    files: tuple[Path, ...]
    shape: tuple[int, int]
    x_coordinate: str
    y_coordinate: str
    x_resolution: float
    y_resolution: float
    resolution: float
    bounds: tuple[float, float, float, float]
    crs: str | None
    unit: str
    start_time: str = ""
    end_time: str = ""




# Output name: source aliases, output unit, selectable input units.
VARIABLES = {
    "pre": (("pre", "tp", "total_precipitation", "Total_precipitation_surface_1_Hour_Accumulation"), "mm", ("mm", "m", "kg m-2", "kg m-2 s-1", "mm/day")),
    "pet": (("pet", "pev", "potential_evaporation"), "mm", ("mm", "m", "kg m-2", "kg m-2 s-1", "mm/day")),
    "tavg": (("tavg", "t2m", "2m_temperature", "Temperature_height_above_ground"), "Celsius", ("Celsius", "K")),
    "tmin": (("tmin", "minimum_temperature", "t2m", "2m_temperature", "Temperature_height_above_ground"), "Celsius", ("Celsius", "K")),
    "tmax": (("tmax", "maximum_temperature", "t2m", "2m_temperature", "Temperature_height_above_ground"), "Celsius", ("Celsius", "K")),
    "tann": (("tann", "annual_mean_temperature"), "Celsius", ("Celsius", "K")),
    "ssrd": (("ssrd", "surface_solar_radiation_downwards"), "W m-2", ("W m-2", "J m-2")),
    "strd": (("strd", "surface_thermal_radiation_downwards"), "W m-2", ("W m-2", "J m-2")),
    "netrad": (("netrad", "net_radiation", "surface_net_radiation"), "W m-2", ("W m-2", "J m-2")),
    "eabs": (("eabs", "vapour_pressure", "vapor_pressure"), "Pa", ("Pa", "hPa", "kPa")),
    "windspeed": (("windspeed", "wind_speed"), "m s-1", ("m s-1", "km/h")),
    # Additional observations are already expressed in their model units.
    "et": (("et", "evapotranspiration"), "mm", ("mm",)),
    "sm": (("sm", "soil_moisture"), "m3 m-3", ("m3 m-3",)),
    "neutrons": (("neutrons",), "counts", ("counts",)),
    "twsa": (("twsa", "tws"), "mm", ("mm",)),
}
CADENCES = {"Hourly": "h", "Daily": "D", "Monthly": "MS", "Yearly": "YS"}
# `datetime64` units are calendar units: `[M] + 1` is the next month and
# `[Y] + 1` the next year, so numpy does the 28/29/30/31-day and leap handling
# that pandas offsets used to -- in C, over a whole chunk at once.
VARIABLE_LABELS = {
    "pre": "Precipitation", "pet": "Potential evapotranspiration", "tavg": "Mean temperature",
    "tmin": "Minimum temperature", "tmax": "Maximum temperature", "tann": "Annual mean temperature",
    "ssrd": "Shortwave radiation", "strd": "Longwave radiation", "netrad": "Net radiation",
    "eabs": "Vapour pressure", "windspeed": "Wind speed",
}
PERIOD_UNITS = {"Hourly": "h", "Daily": "D", "Monthly": "M", "Yearly": "Y"}
# The last hour, day or month an accumulation covers before it resets; the last one by default.
RESETS = {"Daily": range(24), "Monthly": range(1, 32), "Yearly": range(1, 13)}
AGGREGATIONS = ("Average", "Sum", "Minimum", "Maximum")
ERA_END_VARIABLES = {"tp", "total_precipitation", "pev", "potential_evaporation",
                     "ssrd", "strd", "surface_solar_radiation_downwards",
                     "surface_thermal_radiation_downwards"}


def normalize_unit(unit):
    """Normalize common CF spellings, never silently change dimensions."""
    compact = str(unit).strip().lower().replace("**", "").replace("^", "").replace(" ", "")
    aliases = {"c": "Celsius", "degc": "Celsius", "degree_celsius": "Celsius",
               "degrees_celsius": "Celsius", "celsius": "Celsius", "kelvin": "K", "k": "K",
               "w/m2": "W m-2", "wm-2": "W m-2", "j/m2": "J m-2", "jm-2": "J m-2",
               "m/s": "m s-1", "ms-1": "m s-1", "kg/m2": "kg m-2",
               "kgm-2": "kg m-2", "kgm-2s-1": "kg m-2 s-1", "mmday-1": "mm/day",
               "mmd-1": "mm/day", "mm/d": "mm/day",
               "pa": "Pa", "hpa": "hPa", "kpa": "kPa",
               "m3s-1": "m3 s-1", "m3m-3": "m3 m-3", "counts": "counts"}
    return aliases.get(compact, compact)


def validate_spec(spec):
    """Reject unsupported units, cadence and cumulative combinations."""
    if spec.temporal_resolution not in CADENCES:
        msg = "Select the input temporal resolution."
        raise ValueError(msg)
    output = spec.output_temporal_resolution or spec.temporal_resolution
    if output not in CADENCES:
        msg = "Select the output temporal resolution."
        raise ValueError(msg)
    input_index = list(CADENCES).index(spec.temporal_resolution)
    output_index = list(CADENCES).index(output)
    if output_index < input_index:
        msg = "Output temporal resolution cannot be finer than input."
        raise ValueError(msg)
    if output_index == input_index and spec.aggregation:
        msg = "Aggregation must be <None> when input and output resolutions match."
        raise ValueError(msg)
    if output_index > input_index and spec.aggregation not in AGGREGATIONS:
        msg = "Select an aggregation type for the coarser output resolution."
        raise ValueError(msg)
    unit = normalize_unit(spec.unit)
    if unit not in VARIABLES[spec.kind][2]:
        msg = f"Select a supported unit: {', '.join(VARIABLES[spec.kind][2])}."
        raise ValueError(msg)
    if spec.cumulative:
        if spec.cumulative_resolution not in CADENCES:
            msg = "Select the accumulation reset frequency."
            raise ValueError(msg)
        if list(CADENCES).index(spec.cumulative_resolution) < list(CADENCES).index(spec.temporal_resolution):
            msg = "Accumulation cannot reset more often than the input sampling."
            raise ValueError(msg)
        if unit not in ("m", "mm", "kg m-2", "J m-2"):
            msg = "Cumulative data must be an amount, not a rate or temperature."
            raise ValueError(msg)
        if spec.cumulative_reset is not None and spec.cumulative_reset not in RESETS.get(spec.cumulative_resolution, ()):
            msg = "Select when the accumulation resets."
            raise ValueError(msg)


def output_periods(starts, cadence):
    """Return the calendar interval key containing each input interval start.

    One truncation for the whole array. The per-timestep `pd.Timestamp` plus
    `pd.DateOffset` this replaced fell through to `dateutil.relativedelta` and
    cost about 38 us a step, which was the largest single item in a long run.
    """
    import numpy as np

    return np.asarray(starts).astype(f"datetime64[{PERIOD_UNITS[cadence]}]")


def period_bounds(key):
    """Return the nanosecond start and end of one calendar interval key."""
    return key.astype("datetime64[ns]"), (key + 1).astype("datetime64[ns]")


def covered_until(last_start, cadence) -> str:
    """Return the end of the last complete `cadence` period, which starts at `last_start`."""
    import numpy as np

    return str(period_bounds(output_periods(np.datetime64(last_start[:16]), cadence))[1])[:16]


def covers_end(sim_end, last_start, cadence, v6) -> bool:
    """v6 runs up to `sim_end` (exclusive); v5.13 through the whole day `sim_end`."""
    covered = covered_until(last_start, cadence)
    return sim_end[:16] <= covered if v6 else sim_end[:10] < covered[:10]


def time_index(values, cadence, *, continuous=True):
    """Validate an ordered continuous Gregorian coordinate against its declared cadence."""
    import numpy as np
    import pandas as pd
    values = np.asarray(values)
    if not np.issubdtype(values.dtype, np.datetime64):
        msg = "Meteorology requires a Gregorian datetime coordinate."
        raise ValueError(msg)
    index = pd.DatetimeIndex(values)
    if not len(index) or index.hasnans or index.has_duplicates or not index.is_monotonic_increasing:
        msg = "Meteorology timestamps must be nonempty, unique and ordered."
        raise ValueError(msg)
    offset = {"Hourly": pd.DateOffset(hours=1), "Daily": pd.DateOffset(days=1),
              "Monthly": pd.DateOffset(months=1), "Yearly": pd.DateOffset(years=1)}[cadence]
    # one stamp in each consecutive calendar period, wherever in it (mid-month, month end, 23:00)
    if continuous and len(index) > 1 and not (np.diff(output_periods(index.values, cadence).astype("int64")) == 1).all():
        msg = f"Timestamps are not continuous {cadence.lower()} input."
        raise ValueError(msg)
    return index, offset


def interval_bounds(ds, time_name, spec, source_name):
    """Read CF intervals or the known ERA interval-end convention."""
    import numpy as np
    bounds_name = ds[time_name].attrs.get("bounds")
    index, offset = time_index(
        ds[time_name].values, spec.temporal_resolution,
        continuous=not (bounds_name and bounds_name in ds),
    )
    if bounds_name and bounds_name in ds:
        bounds = np.asarray(ds[bounds_name].values).astype("datetime64[ns]")
    elif source_name in ERA_END_VARIABLES and (
            spec.cumulative or (
                spec.aggregation in {"Minimum", "Maximum"}
                and spec.temporal_resolution == "Hourly"
                and spec.output_temporal_resolution == "Daily"
            )):
        bounds = np.column_stack(((index - offset).values, index.values))
    elif spec.cumulative:
        msg = "Cumulative input requires CF time bounds or a recognized ERA interval-end variable."
        raise ValueError(msg)
    else:  # the calendar period holding each stamp: month-end or 23:00 stamps are whole months and days
        bounds = np.column_stack(period_bounds(output_periods(index.values, spec.temporal_resolution)))
    if (bounds.shape != (len(index), 2) or np.isnat(bounds).any()
            or (bounds[:, 1] <= bounds[:, 0]).any()
            or (bounds[:-1, 1] != bounds[1:, 0]).any()):
        msg = "Meteorology time bounds must describe positive, contiguous intervals."
        raise ValueError(msg)
    return bounds


def complete_window(bounds, input_cadence, output_cadence):
    """Return the first through last complete output periods, excluding edge fragments."""
    import numpy as np

    def complete(starts, ends, cadence):
        keys = output_periods(starts, cadence)
        expected_start, expected_end = period_bounds(keys)
        return (starts == expected_start) & (ends == expected_end)

    input_full = complete(bounds[:, 0], bounds[:, 1], input_cadence)
    if not input_full[1:-1].all():
        msg = "Meteorology contains an incomplete input interval inside the record."
        raise ValueError(msg)
    keys = output_periods(bounds[:, 0], output_cadence)
    edges = np.r_[0, np.flatnonzero(keys[1:] != keys[:-1]) + 1, len(bounds)]
    starts, ends = bounds[edges[:-1], 0], bounds[edges[1:] - 1, 1]
    full = complete(starts, ends, output_cadence)
    selected = np.flatnonzero(full)
    if not len(selected) or not full[selected[0]:selected[-1] + 1].all():
        msg = "Meteorology has no continuous complete output periods."
        raise ValueError(msg)
    return starts[selected[0]], ends[selected[-1]], starts[selected[-1]]


def deaccumulate_values(values, bounds, spec, previous=None):
    """Return interval values while retaining the last cumulative plane."""
    import numpy as np
    import pandas as pd

    # Only the cumulative branch writes into the block, so a non-cumulative
    # variable must not pay a full copy of every chunk for nothing.
    result = np.asarray(values, dtype="float64")
    if spec.cumulative:
        result = result.copy()
        opens = cycle_starts(pd.DatetimeIndex(bounds[:, 0]), spec.cumulative_resolution, spec.cumulative_reset)
        for number, opened in enumerate(opens):
            raw = result[number].copy()
            if not opened and previous is None:  # the record's leading fragment: before its window
                result[number] = np.nan
            elif not opened:
                if previous[1] != bounds[number, 0]:
                    msg = "Cumulative data starts mid-cycle without its preceding sample."
                    raise ValueError(msg)
                result[number] -= previous[0]
            previous = (raw, bounds[number, 1])
    return result, previous


def cycle_starts(starts, cadence, reset=None):
    """Return whether each interval start opens an accumulation cycle.

    A cycle opens when the `reset` hour, day or month (the last a cycle covers) has
    just ended. A day past the month's length is its last day.
    """
    import pandas as pd

    last = RESETS[cadence][-1] if reset is None else reset
    if cadence == "Daily":
        return (starts.hour == (last + 1) % 24) & (starts == starts.floor("h"))
    midnight = starts == starts.normalize()
    if cadence == "Monthly":
        return midnight & ((starts.day == last + 1) | ((starts.day == 1) & ((starts - pd.Timedelta(days=1)).days_in_month <= last)))
    return midnight & (starts.day == 1) & (starts.month == last % 12 + 1)


@contextmanager
def about_variable(kind):
    """Name the variable in a failure inside, so the user knows which input to fix."""
    try:
        yield
    except ValueError as error:
        msg = f"{VARIABLE_LABELS.get(kind, kind)}: {error}"
        raise ValueError(msg) from error


class NegativeAmounts(Exception):
    """Amounts slightly below zero, as de-accumulated ERA5 leaves them; clipped once the user agrees."""

    def __init__(self, kind, lowest):
        self.kind = kind
        super().__init__(f"{VARIABLE_LABELS.get(kind, kind)} has small negative amounts (down to {lowest:.3g} mm), "
                         "most likely rounding noise of the source data.")


def convert_units(values, bounds, spec, source_name, *, input_seconds=None):
    """Convert already aggregated values to the model unit."""
    import numpy as np

    unit = normalize_unit(spec.unit)
    result = np.asarray(values, dtype="float64").copy()
    seconds = (bounds[:, 1] - bounds[:, 0]) / np.timedelta64(1, "s")
    seconds = seconds.reshape((-1,) + (1,) * (result.ndim - 1))
    source_seconds = float(input_seconds or seconds.reshape(-1)[0])
    if unit == "K":
        result -= 273.15
    elif unit == "m":
        result *= 1000.0
    elif unit == "kg m-2 s-1":
        result *= source_seconds if spec.aggregation == "Sum" else seconds
    elif unit == "mm/day":
        result *= (source_seconds if spec.aggregation == "Sum" else seconds) / 86400.0
    elif unit == "J m-2":
        result /= seconds if spec.aggregation == "Sum" else source_seconds
    elif unit == "hPa":
        result *= 100.0
    elif unit == "kPa":
        result *= 1000.0
    elif unit == "km/h":
        result /= 3.6
    if spec.kind == "pet" and source_name in {"pev", "potential_evaporation"}:
        result *= -1
    if spec.kind in {"pre", "pet"} and (negative := result < 0).any():
        lowest = float(result[negative].min())
        if lowest < -0.01:  # a wrong unit, accumulation or sign, not round-off
            sign = " or the sign of the evaporation" if spec.kind == "pet" else ""
            msg = (
                f"The input has negative amounts (down to {lowest:.3g} mm): check its unit, "
                             f"the cumulative setting{sign}."
            )
            raise ValueError(msg)
        if lowest < -1e-7 and not spec.clip_negative:  # float round-off below that is clipped silently
            raise NegativeAmounts(spec.kind, lowest)
        np.maximum(result, 0, out=result)
    return result


@dataclass(frozen=True)
class _TargetGrid:
    """Target cell centres, their exact header and WGS84 sample coordinates."""

    lon: list
    lat: list
    header: dict
    crs: str | None = None
    sample_lon: object | None = None
    sample_lat: object | None = None


def _create_target_grid(header, crs=None) -> _TargetGrid:
    """Return the target grid of an ASCII grid header.

    Parameters
    ----------
    header : dict
        Grid header with ncols, nrows, xllcorner, yllcorner and cellsize.
    crs : str, optional
        CRS of the header coordinates; None treats them as lon/lat.

    Returns
    -------
    _TargetGrid
        Cell-centre axes, the header, its CRS and the WGS84 sample meshes.
    """
    import numpy as np

    x_mesh, y_mesh = np.meshgrid(*get_header_cell_centers(header))
    lon_mesh, lat_mesh = x_mesh, y_mesh
    if crs:
        from pyproj import CRS, Transformer

        source = CRS.from_user_input(crs)
        if not source.is_geographic:
            lon_mesh, lat_mesh = Transformer.from_crs(
                source, "EPSG:4326", always_xy=True
            ).transform(x_mesh, y_mesh)
    middle_row, middle_column = lon_mesh.shape[0] // 2, lon_mesh.shape[1] // 2
    return _TargetGrid(
        lon=lon_mesh[middle_row, :].tolist(),
        lat=lat_mesh[:, middle_column].tolist(),
        header=dict(header),
        crs=crs,
        sample_lon=lon_mesh,
        sample_lat=lat_mesh,
    )


def _same_crs(left: str | None, right: str | None) -> bool:
    """Return whether two CRS strings describe the same CRS."""
    if not left or not right:
        return not left and not right
    try:
        from pyproj import CRS

        return bool(CRS.from_user_input(left) == CRS.from_user_input(right))
    except Exception:
        return str(left).strip().casefold() == str(right).strip().casefold()


def _axis_resolution(values, label: str = "coordinate") -> float:
    """Return the spacing of a regular one-dimensional coordinate axis."""
    import numpy as np

    values = np.asarray(values)
    unique = np.unique(values[np.isfinite(values)])
    if len(unique) < 2:
        msg = f"{label} needs at least two values."
        raise ValueError(msg)
    if len(unique) != len(values):
        msg = f"{label} contains duplicates."
        raise ValueError(msg)
    diffs = np.abs(np.diff(np.sort(unique)))
    resolution = float(np.nanmedian(diffs))
    if not np.allclose(diffs, resolution, rtol=1e-6, atol=1e-9):
        msg = f"{label} is not regularly spaced."
        raise ValueError(msg)
    return resolution


def _axes_match_header(x_values, y_values, header) -> bool:
    """Return whether both axes lie on the header's cell centres with its spacing."""
    import numpy as np

    cellsize = float(header["cellsize"])
    tolerance = abs(cellsize) * 1e-6
    target_x, target_y = get_header_cell_centers(header)
    for axis_values, target in ((x_values, target_x), (y_values, sorted(target_y))):
        values = np.sort(np.asarray(axis_values, dtype="float64").ravel())
        if values.size < 1:
            return False
        if values.size > 1 and abs(float(np.abs(np.diff(values)).max()) - cellsize) > tolerance:
            return False
        offset = (float(values[0]) - float(target[0])) / cellsize
        if abs(offset - round(offset)) > 1e-6:
            return False
    return True


def _shift_onto_header(x_values, y_values, header):
    """Move axes with the header's spacing onto its lattice, by under half a cell.

    These are the cells a nearest-neighbour pick would take, without interpolating.
    Other axes are kept.
    """
    import numpy as np

    cellsize = float(header["cellsize"])

    def shift(axis, origin):
        values = np.asarray(axis, dtype="float64")
        steps = np.abs(np.diff(values))
        if steps.size and float(np.abs(steps - cellsize).max()) <= cellsize * 1e-6:
            first = np.floor((values[0] - origin) / cellsize + 0.5)  # one shift for the axis, ties too
            return origin + (first + np.rint((values - values[0]) / cellsize)) * cellsize
        return values

    target_x, target_y = get_header_cell_centers(header)
    return shift(x_values, target_x[0]), shift(y_values, target_y[0])


def _assert_matches_header(dataset, variable: str, header) -> None:
    """Raise when a prepared block does not have the header's rows and columns."""
    rows, cols = int(header["nrows"]), int(header["ncols"])
    sizes = dataset[variable].sizes
    if sizes.get("lat") != rows or sizes.get("lon") != cols:
        msg = (
            f"{variable}: prepared L2 grid is {sizes.get('lat')}x{sizes.get('lon')}, "
            f"expected {rows}x{cols}."
        )
        raise ValueError(
            msg
        )


@dataclass(frozen=True)
class _SpatialSignature:
    metadata: SpatialMetadata
    x_values: object
    y_values: object


def _check_cancel(cancelled):
    if cancelled is not None and cancelled():
        msg = "Meteorology task cancelled."
        raise RuntimeError(msg)


def read_meteo_metadata(folder, kind=None, source=AUTO, crs_fallback=None,
                         *, cancelled=None, **options):
    """Inspect metadata only; no forcing arrays are read or prepared."""
    import numpy as np
    spec = (folder if isinstance(folder, MeteoFolderSpec) else
            MeteoFolderSpec(kind, folder, source, crs_fallback, **options)).normalized()
    validate_spec(spec)
    signatures, records = [], []
    for path in get_meteo_files(spec):
        _check_cancel(cancelled)
        signatures.append(_inspect_file(path, spec))
        with _open_dataset(path) as ds:
            variable = _find_variable(ds, spec)
            units = ds[variable].attrs.get("units")
            if units and normalize_unit(units) != normalize_unit(spec.unit):
                msg = f"{path.name}: metadata unit {units!r} differs from selected {spec.unit!r}."
                raise ValueError(msg)
            time_name = "time" if "time" in ds[variable].dims else "valid_time"
            bounds = interval_bounds(ds, time_name, spec, variable)
            records.append((path, ds[time_name].values.copy(), bounds))
    _validate_spatial_signatures(signatures)
    records.sort(key=lambda item: item[1][0])
    times = np.concatenate([item[1] for item in records])
    bounds = np.concatenate([item[2] for item in records])
    time_index(times, spec.temporal_resolution, continuous=False)
    if (bounds[:-1, 1] != bounds[1:, 0]).any():
        msg = "Meteorology files contain overlapping or missing intervals."
        raise ValueError(msg)
    if spec.cumulative:  # e.g. ERA5-Land's first 00:00 stamp: the previous day's total, no cycle of its own
        import pandas as pd

        opens = cycle_starts(pd.DatetimeIndex(bounds[:, 0]), spec.cumulative_resolution, spec.cumulative_reset)
        if not opens.any():
            msg = "Cumulative data hold no start of an accumulation cycle: check the reset."
            raise ValueError(msg)
        bounds = bounds[int(np.argmax(opens)):]
    start, _, last = complete_window(
        bounds, spec.temporal_resolution, spec.output_temporal_resolution)
    return replace(signatures[0].metadata, files=tuple(item[0] for item in records),
                   start_time=str(start), end_time=str(last))



# Aggregated intervals are stacked before they are aligned and written: doing
# that per day meant one resample, one Dataset build and one file append each,
# which is what made a multi-decade hourly record take hours. The batch is
# capped again per grid so a large L2 grid cannot buffer gigabytes.
INTERVAL_BATCH = 256


def _prepared_blocks(spec, metadata, target, *, window=None, on_file=None, cancelled=None):
    """Aggregate intervals, then convert and align them in batches."""
    import numpy as np
    import xarray as xr
    previous = None
    pending = None
    last_path = None
    finished = []
    for path in metadata.files:
        _check_cancel(cancelled)
        last_path = path
        with _open_dataset(path) as ds:
            variable = _find_variable(ds, spec)
            raw = ds[variable]
            time_name = "time" if "time" in raw.dims else "valid_time"
            bounds = interval_bounds(ds, time_name, spec, variable)
            y_dim, x_dim = _spatial_dims(raw)
            for dim in tuple(raw.dims):
                if dim not in {time_name, y_dim, x_dim}:
                    raw = raw.isel({dim: 0}, drop=True)
            raw = raw.rename({time_name: "time"}) if time_name != "time" else raw
            raw = raw.transpose("time", y_dim, x_dim)
            cells = max(raw.sizes[y_dim] * raw.sizes[x_dim],
                        int(target.header["nrows"]) * int(target.header["ncols"]))
            block_size = min(1024, max(1, 64 * 1024 * 1024 // (8 * cells)))
            # Buffered intervals are held in memory, so bound them the same way.
            batch = min(INTERVAL_BATCH, max(1, 64 * 1024 * 1024 // (8 * cells)))
            for start in range(0, raw.sizes["time"], block_size):
                _check_cancel(cancelled)
                chunk = raw.isel(time=slice(start, start + block_size)).load()
                chunk_bounds = bounds[start:start + block_size]
                values, previous = deaccumulate_values(
                    chunk.values, chunk_bounds, spec, previous)
                if window is not None:
                    selected = ((chunk_bounds[:, 0] >= window[0])
                                & (chunk_bounds[:, 1] <= window[1]))
                    if not selected.any():
                        continue
                    chunk = chunk.isel(time=np.flatnonzero(selected))
                    chunk_bounds = chunk_bounds[selected]
                    values = values[selected]
                if spec.output_temporal_resolution == spec.temporal_resolution:
                    chunk = chunk.assign_coords(time=chunk_bounds[:, 0])  # v5.13 reads only `time`
                    converted = convert_units(values, chunk_bounds, spec, variable)
                    yield _prepared_dataset(
                        chunk.copy(data=converted), chunk_bounds, ds, spec,
                        target, np, xr)
                    continue
                periods = output_periods(
                    chunk_bounds[:, 0], spec.output_temporal_resolution)
                edges = _group_edges(periods, np)
                group_value, group_count = _reduce_groups(
                    values, edges, spec.aggregation, np)
                for number in range(len(edges) - 1):
                    first, stop = int(edges[number]), int(edges[number + 1])
                    expected = periods[first]
                    if pending is not None and pending["expected"] != expected:
                        finished.append(_finish_interval(pending, spec, np))
                        pending = None
                        if len(finished) >= batch:
                            yield _stacked_dataset(finished, ds, spec, target, np, xr)
                            finished = []
                    if pending is None:
                        pending = {
                            "expected": expected,
                            "start": chunk_bounds[first, 0],
                            "end": chunk_bounds[stop - 1, 1],
                            "value": group_value[number],
                            "count": group_count[number],
                            "template": chunk.isel(time=slice(first, first + 1)),
                            "source": variable,
                            "input_seconds": float(
                                (chunk_bounds[first, 1] - chunk_bounds[first, 0])
                                / np.timedelta64(1, "s")),
                        }
                    else:
                        _merge_group(
                            pending, group_value[number], group_count[number],
                            spec.aggregation, np)
                        pending["end"] = chunk_bounds[stop - 1, 1]
        if on_file is not None:
            on_file(spec.kind)
    if pending is not None:
        finished.append(_finish_interval(pending, spec, np))
    if finished and last_path is not None:
        with _open_dataset(last_path) as ds:
            yield _stacked_dataset(finished, ds, spec, target, np, xr)


def _finish_interval(pending, spec, np):
    """Return one aggregated interval as (single-step template, bounds)."""
    actual = np.asarray([[pending["start"], pending["end"]]], dtype="datetime64[ns]")
    if tuple(actual[0]) != period_bounds(pending["expected"]):
        logger.warning(
            f"{spec.kind}: partial {spec.output_temporal_resolution.lower()} "
            f"interval {actual[0, 0]} to {actual[0, 1]}; aggregating available values."
        )
    values = _aggregate_result(pending, spec.aggregation, np)[None, ...]
    values = convert_units(
        values, actual, spec, pending["source"],
        input_seconds=pending["input_seconds"])
    return pending["template"].assign_coords(time=[actual[0, 0]]).copy(data=values), actual


def _stacked_dataset(finished, ds, spec, target, np, xr):
    """Align and convert one batch of finished intervals in a single pass."""
    chunk = (finished[0][0] if len(finished) == 1
             else xr.concat([template for template, _ in finished], dim="time"))
    bounds = np.concatenate([interval for _, interval in finished])
    return _prepared_dataset(chunk, bounds, ds, spec, target, np, xr)


def _group_edges(periods, np):
    """Return the index boundaries of each run of equal interval keys."""
    changed = np.flatnonzero(periods[1:] != periods[:-1]) + 1
    return np.concatenate(([0], changed, [len(periods)]))


def _reduce_groups(values, edges, method, np):
    """Reduce every interval in one chunk along time, without a Python loop.

    Uniform groups -- the usual hourly-to-daily case -- reduce as a single
    reshaped call. Monthly and yearly intervals have unequal lengths, so those
    fall back to one vectorised reduction per interval: still tens of
    iterations per file instead of one per timestep.
    """
    lengths = np.diff(edges)
    if len(lengths) > 1 and bool(np.all(lengths == lengths[0])):
        block = values.reshape(len(lengths), int(lengths[0]), *values.shape[1:])
        return _reduce_axis(block, 1, method, np)
    reduced = [_reduce_axis(values[first:stop], 0, method, np)
               for first, stop in itertools.pairwise(edges)]
    return (np.stack([value for value, _ in reduced]),
            np.stack([count for _, count in reduced]))


def _reduce_axis(values, axis, method, np):
    """Return the (partial aggregate, valid count) of one axis."""
    if method in {"Average", "Sum"}:
        finite = np.isfinite(values)
        return (np.add.reduce(np.where(finite, values, 0.0), axis=axis),
                finite.sum(axis=axis, dtype="int32"))
    # fmin/fmax already ignore NaN, so no masked copy of the block is needed;
    # the result is NaN exactly where an interval had no valid sample.
    reducer = np.fmin if method == "Minimum" else np.fmax
    value = reducer.reduce(values, axis=axis)
    return value, np.isfinite(value).astype("int32")


def _merge_group(pending, value, count, method, np):
    """Fold one reduced interval into a partial one carried across a boundary."""
    if method in {"Average", "Sum"}:
        pending["value"] = pending["value"] + value
        pending["count"] = pending["count"] + count
    else:
        reducer = np.fmin if method == "Minimum" else np.fmax
        pending["value"] = reducer(pending["value"], value)
        pending["count"] = np.isfinite(pending["value"]).astype("int32")


def _aggregate_result(pending, method, np):
    if method == "Average":
        return np.divide(
            pending["value"], pending["count"],
            out=np.full_like(pending["value"], np.nan),
            where=pending["count"] > 0)
    return np.where(pending["count"] > 0, pending["value"], np.nan)


def _prepared_dataset(chunk, bounds, ds, spec, target, np, xr):
    chunk = _resample_ready(chunk, ds, spec.crs, target, np, xr).load()
    chunk.name = spec.kind
    chunk.attrs = {
        "units": VARIABLES[spec.kind][1],
        "long_name": spec.kind,
        "input_cadence": spec.temporal_resolution,
        "output_cadence": spec.output_temporal_resolution,
    }
    output = chunk.to_dataset()
    output["time_bnds"] = (("time", "bnds"), bounds)
    output["time"].attrs = {
        "standard_name": "time", "axis": "T", "bounds": "time_bnds"}
    output["lat"].attrs = {"standard_name": "latitude", "units": "degrees_north"}
    output["lon"].attrs = {"standard_name": "longitude", "units": "degrees_east"}
    output.attrs = {"Conventions": "CF-1.8", "history": "Prepared by mhm-tools format-meteo"}
    return _attach_target_coordinates(output, spec.kind, target)



def get_meteo_files(spec: MeteoFolderSpec) -> list[Path]:
    """Return the NetCDF files the glob of `spec` matches in its folder, sorted."""
    folder = Path(spec.folder)
    if not folder.is_dir():
        msg = f"Meteorology folder does not exist: {folder}"
        raise FileNotFoundError(msg)
    pattern = spec.pattern.strip()
    if not pattern or Path(pattern).is_absolute() or ".." in Path(pattern).parts:
        msg = "Enter a relative NetCDF glob within the selected folder."
        raise ValueError(msg)
    files = sorted(path for path in folder.glob(pattern)
                   if path.is_file() and path.suffix.lower() == ".nc")
    if not files:
        msg = f"No NetCDF files match {pattern!r} in {folder}."
        raise ValueError(msg)
    return files


def _open_dataset(path: Path):
    import xarray as xr

    first_error = None  # netCDF4's names the file's problem; the auto engine's only lists backends
    for engine in ("netcdf4", "h5netcdf", "scipy", None):
        try:
            kwargs = {} if engine is None else {"engine": engine}
            return xr.open_dataset(
                path, decode_times=True, mask_and_scale=True, **kwargs)
        except Exception as exc:
            first_error = first_error or exc
    msg = f"Could not open {path} ({first_error}): check that it is a complete NetCDF file."
    raise RuntimeError(msg)


def _cadence(times):
    """Return the step at which `times` hold one stamp in every period, or None."""
    for cadence in CADENCES if len(times) > 1 else ():
        try:
            time_index(times, cadence)
        except ValueError:
            continue
        return cadence
    return None


def get_meteo_time_window(folder, pattern, time_name=""):
    """Return the time window of every file the glob matches, from its time variable alone.

    The result holds the candidate `times` variables, the `time` used, its `step`, the
    first period's `start`, the last period's start as `end` and the number of `files`.
    """
    import numpy as np

    files, stamps, names, name = get_meteo_files(MeteoFolderSpec("pre", Path(folder), pattern=pattern)), [], [], ""
    for path in files:
        with _open_dataset(path) as ds:
            names = names or [key for key, item in ds.variables.items()
                              if item.ndim == 1 and np.issubdtype(item.dtype, np.datetime64)]
            name = time_name or (names[0] if names else "")
            if name not in ds.variables or not np.issubdtype(ds[name].dtype, np.datetime64):
                raise ValueError(f"{path.name} has no time variable {name!r}." if name else
                                 f"{path.name} has no time information.")
            stamps.append(ds[name].values)
    times = np.sort(np.concatenate(stamps))
    step = _cadence(times)
    if step is None:
        msg = (
            "The files do not hold one time stamp in every period from start to end "
                         "(a gap, an overlap or an irregular step)."
        )
        raise ValueError(msg)
    keys = output_periods(times, step)
    return {"files": len(files), "times": names, "time": name, "step": step,
            "start": str(period_bounds(keys[0])[0])[:16], "end": str(period_bounds(keys[-1])[0])[:16]}


def get_meteo_file_defaults(spec):
    """Return what the first matching file declares.

    These are its time-dependent `variables`, and the `variable`, time step, unit and
    CRS; undeclared ones are left out (no `variable` when its name is not known).
    """
    path = get_meteo_files(spec)[0]
    found = {"file": path.name}
    with _open_dataset(path) as ds:
        bounds = {item.attrs.get("bounds") for item in ds.variables.values()}  # e.g. time_bnds
        found["variables"] = [name for name, data in ds.data_vars.items()
                              if {"time", "valid_time"} & set(data.dims) and name not in bounds]
        try:
            variable = _find_variable(ds, spec if spec.variable in found["variables"] else replace(spec, variable=""))
        except ValueError:
            return found
        found["variable"] = variable
        if (unit := normalize_unit(ds[variable].attrs.get("units", ""))) in VARIABLES[spec.kind][2]:
            found["unit"] = unit
        times = ds["time" if "time" in ds[variable].dims else "valid_time"].values
    if cadence := _cadence(times):
        found["temporal_resolution"] = cadence
    if crs := _inspect_file(path, replace(spec, crs=None), variable).metadata.crs:
        found["crs"] = crs
    return found


def get_netcdf_grid(path, variable, crs=None):
    """Return a NetCDF variable's grid as the forcing reads it, from its axes only.

    GDAL builds a band per time step, minutes for a long series. The result is the
    cell-edge bounds (xmin, xmax, ymin, ymax), columns, rows and the CRS the file
    declares (else `crs`).
    """
    try:
        metadata = _inspect_file(Path(path), MeteoFolderSpec("pre", Path(path).parent, crs=crs), variable).metadata
    except (KeyError, ValueError) as error:
        msg = (
            f"{Path(path).name}: no grid information (lower-left corner and cell size): {error}. "
                         "Select another file."
        )
        raise ValueError(msg) from error
    west, east, south, north = metadata.bounds
    half_x, half_y = abs(metadata.x_resolution) / 2, abs(metadata.y_resolution) / 2
    bounds = (west - half_x, east + half_x, south - half_y, north + half_y)
    return bounds, metadata.shape[1], metadata.shape[0], metadata.crs



def read_netcdf_first_slice(path, variable):
    """Return a variable's first time step as a 2-D array, north row first."""
    import numpy as np

    with _open_dataset(Path(path)) as ds:
        da = ds[variable]
        y_dim, x_dim = _spatial_dims(da)
        da = da.isel({dim: 0 for dim in da.dims if dim not in (y_dim, x_dim)}).transpose(y_dim, x_dim)
        y = _axis_values(ds, y_dim, "y")[1]
        values = np.asarray(da.values, dtype="float32")
    return values[::-1] if y is not None and len(y) > 1 and float(y[0]) < float(y[-1]) else values


def _inspect_file(
        path: Path,
        spec: MeteoFolderSpec,
        variable: str | None = None) -> _SpatialSignature:
    import numpy as np

    from mhm_tools.common.xarray_utils import calculate_mesh_resolution

    with _open_dataset(path) as ds:
        variable = variable or _find_variable(ds, spec)
        da = ds[variable]
        y_dim, x_dim = _spatial_dims(da)
        x_name, x_values = _axis_values(ds, x_dim, "x")
        y_name, y_values = _axis_values(ds, y_dim, "y")

        if x_values is None or y_values is None:
            lon_name, lon = _geographic_coordinate(ds, "lon", (y_dim, x_dim))
            lat_name, lat = _geographic_coordinate(ds, "lat", (y_dim, x_dim))
            if lon is None or lat is None:
                msg = f"{path.name}: could not identify spatial coordinates."
                raise ValueError(
                    msg)
            x_name, y_name = lon_name, lat_name
            x_values, y_values = lon, lat

        crs = _dataset_crs(ds, da, x_name, y_name) or spec.crs
        x_values = np.asarray(x_values, dtype="float64")
        y_values = np.asarray(y_values, dtype="float64")
        if x_values.ndim == 1 and y_values.ndim == 1:
            x_values, y_values = _centre_axes(ds, (y_dim, x_dim), x_name, y_name, x_values, y_values, crs)
            dx = _axis_resolution(x_values, f"{path.name}: coordinate {x_name}")
            dy = _axis_resolution(y_values, f"{path.name}: coordinate {y_name}")
            bounds = (
                float(np.nanmin(x_values)), float(np.nanmax(x_values)),
                float(np.nanmin(y_values)), float(np.nanmax(y_values)))
            shape = (len(y_values), len(x_values))
        elif x_values.shape == y_values.shape == (
                da.sizes[y_dim], da.sizes[x_dim]):
            dx = calculate_mesh_resolution(
                x_values, axis=1, label=f"{path.name}: coordinate {x_name}"
            )
            dy = calculate_mesh_resolution(
                y_values, axis=0, label=f"{path.name}: coordinate {y_name}"
            )
            bounds = (
                float(np.nanmin(x_values)), float(np.nanmax(x_values)),
                float(np.nanmin(y_values)), float(np.nanmax(y_values)))
            shape = x_values.shape
        else:
            msg = f"{path.name}: spatial coordinates are not a 2-D grid."
            raise ValueError(msg)

        unit = _coordinate_unit(ds, x_name, crs)
        metadata = SpatialMetadata(
            kind=spec.kind,
            source=spec.source,
            files=(path,),
            shape=shape,
            x_coordinate=x_name,
            y_coordinate=y_name,
            x_resolution=dx,
            y_resolution=dy,
            resolution=(dx + dy) / 2.0,
            bounds=bounds,
            crs=crs,
            unit=unit,
        )
        return _SpatialSignature(metadata, x_values, y_values)


def _find_variable(ds, spec):
    """Return the chosen source variable, else the output name or one known alias."""
    if spec.variable:
        if spec.variable not in ds.data_vars:
            msg = f"No variable {spec.variable!r}; the file has {list(ds.data_vars)}."
            raise ValueError(msg)
        return spec.variable
    kind = spec.kind
    candidates = [name for name in VARIABLES[kind][0] if name in ds.data_vars
                  and ("time" in ds[name].dims or "valid_time" in ds[name].dims)]
    if kind in candidates:
        return kind
    if len(candidates) != 1:
        msg = f"Expected one source variable; found {candidates or list(ds.data_vars)}."
        raise ValueError(msg)
    return candidates[0]


def _spatial_dims(da) -> tuple[str, str]:
    dims = [dim for dim in da.dims if dim not in {"time", "valid_time", "bnds"}]
    extra = [dim for dim in dims if da.sizes[dim] > 1]
    if len(extra) != 2:
        msg = f"Variable {da.name!r} must have exactly two spatial dimensions."
        raise ValueError(
            msg)
    return extra[-2], extra[-1]


def _axis_values(ds, dim: str, axis: str):
    import numpy as np

    candidates = (
        ("lon", "longitude", "xc", "x", "easting")
        if axis == "x" else
        ("lat", "latitude", "yc", "y", "northing")
    )
    for name in (dim, *candidates):
        if name not in ds.variables:
            continue
        values = np.asarray(ds[name].values)
        if values.ndim == 1 and ds[name].dims == (dim,):
            return name, values
    return dim, None


def _centre_axes(ds, dims, x_name: str, y_name: str, x_values, y_values, crs):
    """Return cell-centre 1-D axes.

    Lat/lon centres projected to `crs` place the grid, while the axes keep its order
    (lat/lon rows may be stored reversed); without them `xc`/`yc` are the left/upper
    edges.
    """
    import numpy as np

    x, y = np.asarray(x_values, dtype="float64"), np.asarray(y_values, dtype="float64")
    if x_name.lower() in {"lon", "longitude"} or x.size < 2 or y.size < 2:
        return x, y
    dx, dy = abs(float(x[1] - x[0])), abs(float(y[1] - y[0]))
    lon, lat = (_geographic_coordinate(ds, axis, dims)[1] for axis in ("lon", "lat"))
    if lon is not None and lat is not None and crs:
        from pyproj import Transformer

        lon, lat = np.asarray(lon, dtype="float64"), np.asarray(lat, dtype="float64")
        rims = (np.concatenate([grid[0], grid[-1], grid[:, 0], grid[:, -1]]) for grid in (lon, lat))
        px, py = Transformer.from_crs("EPSG:4326", crs, always_xy=True).transform(*rims)
        west, east = np.nanmin(px) - x.min(), np.nanmax(px) - x.max()
        south, north = np.nanmin(py) - y.min(), np.nanmax(py) - y.max()
        if max(abs(west), abs(east)) > dx or max(abs(south), abs(north)) > dy:
            msg = (
                f"The lat/lon of the data do not match its {x_name}/{y_name} in {crs}; "
                             "check the meteo CRS."
            )
            raise ValueError(msg)
        return x + west, y + south
    if (x_name.lower(), y_name.lower()) == ("xc", "yc"):
        return x + dx / 2, y - dy / 2
    return x, y


def _geographic_coordinate(ds, axis: str, dims: tuple[str, str]):
    aliases = ("lon", "longitude") if axis == "lon" else ("lat", "latitude")
    for name in aliases:
        if name in ds.variables and ds[name].dims == dims:
            return name, ds[name].values
    return axis, None


def _dataset_crs(ds, da, x_name: str, y_name: str) -> str | None:
    if x_name.lower() in {"lon", "longitude"} and y_name.lower() in {
            "lat", "latitude"}:
        return "EPSG:4326"

    mapping_name = da.attrs.get("grid_mapping")
    candidates = [ds.attrs]
    if mapping_name and mapping_name in ds.variables:
        candidates.append(ds[mapping_name].attrs)
    for name in ("spatial_ref", "crs"):
        if name in ds.variables:
            candidates.append(ds[name].attrs)
    for attrs in candidates:
        for key in ("spatial_ref", "crs_wkt", "crs"):
            value = attrs.get(key)
            if value:
                return str(value)
    return None


def _coordinate_unit(ds, x_name: str, crs: str | None) -> str:
    units = str(ds[x_name].attrs.get("units", "")) if x_name in ds else ""
    if "degree" in units.lower() or x_name.lower() in {"lon", "longitude"}:
        return "deg"
    if crs:
        try:
            from pyproj import CRS

            return _crs_unit(CRS.from_user_input(crs))
        except (ImportError, RuntimeError):  # no pyproj, or a CRS it can't read (CRSError)
            pass
    return units or ""


def _crs_unit(crs) -> str:
    if getattr(crs, "is_geographic", False):
        return "deg"
    name = crs.axis_info[0].unit_name.lower() if crs.axis_info else ""
    if name in {"metre", "meter"}:
        return "m"
    if "foot" in name or "feet" in name:
        return "ft"
    return name


def _validate_spatial_signatures(signatures: list[_SpatialSignature]) -> None:
    import numpy as np

    first = signatures[0]
    for current in signatures[1:]:
        if current.metadata.shape != first.metadata.shape:
            msg = (
                "NetCDF files do not share the same spatial dimensions: "
                f"{first.metadata.files[0].name}={first.metadata.shape}, "
                f"{current.metadata.files[0].name}={current.metadata.shape}."
            )
            raise ValueError(
                msg)
        if (
                np.shape(current.x_values) != np.shape(first.x_values)
                or np.shape(current.y_values) != np.shape(first.y_values)
                or not np.allclose(current.x_values, first.x_values,
                                   rtol=1e-9, atol=1e-9, equal_nan=True)
                or not np.allclose(current.y_values, first.y_values,
                                   rtol=1e-9, atol=1e-9, equal_nan=True)):
            msg = (
                "NetCDF files do not share identical spatial coordinates: "
                f"{first.metadata.files[0].name} and "
                f"{current.metadata.files[0].name}."
            )
            raise ValueError(
                msg)
        left, right = first.metadata.crs, current.metadata.crs
        if (left is None) != (right is None) or (
                left and right and not _same_crs(left, right)):
            msg = (
                "NetCDF files do not share the same CRS: "
                f"{first.metadata.files[0].name} and "
                f"{current.metadata.files[0].name}."
            )
            raise ValueError(
                msg)


def _validate_target_coverage(
        metadata: SpatialMetadata,
        target: _TargetGrid) -> None:
    """Reject disjoint grids; allow padding around overlapping input."""
    import numpy as np

    target_x = np.asarray(
        target.sample_lon
        if target.sample_lon is not None else target.lon,
        dtype="float64",
    )
    target_y = np.asarray(
        target.sample_lat
        if target.sample_lat is not None else target.lat,
        dtype="float64",
    )
    if (
            target.crs
            and metadata.crs
            and _same_crs(metadata.crs, target.crs)):
        target_x, target_y = get_header_cell_centers(target.header)
    elif metadata.crs and not _same_crs(metadata.crs, "EPSG:4326"):
        try:
            from pyproj import Transformer
        except Exception as exc:
            msg = "pyproj is required to validate projected meteorology data."
            raise RuntimeError(
                msg) from exc
        transform = Transformer.from_crs(
            "EPSG:4326", metadata.crs, always_xy=True)
        target_x, target_y = transform.transform(target_x, target_y)
    elif metadata.crs is None and metadata.unit not in {"deg", "degree"}:
        return

    west, east, south, north = metadata.bounds
    x_tolerance = metadata.x_resolution / 2.0 + 1e-9
    y_tolerance = metadata.y_resolution / 2.0 + 1e-9
    if (
            float(target_x.max()) < west - x_tolerance
            or float(target_x.min()) > east + x_tolerance
            or float(target_y.max()) < south - y_tolerance
            or float(target_y.min()) > north + y_tolerance):
        msg = (
            f"The model grid is outside the {VARIABLE_LABELS.get(metadata.kind, metadata.kind).lower()} "
                         "data: check the input's extent and CRS."
        )
        raise ValueError(msg)


def validate_coverage(metadata: SpatialMetadata, bounds, crs, l0_resolution=0.0) -> None:
    """Raise unless the source's cell edges contain the study `bounds`.

    `bounds` are xmin, xmax, ymin, ymax in `crs`, less 1% of an L0 cell for
    coordinate rounding (lat/lon stored as float32).
    """
    margin = 0.01 * float(l0_resolution)
    xmin, xmax, ymin, ymax = (float(value) + sign * margin for value, sign in zip(bounds, (1, -1, 1, -1), strict=True))
    source_crs = metadata.crs or ("EPSG:4326" if metadata.unit in {"deg", "degree"} else None)
    if not source_crs:
        return
    if crs and not _same_crs(source_crs, crs):
        from pyproj import Transformer
        xmin, ymin, xmax, ymax = Transformer.from_crs(crs, source_crs, always_xy=True).transform_bounds(
            xmin, ymin, xmax, ymax, densify_pts=21)
    west, east, south, north = metadata.bounds  # cell centres
    half_x, half_y = metadata.x_resolution / 2, metadata.y_resolution / 2
    tolerance = max(metadata.x_resolution, metadata.y_resolution) * 1e-6
    edges = (west - half_x, east + half_x, south - half_y, north + half_y)
    if (xmin < edges[0] - tolerance or xmax > edges[1] + tolerance
            or ymin < edges[2] - tolerance or ymax > edges[3] + tolerance):
        msg = (
            f"The {VARIABLE_LABELS.get(metadata.kind, metadata.kind).lower()} data "
            f"({', '.join(f'{value:.10g}' for value in edges)}) do not cover the study extent "
            f"({xmin:.10g}, {xmax:.10g}, {ymin:.10g}, {ymax:.10g}): use inputs that cover the whole domain."
        )
        raise ValueError(
            msg)



def grid_note(metadata: SpatialMetadata, grid, crs, level="L2") -> str:
    """Describe how an inspected input reaches the common `level` grid.

    `grid` is the saved grid metadata. The input is copied, shifted onto the lattice
    or resampled (nearest), as `_resample_ready` will do it. ValueError when it does
    not cover the common extent.
    """
    validate_coverage(metadata, grid["study_bounds"], crs, grid.get("l0_resolution", 0.0))
    header, source_crs = grid[f"{level.lower()}_header"], metadata.crs or ("EPSG:4326" if metadata.unit in {"deg", "degree"} else None)
    cell = float(header["cellsize"])
    if source_crs and crs and _same_crs(source_crs, crs) and all(
            math.isclose(value, cell, rel_tol=1e-6) for value in (metadata.x_resolution, metadata.y_resolution)):
        offsets = []
        for centre, corner in ((metadata.bounds[0], header["xllcorner"]), (metadata.bounds[2], header["yllcorner"])):
            steps = (centre - float(corner)) / cell - 0.5
            offsets.append((round(steps) - steps) * cell)
        if all(abs(value) <= 1e-6 * cell for value in offsets):
            return f"Covers the common extent; already on the {level} grid: copied."
        return (f"Covers the common extent; shifted by ({offsets[0]:.6g}, {offsets[1]:.6g}) onto the {level} grid "
                "(nearest cells), not resampled.")
    return (f"Covers the common extent; resampled (nearest) from {metadata.x_resolution:.6g} {metadata.unit} "
            f"({source_crs or 'no CRS'}) to the {level} grid ({cell:.6g}).")

def _resample_ready(da, ds, source_crs, target, np, _xr):
    from mhm_tools.common.xarray_utils import (
        reindex_to_header,
        resample_nearest_curvilinear,
        resample_nearest_rectilinear,
    )

    y_dim, x_dim = _spatial_dims(da)
    x_name, x_values = _axis_values(ds, x_dim, "x")
    y_name, y_values = _axis_values(ds, y_dim, "y")
    target_lon = np.asarray(target.lon, dtype="float64")
    target_lat = np.asarray(target.lat, dtype="float64")
    sample_lon = (
        np.asarray(target.sample_lon, dtype="float64")
        if target.sample_lon is not None else None
    )
    sample_lat = (
        np.asarray(target.sample_lat, dtype="float64")
        if target.sample_lat is not None else None
    )

    if x_values is not None and y_values is not None:
        source_crs = _dataset_crs(ds, da, x_name, y_name) or source_crs
        _lon_name, lon = _geographic_coordinate(ds, "lon", (y_dim, x_dim))
        _lat_name, lat = _geographic_coordinate(ds, "lat", (y_dim, x_dim))
        if source_crs is None and lon is not None and lat is not None:
            return resample_nearest_curvilinear(
                da, np.asarray(lon), np.asarray(lat),
                target_lon, target_lat, sample_lon, sample_lat)
        x_values, y_values = _centre_axes(ds, (y_dim, x_dim), x_name, y_name, x_values, y_values, source_crs)
        x_target, y_target = target_lon, target_lat
        target_crs_matches = bool(
            source_crs and target.crs and _same_crs(source_crs, target.crs)
        )
        if target_crs_matches:
            x_target, y_target = get_header_cell_centers(target.header)
        elif sample_lon is not None:
            x_target, y_target = sample_lon, sample_lat
            if source_crs and not _same_crs(source_crs, "EPSG:4326"):
                try:
                    from pyproj import Transformer
                except Exception as exc:
                    msg = (
                        "pyproj is required to resample projected "
                        "mHM-ready data."
                    )
                    raise RuntimeError(
                        msg
                    ) from exc
                transform = Transformer.from_crs(
                    "EPSG:4326",
                    source_crs,
                    always_xy=True,
                )
                x_target, y_target = transform.transform(
                    x_target,
                    y_target,
                )
            return resample_nearest_rectilinear(
                da,
                np.asarray(x_values),
                np.asarray(y_values),
                x_target,
                y_target,
                target_lon,
                target_lat,
            )
        elif source_crs and not _same_crs(source_crs, "EPSG:4326"):
            try:
                from pyproj import Transformer
            except Exception as exc:
                msg = "pyproj is required to resample projected mHM-ready data."
                raise RuntimeError(
                    msg) from exc
            transform = Transformer.from_crs(
                "EPSG:4326", source_crs, always_xy=True)
            center_lat = float((target_lat.min() + target_lat.max()) / 2.0)
            center_lon = float((target_lon.min() + target_lon.max()) / 2.0)
            x_target = np.asarray([
                transform.transform(lon, center_lat)[0]
                for lon in target_lon])
            y_target = np.asarray([
                transform.transform(center_lon, lat)[1]
                for lat in target_lat])

        if target_crs_matches:
            x_values, y_values = _shift_onto_header(x_values, y_values, target.header)
        source = da.assign_coords(
            {x_dim: np.asarray(x_values), y_dim: np.asarray(y_values)})
        source = source.sortby(x_dim).sortby(y_dim)
        if target_crs_matches and _axes_match_header(
                x_values, y_values, target.header):
            result = reindex_to_header(source, x_dim, y_dim, target.header)
        else:
            result = source.interp(
                {x_dim: x_target, y_dim: y_target}, method="nearest")
        result = result.reset_coords(drop=True).rename(  # the source lat/lon go: the target ones follow
            {x_dim: "lon", y_dim: "lat"})
        return result.assign_coords(lon=target_lon, lat=target_lat).transpose(
            "time", "lat", "lon")

    _lon_name, lon = _geographic_coordinate(ds, "lon", (y_dim, x_dim))
    _lat_name, lat = _geographic_coordinate(ds, "lat", (y_dim, x_dim))
    if lon is None or lat is None:
        msg = "mHM-ready data has no usable spatial coordinates."
        raise ValueError(msg)
    return resample_nearest_curvilinear(
        da,
        np.asarray(lon),
        np.asarray(lat),
        target_lon,
        target_lat,
        sample_lon,
        sample_lat,
    )


def _attach_target_coordinates(ds, variable: str, target: _TargetGrid):
    """Attach exact 2-D cell-centre coordinates for projected target grids."""
    if target.sample_lon is None or target.sample_lat is None:
        return ds

    ds = ds.assign_coords({
        "lon2d": (
            ("lat", "lon"),
            target.sample_lon,
            {
                "standard_name": "longitude",
                "units": "degrees_east",
            },
        ),
        "lat2d": (
            ("lat", "lon"),
            target.sample_lat,
            {
                "standard_name": "latitude",
                "units": "degrees_north",
            },
        ),
    })
    ds["lon"].attrs = {"standard_name": "projection_x_coordinate", "axis": "X"}
    ds["lat"].attrs = {"standard_name": "projection_y_coordinate", "axis": "Y"}
    ds[variable].encoding.pop("coordinates", None)
    ds[variable].attrs["coordinates"] = "lat2d lon2d"
    described = _crs_description(target.crs)
    if described is not None:
        unit, crs_attrs = described
        ds["lon"].attrs["units"] = ds["lat"].attrs["units"] = unit
        ds["crs"] = 0
        ds["crs"].attrs.update(crs_attrs)
        ds[variable].attrs["grid_mapping"] = "crs"
    return ds


@lru_cache(maxsize=8)
def _crs_description(crs_text: str):
    """Return (axis unit, CF grid-mapping attributes) for one CRS, once.

    This is constant for a whole run but used to be rebuilt for every written
    block, which on a multi-decade hourly record meant tens of thousands of
    pyproj calls from a Qt worker thread.
    """
    if not crs_text:
        return None
    try:
        from pyproj import CRS

        crs = CRS.from_user_input(crs_text)
        attrs = dict(crs.to_cf())
        attrs["spatial_ref"] = crs.to_wkt()
        return (crs.axis_info[0].unit_name if crs.axis_info else "metre"), attrs
    except Exception:
        return None


def format_meteo_data(
    spec,
    output_file,
    grid_file,
    *,
    grid_crs=None,
    start_time=None,
    end_time=None,
    metadata=None,
    progress_callback=None,
    cancelled=None,
    compression=None,
) -> int:
    """Format one meteorological or observation variable onto an mHM grid.

    Parameters
    ----------
    spec : MeteoFolderSpec
        Input folder and glob, source variable, unit, temporal resolutions,
        aggregation and accumulation of the variable `spec.kind`.
    output_file : str or pathlib.Path
        NetCDF to write; its stem names the written variable.
    grid_file : str or pathlib.Path
        ASCII grid header of the target (L2) grid.
    grid_crs : str, optional
        CRS of the target grid; None treats it as lon/lat.
    start_time, end_time : str or numpy.datetime64, optional
        Start and exclusive end of the intervals to keep, each clamped to the
        complete output periods of the input, which are the default.
    metadata : SpatialMetadata, optional
        `read_meteo_metadata(spec)`, when the caller already has it.
    progress_callback : callable, optional
        Called with `spec.kind` after each input file.
    cancelled : callable, optional
        Returns True to stop; checked between blocks.
    compression : mhm_tools.common.netcdf.NetcdfCompression, optional
        Output compression; None keeps zlib level 4 with shuffle.

    Returns
    -------
    int
        Number of time steps written.
    """
    import numpy as np

    from mhm_tools.common.esri_grid import read_header
    from mhm_tools.common.merge import write_time_block

    spec = spec.normalized()
    validate_spec(spec)
    target = _create_target_grid(read_header(grid_file), grid_crs)
    metadata = metadata or read_meteo_metadata(spec, cancelled=cancelled)
    _validate_target_coverage(metadata, target)
    first = np.datetime64(metadata.start_time)
    last = period_bounds(output_periods(
        np.datetime64(metadata.end_time), spec.output_temporal_resolution))[1]
    window = (first if start_time is None else max(first, np.datetime64(start_time)),
              last if end_time is None else min(last, np.datetime64(end_time)))
    output_file = Path(output_file)
    name = output_file.stem
    # a failed or cancelled run must not leave a truncated file or destroy the last good one
    partial = output_file.with_name(f".{output_file.name}.part")
    count, handle = 0, None
    try:
        try:
            for block in _prepared_blocks(spec, metadata, target, window=window,
                                          on_file=progress_callback, cancelled=cancelled):
                _check_cancel(cancelled)
                _assert_matches_header(block, spec.kind, target.header)
                handle = write_time_block(block.rename({spec.kind: name}), name, partial,
                                          count, handle, compression)
                count += block.sizes["time"]
                block.close()
        finally:
            if handle is not None:
                handle.close()
        if not count:
            msg = "No time steps to write."
            raise ValueError(msg)
        partial.replace(output_file)
    finally:
        partial.unlink(missing_ok=True)
    return count
