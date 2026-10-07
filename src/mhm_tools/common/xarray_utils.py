"""Provides basic xarray utils."""

import logging
import warnings
from typing import Optional, Union

import matplotlib as mpl
import numpy as np
import pandas as pd
import xarray as xr
from scipy.stats import rankdata, spearmanr
from scipy.stats import t as student_t

from mhm_tools.common.constants import (
    LAT_KEYS,
    LON_KEYS,
    TIME_KEYS,
    WMO_REGION_BOUNDS,
)
from mhm_tools.common.logger import ErrorLogger
from mhm_tools.common.netcdf import (
    generate_bounds,
    generate_bounds_for_all_coords,
    get_netcdf_metadata_data_vars,
)
from mhm_tools.common.time_utils import get_data_coverage, timedelta_to_alias
from mhm_tools.common.units import calculate_conversion_factor

logger = logging.getLogger(__name__)
MIN_PAIRS_FOR_CORRELATION = 2
# Measured working set of calculate_spearman_over_time, which needs several
# float64 copies of what it is handed. Callers size their blocks with it.
SPEARMAN_BYTES_PER_ELEMENT = 66


def create_mask_from_polygon(data_array, vertices):
    """Create a boolean mask for grid cells whose center falls inside a polygon.

    The input `data_array` is a 2D array with `lat` and `lon` coordinates; the
    mask is True for cells whose (lon, lat) center falls inside the polygon
    defined by `vertices`.

    Parameters
    ----------
    data_array : xarray.DataArray
        2D data array with coordinates `lat` and `lon`.
    vertices : sequence[tuple[float, float]]
        Polygon vertices as (lon, lat) pairs.

    Returns
    -------
    numpy.ndarray
        Boolean mask with the same shape as `data_array`, True inside the polygon.
    """
    polygon = mpl.path.Path(vertices)
    # mask out only the values in data_array that fall within bbox of polygon, convert them to points
    bbox = polygon.get_extents()
    bbox_lon_mask = (bbox.xmin < data_array.lon) & (bbox.xmax > data_array.lon)
    bbox_lat_mask = (bbox.ymin < data_array.lat) & (bbox.ymax > data_array.lat)
    lon2d, lat2d = np.meshgrid(
        data_array.lon[bbox_lon_mask], data_array.lat[bbox_lat_mask]
    )
    points = np.hstack((lon2d.reshape(-1, 1), lat2d.reshape(-1, 1)))
    # mask out the values
    bbox_mask = polygon.contains_points(points).reshape(
        int(bbox_lat_mask.sum()), int(bbox_lon_mask.sum())
    )

    # global mask, set to False
    mask = np.zeros_like(data_array.data, dtype=bool)
    # insert the local mask into the global one
    mask[np.ix_(bbox_lat_mask, bbox_lon_mask)] = bbox_mask
    return mask


def combine_region_grids(priority_grid, fallback_grid):
    """Combine two region grids, preferring priority_grid and filling its gaps from fallback_grid.

    Both grids are expected to use NaN for unset cells on the same lat/lon grid.

    Parameters
    ----------
    priority_grid : xarray.DataArray
        Region grid whose values are kept wherever finite.
    fallback_grid : xarray.DataArray
        Region grid used only where `priority_grid` is NaN.

    Returns
    -------
    xarray.DataArray
        Combined region grid.
    """
    return fallback_grid.where(~np.isfinite(priority_grid), priority_grid)


def create_valid_data_mask(data_array, treat_zero_as_missing=True):
    """Return a boolean validity mask, treating NaN (and optionally 0) as missing.

    Guards against the common bug where a mask stores 0/1 without a declared
    `_FillValue`: checking only `np.isfinite` then treats every 0 cell as valid.

    Parameters
    ----------
    data_array : xarray.DataArray or numpy.ndarray
        Data to derive validity from.
    treat_zero_as_missing : bool, optional
        Also treat exact 0 values as missing, by default True.

    Returns
    -------
    xarray.DataArray or numpy.ndarray
        Boolean mask, True where data is valid.
    """
    valid = np.isfinite(data_array)
    if treat_zero_as_missing:
        valid = valid & (data_array != 0)
    return valid


def materialize_if_within_budget(da, max_memory_gib, label):
    """Compute a lazy field once when it fits the memory budget.

    A field every later step reads again is rebuilt from its whole graph each
    time it is touched, which for a record that was read, regridded and
    resampled means doing all of that per output. Computing it once trades that
    for one array. A field too large for the budget stays lazy and correct,
    only slower.

    Args:
        da: Dataset or DataArray, lazy or not.
        max_memory_gib: Memory budget in GiB.
        label: Name used in the log messages.

    Returns
    -------
        The computed or the unchanged object.
    """
    estimated_gib = da.nbytes / 1024**3
    if estimated_gib <= max_memory_gib:
        logger.info(f"Computing {label} once ({estimated_gib:.2f} GiB).")
        return da.compute()
    logger.warning(
        f"Keeping {label} lazy because {estimated_gib:.2f} GiB exceeds the budget "
        f"of {max_memory_gib} GiB. Every step that reads it rebuilds it, so this "
        f"is markedly slower; raise --available-mem if the memory is there."
    )
    return da


def normalize_longitude_range(ds, lon_key="lon"):
    """Shift a global 0 to 360 longitude axis onto -180 to 180.

    A satellite product often counts longitude from 0 eastwards, while a model
    grid and the evaluation regions run from -180. Comparing the two then finds
    no overlap at all. On a global axis the shift is a rotation, so the cells
    are only put back in ascending order.

    A regional axis reaching past 180 crosses the antimeridian, and shifting it
    would tear the domain into a piece at each end of the axis with a gap
    between them, which every later crop would read as a near global extent.
    Such an axis is refused instead. An axis that already runs from -180 but
    crosses the antimeridian cannot be made ascending at all and is only warned
    about, because a label based selection on it is unreliable. An axis reaching
    below -180 or above 360 holds no longitudes at all, such as the x axis of a
    projected grid in metres, and is left unchanged.

    Args:
        ds: Dataset or DataArray with a longitude coordinate.
        lon_key: Name of the longitude coordinate.

    Returns
    -------
        The object on a -180 to 180 axis, unchanged when it already is or when
        the axis holds no longitudes.
    """
    if lon_key not in getattr(ds, "coords", {}):
        return ds
    longitudes = np.asarray(ds[lon_key].values, dtype=float)
    if longitudes.size < 2:
        return ds
    lowest, highest = float(np.nanmin(longitudes)), float(np.nanmax(longitudes))
    if lowest < -180.0 or highest > 360.0:
        logger.debug(
            f"The axis {lon_key!r} runs {lowest:g}..{highest:g} and holds no "
            f"longitudes, so it is left unchanged."
        )
        return ds
    if highest <= 180.0:
        if not np.all(np.diff(longitudes) > 0):
            logger.warning(
                f"The longitude axis runs {longitudes[0]:g}..{longitudes[-1]:g} "
                f"and is not ascending, so it crosses the antimeridian. A label "
                f"based selection on it is unreliable."
            )
        return ds
    step = float(np.nanmedian(np.abs(np.diff(longitudes))))
    if highest - lowest + step < 360.0 - step:
        msg = (
            f"The longitude axis runs {lowest:g}..{highest:g}, which reaches "
            f"past 180 without covering the globe, so the domain crosses the "
            f"antimeridian. Shifting it onto -180..180 would split it into a "
            f"piece at each end of the axis and every later crop would read "
            f"that as a near global extent. Provide the domain on a -180..180 "
            f"axis instead."
        )
        with ErrorLogger(logger):
            raise ValueError(msg)
    logger.info(
        f"Shifting the global longitude axis from 0..360 to -180..180 "
        f"({lowest:g}..{highest:g})."
    )
    shifted = ((longitudes + 180.0) % 360.0) - 180.0
    return ds.assign_coords({lon_key: shifted}).sortby(lon_key)


def normalize_lat_lon(
    ds: Union[xr.Dataset, xr.DataArray],
    lat_key: Optional[str] = None,
    lon_key: Optional[str] = None,
    new_lat_key: str = "lat",
    new_lon_key: str = "lon",
    raise_exceptions: bool = False,
    log_warning: bool = False,
) -> Union[xr.Dataset, xr.DataArray]:
    """
    Normalize latitude and longitude dimension and coordinate names to 'lat' and 'lon'.

    Handles both dimensions and coordinate variables.
    """
    try:
        rename_dict = {}
        # A file can store its coordinates as data variables, which leaves the
        # dimension without an index. Promoting them first lets the rename below
        # see them at all.
        if isinstance(ds, xr.Dataset):
            coord_like_vars = [
                name
                for name in ds.data_vars
                if name in LAT_KEYS + LON_KEYS and ds[name].ndim == 1
            ]
            if coord_like_vars:
                logger.info(f"Promoting the coordinate variables {coord_like_vars}.")
                ds = ds.set_coords(coord_like_vars)
        if lat_key is None:
            lat_key = get_coord_key(ds, lat=True)
        if lon_key is None:
            lon_key = get_coord_key(ds, lon=True)

        coords_and_dims = list(ds.coords) + list(ds.dims)
        if log_warning and (lat_key != new_lat_key or lon_key != new_lon_key):
            logger.warning(
                f"The coordinates were normalised from {lon_key}->{new_lon_key} and {lat_key}->{new_lat_key}"
            )

        def _needs_rename(key, new_key):
            """Tell whether a coordinate can be renamed to the target name."""
            if key is None or key == new_key or key not in coords_and_dims:
                return False
            # a coordinate of that name already is the normalized one
            if new_key in ds.coords:
                return False
            # the target may be the bare dimension this coordinate spans, and
            # renaming then turns it into a dimension coordinate; a dimension of
            # that name the coordinate does not span is unrelated and would clash
            return new_key not in ds.dims or new_key in getattr(ds[key], "dims", ())

        def _aliased_coord_of_dim(dim, keys):
            """Return a coordinate named like an alias that spans only `dim`."""
            for key in keys:
                if key in ds.coords and tuple(ds[key].dims) == (dim,):
                    return key
            return None

        if _needs_rename(lat_key, new_lat_key):
            rename_dict[lat_key] = new_lat_key
        if _needs_rename(lon_key, new_lon_key):
            rename_dict[lon_key] = new_lon_key
        # A dimension can already carry the target name while its coordinate
        # keeps an alias, which leaves the dimension without an index and makes
        # every label based selection fall back to positional indexing.
        for new_key, alias_keys in ((new_lat_key, LAT_KEYS), (new_lon_key, LON_KEYS)):
            if new_key in ds.dims and new_key not in ds.coords:
                alias = _aliased_coord_of_dim(new_key, alias_keys)
                if alias is not None and alias not in rename_dict:
                    rename_dict[alias] = new_key
        if rename_dict:
            logger.info(f"Normalizing coordinate names: {rename_dict}")

        with warnings.catch_warnings():
            # renaming a coordinate onto its own dimension name warns that it is
            # not an index yet, which the set_index below is what fixes
            warnings.filterwarnings("ignore", message="rename .* does not create")
            normalized = ds.rename(rename_dict)
        # a renamed coordinate is not an index yet, and without one every label
        # based selection silently falls back to positional indexing
        missing_index = [
            key
            for key in (new_lat_key, new_lon_key)
            if key in normalized.dims
            and key in normalized.coords
            and key not in normalized.indexes
        ]
        if missing_index:
            logger.info(f"Indexing the normalized coordinates {missing_index}.")
            normalized = normalized.set_index({key: key for key in missing_index})
        return normalize_longitude_range(normalized, lon_key=new_lon_key)
    except Exception as e:
        if raise_exceptions:
            with ErrorLogger(logger):
                raise (e)
        else:
            logger.warning(f"Exception in normalize lat lon: {e}")
            return ds


def snap_to_target(
    ds: Union[xr.Dataset, xr.DataArray],
    lat_key: str,
    lon_key: str,
    target_lat_array,
    target_lon_array,
    new_lat_key: str = "lat",
    new_lon_key: str = "lon",
) -> Union[xr.Dataset, xr.DataArray]:
    """
    Rename latitude/longitude dimensions and assign exact target coordinates.

    This is useful after nearest-neighbor grid matching, where selected values
    are correct by position but coordinate labels still need to match exactly
    for xarray alignment.
    """
    ds = normalize_lat_lon(
        ds,
        lat_key=lat_key,
        lon_key=lon_key,
        new_lat_key=new_lat_key,
        new_lon_key=new_lon_key,
        raise_exceptions=True,
    )
    return ds.assign_coords(
        {
            new_lat_key: np.asarray(target_lat_array),
            new_lon_key: np.asarray(target_lon_array),
        }
    )


def regrid_mask(
    mask_ds,
    lon_key_mask,
    lat_key_mask,
    target_lon,
    target_lat,
    mask_key=None,
    lon_key_target=None,
    lat_key_target=None,
    target_res=None,
    mask_res=None,
):
    """Regrid a xarray mask dataset mask_ds to the resolution of a second dataset ds2."""

    def _select_mask_var(mask_obj):
        if isinstance(mask_obj, xr.DataArray):
            return mask_obj
        if isinstance(mask_obj, xr.Dataset):
            key = mask_key or get_single_data_var(mask_obj)
            if key is None:
                no_key_msg = "Mask dataset has multiple data_vars; provide mask_key."
                with ErrorLogger(logger):
                    raise ValueError(no_key_msg)
            return mask_obj[key]
        wrong_type_msg = f"Unsupported mask type: {type(mask_obj)}"
        with ErrorLogger(logger):
            raise ValueError(wrong_type_msg)

    if lon_key_target is None:
        lon_key_target = lon_key_mask
    if lat_key_target is None:
        lat_key_target = lat_key_mask
    mask_lon = mask_ds[lon_key_mask].data
    mask_lat = mask_ds[lat_key_mask].data
    if mask_res is None or target_res is None:
        from mhm_tools.common.resolution_handler import get_file_res

        if mask_res is None:
            mask_res = get_file_res(lon=mask_lon, lat=mask_lat)
        if target_res is None:
            target_res = get_file_res(lon=target_lon, lat=target_lat)
    if (target_res - mask_res) > 1e-5:
        if target_res % mask_res > 1e-5:
            logger.warning(
                f"Target resolution {target_res} is not an integer muptiple of mask resolution {mask_res}. Factor: {target_res / mask_res}"
            )
        results = np.full((len(target_lat), len(target_lon)), 0.0)
        for i, lat in enumerate(target_lat):
            for j, lon in enumerate(target_lon):
                for n, mlat in enumerate(mask_lat):
                    if mlat < (lat - target_res / 2) or mlat > (lat + target_res / 2):
                        continue
                    for m, mlon in enumerate(mask_lon):
                        if mlon < lon - target_res / 2 or mlon > lon + target_res / 2:
                            continue
                        if mask_key is not None:
                            results[i][j] += mask_ds[mask_key].data[n, m]
                        else:
                            results[i][j] += mask_ds.data[n, m]
        results = np.where(np.isfinite(results), results, 0.0)
        max_result = np.max(results) if results.size else 0.0
        if max_result <= 0:
            logger.warning("Regridded mask has no positive cells on target grid.")
            return xr.DataArray(
                results,
                dims=[lat_key_target, lon_key_target],
                coords={lat_key_target: target_lat, lon_key_target: target_lon},
            )
        results /= max_result
        mask = results > 1e-3
        results[mask] = 1
        results[~mask] = 0
        return xr.DataArray(
            results,
            dims=[lat_key_target, lon_key_target],
            coords={lat_key_target: target_lat, lon_key_target: target_lon},
        )
    if abs(target_res - mask_res) <= 1e-5:
        logger.debug("Target resolution equals mask resolution (within tolerance).")

        try:
            # quick path: if coords are almost equal, reuse data but snap labels
            if (
                len(mask_lon) == len(target_lon)
                and len(mask_lat) == len(target_lat)
                and np.allclose(mask_lon, target_lon, rtol=0, atol=1e-9)
                and np.allclose(mask_lat, target_lat, rtol=0, atol=1e-9)
            ):
                return snap_to_target(
                    _select_mask_var(mask_ds),
                    lat_key=lat_key_mask,
                    lon_key=lon_key_mask,
                    target_lat_array=target_lat,
                    target_lon_array=target_lon,
                    new_lat_key=lat_key_target,
                    new_lon_key=lon_key_target,
                )

            tol = max(mask_res, target_res) * 1e-3  # generous but safe snapping tol
            reindexed = mask_ds.reindex(
                {
                    lat_key_mask: np.asarray(target_lat),
                    lon_key_mask: np.asarray(target_lon),
                },
                method="nearest",
                tolerance=tol,
            )
            min_lon = min(len(mask_lon), len(target_lon))
            min_lat = min(len(mask_lat), len(target_lat))
            logger.debug(
                f"Reindexed mask to target grid with tolerance {tol}; "
                f"delta lon={float(np.nanmax(np.abs(mask_lon[:min_lon] - target_lon[:min_lon]))):.3g}, "
                f"delta lat={float(np.nanmax(np.abs(mask_lat[:min_lat] - target_lat[:min_lat]))):.3g}"
            )
            return snap_to_target(
                _select_mask_var(reindexed),
                lat_key=lat_key_mask,
                lon_key=lon_key_mask,
                target_lat_array=target_lat,
                target_lon_array=target_lon,
                new_lat_key=lat_key_target,
                new_lon_key=lon_key_target,
            )
        except Exception:
            logger.debug(
                "Mask reindex to target grid failed; using original mask", exc_info=True
            )
            return _select_mask_var(mask_ds)
    else:
        msg = "mask coarser than file not yet implemented"
        with ErrorLogger(logger):
            raise Exception(msg)


def get_coord_key(
    ds, lat=False, lon=False, time=False, raise_exception=True, is_retry=False
):
    """Return the lat or lon coordinate name used in the xarray dataset."""
    if lat + lon + time != 1:
        with ErrorLogger(logger):
            msg = f"one of lon, lat or time should be true but lon={lon} and lat={lat} and time={time}"
            raise ValueError(msg)
    ds_dims = ds.dims if isinstance(ds, xr.DataArray) or is_retry else ds.coords
    # first check if there are dimensions with a fitting axis attribute
    try:
        for dim in ds_dims:
            if (
                (lat and ds[dim].axis == "Y")
                or (lon and ds[dim].axis == "X")
                or (time and ds[dim].axis == "T")
            ):
                return dim
    except AttributeError:
        pass
    # then select possible keys from the following lists and try them until a fitting one is found.
    if lat:
        keys = LAT_KEYS
    elif lon:
        keys = LON_KEYS
    else:
        keys = TIME_KEYS
    for key in keys:
        if key in ds_dims and len(ds[key].shape) == 1:
            return key
    for key in keys:
        if key in ds_dims:
            logger.warning(
                f"{type(ds)} contains key: {key} but ds[key] has shape {ds[key].shape}."
            )
            return key
    if not is_retry and isinstance(ds, xr.Dataset):
        logger.warning(
            f"{type(ds)} does not contain fitting coordinates. Trying again looking for dimensions"
        )
        return get_coord_key(
            ds,
            lat=lat,
            lon=lon,
            time=time,
            raise_exception=raise_exception,
            is_retry=True,
        )
    if raise_exception:
        with ErrorLogger(logger):
            msg = f"None of {keys} in {type(ds).__name__} keys {ds_dims}."
            raise ValueError(msg)
    return None


def get_single_data_var(ds: xr.Dataset, proposed_vars: Optional[list] = None):
    """Get the data var name from a dataset that only contains one data variable."""
    metadata_vars = get_netcdf_metadata_data_vars(ds)
    data_vars = [name for name in ds.data_vars if name not in metadata_vars]
    if isinstance(proposed_vars, list):
        for var in data_vars:
            if var in proposed_vars:
                return var
    if len(data_vars) > 1:
        # remove coords without mutating while iterating
        coords = [
            coord for coord in LAT_KEYS + LON_KEYS + TIME_KEYS if coord in data_vars
        ]
        for coord in coords:
            logger.debug(f"Removing coordinate data_var {coord} from consideration.")
        data_vars = [dv for dv in data_vars if dv not in coords]
        logger.debug(f"data_vars after removing coords: {data_vars}")

        # remove bounds variables; iterate over snapshot to avoid skipping
        bounds_removed = []
        filtered = []
        for data_var in data_vars.copy():
            if "bounds" in ds[data_var].attrs or data_var.endswith("_bnds"):
                bounds_removed.append(data_var)
            else:
                filtered.append(data_var)
                logger.debug(f"Keeping data_var {data_var}.")
        for bnd in bounds_removed:
            logger.debug(f"Removing bounds data_var {bnd} from consideration.")
        data_vars = filtered
        logger.debug(f"data_vars after removing bounds: {data_vars}")

        if len(data_vars) > 1:
            logger.error(f"Only single data_var allowed but has {data_vars}")
            return None
        if len(data_vars) == 0:
            logger.error("No datavar that is not coordinate.")
            return None
    logger.debug(f"data_vars: {data_vars}")
    if isinstance(data_vars, list) and len(data_vars) == 1:
        return data_vars[0]
    return None


def induce_data_var_from_file_name(ds, file_path):
    """Check if one of the data_vars is part of the file name and select it as the most probable data_var."""
    logger.info("Searching for more than one datavar by comparing with file name.")
    name = file_path.stem
    data_vars = list(ds.data_vars)
    logger.debug(f"{name} - {data_vars}")
    for dv in data_vars:
        if dv in name:
            return dv
        if name in dv:
            return dv
    return None


def get_overlapping_time_slice(input_ds, ref_ds):
    """Return the inclusive overlapping time window of two time-indexed objects.

    The overlap is computed from the first/last non-all-NaN timesteps of each
    input. Time-of-day components are dropped before creating the slice so
    small sub-daily offsets (e.g. 00:00 vs 11:00 for daily data) still produce
    a common calendar-day window.

    Args:
        input_ds: Simulation (or first) xarray object with a ``time`` dimension.
        ref_ds: Reference (or second) xarray object with a ``time`` dimension.

    Returns
    -------
        ``slice(start, end)`` with ``pandas.Timestamp`` endpoints.
    """
    input_non_nan_time = input_ds.dropna(dim="time", how="all").time.data
    reference_non_nan_time = ref_ds.dropna(dim="time", how="all").time.data

    if input_non_nan_time.size == 0 or reference_non_nan_time.size == 0:
        msg = (
            "Cannot determine temporal overlap because one dataset has no valid "
            "non-NaN timesteps after preprocessing. "
            f"Input valid timesteps: {input_non_nan_time.size}; "
            f"Reference valid timesteps: {reference_non_nan_time.size}. "
            "This usually means the selected variable, mask, spatial crop, "
            "regridding, or time resampling removed all usable data. Check the "
            "input/ref variable names, mask coverage, coordinate slice, and target "
            "time frequency."
        )
        with ErrorLogger(logger):
            raise ValueError(msg)

    logger.debug(f"input {input_non_nan_time[0]} till {input_non_nan_time[-1]}")
    logger.debug(f"ref {reference_non_nan_time[0]} till {reference_non_nan_time[-1]}")

    # Infer temporal buckets. We compare on calendar buckets (day/month/week)
    # instead of exact timestamps so small offsets like 00:00 vs 11:00 do not
    # break overlap detection for daily/monthly data.
    _, input_alias = timedelta_to_alias(input_ds)
    _, reference_alias = timedelta_to_alias(ref_ds)

    def _alias_to_bucket(alias):
        alias = str(alias).upper()
        if alias in {"ME", "MS", "M"}:
            return "M"
        if alias.startswith("W"):
            return "W"
        if alias == "D":
            return "D"
        return None

    input_bucket = _alias_to_bucket(input_alias)
    reference_bucket = _alias_to_bucket(reference_alias)
    bucket = (
        "M"
        if "M" in (input_bucket, reference_bucket)
        else (
            "W"
            if "W" in (input_bucket, reference_bucket)
            else "D" if "D" in (input_bucket, reference_bucket) else None
        )
    )

    # Find overlapping range between both available periods.
    if bucket is None:
        # Sub-daily case: use exact timestamps.
        overlap_start = pd.to_datetime(
            max(input_non_nan_time[0], reference_non_nan_time[0])
        )
        overlap_end = pd.to_datetime(
            min(input_non_nan_time[-1], reference_non_nan_time[-1])
        )
    else:
        # Bucketed comparison (D/W/M): compare period overlap and expand to
        # full bucket boundaries for robust `.sel(time=slice(...))`.
        input_period = pd.DatetimeIndex(input_non_nan_time).to_period(bucket)
        reference_period = pd.DatetimeIndex(reference_non_nan_time).to_period(bucket)
        overlap_start_period = max(input_period.min(), reference_period.min())
        overlap_end_period = min(input_period.max(), reference_period.max())
        overlap_start = overlap_start_period.start_time
        overlap_end = overlap_end_period.end_time

    if overlap_end < overlap_start:
        logger.warning(
            "The two datasets are not overlapping. "
            f"Sim data has non nan data from {input_non_nan_time[0]} to {input_non_nan_time[-1]} "
            f"and obs from {reference_non_nan_time[0]} to {reference_non_nan_time[-1]}."
        )
    logger.info(
        f"Cropping data to timeframe {overlap_start} to {overlap_end}"
        + (f" using {bucket} buckets." if bucket is not None else ".")
    )
    return slice(overlap_start, overlap_end)


def crop_ds(
    ds: xr.Dataset,
    lon_min: float,
    lon_max: float,
    lat_min: float,
    lat_max: float,
    lon_name: str = "lon",
    lat_name: str = "lat",
) -> xr.Dataset:
    """Crop an xarray.Dataset to the given lon/lat bounds, handling coordinate order."""
    # ensure min < max
    lon_low, lon_high = sorted([lon_min, lon_max])
    lat_low, lat_high = sorted([lat_min, lat_max])

    # grab the coordinate arrays by name
    lon_vals = ds[lon_name].data
    lat_vals = ds[lat_name].data

    # if the coordinate axis is ascending, slice low->high; else high->low
    if lon_vals[0] <= lon_vals[-1]:
        lon_slice = slice(lon_low, lon_high)
    else:
        lon_slice = slice(lon_high, lon_low)

    if lat_vals[0] <= lat_vals[-1]:
        lat_slice = slice(lat_low, lat_high)
    else:
        lat_slice = slice(lat_high, lat_low)

    # select using a dict so named dims are respected
    return ds.sel({lon_name: lon_slice, lat_name: lat_slice})


def climatology(data):
    """Calculate the climatology from an xarray DataArray."""
    if "time" not in data.dims or data.sizes["time"] == 0:
        msg = "Input data for climatology calculation has no valid time dimension."
        with ErrorLogger(logger):
            raise ValueError(msg)
    # group into monthly mean data
    data_clim = data.groupby("time.month").mean(dim="time", skipna=True)
    # Ensure the climatology has all 12 months, filling missing months with NaNs
    return data_clim.reindex(month=np.arange(1, 13), fill_value=np.nan)


def get_clim_from_ds(ds, input_var=None, factor=1):
    """Calculate climatology from a Dataset or DataArray.

    Multiplies the selected data by `factor` before computing the monthly
    climatology.
    """
    data = ds * factor if input_var is None else ds[input_var] * factor
    return climatology(data)


def normalize_time(obj, new_time_key="time"):
    """Rename the time dimension and coordinate to ``time``.

    Args:
        obj: Dataset or DataArray to rename.
        new_time_key: Target time name.

    Returns
    -------
        The renamed object.
    """
    time_key = get_coord_key(obj, time=True)
    if time_key == new_time_key:
        return obj
    logger.info(f"Normalizing time coordinate name: {time_key} -> {new_time_key}")
    return obj.rename({time_key: new_time_key})


def crop_to_region(da, region_name):
    """Crop a field to the bounds of a WMO evaluation region.

    Args:
        da: DataArray with ``lat`` and ``lon`` coordinates.
        region_name: Key of ``WMO_REGION_BOUNDS``.

    Returns
    -------
        The DataArray reduced to the region's bounds.
    """
    bounds = WMO_REGION_BOUNDS[region_name]
    lon_slice = bounds["lon_slice"]
    lat_slice = bounds["lat_slice"]
    return crop_ds(da, lon_slice.start, lon_slice.stop, lat_slice.start, lat_slice.stop)


def calculate_region_medians(ds, variables, region_names):
    """Calculate the median of every variable per WMO region and for the domain.

    Inf counts as missing. A region without any valid cell is left out.

    Args:
        ds: Dataset with ``lat`` and ``lon`` coordinates holding the variables.
        variables: Names of the variables to summarise.
        region_names: Keys of ``WMO_REGION_BOUNDS``.

    Returns
    -------
        DataFrame with a first row "Global" for the whole domain and one row
        per region, holding ``region``, the number of grid ``cells`` with a
        valid value and the median of every variable.
    """
    rows = []
    for region_name in [None, *region_names]:
        region_ds = ds[variables]
        if region_name is not None:
            region_ds = crop_to_region(region_ds, region_name)
            if region_ds.sizes["lat"] == 0 or region_ds.sizes["lon"] == 0:
                continue
        row = {"region": region_name or "Global"}
        valid_cells = np.zeros((region_ds.sizes["lat"], region_ds.sizes["lon"]), bool)
        for name in variables:
            da = region_ds[name].transpose("lat", "lon", ...)
            values = np.asarray(da.values)
            finite = np.isfinite(values)
            # a cell counts once, however many years or months it holds
            valid_cells |= finite.reshape(*valid_cells.shape, -1).any(axis=-1)
            row[name] = float(np.median(values[finite])) if finite.any() else np.nan
        if region_name is not None and not valid_cells.any():
            continue
        row["cells"] = int(valid_cells.sum())
        rows.append(row)
    return pd.DataFrame(rows, columns=["region", "cells", *variables])


def convert_dataarray_units(da, to_units, from_units=None):
    """Convert a DataArray to other units of the same kind, keeping its attributes.

    Args:
        da: DataArray to convert.
        to_units: Target units, e.g. "m3 s-1" or "mm".
        from_units: Units of `da`, None to read its units attribute.

    Returns
    -------
        The DataArray in `to_units` with its units attribute set. Raises
        ``ValueError`` for missing or incompatible units.
    """
    from_units = da.attrs.get("units") if from_units is None else from_units
    if from_units is None:
        msg = f"{da.name!r} has no units to convert from."
        raise ValueError(msg)
    factor = calculate_conversion_factor(from_units, to_units)
    converted = da.copy(deep=False) if np.isclose(factor, 1.0) else da * factor
    converted.attrs = {**da.attrs, "units": to_units}
    return converted


def convert_water_storage_to_mm(da, scale_factor=None, label="input"):
    """Convert a water storage field to millimetres of water.

    A scale factor wins over the `units` attribute. Without either the field
    cannot be interpreted physically, so this raises instead of assuming a unit.

    Args:
        da: Storage DataArray, e.g. TWS, a TWS anomaly or one mHM storage.
        scale_factor: Millimetres per data unit, or None to use `units`.
        label: Name used in the log and error messages.

    Returns
    -------
        The DataArray in mm, keeping its other attributes.
    """
    attrs = dict(da.attrs)
    if scale_factor is not None:
        logger.info(f"Scaling {label} by the given factor {scale_factor} to mm.")
        converted = da * float(scale_factor)
    else:
        units = da.attrs.get("units") or da.encoding.get("units")
        try:
            factor = calculate_conversion_factor(units, "mm") if units else None
        except ValueError:
            factor = None
        if factor is None:
            reason = (
                "carries no 'units' attribute"
                if units is None
                else f"carries the unrecognized unit '{units}'"
            )
            msg = (
                f"The {label} variable '{da.name}' {reason}, so it cannot be "
                f"converted to mm. Pass an explicit scale factor in mm per data "
                f"unit, or set a depth of water such as mm, cm, m or kg m-2."
            )
            with ErrorLogger(logger):
                raise ValueError(msg)
        logger.info(f"Converting {label} from '{units}' to mm by factor {factor}.")
        converted = da * factor
    # arithmetic drops attrs, so restore them and record the new unit
    converted.attrs = {**attrs, "units": "mm"}
    converted.name = da.name
    return converted


def calculate_anomaly(
    data, baseline_slice=None, require_full_coverage=False, label="data"
):
    """Subtract the mean of a baseline period from every time step.

    The baseline mean is one constant per grid cell, so the seasonal cycle stays
    in the result. This is the convention of a GRACE style storage anomaly.

    Args:
        data: DataArray or Dataset with a ``time`` dimension.
        baseline_slice: ``slice`` of the baseline period, or None to use the
            whole record.
        require_full_coverage: Raise when the record does not cover the whole
            baseline period, instead of averaging the covered part.
        label: Name used in the log and error messages.

    Returns
    -------
        Tuple of (anomaly, baseline mean per cell).
    """
    if "time" not in data.dims or data.sizes["time"] == 0:
        msg = f"The {label} for the anomaly calculation has no valid time dimension."
        with ErrorLogger(logger):
            raise ValueError(msg)
    if baseline_slice is None:
        baseline = data
    else:
        if require_full_coverage:
            validate_period_coverage(data, baseline_slice, label)
        baseline = data.sel(time=baseline_slice)
    if baseline.sizes["time"] == 0:
        msg = (
            f"The baseline period {baseline_slice.start} to {baseline_slice.stop} "
            f"holds no time step of the {label} record "
            f"{str(data.time.values[0])[:10]} to {str(data.time.values[-1])[:10]}."
        )
        with ErrorLogger(logger):
            raise ValueError(msg)
    baseline_mean = baseline.mean("time", skipna=True)
    logger.info(
        f"Removing the {label} baseline mean of {baseline.sizes['time']} steps "
        f"between {baseline_slice.start} and {baseline_slice.stop}."
        if baseline_slice is not None
        else f"Removing the {label} mean of the whole record."
    )
    return data - baseline_mean, baseline_mean


def validate_period_coverage(data, period_slice, label="data"):
    """Fail when a record does not cover a whole requested period.

    A stamp labels the period it was aggregated from, so the coverage of the
    record is compared instead of its bare first and last stamp.

    Args:
        data: DataArray or Dataset with a ``time`` dimension.
        period_slice: ``slice`` of the requested period.
        label: Name used in the error message.

    Returns
    -------
        None. Raises ``ValueError`` when the period is not fully covered.
    """
    coverage_start, coverage_end = get_data_coverage(data)
    period_start = pd.Timestamp(period_slice.start)
    period_end = pd.Timestamp(period_slice.stop)
    if coverage_start > period_start or coverage_end < period_end:
        msg = (
            f"The {label} record covers {coverage_start.date()} to "
            f"{coverage_end.date()} and does not fully cover the requested period "
            f"{period_start.date()} to {period_end.date()}. Pass a period the "
            f"record covers."
        )
        with ErrorLogger(logger):
            raise ValueError(msg)


def calculate_coordinate_resolution(coord):
    """Return the median absolute spacing of a 1-D coordinate.

    Args:
        coord: Coordinate DataArray or array with at least two values.

    Returns
    -------
        The spacing as float.
    """
    values = np.asarray(coord)
    if values.ndim != 1 or values.size < 2:
        msg = (
            f"Cannot determine the resolution of a coordinate with shape "
            f"{values.shape}. Two-dimensional or single-cell grids are not supported."
        )
        with ErrorLogger(logger):
            raise ValueError(msg)
    return float(np.nanmedian(np.abs(np.diff(values))))


def create_cell_edges(centers):
    """Create a contiguous edge vector from monotonic cell centers.

    Reuses the CF bounds of `generate_bounds` and folds the per cell
    ``[lower, upper]`` pairs into the single edge vector that binning needs.

    Args:
        centers: 1-D array of cell centers.

    Returns
    -------
        Array of ``len(centers) + 1`` cell edges in the order of the centers.
    """
    centers = np.asarray(centers, dtype=float)
    if centers.size < 2:
        msg = "Cannot create cell edges from less than two cell centers."
        with ErrorLogger(logger):
            raise ValueError(msg)
    coord = xr.DataArray(centers, dims=("cell",), coords={"cell": centers})
    bounds = generate_bounds(coord).values
    return np.append(bounds[:, 0], bounds[-1, 1])


def align_to_target_grid(da, target_lat, target_lon):
    """Snap a DataArray onto target coordinates of the same resolution.

    Args:
        da: DataArray with ``lat`` and ``lon``.
        target_lat: Target latitude values.
        target_lon: Target longitude values.

    Returns
    -------
        The DataArray reindexed to the target coordinates.
    """
    tolerance = (
        max(
            calculate_coordinate_resolution(target_lat),
            calculate_coordinate_resolution(target_lon),
        )
        / 2
    )
    return da.reindex(
        lat=np.asarray(target_lat),
        lon=np.asarray(target_lon),
        method="nearest",
        tolerance=tolerance,
    )


def aggregate_to_target_grid(da, target_lat, target_lon):
    """Aggregate a fine DataArray onto a coarser target grid by spatial mean.

    Uses ``coarsen`` when the target grid is an aligned integer multiple of the
    source grid and falls back to binning the source cells into the target
    cells otherwise.

    Args:
        da: Fine DataArray with ``lat`` and ``lon``.
        target_lat: Coarse latitude values.
        target_lon: Coarse longitude values.

    Returns
    -------
        The aggregated DataArray on the target grid.
    """
    lat_factor = round(
        calculate_coordinate_resolution(target_lat)
        / calculate_coordinate_resolution(da["lat"])
    )
    lon_factor = round(
        calculate_coordinate_resolution(target_lon)
        / calculate_coordinate_resolution(da["lon"])
    )
    if (
        lat_factor > 1
        and lon_factor > 1
        and da.sizes["lat"] % lat_factor == 0
        and da.sizes["lon"] % lon_factor == 0
    ):
        coarsened = da.coarsen(lat=lat_factor, lon=lon_factor, boundary="trim").mean()
        if coarsened.sizes["lat"] == len(target_lat) and coarsened.sizes["lon"] == len(
            target_lon
        ):
            logger.info(
                f"Aggregating with coarsen by factors lat={lat_factor}, lon={lon_factor}."
            )
            return coarsened.assign_coords(
                lat=np.asarray(target_lat), lon=np.asarray(target_lon)
            )
        logger.debug(
            f"Coarsen produced {coarsened.sizes['lat']}x{coarsened.sizes['lon']} cells "
            f"instead of {len(target_lat)}x{len(target_lon)}; binning instead."
        )

    logger.info("Aggregating fine cells into the coarse target cells by mean.")
    lat_edges = create_cell_edges(target_lat)
    lon_edges = create_cell_edges(target_lon)
    # groupby_bins needs ascending edges, so reverse the labels for descending grids
    lat_labels = target_lat if lat_edges[0] < lat_edges[-1] else target_lat[::-1]
    lon_labels = target_lon if lon_edges[0] < lon_edges[-1] else target_lon[::-1]
    binned = (
        da.groupby_bins("lat", np.sort(lat_edges), labels=np.asarray(lat_labels))
        .mean("lat")
        .groupby_bins("lon", np.sort(lon_edges), labels=np.asarray(lon_labels))
        .mean("lon")
        .rename({"lat_bins": "lat", "lon_bins": "lon"})
    )
    return binned.reindex(lat=np.asarray(target_lat), lon=np.asarray(target_lon))


def regrid_to_coarser_grid(input_da, ref_da):
    """Bring two DataArrays or Datasets onto the coarser of their two grids.

    Args:
        input_da: Input DataArray or Dataset with ``lat`` and ``lon``.
        ref_da: Reference DataArray or Dataset with ``lat`` and ``lon``.

    Returns
    -------
        Tuple of the two objects on the common coarse grid.
    """
    input_cell_area = calculate_coordinate_resolution(
        input_da["lat"]
    ) * calculate_coordinate_resolution(input_da["lon"])
    ref_cell_area = calculate_coordinate_resolution(
        ref_da["lat"]
    ) * calculate_coordinate_resolution(ref_da["lon"])

    if np.isclose(input_cell_area, ref_cell_area, rtol=1e-3):
        logger.info("Input and reference share the same grid resolution.")
        return align_to_target_grid(input_da, ref_da["lat"], ref_da["lon"]), ref_da
    if input_cell_area > ref_cell_area:
        logger.info("Reference grid is finer; aggregating it onto the input grid.")
        return input_da, aggregate_to_target_grid(
            ref_da, input_da["lat"].values, input_da["lon"].values
        )
    logger.info("Input grid is finer; aggregating it onto the reference grid.")
    return (
        aggregate_to_target_grid(input_da, ref_da["lat"].values, ref_da["lon"].values),
        ref_da,
    )


def spearman_correlation(data1, data2):
    """Calculate Spearman rank correlation between two xarray DataArrays."""
    # Check that both arrays are of the same size and flatten them
    if data1.shape != data2.shape:
        with ErrorLogger(logger):
            msg = "Both DataArrays must have the same shape"
            raise ValueError(msg)
    # Accept either xarray objects or plain numpy arrays.
    data1 = np.asarray(getattr(data1, "values", data1)).flatten()
    data2 = np.asarray(getattr(data2, "values", data2)).flatten()
    valid = np.isfinite(data1) & np.isfinite(data2)
    data1 = data1[valid]
    data2 = data2[valid]
    if data1.size < 2:
        return np.nan, np.nan
    # Calculate Spearman rank correlation using scipy
    corr, p_value = spearmanr(data1, data2)
    return corr, p_value


def calculate_spearman_over_time(input_values, ref_values):
    """Correlate two time series stacks cell by cell over the trailing axis.

    Reproduces the average ranks, Pearson and t statistic chain that
    ``scipy.stats.spearmanr`` uses, but vectorized over all leading axes.

    Parameters
    ----------
    input_values, ref_values : numpy.ndarray
        Arrays of the same shape ``(..., time)``.

    Returns
    -------
    tuple of numpy.ndarray
        Rho and two-sided p-value arrays with the trailing axis reduced.
    """
    input_ranked = np.array(input_values, dtype=np.float64)
    ref_ranked = np.array(ref_values, dtype=np.float64)
    # A cell is ranked over the steps both stacks cover, so the steps that are
    # not finite in both are blanked in both before ranking. One buffer holds
    # the paired steps first and is then inverted into the blanking mask.
    paired = np.isfinite(input_ranked)
    paired &= np.isfinite(ref_ranked)
    n_paired = np.count_nonzero(paired, axis=-1)
    np.logical_not(paired, out=paired)
    input_ranked[paired] = np.nan
    ref_ranked[paired] = np.nan
    del paired

    input_ranked = rankdata(input_ranked, axis=-1, nan_policy="omit")
    ref_ranked = rankdata(ref_ranked, axis=-1, nan_policy="omit")
    divisor = np.where(n_paired > 0, n_paired, 1)
    input_ranked -= np.nansum(input_ranked, axis=-1, keepdims=True) / divisor[..., None]
    ref_ranked -= np.nansum(ref_ranked, axis=-1, keepdims=True) / divisor[..., None]
    input_norm = np.sqrt(np.nansum(input_ranked * input_ranked, axis=-1))
    ref_norm = np.sqrt(np.nansum(ref_ranked * ref_ranked, axis=-1))

    with np.errstate(invalid="ignore", divide="ignore"):
        rho = np.nansum(input_ranked * ref_ranked, axis=-1) / (input_norm * ref_norm)
        rho = np.where(
            (n_paired >= MIN_PAIRS_FOR_CORRELATION) & (input_norm > 0) & (ref_norm > 0),
            rho,
            np.nan,
        )
        # two parameters are estimated per cell, so the t test loses two steps
        degrees_of_freedom = n_paired - 2
        t_statistic = rho * np.sqrt(
            np.clip(degrees_of_freedom / ((rho + 1.0) * (1.0 - rho)), 0, None)
        )
    p_value = np.where(
        np.isnan(rho),
        np.nan,
        2 * student_t.sf(np.abs(t_statistic), degrees_of_freedom),
    )
    return rho.astype(np.float32), p_value.astype(np.float32)


def get_dtype(ds):
    """Return a simple dtype string without forcing data into memory."""
    try:
        if isinstance(ds, xr.Dataset):
            v = get_single_data_var(ds)
            if v is None:
                try:
                    dt = ds.dtype  # check if dataset has single dtype
                    da = ds  # use dataset directly
                except AttributeError as e:
                    msg = "Dataset has no single data variable to inspect."
                    raise ValueError(msg) from e
            else:
                da = ds[v]
        elif isinstance(ds, xr.DataArray):
            da = ds
        else:
            msg = f"Unsupported type {type(ds)}"
            raise ValueError(msg)

        dt = da.dtype  # cheap: does not load data
        if dt is None:
            return "f4"

        if np.issubdtype(dt, np.floating):
            return "f4" if dt.itemsize <= 4 else "f8"
        if np.issubdtype(dt, np.integer) or np.issubdtype(dt, np.unsignedinteger):
            # prefer signed integers for ASCII grid
            return "i4" if dt.itemsize <= 4 else "i8"

        msg = f"write_grid: cannot infer dtype from data with numpy dtype {dt}"
        with ErrorLogger(logger):
            raise ValueError(msg)
    except Exception:
        return "f4"


def _snap_coord_bound(value, resolution=None):
    """Snap coordinate bounds that only differ from the grid by roundoff."""
    value = float(value)
    if not np.isfinite(value):
        return value
    scale = max(abs(value), 1.0)
    tolerance = np.finfo(float).eps * scale * 4096
    if resolution is not None:
        resolution = abs(float(resolution))
        if np.isfinite(resolution) and resolution > 0:
            snapped = round(value / resolution) * resolution
            if abs(value - snapped) <= tolerance:
                value = float(snapped)
    nearest_integer = round(value)
    if abs(value - nearest_integer) <= tolerance:
        return float(nearest_integer)
    return value


def _snap_coord_bounds(bounds, resolution=None):
    return tuple(_snap_coord_bound(bound, resolution) for bound in bounds)


def _coord_bound_resolution(ds, lon_key, lat_key, lon_bnds_key=None, lat_bnds_key=None):
    if "spatial_resolution" in ds.attrs:
        return ds.attrs["spatial_resolution"]
    try:
        from mhm_tools.common.resolution_handler import get_file_res

        return get_file_res(ds[lon_key], ds[lat_key], None)
    except ValueError:
        pass
    for bounds_key in (lon_bnds_key, lat_bnds_key):
        if bounds_key is None or bounds_key not in ds:
            continue
        bounds = np.asarray(ds[bounds_key].values, dtype=float)
        widths = np.abs(bounds[..., 1] - bounds[..., 0])
        widths = widths[np.isfinite(widths) & (widths > 0)]
        if widths.size:
            return float(np.nanmedian(widths))
    return None


def get_ds_extend(ds, var=None, recursive_search=True, resolutions=None):
    """Get the spatial extent of a dataset as (lon_min, lon_max, lat_min, lat_max) from its bounds."""
    from mhm_tools.common.resolution_handler import get_file_res

    if var is not None:
        # get the coordinate keys from the variable if possible, otherwise from the dataset
        lon_key = get_coord_key(ds[var], lon=True)
        lat_key = get_coord_key(ds[var], lat=True)
    else:
        lon_key = get_coord_key(ds, lon=True)
        lat_key = get_coord_key(ds, lat=True)
    lon = ds[lon_key]
    lat = ds[lat_key]
    lon_bnds_key = lon.attrs.get("bounds", None)
    lat_bnds_key = lat.attrs.get("bounds", None)
    res = _coord_bound_resolution(ds, lon_key, lat_key, lon_bnds_key, lat_bnds_key)
    # a coordinate can name a bounds variable that is not there, for instance
    # after selecting one variable out of a bounded dataset, so check both.
    # `in` on a DataArray tests its values, hence the explicit container.
    present = getattr(ds, "variables", ds.coords)
    if (
        lon_bnds_key is not None
        and lat_bnds_key is not None
        and lon_bnds_key in present
        and lat_bnds_key in present
    ):
        lon_min = float(ds[lon_bnds_key].values.min())
        lon_max = float(ds[lon_bnds_key].values.max())
        lat_min = float(ds[lat_bnds_key].values.min())
        lat_max = float(ds[lat_bnds_key].values.max())
        return _snap_coord_bounds((lon_min, lon_max, lat_min, lat_max), res)
    if recursive_search:
        ds = generate_bounds_for_all_coords(ds)
        return get_ds_extend(ds, var=var, recursive_search=False)
    logger.warning(
        "Could not find coordinate bounds for dataset; estimating spatial extent from coordinate values."
    )
    lon_vals = np.asarray(ds[lon_key].values)
    lat_vals = np.asarray(ds[lat_key].values)
    res = (
        ds.attrs.get("spatial_resolution")
        if "spatial_resolution" in ds.attrs
        else (get_file_res(ds[lon_key], ds[lat_key], resolutions=resolutions))
    )
    if recursive_search:
        ds = generate_bounds_for_all_coords(ds, res=res)
        return get_ds_extend(
            ds, var=var, recursive_search=False, resolutions=resolutions
        )
    return _snap_coord_bounds(
        (
            float(np.nanmin(lon_vals)) - float(res) / 2,
            float(np.nanmax(lon_vals)) + float(res) / 2,
            float(np.nanmin(lat_vals)) - float(res) / 2,
            float(np.nanmax(lat_vals)) + float(res) / 2,
        ),
        res,
    )
