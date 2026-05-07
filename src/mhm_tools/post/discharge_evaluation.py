"""
Compare simulated with observed discharge either directly or using a bootstraping approach.

Authors
-------
- Jeisson Leal
- Simon Lüdke
"""

import itertools
import logging
from pathlib import Path
from types import SimpleNamespace

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import xarray as xr
from joblib import Parallel, delayed
from matplotlib import colors as mcolors
from scipy.spatial import cKDTree

from mhm_tools.common.file_handler import (
    get_dataset_from_path,
    write_xarray_to_file,
)
from mhm_tools.common.logger import ErrorLogger, log_arguments, log_errors
from mhm_tools.common.utils import coord_to_index, find_best_gauge_location_by_area
from mhm_tools.common.xarray_utils import (
    get_clim_from_ds,
    get_coord_key,
    get_overlapping_time_slice,
    get_single_data_var,
    spearman_correlation,
)
from mhm_tools.post.gridded_data_evaluation import (
    resample_to_coarser_calendar,
)
from mhm_tools.post.hydrograph import gen_hydrograph_by_data_sets

logger = logging.getLogger(__name__)


def flatten_list(nested_list):
    """Flatten a list."""
    flat_list = []
    for item in nested_list:
        if item is None:
            continue
        if isinstance(item, list):  # If the item is a list, extend the flat list
            flat_list.extend(flatten_list(item))
        else:  # If it's not a list, append it directly
            flat_list.append(item)
    return flat_list


def gen_list_of_result_dicts(data, id, datatype="", facc=None):
    """Generate result dicts with time, id, discharge, and year."""
    # Check if all values in the array are NaN
    discharge = data.data
    are_all_nan = np.all(np.isnan(discharge))
    time = data.time.data
    year = data.time.dt.year.data

    if not are_all_nan:
        logger.info(f" values found at gauge number: {id}...\n")
        res = []
        for t, val, yr in zip(time, discharge, year):
            res_dict = {"time": t, "id": int(id), "Q": val, "year": yr}
            if facc is not None:
                res_dict["facc"] = facc
            res.append(res_dict)
        return res
    logger.warning(f"All datapoints for {datatype} at gauge {id} are missing.")
    return None


def get_grdc_for_one_gauge(id, observed_data_by_id):
    """Read observed data for one gauge."""
    # id, index = id.values, index=id['index'].values
    if observed_data_by_id.size == 0:
        logger.info(f"No data found for ID: {id}. Skipping...\n")
        return None
    return observed_data_by_id.where(~np.isnan(observed_data_by_id), drop=True)


def get_sim_data_for_one_gauge(
    id, index, sim_data, yarr, xarr, resolution, lat_key="lat", lon_key="lon"
):
    """Read out simulation data for one gauge."""
    if xarr[index] is None and yarr[index] is None:
        return None
    # This will ensure that x & y values always match lat and lon in mRM dataset
    try:
        x = np.round(xarr[index], 9)
        y = np.round(yarr[index], 9)
    except KeyError:
        logger.error(f"index {index} not found")
        return None
    sim_data_loc = sim_data.sel({lat_key: y - resolution, lon_key: x}, method="nearest")
    return sim_data_loc.expand_dims(dim={"id": [id]})


def _find_node_xy_vars(ds):
    """Find x/y node coordinate variables in a node-output dataset."""
    candidates = [
        ("river_node_x", "river_node_y"),
        ("river_node_lon", "river_node_lat"),
        ("node_x", "node_y"),
        ("x", "y"),
        ("lon", "lat"),
    ]
    for x_name, y_name in candidates:
        if x_name in ds and y_name in ds:
            return x_name, y_name
        if x_name in ds.coords and y_name in ds.coords:
            return x_name, y_name

    x_vars = [
        name
        for name in list(ds.data_vars) + list(ds.coords)
        if "river_node" in name.lower() and "x" in name.lower()
    ]
    y_vars = [
        name
        for name in list(ds.data_vars) + list(ds.coords)
        if "river_node" in name.lower() and "y" in name.lower()
    ]
    if x_vars and y_vars:
        return x_vars[0], y_vars[0]

    msg = f"Could not find river node x/y coordinates in model dataset with coords {ds.coords} and data_vars {ds.data_vars}."
    with ErrorLogger(logger):
        raise ValueError(msg)


def _find_node_dim(discharge_da, n_nodes):
    """Find the node dimension in discharge data by size."""
    for dim in discharge_da.dims:
        if discharge_da.sizes.get(dim) == n_nodes:
            return dim
    msg = (
        "Could not identify node dimension matching river_node_x/y length "
        f"({n_nodes}) in discharge data dims {discharge_da.dims}."
    )
    with ErrorLogger(logger):
        raise ValueError(msg)


def _reduce_node_coord(da):
    """Reduce node coordinate array to 1D if it carries extra dimensions."""
    if da.ndim <= 1:
        return da
    if "time" in da.dims:
        da = da.isel(time=0)
    da = da.squeeze(drop=True)
    if da.ndim <= 1:
        return da
    if "node" in da.dims:
        drop_dims = [d for d in da.dims if d != "node"]
        if drop_dims:
            da = da.isel(dict.fromkeys(drop_dims, 0))
        return da
    drop_dims = list(da.dims[:-1])
    if drop_dims:
        da = da.isel(dict.fromkeys(drop_dims, 0))
    return da


def _as_scalar_or_nan(value):
    """Return a scalar from array-like inputs, else NaN for unsupported shapes."""
    if value is None:
        return np.nan
    if isinstance(value, (str, bytes)):
        return value
    try:
        arr = np.asarray(value)
    except Exception:
        return np.nan
    if arr.ndim == 0:
        return arr.item()
    if arr.size == 1:
        return arr.reshape(-1)[0]
    return np.nan


def _load_discharge_nc_collection(model_data_path, date_slice=None):
    """Load discharge.nc files containing Qsim/Qobs_<id> and return sim/obs datasets."""

    def _expand_paths(path_like):
        if isinstance(path_like, (list, tuple)):
            return [Path(p) for p in path_like]
        path_str = str(path_like)
        if any(w in path_str for w in ("*", "?", "[", "]")):
            matches = list(Path().glob(path_str))
            if not matches:
                msg = f"No paths match pattern {path_str}"
                with ErrorLogger(logger):
                    raise FileNotFoundError(msg)
            return matches
        return [Path(path_like)]

    discharge_files = []
    for path in _expand_paths(model_data_path):
        if path.is_dir():
            discharge_files.extend(sorted(path.rglob("discharge.nc")))
        elif path.is_file():
            discharge_files.append(path)
        else:
            msg = f"Path {path} does not exist."
            with ErrorLogger(logger):
                raise FileNotFoundError(msg)
    if not discharge_files:
        msg = f"No discharge.nc files found under {model_data_path}"
        with ErrorLogger(logger):
            raise FileNotFoundError(msg)

    sim_list = []
    obs_list = []
    for f in discharge_files:
        with load_ds(f) as ds:
            ds_sel = ds.sel(time=date_slice) if date_slice is not None else ds
            for var in ds_sel.data_vars:
                lower = var.lower()
                if not lower.startswith(("qsim_", "qobs_")):
                    continue
                # parse gauge id, stripping leading zeros
                try:
                    gid_str = var.split("_", 1)[1]
                    gid = int(gid_str.lstrip("0") or "0")
                except Exception:
                    logger.debug(f"Could not parse gauge id from {var} in {f}")
                    continue
                da = ds_sel[var]
                da = da.assign_coords(id=gid).expand_dims("id")
                da.name = "discharge"
                if lower.startswith("qsim_"):
                    sim_list.append(da)
                else:
                    obs_list.append(da)

    if not sim_list or not obs_list:
        msg = "Missing Qsim/Qobs variables in discharge.nc collection."
        with ErrorLogger(logger):
            raise ValueError(msg)

    sim_da = xr.concat(sim_list, dim="id").sortby("id")
    obs_da = xr.concat(obs_list, dim="id").sortby("id")
    sim_ds = xr.Dataset({"discharge": sim_da})
    obs_ds = xr.Dataset({"discharge": obs_da})
    sim_ds, obs_ds = xr.align(sim_ds, obs_ds, join="outer")
    return sim_ds, obs_ds


def _expand_input_files(path_like, file_name=None):
    """Resolve file, directory, list, or glob input into a sorted file list."""

    def _iter_wildcard_matches(pattern_str):
        pattern_path = Path(pattern_str)
        if pattern_path.is_absolute():
            anchor = Path(pattern_path.anchor)
            relative_pattern = str(pattern_path.relative_to(anchor))
            return anchor.glob(relative_pattern)
        return Path().glob(pattern_str)

    pattern = file_name if file_name not in (None, "") else "*.nc"
    inputs = path_like if isinstance(path_like, (list, tuple)) else [path_like]
    files = []

    for item in inputs:
        if item is None:
            continue
        item_str = str(item)
        if any(token in item_str for token in ("*", "?", "[")):
            files.extend(sorted(_iter_wildcard_matches(item_str)))
            continue

        path = Path(item)
        if path.is_file():
            files.append(path)
        elif path.is_dir():
            files.extend(sorted(path.rglob(pattern)))
        else:
            # Defer path validation to downstream loader for compatibility with
            # tests/mocked loaders and virtual file handlers.
            files.append(path)

    # Keep order deterministic and drop duplicates
    unique_files = []
    seen = set()
    for p in files:
        key = str(p.resolve()) if p.exists() else str(p)
        if key in seen:
            continue
        seen.add(key)
        unique_files.append(p)
    return unique_files


def _concat_time_dataarrays(parts):
    """Concatenate time-sliced DataArrays and drop duplicate timestamps."""
    if not parts:
        return None
    out = parts[0] if len(parts) == 1 else xr.concat(parts, dim="time", join="outer")
    if "time" in out.dims:
        out = out.sortby("time")
        # Keep first occurrence for duplicate timestamps
        if "time" in out.coords:
            time_vals = np.asarray(out["time"].values)
            if time_vals.size:
                _, unique_pos = np.unique(time_vals, return_index=True)
                if unique_pos.size != time_vals.size:
                    out = out.isel(time=np.sort(unique_pos))
    return out


def _read_observed_file_part(obs_file, observed_variable, eval_gauge_ids, date_slice):
    """Read one observed file and return filtered discharge + metadata."""
    with load_ds(obs_file) as observed_data_in:
        obs_var_local = (
            get_single_data_var(observed_data_in)
            if observed_variable is None
            else observed_variable
        )
        if obs_var_local not in observed_data_in:
            logger.warning(
                f"Variable '{obs_var_local}' not found in observed file {obs_file}. Skipping."
            )
            return None

        ids = np.asarray(observed_data_in["id"].values)
        keep_mask = np.ones(ids.shape, dtype=bool)
        if eval_gauge_ids is not None:
            keep_mask &= np.isin(ids, eval_gauge_ids)
        kept_ids = ids[keep_mask]
        if kept_ids.size == 0:
            return None

        obs_part = (
            observed_data_in[obs_var_local].sel(id=kept_ids).sel(time=date_slice).load()
        )

        def _meta_values(name):
            if name in observed_data_in or name in observed_data_in.coords:
                return np.asarray(
                    observed_data_in[name].sel(id=kept_ids).values, dtype=float
                )
            return np.full(kept_ids.shape, np.nan, dtype=float)

        meta = pd.DataFrame(
            {
                "id": kept_ids,
                "geo_x": _meta_values("geo_x"),
                "geo_y": _meta_values("geo_y"),
                "area": _meta_values("area"),
            }
        )
        return obs_var_local, obs_part, meta


def _materialize_data(obj, num_workers=1):
    """Load xarray object into memory; use threaded compute when possible."""
    workers = max(1, int(num_workers)) if num_workers is not None else 1
    if workers > 1:
        try:
            return obj.load(scheduler="threads", num_workers=workers)
        except TypeError:
            return obj.load()
    return obj.load()


def _read_model_file_part(  # noqa: PLR0911
    sim_file,
    sim_variable,
    date_slice,
    do_spatial_crop=False,
    lat_min=None,
    lat_max=None,
    lon_min=None,
    lon_max=None,
    reduce_coords=False,
    extraction_mode="raw",
    gauge_ids_with_values=None,
    x_new=None,
    y_new=None,
    resolution=None,
    load_num_workers=1,
):
    """Read one model file and return either cropped data or extracted gauge series."""
    with load_ds(sim_file) as ds_sim:
        sim_var_local = (
            get_single_data_var(ds_sim) if sim_variable is None else sim_variable
        )
        if sim_var_local is None or sim_var_local not in ds_sim:
            logger.warning(
                f"Could not resolve simulation variable in {sim_file}. Skipping file."
            )
            return None
        if extraction_mode == "node":
            if gauge_ids_with_values is None or x_new is None or y_new is None:
                return None
            ds_part = ds_sim
            try:
                x_name, y_name = _find_node_xy_vars(ds_sim)
                keep_vars = [sim_var_local]
                if x_name in ds_sim.data_vars:
                    keep_vars.append(x_name)
                if y_name in ds_sim.data_vars:
                    keep_vars.append(y_name)
                ds_part = ds_sim[keep_vars]
                if x_name in ds_sim.coords and x_name not in ds_part.coords:
                    ds_part = ds_part.assign_coords({x_name: ds_sim.coords[x_name]})
                if y_name in ds_sim.coords and y_name not in ds_part.coords:
                    ds_part = ds_part.assign_coords({y_name: ds_sim.coords[y_name]})
            except Exception:
                # Keep compatibility with tests that mock the downstream extractor
                # and do not provide node coordinate variables on the input dataset.
                logger.debug(
                    f"Could not pre-reduce node variables for {sim_file}; using full dataset."
                )
            ds_part = ds_part.sel(time=date_slice)
            sim_sel, matched_x, matched_y, ids_arr = get_sim_data_for_gauges_from_nodes(
                sim_ds=ds_part,
                sim_variable=sim_var_local,
                x_new=x_new,
                y_new=y_new,
                gauge_ids=gauge_ids_with_values,
                resolution=resolution,
            )
            # Important: file-wise extraction is loaded here to free backing file handles
            # before concatenation across many files.
            sim_sel = _materialize_data(sim_sel, num_workers=load_num_workers)
            return (
                sim_var_local,
                sim_sel,
                np.asarray(matched_x),
                np.asarray(matched_y),
                np.asarray(ids_arr),
            )

        ds_part = ds_sim[[sim_var_local]] if reduce_coords else ds_sim

        if do_spatial_crop:
            lat_key = get_coord_key(ds_part, lat=True, raise_exception=False)
            lon_key = get_coord_key(ds_part, lon=True, raise_exception=False)
            if lat_key is not None and lon_key is not None:
                ds_part = ds_part.sel(
                    {
                        lat_key: slice(lat_max, lat_min),
                        lon_key: slice(lon_min, lon_max),
                    }
                )
            else:
                logger.warning(
                    f"Spatial crop requested but lat/lon keys missing in {sim_file}. Skipping spatial crop for this file."
                )

        if extraction_mode == "coords":
            if gauge_ids_with_values is None or x_new is None or y_new is None:
                return None
            ds_part = ds_part.sel(time=date_slice)
            lat_key = get_coord_key(
                ds_part[sim_var_local], lat=True, raise_exception=False
            )
            lon_key = get_coord_key(
                ds_part[sim_var_local], lon=True, raise_exception=False
            )
            # Keep compatibility with unit tests that mock per-gauge extraction and
            # provide synthetic sim data without lat/lon coords.
            if lat_key is None:
                lat_key = "lat"
            if lon_key is None:
                lon_key = "lon"
            sim_series = []
            for i, gid in enumerate(np.asarray(gauge_ids_with_values)):
                da = get_sim_data_for_one_gauge(
                    id=gid,
                    index=i,
                    sim_data=ds_part[sim_var_local],
                    yarr=np.asarray(y_new),
                    xarr=np.asarray(x_new),
                    resolution=resolution,
                    lat_key=lat_key,
                    lon_key=lon_key,
                )
                if da is not None:
                    sim_series.append(
                        da.drop_vars([c for c in (lat_key, lon_key) if c in da.coords])
                    )
            if not sim_series:
                return None
            sim_sel = xr.concat(
                sim_series, dim="id", coords="minimal", compat="override"
            )
            sim_sel = _materialize_data(sim_sel, num_workers=load_num_workers)
            return sim_var_local, sim_sel

        ds_part = ds_part.sel(time=date_slice)

        ds_part = _materialize_data(ds_part, num_workers=load_num_workers)
        return sim_var_local, ds_part


def get_sim_data_for_gauges_from_nodes(
    sim_ds,
    sim_variable,
    x_new,
    y_new,
    gauge_ids,
    resolution=None,
):
    """Extract discharge series by nearest river node to gauge coordinates."""
    logger.info("Extracting gauge discharge from node output: start.")
    x_name, y_name = _find_node_xy_vars(sim_ds)
    logger.info(
        "Resolved node coordinate vars",
    )
    node_x_da = _reduce_node_coord(sim_ds[x_name])
    node_y_da = _reduce_node_coord(sim_ds[y_name])
    node_x = np.asarray(node_x_da.values).ravel()
    node_y = np.asarray(node_y_da.values).ravel()
    logger.info(
        "Loaded node coordinates",
    )
    if node_x.size != node_y.size:
        msg = (
            f"Node coordinate sizes differ: {x_name}={node_x.size}, "
            f"{y_name}={node_y.size}"
        )
        with ErrorLogger(logger):
            raise ValueError(msg)

    sim_da = sim_ds[sim_variable]
    node_dim = _find_node_dim(sim_da, node_x.size)
    logger.info("Resolved node dimension")

    x_arr = np.asarray(x_new)
    y_arr = np.asarray(y_new)
    ids_arr = np.asarray(gauge_ids)
    valid = np.isfinite(x_arr) & np.isfinite(y_arr)
    if not np.any(valid):
        msg = "No valid gauge coordinates found in scc gauges file."
        with ErrorLogger(logger):
            raise ValueError(msg)
    if not np.all(valid):
        dropped = np.where(~valid)[0]
        logger.warning(f"Dropping {dropped.size} gauges with invalid coordinates.")
    x_arr = x_arr[valid]
    y_arr = y_arr[valid]
    ids_arr = ids_arr[valid]
    logger.info(f"Prepared {x_arr.size} gauge coordinates")

    tree = cKDTree(np.column_stack([node_x, node_y]))
    distances, indices = tree.query(np.column_stack([x_arr, y_arr]), k=1)
    logger.info(f"KDTree query for {x_arr.size} gauges.")

    max_dist = resolution * 1.5 if resolution is not None else None
    if max_dist is not None:
        far_mask = distances > max_dist
        if np.any(far_mask):
            logger.warning(
                f"Some gauges are farther than {max_dist:.6f} from nearest node; "
                f"using nearest anyway (max {float(np.max(distances[far_mask])):.6f})."
            )

    id_indexer = xr.DataArray(ids_arr, dims="id", coords={"id": ids_arr})
    sim_sel = sim_da.isel({node_dim: xr.DataArray(indices, dims="id")})
    sim_sel = sim_sel.assign_coords(id=id_indexer)
    if "id" not in sim_sel.dims and node_dim in sim_sel.dims:
        sim_sel = sim_sel.rename({node_dim: "id"})
    logger.info(
        "Selected discharge for all gauges.",
    )

    matched_x = node_x[indices]
    matched_y = node_y[indices]
    return sim_sel, matched_x, matched_y, ids_arr


def get_gauge_coords(  # noqa: PLR0912
    ds,
    ref_facc,
    facc_variable="L11_fAcc",
    lonlat=None,
    xy=None,
    max_distance_cells=3,
    max_error=0.1,
    method="basinex",
    id=None,
):
    """Find gauge coordinates by matching reference and simulated catchment area.

    Uses :func:`mhm_tools.common.utils.find_best_gauge_location`.
    """
    method_norm = str(method).strip().lower().replace("-", "")
    method_norm = {"basinex": "basinex", "burek": "burek"}.get(method_norm, method_norm)
    if method_norm not in {"basinex", "burek"}:
        msg = f"Unknown gauge optimization method '{method}'. Use 'basinex' or 'burek'."
        with ErrorLogger(logger):
            raise ValueError(msg)

    max_error = float(max_error)
    if max_error < 0:
        max_error = 0.0
    max_distance_cells = max(0, int(max_distance_cells))

    if facc_variable not in ds:
        logger.warning(
            f"Variable '{facc_variable}' not found in flow-accumulation dataset for gauge {id}."
        )
        return None, None, None
    facc_da = ds[facc_variable].squeeze()
    upstream_area = np.asarray(facc_da.data)
    if upstream_area.ndim != 2:
        logger.warning(
            f"Expected 2D upstream area in '{facc_variable}', got shape {upstream_area.shape} for gauge {id}."
        )
        return None, None, None
    if not np.isfinite(ref_facc):
        logger.warning(f"Invalid reference catchment area for gauge {id}: {ref_facc}")
        return None, None, None

    search_upstream = upstream_area
    latlon_search = False
    l0_resolution = 1.0

    # mRM restart structure fallback: L11_fAcc typically [lon, lat], coords in L1_domain_lon/lat.
    if "L1_domain_lon" in ds and "L1_domain_lat" in ds:
        lon_data = np.asarray(ds["L1_domain_lon"].data)
        lat_data = np.asarray(ds["L1_domain_lat"].data)
        if (
            lon_data.ndim == 2
            and lat_data.ndim == 2
            and lon_data.shape == upstream_area.shape
            and lat_data.shape == upstream_area.shape
        ):
            search_upstream = upstream_area.T
            search_ds = xr.Dataset(
                coords={
                    "lat": np.asarray(lat_data[0, :], dtype=float),
                    "lon": np.asarray(lon_data[:, 0], dtype=float),
                }
            )
            latlon_search = True
            if search_ds["lon"].size > 1:
                l0_resolution = float(
                    abs(search_ds["lon"].values[1] - search_ds["lon"].values[0])
                )
            elif search_ds["lat"].size > 1:
                l0_resolution = float(
                    abs(search_ds["lat"].values[1] - search_ds["lat"].values[0])
                )
            elif hasattr(ds, "cellsize_L1"):
                l0_resolution = float(ds.cellsize_L1)
        else:
            search_ds = xr.Dataset(
                coords={
                    "lat": np.arange(search_upstream.shape[0], dtype=float),
                    "lon": np.arange(search_upstream.shape[1], dtype=float),
                }
            )
    else:
        # Standard georeferenced facc file (lat/lon coords on facc variable)
        lat_key = get_coord_key(facc_da, lat=True, raise_exception=False)
        lon_key = get_coord_key(facc_da, lon=True, raise_exception=False)
        if (
            lat_key is not None
            and lon_key is not None
            and lat_key in facc_da.dims
            and lon_key in facc_da.dims
        ):
            facc_std = facc_da.transpose(lat_key, lon_key)
            search_upstream = np.asarray(facc_std.data)
            search_ds = xr.Dataset(
                coords={
                    "lat": np.asarray(facc_std[lat_key].values, dtype=float),
                    "lon": np.asarray(facc_std[lon_key].values, dtype=float),
                }
            )
            latlon_search = True
            if search_ds["lon"].size > 1:
                l0_resolution = float(
                    abs(search_ds["lon"].values[1] - search_ds["lon"].values[0])
                )
            elif search_ds["lat"].size > 1:
                l0_resolution = float(
                    abs(search_ds["lat"].values[1] - search_ds["lat"].values[0])
                )
        else:
            search_ds = xr.Dataset(
                coords={
                    "lat": np.arange(search_upstream.shape[0], dtype=float),
                    "lon": np.arange(search_upstream.shape[1], dtype=float),
                }
            )

    if lonlat is not None:
        lon, lat = float(lonlat[0]), float(lonlat[1])
        try:
            idx0_guess, idx1_guess = coord_to_index(search_ds, lat, lon)
        except Exception:
            idx0_guess = int(search_upstream.shape[0] / 2)
            idx1_guess = int(search_upstream.shape[1] / 2)
    else:
        idx0_guess = int(xy[0])
        idx1_guess = int(xy[1])

    idx0_guess = int(np.clip(idx0_guess, 0, search_upstream.shape[0] - 1))
    idx1_guess = int(np.clip(idx1_guess, 0, search_upstream.shape[1] - 1))

    resolutions = SimpleNamespace(l0_resolution=l0_resolution)

    try:
        best_idx, area_error, distance_100m = find_best_gauge_location_by_area(
            ds=search_ds,
            upstream_area=search_upstream,
            gauge_coords=(idx0_guess, idx1_guess),
            ref_catchment_area=float(ref_facc),
            resolutions=resolutions,
            max_distance_cells=max_distance_cells,
            max_error=max_error,
            method=method_norm,
            raise_on_fallback=True,
            latlon=latlon_search,
        )
    except Exception:
        logger.warning(
            f"No suitable gauge match found for gauge {id} within {max_distance_cells} "
            f"cells and {max_error * 100.0:.2f}% error."
        )
        return None, None, None

    idx0, idx1 = int(best_idx[0]), int(best_idx[1])
    found_facc = float(search_upstream[idx0, idx1])

    if latlon_search:
        lon_x = float(search_ds["lon"].values[idx1])
        lat_y = float(search_ds["lat"].values[idx0])
    elif lonlat is not None:
        lon_x, lat_y = lonlat[0], lonlat[1]
    else:
        lon_x, lat_y = idx0, idx1

    logger.info(
        f"Gauge {id} matched at lon={lon_x} lat={lat_y} with facc={found_facc:.3f} "
        f"(target={float(ref_facc):.3f}, error={float(area_error) * 100.0:.2f}%, "
        f"distance={float(distance_100m):.2f} x100m)."
    )
    return lon_x, lat_y, found_facc


def load_ds(file_path, file_name=None):
    """Load an xarray dataset from disk."""
    if file_name is not None:
        logger.info(f"Loading dataset from file {file_path}/{file_name}")
    else:
        logger.info(f"Loading dataset from file {file_path}")
    return get_dataset_from_path(file_path, file_name=file_name, use_mfdataset=True)


def filter_ids_by_observed_years(valid_ids, obs_discharge_data, min_overlapping_years):
    """Filter ids by minimum observed years while preserving input order."""
    ids_arr = np.asarray(valid_ids)
    if min_overlapping_years is None:
        return ids_arr, []
    if min_overlapping_years <= 0:
        min_overlapping_years = 1
    if ids_arr.size == 0:
        return ids_arr, []

    obs_sel = obs_discharge_data.sel(id=ids_arr)
    obs_years = obs_sel["time"].dt.year
    n_years = obs_sel.notnull().groupby(obs_years).any(dim="time").sum(dim="year")

    enough_ids = np.asarray(
        obs_sel["id"].where(n_years >= min_overlapping_years, drop=True).values
    )
    keep_mask = np.isin(ids_arr, enough_ids)
    filtered_ids = ids_arr[keep_mask]

    years_map = {
        sid: int(years)
        for sid, years in zip(
            np.asarray(obs_sel["id"].values), np.asarray(n_years.values, dtype=int)
        )
    }
    dropped = [(sid, years_map.get(sid, 0)) for sid in ids_arr[~keep_mask]]
    return filtered_ids, dropped


def Q_data_to_xarray(  # noqa: PLR0913, PLR0915, PLR0912
    model_data_path,
    observed_data_path,
    sim_variable,
    observed_variable,
    model_keyword,
    facc_file=None,
    facc_variable="L11_fAcc",
    scc_gauges_file=None,
    evaluation_gauges=None,
    saving_path=None,
    model_file_name=None,
    lon_min=None,
    lon_max=None,
    lat_min=None,
    lat_max=None,
    resolution=0.1,
    n_jobs=1,
    date_slice=None,
    overwrite=False,
    direct_comparison=False,
    write_input_data_cache=False,
    min_overlapping_years=1,
    gauge_location_method="basinex",
    gauge_max_distance_cells=3,
    gauge_max_error=0.1,
):
    """Get observed and model Q data and save it as CSV files to be opened later.

    This function is not part of the workflow, but a pre processing tool
    that for comodity has been stored here. This was done bc mrm_data_by_id.values
    was taking to much time to execute

    Args:
    - mrm_data (xarray.DataArray): The mRM simulated data as an xarray DataArray.
    - observed_data (xarray.DataArray): The observed data as an xarray DataArray.
    - model_keyword (str): dir to be added to the path were files will be stored.
    - output_path (str): optional, saving path

    Note:
    The gauge information dataset should contain the following variables:
    - "id1": Gauge IDs
    - "new_x": X-coordinates of gauges
    - "new_y": Y-coordinates of gauges
    """
    if date_slice is None:
        date_slice = slice(None, None)
    sim_output_file = Path(f"{saving_path}/{model_keyword}_data.nc")
    obs_output_file = Path(f"{saving_path}/GRDC_data.nc")

    # if both sim and obs data cachefiles exist and overwrite is False, try to load and return them
    sim_data = None
    observed_data = None
    if not overwrite:
        if sim_output_file.is_file() and not overwrite:
            logger.info("reading sim data from file...")
            try:
                sim_data = load_ds(sim_output_file)
                sim_data = sim_data.sel(time=date_slice)
            except Exception:
                logger.warning(
                    f"Failed reading cached simulation data from {sim_output_file}. Recomputing."
                )
        if obs_output_file.is_file() and not overwrite:
            logger.info("reading obs data from file...")
            try:
                observed_data = load_ds(obs_output_file)
                observed_data = observed_data.sel(time=date_slice)
            except Exception:
                logger.warning(
                    f"Failed reading cached observation data from {obs_output_file}. Recomputing."
                )
        if sim_data is not None and observed_data is not None:
            logger.info(
                "Successfully loaded cached data. Selecting stations with sufficient overlap."
            )
            if min_overlapping_years is not None:
                eligible_ids, dropped_ids = _filter_ids_by_overlapping_years(
                    model_da=sim_data["discharge"],
                    observed_da=observed_data["discharge"],
                    min_overlapping_years=min_overlapping_years,
                )
                if dropped_ids:
                    logger.info(
                        f"Dropping {len(dropped_ids)} gauges with less than {min_overlapping_years} overlapping years."
                    )
                    logger.debug(
                        f"Dropped gauges due to overlap threshold (id, years): {dropped_ids}"
                    )
                if eligible_ids.size == 0:
                    msg = (
                        "No gauges meet the minimum overlap requirement of "
                        f"{min_overlapping_years} years."
                    )
                    with ErrorLogger(logger):
                        raise ValueError(msg)
                return observed_data.sel(id=eligible_ids), sim_data.sel(id=eligible_ids)
            return observed_data, sim_data
        overwrite = True  # if either data is not loaded, we need to overwrite
        logger.info(
            "Cached data not available or failed to load. Computing from source data."
        )

    # creating saving path
    saving_path = Path(saving_path)
    if not saving_path.is_dir():
        saving_path.mkdir(parents=True)

    # Load data and get gauge info for multiple different cases:
    # 1. discharge.nc files with Qsim/Qobs_<id> variables available (e.g. mRM v5+ output)
    discharge_nc_files = model_file_name == "discharge.nc"
    if discharge_nc_files:
        sim_rs, obs_rs = _load_discharge_nc_collection(
            model_data_path, date_slice=date_slice
        )
        observed_variable = "discharge"
        logger.info("Loaded discharge.nc collection for sim/obs data.")
        logger.info("Selecting data for date slice...")
        obs_discharge_data = obs_rs[observed_variable].sel(time=date_slice)
        sim_data_cropped = sim_rs[observed_variable].sel(time=date_slice)
        if direct_comparison:
            overlapping_time_slice = get_overlapping_time_slice(sim_rs, obs_rs)
            logger.info(
                f"Overlapping time is from {overlapping_time_slice.start} "
                f"to {overlapping_time_slice.stop}"
            )
            obs_discharge_data = obs_discharge_data.sel(time=overlapping_time_slice)
            sim_data_cropped = sim_data_cropped.sel(time=overlapping_time_slice)
        gauge_ids = obs_discharge_data["id"]
        # Ensure unique gauge ids (xarray .sel requires unique index)
        id_index = pd.Index(gauge_ids.values)
        if not id_index.is_unique:
            _, unique_pos = np.unique(id_index.values, return_index=True)
            unique_pos = np.sort(unique_pos)
            dup_count = len(id_index) - len(unique_pos)
            logger.warning(
                f"Found {dup_count} duplicate gauge id(s). Keeping first occurrence."
            )
            gauge_ids = gauge_ids.isel(id=unique_pos)
            obs_discharge_data = obs_discharge_data.isel(id=unique_pos)
            sim_data_cropped = sim_data_cropped.isel(id=unique_pos)
        # facc/x/y not available in discharge.nc
        x = xr.DataArray(
            np.full(gauge_ids.size, np.nan), dims=["id"], coords={"id": gauge_ids}
        )
        y = xr.DataArray(
            np.full(gauge_ids.size, np.nan), dims=["id"], coords={"id": gauge_ids}
        )
        facc = xr.DataArray(
            np.full(gauge_ids.size, np.nan), dims=["id"], coords={"id": gauge_ids}
        )
    # 2. no discharge.nc files provided. In this case the observed data is read from a given path and the sim data is extracted based on gauge coordinates from either a provided scc gauges file or by matching flow accumulation and coordinates in a facc file.
    else:
        logger.info("Loading observed data and gauge info.")
        observed_files = _expand_input_files(observed_data_path)
        if not observed_files:
            msg = f"No observed input files resolved from {observed_data_path}."
            with ErrorLogger(logger):
                raise FileNotFoundError(msg)
        if len(observed_files) > 1:
            logger.info(
                f"Observed input resolves to {len(observed_files)} files; processing incrementally."
            )

        eval_gauge_ids = None
        if evaluation_gauges is not None:
            logger.info("Filtering gauges for evaluation.")
            eval_gauge_ids = (
                pd.read_csv(evaluation_gauges, header=None).iloc[:, 0].values
            )

        n_jobs_eff = (
            min(max(1, int(n_jobs)), len(observed_files)) if n_jobs is not None else 1
        )
        if n_jobs_eff > 1:
            logger.info(
                f"Reading {len(observed_files)} observed files in parallel with n_jobs={n_jobs_eff}."
            )
            obs_results = Parallel(n_jobs=n_jobs_eff, backend="loky")(
                delayed(_read_observed_file_part)(
                    obs_file=obs_file,
                    observed_variable=observed_variable,
                    eval_gauge_ids=eval_gauge_ids,
                    date_slice=date_slice,
                )
                for obs_file in observed_files
            )
        else:
            if len(observed_files) > 1:
                logger.info(
                    f"Processing {len(observed_files)} observed files sequentially."
                )
            else:
                logger.info("Processing observed file.")
            obs_results = [
                _read_observed_file_part(
                    obs_file=obs_file,
                    observed_variable=observed_variable,
                    eval_gauge_ids=eval_gauge_ids,
                    date_slice=date_slice,
                )
                for obs_file in observed_files
            ]
        logger.info("Finished reading observed files. Processing results.")
        obs_results = [res for res in obs_results if res is not None]
        if not obs_results:
            msg = "No observed discharge data found after applying filters."
            with ErrorLogger(logger):
                raise ValueError(msg)

        if observed_variable is None:
            observed_variable = obs_results[0][0]
        obs_parts = [
            part for var_name, part, _ in obs_results if var_name == observed_variable
        ]
        obs_meta_frames = [
            meta for var_name, _, meta in obs_results if var_name == observed_variable
        ]
        dropped_obs = sum(
            1 for var_name, *_ in obs_results if var_name != observed_variable
        )
        if dropped_obs > 0:
            logger.warning(
                f"Dropped {dropped_obs} observed file(s) due to variable-name mismatch with '{observed_variable}'."
            )
        if not obs_parts:
            msg = (
                f"No observed discharge data left for variable '{observed_variable}' "
                "after processing all files."
            )
            with ErrorLogger(logger):
                raise ValueError(msg)
        obs_discharge_data = _concat_time_dataarrays(obs_parts)
        meta_df = pd.concat(obs_meta_frames, ignore_index=True)
        meta_df = meta_df.drop_duplicates(subset=["id"], keep="first").sort_values("id")
        gauge_id_vals = meta_df["id"].to_numpy()
        gauge_ids = xr.DataArray(
            gauge_id_vals, dims=["id"], coords={"id": gauge_id_vals}
        )
        x = xr.DataArray(
            meta_df["geo_x"].to_numpy(dtype=float),
            dims=["id"],
            coords={"id": gauge_id_vals},
        )
        y = xr.DataArray(
            meta_df["geo_y"].to_numpy(dtype=float),
            dims=["id"],
            coords={"id": gauge_id_vals},
        )
        facc = xr.DataArray(
            meta_df["area"].to_numpy(dtype=float),
            dims=["id"],
            coords={"id": gauge_id_vals},
        )
        obs_discharge_data = obs_discharge_data.sel(id=gauge_ids.data)

        slicing_condition = None
        if (
            lon_min is not None
            and lon_max is not None
            and lat_min is not None
            and lat_max is not None
        ):
            slicing_condition = (
                (x >= lon_min) & (x <= lon_max) & (y >= lat_min) & (y <= lat_max)
            )
            x = x.where(slicing_condition, drop=True)
            y = y.where(slicing_condition, drop=True)
            facc = facc.where(slicing_condition, drop=True)
            gauge_ids = gauge_ids.where(slicing_condition, drop=True)
            obs_discharge_data = obs_discharge_data.sel(id=gauge_ids.data)
        logger.info(f"There are {len(gauge_ids.data)} gauges in total.")
        logger.debug(f"Using observed variable: {observed_variable}")
        logger.debug(
            f"Observed discharge data after time selection: {obs_discharge_data}"
        )

        extraction_mode = "raw"
        gauge_ids_with_values_pre = None
        x_new_pre = None
        y_new_pre = None
        facc_new_pre = None

        if scc_gauges_file is not None:
            extraction_mode = "node"
            with load_ds(scc_gauges_file) as ds:
                logger.info("Getting gauge coordinates from scc gauges file.")
                try:
                    scc_x = np.asarray(ds.lon.values, dtype=float)
                    scc_y = np.asarray(ds.lat.values, dtype=float)
                except Exception:
                    scc_x = np.asarray(ds.x.values, dtype=float)
                    scc_y = np.asarray(ds.y.values, dtype=float)
                scc_ids = np.asarray(ds.station.values)
            valid_id_set = set(np.asarray(gauge_ids.values))
            keep_mask = np.isin(scc_ids, list(valid_id_set))
            if not np.any(keep_mask):
                msg = "No gauges with observed data found in scc gauges file."
                with ErrorLogger(logger):
                    raise ValueError(msg)
            x_new_pre = scc_x[keep_mask]
            y_new_pre = scc_y[keep_mask]
            gauge_ids_with_values_pre = scc_ids[keep_mask]
            try:
                facc_new_pre = facc.sel(id=gauge_ids_with_values_pre).values
            except Exception:
                facc_new_pre = np.full(len(gauge_ids_with_values_pre), np.nan)
        elif facc_file is not None:
            extraction_mode = "coords"
            with load_ds(facc_file) as ds:
                logger.info(
                    "Getting gauge coordinates by matching coordinates and flow accumulation."
                )
                out = Parallel(n_jobs=n_jobs, backend="loky")(
                    delayed(get_gauge_coords)(
                        ds,
                        ref_facc=facc_i,
                        facc_variable=facc_variable,
                        lonlat=[x_i, y_i],
                        method=gauge_location_method,
                        max_distance_cells=gauge_max_distance_cells,
                        max_error=gauge_max_error,
                        id=id_i,
                    )
                    for x_i, y_i, facc_i, id_i in zip(
                        x.values, y.values, facc.values, gauge_ids.values
                    )
                )
            x_new_pre, y_new_pre, facc_new_pre, gauge_ids_with_values_pre = (
                [],
                [],
                [],
                [],
            )
            for i, (xn, yn, fan) in enumerate(out):
                if xn is None or yn is None or fan is None:
                    continue
                x_new_pre.append(xn)
                y_new_pre.append(yn)
                facc_new_pre.append(fan)
                gauge_ids_with_values_pre.append(gauge_ids.data[i])
            x_new_pre = np.asarray(x_new_pre, dtype=float)
            y_new_pre = np.asarray(y_new_pre, dtype=float)
            facc_new_pre = np.asarray(facc_new_pre, dtype=float)
            gauge_ids_with_values_pre = np.asarray(gauge_ids_with_values_pre)

        # Read simulation data file-by-file and merge afterwards to reduce peak memory.
        model_files = _expand_input_files(model_data_path, file_name=model_file_name)
        if not model_files:
            msg = f"No model input files resolved from {model_data_path}."
            with ErrorLogger(logger):
                raise FileNotFoundError(msg)
        if len(model_files) > 1:
            logger.info(
                f"Model input resolves to {len(model_files)} files; processing incrementally."
            )

        do_spatial_crop = slicing_condition is not None
        n_jobs_eff = (
            min(max(1, int(n_jobs)), len(model_files)) if n_jobs is not None else 1
        )
        if n_jobs_eff > 1:
            logger.info(
                f"Reading {len(model_files)} model files in parallel with n_jobs={n_jobs_eff}."
            )
            sim_results = Parallel(n_jobs=n_jobs_eff, backend="loky")(
                delayed(_read_model_file_part)(
                    sim_file=sim_file,
                    sim_variable=sim_variable,
                    date_slice=date_slice,
                    do_spatial_crop=do_spatial_crop,
                    lat_min=lat_min,
                    lat_max=lat_max,
                    lon_min=lon_min,
                    lon_max=lon_max,
                    reduce_coords=extraction_mode != "node",
                    extraction_mode=extraction_mode,
                    gauge_ids_with_values=gauge_ids_with_values_pre,
                    x_new=x_new_pre,
                    y_new=y_new_pre,
                    resolution=resolution,
                    load_num_workers=n_jobs_eff,
                )
                for sim_file in model_files
            )
        else:
            if len(model_files) > 1:
                logger.info(f"Processing {len(model_files)} model files sequentially.")
            else:
                logger.info("Processing model file.")
            sim_results = [
                _read_model_file_part(
                    sim_file=sim_file,
                    sim_variable=sim_variable,
                    date_slice=date_slice,
                    do_spatial_crop=do_spatial_crop,
                    lat_min=lat_min,
                    lat_max=lat_max,
                    lon_min=lon_min,
                    lon_max=lon_max,
                    reduce_coords=extraction_mode != "node",
                    extraction_mode=extraction_mode,
                    gauge_ids_with_values=gauge_ids_with_values_pre,
                    x_new=x_new_pre,
                    y_new=y_new_pre,
                    resolution=resolution,
                    load_num_workers=n_jobs_eff,
                )
                for sim_file in model_files
            ]
        logger.info("Finished reading model files. Processing results.")
        sim_results = [res for res in sim_results if res is not None]
        if not sim_results:
            msg = "No simulation data could be loaded from model input files."
            with ErrorLogger(logger):
                raise ValueError(msg)

        sim_variable = sim_results[0][0] if sim_variable is None else sim_variable
        if extraction_mode == "node":
            filtered = [res for res in sim_results if res[0] == sim_variable]
            dropped_sim = len(sim_results) - len(filtered)
            if dropped_sim > 0:
                logger.warning(
                    f"Dropped {dropped_sim} model file(s) due to variable-name mismatch with '{sim_variable}'."
                )
            if not filtered:
                msg = (
                    f"No simulation data left for variable '{sim_variable}' after "
                    "processing all files."
                )
                with ErrorLogger(logger):
                    raise ValueError(msg)
            sim_parts = [res[1] for res in filtered]
            # keep node-matched coordinates/ids from the first valid file
            x_new_pre = np.asarray(filtered[0][2])
            y_new_pre = np.asarray(filtered[0][3])
            gauge_ids_with_values_pre = np.asarray(filtered[0][4])
            try:
                facc_new_pre = facc.sel(id=gauge_ids_with_values_pre).values
            except Exception:
                facc_new_pre = np.full(len(gauge_ids_with_values_pre), np.nan)
            sim_data_cropped = _concat_time_dataarrays(sim_parts).to_dataset(
                name=sim_variable
            )
        elif extraction_mode == "coords":
            filtered = [res for res in sim_results if res[0] == sim_variable]
            dropped_sim = len(sim_results) - len(filtered)
            if dropped_sim > 0:
                logger.warning(
                    f"Dropped {dropped_sim} model file(s) due to variable-name mismatch with '{sim_variable}'."
                )
            if not filtered:
                msg = (
                    f"No simulation data left for variable '{sim_variable}' after "
                    "processing all files."
                )
                with ErrorLogger(logger):
                    raise ValueError(msg)
            sim_parts = [res[1] for res in filtered]
            sim_data_cropped = _concat_time_dataarrays(sim_parts).to_dataset(
                name=sim_variable
            )
        else:
            sim_parts = [
                part for var_name, part in sim_results if var_name == sim_variable
            ]
            dropped_sim = sum(
                1 for var_name, _ in sim_results if var_name != sim_variable
            )
            if dropped_sim > 0:
                logger.warning(
                    f"Dropped {dropped_sim} model file(s) due to variable-name mismatch with '{sim_variable}'."
                )
            if not sim_parts:
                msg = (
                    f"No simulation data left for variable '{sim_variable}' after "
                    "processing all files."
                )
                with ErrorLogger(logger):
                    raise ValueError(msg)

            if len(sim_parts) == 1:
                sim_data_cropped = sim_parts[0]
            else:
                sim_data_cropped = xr.concat(
                    sim_parts, dim="time", join="outer"
                ).sortby("time")
                if "time" in sim_data_cropped.coords:
                    time_vals = np.asarray(sim_data_cropped["time"].values)
                    if time_vals.size:
                        _, unique_pos = np.unique(time_vals, return_index=True)
                        if unique_pos.size != time_vals.size:
                            sim_data_cropped = sim_data_cropped.isel(
                                time=np.sort(unique_pos)
                            )
        logger.debug(f"Sim data after cropping (time and space): {sim_data_cropped}")

        logger.info("Resampling to coarser calendar if needed...")
        # Harmonize temporal frequency between simulation and observations.
        # The helper returns the possibly resampled simulation first.
        resampled_sim_data, resampled_obs_data = resample_to_coarser_calendar(
            sim_data_cropped, obs_discharge_data
        )
        logger.debug(f"Obs data.time after resampling: {resampled_obs_data.time}")
        logger.debug(f"Sim data.time after resampling: {resampled_sim_data.time}")

        # Ensure unique gauge ids (xarray .sel requires unique index)
        id_index = pd.Index(gauge_ids.values)
        if not id_index.is_unique:
            _, unique_pos = np.unique(id_index.values, return_index=True)
            unique_pos = np.sort(unique_pos)
            dup_count = len(id_index) - len(unique_pos)
            logger.warning(
                f"Found {dup_count} duplicate gauge id(s). Keeping first occurrence."
            )
            gauge_ids = gauge_ids.isel(id=unique_pos)
            x = x.isel(id=unique_pos)
            y = y.isel(id=unique_pos)
            facc = facc.isel(id=unique_pos)
            resampled_obs_data = resampled_obs_data.isel(id=unique_pos)

        if direct_comparison:
            logger.info("Selecting overlapping time period for direct comparison...")
            # Determine one common inclusive time window and apply it to both
            # series to ensure metric computation compares the same period.
            common_time_window = get_overlapping_time_slice(
                sim_data_cropped, obs_discharge_data
            )
            logger.info(
                f"Overlapping time is from {common_time_window.start} "
                f"to {common_time_window.stop}"
            )
            obs_discharge_data = obs_discharge_data.sel(time=common_time_window)
            sim_data_cropped = sim_data_cropped.sel(time=common_time_window)
            logger.debug(
                f"Observed data.time after overlapping slice: {obs_discharge_data.time}"
            )
            logger.debug(
                f"Sim data.time after overlapping slice: {sim_data_cropped.time}"
            )

    # Drop gauges without any observed data in the selected period
    logger.info("Filtering gauges without observed data in the provided time range...")
    obs_discharge_data = obs_discharge_data.sel(id=gauge_ids.data)
    valid_obs = obs_discharge_data.notnull().any(dim="time")
    valid_ids = obs_discharge_data["id"].where(valid_obs, drop=True).values
    # If min_overlapping_years is set, further drop gauges that do not have
    # enough observed years in the already aligned comparison period.
    if min_overlapping_years is not None:
        ids_before = np.asarray(valid_ids)
        logger.info(
            f"Applying minimum observed-year filter in aligned period: >= {int(min_overlapping_years)} years."
        )
        valid_ids, dropped_year_filter = filter_ids_by_observed_years(
            valid_ids=ids_before,
            obs_discharge_data=obs_discharge_data,
            min_overlapping_years=min_overlapping_years,
        )
        logger.info(
            f"Gauge count after observed-year filter: {int(ids_before.size)} -> {int(np.asarray(valid_ids).size)}"
        )
        if dropped_year_filter:
            logger.debug(
                f"Dropped gauges by observed-year filter (id, observed_years): {dropped_year_filter}"
            )

    if valid_ids.size == 0:
        msg = "No gauges with observed data in the selected time range."
        with ErrorLogger(logger):
            raise ValueError(msg)
    gauge_ids = gauge_ids.sel(id=valid_ids)
    x = x.sel(id=valid_ids)
    y = y.sel(id=valid_ids)
    facc = facc.sel(id=valid_ids)
    obs_discharge_data = obs_discharge_data.sel(id=valid_ids)
    logger.info(
        f"There are {len(gauge_ids.data)} gauges with observed data in the selected time range."
    )

    # Preparing observed data
    logger.info("preparing obs data...")
    obs_discharge_data = obs_discharge_data.reindex(id=gauge_ids.data)
    facc_da = xr.DataArray(
        facc, name="facc", dims=["id"], coords={"id": gauge_ids.data}
    )
    observed_data = xr.Dataset({"facc": facc_da, "discharge": obs_discharge_data})
    if write_input_data_cache:
        logger.info(f"Saving obs data to {obs_output_file}...")
        write_xarray_to_file(observed_data, obs_output_file)
    else:
        logger.info(
            "Skipping saving obs data to file as write_input_data_cache is False."
        )
        observed_data.load()

    # Preaparing simulation data for gauges
    logger.info("preparing sim data...")
    # Case 1: discharge.nc files with Qsim/Qobs_<id> variables available (e.g. mRM v5+ output)
    if discharge_nc_files:
        simulation_discharge = sim_data_cropped
        gauge_ids_with_values = np.asarray(gauge_ids.values)
        x_new = np.asarray(x.values)
        y_new = np.asarray(y.values)
        facc_new = (
            np.asarray(facc.values)
            if facc is not None
            else np.full(len(gauge_ids_with_values), np.nan)
        )
    # Case 2: scc gauges file provided. Input is node based (mRM v6+)
    elif scc_gauges_file is not None:
        if sim_variable is None:
            sim_variable = (
                "discharge"
                if "discharge" in sim_data_cropped.data_vars
                else get_single_data_var(sim_data_cropped)
            )
        if sim_variable is None:
            msg = "Could not determine simulation discharge variable."
            with ErrorLogger(logger):
                raise ValueError(msg)
        logger.info(
            f"Using pre-extracted node discharge variable '{sim_variable}' from file-wise processing."
        )
        sim = sim_data_cropped[sim_variable]
        gauge_ids_with_values = np.asarray(gauge_ids_with_values_pre)
        x_new = np.asarray(x_new_pre)
        y_new = np.asarray(y_new_pre)
        facc_new = np.asarray(facc_new_pre)
    # Case 3: flow-accumulation file provided. This enables matching based on coordinates and flow accumulation. (mRM v5 style)
    elif facc_file is not None:
        sim_variable = (
            get_single_data_var(sim_data_cropped)
            if sim_variable is None
            else sim_variable
        )
        logger.info(
            f"Using pre-extracted coordinate-based discharge variable '{sim_variable}' from file-wise processing."
        )
        sim = sim_data_cropped[sim_variable]
        gauge_ids_with_values = np.asarray(gauge_ids_with_values_pre)
        x_new = np.asarray(x_new_pre)
        y_new = np.asarray(y_new_pre)
        facc_new = np.asarray(facc_new_pre)
    else:
        error_msg = (
            "Neither facc_file or scc gauges file are provided. "
            "Gauge location can not be determined"
        )
        with ErrorLogger(logger):
            raise ValueError(error_msg)

    if len(x_new) == 0:
        msg = "There are no gauges that could be found."
        with ErrorLogger(logger):
            raise ValueError(msg)
    logger.info(f"There are {len(x_new)} gauges")
    logger.info("creating sim dataset")
    if not discharge_nc_files:
        if isinstance(sim, list):
            # Drop per-gauge location coords before concat; they differ across ids and
            # trigger expensive coord-merging logic without adding value here.
            sim = [
                da.drop_vars([coord for coord in ("lat", "lon") if coord in da.coords])
                for da in sim
            ]
            simulation_discharge = xr.concat(
                sim, dim="id", coords="minimal", compat="override"
            )
        else:
            simulation_discharge = sim
    # Ensure sim ids align with observed valid_ids (node outputs may miss some)
    sim_ids = np.asarray(simulation_discharge["id"].values)
    valid_id_values = np.asarray(valid_ids)
    common_ids = np.intersect1d(sim_ids, valid_id_values)
    if common_ids.size == 0:
        msg = "No overlapping gauge ids between simulation and observations."
        with ErrorLogger(logger):
            raise ValueError(msg)
    ids_arr = np.asarray(gauge_ids_with_values)
    keep_mask = np.isin(ids_arr, common_ids)
    if not np.all(keep_mask):
        logger.info(
            f"Dropping {int((~keep_mask).sum())} gauges missing in simulation output."
        )
    gauge_ids_with_values = ids_arr[keep_mask]
    x_new = np.asarray(x_new)[keep_mask]
    y_new = np.asarray(y_new)[keep_mask]
    facc_new = np.asarray(facc_new)[keep_mask]
    simulation_discharge = simulation_discharge.sel(id=gauge_ids_with_values)
    # DataArray has no drop_dims(); remove leftover singleton dims/coords safely.
    for dim in ("lat", "lon"):
        if dim in simulation_discharge.dims:
            if simulation_discharge.sizes.get(dim, 0) == 1:
                simulation_discharge = simulation_discharge.isel({dim: 0}, drop=True)
            else:
                logger.warning(
                    f"Simulation discharge still has non-singleton '{dim}' dimension; keeping it."
                )
    simulation_discharge = simulation_discharge.drop_vars(
        [coord for coord in ("lat", "lon") if coord in simulation_discharge.coords]
    )
    # if "lat" in simulation_discharge.data_vars:
    #     simulation_discharge = simulation_discharge.drop_vars(["lat"])
    # if "lon" in simulation_discharge.data_vars:
    #     simulation_discharge = simulation_discharge.drop_vars(["lon"])
    facc_ids = xr.DataArray(
        data=np.array(facc_new), dims=["id"], coords={"id": gauge_ids_with_values}
    )

    x_ids = xr.DataArray(
        data=np.array(x_new), dims=["id"], coords={"id": gauge_ids_with_values}
    )
    y_ids = xr.DataArray(
        data=np.array(y_new), dims=["id"], coords={"id": gauge_ids_with_values}
    )

    # 4) Build a new Dataset from this 2D DataArray
    sim_data = xr.Dataset(
        {
            "discharge": simulation_discharge,
            "facc": facc_ids,
            "x": x_ids,
            "y": y_ids,
        }
    )
    if write_input_data_cache:
        logger.info(f"Saving sim data to {sim_output_file}...")
        write_xarray_to_file(sim_data, sim_output_file)
    else:
        logger.info(
            "Skipping saving sim data to file as write_input_data_cache is False."
        )
        sim_data.load()
    return observed_data, sim_data


def boostap_statistics(
    index,
    id,
    model_da,
    observed_da,
    total_years_sim,
    total_years_obs,
    n_bootstrap_years,
):
    """Calculate the statistics for one boostrap selection."""
    np.random.seed(index)
    years_sim = np.random.choice(total_years_sim, size=n_bootstrap_years)
    years_obs = np.random.choice(total_years_obs, size=n_bootstrap_years)
    logger.debug(f"years_sim: {years_sim}")
    logger.debug(f"years_obs: {years_obs}")
    sim_da_sel = xr.concat(
        [model_da.where(model_da.time.dt.year == year) for year in years_sim],
        dim="time",
    )
    obs_da_sel = xr.concat(
        [observed_da.where(observed_da.time.dt.year == year) for year in years_sim],
        dim="time",
    )
    alpha, beta, gamma = np.nan, np.nan, np.nan
    if (
        (id in obs_da_sel.id and id in sim_da_sel.id)
        and not sim_da_sel.isnull().all()
        and not obs_da_sel.isnull().all()
    ):
        try:
            sim_id = sim_da_sel.sel(id=id)
            obs_id = obs_da_sel.sel(id=id)
            clim_sim = get_clim_from_ds(sim_id)
            clim_obs = get_clim_from_ds(obs_id)
            alpha = sim_id.mean(skipna=True) / obs_id.mean(skipna=True)
            beta = sim_id.std(skipna=True) / obs_id.std(skipna=True)
            gamma = spearman_correlation(clim_sim, clim_obs)[0]
            logger.debug(
                f"results for index {index} and gauge {id}: alpha={alpha:.3f}, beta={beta:.3f}, gamma={gamma:.3f}"
            )
            if np.isnan(alpha):
                logger.debug("Alpha is none for:")
                logger.debug(f"sim: {sim_id} and clim {clim_sim}")
                logger.debug(f"obs: {obs_id} and clim {clim_obs}")
        except Exception as e:
            logger.error(f"Error for index {index} and id {id} with error {e}")
    else:
        logger.warning(
            f"(id in obs_da_sel.id = {id in obs_da_sel.id} and id in sim_da_sel.id = {id in sim_da_sel.id}) or not sim_da_sel.isnull().all() = {sim_da_sel.isnull().all()} or not obs_da_sel.isnull().all() = {obs_da_sel.isnull().all()}"
        )
    return {
        "index": index,
        "id": id,
        "alpha": float(alpha),
        "beta": float(beta),
        "gamma": float(gamma),
    }


def _filter_ids_by_overlapping_years(model_da, observed_da, min_overlapping_years):
    """Return ids with at least the requested overlap years plus dropped-id details."""
    model_ids = set(np.asarray(model_da["id"].values).tolist())
    """Return ids with enough obs/sim overlap years plus dropped-id details.

    A year counts as overlapping for a station if at least one timestep in that
    calendar year has non-NaN values in both observed and simulated series.
    """
    if min_overlapping_years <= 0:
        min_overlapping_years = 1

    obs_ids = pd.Index(np.asarray(observed_da["id"].values))
    model_ids = pd.Index(np.asarray(model_da["id"].values))
    common_ids = obs_ids.intersection(model_ids)

    overlap_counts = pd.Series(dtype=int)
    if len(common_ids) > 0:
        sim_common = model_da.sel(id=common_ids.values)
        obs_common = observed_da.sel(id=common_ids.values)
        sim_aligned, obs_aligned = xr.align(sim_common, obs_common, join="inner")

        has_time = "time" in sim_aligned.dims and sim_aligned.sizes.get("time", 0) > 0
        if has_time:
            overlap = sim_aligned.notnull() & obs_aligned.notnull()
            overlap_yearly = overlap.groupby("time.year").any(dim="time")
            yearly_counts = overlap_yearly.sum(dim="year").astype(int)
            overlap_counts = pd.Series(
                np.asarray(yearly_counts.values, dtype=int),
                index=pd.Index(np.asarray(yearly_counts["id"].values)),
            )

    eligible_ids = []
    dropped_ids = []
    for gauge_id in obs_ids:
        years = int(overlap_counts.get(gauge_id, 0))
        if gauge_id in model_ids and years >= min_overlapping_years:
            eligible_ids.append(gauge_id)
        else:
            dropped_ids.append((gauge_id, years))

    return np.asarray(eligible_ids), dropped_ids


@log_arguments()
def evaludate_discharge_data(  # noqa: PLR0913
    model_data_path,
    observed_data_path,
    facc_file=None,
    facc_variable="L11_fAcc",
    scc_gauges_file=None,
    output_path=None,
    evaluation_gauges=None,
    n_jobs=1,
    sim_variable=None,
    observed_variable=None,
    model_file_name=None,
    lon_min=-180,
    lon_max=180,
    lat_min=-56,
    lat_max=84,
    direct_comparison=True,
    resolution=0.1,
    n_bootstrap_years=5,
    n_boostrap_selections=0,
    start_date=None,
    end_date=None,
    overwrite=False,
    only_plot=False,
    save_hydrograph=False,
    min_overlapping_years=1,
    write_input_data_cache=False,
    gauge_location_method="basinex",
    gauge_max_distance_cells=3,
    gauge_max_error=0.1,
    hydrograph_plots=None,
):
    """Compare simulated with observed discharge directly or via bootstrapping.

    Loads/caches data, harmonizes temporal resolution, optionally bootstraps
    across years, and writes per-gauge plots and a results table.
    """
    output_path = Path(output_path)
    stats_output_file = output_path / "results.csv"
    if not only_plot or not stats_output_file.is_file():
        observed_ds, model_ds = Q_data_to_xarray(
            model_data_path=model_data_path,
            observed_data_path=observed_data_path,
            facc_file=facc_file,
            facc_variable=facc_variable,
            scc_gauges_file=scc_gauges_file,
            observed_variable=observed_variable,
            sim_variable=sim_variable,
            model_file_name=model_file_name,
            evaluation_gauges=evaluation_gauges,
            model_keyword="mrm",
            saving_path=output_path,
            lon_min=lon_min,
            lon_max=lon_max,
            lat_min=lat_min,
            lat_max=lat_max,
            resolution=resolution,
            n_jobs=n_jobs,
            date_slice=slice(start_date, end_date),
            overwrite=overwrite,
            direct_comparison=direct_comparison,
            write_input_data_cache=write_input_data_cache,
            min_overlapping_years=min_overlapping_years,
            gauge_location_method=gauge_location_method,
            gauge_max_distance_cells=gauge_max_distance_cells,
            gauge_max_error=gauge_max_error,
        )
        logger.info("Procured discharge data")
        model_da = model_ds["discharge"]
        observed_da = observed_ds["discharge"]
        logger.debug(f"Model dataarray: {model_da}")
        logger.debug(f"Observed dataarray: {observed_da}")
        logger.info("Starting to calculate metrics")
        results = []
        if (
            n_bootstrap_years is not None
            and n_boostrap_selections is not None
            and n_boostrap_selections > 0
            and not direct_comparison
        ):
            logger.info(
                f"Bootstrapping with {n_boostrap_selections} selections with {n_bootstrap_years} years each."
            )
            results = []
            model_da = model_da.dropna(dim="time", how="all")
            observed_da = observed_da.dropna(dim="time", how="all")
            total_years_sim = np.unique(model_da.time.dt.year.data)
            total_years_obs = np.unique(observed_da.time.dt.year.data)
            logger.info(f"Observed years with non nan values: {total_years_obs}")
            logger.info(f"Simulated years with non nan values: {total_years_obs}")
            # for index in range(n_boostrap_selections):
            ids_sim = np.unique(model_da.id.values)
            ids_obs = np.unique(observed_da.id.values)
            results = Parallel(n_jobs=n_jobs, backend="loky")(
                delayed(boostap_statistics)(
                    index=index,
                    id=id,
                    model_da=model_da,
                    observed_da=observed_da,
                    total_years_sim=total_years_sim,
                    total_years_obs=total_years_obs,
                    n_bootstrap_years=n_bootstrap_years,
                )
                for index, id in itertools.product(
                    range(n_boostrap_selections), ids_sim
                )
                if id in ids_obs
            )
        results_direct = Parallel(n_jobs=n_jobs, backend="loky")(
            delayed(gen_hydrograph_by_data_sets)(
                simulations=model_da.sel(id=id),
                observation=observed_da.sel(id=id),
                precipitation=None,
                output_file=output_path / f"hydrograph_{int(id)}.pdf",
                area=model_ds["facc"].sel(id=id).data,
                id=id,
                calc_stats=direct_comparison,
                raise_exceptions=False,
                title=f"{id} at {model_ds['x'].sel(id=id).data} - {model_ds['y'].sel(id=id).data}",
                x=model_ds["x"].sel(id=id).data,
                y=model_ds["y"].sel(id=id).data,
                save=save_hydrograph,
                plot_code=hydrograph_plots,
            )
            for id in observed_ds.id.values
            if id in model_ds.id.values
        )
        if not results:
            logger.info("Using the results from direct comparison.")
            logger.info(results_direct)
            results_df = pd.DataFrame(results_direct)
        else:
            logger.info("Using the results from bootstraping.")
            results_df = pd.DataFrame(results)
        results_df.to_csv(stats_output_file)
    else:
        logger.info(f"Reading results from {stats_output_file}...")
        results_df = pd.read_csv(stats_output_file, index_col=0)
    # only plot cdf if more than 5 results to avoid plots without enough data points
    if len(results_df) > 5:
        plot_cdf(results_df, output_path, boostrap_iterations=n_boostrap_selections)
    else:
        logger.info("Too few gauges for CDF plots.")
    plot_map(results_df, output_path)


def plot_kde(results_df, output_path):
    """Create kde plots of alpha, beta and gamma."""
    sns.kdeplot(
        data=results_df,  # The DataFrame with results
        x="alpha",  # The x-axis is the alpha value
        hue="id",  # Use column names for coloring
        palette="tab10",  # Set a color palette
        fill=True,  # Fill the KDE areas for better visualization
        alpha=0.6,  # Set transparency
        common_norm=False,  # Ensure each column has its own normalization
    )
    sns.kdeplot(
        data=results_df,
        x="alpha",
        hue="id",
        palette="tab10",
        cumulative=True,
        linestyle="--",
        common_norm=False,
    )
    plt.axvline(x=1, color="black", linestyle="--", linewidth=1)
    plt.xlim(-0.05, 2)
    plt.savefig(output_path / "alpha.png")
    plt.close()

    sns.kdeplot(
        data=results_df,  # The DataFrame with results
        x="beta",  # The x-axis is the alpha value
        hue="id",  # Use column names for coloring
        palette="tab10",  # Set a color palette
        fill=True,  # Fill the KDE areas for better visualization
        alpha=0.6,  # Set transparency
        common_norm=False,  # Ensure each column has its own normalization
    )
    sns.kdeplot(
        data=results_df,
        x="beta",
        hue="id",
        palette="tab10",
        cumulative=True,
        linestyle="--",
        common_norm=False,
    )
    plt.axvline(x=1, color="black", linestyle="--", linewidth=1)
    plt.xlim(-0.05, 2)
    plt.savefig(output_path / "beta.png")
    plt.close()

    sns.kdeplot(
        data=results_df,  # The DataFrame with results
        x="gamma",  # The x-axis is the alpha value
        hue="id",  # Use column names for coloring
        palette="tab10",  # Set a color palette
        fill=True,  # Fill the KDE areas for better visualization
        alpha=0.6,  # Set transparency
        common_norm=False,  # Ensure each column has its own normalization
    )
    sns.kdeplot(
        data=results_df,
        x="gamma",
        hue="id",
        palette="tab10",
        cumulative=True,
        linestyle="--",
        common_norm=False,
    )
    plt.axvline(x=1, color="black", linestyle="--", linewidth=1)
    plt.xlim(-0.05, 2)
    plt.savefig(output_path / "gamma.png")
    plt.close()


def plot_map(  # noqa: PLR0915
    results_df,
    output_path,
    variables=None,
    lon_col="x",
    lat_col="y",
    cmap="viridis",
    point_size=6,
    dpi=200,
):
    """Plot gauge maps for each variable using color-coded points.

    The map extent is expanded by 10% of the lon/lat range in each direction.
    """
    try:
        import cartopy.crs as ccrs
        import cartopy.feature as cfeature
    except Exception as exc:
        logger.error("cartopy is required for plot_map but could not be imported.")
        raise exc

    output_path = Path(output_path)
    if not output_path.is_dir():
        output_path.mkdir(parents=True, exist_ok=True)

    if lon_col not in results_df.columns or lat_col not in results_df.columns:
        msg = f"Missing lon/lat columns '{lon_col}'/'{lat_col}' in results_df."
        with ErrorLogger(logger):
            raise ValueError(msg)

    df = results_df.copy()

    # Coerce lon/lat to numeric (handles object values like array(nan)/0-d arrays)
    df = df.dropna(subset=[lon_col, lat_col])
    for col in (lon_col, lat_col):
        if col in df.columns:
            df[col] = df[col].map(_as_scalar_or_nan)
            df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.replace([np.inf, -np.inf], np.nan)
    if df.empty:
        logger.warning("No valid gauge coordinates available for map plotting.")
        return

    # Determine variables to plot
    if variables is None:
        skip = {lon_col, lat_col, "id"}
        variables = [
            c
            for c in df.columns
            if c not in skip and np.issubdtype(df[c].dtype, np.number)
        ]
    else:
        variables = [v for v in variables if v in df.columns]

    if not variables:
        logger.warning("No numeric variables found for map plotting.")
        return

    lons = df[lon_col].astype(float).values
    lats = df[lat_col].astype(float).values
    min_lon, max_lon = np.nanmin(lons), np.nanmax(lons)
    min_lat, max_lat = np.nanmin(lats), np.nanmax(lats)
    lon_range = max_lon - min_lon
    lat_range = max_lat - min_lat
    lon_pad = lon_range * 0.1 if lon_range > 0 else 0.1
    lat_pad = lat_range * 0.1 if lat_range > 0 else 0.1

    size_scale = 1.0
    if len(df) > 200:
        size_scale = (200 / len(df)) ** 0.5
    point_size = max(4, point_size * size_scale)

    for var in variables:
        vals = df[var].astype(float).values
        if np.all(np.isnan(vals)):
            logger.warning(f"Skipping map for {var}: all values are NaN.")
            continue

        # Determine value range and colorbar extension
        extend = "neither"
        if var == "kge":
            logger.info("Setting kge colorbar limits to -0.5 and 1.0")
            vmin, vmax = -0.5, 1.0
            if np.nanmin(vals) < vmin:
                extend = "min"
        elif var == "nse":
            logger.info("Setting nse colorbar limits to -0.1 and 1.0")
            vmin, vmax = -0.1, 1.0
            if np.nanmin(vals) < vmin:
                extend = "min"
        else:
            logger.info(f"Setting colorbar limits for {var} based on data range.")
            vmin = np.nanmin(vals)
            vmax = np.nanmax(vals)

        if np.isfinite(vmin) and np.isfinite(vmax) and vmin == vmax:
            vmin -= 1.0
            vmax += 1.0

        fig = plt.figure(figsize=(7, 5))
        ax = plt.axes(projection=ccrs.PlateCarree())
        ax.set_extent(
            [
                min_lon - lon_pad,
                max_lon + lon_pad,
                min_lat - lat_pad,
                max_lat + lat_pad,
            ],
            crs=ccrs.PlateCarree(),
        )
        ax.add_feature(cfeature.BORDERS, linewidth=0.6)
        ax.add_feature(cfeature.COASTLINE, linewidth=0.5)
        ax.add_feature(cfeature.LAND, facecolor="0.95")
        ax.add_feature(cfeature.OCEAN, facecolor="0.97")

        cmap_obj = plt.get_cmap(cmap).copy()
        if extend == "min":
            cmap_obj.set_under("lightgray")
        norm = mcolors.Normalize(vmin=vmin, vmax=vmax, clip=False)
        sc = ax.scatter(
            lons,
            lats,
            c=vals,
            cmap=cmap_obj,
            norm=norm,
            s=point_size,
            alpha=0.85,
            edgecolor="white",
            linewidth=0.25,
            transform=ccrs.PlateCarree(),
        )
        cb = plt.colorbar(sc, ax=ax, orientation="vertical", shrink=0.8, extend=extend)
        cb.set_label(var)
        ax.set_title(f"{var} by gauge")
        fig.tight_layout()
        fig.savefig(output_path / f"map_{var}.png", dpi=dpi)
        plt.close(fig)


@log_errors()
def plot_cdf(df, output_path, boostrap_iterations=None):
    """Create CDF plots for alpha, beta, and gamma.

    The plots are generated for different subselections
    (by catchment, bootstrap-mean, or all results).
    """
    # The variables to plot
    logger.info(f"In total there are {len(df['id'].unique())} catchments of which ")
    logger.info(
        f"   {len(df.dropna(subset=['alpha'], how='any')['id'].unique())} have all alpha values"
    )
    logger.info(
        f"   {len(df.dropna(subset=['beta'], how='any')['id'].unique())} have beta values "
    )
    logger.info(
        f"   {len(df.dropna(subset=['gamma'], how='any')['id'].unique())} have all gamma values "
    )
    logger.info(
        f"   {len(df.dropna(subset=['alpha', 'beta', 'gamma'], how='any')['id'].unique())} have all values "
    )
    df = df.dropna(subset=["alpha", "beta", "gamma"], how="any")
    logger.info(df.head())
    variables = ["alpha", "beta", "gamma", "kge", "nse"]
    regions = {
        1: "Africa",
        2: "Asia",
        3: "South America",
        4: "North/Central America",
        5: "SW Pacific",
        6: "Europe",
    }
    cb_colors = [
        "#000000",
        "#E69F00",
        "#56B4E9",
        "#009E73",
        "#F0E442",
        "#0072B2",
        "#D55E00",
        "#CC79A7",
    ]

    unique_ids = df["id"].unique()
    logger.info(f"Creating a cdf plot with {len(unique_ids)} stations")

    logger.info("All values")
    plot_modes = ["global", "global_color_by_region", "regions"]
    region_colors = {
        region: cb_colors[i % len(cb_colors)]
        for i, region in enumerate(regions.values())
    }

    def _region_from_id(gauge_id):
        gauge_id_str = str(gauge_id)
        for region_id, region_name in regions.items():
            if gauge_id_str.startswith(str(region_id)):
                return region_name
        return "Unknown"

    for plot in plot_modes:
        for var in variables:
            var_df = df[["id", var]].dropna(subset=[var]).copy()
            if var_df.empty:
                logger.warning("No valid values for %s. Skipping %s plot.", var, plot)
                continue

            var_df["region"] = var_df["id"].apply(_region_from_id)
            global_sorted = var_df.sort_values(by=var).reset_index(drop=True)
            global_sorted["cdf"] = np.arange(1, len(global_sorted) + 1) / len(
                global_sorted
            )

            fig, ax = plt.subplots(figsize=(6, 4))
            med = float(np.nanmedian(global_sorted[var].values))

            if plot == "global":
                ax.plot(
                    global_sorted[var].values,
                    global_sorted["cdf"].values,
                    color="lightgray",
                    linewidth=1.0,
                )
                ax.scatter(
                    global_sorted[var].values,
                    global_sorted["cdf"].values,
                    s=16,
                    label=f"all stations (n={len(global_sorted)})",
                )
            elif plot == "global_color_by_region":
                ax.plot(
                    global_sorted[var].values,
                    global_sorted["cdf"].values,
                    color="lightgray",
                    linewidth=1.0,
                )
                for region_name in regions.values():
                    region_df = global_sorted[global_sorted["region"] == region_name]
                    if region_df.empty:
                        continue
                    ax.scatter(
                        region_df[var].values,
                        region_df["cdf"].values,
                        s=16,
                        color=region_colors[region_name],
                        label=f"{region_name} (n={len(region_df)})",
                    )
            elif plot == "regions":
                for region_name in regions.values():
                    region_df = var_df[var_df["region"] == region_name].sort_values(
                        by=var
                    )
                    if region_df.empty:
                        continue
                    region_cdf = np.arange(1, len(region_df) + 1) / len(region_df)
                    ax.plot(
                        region_df[var].values,
                        region_cdf,
                        color=region_colors[region_name],
                        linewidth=1.0,
                    )
                    ax.scatter(
                        region_df[var].values,
                        region_cdf,
                        s=16,
                        color=region_colors[region_name],
                        label=f"{region_name} (n={len(region_df)})",
                    )

            ax.axvline(
                med,
                color="red",
                linestyle="dotted",
                linewidth=1,
                label=f"median = {med:.3f}",
            )
            title = f"CDF of {var} "
            if plot == "global_color_by_region":
                title += " (global, points colored by region)"
            elif plot == "regions":
                title += " (per-region CDFs)"
            if boostrap_iterations is not None and boostrap_iterations > 0:
                title += f" and {boostrap_iterations} bootstrap iterations"
            ax.set_title(title)
            ax.set_xlabel(var)
            ax.set_ylabel("CDF")
            ax.legend()
            if var in ["kge", "nse"]:
                ax.set_xlim(-0.5, 1.0)
                ax.set_ylim(0.0, 1.01)
            else:
                ax.set_ylim(0.0, 1.01)
                values = global_sorted[var].values
                xmin = values.min() if values.min() > -2 else np.quantile(values, 0.05)
                xmax = values.max() if values.max() < 3 else np.quantile(values, 0.95)
                if xmax - xmin > 6:
                    median_val = float(np.median(values))
                    xmin = max(xmin, median_val - 3)
                    xmax = min(xmax, median_val + 3)
                ax.set_xlim(xmin, xmax)

            plt.tight_layout()
            plt.savefig(
                output_path / f"cdf_{var}_{plot.strip().lower().replace(' ', '_')}.png",
                dpi=450,
            )
            plt.close()

    logger.info("Done! Check the saved PNG files for your CDF plots.")
