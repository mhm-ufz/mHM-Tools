"""
Compare simulated with observed discharge either directly or using a bootstraping approach.

Authors
-------
- Jeisson Leal
- Simon Lüdke
"""

import itertools
import logging
from glob import glob
from pathlib import Path

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

    msg = "Could not find river node x/y coordinates in model dataset."
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


def _load_discharge_nc_collection(model_data_path, date_slice=None):
    """Load discharge.nc files containing Qsim/Qobs_<id> and return sim/obs datasets."""
    def _expand_paths(path_like):
        if isinstance(path_like, (list, tuple)):
            return [Path(p) for p in path_like]
        path_str = str(path_like)
        if any(w in path_str for w in ("*", "?", "[", "]")):
            matches = [Path(p) for p in glob(path_str)]
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
                "Some gauges are farther than %.6f from nearest node; "
                "using nearest anyway (max %.6f).",
                max_dist,
                float(np.max(distances[far_mask])),
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


def get_gauge_coords(
    ds,
    facc,
    lonlat=None,
    xy=None,
    cell_diff=1,
    max_cell_diff=3,
    diff_percent=10,
    id=None,
):
    """Find correct gauge location.

    Takes mrm_restart file, an approximate lat lon value and and then,
    finds the cell with the value closest to a given flow accumulation.
    Returns lon,lat, flow accumulation.
    """
    if lonlat is not None:
        lon = lonlat[0]
        lat = lonlat[1]
        lon_col = int((lon - ds.xllcorner_L0) / ds.cellsize_L1)
        lat_row = int(ds.nrows_L1 - (lat - (ds.yllcorner_L0)) / ds.cellsize_L1)
    else:
        lon_col = xy[0]
        lat_row = xy[1]
    if cell_diff > 0:
        ds_cut = ds.sel(
            nrows0=slice(lat_row - cell_diff, lat_row + cell_diff),
            nrows1=slice(lat_row - cell_diff, lat_row + cell_diff),
            nrows11=slice(lat_row - cell_diff, lat_row + cell_diff),
            ncols0=slice(lon_col - cell_diff, lon_col + cell_diff),
            ncols1=slice(lon_col - cell_diff, lon_col + cell_diff),
            ncols11=slice(lon_col - cell_diff, lon_col + cell_diff),
        )
        abs_diff = np.abs(ds_cut.L11_fAcc - facc)
        try:
            min_index = np.unravel_index(
                np.argmin(abs_diff.data), ds_cut.L11_fAcc.shape
            )
        except ValueError as ve:
            logger.error(str(ve))
            logger.info(abs_diff)
            logger.info(ds_cut.L11_fAcc)
            logger.info(facc)
            return None, None, None
        if lonlat:
            lon_x = ds_cut.L1_domain_lon.data[min_index[0], 0]
            lat_y = ds_cut.L1_domain_lat.data[0, min_index[1]]
        # else:
        #     lon_x, lat_y = ds_cut.nrows0.values[1], ds_cut.nrows0.values[1] ## Not yet working
        found_facc = ds_cut.L11_fAcc.data[min_index[0], min_index[1]]
    else:
        ds_cut = ds.sel(
            nrows0=lat_row,
            nrows1=lat_row,
            nrows11=lat_row,
            ncols0=lon_col,
            ncols1=lon_col,
            ncols11=lon_col,
        )
        if lonlat:
            lon_x = ds_cut.L1_domain_lon.data
            lat_y = ds_cut.L1_domain_lat.data
        else:
            lon_x = lon_col
            lat_y = lat_row
        found_facc = ds_cut.L11_fAcc.data
    if abs(found_facc - facc) < diff_percent / 100 * facc:
        logger.info(f"{lon_x}, {lat_y}, {found_facc}, {facc}, {cell_diff}")
        return lon_x, lat_y, found_facc
    if cell_diff < max_cell_diff:
        logger.debug(
            f"No similar flow acc found. Increasing search radius to {cell_diff + 1} cells in each direction."
        )
        return get_gauge_coords(
            ds,
            facc,
            lonlat=[lon, lat],
            cell_diff=cell_diff + 1,
            max_cell_diff=3,
            diff_percent=10,
            id=id,
        )
    logger.warning(f"No similar flow accumulation found nearby for gauge {id}.")
    logger.debug("None, None, None")
    return None, None, None


def load_ds(file_path, file_name=None):
    """Load an xarray dataset from disk."""
    if file_name is not None:
        logger.info(f"Loading dataset from file {file_path}/{file_name}...")
    else:
        logger.info(f"Loading dataset from file {file_path}...")
    return get_dataset_from_path(file_path, file_name=file_name, use_mfdataset=True)


def Q_data_to_xarray(  # noqa: PLR0913, PLR0915, PLR0912
    model_data_path,
    observed_data_path,
    sim_variable,
    observed_variable,
    model_keyword,
    mrm_restart_file=None,
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

    # ignored if overwrite is True
    if sim_output_file.is_file() and not overwrite:
        logger.info("reading sim data from file...")
        sim_data = load_ds(sim_output_file)
        sim_data = sim_data.sel(time=date_slice)
    if obs_output_file.is_file() and not overwrite:
        logger.info("reading obs data from file...")
        observed_data = load_ds(obs_output_file)
        observed_data = observed_data.sel(time=date_slice)
    if obs_output_file.is_file() and sim_output_file.is_file() and not overwrite:
        return observed_data, sim_data

    # creating saving path
    saving_path = Path(saving_path)
    if not saving_path.is_dir():
        saving_path.mkdir(parents=True)

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
    else:
        # getting gauge infos
        with load_ds(observed_data_path) as observed_data_in:
            gauge_ids = observed_data_in["id"]
            x = observed_data_in["geo_x"]
            y = observed_data_in["geo_y"]
            facc = observed_data_in["area"]
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
            logger.info(f"There are {len(gauge_ids.data)} gauges in total.")
            observed_variable = (
                get_single_data_var(observed_data_in)
                if observed_variable is None
                else observed_variable
            )
            logger.info(observed_variable)
            # prepare for later resampling
            with load_ds(model_data_path, file_name=model_file_name) as sim_data_in:
                logger.info("Resampling to coarser calendar if needed...")
                sim_rs, obs_rs = resample_to_coarser_calendar(
                    sim_data_in, observed_data_in
                )

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
                obs_rs = obs_rs.isel(id=unique_pos)

            logger.info("Cropping data...")
            if date_slice is not None and date_slice != slice(None, None):
                logger.info(
                    f"Selecting data from {date_slice.start} to {date_slice.stop}..."
                )
            obs_discharge_data = obs_rs[observed_variable].sel(time=date_slice)
            if slicing_condition is not None:
                logger.info("Applying spatial slicing to sim data...")
                sim_data_cropped = sim_rs.sel(
                    {
                        get_coord_key(sim_rs, lat=True): slice(lat_max, lat_min),
                        get_coord_key(sim_rs, lon=True): slice(lon_min, lon_max),
                    }
                ).sel(time=date_slice)
            else:
                sim_data_cropped = sim_rs.sel(time=date_slice)

            if direct_comparison:
                logger.info(
                    "Selecting overlapping time period for direct comparison..."
                )
                overlapping_time_slice = get_overlapping_time_slice(sim_rs, obs_rs)
                logger.info(
                    f"Overlapping time is from {overlapping_time_slice.start} "
                    f"to {overlapping_time_slice.stop}"
                )
                obs_discharge_data = obs_discharge_data.sel(time=overlapping_time_slice)
                sim_data_cropped = sim_data_cropped.sel(time=overlapping_time_slice)

    # Drop gauges without any observed data in the selected period
    obs_discharge_data = obs_discharge_data.sel(id=gauge_ids.data)
    valid_obs = obs_discharge_data.notnull().any(dim="time")
    valid_ids = obs_discharge_data["id"].where(valid_obs, drop=True).values
    if evaluation_gauges is not None:
        eval_gauge_ids = pd.read_csv(evaluation_gauges, header=None).iloc[:, 0].values
        valid_ids = np.intersect1d(valid_ids, eval_gauge_ids)
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
    if not obs_output_file.is_file() or overwrite:
        logger.info("preparing obs data...")
        # observed_data = observed_data.rename({observed_variable: "discharge"})
        obs_discharge_data = obs_discharge_data.reindex(id=gauge_ids.data)
        facc_da = xr.DataArray(
            facc, name="facc", dims=["id"], coords={"id": gauge_ids.data}
        )
        observed_data = xr.Dataset({"facc": facc_da, "discharge": obs_discharge_data})

        logger.info(f"Saving obs data to {obs_output_file}...")
        write_xarray_to_file(observed_data, obs_output_file)

    if not sim_output_file.is_file() or overwrite:
        logger.info("preparing sim data...")
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
        elif scc_gauges_file is not None:
            x_new, y_new = [], []
            with load_ds(scc_gauges_file) as ds:
                logger.info("get the gauge coordinates from scc gauges file")
                x_new = ds.lon.values
                y_new = ds.lat.values
                gauge_ids_with_values = ds.station.values
            valid_id_set = set(np.asarray(gauge_ids.values))
            keep_mask = np.isin(gauge_ids_with_values, list(valid_id_set))
            if not np.any(keep_mask):
                msg = "No gauges with observed data found in scc gauges file."
                with ErrorLogger(logger):
                    raise ValueError(msg)
            x_new = np.asarray(x_new)[keep_mask]
            y_new = np.asarray(y_new)[keep_mask]
            gauge_ids_with_values = np.asarray(gauge_ids_with_values)[keep_mask]
        elif mrm_restart_file is not None:
            with load_ds(mrm_restart_file) as ds:
                logger.info(
                    "get the gauge coordinates by matching coordinates and flow accumulation"
                )
                out = Parallel(n_jobs=n_jobs, backend="loky")(
                    delayed(get_gauge_coords)(
                        ds,
                        facc=facc_i,
                        lonlat=[x_i, y_i],
                        cell_diff=1,
                        max_cell_diff=3,
                        diff_percent=10,
                        id=id_i,
                    )
                    for x_i, y_i, facc_i, id_i in zip(
                        x.values, y.values, facc.values, gauge_ids.values
                    )
                )
                x_new, y_new, facc_new, gauge_ids_with_values = [], [], [], []
                for i, (xn, yn, fan) in enumerate(out):
                    if xn is None or yn is None or fan is None:
                        continue
                    x_new.append(xn)
                    y_new.append(yn)
                    facc_new.append(fan)
                    gauge_ids_with_values.append(gauge_ids.data[i])
        else:
            error_msg = "Neither mrm restart file or scc gauges file are provided. Gauge location can not be determined"
            with ErrorLogger(logger):
                raise ValueError(error_msg)

        if len(x_new) == 0:
            msg = "There are no gauges that could be found."
            with ErrorLogger(logger):
                raise ValueError(msg)
        logger.info(f"There are {len(x_new)} gauges")
        logger.info("creating sim dataset")
        if discharge_nc_files:
            # simulation_discharge already prepared
            pass
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
            logger.info(f"Extracting discharge variable '{sim_variable}' from nodes...")
            sim, matched_x, matched_y, gauge_ids_with_values = (
                get_sim_data_for_gauges_from_nodes(
                    sim_data_cropped,
                    sim_variable,
                    x_new,
                    y_new,
                    gauge_ids_with_values,
                    resolution=resolution,
                )
            )
            x_new = matched_x
            y_new = matched_y
            try:
                facc_new = facc.sel(id=gauge_ids_with_values).values
            except Exception:
                facc_new = np.full(len(gauge_ids_with_values), np.nan)
        elif mrm_restart_file is not None:
            sim_variable = (
                get_single_data_var(sim_data_cropped)
                if sim_variable is None
                else sim_variable
            )
            sim = Parallel(n_jobs=n_jobs, backend="loky")(
                delayed(get_sim_data_for_one_gauge)(
                    id=id,
                    index=i,
                    sim_data=sim_data_cropped[sim_variable],
                    yarr=y_new,
                    xarr=x_new,
                    resolution=resolution,
                    lat_key="lat",
                    lon_key="lon",
                )
                for i, id in enumerate(gauge_ids_with_values)
            )
        if not discharge_nc_files:
            if isinstance(sim, list):
                simulation_discharge = xr.concat(sim, dim="id")
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
                "Dropping %d gauges missing in simulation output.",
                int((~keep_mask).sum()),
            )
        gauge_ids_with_values = ids_arr[keep_mask]
        x_new = np.asarray(x_new)[keep_mask]
        y_new = np.asarray(y_new)[keep_mask]
        facc_new = np.asarray(facc_new)[keep_mask]
        simulation_discharge = simulation_discharge.sel(id=gauge_ids_with_values)
        if "lat" in simulation_discharge.dims or "lat" in simulation_discharge.coords:
            simulation_discharge = simulation_discharge.drop_dims(["lat"])
        if "lon" in simulation_discharge.dims or "lon" in simulation_discharge.coords:
            simulation_discharge = simulation_discharge.drop_dims(["lon"])
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
        logger.info(f"Saving sim data to {sim_output_file}...")
        write_xarray_to_file(sim_data, sim_output_file)
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


@log_arguments()
def evaludate_discharge_data(  # noqa: PLR0913
    model_data_path,
    observed_data_path,
    mrm_restart_file=None,
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
    save_hydrograph=True
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
            mrm_restart_file=mrm_restart_file,
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
                save=save_hydrograph
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


def plot_map(
    results_df,
    output_path,
    variables=None,
    lon_col="x",
    lat_col="y",
    cmap="viridis",
    point_size=18,
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
    # Coerce lon/lat to numeric (handles object values like array(nan))
    def _as_float(val):
        if val is None:
            return np.nan
        if isinstance(val, (np.floating, float, int)):
            return float(val)
        try:
            arr = np.asarray(val)
            if arr.ndim == 0:
                return float(arr)
            if arr.size == 0:
                return np.nan
            if arr.size == 1:
                return float(arr.ravel()[0])
        except Exception:
            return np.nan
        return np.nan

    for col in (lon_col, lat_col):
        if col in df.columns:
            df[col] = df[col].map(_as_float)
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna(subset=[lon_col, lat_col])
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
            edgecolor="black",
            linewidth=0.2,
            transform=ccrs.PlateCarree(),
        )
        cb = plt.colorbar(sc, ax=ax, orientation="vertical", shrink=0.8, extend=extend)
        cb.set_label(var)
        ax.set_title(f"{var} by gauge")
        fig.tight_layout()
        fig.savefig(output_path / f"map_{var}.png", dpi=dpi)
        plt.close(fig)


@log_errors()
def plot_cdf(df, output_path, boostrap_iterations=None):  # noqa: PLR0915
    """Create CDF plots for alpha, beta, and gamma.

    The plots are generated for different subselections
    (by catchment, bootstrap-mean, or all results).
    """
    # --- 1) Read your CSV ---
    # Adjust 'mydata.csv' to your actual file path
    # df = pd.read_csv('/work/kelbling/ecflow_work/gloria_hourly_t2k/output/gloria_0p05deg/discharge_validation/results.csv', index_col=0)
    # output_path = Path('/work/luedke/tmp/test_discharge_plots')
    # if not output_path.is_dir():
    #     output_path.mkdir()
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
    df_all = df.copy()
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

    # --- 2) Check number of unique IDs ---
    unique_ids = df["id"].unique()
    logger.info(f"Creating a cdf plot with {len(unique_ids)} stations")

    logger.info("All values")
    for var in variables:
        fig, ax = plt.subplots(figsize=(6, 4))
        # Extract all values of `var` for the given ID
        da = xr.DataArray(df[var], dims=["index"], name=var).dropna(
            how="all", dim="index"
        )
        da_sorted = da.sortby(da)  # sort by the data values
        n = da_sorted.sizes["index"]

        # Create an array of ranks: [1, 2, ..., n]
        ranks = xr.DataArray(np.arange(1, n + 1), dims=["index"])

        # Compute the fraction => empirical CDF
        cdf = ranks / n
        med = np.nanmedian(da_sorted.values)
        ax.plot(
            da_sorted,
            cdf,
            marker=".",
            linestyle="none",
            color='darkblue',
            markersize=4,
        )
        ax.axvline(med, color='red', linestyle="dotted", linewidth=1, label=f"median = {med:.3f}")
        title = f"CDF of {var} for {len(unique_ids)} stations"
        if boostrap_iterations is not None and boostrap_iterations > 0:
            title += f" and {boostrap_iterations} bootstrap iterations"
        ax.set_title(title)
        ax.set_xlabel(var)
        ax.set_ylabel("CDF")
        ax.legend()
        ax.set_xlim(np.nanmin([da_sorted.min(), 0]), np.nanmax([da_sorted.max(), 1]))
        if var in ["kge", "nse"]:
            ax.set_xlim(-0.5, 1.0)
            ax.set_ylim(0.0, 1.01)
        else:
            ax.set_ylim(0, 1.05)
        fig.tight_layout()
        fig.savefig(output_path / f"cdf_{var}_all_stations.png", dpi=450)
        plt.close()

    # --- 3) CDF by WMO region (only if more than one region present) ---
    def _wmo_region_from_id(val):
        if pd.isna(val):
            return np.nan
        try:
            s = str(int(val))
        except Exception:
            s = str(val)
        s = s.lstrip("0")
        if not s or not s[0].isdigit():
            return np.nan
        return int(s[0])

    df_region = df.copy()
    df_region["wmo_region"] = df_region["id"].apply(_wmo_region_from_id)
    region_ids = np.sort(df_region["wmo_region"].dropna().unique())
    if len(region_ids) < 2:
        return 
    for var in variables:
        fig, ax = plt.subplots(figsize=(6, 4))
        # Extract all values of `var` for the given ID
        for i, region_id in enumerate(region_ids):
            df_var = df_region.dropna(subset=[var, "wmo_region"]).copy()
            da = xr.DataArray(df_var, dims=["index"], name=var).dropna(
                how="all", dim="index"
            )
            da_sorted = da.sortby(da)  # sort by the data values
            n = da_sorted.sizes["index"]

            # Create an array of ranks: [1, 2, ..., n]
            ranks = xr.DataArray(np.arange(1, n + 1), dims=["index"])

            # Compute the fraction => empirical CDF
            cdf = ranks / n
            plt.plot(da_sorted, cdf, marker="+", linestyle="-", )  # step or line is typical
            med = np.nanmedian(da_sorted)
            region_name = regions.get(int(region_id), f"Region {region_id}")
            color = cb_colors[i % len(cb_colors)]
            ax.plot(
                    da_sorted,
                    cdf,
                    marker=".",
                    linestyle="none",
                    color=color,
                    markersize=4,
                )
            ax.axvline(med, color=color, linestyle="dotted", linewidth=1, label=f"{region_name} = {med:.3f}")

        title = f"CDF of {var} for {len(unique_ids)} stations"
        if boostrap_iterations is not None and boostrap_iterations > 0:
            title += f" and {boostrap_iterations} bootstrap iterations"
        ax.set_xlabel(var)
        ax.set_ylabel("CDF")
        ax.grid(True, alpha=0.3)
        ax.legend(title="WMO Region median", ncols=min(len(region_ids), 6), fontsize=8)
        ax.set_title(f"CDF of {var} by WMO region")
        ax.set_xlim(np.nanmin([df_var[var].min(), 0]), np.nanmax([df_var[var].max(), 1]))
        if var in ["kge", "nse"]:
            ax.set_xlim(-0.5, 1.0)
            ax.set_ylim(0.0, 1.01)
        else:
            ax.set_ylim(0, 1.05)
        fig.tight_layout()
        fig.savefig(output_path / f"cdf_{var}_by_wmo_region.png", dpi=450)
        plt.close(fig)
