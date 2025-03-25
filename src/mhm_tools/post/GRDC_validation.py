"""Compare simulated with observed discharge either directly or using a bootstraping approach."""

import itertools
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import xarray as xr
from joblib import Parallel, delayed

from mhm_tools.common.file_handler import get_xarray_ds_from_file, write_xarray_to_file
from mhm_tools.common.logger import log_arguments, log_errors
from mhm_tools.common.xarray_utils import get_coord_key
from mhm_tools.post.hydrograph import gen_hydrograph_by_data_sets
from mhm_tools.post.seasonality_grid_validation import (
    get_clim_from_ds,
    spearman_correlation,
)

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
    """Generate a list of result dictionaries containing time, id, discharge and year."""
    # Check if all values in the array are NaN
    discharge = data.values
    are_all_nan = np.all(np.isnan(discharge))
    time = data.time.values
    year = data.time.dt.year.values

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
    """Read in observed data for one gauge."""
    # id, index = id.values, index=id['index'].values
    if observed_data_by_id.size == 0:
        logger.info(f"No data found for ID: {id}. Skipping...\n")
        return None
    return observed_data_by_id.where(~np.isnan(observed_data_by_id), drop=True)


def get_sim_data_for_one_gauge(id, index, sim_data, yarr, xarr, resolution):
    """Read out simulation data for one gauge."""
    if xarr[index] is None and yarr[index] is None:
        return None
    # This will ensure that x & y values always match lat and lon in mRM dataset
    try:
        x = np.round(xarr[index], 5)
        y = np.round(yarr[index], 5)
    except KeyError:
        logger.error(f"index {index} not found")
        return None
    sim_data_loc = sim_data.sel(lat=y - resolution, lon=x, method="nearest")
    return sim_data_loc.expand_dims(dim={"id": [id]})


def get_gauge_coords(
    ds, facc, lonlat=None, xy=None, cell_diff=1, max_cell_diff=3, diff_percent=10
):
    """
    Find correct gauge location.

    Takes mrm_restart file, an approximate lat lon value and and then finds the cell with the value closest to a given flow accumulation.

    return lon,lat, flow accumulation
    """
    if lonlat is not None:
        # print('lonlat:', lonlat)
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
        min_index = np.unravel_index(np.argmin(abs_diff.values), ds_cut.L11_fAcc.shape)
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
        # print(ds_cut.L11_fAcc.data)
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
            f"No similar flow acc found. Increasing search radius to {cell_diff+1} cells in each direction."
        )
        return get_gauge_coords(
            ds,
            facc,
            lonlat=[lon, lat],
            cell_diff=cell_diff + 1,
            max_cell_diff=3,
            diff_percent=10,
        )
    logger.warning("No similar flow accumulation found nearby.")
    logger.debug("None, None, None")
    return None, None, None


def Q_data_to_xarray(
    model_data_path,
    observed_data_path,
    mrm_restart_file,
    sim_variable,
    observed_variable,
    model_keyword,
    saving_path=None,
    lon_min=None,
    lon_max=None,
    lat_min=None,
    lat_max=None,
    resolution=0.1,
    n_jobs=1,
    date_slice=None,
    overwrite=False,
):
    """
    Get observed and model Q data and save it as CSV files to be opened later.

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
    if sim_output_file.is_file() and not overwrite:
        logger.info("reading sim data from file...")
        sim_data = get_xarray_ds_from_file(sim_output_file)
        sim_data = sim_data.sel(time=date_slice)
    if obs_output_file.is_file() and not overwrite:
        logger.info("reading obs data from file...")
        observed_data = get_xarray_ds_from_file(obs_output_file)
        observed_data = observed_data.sel(time=date_slice)
    if obs_output_file.is_file() and sim_output_file.is_file() and not overwrite:
        return observed_data, sim_data
    # creating saving path
    saving_path = Path(saving_path)
    if not saving_path.is_dir():
        saving_path.mkdir(parents=True)

    # getting gauge infos
    with xr.open_dataset(observed_data_path) as gauge_info:
        gauge_ids = gauge_info["id"]
        x = gauge_info["geo_x"]
        y = gauge_info["geo_y"]
        facc = gauge_info["area"]
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
    logger.info(f"There are {len(gauge_ids.values)} gauges")

    if not obs_output_file.is_file() or overwrite:
        with xr.open_dataset(observed_data_path) as observed_data_in:
            obs_discharge_data = observed_data_in[observed_variable]
            obs_discharge_data = obs_discharge_data.sel(time=date_slice)
            # observed_data = observed_data.rename({observed_variable: "discharge"})
            obs_discharge_data = obs_discharge_data.sel(id=gauge_ids.values)
            obs_discharge_data = obs_discharge_data.reindex(id=gauge_ids.values)
            facc_da = xr.DataArray(
                facc, name="facc", dims=["id"], coords={"id": gauge_ids.values}
            )
            observed_data = xr.Dataset(
                {"facc": facc_da, "discharge": obs_discharge_data}
            )

            logger.info(f"Saving obs data to {obs_output_file}...")
            write_xarray_to_file(observed_data, obs_output_file)

    if not sim_output_file.is_file() or overwrite:
        with xr.open_dataset(mrm_restart_file) as ds:
            # get the gauge coordinates by matching coordinates and flow accumulation
            out = Parallel(n_jobs=n_jobs, backend="loky")(
                delayed(get_gauge_coords)(
                    ds,
                    facc=facc_i,
                    lonlat=[x_i, y_i],
                    cell_diff=1,
                    max_cell_diff=3,
                    diff_percent=10,
                )
                for x_i, y_i, facc_i in zip(x.values, y.values, facc.values)
            )
            x_new, y_new, facc_new, gauge_ids_with_values = [], [], [], []
            for i, (xn, yn, fan) in enumerate(out):
                if xn is None or yn is None or fan is None:
                    continue
                x_new.append(xn)
                y_new.append(yn)
                facc_new.append(fan)
                gauge_ids_with_values.append(gauge_ids.values[i])
        logger.info(f"There are {len(x_new)} gauges")
        logger.info("creating sim dataset")
        with xr.open_dataset(model_data_path) as sim_data_in:
            if slicing_condition is not None:
                sim_data_cropped = sim_data_in.sel(
                    {
                        get_coord_key(sim_data_in, lat=True): slice(lat_max, lat_min),
                        get_coord_key(sim_data_in, lon=True): slice(lon_min, lon_max),
                    }
                ).sel(time=date_slice)
            else:
                sim_data_cropped = sim_data_in.sel(time=date_slice)
            sim = Parallel(n_jobs=n_jobs, backend="loky")(
                delayed(get_sim_data_for_one_gauge)(
                    id=id,
                    index=i,
                    sim_data=sim_data_cropped[sim_variable],
                    yarr=y_new,
                    xarr=x_new,
                    resolution=resolution
                )
                for i, id in enumerate(gauge_ids_with_values)
            )
        simulation_discharge = xr.concat(sim, dim="id").drop_vars(["lat", "lon"])
        facc_ids = xr.DataArray(
            data=np.array(facc_new), dims=["id"], coords={"id": gauge_ids_with_values}
        )

        # 4) Build a new Dataset from this 2D DataArray
        sim_data = xr.Dataset({"discharge": simulation_discharge, "facc": facc_ids})
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
        or not sim_da_sel.isnull().all()
        or not obs_da_sel.isnull().all()
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
def evaludate_grdc_data(  # noqa: PLR0913
    model_data_path,
    observed_data_path,
    mrm_restart_file="/work/kelbling/ecflow_work/new_gloria_historical/output/gloria_0p05deg/mrm_restart_file/mRM_restart_001.nc",
    output_path=None,
    n_jobs=1,
    sim_variable="Qrouted",
    observed_variable="runoff_mean_mm",
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
):
    """Compare simulated with observed discharge either directly or using a bootstraping approach."""
    output_path = Path(output_path)
    observed_ds, model_ds = Q_data_to_xarray(
        model_data_path=model_data_path,
        observed_data_path=observed_data_path,
        mrm_restart_file=mrm_restart_file,
        observed_variable=observed_variable,
        sim_variable=sim_variable,
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
    )
    model_da = model_ds["discharge"]
    observed_da = observed_ds["discharge"]
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
        total_years_sim = np.unique(
            model_da.dropna(dim="time", how="all").time.dt.year.data
        )
        total_years_obs = np.unique(
            observed_da.dropna(dim="time", how="all").time.dt.year.data
        )
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
            for index, id in itertools.product(range(n_boostrap_selections), ids_sim)
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
        )
        for id in observed_ds.id.values
        if id in model_ds.id.values
    )
    results_df = pd.DataFrame(results_direct) if not results else pd.DataFrame(results)
    results_df = pd.DataFrame(results)
    results_df.to_csv(output_path / "results.csv")
    if results:
        plot_cdf(results_df, output_path, boostrap_iterations=n_boostrap_selections)
    else:
        plot_cdf(results_df, output_path)


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


@log_errors()
def plot_cdf(df, output_path, boostrap_iterations=None):
    """Create cdf plots for alpha, beat and gamma for different subselections (by catchment, boostrap-mean or all results)."""
    # --- 1) Read your CSV ---
    # Adjust 'mydata.csv' to your actual file path
    # df = pd.read_csv('/work/kelbling/ecflow_work/gloria_hourly_t2k/output/gloria_0p05deg/discharge_validation/results.csv', index_col=0)
    # output_path = Path('/work/luedke/tmp/test_discharge_plots')
    # if not output_path.is_dir():
    #     output_path.mkdir()
    # The variables to plot
    df = df.dropna(subset=["alpha", "beta", "gamma"], how="any")
    logger.info(df.head())
    variables = ["alpha", "beta", "gamma"]

    # --- 2) Check number of unique IDs ---
    unique_ids = df["id"].unique()
    n_ids = len(unique_ids)
    logger.info(f"Found {n_ids} unique IDs.")

    # --- 3 & 4) Branch based on the number of unique IDs ---
    # if n_ids < 9 or plot_all:
    #     logger.info("Single ids")
    #     # Plot a separate CDF for each ID, for each variable
    #     for var in variables:
    #         plt.figure(figsize=(6, 4))
    #         for uid in unique_ids:
    #             # Extract all values of `var` for the given ID
    #             subdata = df.loc[df["id"] == uid, var].sort_values()
    #             if len(subdata) == 0:
    #                 continue  # no data for this ID

    #             # Compute the empirical CDF
    #             cdfvals = np.arange(1, len(subdata) + 1) / float(len(subdata))

    #             # Plot
    #             plt.plot(subdata, cdfvals, label=f"id = {int(uid)}")

    #         plt.title(f"CDF of {var} (Separate lines per ID)")
    #         plt.xlabel(var)
    #         plt.ylabel("CDF")
    #         plt.legend()
    #         plt.tight_layout()
    #         plt.savefig(output_path / f"cdf_{var}_by_id.png", dpi=150)
    #         plt.close()

    # if n_ids < 10 or plot_all:
    logger.info("All values")
    for var in variables:
        plt.figure(figsize=(6, 4))
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
        # subdata = df[var].sort_values()
        # if len(subdata) == 0:
        # continue  # no data for this ID
        # logger.info(float(len(subdata)))
        logger.info(n)
        logger.info(da_sorted.values)
        logger.info(cdf.values)
        plt.plot(da_sorted, cdf, marker="+", linestyle="-")  # step or line is typical
        # Compute the empirical CDF
        # cdfvals = np.arange(1, len(subdata) + 1) / float(len(subdata))
        # Plot
        # plt.scatter(subdata, cdfvals, s=0.5, color='blue')
        # plt.plot(subdata, cdfvals, linewidth=0.3, color='blue')
        title = f"CDF of {var} for {len(unique_ids)} stations"
        if boostrap_iterations is not None:
            title += f" and {boostrap_iterations} bootstrap iterations"
        plt.title(title)
        plt.xlabel(var)
        plt.ylabel("CDF")
        plt.legend()
        plt.xlim(min(da_sorted.min(), 0), max(da_sorted.max(), 1))
        plt.ylim(0, 1.05)
        plt.tight_layout()
        plt.savefig(output_path / f"cdf_{var}_all_stations.png", dpi=450)
        plt.close()

    # if n_ids > 9 or plot_all:
    #     # Many IDs => plot the distribution of mean values by ID, for each variable
    #     # 1) Compute average (mean) across all rows belonging to each ID
    #     means_by_id = df.groupby("id")[variables].mean()

    #     for var in variables:
    #         # This is now one mean value per ID
    #         data = means_by_id[var].sort_values()
    #         cdfvals = np.arange(1, len(data) + 1) / float(len(data))

    #         plt.figure(figsize=(6, 4))
    #         plt.plot(data, cdfvals, marker="o")
    #         plt.title(f"CDF of mean {var} across {n_ids} IDs")
    #         plt.xlabel(f"mean {var}")
    #         plt.ylabel("CDF")
    #         plt.tight_layout()
    #         plt.savefig(output_path / f"cdf_{var}_mean_across_ids.png", dpi=150)
    #         plt.close()

    logger.info("Done! Check the saved PNG files for your CDF plots.")
