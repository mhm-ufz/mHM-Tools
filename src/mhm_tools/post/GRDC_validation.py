import random
from pathlib import Path

import matplotlib.pyplot as plt
from mhm_tools.common.logger import ErrorLogger, log_arguments
from mhm_tools.common.xarray_utils import get_coord_key
import numpy as np
import pandas as pd
import seaborn as sns
import xarray as xr
from joblib import Parallel, delayed
import seaborn as sns

from mhm_tools.post.seasonality_grid_validation import climatology, spearman_correlation
import logging

logger = logging.getLogger(__name__)

# make sure that the gauge location is correct basin extractor ...
# make sample size the same length as simulation dataset, pick periods and use that for uncertainty estimate
# how to deal with climate variablity:
#   - trend correction?
#   - bootstrap years around event

# currently unused
def calculate_statistics(sim_data_by_id, obs_data_by_id):
    # replace the following with bootstrap alorythem What takes long seems to be
    logger.info("get mean values")
    mean_sim = sim_data_by_id.mean(skipna=True)
    logger.info("sim done")
    mean_obs = obs_data_by_id.mean(skipna=True)
    logger.info("obs done")

    logger.info("create climatologies")
    clim_sim = climatology(sim_data_by_id)
    logger.info("sim done")
    clim_obs = climatology(obs_data_by_id)
    logger.info("obs done")

    logger.info(f"sim: {sim_data_by_id.shape}")
    logger.info(f" obs: {obs_data_by_id.shape}")

    logger.info("get standard deviation")
    std_obs = obs_data_by_id.std(skipna=True)
    std_sim = sim_data_by_id.std(skipna=True)

    logger.info("calc statistics")
    alpha = float((mean_sim / mean_obs).values)
    beta = float((std_sim / std_obs).values)
    spearman, spera_var = spearman_correlation(clim_sim, clim_obs)
    logger.info(f"{type(id)}, {type(alpha)}, {type(beta)}, {type(spearman)}")
    logger.info(f"{id}, {alpha}, {beta}, {spearman}")
    logger.info(f"sim_mean = {mean_sim.values}; obs_mean = {mean_obs.values}")
    return {"id": id, "alpha": alpha, "beta": beta, "spearman": spearman}

# currently unused
def bootstrap_years(sim_data_by_id, obs_data_by_id, n_selections, n_years):
    total_years_sim = list(set(sim_data_by_id.time.dt.year.values))
    total_years_obs = list(set(sim_data_by_id.time.dt.year.values))
    results = []
    for index in range(n_selections):
        random.seed(index)
        years_sim = random.choices(total_years_sim, k=n_years)
        years_obs = random.choices(total_years_obs, k=n_years)
        ds_sim_sel = sim_data_by_id.where(
            sim_data_by_id.time.dt.year.isin(years_sim), drop=True
        )
        ds_obs_sel = obs_data_by_id.where(
            obs_data_by_id.time.dt.year.isin(years_obs), drop=True
        )
        res = calculate_statistics(ds_sim_sel, ds_obs_sel)
        res["index"] = index
        results.append(res)
    return results

# currently unused
def evaluate_one_gauge(index, id, observed_data, sim_data, x, y, n_years, n_selections):
    logger.info(f"working on gauge number: {index}...\n")
    obs_data_by_id = observed_data.sel(id=id)
    if obs_data_by_id.size == 0:
        logger.info(f"No data found for ID: {id}. Skipping...\n")
        return None

    are_all_nan = obs_data_by_id.isnull().all()
    if are_all_nan:
        logger.info("all values are nan")
        return None
    logger.info(f"values found at gauge: {id}\n")
    # This will ensure that x & y values always match lat and lon in sim dataset
    x = np.round(x.isel(index).values, 2)
    y = np.round(y.isel(index).values, 2)
    logger.info("get sim data point")
    # x,y, facc = 
    sim_data_by_id = sim_data.sel(lat=y, lon=x, method="nearest")
        # logger.info(sim_data_by_id)
    if n_years is not None and n_selections is not None:
        return bootstrap_years(sim_data_by_id, obs_data_by_id, n_selections, n_years)
    return calculate_statistics(sim_data_by_id, obs_data_by_id)

def flatten_list(nested_list):
    flat_list = []
    for item in nested_list:
        if item is None:
            continue
        elif isinstance(item, list):  # If the item is a list, extend the flat list
            flat_list.extend(flatten_list(item))
        else:  # If it's not a list, append it directly
            flat_list.append(item)
    return flat_list

def gen_list_of_result_dicts(data, id, datatype='', facc=None):
    # Check if all values in the array are NaN
    discharge = data.values
    are_all_nan = np.all(np.isnan(discharge))
    time = data.time.values
    year = data.time.dt.year.values

    if not are_all_nan:
        logger.info(f" values found at gauge number: {id}...\n")
        res = []
        for t, val, yr in zip(time, discharge, year):
            res_dict = {'time': t, 'id': int(id), 'Q': val, 'year': yr}
            if facc is not None: 
                res_dict['facc'] = facc
            res.append(res_dict)
        return res
    logger.warning('All datapoints for {datatype} at gauge {id} are missing.')

def get_grdc_for_one_gauge(id, observed_data_by_id, facc=None):
    # id, index = id.values, index=id['index'].values
    id = id
    if observed_data_by_id.size == 0:
        logger.info(f"No data found for ID: {id}. Skipping...\n")
        return
    observed_data_by_id = observed_data_by_id.where(~np.isnan(observed_data_by_id), drop=True)
    return gen_list_of_result_dicts(observed_data_by_id, id=id, datatype='observation', facc=facc)

def get_sim_data_for_one_gauge(id, index, sim_data, yarr, xarr, resolution, facc=None):
    id = id
    if xarr[index] is None and yarr[index] is None:
        return
    # This will ensure that x & y values always match lat and lon in mRM dataset
    try:    
        x = np.round(xarr[index],3)
        y = np.round(yarr[index],3)
    except KeyError as e: 
        logger.error(f'index {index} not found')
        return
    mrm_data_by_id = sim_data.sel(lat=y - resolution, lon=x, method="nearest")
    mrm_data_by_id = mrm_data_by_id.where(~np.isnan(mrm_data_by_id), drop=True)

    return gen_list_of_result_dicts(mrm_data_by_id, id=id, datatype='simulation', facc=facc)

def get_gauge_coords(ds, facc, lonlat=None, xy=None,  cell_diff=1, max_cell_diff = 3, diff_percent=10):
    """
    Find correct gauge location.
    
    Takes mrm_restart file, an approximate lat lon value and and then finds the cell with the value closest to a given flow accumulation.

    return lon,lat, flow accumulation
    """
    if lonlat is not None:
        # print('lonlat:', lonlat)
        lon = lonlat[0]
        lat = lonlat[1]
        lon_col = int((lon-ds.xllcorner_L0)/ds.cellsize_L1)
        lat_row = int(ds.nrows_L1 - (lat-(ds.yllcorner_L0))/ds.cellsize_L1)
    else:
        lon_col=xy[0]
        lat_row=xy[1]
    if cell_diff > 0:
        ds_cut = ds.sel(nrows0=slice(lat_row-cell_diff, lat_row+cell_diff),nrows1=slice(lat_row-cell_diff, lat_row+cell_diff),nrows11=slice(lat_row-cell_diff, lat_row+cell_diff), ncols0=slice(lon_col-cell_diff, lon_col+cell_diff),ncols1=slice(lon_col-cell_diff, lon_col+cell_diff),ncols11=slice(lon_col-cell_diff, lon_col+cell_diff))
        abs_diff = np.abs(ds_cut.L11_fAcc - facc)
        min_index = np.unravel_index(np.argmin(abs_diff.values), ds_cut.L11_fAcc.shape)
        if lonlat:
            lon_x = ds_cut.L1_domain_lon.data[min_index[0],0]
            lat_y = ds_cut.L1_domain_lat.data[0,min_index[1]]
        # else:
        #     lon_x, lat_y = ds_cut.nrows0.values[1], ds_cut.nrows0.values[1] ## Not yet working
        found_facc = ds_cut.L11_fAcc.data[min_index[0],min_index[1]]
    else:
        ds_cut = ds.sel(nrows0=lat_row, nrows1=lat_row, nrows11=lat_row, ncols0=lon_col, ncols1=lon_col, ncols11=lon_col)
        # print(ds_cut.L11_fAcc.data)
        if lonlat:
            lon_x = ds_cut.L1_domain_lon.data
            lat_y = ds_cut.L1_domain_lat.data
        else:
            lon_x = lon_col
            lat_y = lat_row
        found_facc = ds_cut.L11_fAcc.data
    if abs(found_facc- facc) < diff_percent/100 * facc:
        logger.info(f"{lon_x}, {lat_y}, {found_facc}, {facc}, {cell_diff}")
        return lon_x, lat_y, found_facc
    elif cell_diff < max_cell_diff:
        logger.debug(f'No similar flow acc found. Increasing search radius to {cell_diff+1} cells in each direction.')
        return get_gauge_coords(ds, facc, lonlat=[lon, lat], cell_diff=cell_diff+1, max_cell_diff=3, diff_percent=10)
    else: 
        logger.debug(f'No similar flow accumulation found nearby.')
        logger.debug("None, None, None")
        return None, None, None




def Q_data_to_CSV(
    model_data_path, observed_data_path, mrm_restart_file, sim_variable, observed_variable, gauge_info_path, model_keyword, saving_path=None, lon_min=None, lon_max=None, lat_min=None, lat_max=None, resolution=0.1, n_jobs=1,
):
    """
    This is a function that gets observed and model Q data and
    saves it as CSV files to be open later

    This function is not part of the workflow, but a pre processing tool
    that for comodity has been stored here. This was done bc mrm_data_by_id.values
    was taking to much time to execute

    Args:
    - mrm_data (xarray.DataArray): The mRM simulated data as an xarray DataArray.
    - observed_data (xarray.DataArray): The observed data as an xarray DataArray.
    - gauge_info_path (str): The file path to the gauge information dataset.
    - model_keyword (str): dir to be added to the path were files will be stored.
    - output_path (str): optional, saving path

    Note:
    The gauge information dataset should contain the following variables:
    - "id1": Gauge IDs
    - "new_x": X-coordinates of gauges
    - "new_y": Y-coordinates of gauges

    """
    sim_output_file = Path(f"{saving_path}/{model_keyword}_dataframe.csv")
    obs_output_file = Path(f"{saving_path}/GRDC_dataframe.csv")
    if sim_output_file.is_file():
        logger.info('reading sim data from file...')
        sim_dataframe = pd.read_csv(sim_output_file) 
    if obs_output_file.is_file():
        logger.info('reading obs data from file...')
        obs_dataframe = pd.read_csv(obs_output_file) 
    if obs_output_file.is_file() and sim_output_file.is_file():
        return obs_dataframe, sim_dataframe
        # creating saving path
    saving_path = Path(saving_path)
    if not saving_path.is_dir():
        saving_path.mkdir(parents=True)
    
    
    # getting gauge infos
    with  xr.open_dataset(gauge_info_path) as gauge_info:
        gauge_ids = gauge_info["id1"]
        x = gauge_info["gauge_x"]
        y = gauge_info["gauge_y"]
        facc = gauge_info['gauge_size']
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
    # logger.info(f'xarr {x}')
    # logger.info(f'yarr {y}')
    
    # IMPORTANT: The id's in GRDC observed_data have the same index as in gauge_info
    if not obs_output_file.is_file():
        observed_data = xr.open_dataset(observed_data_path)
        observed_data = observed_data.sel()
        observed_data = observed_data[observed_variable]
        obs = Parallel(n_jobs=n_jobs, backend="loky")(
                delayed(get_grdc_for_one_gauge)(id=id, observed_data_by_id=observed_data.sel(id=id), facc=facc_i)
                for id, facc_i in zip(gauge_ids.values, facc.values)
            )
        obs = flatten_list(obs)
        obs_dataframe = pd.DataFrame(obs)
        logger.info(f'Saving obs data to {obs_output_file}...')
        obs_dataframe.to_csv(obs_output_file)
    if not sim_output_file.is_file():
        with xr.open_dataset(mrm_restart_file) as ds:
            # for x_i, y_i, facc_i in zip(x.values, y.values, facc.values):
            #     print(get_gauge_coords(ds, facc=facc_i, lonlat=[x_i, y_i], cell_diff=1, max_cell_diff=3, diff_percent=10))
            out = Parallel(n_jobs=n_jobs, backend="loky")(
                    delayed(get_gauge_coords)(ds, facc=facc_i, lonlat=[x_i, y_i], cell_diff=1, max_cell_diff=3, diff_percent=10)
                    for x_i, y_i, facc_i in zip(x.values,y.values,facc.values)
            )
            x_new, y_new, facc_new = [], [], []
            for xn,yn,fan in out: 
                if xn is None or yn is None or fan is None:
                    continue
                x_new.append(xn)
                y_new.append(yn)
                facc_new.append(fan)
        logger.info(f"There are {x_new} gauges")
        logger.info('creating sim dataframe')
        sim_data = xr.open_dataset(model_data_path)
        if slicing_condition is not None:
            sim_data = sim_data.sel({get_coord_key(sim_data, lat=True): slice(lat_max, lat_min), get_coord_key(sim_data, lon=True): slice(lon_min, lon_max)})
        sim_data= sim_data[sim_variable]
        sim = Parallel(n_jobs=n_jobs, backend="loky")(
                delayed(get_sim_data_for_one_gauge)(id=id, index=i, sim_data=sim_data, yarr=y_new, xarr=x_new, resolution=resolution, facc=facc_i)
                for i, (id, facc_i) in enumerate(zip(gauge_ids.values, facc_new))
            )
        sim = flatten_list(sim)
        sim_dataframe = pd.DataFrame(sim)
        logger.info(f'Saving sim data to {sim_output_file}...')
        sim_dataframe.to_csv(sim_output_file)
    return obs_dataframe, sim_dataframe

def calc_clim_from_pandas(df):
    # Group by month and calculate the mean for the relevant columns
    climatologies = df.groupby('month').mean()

    # Rename the index for clarity (optional)
    climatologies.index.name = 'month'
    climatologies.reset_index(inplace=True)

    return climatologies

def add_month_column(df):
    if 'time' in df.columns:
        if not np.issubdtype(df['time'].dtype, np.datetime64):
            df['time'] = pd.to_datetime(df['time'], errors='coerce')
        if df['time'].isna().any():
            df = df.dropna(subset=['time'])
        df['month'] = df['time'].dt.month
    else:
        with ErrorLogger(logger):
            raise KeyError("The 'time' column is missing from the DataFrame.")
    return df

@log_arguments()
def evaludate_grdc_data(
    model_data_path,
    observed_data_path,
    gauge_info_path,
    mrm_restart_file='/work/kelbling/ecflow_work/new_gloria_historical/output/gloria_0p05deg/mrm_restart_file/mRM_restart_001.nc',
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
    n_boostrap_selections = 0,
):
    output_path = Path(output_path)
    observed_df, model_df = Q_data_to_CSV(model_data_path=model_data_path, observed_data_path=observed_data_path, mrm_restart_file=mrm_restart_file, observed_variable=observed_variable, sim_variable=sim_variable, gauge_info_path=gauge_info_path,model_keyword="mrm", saving_path=output_path, lon_min=lon_min, lon_max=lon_max, lat_min=lat_min, lat_max=lat_max, resolution=resolution, n_jobs=n_jobs)

    if direct_comparison: 
        return # use jeissons function directly
    logger.info('start boostraping')
    if n_bootstrap_years is not None and n_boostrap_selections is not None and n_boostrap_selections > 0:
        results = []
        total_years_sim = model_df['year'].unique()
        total_years_obs = observed_df['year'].unique()
        # for index in range(n_boostrap_selections):
        for index in range(n_boostrap_selections):
            logger.info(f"index: {index}")
            years_sim = random.choices(total_years_sim, k=n_bootstrap_years)
            years_obs = random.choices(total_years_obs, k=n_bootstrap_years)
            logger.debug(f"years_sim: {years_sim}")
            logger.debug(f"years_obs: {years_obs}")
            sim_df_sel = pd.concat([model_df[model_df['year'] == year] for year in years_sim])
            obs_df_sel = pd.concat([observed_df[observed_df['year'] == year] for year in years_obs])
            
            obs_df_sel = add_month_column(obs_df_sel)
            sim_df_sel = add_month_column(sim_df_sel)
            ids_sim = [int(id) for id in sim_df_sel['id'].unique()]
            ids_obs = [int(id) for id in obs_df_sel['id'].unique()]
            logger.debug(f"OBS ids: {ids_obs}")
            logger.debug(f"SIM ids: {ids_sim}")
            for id in ids_sim:
                if id not in ids_obs:
                    continue
                try:
                    sim_id = sim_df_sel[sim_df_sel['id']==id]
                    obs_id = obs_df_sel[obs_df_sel['id']==id]
                    clim_sim = calc_clim_from_pandas(sim_id)
                    clim_obs = calc_clim_from_pandas(obs_id)
                    alpha = sim_id['Q'].mean(skipna=True) / obs_id['Q'].mean(skipna=True)
                    beta = sim_id['Q'].std(skipna=True) / obs_id['Q'].std(skipna=True)
                    gamma = spearman_correlation(clim_sim['Q'], clim_obs['Q'])[0]
                    logger.debug(f'results for index {index} and gauge {id}: alpha={alpha:.3f}, beta={beta:.3f}, gamma={gamma:.3f}')
                    results.append({
                        "index": index,
                        "id": id,
                        "alpha": alpha,
                        "beta": beta,
                        "gamma": gamma,
                    })
                except Exception as e: 
                    logger.error(f'Error for index {index} and id {id} with error {e}')
    else: 
        msg = 'Direct comparison is not yet implemented'
        raise NotImplementedError(msg)
    results_df = pd.DataFrame(results)
    results_df.to_csv(output_path / 'results.csv')
    plot_cdf(results_df, output_path)
    
def plot_kdf(results_df, output_path):
    sns.kdeplot(
        data=results_df,     # The DataFrame with results
        x="alpha",           # The x-axis is the alpha value
        hue="id",        # Use column names for coloring
        palette="tab10",     # Set a color palette
        fill=True,           # Fill the KDE areas for better visualization
        alpha=0.6,           # Set transparency
        common_norm=False,   # Ensure each column has its own normalization
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
    plt.axvline(x=1, color='black', linestyle='--', linewidth=1)
    plt.xlim(-0.05, 2)
    plt.savefig(output_path / 'alpha.png')
    plt.close()

    sns.kdeplot(
        data=results_df,     # The DataFrame with results
        x="beta",           # The x-axis is the alpha value
        hue="id",        # Use column names for coloring
        palette="tab10",     # Set a color palette
        fill=True,           # Fill the KDE areas for better visualization
        alpha=0.6,           # Set transparency
        common_norm=False,   # Ensure each column has its own normalization
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
    plt.axvline(x=1, color='black', linestyle='--', linewidth=1)
    plt.xlim(-0.05, 2)
    plt.savefig(output_path / 'beta.png')
    plt.close()

    sns.kdeplot(
        data=results_df,     # The DataFrame with results
        x="gamma",           # The x-axis is the alpha value
        hue="id",        # Use column names for coloring
        palette="tab10",     # Set a color palette
        fill=True,           # Fill the KDE areas for better visualization
        alpha=0.6,           # Set transparency
        common_norm=False,   # Ensure each column has its own normalization
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
    plt.axvline(x=1, color='black', linestyle='--', linewidth=1)
    plt.xlim(-0.05, 2)
    plt.savefig(output_path / 'gamma.png')
    plt.close()


def plot_cdf(df, output_path, plot_all=True):
    # --- 1) Read your CSV ---
    # Adjust 'mydata.csv' to your actual file path
    # df = pd.read_csv('/work/kelbling/ecflow_work/gloria_hourly_t2k/output/gloria_0p05deg/discharge_validation/results.csv', index_col=0)
    # output_path = Path('/work/luedke/tmp/test_discharge_plots')
    # if not output_path.is_dir():
    #     output_path.mkdir()
    # The variables to plot
    variables = ['alpha', 'beta', 'gamma']

    # --- 2) Check number of unique IDs ---
    unique_ids = df['id'].unique()
    n_ids = len(unique_ids)
    logger.info(f"Found {n_ids} unique IDs.")

    # --- 3 & 4) Branch based on the number of unique IDs ---
    if n_ids < 9 or plot_all:
        logger.info("Single ids")
        # Plot a separate CDF for each ID, for each variable
        for var in variables:
            plt.figure(figsize=(6,4))
            for uid in unique_ids:
                # Extract all values of `var` for the given ID
                subdata = df.loc[df['id'] == uid, var].sort_values()
                if len(subdata) == 0:
                    continue  # no data for this ID

                # Compute the empirical CDF
                cdfvals = np.arange(1, len(subdata)+1) / float(len(subdata))

                # Plot
                plt.plot(subdata, cdfvals, label=f"id = {int(uid)}")

            plt.title(f"CDF of {var} (Separate lines per ID)")
            plt.xlabel(var)
            plt.ylabel('CDF')
            plt.legend()
            plt.tight_layout()
            plt.savefig(output_path/f"cdf_{var}_by_id.png", dpi=150)
            plt.close()

    if n_ids < 10 or plot_all:
        logger.info("All values")
        # Plot a separate CDF for each ID, for each variable
        for var in variables:
            plt.figure(figsize=(6,4))
            # Extract all values of `var` for the given ID
            subdata = df[var].sort_values()
            if len(subdata) == 0:
                continue  # no data for this ID

            # Compute the empirical CDF
            cdfvals = np.arange(1, len(subdata)+1) / float(len(subdata))

            # Plot
            plt.plot(subdata, cdfvals)

            plt.title(f"CDF of {var} (Separate lines per ID)")
            plt.xlabel(var)
            plt.ylabel('CDF')
            plt.legend()
            plt.tight_layout()
            plt.savefig(output_path/f"cdf_{var}_all.png", dpi=150)
            plt.close()


    if n_ids > 9 or plot_all:
        # Many IDs => plot the distribution of mean values by ID, for each variable
        # 1) Compute average (mean) across all rows belonging to each ID
        means_by_id = df.groupby('id')[variables].mean()

        for var in variables:
            # This is now one mean value per ID
            data = means_by_id[var].sort_values()
            cdfvals = np.arange(1, len(data)+1) / float(len(data))

            plt.figure(figsize=(6,4))
            plt.plot(data, cdfvals, marker='o')
            plt.title(f"CDF of mean {var} across {n_ids} IDs")
            plt.xlabel(f"mean {var}")
            plt.ylabel("CDF")
            plt.tight_layout()
            plt.savefig(output_path/f"cdf_{var}_mean_across_ids.png", dpi=150)
            plt.close()

    logger.info("Done! Check the saved PNG files for your CDF plots.")