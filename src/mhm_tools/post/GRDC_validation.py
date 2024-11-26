import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import xarray as xr
from joblib import Parallel, delayed
import seaborn as sns

from mhm_tools.common.logger import log_arguments, logger, set_log_level
from mhm_tools.post.seasonality_grid_validation import climatology, spearman_correlation

# make sure that the gauge location is correct basin extractor ...
# make sample size the same length as simulation dataset, pick periods and use that for uncertainty estimate
# how to deal with climate variablity:
#   - trend correction?
#   - bootstrap years around event


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
    sim_data_by_id = sim_data.sel(lat=y - 0.1, lon=x, method="nearest")
        # logger.info(sim_data_by_id)
    if n_years is not None and n_selections is not None:
        return bootstrap_years(sim_data_by_id, obs_data_by_id, n_selections, n_years)
    return calculate_statistics(sim_data_by_id, obs_data_by_id)

def flatten_list(nested_list):
    flat_list = []
    for item in nested_list:
        if isinstance(item, list):  # If the item is a list, extend the flat list
            flat_list.extend(flatten_list(item))
        else:  # If it's not a list, append it directly
            flat_list.append(item)
    return flat_list

def get_grdc_for_one_gauge(id, observed_data_by_id, index):
    if observed_data_by_id.size == 0:
        logger.info(f"No data found for ID: {id}. Skipping...\n")
        return
    observed_data_by_id = observed_data_by_id.where(~np.isnan(observed_data_by_id), drop=True)
    # Check if all values in the array are NaN
    np_observed = np.array(observed_data_by_id.values)
    are_all_nan = np.all(np.isnan(np_observed))
    time = observed_data_by_id.time.values
    year = observed_data_by_id.time.dt.year.values

    if not are_all_nan:
        logger.info(f"observation values found at gauge number: {index}...\n")
        res = []
        for t, val, yr in zip(time, np_observed, year):
            res.append({'time': t, 'id': id, 'Q': val, 'year': yr})
        return res
        # return {'time': observed_data_by_id.time, int(id): np_observed, 'year': observed_data_by_id.time.dt.year}

def get_sim_data_for_one_gauge(id, sim_data, yarr, xarr, index, resolution):
    logger.info(f"values found at gauge number: {index}...\n")
    # This will ensure that x & y values always match lat and lon in mRM dataset
    try:
        x = np.round(xarr.sel(index=index).values, 2)
        y = np.round(yarr.sel(index=index).values, 2) 
    except KeyError as e: 
        logger.error(f'index {index} not found')
        return
    mrm_data_by_id = sim_data.sel(lat=y - resolution, lon=x, method="nearest")
    mrm_data_by_id = mrm_data_by_id.where(~np.isnan(mrm_data_by_id), drop=True)

    # Compute to get the NumPy array
    # np_model = np.array(mrm_data_by_id.values)
    # return {'time': mrm_data_by_id.time, int(id): np_model, 'year': mrm_data_by_id.time.dt.year}
    np_model = mrm_data_by_id.values
    time = mrm_data_by_id.time.values
    year = mrm_data_by_id.time.dt.year.values

    # Append results to the list as dictionaries
    res = []
    for t, val, yr in zip(time, np_model, year):
        res.append({'time': t, 'id': id, 'Q': val, 'year': yr})
    return res

def Q_data_to_CSV(
    model_data_path, observed_data_path, sim_variable, observed_variable, gauge_info_path, model_keyword, saving_path=None, lon_min=None, lon_max=None, lat_min=None, lat_max=None, resolution=0.1, n_jobs=1
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
    - save_path (str): optional, saving path

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
    gauge_info = xr.open_dataset(gauge_info_path)
    gauge_ids = gauge_info["id1"]
    x = gauge_info["new_x"]
    logger.debug(x)
    y = gauge_info["new_y"]
    if (
        lon_min is not None
        and lon_max is not None
        and lat_min is not None
        and lat_max is not None
    ):
        sliceing_condtion = (
            (x >= lon_min) & (x <= lon_max) & (y >= lat_min) & (y <= lat_max)
        )
        x = x.where(sliceing_condtion, drop=True)
        y = y.where(sliceing_condtion, drop=True)
        gauge_ids = gauge_ids.where(sliceing_condtion, drop=True)
    logger.info(f"There are {len(gauge_ids.values)} gauges")
    logger.info(f'xarr {x}')
    logger.info(f'yarr {y}')
    for index, id in enumerate(gauge_ids.values):
        logger.debug(f"index: {index} --- id: {id}")
        logger.debug(f"x: {x[index].values} --- y: {y[index].values}")

    # IMPORTANT: The id's in GRDC observed_data have the same index as in gauge_info
    if not obs_output_file.is_file():
        observed_data = xr.open_dataset(observed_data_path)
        observed_data = observed_data[observed_variable]
        obs = Parallel(n_jobs=n_jobs, backend="loky")(
                delayed(get_grdc_for_one_gauge)(id=id, index=index, observed_data_by_id=observed_data.sel(id=id))
                for index, id in enumerate(gauge_ids.values)
            )
        obs = flatten_list(obs)
        obs_dataframe = pd.DataFrame(obs)
        logger.info(f'Saving obs data to {obs_output_file}...')
        obs_dataframe.to_csv(obs_output_file)
    if not sim_output_file.is_file():
        logger.info('creating sim dataframe')
        sim_data = xr.open_dataset(model_data_path)
        sim_data= sim_data[sim_variable]
        sim = Parallel(n_jobs=n_jobs, backend="loky")(
                delayed(get_sim_data_for_one_gauge)(id=id.values, index=id['index'].values, sim_data=sim_data, yarr=y, xarr=x, resolution=resolution)
                for id in gauge_ids
            )
        sim = flatten_list(sim)
        logger.debug(sim)
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
        raise KeyError("The 'time' column is missing from the DataFrame.")
    return df

@log_arguments()
def evaludate_grdc_data(
    model_data_path,
    observed_data_path,
    gauge_info_path,
    save_path=None,
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
    save_path = Path(save_path)
    observed_df, model_df = Q_data_to_CSV(model_data_path=model_data_path, observed_data_path=observed_data_path, observed_variable=observed_variable, sim_variable=sim_variable, gauge_info_path=gauge_info_path,model_keyword="mrm", saving_path=save_path, lon_min=lon_min, lon_max=lon_max, lat_min=lat_min, lat_max=lat_max, resolution=resolution, n_jobs=n_jobs)

    if direct_comparison: 
        return # use jeissons function directly
    logger.info('start boostraping')
    if n_bootstrap_years is not None and n_boostrap_selections is not None and n_boostrap_selections > 0:
        results = []
        total_years_sim = model_df['year'].unique()
        total_years_obs = observed_df['year'].unique()
        for index in range(n_boostrap_selections):
            years_sim = random.choices(total_years_sim, k=n_bootstrap_years)
            years_obs = random.choices(total_years_obs, k=n_bootstrap_years)
            sim_df_sel = model_df[model_df['year'].isin(years_sim)].dropna()
            obs_df_sel = observed_df[observed_df['year'].isin(years_obs)].dropna()
            logger.info(index)
            logger.info(sim_df_sel.head())
            logger.info(obs_df_sel.head())
            
            obs_df_sel = add_month_column(obs_df_sel)
            sim_df_sel = add_month_column(sim_df_sel)
            ids = sim_df_sel['id'].unique()
            for id in ids:
                sim_id = sim_df_sel[sim_df_sel['id']==id]
                obs_id = obs_df_sel[obs_df_sel['id']==id]
                clim_sim = calc_clim_from_pandas(sim_id)
                clim_obs = calc_clim_from_pandas(obs_id)
                alpha = sim_id['Q'].mean(skipna=True) / obs_id['Q'].mean(skipna=True)
                beta = sim_id['Q'].std(skipna=True) / obs_id['Q'].std(skipna=True)
                gamma = spearman_correlation(clim_sim['Q'], clim_obs['Q'])[0]
                logger.info(f'alpha: {alpha}, beta: {beta}, gamma: {gamma}')
                results.append({
                    "index": index,
                    "id": id,
                    "alpha": alpha,
                    "beta": beta,
                    "gamma": gamma,
                })
    results_df = pd.DataFrame(results)
    results_df.to_csv(save_path / 'results.csv')
    logger.info(results_df.head())
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
    plt.savefig(save_path / 'alpha.png')
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
    plt.savefig(save_path / 'beta.png')
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
    plt.savefig(save_path / 'gamma.png')
    plt.close()

