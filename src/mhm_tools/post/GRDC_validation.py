import random
from pathlib import Path

import matplotlib.pyplot as plt
from mhm_tools.common.xarray_utils import get_coord_key
import numpy as np
import pandas as pd
import seaborn as sns
import xarray as xr
from joblib import Parallel, delayed
import seaborn as sns

from mhm_tools.common.logger import logger
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

def get_grdc_for_one_gauge(id, observed_data_by_id):
    # id, index = id.values, index=id['index'].values
    id = id
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
        logger.info(f"observation values found at gauge number: {id}...\n")
        res = []
        for t, val, yr in zip(time, np_observed, year):
            res.append({'time': t, 'id': id, 'Q': val, 'year': yr})
        return res
        # return {'time': observed_data_by_id.time, int(id): np_observed, 'year': observed_data_by_id.time.dt.year}

def get_sim_data_for_one_gauge(id, index, sim_data, yarr, xarr, resolution):
    id = id.values
    # index = index=id['index'].values
    # logger.info(f"values found at gauge number: {index}...\n")
    if xarr[index] is None and yarr[index] is None:
        return
    # This will ensure that x & y values always match lat and lon in mRM dataset
    try:    
        x = np.round(xarr[index],3)
        y = np.round(yarr[index],3)
        # x = np.round(xarr.sel(index=index).values, 2)
        # y = np.round(yarr.sel(index=index).values, 2)
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
        logger.warning(f'No similar flow acc found. Increasing search radius to {cell_diff+1} cells in each direction.')
        return get_gauge_coords(ds, facc, lonlat=[lon, lat], cell_diff=cell_diff+1, max_cell_diff=3, diff_percent=10)
    else: 
        logger.error(f'No similar flow accumulation found nearby.')
        logger.info("None, None, None")
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
    logger.info(f"There are {len(gauge_ids.values)} gauges {gauge_ids[0]['index']}")
    # logger.info(f'xarr {x}')
    # logger.info(f'yarr {y}')
    
    # IMPORTANT: The id's in GRDC observed_data have the same index as in gauge_info
    if not obs_output_file.is_file():
        observed_data = xr.open_dataset(observed_data_path)
        observed_data = observed_data.sel()
        observed_data = observed_data[observed_variable]
        obs = Parallel(n_jobs=n_jobs, backend="loky")(
                delayed(get_grdc_for_one_gauge)(id=id, observed_data_by_id=observed_data.sel(id=id))
                for id in gauge_ids
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
            x_new, y_new = [], []
            for xn,yn,fan in out: 
                x_new.append(xn)
                y_new.append(yn)
        logger.info('creating sim dataframe')
        sim_data = xr.open_dataset(model_data_path)
        if slicing_condition is not None:
            sim_data = sim_data.sel({get_coord_key(sim_data, lat=True): slice(lat_max, lat_min), get_coord_key(sim_data, lon=True): slice(lon_min, lon_max)})
        sim_data= sim_data[sim_variable]
        logger.info(gauge_ids)
        sim = Parallel(n_jobs=n_jobs, backend="loky")(
                delayed(get_sim_data_for_one_gauge)(id=id, index=i, sim_data=sim_data, yarr=y_new, xarr=x_new, resolution=resolution)
                for i, id in enumerate(gauge_ids)
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
        raise KeyError("The 'time' column is missing from the DataFrame.")
    return df

def evaludate_grdc_data(
    model_data_path,
    observed_data_path,
    gauge_info_path,
    mrm_restart_file='/work/kelbling/ecflow_work/new_gloria_historical/output/gloria_0p05deg/mrm_restart_file/mRM_restart_001.nc',
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
    observed_df, model_df = Q_data_to_CSV(model_data_path=model_data_path, observed_data_path=observed_data_path, mrm_restart_file=mrm_restart_file, observed_variable=observed_variable, sim_variable=sim_variable, gauge_info_path=gauge_info_path,model_keyword="mrm", saving_path=save_path, lon_min=lon_min, lon_max=lon_max, lat_min=lat_min, lat_max=lat_max, resolution=resolution, n_jobs=n_jobs)

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
            logger.debug(f"years_sim: {years_sim}")
            logger.debug(f"years_obs: {years_obs}")
            sim_df_sel = pd.concat([model_df[model_df['year'] == year] for year in years_sim])
            obs_df_sel = pd.concat([observed_df[observed_df['year'] == year] for year in years_obs])
            logger.info(index)
            logger.info(sim_df_sel.head())
            logger.info(obs_df_sel.head())
            
            obs_df_sel = add_month_column(obs_df_sel)
            sim_df_sel = add_month_column(sim_df_sel)
            logger.info(type(sim_df_sel))
            ids_sim = [int(id) for id in sim_df_sel['id'].unique()]
            ids_obs = [int(id) for id in obs_df_sel['id'].unique()]
            logger.debug(f"OBS: {ids_obs}")
            logger.debug(f"SIM: {ids_sim}")
            for id in ids_sim:
                if id not in ids_obs:
                    continue
                sim_id = sim_df_sel[sim_df_sel['id']==id]
                obs_id = obs_df_sel[obs_df_sel['id']==id]
                clim_sim = calc_clim_from_pandas(sim_id)
                clim_obs = calc_clim_from_pandas(obs_id)
                alpha = sim_id['Q'].mean(skipna=True) / obs_id['Q'].mean(skipna=True)
                beta = sim_id['Q'].std(skipna=True) / obs_id['Q'].std(skipna=True)
                logger.debug(f'Clim Shapes sim: {clim_sim['Q']}; obs: {clim_obs['Q']}')
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
    plt.axvline(x=1, color='black', linestyle='--', linewidth=1)
    plt.xlim(-0.05, 2)
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
    plt.axvline(x=1, color='black', linestyle='--', linewidth=1)
    plt.xlim(-0.05, 2)
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
    plt.axvline(x=1, color='black', linestyle='--', linewidth=1)
    plt.xlim(-0.05, 2)
    plt.savefig(save_path / 'gamma.png')
    plt.close()

