from pathlib import Path
from joblib import Parallel, delayed
import seaborn as sns
import pandas as pd
import xarray as xr 
import numpy as np
import matplotlib.pyplot as plt

from mhm_tools.common.logger import logger, set_log_level
from mhm_tools.post.seasonality_grid_validation import climatology, get_clim_from_ds, get_std_from_ds, spearman_correlation

# make sure that the gauge location is correct basin extractor ...
# make sample size the same length as simulation dataset, pick periods and use that for uncertainty estimate
# how to deal with climate variablity: 
#   - trend correction?
#   - bootstrap years around event 


def evaluate_one_gauge(index, id, observed_data, sim_data, x, y, remove_seasonality=True):
    logger.info(f"working on gauge number: {index}...\n")
    observed_data_by_id = observed_data.sel(id=id)
    if observed_data_by_id.size == 0:
        logger.info(f"No data found for ID: {id}. Skipping...\n")
        return

    are_all_nan = observed_data_by_id.isnull().all()
    if are_all_nan:
        logger.info('all values are nan')
        return
    else:
        logger.info(f"values found at gauge: {id}\n")
        # This will ensure that x & y values always match lat and lon in sim dataset
        x = np.round(x.sel(index=index).values, 2)
        y = np.round(y.sel(index=index).values, 2) 
        logger.info('get sim data point')
        sim_data_by_id = sim_data.sel(lat=y - 0.1, lon=x, method="nearest")
        # logger.info(sim_data_by_id)


        # replace the following with bootstrap alorythem What takes long seems to be
        logger.info('get mean values')
        mean_sim = sim_data_by_id.mean(dim='time', skipna=True)
        logger.info('sim done')
        mean_obs = observed_data_by_id.mean(dim='time', skipna=True)
        logger.info('obs done')

        logger.info('create climatologies')
        clim_sim = climatology(sim_data_by_id)
        logger.info('sim done')
        clim_obs = climatology(observed_data_by_id)
        logger.info('obs done')

        logger.info('get standard deviation')
        std_obs = observed_data_by_id.std(skipna=True)
        std_sim = sim_data_by_id.std(skipna=True)

        logger.info('calc statistics')
        alpha = mean_sim / mean_obs
        beta = std_sim / std_obs
        spearman = spearman_correlation(clim_sim, clim_obs)
        logger.info(f"{type(id)}, {type(alpha)}, {type(beta)}, {type(spearman)}")
        logger.info(f"{id}, {alpha}, {beta}, {spearman}")
        return {'id': id, 'alpha':alpha, 'beta': beta, 'spearman': spearman}


def evaludate_grdc_data(
    model_data_path, observed_data_path, gauge_info_path, save_path=None, n_jobs=1, sim_variable='Qrouted', observed_variable='runoff_mean_mm',
    lon_min=-180, lon_max=180, lat_min=-56, lat_max=84
):  
    observed_data = xr.open_dataset(observed_data_path)
    # logger.info(observed_data.keys()) # runoff_mean_mm
    sim_data = xr.open_dataset(model_data_path)
    gauge_info = xr.open_dataset(gauge_info_path)
    sim_data, observed_data = sim_data[sim_variable], observed_data[observed_variable]
    # getting variables needed
    gauge_ids = gauge_info["id1"]
    x = gauge_info["new_x"]
    y = gauge_info["new_y"]
    if lon_min is not None and lon_max is not None and lat_min is not None and lat_max is not None:
        sliceing_condtion = (x>=lon_min) & (x<=lon_max) & (y>=lat_min) & (y<=lat_max)
        x = x.where(sliceing_condtion, drop=True)
        y = y.where(sliceing_condtion, drop=True)
        gauge_ids = gauge_ids.where(sliceing_condtion, drop=True)
    # Create an empty pandas dataframes
    model_dataframe, obs_dataframe = pd.DataFrame(), pd.DataFrame()

    # Initialize an empty DataFrame
    model_to_concat = []
    obs_to_concat = []

    # IMPORTANT: The id's in GRDC observed_data have the same index as in gauge_info
    logger.info(f"There are {len(gauge_ids.values)} gauges")
    if n_jobs == 1:
        results_per_id = []
        for index, id in enumerate(gauge_ids.values[2:8]):
            results_per_id.append(evaluate_one_gauge(index, id, observed_data, sim_data, x, y))
    else:
        results_per_id = Parallel(n_jobs=n_jobs, backend="loky")(
                            delayed(evaluate_one_gauge)(index, id, observed_data, sim_data, x, y)
                            for index, id in enumerate(gauge_ids.values[:])
        
    )
    results_df = pd.DataFrame(results_per_id)
    logger.info(results_df)
    results_df.to_csv('/work/luedke/grdc_results.csv')

    if save_path is None:
        saving_path = Path(
            "/work/luedke/"
        )
    else:
        saving_path = Path(save_path)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Plot CDF for each variable
    sns.ecdfplot(data=results_df, x="alpha", ax=axes[0])
    axes[0].set_title("CDF of Alpha")

    sns.ecdfplot(data=results_df, x="beta", ax=axes[1])
    axes[1].set_title("CDF of Beta")

    sns.ecdfplot(data=results_df, x="spearman", ax=axes[2])
    axes[2].set_title("CDF of Spearman")

    plt.tight_layout()
    logger.info(pd.DataFrame(results_per_id))
    plt.savefig(saving_path / 'grdc.png', dpi=1000)