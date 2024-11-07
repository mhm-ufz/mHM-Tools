from mhm_tools.post.GRDC_validation import evaludate_grdc_data


def run():
    gauge_info_path = "/data/cats/data/DestinE/forcings_IFS_NEMO_gen_1_produciton_data/mhm_results_validation/validation_data/gauge_info_selected.nc"
    observed_data_path = "/data/cats/data/DestinE/forcings_IFS_NEMO_gen_1_produciton_data/mhm_results_validation/validation_data/GRDC_mean_daily_ulysses_1981-01-01-2019-12-31.nc"
    working_dir = f"/gpfs1/data/cats/data/DestinE/forcings_IFS_NEMO_gen_1_produciton_data/IFS_NEMO_mHM_Results/mRM/historic/daily_means"
    model_data = f"{working_dir}/daily_fluxes.nc" # MRM results
    # logger.info(sim_data.keys()) #Qrouted
    evaludate_grdc_data(
        model_data, observed_data_path, gauge_info_path, save_path=None, n_jobs=1
    )   
