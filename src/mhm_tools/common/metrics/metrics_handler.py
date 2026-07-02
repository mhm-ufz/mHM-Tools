"""
Dispatch metric calculations and write metric CSV files.

Authors
-------
- Simon Lüdke
- Carla Peter
"""

import logging

import numpy as np
import pandas as pd

from mhm_tools.common.metrics.esp import ESP
from mhm_tools.common.metrics.mspaef import MSPAEF
from mhm_tools.common.metrics.spaef import SPAEF
from mhm_tools.common.metrics.tsm import calculate_tsm_for_gridded_data
from mhm_tools.common.metrics.waspaef import WASPAEF
from mhm_tools.common.utils import pretty_print_df

logger = logging.getLogger(__name__)
RESULT_METRIC_TSM = "TSM"
RESULT_METRIC_SPAEF = "SPAEF"
RESULT_METRIC_ESP = "ESP"
RESULT_METRIC_WASPAEF = "WASPAEF"
RESULT_METRIC_MSPAEF = "MSPAEF"
ACCEPTED_RESULT_METRICS = (
    RESULT_METRIC_TSM,
    RESULT_METRIC_SPAEF,
    RESULT_METRIC_ESP,
    RESULT_METRIC_WASPAEF,
    RESULT_METRIC_MSPAEF,
)


def normalize_results_metric(metric):
    """Normalize and validate a result CSV metric name."""
    if metric is None:
        return RESULT_METRIC_TSM
    normalized = str(metric).strip().upper()
    if normalized == "ALL":
        return ACCEPTED_RESULT_METRICS
    if normalized not in ACCEPTED_RESULT_METRICS:
        accepted = ", ".join(ACCEPTED_RESULT_METRICS)
        msg = f"Unsupported result metric {metric!r}. Use one of: {accepted}."
        raise ValueError(msg)
    return normalized


def calculate_metric_per_timestep_and_average(map1, map2, func, *args, **kwargs):
    """Apply `func` to matching 2D timesteps of two 3D arrays,then average the per-timestep results.

    Parameters
    ----------
    func: callable
        Function applied to matching 2D timestep slices from both arrays.
    map1: numpy.ndarray
        observed data 3D array with shape ``(time, lat, lon)``.
    map2: numpy.ndarray
        simulated data 3D array with shape ``(time, lat, lon)``.
    *args
        Additional positional arguments passed to ``func``.
    **kwargs
        Additional keyword arguments passed to ``func``.

    Returns
    -------
    numpy.ndarray
        Mean metric values over all timesteps.
    """
    results_per_timestep = np.asarray(
        [
            func(simulated, observed, *args, **kwargs)
            for simulated, observed in zip(map1, map2)
        ]
    )

    return np.nanmean(results_per_timestep, axis=0)


def calculate_spaef_for_gridded_data(map1, map2, ds1_name, ds2_name):
    """Calculate SPAEF metrics for two gridded datasets."""
    spaef, alpha, beta, gamma = calculate_metric_per_timestep_and_average(
        map1, map2, SPAEF
    )
    return {
        "name": ds1_name + "-" + ds2_name,
        "avg_spaef": spaef,
        "avg_alpha": alpha,
        "avg_beta": beta,
        "avg_gamma": gamma,
    }


def calculate_esp_for_gridded_data(map1, map2, ds1_name, ds2_name):
    """Calculate ESP metrics for two gridded datasets."""
    esp, rs, gamma, alpha = calculate_metric_per_timestep_and_average(map1, map2, ESP)
    return {
        "name": ds1_name + "-" + ds2_name,
        "avg_esp": esp,
        "avg_rs": rs,
        "avg_gamma": gamma,
        "avg_alpha": alpha,
    }


def calculate_waspaef_for_gridded_data(map1, map2, ds1_name, ds2_name):
    """Calculate WASPAEF metrics for two gridded datasets."""
    waspaef, rho, sigma, wd = calculate_metric_per_timestep_and_average(
        map1, map2, WASPAEF
    )
    return {
        "name": ds1_name + "-" + ds2_name,
        "avg_waspaef": waspaef,
        "avg_rho": rho,
        "avg_sigma": sigma,
        "avg_wd": wd,
    }


def calculate_mspaef_for_gridded_data(map1, map2, ds1_name, ds2_name):
    """Calculate MSPAEF metrics for two gridded datasets."""
    mspaef, nrmse, sigma, sigma_error, mean_bias, rho = (
        calculate_metric_per_timestep_and_average(map1, map2, MSPAEF)
    )
    return {
        "name": ds1_name + "-" + ds2_name,
        "avg_mspaef": mspaef,
        "avg_nrmse": nrmse,
        "avg_sigma": sigma,
        "avg_sigma_error": sigma_error,
        "avg_mean_bias": mean_bias,
        "avg_rho": rho,
    }


def calculate_results_metric(map1, map2, ds1_name, ds2_name, metric=RESULT_METRIC_TSM):
    """Calculate the requested gridded result metric."""
    metric = normalize_results_metric(metric)
    if metric == RESULT_METRIC_TSM:
        return calculate_tsm_for_gridded_data(
            map1=map1, map2=map2, ds1_name=ds1_name, ds2_name=ds2_name
        )
    if metric == RESULT_METRIC_SPAEF:
        return calculate_spaef_for_gridded_data(
            map1=map1, map2=map2, ds1_name=ds1_name, ds2_name=ds2_name
        )
    if metric == RESULT_METRIC_ESP:
        return calculate_esp_for_gridded_data(
            map1=map1, map2=map2, ds1_name=ds1_name, ds2_name=ds2_name
        )
    if metric == RESULT_METRIC_WASPAEF:
        return calculate_waspaef_for_gridded_data(
            map1=map1, map2=map2, ds1_name=ds1_name, ds2_name=ds2_name
        )
    if metric == RESULT_METRIC_MSPAEF:
        return calculate_mspaef_for_gridded_data(
            map1=map1, map2=map2, ds1_name=ds1_name, ds2_name=ds2_name
        )
    msg = f"Unsupported result metric {metric!r}."
    raise ValueError(msg)


def create_csv_from_dict(results_dict: dict, out_path):
    """Create a CSV file from the provided dictionary."""
    df = pd.DataFrame(results_dict, index=[0])
    logger.info(f"Written metrics to {out_path}")
    df.to_csv(out_path)


def create_results_csv(
    map1,
    map2,
    ds1_name,
    ds2_name,
    out_dir,
    out_name="",
    metric="all",
):
    """Calculate the selected metric and create a CSV file."""
    norm_metric = normalize_results_metric(metric)
    if isinstance(norm_metric, tuple):
        logger.info("Create csv for all metrics")
        for nm in norm_metric:
            create_results_csv(
                map1=map1,
                map2=map2,
                ds1_name=ds1_name,
                ds2_name=ds2_name,
                out_dir=out_dir,
                out_name=out_name,
                metric=nm,
            )
        return
    logger.info(f"Calculating metrics for {metric}")
    results_dict = calculate_results_metric(
        map1=map1,
        map2=map2,
        ds1_name=ds1_name,
        ds2_name=ds2_name,
        metric=norm_metric,
    )
    logger.info(f"Spatial metrics: {results_dict}")
    metric_name = norm_metric.lower()
    file_name = f"{out_name}_{metric_name}.csv" if out_name else f"{metric_name}.csv"
    create_csv_from_dict(results_dict=results_dict, out_path=out_dir / file_name)
    df = pd.DataFrame(results_dict, index=[0])
    pretty_print_df(df, title=norm_metric)
