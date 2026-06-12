"""Dispatch metric calculations and write metric CSV files."""

import logging

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


def calculate_spaef_for_gridded_data(map1, map2, ds1_name, ds2_name):
    """Calculate SPAEF metrics for two gridded datasets."""
    spaef, alpha, beta, gamma = SPAEF(map1, map2)
    return {
        "name": ds1_name + "-" + ds2_name,
        "spaef": spaef,
        "alpha": alpha,
        "beta": beta,
        "gamma": gamma,
    }


def calculate_esp_for_gridded_data(map1, map2, ds1_name, ds2_name):
    """Calculate ESP metrics for two gridded datasets."""
    esp, rs, gamma, alpha = ESP(map1, map2)
    return {
        "name": ds1_name + "-" + ds2_name,
        "esp": esp,
        "rs": rs,
        "gamma": gamma,
        "alpha": alpha,
    }


def calculate_waspaef_for_gridded_data(map1, map2, ds1_name, ds2_name):
    """Calculate WASPAEF metrics for two gridded datasets."""
    waspaef, rho, sigma, wd = WASPAEF(map1, map2)
    return {
        "name": ds1_name + "-" + ds2_name,
        "waspaef": waspaef,
        "rho": rho,
        "sigma": sigma,
        "wd": wd,
    }


def calculate_mspaef_for_gridded_data(map1, map2, ds1_name, ds2_name):
    """Calculate MSPAEF metrics for two gridded datasets."""
    mspaef, nrmse, sigma, sigma_error, mean_bias, rho = MSPAEF(map1, map2)
    return {
        "name": ds1_name + "-" + ds2_name,
        "mspaef": mspaef,
        "nrmse": nrmse,
        "sigma": sigma,
        "sigma_error": sigma_error,
        "mean_bias": mean_bias,
        "rho": rho,
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
