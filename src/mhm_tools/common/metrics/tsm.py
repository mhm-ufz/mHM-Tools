"""Calculate the temporal-spatial metric used by mhm-tools."""

import numpy as np
import pandas as pd
from scipy.stats import spearmanr


def filter_nan(s, o):
    """Remove rows containing NaNs from paired arrays."""
    data = np.transpose(np.array([s.flatten(), o.flatten()]))
    data = data[~np.isnan(data).any(1)]
    return data[:, 0], data[:, 1]


def objective_functions(s, o, metrics=None, param=""):
    """Calculate requested objective metrics for paired arrays."""
    if metrics is None:
        metrics = ["pearson", "bias", "variance"]
    result = {}
    s, o = filter_nan(s, o)

    param = param + "-" if param != "" else param
    if "pearson" in metrics:
        result[param + "gamma"] = np.corrcoef(s, o)[1, 0]
    elif "spearman" in metrics:
        result[param + "gamma"] = spearmanr(s, o)[0]
    if "variance" in metrics:
        result[param + "alpha"] = np.nanstd(s) / np.nanstd(o)

    if "bias" in metrics:
        result[param + "beta"] = np.nanmean(s) / np.nanmean(o)
    return result


def norm_deviation(data):
    """Calculate normalized deviation from spatial mean at that point in time."""
    return data - np.nanmean(data, axis=(1, 2), keepdims=True) / np.nanmean(
        data, axis=(1, 2), keepdims=True
    )


def nothing(data):
    """Return data unchanged."""
    return data


def nan_area_mean(data):
    """Calculate area mean for each point in time."""
    return np.nanmean(data, axis=(1, 2))


def desesonalized_nan_area_mean(data, time_index=None):
    """Create a one-year daily climatology from a multi-year gridded time series.

    Parameters
    ----------
    data : np.ndarray
        Array with shape (time, y, x).
    time_index : array-like, optional
        Time stamps matching the first dimension of ``data``. If provided, values
        are grouped by calendar day and returned as a 366-day climatology.
        Feb 29 is interpolated as the mean of Feb 28 and Mar 1.
    """
    area_mean = nan_area_mean(data)

    if time_index is not None:
        time_index = pd.to_datetime(time_index)
        if len(time_index) != len(area_mean):
            msg = (
                "Length mismatch between time_index and data time dimension: "
                f"{len(time_index)} != {len(area_mean)}"
            )
            raise ValueError(msg)
        df = pd.DataFrame(
            {"value": area_mean, "month_day": pd.Index(time_index).strftime("%m-%d")}
        )
        climatology = df.groupby("month_day")["value"].mean()

        feb_28 = climatology.get("02-28", np.nan)
        mar_01 = climatology.get("03-01", np.nan)
        climatology.loc["02-29"] = np.nanmean([feb_28, mar_01])

        clim_dates = pd.date_range("2000-01-01", "2000-12-31", freq="D")
        clim_month_day = clim_dates.strftime("%m-%d")
        return np.array(
            [climatology.get(month_day, np.nan) for month_day in clim_month_day],
            dtype=float,
        )

    if len(area_mean) % 365 == 0:
        days_per_year = 365
    elif len(area_mean) % 366 == 0:
        days_per_year = 366
    else:
        msg = (
            "Cannot infer daily climatology without time_index. "
            "Provide time_index or use a time dimension divisible by 365 or 366."
        )
        raise ValueError(msg)

    reshaped = area_mean.reshape(-1, days_per_year)
    daily_climatology = np.nanmean(reshaped, axis=0)
    if days_per_year == 366:
        daily_climatology[59] = np.nanmean(
            [daily_climatology[58], daily_climatology[60]]
        )
    return daily_climatology


def create_dic_of_objective_functions(arr_s, arr_o, metrics, func=nothing, param=""):
    """Calculate objective functions after applying an optional transform."""
    return objective_functions(func(arr_s), func(arr_o), metrics=metrics, param=param)


def calculate_tsm_for_gridded_data(map1, map2, ds1_name, ds2_name, eval_params=None):
    """Calculate TSM components for two gridded datasets."""
    if eval_params is None:
        eval_params = {
            "general": {"metrics": ["bias"], "func": nothing},
            "temporal": {
                "metrics": ["spearman", "variance"],
                "func": nan_area_mean,
            },
            "spatial": {
                "metrics": ["spearman", "variance"],
                "func": norm_deviation,
            },
        }
    evaluation_results_dict = {"name": ds1_name + "-" + ds2_name}
    for eval_param, eval_param_dict in eval_params.items():
        evaluation_results_dict.update(
            create_dic_of_objective_functions(
                map1,
                map2,
                metrics=eval_param_dict["metrics"],
                func=eval_param_dict["func"],
                param=eval_param,
            )
        )
    m1s_beta = (1 - evaluation_results_dict["general-beta"]) ** 2
    m1s_spatial_alpha = (1 - evaluation_results_dict["spatial-alpha"]) ** 2
    m1s_spatial_gamma = (1 - evaluation_results_dict["spatial-gamma"]) ** 2
    m1s_temporal_alpha = (1 - evaluation_results_dict["temporal-alpha"]) ** 2
    m1s_temporal_gamma = (1 - evaluation_results_dict["temporal-gamma"]) ** 2

    evaluation_results_dict["comb"] = 1 - np.sqrt(
        (
            m1s_beta
            + m1s_spatial_alpha
            + m1s_spatial_gamma
            + m1s_temporal_alpha
            + m1s_temporal_gamma
        )
        * 3
        / 5
    )
    return evaluation_results_dict


calculate_objectives_for_gridded_data = calculate_tsm_for_gridded_data
