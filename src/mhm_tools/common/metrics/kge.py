"""Kling-Gupta efficiency and its components."""

import logging

import numpy as np
import xarray as xr

logger = logging.getLogger(__name__)

MIN_VALID_PAIRS = 2


def calculate_kling_gupta_efficiency(simulated, observed, min_valid_pairs=None):
    """Calculate the Kling-Gupta efficiency and its three components.

    Only pairs where both series are finite are used, because the correlation
    is not NaN aware. A constant series leaves the correlation undefined, so
    all four values are NaN in that case.

    Args:
        simulated: Simulated values, array like of any shape.
        observed: Observed values, array like of the same shape.
        min_valid_pairs: Minimum number of finite pairs needed for a result,
            by default 2.

    Returns
    -------
        Tuple of (kge, alpha, beta, gamma), all NaN when the input does not
        allow a result. ``alpha`` is the ratio of the standard deviations,
        ``beta`` the ratio of the means and ``gamma`` the correlation.
    """
    min_valid_pairs = MIN_VALID_PAIRS if min_valid_pairs is None else min_valid_pairs
    simulated = np.asarray(getattr(simulated, "values", simulated), dtype=float).ravel()
    observed = np.asarray(getattr(observed, "values", observed), dtype=float).ravel()
    if simulated.shape != observed.shape:
        msg = (
            f"Simulated and observed must have the same shape, got "
            f"{simulated.shape} and {observed.shape}."
        )
        raise ValueError(msg)

    valid = np.isfinite(simulated) & np.isfinite(observed)
    simulated = simulated[valid]
    observed = observed[valid]
    if simulated.size < max(min_valid_pairs, MIN_VALID_PAIRS):
        return np.nan, np.nan, np.nan, np.nan

    observed_std = np.std(observed)
    simulated_std = np.std(simulated)
    observed_mean = np.mean(observed)
    # a constant series has no correlation and no variability ratio
    if observed_std == 0 or simulated_std == 0 or observed_mean == 0:
        return np.nan, np.nan, np.nan, np.nan

    alpha = simulated_std / observed_std
    beta = np.mean(simulated) / observed_mean
    gamma = np.corrcoef(observed, simulated)[1, 0]
    kge = 1 - np.sqrt((gamma - 1) ** 2 + (alpha - 1) ** 2 + (beta - 1) ** 2)
    return float(kge), float(alpha), float(beta), float(gamma)


def calculate_kling_gupta_efficiency_per_cell(
    simulated, observed, min_valid_pairs=None
):
    """Calculate KGE and its components for every grid cell of two records.

    Holds the same definition as `calculate_kling_gupta_efficiency`, written as
    reductions over ``time`` so the whole grid is done in one pass and stays
    lazy. Keep both in step when the formula changes.

    Args:
        simulated: Simulated DataArray with a ``time`` dimension.
        observed: Observed DataArray on the same grid and time axis.
        min_valid_pairs: Fewest paired finite steps a cell needs, by default 2.

    Returns
    -------
        Dataset with ``kge``, ``alpha``, ``beta``, ``gamma`` and the per cell
        counts and reference moments (``valid_pairs``, ``observed_mean``,
        ``observed_std``) that make an unstable ``beta`` readable.
    """
    min_valid_pairs = MIN_VALID_PAIRS if min_valid_pairs is None else min_valid_pairs
    min_valid_pairs = max(min_valid_pairs, MIN_VALID_PAIRS)

    valid = simulated.notnull() & observed.notnull()
    sim = simulated.where(valid)
    obs = observed.where(valid)
    valid_pairs = valid.sum("time")

    simulated_mean = sim.mean("time")
    observed_mean = obs.mean("time")
    # ddof=0 on both sides, so it cancels in alpha and in the correlation
    simulated_std = sim.std("time")
    observed_std = obs.std("time")

    covariance = ((sim - simulated_mean) * (obs - observed_mean)).mean("time")
    gamma = covariance / (simulated_std * observed_std)
    alpha = simulated_std / observed_std
    beta = simulated_mean / observed_mean
    kge = 1 - ((gamma - 1) ** 2 + (alpha - 1) ** 2 + (beta - 1) ** 2) ** 0.5

    # a constant or zero mean reference leaves the components undefined
    usable = (
        (valid_pairs >= min_valid_pairs)
        & (observed_std > 0)
        & (simulated_std > 0)
        & (observed_mean != 0)
    )
    return xr.Dataset(
        {
            "kge": kge.where(usable),
            "alpha": alpha.where(usable),
            "beta": beta.where(usable),
            "gamma": gamma.where(usable),
            "valid_pairs": valid_pairs,
            "observed_mean": observed_mean,
            "observed_std": observed_std,
        }
    )
