"""
Calculate the modified SPAtial EFficiency metric.

Based on:
Andreas Karpasitis, Panos Hadjinicolaou, and George Zittis
A new efficiency metric for the spatial evaluation and inter-comparison of climate and geoscientific model output
Geoscientific Model Development 19, https://doi.org/10.5194/gmd-19-345-2026, 2026

Implementation based on:
Karpasitis, A.: Code for the MSPAEF metric, Zenodo [code], https://doi.org/10.5281/zenodo.15094921, 2025. a
"""

import numpy as np

from mhm_tools.common.metrics.spaef import filter_nan


def MSPAEF(s, o):
    """Calculate MSPAEF and its normalized error components."""
    s, o = filter_nan(s, o)

    q75, q25 = np.percentile(o, [75, 25])
    iqr = q75 - q25

    nrmse = np.sqrt(np.mean((s - o) ** 2)) / iqr
    sigma = np.std(s) / np.std(o)
    sigma_error = np.sqrt((sigma**2 - 1) ** 2 + (1 / sigma**2 - 1) ** 2)
    mean_bias = np.abs(np.mean(s) - np.mean(o)) / iqr
    rho = np.corrcoef(s, o)[0, 1]

    mspaef_error = np.sqrt(
        nrmse**2 + sigma_error**2 + mean_bias**2 + (1 - rho) ** 2
    ) / np.sqrt(4)
    mspaef = 1 - mspaef_error

    return mspaef, nrmse, sigma, sigma_error, mean_bias, rho
