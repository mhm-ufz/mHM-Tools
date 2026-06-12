"""
Calculate the SPAtial EFficiency metric.

Based on:
Julian Koch, Mehmet Cüneyd Demirel, and Simon Stisen:
The SPAtial EFficiency metric (SPAEF): multiple-component evaluation of
spatial patterns for optimization of hydrological models,
Geoscientific Model Development 11, https://doi.org/10.5194/gmd-11-1873-2018, 2018
"""

import math

import numpy as np
from scipy.stats import variation, zscore


def filter_nan(s, o):
    """Remove paired NaN values before calculating SPAEF."""
    data = np.transpose(np.array([s.flatten(), o.flatten()]))
    data = data[~np.isnan(data).any(1)]
    return data[:, 0], data[:, 1]


def SPAEF(s, o):
    """Calculate SPAEF and its alpha, beta, and gamma components."""
    s, o = filter_nan(s, o)

    bins = int(np.around(math.sqrt(len(o)), 0))
    alpha = np.corrcoef(s, o)[0, 1]
    beta = variation(s) / variation(o)

    observed = zscore(o)
    simulated = zscore(s)
    observed_histogram, _ = np.histogram(observed, bins)
    simulated_histogram, _ = np.histogram(simulated, bins)
    observed_histogram = np.float64(observed_histogram)
    simulated_histogram = np.float64(simulated_histogram)

    minima = np.minimum(simulated_histogram, observed_histogram)
    gamma = np.sum(minima) / np.sum(observed_histogram)
    spaef = 1 - np.sqrt((alpha - 1) ** 2 + (beta - 1) ** 2 + (gamma - 1) ** 2)

    return spaef, alpha, beta, gamma
