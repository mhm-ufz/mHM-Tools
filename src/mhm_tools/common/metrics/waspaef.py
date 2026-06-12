"""
Calculate the Wasserstein SPAtial EFficiency metric.

Values go from 0 to infinity and can be interpreted as a distance from reference dataset.

Based on
Gómez, M. J., Barboza, L. A., Hidalgo, H. G., and Alfaro, E. J.:
Comparison of indicators to evaluate the performance of climate models,
International Journal of Climatology, 44, 4907-4924, https://doi.org/10.1002/joc.8619, 2024.a, b, c

Implementation based on:
Karpasitis, A.: Code for the MSPAEF metric, Zenodo [code], https://doi.org/10.5281/zenodo.15094921, 2025.a
"""

import numpy as np

from mhm_tools.common.metrics.spaef import filter_nan


def WASPAEF(s, o):
    """Calculate WASPAEF and its correlation, spread, and distance components."""
    s, o = filter_nan(s, o)

    rho = np.corrcoef(s, o)[0, 1]
    sigma = np.std(s) / np.std(o)

    observed = np.sort(o)
    simulated = np.sort(s)
    # Original implementation changes are possible since flatten forces same length
    # for this case this implementation matches paper description
    #     data_min = min(np.min(simulated), np.min(observed))
    #     data_max = max(np.max(simulated), np.max(observed))
    #     n_bins = int(np.around(np.sqrt(len(o)), 0))
    #     # Calculate the bin edges, ensuring they are integers
    #     bin_edges = np.linspace(
    #         int(np.floor(data_min)), int(np.ceil(data_max)), n_bins + 1
    #     )  ### Bin edges for histogram using original data
    #     data_comp_pdf, _ = np.histogram(
    #         observed.flatten(), bins=bin_edges, density=False
    #     )  ### Histogram bins for original data of comparison dataset
    #     data_pdf, _ = np.histogram(
    #         simulated.flatten(), bins=bin_edges, density=False
    #     )  ### Histogram bins for original data of model dataset

    #     wd = wasserstein_distance(
    #         bin_edges[:-1], bin_edges[:-1], data_comp_pdf, data_pdf
    #     )  ##### Wasserstein distance of histograms of the original data
    wd = np.sqrt(np.mean((observed - simulated) ** 2))
    waspaef = np.sqrt((rho - 1) ** 2 + (sigma - 1) ** 2 + wd**2)

    return waspaef, rho, sigma, wd
