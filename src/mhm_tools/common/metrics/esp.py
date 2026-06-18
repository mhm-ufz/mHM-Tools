"""
Calculate the Error-SPAtial Pattern metric.

Based on
Dembélé, M., Hrachowitz, M., Savenije, H. H., Mariéthoz, G., and
Schaefli, B.: Improving the predictive skill of a distributed hy-
drological model by calibration on spatial patterns with multi-
ple satellite datasets, Water Resour. Res., 56, e2019WR026085,
https://doi.org/10.1029/2019WR026085, 2020b.
"""

import numpy as np
from scipy.stats import spearmanr, variation, zscore

from mhm_tools.common.metrics.spaef import filter_nan


def ESP(s, o):
    """Calculate ESP and its rank, variability, and location components."""
    s, o = filter_nan(s, o)

    rs = spearmanr(s, o)[0]
    gamma = variation(s) / variation(o)

    observed = zscore(o)
    simulated = zscore(s)
    alpha = np.sqrt(np.mean((simulated - observed) ** 2))
    esp = 1 - np.sqrt((rs - 1) ** 2 + (gamma - 1) ** 2 + alpha**2)

    return esp, rs, gamma, alpha
