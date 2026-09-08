"""Root mean square error of two records.

Holds the per cell form used by the gridded evaluations, written as reductions
over ``time`` so a whole grid is scored in one pass and stays lazy.

Authors
-------
- Simon Lüdke
"""

import logging

logger = logging.getLogger(__name__)

MIN_VALID_PAIRS = 1


def calculate_root_mean_square_error_per_cell(
    simulated, observed, min_valid_pairs=None
):
    """Calculate the root mean square error for every grid cell of two records.

    A step missing in either record cannot be compared and is left out of the
    mean, so a cell is scored on the steps both records hold.

    Args:
        simulated: Simulated DataArray with a ``time`` dimension.
        observed: Observed DataArray on the same grid and time axis.
        min_valid_pairs: Fewest paired finite steps a cell needs, by default 1.

    Returns
    -------
        DataArray of the root mean square error in the unit of the two records,
        NaN in a cell holding fewer pairs than required.
    """
    min_valid_pairs = MIN_VALID_PAIRS if min_valid_pairs is None else min_valid_pairs
    min_valid_pairs = max(min_valid_pairs, MIN_VALID_PAIRS)

    valid = simulated.notnull() & observed.notnull()
    squared_error = ((simulated - observed) ** 2).where(valid)
    rmse = squared_error.mean("time") ** 0.5
    return rmse.where(valid.sum("time") >= min_valid_pairs)
