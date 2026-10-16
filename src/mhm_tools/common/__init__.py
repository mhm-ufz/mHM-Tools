"""Common routines and constants.

Subpackages
===========

.. autosummary::
   :toctree:

   ~mhm_tools.common.metrics

Files
=====

.. autosummary::
   :toctree:

   ~mhm_tools.common.cli_utils
   ~mhm_tools.common.constants
   ~mhm_tools.common.crs_handler
   ~mhm_tools.common.esri_grid
   ~mhm_tools.common.file_handler
   ~mhm_tools.common.format_data
   ~mhm_tools.common.logger
   ~mhm_tools.common.netcdf
   ~mhm_tools.common.plotter
   ~mhm_tools.common.provenance
   ~mhm_tools.common.resolution_handler
   ~mhm_tools.common.time_utils
   ~mhm_tools.common.utils
   ~mhm_tools.common.xarray_utils
"""

import importlib

from . import constants
from .constants import ESRI_REQ, ESRI_TYPES, NC_ENCODE_DEFAULTS, NO_DATA
from .esri_grid import (
    check_grid_compatibility,
    check_resolutions,
    read_grid,
    read_header,
    rescale_grid,
    standardize_header,
    write_grid,
    write_header,
)

# netcdf loads xarray, so it and its exports are imported on first access
_NETCDF_EXPORTS = ("generate_bounds", "set_netcdf_encoding")

__all__ = ["constants", "netcdf"]
__all__ += ["ESRI_REQ", "ESRI_TYPES", "NC_ENCODE_DEFAULTS", "NO_DATA"]
__all__ += [
    "check_grid_compatibility",
    "check_resolutions",
    "read_grid",
    "read_header",
    "rescale_grid",
    "standardize_header",
    "write_grid",
    "write_header",
]
__all__ += ["generate_bounds", "set_netcdf_encoding"]


def __getattr__(name):
    """Import the netcdf module and its exports on first access.

    Args:
        name (str): Attribute name.

    Returns
    -------
        The netcdf module or the requested netcdf function.
    """
    if name == "netcdf" or name in _NETCDF_EXPORTS:
        netcdf = importlib.import_module(".netcdf", __name__)
        return netcdf if name == "netcdf" else getattr(netcdf, name)
    error_msg = f"module '{__name__}' has no attribute '{name}'"
    raise AttributeError(error_msg)


def __dir__():
    """List the package attributes including the not yet imported exports."""
    return sorted(set(globals().keys()) | set(__all__))
