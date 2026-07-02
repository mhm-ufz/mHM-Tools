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
   ~mhm_tools.common.esri_grid
   ~mhm_tools.common.file_handler
   ~mhm_tools.common.logger
   ~mhm_tools.common.netcdf
   ~mhm_tools.common.plotter
   ~mhm_tools.common.provenance
   ~mhm_tools.common.resolution_handler
   ~mhm_tools.common.time_utils
   ~mhm_tools.common.utils
   ~mhm_tools.common.xarray_utils
"""

from . import constants, netcdf
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
from .netcdf import generate_bounds, set_netcdf_encoding

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
