"""Common routines and constants.

Subpackages
===========

.. autosummary::
   :toctree:

   constants

NetCDF
======

.. autosummary::
   :toctree:

   set_netcdf_encoding
   generate_bounds

ESRI grids
==========

.. autosummary::
   :toctree:

   read_header
   read_grid
   write_header
   write_grid
   standardize_header
   rescale_grid
   check_resolutions
   check_grid_compatibility

Constants
=========

.. currentmodule:: mhm_tools.common.constants

.. autosummary::

   NO_DATA
   NC_ENCODE_DEFAULTS
   ESRI_TYPES
   ESRI_REQ
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
__all__ += ["NO_DATA", "NC_ENCODE_DEFAULTS", "ESRI_TYPES", "ESRI_REQ"]
__all__ += [
    "read_header",
    "read_grid",
    "standardize_header",
    "rescale_grid",
    "check_resolutions",
    "check_grid_compatibility",
    "write_header",
    "write_grid",
]
__all__ += ["set_netcdf_encoding", "generate_bounds"]
