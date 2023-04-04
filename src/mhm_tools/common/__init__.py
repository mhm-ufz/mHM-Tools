"""Common routines and constants.

Constants
=========

.. autosummary::
   :toctree: generated

    NO_DATA
    NC_ENCODE_DEFAULTS

NetCDF
======

.. autosummary::
   :toctree: generated

    set_netcdf_encoding

----
.. autodata:: NO_DATA
.. autodata:: NC_ENCODE_DEFAULTS
"""

from . import constants, netcdf
from .constants import NC_ENCODE_DEFAULTS, NO_DATA
from .netcdf import set_netcdf_encoding

__all__ = ["constants", "netcdf"]
__all__ += ["NO_DATA", "NC_ENCODE_DEFAULTS"]
__all__ += ["set_netcdf_encoding"]
