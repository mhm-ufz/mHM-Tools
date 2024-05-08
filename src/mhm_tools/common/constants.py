"""
Common constants.

Constants
=========

.. autosummary::
    NO_DATA
    NC_ENCODE_DEFAULTS
    ESRI_TYPES
    ESRI_REQ

----

.. autodata:: NO_DATA

.. autodata:: NC_ENCODE_DEFAULTS

.. autodata:: ESRI_TYPES

.. autodata:: ESRI_REQ

"""

__all__ = ["NO_DATA", "NC_ENCODE_DEFAULTS"]

NO_DATA = -9999.0
"""Default no data value for mHM."""

NC_ENCODE_DEFAULTS = {"_FillValue": NO_DATA, "missing_value": NO_DATA}
"""Default netcdf encoding settings."""

ESRI_TYPES = {
    "ncols": int,
    "nrows": int,
    "xllcorner": float,
    "yllcorner": float,
    "xllcenter": float,
    "yllcenter": float,
    "cellsize": float,
    "nodata_value": float,
}
"""types for ESRI ASCII grid header information."""

ESRI_REQ = {"ncols", "nrows", "xllcorner", "yllcorner", "cellsize"}
"""Required ESRI ASCII grid header information."""
