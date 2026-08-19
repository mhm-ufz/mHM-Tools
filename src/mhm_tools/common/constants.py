"""Common constants.

Constants
=========

.. autosummary::
    NO_DATA
    NC_ENCODE_DEFAULTS
    ESRI_TYPES
    ESRI_REQ
    GREENLAND_COORDS

----

.. autodata:: NO_DATA

.. autodata:: NC_ENCODE_DEFAULTS

.. autodata:: ESRI_TYPES

.. autodata:: ESRI_REQ

.. autodata:: GREENLAND_COORDS
"""

import numpy as np

__all__ = ["NC_ENCODE_DEFAULTS", "NO_DATA"]

NO_DATA = -9999.0
"""Default no data value for mHM."""

NC_ENCODE_DEFAULTS = {"_FillValue": NO_DATA, "missing_value": NO_DATA}
NC_ENCODE_MASK = {"_FillValue": 0, "missing_value": 0}
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
"""Types for ESRI ASCII grid header information."""

ESRI_REQ = {"ncols", "nrows", "xllcorner", "yllcorner", "cellsize"}
"""Required ESRI ASCII grid header information."""


LOG_LEVELS = {
    "info": 20,
    "warning": 30,
    "warn": 30,
    "debug": 10,
    "error": 40,
    "INFO": 20,
    "WARNING": 30,
    "WARN": 30,
    "DEBUG": 10,
    "ERROR": 40,
}

LOG_LEVEL_STR = {
    10: "DEBUG",
    20: "INFO",
    30: "WARNING",
    40: "ERROR",
    50: "CRITICAL",
}

# Characters indicating wildcard patterns in NetCDF filenames
WILDCARDS = ("*", "?", "[", "]")


# possible coordinate keys
LAT_KEYS = ["lat", "latitude", "northing", "y", "new_y", "Y", "geo_y", "lat_l2"]
LON_KEYS = ["lon", "longitude", "easting", "x", "new_x", "X", "geo_x", "lon_l2"]
TIME_KEYS = ["time", "month_of_year", "valid_time"]

# Coordinate array for the shape of Greenland in lons, lats
GREENLAND_COORDS = np.array(
    [
        [-43.65, -17.25, -7.05, -56.05, -60.05, -69.85, -73.65, -71.85, -46.65],
        [58.25, 69.85, 84.65, 85.85, 82.65, 79.45, 79.05, 75.25, 57.65],
    ]
).T
"""Polygon vertices (lon, lat) approximating Greenland's coastline."""
