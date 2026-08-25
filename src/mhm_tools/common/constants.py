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
WMO_REGION_BOUNDS = {
    "Africa": {"lon_slice": slice(-20, 55), "lat_slice": slice(-35, 38)},
    "Asia": {"lon_slice": slice(25, 180), "lat_slice": slice(0, 85)},
    "South America": {"lon_slice": slice(-82, -34), "lat_slice": slice(-56, 13)},
    "North/Central America": {"lon_slice": slice(-168, -52), "lat_slice": slice(5, 84)},
    "SW Pacific": {"lon_slice": slice(95, 180), "lat_slice": slice(-50, 25)},
    "Europe": {"lon_slice": slice(-25, 60), "lat_slice": slice(35, 82)},
}
WMO_INDEX_TO_REGION = {
    1: "Africa",
    2: "Asia",
    3: "South America",
    4: "North/Central America",
    5: "SW Pacific",
    6: "Europe",
}
WMO_REGION_TO_INDEX = {v: k for k, v in WMO_INDEX_TO_REGION.items()}
