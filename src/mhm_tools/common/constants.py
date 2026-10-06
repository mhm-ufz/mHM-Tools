"""Common constants.

Constants
=========

.. autosummary::
    NO_DATA
    NC_ENCODE_DEFAULTS
    EARTH_RADIUS_M
    METERS_PER_DEGREE
    ESRI_TYPES
    ESRI_REQ
    GREENLAND_COORDS

----

.. autodata:: NO_DATA

.. autodata:: NC_ENCODE_DEFAULTS

.. autodata:: EARTH_RADIUS_M

.. autodata:: METERS_PER_DEGREE

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

EARTH_RADIUS_M = 6_371_008.8
"""Mean earth radius in meters (IUGG), the sphere every distance here assumes."""
EARTH_RADIUS_KM = EARTH_RADIUS_M / 1000
"""Mean earth radius in kilometers."""
METERS_PER_DEGREE = EARTH_RADIUS_M * np.pi / 180
"""Length of one degree along a meridian in meters."""
MIN_COS_LATITUDE = 0.01
"""Lower limit for cos(latitude), so a conversion near the poles stays finite."""

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
MHM_TWS_STORAGE_VARS = (
    "interception",
    "swe",
    "sealedSTW",
    "unsatSTW",
    "satSTW",
)
"""mHM storage variables that sum up to total water storage, without the soil layers."""
MHM_SOIL_MOISTURE_PREFIX = "SWC_L"
"""Prefix of the per-layer mHM soil water content variables (SWC_L01, SWC_L02, ...)."""
KGE_CONSTANT_MEAN_BOUND = -0.41
"""Lower end of every KGE scale, the score of a constant mean value predictor."""
NSE_CONSTANT_MEAN_BOUND = 0.0
"""Lower end of every NSE scale, the score of a constant mean value predictor."""
DEFAULT_DISCHARGE_UNITS = "m3 s-1"
"""Unit assumed for a discharge record without a units attribute, as GRDC and mRM use."""
SECONDS_PER_YEAR = 365.25 * 86400.0
"""Length of a year of units such as mm/year, the mean Julian year."""
TIME_UNIT_SECONDS = {
    "s": 1.0,
    "min": 60.0,
    "h": 3600.0,
    "d": 86400.0,
    "month": SECONDS_PER_YEAR / 12,
    "year": SECONDS_PER_YEAR,
}
"""Seconds per time unit of a rate, keyed by the canonical names of `common.units`."""
AMOUNT_UNITS = {
    (("m", 3),): ("volume", 1.0),
    (("hm", 3),): ("volume", 1e6),
    (("km", 3),): ("volume", 1e9),
    (("l", 1),): ("volume", 1e-3),
    (("mm", 1),): ("depth", 1e-3),
    (("cm", 1),): ("depth", 1e-2),
    (("m", 1),): ("depth", 1.0),
    (("kg", 1), ("m", -2)): ("depth", 1e-3),
}
"""Kind and size of an amount, keyed by its unit exponents: m3 per volume unit or
metres of water per depth unit, where 1 kg/m2 of water is 1 mm."""
