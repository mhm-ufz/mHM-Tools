"""
Prepare NetCDF MHM forcing files.

This module provides functions to:
- Convert meteorological time series into the units expected by MHM
- Crop spatial fields to a user-defined region
- Write the pre-processed data out as CF-compliant NetCDF

Authors
-------
- Jeisson Leal
"""

import logging
from pathlib import Path
from typing import Optional

import pandas as pd
from mhm_tools.common.time_utils import resample_to_daily_or_hourly_adaptive
import xarray as xr

from mhm_tools.common.file_handler import get_xarray_ds_from_file, write_xarray_to_file
from mhm_tools.common.logger import ErrorLogger
from mhm_tools.common.xarray_utils import crop_ds, get_single_data_var

logger = logging.getLogger(__name__)

# Define acceptable input unit lists for detection
TEMPERATURE_UNITS = [
    "K",
    "Kelvin",
    "kelvin",
    "C",
    "°C",
    "degC",
    "celsius",
    "F",
    "°F",
    "degF",
    "fahrenheit",
]
PRECIPITATION_UNITS = ["m", "kg m-2", "mm"]
PRECIPITATION_RATE_UNITS = ["kg m-2 s-1", "mm s-1", "mm d-1"]


def convert_units(ds: xr.Dataset, var: str) -> xr.DataArray:
    """Convert variable to standard units.

    Temperature variables are converted to degrees Celsius (degC),
    and precipitation variables are converted to millimeters (mm).
    """
    units = ds[var].attrs.get("units")
    if not units:
        msg = f"Variable '{var}' missing 'units' attribute."
        raise ValueError(msg)
    logger.info(f"units are: {units}")
    # Temperature
    if units in TEMPERATURE_UNITS:
        new_var = "tavg"
        ds = ds.rename({var: new_var})
        if units in ["K", "Kelvin", "kelvin"]:
            ds[new_var] = ds[new_var] - 273.15
        elif units in ["F", "°F", "degF", "fahrenheit"]:
            ds[new_var] = (ds[new_var] - 32) * (5 / 9)
        ds[new_var].attrs["units"] = "degC"

    # Total precipitation
    elif units in PRECIPITATION_UNITS:
        new_var = "pre"
        ds = ds.rename({var: new_var})
        if units in ["m", "kg m-2"]:
            ds[new_var] = ds[new_var] * 1000
        ds[new_var].attrs["units"] = "mm"

    # Precipitation rate
    elif units in PRECIPITATION_RATE_UNITS:
        new_var = "pre"
        ds = ds.rename({var: new_var})
        freq = pd.infer_freq(ds.indexes["time"])
        if 'kg' in units and "s-1" in units:
            if freq and freq.startswith("D"):
                factor = 86400
            elif freq and freq.startswith("H"):
                factor = 3600
            else:
                factor = 1
        elif units == "mm d-1":
            if freq and freq.startswith("D"):
                factor = 1
            elif freq and freq.startswith("H"):
                factor = 1 / 24
        elif units == "mm s-1":
            if freq and freq.startswith("D"):
                factor = 90000
            elif freq and freq.startswith("H"):
                factor = 3600
            
        ds[new_var] = ds[new_var] * factor
        ds[new_var].attrs["units"] = "mm"
    else:
        msg = f"Unexpected units '{units}' for variable '{var}'."
        raise ValueError(msg)

    mv = -9999.0
    encoding = {"_FillValue": mv, "missing_value": mv}
    ds[new_var].attrs.update({"_FillValue": mv, "missing_value": mv})
    return ds[new_var], encoding


def ensure_lat_lon_order(ds: xr.Dataset) -> xr.Dataset:
    """Ensure latitude and longitude axes are ordered correctly.

    Latitudes will be sorted descending if needed, and longitudes ascending.
    """
    if not (ds["lat"].values[1:] < ds["lat"].values[:-1]).all():
        ds = ds.sortby("lat", ascending=False)
    if not (ds["lon"].values[1:] > ds["lon"].values[:-1]).all():
        ds = ds.sortby("lon", ascending=True)
    return ds


def prepare_forcings(
    in_dir: str,
    in_file: str,
    out_dir: str,
    out_file: str,
    var: str = None,
    crop: bool = False,
    lon_min: Optional[float] = None,
    lon_max: Optional[float] = None,
    lat_min: Optional[float] = None,
    lat_max: Optional[float] = None,
    use_mfdataset: bool = False,
    target_frequency: str = None
) -> None:
    """Loop through all files matching in_file in in_dir, convert units.

    Optionally crop, and write to NetCDF in out_dir with naming controlled by out_file.
    """
    files = sorted(Path(in_dir).glob(in_file))
    if not files:
        with ErrorLogger(logger):
            msg = f"No files match pattern {in_file!r} in directory {in_dir!r}"
            raise FileNotFoundError(msg)

    Path(out_dir).mkdir(parents=True, exist_ok=True)
    for path in files:
        # Load dataset
        ds = get_xarray_ds_from_file(
            file_path=str(path),
            use_mfdataset=use_mfdataset,
            normalize_latlon_coords=True,
        )
        if var is None: 
            var = get_single_data_var(ds)
        # Sort lat/lon if needed
        ds = ensure_lat_lon_order(ds)

        # needs to be before unit conversion because that changes rates to quantities
        if target_frequency is not None:
            ds = resample_to_daily_or_hourly_adaptive(ds, target_frequency)
        
        # Convert units and get DataArray
        da, encoding = convert_units(ds, var)

        # Crop spatially
        if crop:
            if None in (lon_min, lon_max, lat_min, lat_max):
                with ErrorLogger(logger):
                    msg = "All lon/lat bounds must be provided when crop=True."
                    raise ValueError(msg)
            da = crop_ds(da, lon_min, lon_max, lat_min, lat_max)

        # Determine output name
        name = path.name if out_file == "*" else out_file

        # Write output
        logger.info(da)
        write_xarray_to_file(ds=da, file_path=Path(out_dir) / name)#, encoding=encoding)
