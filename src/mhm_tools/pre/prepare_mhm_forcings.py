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
from typing import Optional, Union

import pandas as pd
import xarray as xr

from mhm_tools.common.file_handler import get_xarray_ds_from_file, write_xarray_to_file
from mhm_tools.common.logger import ErrorLogger, log_arguments
from mhm_tools.common.time_utils import resample_to_daily_or_hourly_adaptive
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


def convert_units(ds: Union[xr.Dataset, xr.DataArray], var: str) -> xr.DataArray:
    """Convert variable to standard units.

    Temperature variables are converted to degrees Celsius (degC),
    and precipitation variables are converted to millimeters (mm).
    """
    logger.info(f"Converting units for variable '{var}'")
    logger.debug(f"Original dataset: {ds}")
    if isinstance(ds, xr.Dataset):
        if var not in ds:
            msg = f"Variable '{var}' not found in dataset."
            raise ValueError(msg)
        da = ds[var]
    else:
        da = ds
    units = da.attrs.get("units")
    if not units:
        msg = f"Variable '{var}' missing 'units' attribute."
        raise ValueError(msg)
    logger.info(f"units are: {units}")
    # Temperature
    if units in TEMPERATURE_UNITS:
        if units in ["K", "Kelvin", "kelvin"]:
            da = da - 273.15
        elif units in ["F", "°F", "degF", "fahrenheit"]:
            da = (da - 32) * (5 / 9)
        da.attrs["units"] = "degC"
    # Total precipitation
    elif units in PRECIPITATION_UNITS:
        if units in ["m", "kg m-2"]:
            da = da * 1000
        da.attrs["units"] = "mm"

    # Precipitation rate
    elif units in PRECIPITATION_RATE_UNITS:
        freq = pd.infer_freq(da.indexes["time"])
        if not freq or not freq.startswith(("h", "D")):
            msg = (
                f"Cannot infer frequency from time coordinate with freq={freq!r}. "
                f"Expected hourly or daily frequency."
            )
            with ErrorLogger(logger):
                raise ValueError(msg)
        factor = 1.0
        if "kg" in units and "s-1" in units:
            factor = (
                86400 if freq.startswith("D") else 3600 if freq.startswith("h") else 1
            )
        elif units == "mm d-1" and freq:
            factor = (
                1 if freq.startswith("D") else 1 / 24 if freq.startswith("h") else 1
            )
        elif units == "mm s-1":
            factor = (
                90000 if freq.startswith("D") else 3600 if freq.startswith("h") else 1
            )
        da = da * factor
        da.attrs["units"] = "mm"
    else:
        msg = f"Unexpected units '{units}' for variable '{var}'."
        raise ValueError(msg)

    mv = -9999.0
    encoding = {"_FillValue": mv, "missing_value": mv}
    da.attrs.update({"_FillValue": mv, "missing_value": mv})
    logger.info(f"Converted variable '{var}' with units {da.attrs['units']}")
    return da, encoding


@log_arguments("DEBUG")
def prepare_forcings(
    in_dir: str,
    in_file: str,
    out_dir: str,
    out_file: str,
    var: Optional[str] = None,
    crop: bool = False,
    lon_min: Optional[float] = None,
    lon_max: Optional[float] = None,
    lat_min: Optional[float] = None,
    lat_max: Optional[float] = None,
    use_mfdataset: bool = False,
    target_frequency: Optional[str] = None,
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
            force_decending_y=True,
        )

        if var is None:
            var = get_single_data_var(ds)

        # needs to be before unit conversion because that changes rates to quantities
        if target_frequency is not None:
            ds = resample_to_daily_or_hourly_adaptive(
                in_obj=ds, target=target_frequency, var=var
            )

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
        write_xarray_to_file(
            ds=da, file_path=Path(out_dir) / name
        )  # , encoding=encoding)
