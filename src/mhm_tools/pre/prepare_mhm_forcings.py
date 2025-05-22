import glob
import logging
from pathlib import Path
from typing import Optional

import pandas as pd
import xarray as xr

from mhm_tools.common.file_handler import get_xarray_ds_from_file
from mhm_tools.common.xarray_utils import crop_ds

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
PRECIPITATION_RATE_UNITS = ["kg m-2 s-1"]


def convert_units(ds: xr.Dataset, var: str) -> xr.Dataset:
    units = ds[var].attrs.get("units")
    if not units:
        raise ValueError(f"Variable '{var}' missing 'units' attribute.")

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
        if freq and freq.startswith("D"):
            factor = 86400
        elif freq and freq.startswith("H"):
            factor = 3600
        else:
            factor = 1
        ds[new_var] = ds[new_var] * factor
        ds[new_var].attrs["units"] = "mm"
    else:
        raise ValueError(f"Unexpected units '{units}' for variable '{var}'.")

    mv = -9999.0
    ds[new_var].attrs.update({"_FillValue": mv, "missing_value": mv})
    return ds[new_var]


def ensure_lat_lon_order(ds: xr.Dataset) -> xr.Dataset:
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
    var: str,
    crop: bool = False,
    lon_min: Optional[float] = None,
    lon_max: Optional[float] = None,
    lat_min: Optional[float] = None,
    lat_max: Optional[float] = None,
    use_mfdataset: bool = False,
) -> None:
    """Loop through all files matching in_file in in_dir, convert units,
    optionally crop, and write to NetCDF in out_dir with naming controlled by
    out_file.
    """
    pattern = str(Path(in_dir) / in_file)
    files = sorted(glob.glob(pattern))
    if not files:
        with ErrorLogger(logger):
            msg = "No files match pattern {pattern}"
            raise FileNotFoundError(msg)

    Path(out_dir).mkdir(parents=True, exist_ok=True)
    for path in files:
        # Load data-array
        ds = get_xarray_ds_from_file(
            file_path=path,
            use_mfdataset=use_mfdataset,
            normalize_latlon_coords=True,
        )

        # Sort lat/lon if needed
        ds = ensure_lat_lon_order(ds)
        # Convert units and return da
        da = convert_units(ds, var)
        # Crop spatially
        if crop:
            if None in (lon_min, lon_max, lat_min, lat_max):
                with ErrorLogger(logger):
                    msg = "All lon/lat bounds must be provided when crop=True."
                    raise ValueError(msg)
            da = crop_ds(da, lon_min, lon_max, lat_min, lat_max)
        # Determine output name
        name = Path(path).name if out_file == "*" else out_file
        # Write output
        da.to_netcdf(Path(out_dir) / name)
