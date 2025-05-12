from pathlib import Path
from typing import Optional, Tuple
import glob

import pandas as pd
import xarray as xr

# Define acceptable input unit lists for detection
TEMPERATURE_UNITS = ["K", "Kelvin", "kelvin", "C", "°C", "degC", "celsius", "F", "°F", "degF", "fahrenheit"]
PRECIPITATION_UNITS = ["m", "kg m-2", "mm"]
PRECIPITATION_RATE_UNITS = ["kg m-2 s-1"]


def get_coord_names(ds: xr.Dataset) -> Tuple[str, str]:
    if 'lon' in ds.coords:
        lon_name = 'lon'
    elif 'longitude' in ds.coords:
        lon_name = 'longitude'
    else:
        raise ValueError("Longitude coordinate ('lon' or 'longitude') not found in dataset.")

    if 'lat' in ds.coords:
        lat_name = 'lat'
    elif 'latitude' in ds.coords:
        lat_name = 'latitude'
    else:
        raise ValueError("Latitude coordinate ('lat' or 'latitude') not found in dataset.")

    return lon_name, lat_name


def convert_units(ds: xr.Dataset, var: str) -> xr.Dataset:
    if var not in ds.data_vars:
        raise KeyError(f"Variable '{var}' not found in dataset.")
    units = ds[var].attrs.get("units")
    if not units:
        raise ValueError(f"Variable '{var}' missing 'units' attribute.")

    # Temperature
    if units in TEMPERATURE_UNITS:
        new_var = 'tavg'
        ds = ds.rename({var: new_var})
        if units in ["K", "Kelvin", "kelvin"]:
            ds[new_var] = ds[new_var] - 273.15
        elif units in ["F", "°F", "degF", "fahrenheit"]:
            ds[new_var] = (ds[new_var] - 32) * (5/9)
        ds[new_var].attrs['units'] = 'degC'

    # Total precipitation
    elif units in PRECIPITATION_UNITS:
        new_var = 'pre'
        ds = ds.rename({var: new_var})
        if units in ["m", "kg m-2"]:
            ds[new_var] = ds[new_var] * 1000
        ds[new_var].attrs['units'] = 'mm'

    # Precipitation rate
    elif units in PRECIPITATION_RATE_UNITS:
        new_var = 'pre'
        ds = ds.rename({var: new_var})
        freq = pd.infer_freq(ds.indexes['time'])
        if freq and freq.startswith('D'):
            factor = 86400
        elif freq and freq.startswith('H'):
            factor = 3600
        else:
            factor = 1
        ds[new_var] = ds[new_var] * factor
        ds[new_var].attrs['units'] = 'mm'
    else:
        raise ValueError(f"Unexpected units '{units}' for variable '{var}'.")

    mv = -9999.0
    ds[new_var].attrs.update({"_FillValue": mv, "missing_value": mv})
    return ds


def ensure_lat_lon_order(ds: xr.Dataset) -> xr.Dataset:
    lon_name, lat_name = get_coord_names(ds)
    if not (ds[lat_name].values[1:] < ds[lat_name].values[:-1]).all():
        ds = ds.sortby(lat_name, ascending=False)
    if not (ds[lon_name].values[1:] > ds[lon_name].values[:-1]).all():
        ds = ds.sortby(lon_name, ascending=True)
    return ds


def crop_forcing(
    ds: xr.Dataset,
    lon_min: float,
    lon_max: float,
    lat_min: float,
    lat_max: float
) -> xr.Dataset:
    """
    Crop an xarray.Dataset to the given lon/lat bounds, handling coordinate order.

    Automatically reverses slice bounds if the coordinate axis is descending.
    """
    lon_name, lat_name = get_coord_names(ds)
    # Determine lon slice bounds for ascending/descending
    lon_vals = ds[lon_name].values
    if lon_vals[0] <= lon_vals[-1]:  # ascending
        lon_slice = slice(lon_min, lon_max)
    else:  # descending
        lon_slice = slice(lon_max, lon_min)
    # Determine lat slice bounds
    lat_vals = ds[lat_name].values
    if lat_vals[0] <= lat_vals[-1]:  # ascending
        lat_slice = slice(lat_min, lat_max)
    else:  # descending
        lat_slice = slice(lat_max, lat_min)
    return ds.sel({lon_name: lon_slice, lat_name: lat_slice})


def process_forcing(
    path: str,
    out_dir: str,
    out_file: str,
    var: str,
    crop: bool = False,
    lon_min: Optional[float] = None,
    lon_max: Optional[float] = None,
    lat_min: Optional[float] = None,
    lat_max: Optional[float] = None
) -> None:
    with xr.open_dataset(path, decode_times=True) as src_ds:
        ds = src_ds.load()
        ds = convert_units(ds, var)
        ds = ensure_lat_lon_order(ds)
        if crop:
            if None in (lon_min, lon_max, lat_min, lat_max):
                raise ValueError("All lon/lat bounds must be provided when crop=True.")
            ds = crop_forcing(ds, lon_min, lon_max, lat_min, lat_max)
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    if out_file == '*':
        # keep the same name as input file
        name = Path(path).name
    else:
        name = out_file
    ds.to_netcdf(Path(out_dir) / name)


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
    lat_max: Optional[float] = None
) -> None:
    pattern = str(Path(in_dir) / in_file)
    files = sorted(glob.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No files match pattern {pattern}")
    for path in files:
        process_forcing(
            path=path,
            out_dir=out_dir,
            out_file=out_file,
            var=var,
            crop=crop,
            lon_min=lon_min,
            lon_max=lon_max,
            lat_min=lat_min,
            lat_max=lat_max
        )
