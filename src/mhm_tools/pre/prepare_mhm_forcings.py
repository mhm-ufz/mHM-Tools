"""
Prepare NetCDF MHM forcing files.

This module provides functions to:
- Convert meteorological time series into the units expected by MHM
- Crop spatial fields to a user-defined region
- Write the pre-processed data out as CF-compliant NetCDF

Authors
-------
- Jeisson Leal
- Simon Lüdke
"""

import logging
from pathlib import Path
from typing import Optional, Tuple, Union

import xarray as xr

from mhm_tools.common.file_handler import get_xarray_ds_from_file, write_xarray_to_file
from mhm_tools.common.logger import ErrorLogger, log_arguments
from mhm_tools.common.time_utils import (
    calculate_median_time_step_seconds,
    resample_to_daily_or_hourly_adaptive,
)
from mhm_tools.common.units import (
    calculate_amount_factor,
    calculate_conversion_factor,
    convert_temperature_to_celsius,
    split_units,
)
from mhm_tools.common.xarray_utils import crop_ds, get_single_data_var

logger = logging.getLogger(__name__)


def convert_precipitation_to_mm(da: xr.DataArray, units: str, var: str):
    """Convert a precipitation amount or rate to millimetres per time step.

    Args:
        da: Precipitation DataArray with a time axis.
        units: Its units, an amount such as "kg m-2" or a rate such as "mm s-1".
        var: Variable name used in the error messages.

    Returns
    -------
        The precipitation in mm per time step of the record.
    """
    try:
        parts = split_units(units)
    except ValueError:
        parts = None
    if parts is None or parts.kind != "depth":
        msg = f"Unexpected units '{units}' for variable '{var}'."
        raise ValueError(msg)
    if parts.time_seconds is None:
        return da * calculate_conversion_factor(units, "mm")
    # a rate becomes the amount that falls within one time step
    step_seconds = calculate_median_time_step_seconds(da["time"].values)
    if step_seconds is None:
        msg = f"Cannot infer the time step of '{var}' to turn {units} into mm."
        with ErrorLogger(logger):
            raise ValueError(msg)
    return da * calculate_amount_factor(units, step_seconds, "mm")


def convert_units(
    ds: Union[xr.Dataset, xr.DataArray], var: str
) -> Tuple[xr.DataArray, dict]:
    """Convert variable to standard units.

    Temperatures become degrees Celsius (degC). Precipitation amounts become
    millimetres (mm), where 1 kg m-2 of water is 1 mm, and precipitation rates
    millimetres per time step of the record, the step taken from its time axis.

    Returns
    -------
        Tuple of (converted DataArray, encoding with the fill values).
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
    original_attrs = dict(da.attrs)
    logger.info(f"units are: {units}")
    try:
        celsius = convert_temperature_to_celsius(da, units)
    except ValueError:
        celsius = None
    if celsius is not None:
        da, new_units = celsius, "degC"
    else:
        da, new_units = convert_precipitation_to_mm(da, units, var), "mm"
    da.attrs = original_attrs
    da.attrs["units"] = new_units

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
    out_var: Optional[str] = None,
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
        da, _encoding = convert_units(ds, var)

        if out_var:
            da = da.rename(out_var)

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
