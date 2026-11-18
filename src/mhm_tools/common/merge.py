"""
Write NetCDF time series block by block.

Authors
-------
- Sanjeev Bashyal
"""

from __future__ import annotations

from pathlib import Path


def write_netcdf(ds, variable: str, output_file: Path, *, unlimited_dims=None, compression=None) -> None:
    """Write one mHM forcing NetCDF with conservative CF/xarray encoding.

    `compression` is a `NetcdfCompression`; None keeps zlib level 4 with shuffle.
    """
    output_file.parent.mkdir(parents=True, exist_ok=True)
    encoding = {
        variable: {
            "dtype": "float64",
            "_FillValue": -9999.0,
            "missing_value": -9999.0,
            "zlib": True,
            "complevel": 4,
        },
        "time": {
            "dtype": "float64",
            "units": "hours since 1900-01-01 00:00:00",
            "calendar": "proleptic_gregorian",
        },
        "time_bnds": {
            "dtype": "float64",
            "units": "hours since 1900-01-01 00:00:00",
            "calendar": "proleptic_gregorian",
        },
        "lat": {
            "dtype": "float64",
            "_FillValue": None,
        },
        "lon": {
            "dtype": "float64",
            "_FillValue": None,
        },
    }
    if compression is not None:
        from mhm_tools.common.netcdf import create_quantization_encoding

        encoding[variable].update(compression.get_lossless_encoding(encoding[variable]))
        encoding[variable].update(create_quantization_encoding(
            ds, [variable], compression.significant_digits, compression.quantize_mode
        ).get(variable, {}))
    # netcdf4 only: the time blocks are appended with netCDF4 and only it quantizes
    _strip_attrs_managed_by_encoding(ds, encoding).to_netcdf(
        output_file, engine="netcdf4", encoding=encoding, unlimited_dims=unlimited_dims)


def write_time_block(ds, variable, path, offset, handle=None, compression=None):
    """Write `ds` into `path` at time index `offset` and return the open handle.

    The first block (offset 0) creates the file with `compression` (see
    `write_netcdf`); later blocks are appended through the returned handle,
    which the caller closes.

    Reopening the file per block cost milliseconds each and grew with the
    unlimited time dimension, which dominated a multi-decade record.
    """
    import numpy as np
    import pandas as pd
    from netCDF4 import Dataset, date2num

    if not offset:
        write_netcdf(ds, variable, path, unlimited_dims=("time",), compression=compression)
        handle = Dataset(path, "a")
        for name in ("units", "calendar"):
            handle["time_bnds"].setncattr(name, handle["time"].getncattr(name))
        return handle
    stop = offset + ds.sizes["time"]
    handle[variable][offset:stop] = np.ma.masked_invalid(ds[variable].values)
    for name in ("time", "time_bnds"):
        target = handle[name]
        values = ds[name].values
        dates = pd.DatetimeIndex(values.reshape(-1)).to_pydatetime()
        target[offset:stop] = np.asarray(date2num(dates, getattr(target, "units", handle["time"].units), getattr(target, "calendar", handle["time"].calendar))).reshape(values.shape)
    return handle


def _strip_attrs_managed_by_encoding(ds, encoding: dict):
    """Remove attrs that xarray expects to own through encoding."""
    ds_to_write = ds.copy(deep=False)
    for variable, options in encoding.items():
        if variable not in ds_to_write.variables:
            continue
        for key in options:
            ds_to_write[variable].attrs.pop(key, None)
    return ds_to_write
