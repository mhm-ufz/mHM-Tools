import numpy as np
import pytest
import xarray as xr

from mhm_tools.common.file_handler import write_xarray_to_file
from mhm_tools.common.netcdf import sanitize_nc_encoding, set_netcdf_encoding


def _make_ds(dtype=float):
    data = np.array([[1, 2], [3, 4]], dtype=dtype)
    da = xr.DataArray(
        data,
        dims=("lat", "lon"),
        coords={"lat": [10.0, 11.0], "lon": [100.0, 101.0]},
        name="v",
    )
    return da.to_dataset()


def _make_time_ds(
    time_values,
    time_bnds_values,
    *,
    time_units="hours since 2000-01-01 00:00:00",
    time_calendar="proleptic_gregorian",
):
    da = xr.DataArray(
        np.random.rand(len(time_values), 2, 2),
        dims=("time", "lat", "lon"),
        coords={
            "time": ("time", time_values),
            "lat": ("lat", [10.0, 11.0]),
            "lon": ("lon", [100.0, 101.0]),
        },
        name="v",
    )
    ds = da.to_dataset()
    ds["time"].attrs["bounds"] = "time_bnds"
    ds["time"].attrs["units"] = time_units
    ds["time"].attrs["calendar"] = time_calendar
    ds["time_bnds"] = xr.DataArray(
        time_bnds_values, dims=("time", "bnds"), coords={"time": ds["time"]}
    )
    return ds


def test_sanitize_nc_encoding_casts_fillvalue_and_preserves_missing_value():
    ds = _make_ds(dtype=np.float32)
    ds["v"].attrs["_FillValue"] = 9999
    ds["v"].attrs["missing_value"] = 9999

    encoding = {"v": {"_FillValue": "9999", "zlib": True, "complevel": 4}}
    out = sanitize_nc_encoding(ds, encoding)

    assert isinstance(out["v"]["_FillValue"], float)
    assert "missing_value" in ds["v"].attrs


def test_set_netcdf_encoding_creates_bounds_and_sets_encodings():
    ds = _make_ds(dtype=np.float32)
    set_netcdf_encoding(ds)
    assert "lat_bnds" in ds.coords
    assert "lon_bnds" in ds.coords
    assert ds["lat"].encoding.get("_FillValue") is None


def test_write_xarray_to_file_handles_reserved_attrs(tmp_path):
    ds = _make_ds(dtype=np.float32)
    ds["v"].attrs["_FillValue"] = 9999
    ds["v"].attrs["missing_value"] = 9999

    out = tmp_path / "test.nc"
    write_xarray_to_file(ds, out)
    assert out.is_file()

    ds_read = xr.open_dataset(out, engine="netcdf4")
    assert "v" in ds_read
    assert ds_read["v"].shape == (2, 2)


def test_write_xarray_to_file_bool_drops_fillvalue(tmp_path):
    ds = _make_ds(dtype=bool)
    ds["v"].attrs["_FillValue"] = True

    out = tmp_path / "test_bool.nc"
    write_xarray_to_file(ds, out)
    ds_read = xr.open_dataset(out, engine="netcdf4")
    assert ds_read["v"].dtype in (bool, np.uint8)


def test_write_xarray_to_file_strips_time_bnds_units_attrs(tmp_path):
    time = np.array(["2017-01-01T00:00", "2017-01-01T01:00"], dtype="datetime64[ns]")
    time_bnds = np.stack(
        [time - np.timedelta64(1, "h"), time + np.timedelta64(1, "h")], axis=1
    )
    ds = _make_time_ds(time, time_bnds)
    ds["time_bnds"].attrs["units"] = "hours since 2017-01-01 00:00:00"
    ds["time_bnds"].attrs["calendar"] = "proleptic_gregorian"

    out = tmp_path / "time_bounds_attrs.nc"
    write_xarray_to_file(ds, out)
    assert out.is_file()


def test_write_xarray_to_file_regenerates_numeric_time_bnds(tmp_path):
    time = np.arange(3, dtype=float)
    # wildly out-of-scale bounds (nanoseconds-like)
    time_bnds = np.stack([time * 1e9, time * 1e9 + 1], axis=1)
    ds = _make_time_ds(time, time_bnds)

    out = tmp_path / "time_bounds_numeric.nc"
    write_xarray_to_file(ds, out)
    ds_read = xr.open_dataset(out, engine="netcdf4", decode_times=False)

    tmax = float(np.nanmax(np.abs(ds_read["time"].values)))
    bmax = float(np.nanmax(np.abs(ds_read["time_bnds"].values)))
    assert tmax > 0
    assert bmax / tmax < 1e6


def test_write_xarray_to_file_regenerates_datetime_bounds(tmp_path):
    time = np.array(["2017-01-01T00:00", "2017-01-01T01:00"], dtype="datetime64[ns]")
    # numeric bounds (wrong type) should be regenerated to datetime
    time_bnds = np.stack([np.array([0, 1]), np.array([1, 2])], axis=1)
    ds = _make_time_ds(time, time_bnds)

    out = tmp_path / "time_bounds_datetime.nc"
    write_xarray_to_file(ds, out)
    ds_read = xr.open_dataset(out, engine="netcdf4")
    assert np.issubdtype(ds_read["time_bnds"].dtype, np.datetime64)
