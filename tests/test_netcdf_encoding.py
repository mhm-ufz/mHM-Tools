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
