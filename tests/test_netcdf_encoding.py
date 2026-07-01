import logging

import numpy as np
import pytest
import xarray as xr

from mhm_tools.common.file_handler import write_xarray_to_file, write_xarray_to_netcdf
from mhm_tools.common.netcdf import (
    apply_cf_baseline_metadata,
    get_netcdf_metadata_data_vars,
    move_reserved_attrs_to_encoding,
    prepare_dataset_for_netcdf_write,
    prepare_time_bounds_encoding,
    sanitize_coordinate_encoding,
    sanitize_nc_encoding,
    set_netcdf_encoding,
)
from mhm_tools.common.provenance import CREATED_ATTR, HISTORY_ATTR


@pytest.fixture(autouse=True)
def _enable_mhm_tools_log_propagation_for_caplog():
    mhm_logger = logging.getLogger("mhm_tools")
    old_propagate = mhm_logger.propagate
    mhm_logger.propagate = True
    try:
        yield
    finally:
        mhm_logger.propagate = old_propagate


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


def test_get_netcdf_metadata_data_vars_detects_bounds_and_grid_mapping():
    """Detect coordinate bounds and grid mappings as metadata variables."""
    ds = _make_ds(dtype=np.float32)
    ds["lon"].attrs["bounds"] = "lon_bnds"
    ds["lat"].attrs["bounds"] = "lat_bnds"
    ds["v"].attrs["grid_mapping"] = "crs"
    ds["lon_bnds"] = (("lon", "bnds"), np.array([[99.5, 100.5], [100.5, 101.5]]))
    ds["lat_bnds"] = (("lat", "bnds"), np.array([[9.5, 10.5], [10.5, 11.5]]))
    ds["crs"] = xr.DataArray(
        0,
        attrs={"grid_mapping_name": "latitude_longitude"},
    )

    metadata_vars = get_netcdf_metadata_data_vars(ds)

    assert metadata_vars == {"lon_bnds", "lat_bnds", "crs"}
    assert "v" not in metadata_vars


def test_apply_cf_baseline_metadata_sets_expected_attrs(caplog):
    """Add CF baseline metadata and warn for incomplete data-variable attrs."""
    ds = xr.Dataset(
        {"v": (("time", "lat", "lon"), np.ones((2, 2, 2), dtype=np.float32))},
        coords={
            "time": np.array(["2001-01-01", "2001-01-02"], dtype="datetime64[ns]"),
            "lat": ("lat", [10.0, 11.0]),
            "lon": ("lon", [100.0, 101.0]),
        },
    )

    with caplog.at_level(logging.WARNING, logger="mhm_tools.common.netcdf"):
        apply_cf_baseline_metadata(ds, ["v"])

    assert ds.attrs.get("Conventions") == "CF-1.12"
    assert ds["lat"].attrs.get("standard_name") == "latitude"
    assert ds["lat"].attrs.get("units") == "degrees_north"
    assert ds["lon"].attrs.get("standard_name") == "longitude"
    assert ds["lon"].attrs.get("units") == "degrees_east"
    assert ds["time"].attrs.get("standard_name") == "time"
    warnings = [r.message for r in caplog.records if r.levelno >= logging.WARNING]
    assert any("has no 'units' attribute" in msg for msg in warnings)


def test_prepare_time_bounds_encoding_regenerates_bad_bounds():
    """Regenerate out-of-scale numeric bounds and wrong-typed datetime bounds."""
    numeric_time = np.arange(3, dtype=float)
    numeric_bnds = np.stack([numeric_time * 1e9, numeric_time * 1e9 + 1], axis=1)
    numeric_ds = _make_time_ds(numeric_time, numeric_bnds)

    numeric_out = prepare_time_bounds_encoding(numeric_ds, strip_time_attrs=True)

    time_max = float(np.nanmax(np.abs(numeric_out["time"].values)))
    bounds_max = float(np.nanmax(np.abs(numeric_out["time_bnds"].values)))
    assert time_max > 0
    assert bounds_max / time_max < 1e6
    assert "units" not in numeric_out["time"].attrs
    assert "units" not in numeric_out["time_bnds"].attrs
    assert (
        numeric_out["time_bnds"].encoding["units"]
        == numeric_out["time"].encoding["units"]
    )

    datetime_time = np.array(
        ["2017-01-01T00:00", "2017-01-01T01:00"], dtype="datetime64[ns]"
    )
    datetime_bnds = np.stack([np.array([0, 1]), np.array([1, 2])], axis=1)
    datetime_ds = _make_time_ds(datetime_time, datetime_bnds)

    datetime_out = prepare_time_bounds_encoding(datetime_ds)

    assert np.issubdtype(datetime_out["time_bnds"].dtype, np.datetime64)


def test_move_reserved_attrs_to_encoding_merges_per_variable_encoding():
    """Move reserved attrs and merge incoming per-variable encoding."""
    ds = _make_ds(dtype=np.float32)
    ds["v"].attrs["_FillValue"] = 9999
    ds["v"].attrs["scale_factor"] = 1.0
    ds["v"].attrs["missing_value"] = 9999

    cleaned, encoding = move_reserved_attrs_to_encoding(
        ds,
        encoding_in={"v": {"zlib": True, "complevel": 4}},
    )

    assert "_FillValue" not in cleaned["v"].attrs
    assert "scale_factor" not in cleaned["v"].attrs
    assert cleaned["v"].attrs["missing_value"] == 9999
    assert encoding["v"]["_FillValue"] == 9999
    assert encoding["v"]["scale_factor"] == 1.0
    assert encoding["v"]["zlib"] is True
    assert encoding["v"]["complevel"] == 4


def test_sanitize_coordinate_encoding_removes_backend_keys():
    """Remove stale backend encoding from coordinates and metadata variables."""
    ds = _make_ds(dtype=np.float32)
    ds["lat"].attrs["bounds"] = "lat_bnds"
    ds["lat_bnds"] = (("lat", "bnds"), np.array([[9.5, 10.5], [10.5, 11.5]]))
    ds["lat"].encoding.update(
        {"dtype": "f4", "zlib": True, "chunksizes": (1,), "source": "in.nc"}
    )
    ds["lat_bnds"].encoding.update(
        {"zlib": True, "complevel": 4, "chunksizes": (1, 1), "dtype": "f4"}
    )
    ds["lat_bnds"].attrs["_FillValue"] = -9999
    ds["lat_bnds"].attrs["missing_value"] = -9999

    sanitize_coordinate_encoding(ds)

    assert ds["lat"].encoding == {"dtype": "f4", "_FillValue": None}
    assert ds["lat_bnds"].encoding == {"dtype": "f4", "_FillValue": None}
    assert "_FillValue" not in ds["lat_bnds"].attrs
    assert "missing_value" not in ds["lat_bnds"].attrs


def test_prepare_dataset_for_netcdf_write_handles_bool_without_default_fillvalue():
    """Keep bool-to-uint8 conversion from receiving the default -9999 fill value."""
    ds = _make_ds(dtype=bool)

    ds_clean, encoding = prepare_dataset_for_netcdf_write(
        ds,
        ["v"],
        {"v": {"_FillValue": -9999, "zlib": True}},
    )

    assert ds_clean["v"].dtype == np.dtype("uint8")
    assert "v" not in encoding
    assert "_FillValue" not in ds_clean["v"].encoding


def test_write_xarray_to_netcdf_writes_dataset_directly(tmp_path):
    """Write a dataset through the direct NetCDF writer."""
    ds = _make_ds(dtype=np.float32)
    out = tmp_path / "direct_writer.nc"

    write_xarray_to_netcdf(ds, out)

    assert out.is_file()
    ds_read = xr.open_dataset(out, engine="netcdf4")
    assert "v" in ds_read
    assert ds_read.attrs.get("Conventions") == "CF-1.12"
    assert CREATED_ATTR in ds_read.attrs
    assert "mhm-tools command:" in ds_read.attrs[HISTORY_ATTR]


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


def test_write_xarray_to_file_scrubs_data_var_bounds_encoding(tmp_path):
    ds = _make_ds(dtype=np.float32)
    ds["lon"].attrs["bounds"] = "lon_bnds"
    ds["lat"].attrs["bounds"] = "lat_bnds"
    ds["lon_bnds"] = (("lon", "bnds"), np.array([[99.5, 100.5], [100.5, 101.5]]))
    ds["lat_bnds"] = (("lat", "bnds"), np.array([[9.5, 10.5], [10.5, 11.5]]))
    ds["lon_bnds"].encoding.update({"zlib": True, "complevel": 4, "chunksizes": (1, 1)})
    ds["lat_bnds"].encoding.update({"zlib": True, "complevel": 4, "chunksizes": (1, 1)})

    out = tmp_path / "data_var_bounds.nc"
    write_xarray_to_file(ds, out)

    ds_read = xr.open_dataset(out, engine="netcdf4")
    assert "lon_bnds" in ds_read
    assert "lat_bnds" in ds_read


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


def test_write_xarray_to_file_adds_cf_baseline_metadata(tmp_path):
    ds = xr.Dataset(
        {"v": (("time", "lat", "lon"), np.random.rand(2, 2, 2))},
        coords={
            "time": np.array(["2001-01-01", "2001-01-02"], dtype="datetime64[ns]"),
            "lat": ("lat", [10.0, 11.0]),
            "lon": ("lon", [100.0, 101.0]),
        },
    )

    out = tmp_path / "cf_baseline.nc"
    write_xarray_to_file(ds, out)
    ds_read = xr.open_dataset(out, engine="netcdf4")

    assert ds_read.attrs.get("Conventions") == "CF-1.12"
    assert ds_read["lat"].attrs.get("standard_name") == "latitude"
    assert ds_read["lat"].attrs.get("units") == "degrees_north"
    assert ds_read["lat"].attrs.get("axis") == "Y"
    assert ds_read["lon"].attrs.get("standard_name") == "longitude"
    assert ds_read["lon"].attrs.get("units") == "degrees_east"
    assert ds_read["lon"].attrs.get("axis") == "X"
    assert ds_read["time"].attrs.get("standard_name") == "time"
    assert ds_read["time"].attrs.get("axis") == "T"


def test_write_xarray_to_file_warns_instead_of_crashing_when_metadata_missing(
    tmp_path, caplog
):
    ds = xr.Dataset(
        {"v": (("y", "x"), np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32))},
        coords={"y": ("y", [0.0, 1.0]), "x": ("x", [0.0, 1.0])},
    )

    out = tmp_path / "missing_metadata.nc"
    with caplog.at_level(logging.WARNING, logger="mhm_tools.common.netcdf"):
        write_xarray_to_file(ds, out)

    assert out.is_file()
    warnings = [r.message for r in caplog.records if r.levelno >= logging.WARNING]
    assert any("Could not infer latitude coordinate" in msg for msg in warnings)
    assert any("Could not infer longitude coordinate" in msg for msg in warnings)
    assert any("Could not infer time coordinate" in msg for msg in warnings)
    assert any("has no 'units' attribute" in msg for msg in warnings)


def test_write_xarray_to_file_warns_and_falls_back_if_var_name_missing(
    tmp_path, caplog
):
    ds = _make_ds(dtype=np.float32)
    out = tmp_path / "fallback_varname.nc"

    with caplog.at_level(logging.WARNING, logger="mhm_tools.common.file_handler"):
        write_xarray_to_file(ds, out, var_name="does_not_exist")

    assert out.is_file()
    ds_read = xr.open_dataset(out, engine="netcdf4")
    assert "v" in ds_read.data_vars
    warnings = [r.message for r in caplog.records if r.levelno >= logging.WARNING]
    assert any("Requested var_name" in msg for msg in warnings)
