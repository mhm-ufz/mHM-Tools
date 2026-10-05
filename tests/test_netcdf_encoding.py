import logging

import netCDF4
import numpy as np
import pytest
import xarray as xr

from mhm_tools.common.file_handler import write_xarray_to_file, write_xarray_to_netcdf
from mhm_tools.common.netcdf import (
    COMPRESSION_ENCODING_KEYS,
    DEFAULT_COMPLEVEL,
    MIN_COORD_COMPRESSION_VALUES,
    NO_QUANTIZATION,
    QUANTIZATION_ENCODING_KEYS,
    QUANTIZE_MODES,
    NetcdfCompression,
    apply_cf_baseline_metadata,
    create_quantization_encoding,
    get_netcdf_metadata_data_vars,
    is_coordinate_like_variable,
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


def _make_field(nt=4, ny=40, nx=50, dtype="float32"):
    """Build a smooth float field with a couple of NaN rows."""
    lat = np.linspace(47.0, 55.0, ny)
    plane = (np.cos(np.deg2rad(lat))[:, None] * np.ones(nx)[None, :] * 10).astype(dtype)
    data = np.stack([plane] * nt)
    return xr.Dataset(
        {"v": (("time", "lat", "lon"), data)},
        coords={
            "time": np.arange(nt),
            "lat": lat,
            "lon": np.linspace(5.0, 15.0, nx),
        },
    )


def _write_source(path, complevel=None, significant_digits=None, mode="BitGroom"):
    """Write an input file with a chosen compression, bypassing the writers."""
    encoding = {}
    if complevel is not None:
        encoding.update({"zlib": complevel > 0, "complevel": complevel})
    if significant_digits is not None:
        encoding.update(
            {"significant_digits": significant_digits, "quantize_mode": mode}
        )
    _make_field().to_netcdf(
        path, engine="netcdf4", encoding={"v": encoding} if encoding else None
    )
    return path


def _on_disk(path, name="v"):
    """Return the complevel, quantization and _Quantize* attrs of a variable."""
    with netCDF4.Dataset(path) as dataset:
        variable = dataset[name]
        return (
            variable.filters()["complevel"],
            variable.quantization(),
            [a for a in variable.ncattrs() if a.startswith("_Quantize")],
        )


# --- coordinates ------------------------------------------------------------


@pytest.mark.parametrize("complevel", [4, 9])
def test_sanitize_coordinate_encoding_never_compresses_a_1d_coordinate(complevel):
    """Leave one-dimensional axes uncompressed, whatever level is requested."""
    ds = _make_ds(dtype=np.float32)

    sanitize_coordinate_encoding(ds, NetcdfCompression(complevel=complevel))

    for coord in ("lat", "lon"):
        assert not COMPRESSION_ENCODING_KEYS & set(ds[coord].encoding)


def test_sanitize_coordinate_encoding_compresses_a_large_2d_coordinate():
    """Compress a two-dimensional coordinate that is payload sized."""
    side = int(np.ceil(np.sqrt(MIN_COORD_COMPRESSION_VALUES))) + 1
    ds = xr.Dataset(
        coords={"lat2d": (("y", "x"), np.zeros((side, side), dtype="float32"))}
    )

    sanitize_coordinate_encoding(ds, NetcdfCompression(complevel=9))

    assert ds["lat2d"].encoding["complevel"] == 9
    assert ds["lat2d"].encoding["zlib"] is True


def test_sanitize_coordinate_encoding_skips_a_small_2d_coordinate():
    """Leave a bounds sized two-dimensional coordinate uncompressed."""
    ds = xr.Dataset(coords={"lat_bnds": (("lat", "bnds"), np.zeros((2, 2)))})

    sanitize_coordinate_encoding(ds, NetcdfCompression(complevel=9))

    assert not COMPRESSION_ENCODING_KEYS & set(ds["lat_bnds"].encoding)


def test_sanitize_coordinate_encoding_never_quantizes_a_coordinate():
    """Keep every coordinate free of lossy quantization keys."""
    side = int(np.ceil(np.sqrt(MIN_COORD_COMPRESSION_VALUES))) + 1
    ds = xr.Dataset(
        coords={"lat2d": (("y", "x"), np.zeros((side, side), dtype="float32"))}
    )

    sanitize_coordinate_encoding(
        ds, NetcdfCompression(complevel=9, significant_digits=3)
    )

    assert not QUANTIZATION_ENCODING_KEYS & set(ds["lat2d"].encoding)


def test_sanitize_coordinate_encoding_without_compression_leaves_coordinates_bare():
    """Add no compression to coordinates when no settings are given."""
    side = int(np.ceil(np.sqrt(MIN_COORD_COMPRESSION_VALUES))) + 1
    ds = xr.Dataset(
        coords={"lat2d": (("y", "x"), np.zeros((side, side), dtype="float32"))}
    )

    sanitize_coordinate_encoding(ds)

    assert not COMPRESSION_ENCODING_KEYS & set(ds["lat2d"].encoding)


@pytest.mark.parametrize(
    ("name", "attrs", "expected"),
    [
        ("x", {}, True),
        ("y", {}, True),
        ("geo_x", {}, True),
        ("gauge_position", {"standard_name": "latitude"}, True),
        ("northing_m", {"standard_name": "projection_x_coordinate"}, True),
        ("some_angle", {"units": "degrees_east"}, True),
        ("discharge", {"units": "m3 s-1"}, False),
        ("interception", {}, False),
    ],
)
def test_is_coordinate_like_variable_classifies_by_name_and_metadata(
    name, attrs, expected
):
    """Detect a variable that holds coordinates rather than payload data."""
    ds = xr.Dataset({name: (("id",), np.zeros(3))})
    ds[name].attrs.update(attrs)

    assert is_coordinate_like_variable(ds, name) is expected


def test_is_coordinate_like_variable_detects_a_bounds_variable():
    """Treat a variable referenced as coordinate bounds as coordinate data."""
    ds = _make_ds(dtype=np.float32)
    ds["lat"].attrs["bounds"] = "lat_bnds"
    ds["lat_bnds"] = (("lat", "bnds"), np.zeros((2, 2)))

    assert is_coordinate_like_variable(ds, "lat_bnds") is True


def test_create_quantization_encoding_skips_coordinate_like_data_variables():
    """Quantize the payload but never the gauge positions beside it."""
    ds = xr.Dataset(
        {
            "discharge": (("id",), np.array([1.0, 2.0, 3.0])),
            "x": (("id",), np.array([10.0, 11.0, 12.0])),
            "y": (("id",), np.array([50.0, 51.0, 52.0])),
        }
    )

    encoding = create_quantization_encoding(ds, ["discharge", "x", "y"], 4)

    assert set(encoding) == {"discharge"}
    assert encoding["discharge"]["significant_digits"] == 4


# --- pre-existing compression, lossless -------------------------------------


@pytest.mark.parametrize(
    ("source_complevel", "requested", "expected"),
    [
        (9, None, 9),
        (9, 1, 1),
        (0, None, 0),
        (0, 1, 1),
    ],
)
def test_write_keeps_or_overrides_the_input_complevel(
    tmp_path, source_complevel, requested, expected
):
    """Keep an input file's level unless a level was explicitly requested."""
    source = _write_source(tmp_path / "src.nc", complevel=source_complevel)
    compression = None if requested is None else NetcdfCompression(complevel=requested)

    out = tmp_path / "out.nc"
    write_xarray_to_file(xr.open_dataset(source), out, compression=compression)

    assert _on_disk(out)[0] == expected


def test_write_falls_back_to_the_default_level_without_an_input_to_inherit(tmp_path):
    """Use the default level for a dataset that carries no encoding."""
    out = tmp_path / "fresh.nc"
    write_xarray_to_file(_make_field(), out, compression=None)

    assert _on_disk(out)[0] == DEFAULT_COMPLEVEL
    with xr.open_dataset(out) as dataset:
        np.testing.assert_array_equal(dataset["time"].values, np.arange(4))


def test_write_level_beats_a_caller_encoding_while_keeping_its_fill_value(tmp_path):
    """Let the settings own compression and the caller own the fill value."""
    out = tmp_path / "enc.nc"
    write_xarray_to_file(
        _make_field(),
        out,
        encoding={"v": {"zlib": True, "complevel": 1, "_FillValue": -9999.0}},
        compression=NetcdfCompression(complevel=9),
    )

    assert _on_disk(out)[0] == 9
    with netCDF4.Dataset(out) as dataset:
        assert dataset["v"]._FillValue == np.float32(-9999.0)


# --- pre-existing compression, lossy ----------------------------------------


def test_quantization_travels_as_an_attribute_on_read(tmp_path):
    """Read a quantized file back as the attribute netcdf-c records."""
    source = _write_source(
        tmp_path / "q.nc", complevel=9, significant_digits=3, mode="GranularBitRound"
    )

    ds = xr.open_dataset(source)

    assert [a for a in ds["v"].attrs if a.startswith("_Quantize")] == [
        "_QuantizeGranularBitRoundNumberOfSignificantDigits"
    ]


@pytest.mark.parametrize("reader", [xr.open_dataset, xr.load_dataset])
def test_write_keeps_an_input_quantization_without_settings(tmp_path, reader):
    """Carry an input file's quantization over whichever reader opened it."""
    source = _write_source(
        tmp_path / "q.nc", complevel=9, significant_digits=3, mode="GranularBitRound"
    )

    out = tmp_path / "out.nc"
    write_xarray_to_file(reader(source), out, compression=None)

    complevel, quantization, attrs = _on_disk(out)
    assert quantization == (3, "GranularBitRound")
    assert complevel == 9
    assert len(attrs) == 1


def test_write_level_only_keeps_the_inherited_quantization(tmp_path):
    """Change the level without disturbing an inherited precision."""
    source = _write_source(
        tmp_path / "q.nc", complevel=9, significant_digits=3, mode="GranularBitRound"
    )

    out = tmp_path / "out.nc"
    write_xarray_to_file(
        xr.open_dataset(source), out, compression=NetcdfCompression(complevel=1)
    )

    complevel, quantization, _attrs = _on_disk(out)
    assert complevel == 1
    assert quantization == (3, "GranularBitRound")


def test_write_significant_digits_replaces_the_inherited_quantization(tmp_path):
    """Leave exactly one quantization on the file when it is overridden."""
    source = _write_source(
        tmp_path / "q.nc", complevel=9, significant_digits=3, mode="GranularBitRound"
    )

    out = tmp_path / "out.nc"
    write_xarray_to_file(
        xr.open_dataset(source),
        out,
        compression=NetcdfCompression(complevel=9, significant_digits=5),
    )

    _complevel, quantization, attrs = _on_disk(out)
    assert quantization == (5, "BitGroom")
    assert len(attrs) == 1


def test_write_no_quantization_removes_an_inherited_quantization(tmp_path):
    """Write full precision when quantization is switched off explicitly."""
    source = _write_source(
        tmp_path / "q.nc", complevel=9, significant_digits=3, mode="GranularBitRound"
    )

    out = tmp_path / "out.nc"
    write_xarray_to_file(
        xr.open_dataset(source),
        out,
        compression=NetcdfCompression(complevel=9, significant_digits=NO_QUANTIZATION),
    )

    complevel, quantization, attrs = _on_disk(out)
    assert quantization is None
    assert attrs == []
    assert complevel == 9


def test_write_no_quantization_is_harmless_without_one_to_remove(tmp_path):
    """Switching quantization off on a plain input changes nothing but stays off."""
    source = _write_source(tmp_path / "plain.nc", complevel=4)

    out = tmp_path / "out.nc"
    write_xarray_to_file(
        xr.open_dataset(source),
        out,
        compression=NetcdfCompression(significant_digits=NO_QUANTIZATION),
    )

    complevel, quantization, attrs = _on_disk(out)
    assert quantization is None
    assert attrs == []
    assert complevel == DEFAULT_COMPLEVEL


# --- lossy and lossless application -----------------------------------------


@pytest.mark.parametrize("mode", list(QUANTIZE_MODES))
def test_write_applies_each_quantize_mode(tmp_path, mode):
    """Record the requested mode and precision on the written variable."""
    digits = 9 if mode == "BitRound" else 3
    out = tmp_path / "q.nc"
    write_xarray_to_file(
        _make_field(),
        out,
        compression=NetcdfCompression(significant_digits=digits, quantize_mode=mode),
    )

    assert _on_disk(out)[1] == (digits, mode)


def test_write_does_not_quantize_an_integer_variable(tmp_path):
    """Skip quantization for integers, which netcdf-c refuses."""
    ds = xr.Dataset({"flag": (("y", "x"), np.ones((20, 20), dtype="int8"))})

    out = tmp_path / "int.nc"
    write_xarray_to_file(ds, out, compression=NetcdfCompression(significant_digits=4))

    with netCDF4.Dataset(out) as dataset:
        assert dataset["flag"].quantization() is None


@pytest.mark.parametrize(("complevel", "zlib"), [(0, False), (4, True), (9, True)])
def test_get_lossless_encoding_matches_the_written_file(tmp_path, complevel, zlib):
    """Report the zlib flag the file ends up carrying."""
    compression = NetcdfCompression(complevel=complevel)
    assert compression.get_lossless_encoding()["zlib"] is zlib

    out = tmp_path / "lossless.nc"
    write_xarray_to_file(_make_field(), out, compression=compression)

    with netCDF4.Dataset(out) as dataset:
        assert dataset["v"].filters()["zlib"] is zlib
        assert dataset["v"].filters()["complevel"] == complevel


def test_bitgroom_keeps_the_round_trip_error_within_four_digits(tmp_path):
    """Hold the relative error of four significant digits below 1e-4."""
    ds = _make_field()
    out = tmp_path / "bg.nc"
    write_xarray_to_file(ds, out, compression=NetcdfCompression(significant_digits=4))

    written = xr.open_dataset(out)["v"].values.astype("float64")
    original = ds["v"].values.astype("float64")
    finite = np.isfinite(original) & np.isfinite(written)
    error = np.abs(written[finite] - original[finite]) / np.abs(original[finite]).max()

    assert error.max() < 1e-4


def test_write_warns_and_drops_quantization_for_another_engine(tmp_path, caplog):
    """Refuse to pretend a non-netcdf4 engine applied the quantization."""
    out = tmp_path / "h5.nc"
    with caplog.at_level(logging.WARNING, logger="mhm_tools.common.file_handler"):
        write_xarray_to_file(
            _make_field(),
            out,
            engine="h5netcdf",
            compression=NetcdfCompression(significant_digits=4),
        )

    assert "does not support it" in caplog.text
    with netCDF4.Dataset(out) as dataset:
        assert dataset["v"].quantization() is None


def test_write_drops_an_inherited_quantization_for_another_engine(tmp_path):
    """Keep an inherited quantization out of a file another engine writes."""
    source = _write_source(tmp_path / "q.nc", complevel=4, significant_digits=3)

    out = tmp_path / "h5.nc"
    write_xarray_to_file(xr.open_dataset(source), out, engine="h5netcdf")

    with netCDF4.Dataset(out) as dataset:
        assert dataset["v"].quantization() is None


@pytest.mark.parametrize(
    "kwargs",
    [
        {"complevel": 12},
        {"complevel": -1},
        {"quantize_mode": "nonsense"},
        {"significant_digits": -1},
    ],
)
def test_netcdf_compression_rejects_invalid_settings(kwargs):
    """Fail on a bad setting before any data is read."""
    with pytest.raises(ValueError):
        NetcdfCompression(**kwargs)


def test_netcdf_compression_accepts_no_quantization():
    """Treat zero significant digits as a valid request for full precision."""
    assert NetcdfCompression(significant_digits=NO_QUANTIZATION).significant_digits == 0


def test_integer_netcdf_preparation_keeps_delayed_payload_lazy():
    """Preparing integer encoding must not evaluate the delayed payload."""
    import dask.array as da
    from dask import delayed

    @delayed
    def unexpected_read():
        """Fail if the writer computes the payload during preparation."""
        msg = "Integer payload was loaded before writing"
        raise AssertionError(msg)

    data = da.from_delayed(unexpected_read(), shape=(2, 2), dtype="int32")
    dataset = xr.Dataset({"classes": (("y", "x"), data)})
    prepared, encoding = prepare_dataset_for_netcdf_write(
        dataset, ["classes"], {"classes": {"_FillValue": -9999}}
    )
    assert prepared["classes"].chunks is not None
    assert prepared["classes"].dtype == np.dtype("int32")
    assert encoding["classes"]["_FillValue"] == -9999


@pytest.mark.parametrize(
    ("arguments", "digits", "shuffle"),
    [
        (["--significant-digits", "4"], 4, True),
        (["--significant-digits", "0"], None, True),
        (["--no-shuffle"], 3, False),
    ],
)
def test_cli_compression_overrides_only_requested_fields(
    tmp_path, arguments, digits, shuffle
):
    """Precision and shuffle options preserve each variable's input level."""
    import argparse

    from mhm_tools.common.cli_utils import (
        add_netcdf_compression_args,
        get_netcdf_compression,
    )

    parser = argparse.ArgumentParser()
    add_netcdf_compression_args(parser)
    compression = get_netcdf_compression(parser.parse_args(arguments))
    dataset = _make_field()
    dataset["second"] = dataset["v"].copy(deep=False)
    for name, level in (("v", 9), ("second", 1)):
        dataset[name].encoding.update(zlib=True, complevel=level, shuffle=True)
        dataset[name].attrs["_QuantizeBitGroomNumberOfSignificantDigits"] = 3
    output = tmp_path / "partial.nc"
    write_xarray_to_file(dataset, output, compression=compression)

    with netCDF4.Dataset(output) as result:
        for name, level in (("v", 9), ("second", 1)):
            assert result[name].filters()["complevel"] == level
            assert result[name].filters()["shuffle"] is shuffle
            assert result[name].quantization() == (
                None if digits is None else (digits, "BitGroom")
            )


def test_direct_compression_helpers_resolve_optional_settings():
    """Direct writer callers get concrete settings with or without input encoding."""
    dataset = _make_field()
    dataset["v"].encoding.update(zlib=True, complevel=9, shuffle=False)
    compression = NetcdfCompression(complevel=None, shuffle=None, significant_digits=4)
    assert compression.get_lossless_encoding() == {
        "zlib": True,
        "complevel": 4,
        "shuffle": True,
    }
    assert compression.create_variable_encoding(dataset, ["v"])["v"] == {
        "zlib": True,
        "complevel": 9,
        "shuffle": False,
        "significant_digits": 4,
        "quantize_mode": "BitGroom",
    }
