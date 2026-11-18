"""Tests for streamed gridded LAI NetCDF formatting."""

import inspect
from pathlib import Path

import netCDF4
import numpy as np
import pandas as pd
import pytest
import rasterio
import rasterio.warp
import xarray as xr
from pyproj import Transformer

from mhm_tools.common.netcdf import NetcdfCompression
from mhm_tools.pre import format_lai
from mhm_tools.pre.format_lai import (
    copy_lai_netcdf_to_grid,
    format_lai_netcdf_data,
    format_lai_netcdf_file,
    lai_time_step,
    prepare_lai_temporal,
)


def _series(values, start, frequency):
    return xr.DataArray(
        np.asarray(values, dtype="float64"),
        dims=("time",),
        coords={"time": pd.date_range(start, periods=len(values), freq=frequency)},
    )


def _write_dem(path: Path) -> None:
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        width=6,
        height=4,
        count=1,
        dtype="float32",
        crs="EPSG:4326",
        transform=rasterio.transform.from_origin(99.0, 31.0, 0.5, 0.5),
        nodata=-9999,
    ) as dataset:
        dataset.write(np.ones((4, 6), dtype="float32"), 1)


def _write_lai(path: Path, encoding=None) -> None:
    """Write a small LAI cube to path with optional per-variable encoding."""
    times = pd.date_range("2001-01-01", periods=12, freq="MS")
    values = np.arange(12 * 3 * 2, dtype="float64").reshape(12, 3, 2) / 7 + 0.123456789
    xr.DataArray(
        values,
        dims=("time", "lat", "lon"),
        coords={"time": times, "lat": [30.0, 30.5, 31.0], "lon": [99.5, 101.5]},
        name="lai",
        attrs={"units": "1"},
    ).to_dataset().to_netcdf(path, engine="netcdf4", encoding=encoding)


def _write_projected_dem(path: Path) -> None:
    x, y = Transformer.from_crs("EPSG:4326", "EPSG:32647", always_xy=True).transform(
        99.5, 31.0
    )
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        width=6,
        height=4,
        count=1,
        dtype="float32",
        crs="EPSG:32647",
        transform=rasterio.transform.from_origin(x, y, 20_000, 20_000),
    ) as dataset:
        dataset.write(np.ones((4, 6), dtype="float32"), 1)


def test_temporal_resolution_is_inferred_from_dates():
    """No caller-supplied input resolution is needed for conversion."""
    daily = prepare_lai_temporal(_series(np.arange(31), "2001-01-01", "D"), "monthly")
    annual = prepare_lai_temporal(_series([2.0, 4.0], "2000-01-01", "YS"), "monthly")

    assert daily.data.values.tolist() == [15.0]
    assert annual.data.sizes["time"] == 24
    assert np.all(annual.data.values[:12] == 2.0)
    assert np.all(annual.data.values[12:] == 4.0)
    assert lai_time_step("daily") == -1
    assert lai_time_step("long-term-mean-monthly") == 1


@pytest.mark.parametrize("temporal_resolution", ["long-term-mean-monthly", "monthly"])
def test_format_lai_netcdf_uses_exact_dem_grid(tmp_path: Path, temporal_resolution):
    """The output cube is float64 and exactly matches the DEM matrix."""
    source = tmp_path / "source.nc"
    dem = tmp_path / "dem.tif"
    _write_lai(source)
    _write_dem(dem)

    output = format_lai_netcdf_data(
        source, dem, tmp_path / "output", output_temporal_resolution=temporal_resolution
    )

    assert output == tmp_path / "output" / "lai.nc"
    with xr.open_dataset(output) as dataset:
        assert dataset["lai"].shape == (12, 4, 6)
        assert dataset["lai"].dtype == np.dtype("float64")
        np.testing.assert_allclose(
            dataset["xc"].values, 99.0 + (np.arange(6) + 0.5) * 0.5
        )
        np.testing.assert_allclose(
            dataset["yc"].values, 31.0 - (np.arange(4) + 0.5) * 0.5
        )
        assert np.all(np.isfinite(dataset["lai"].values))
        if temporal_resolution == "long-term-mean-monthly":
            assert dataset["time"].attrs["units"] == "month"
        else:
            np.testing.assert_array_equal(
                dataset["time"].values,
                pd.date_range("2001-01-01", periods=12, freq="MS"),
            )
        assert "bnds_bnds" not in dataset


def test_format_lai_netcdf_warps_to_a_projected_dem(tmp_path: Path):
    """Rasterio handles CRS conversion while retaining the DEM dimensions."""
    source = tmp_path / "source.nc"
    dem = tmp_path / "projected_dem.tif"
    _write_lai(source)
    _write_projected_dem(dem)

    output = format_lai_netcdf_file(source, dem, tmp_path / "lai.nc")

    with xr.open_dataset(output) as dataset:
        assert dataset["lai"].shape == (12, 4, 6)
        assert dataset.attrs["projection"] == "epsg:32647"
        assert dataset["xc"].attrs["standard_name"] == "projection_x_coordinate"
        assert np.any(dataset["lai"].values > 0)


def _monthly(field):
    return np.broadcast_to(field, (12, *field.shape))


@pytest.mark.parametrize(
    ("crs", "mapping", "step"),
    [
        ("EPSG:32647", True, 5000.0),
        ("EPSG:32647", False, 5000.0),
        ("EPSG:4326", True, 0.1),
    ],
)
def test_lai_on_x_y_axes_is_warped_from_its_own_crs(tmp_path: Path, crs, mapping, step):
    """Projected or lon/lat x/y axes onto a lon/lat DEM, the CRS from the grid mapping or input_crs."""
    from pyproj import CRS

    dem = tmp_path / "dem.tif"
    _write_dem(dem)
    to_source = Transformer.from_crs("EPSG:4326", crs, always_xy=True)
    corners_x, corners_y = to_source.transform(
        [98.5, 102.5, 98.5, 102.5], [28.5, 28.5, 31.5, 31.5]
    )
    x = np.arange(min(corners_x), max(corners_x), step)
    y = np.arange(min(corners_y), max(corners_y), step)
    scale = 1.0 if crs == "EPSG:4326" else 1e-3
    lai = xr.DataArray(
        _monthly((x[None, :] + y[:, None]) * scale),
        dims=("time", "y", "x"),
        coords={
            "time": pd.date_range("2001-01-01", periods=12, freq="MS"),
            "y": y,
            "x": x,
        },
        name="lai",
        attrs={"grid_mapping": "crs"} if mapping else {},
    ).to_dataset()
    if mapping:
        lai["crs"] = xr.DataArray(0, attrs={"crs_wkt": CRS(crs).to_wkt()})
    lai.to_netcdf(tmp_path / "lai.nc")

    output = format_lai_netcdf_file(
        tmp_path / "lai.nc",
        dem,
        tmp_path / "out.nc",
        input_crs=None if mapping else crs,
    )

    with xr.open_dataset(output) as result:
        source_x, source_y = to_source.transform(
            *np.meshgrid(result["xc"].values, result["yc"].values)
        )
        np.testing.assert_allclose(
            result["lai"].values[0], (source_x + source_y) * scale, rtol=1e-3
        )


def _write_curvilinear_lai(path: Path) -> None:
    """Write a rotated lat/lon grid holding lon + lat."""
    rows, cols = np.meshgrid(np.arange(50.0), np.arange(60.0), indexing="ij")
    lat = 28.0 + 0.1 * rows + 0.02 * cols
    lon = 98.0 + 0.1 * cols - 0.02 * rows
    xr.Dataset(
        {"lai": (("time", "y", "x"), _monthly(lon + lat))},
        coords={
            "time": pd.date_range("2001-01-01", periods=12, freq="MS"),
            "lat": (("y", "x"), lat, {"standard_name": "latitude"}),
            "lon": (("y", "x"), lon, {"standard_name": "longitude"}),
        },
    ).to_netcdf(path)


def test_lai_on_2d_lat_lon_is_warped_onto_the_dem(tmp_path: Path):
    """A curvilinear (rotated) lat/lon grid goes straight onto the DEM through its geolocation."""
    if "src_geoloc_array" not in inspect.signature(rasterio.warp.reproject).parameters:
        pytest.skip("geolocation warps need rasterio 1.4 or newer")
    dem = tmp_path / "dem.tif"
    _write_dem(dem)
    _write_curvilinear_lai(tmp_path / "lai.nc")

    output = format_lai_netcdf_file(tmp_path / "lai.nc", dem, tmp_path / "out.nc")

    with xr.open_dataset(output) as result:
        centre_lon, centre_lat = np.meshgrid(result["xc"].values, result["yc"].values)
        np.testing.assert_allclose(
            result["lai"].values[0], centre_lon + centre_lat, atol=0.01
        )


def test_curvilinear_lai_on_an_old_rasterio_says_what_it_needs(
    tmp_path: Path, monkeypatch
):
    """Without geolocation warps the 2D case fails clearly; nothing pins rasterio."""
    monkeypatch.setattr(
        rasterio.warp, "reproject", lambda _source, _destination, **_kwargs: None
    )
    dem = tmp_path / "dem.tif"
    _write_dem(dem)
    _write_curvilinear_lai(tmp_path / "lai.nc")

    with pytest.raises(ValueError, match=r"rasterio 1\.4 or newer"):
        format_lai_netcdf_file(tmp_path / "lai.nc", dem, tmp_path / "out.nc")


@pytest.mark.parametrize("during_write", [False, True])
def test_cancelled_lai_write_leaves_no_output(
    tmp_path: Path, monkeypatch, during_write
):
    """Cancellation cleans staging and preserves any existing completed output."""
    source = tmp_path / "source.nc"
    dem = tmp_path / "dem.tif"
    output = tmp_path / "lai.nc"
    _write_lai(source)
    _write_dem(dem)

    writing = False
    shared_write = format_lai.write_xarray_to_file

    def record_write(*args, **kwargs):
        """Mark entry to final writing and return the shared writer result."""
        nonlocal writing
        writing = True
        assert args[0]["lai"].chunks is not None
        return shared_write(*args, **kwargs)

    monkeypatch.setattr(format_lai, "write_xarray_to_file", record_write)
    if during_write:
        output.write_bytes(b"previous output")
    with pytest.raises(RuntimeError, match="cancelled"):
        format_lai_netcdf_file(
            source,
            dem,
            output,
            is_cancelled=lambda: writing if during_write else True,
        )
    if during_write:
        assert output.read_bytes() == b"previous output"
    else:
        assert not output.exists()
    assert not list(tmp_path.glob(".lai_*"))


def test_window_copy_pads_an_expanded_grid_with_zero(tmp_path: Path):
    """Aligned L0 expansion uses valid zero LAI outside the source extent."""
    source = tmp_path / "source.nc"
    dem = tmp_path / "dem.tif"
    staged = tmp_path / "staged.nc"
    _write_lai(source)
    _write_dem(dem)
    format_lai_netcdf_file(
        source,
        dem,
        staged,
        compression=NetcdfCompression(complevel=7, significant_digits=4),
    )
    target = {
        "ncols": 8,
        "nrows": 6,
        "xllcorner": 98.5,
        "yllcorner": 28.5,
        "cellsize": 0.5,
    }

    output = copy_lai_netcdf_to_grid(
        staged,
        tmp_path / "expanded.nc",
        target,
        "EPSG:4326",
        "expanded LAI",
    )

    with netCDF4.Dataset(output) as raw:
        assert raw["lai"].filters()["complevel"] == 7
        assert raw["lai"].quantization() == (4, "BitGroom")
    with xr.open_dataset(output) as dataset:
        assert dataset["lai"].shape == (12, 6, 8)
        assert np.all(dataset["lai"].values[:, 0, :] == 0)
        assert np.all(dataset["lai"].values[:, -1, :] == 0)
        assert np.all(dataset["lai"].values[:, :, 0] == 0)
        assert np.all(dataset["lai"].values[:, :, -1] == 0)


def test_lai_output_guard_checks_disk(monkeypatch, tmp_path):
    """The streamed writer guards disk volume rather than process memory."""
    import shutil

    required = format_lai.lai_grid_byte_size(468, 6120, 13320)
    monkeypatch.setattr(
        shutil,
        "disk_usage",
        lambda _path: shutil._ntuple_diskusage(0, 0, required * 2),
    )
    assert format_lai.assert_lai_output_fits(468, 6120, 13320, tmp_path) == required

    monkeypatch.setattr(
        shutil,
        "disk_usage",
        lambda _path: shutil._ntuple_diskusage(0, 0, 1024),
    )
    with pytest.raises(MemoryError, match="is free on the output volume"):
        format_lai.assert_lai_output_fits(468, 6120, 13320, tmp_path)


def test_lai_inherits_compression_and_quantizes_only_payload(tmp_path):
    """Resampling retains source compression; disabling quantization changes only precision."""
    source = tmp_path / "source.nc"
    dem = tmp_path / "dem.tif"
    _write_lai(
        source,
        encoding={
            "lai": {
                "zlib": True,
                "complevel": 9,
                "shuffle": False,
                "significant_digits": 3,
                "quantize_mode": "BitGroom",
            }
        },
    )
    _write_dem(dem)
    quantized = format_lai_netcdf_file(source, dem, tmp_path / "quantized.nc")
    full_precision = format_lai_netcdf_file(
        source,
        dem,
        tmp_path / "full.nc",
        compression=NetcdfCompression(
            complevel=None, shuffle=None, significant_digits=0
        ),
    )
    with netCDF4.Dataset(quantized) as raw, netCDF4.Dataset(full_precision) as full:
        assert raw["lai"].filters()["complevel"] == 9
        assert raw["lai"].filters()["shuffle"] is False
        assert raw["lai"].quantization() == (3, "BitGroom")
        assert full["lai"].filters()["complevel"] == 9
        assert full["lai"].quantization() is None
        for name in ("lat", "lon", "xc", "yc", "time", "time_bnds"):
            assert raw[name].quantization() is None
            np.testing.assert_array_equal(raw[name][:], full[name][:])
        np.testing.assert_allclose(raw["lai"][:], full["lai"][:], rtol=1e-3)
        assert np.any(raw["lai"][:] != full["lai"][:])


def test_lai_disk_guard_accounts_for_uncompressed_output(monkeypatch, tmp_path):
    """Free space sufficient for compressed output may not fit an uncompressed write."""
    import shutil

    total_values_bytes = (12 + 2) * 4 * 6 * 8
    free = int(total_values_bytes * 1.8)
    monkeypatch.setattr(
        shutil, "disk_usage", lambda _path: shutil._ntuple_diskusage(0, 0, free)
    )
    format_lai.assert_lai_output_fits(12, 4, 6, tmp_path)
    with pytest.raises(MemoryError, match="including staging"):
        format_lai.assert_lai_output_fits(
            12, 4, 6, tmp_path, compression=NetcdfCompression(complevel=0)
        )


def _write_lai_at(path: Path, times, values=None, encoding=None) -> None:
    """Write an LAI cube on the grid of ``_write_lai`` at the given times.

    Args:
        path (Path): Output NetCDF file.
        times (array-like): Time stamps of the cube.
        values (numpy.ndarray): LAI of shape (time, 3, 2); 1.0 by default.
        encoding (dict): Encoding of the ``lai`` variable, or None.
    """
    values = np.ones((len(times), 3, 2)) if values is None else values
    xr.DataArray(
        values,
        dims=("time", "lat", "lon"),
        coords={"time": times, "lat": [30.0, 30.5, 31.0], "lon": [99.5, 101.5]},
        name="lai",
    ).to_dataset().to_netcdf(path, encoding={"lai": encoding or {}})


def test_mid_month_stamps_fill_their_whole_month():
    """Monthly LAI stamped on the 15th covers its month from the 1st on."""
    times = pd.to_datetime(["2001-01-15", "2001-02-15", "2001-03-15"])
    monthly = xr.DataArray([1.0, 2.0, 3.0], dims=("time",), coords={"time": times})

    daily = prepare_lai_temporal(monthly, "daily").data

    assert pd.Timestamp(daily["time"].values[0]) == pd.Timestamp("2001-01-01")
    assert daily.sizes["time"] == 31 + 28 + 31
    assert daily.sel(time="2001-02-10").item() == 2.0


def test_float32_axes_count_as_regular():
    """Float32 rounding near 180 degrees must not make a 1/12 degree axis irregular."""
    axis = (80.0 + (np.arange(1200) + 0.5) / 12).astype("float32")

    assert format_lai._axis_step(axis, "longitude") == pytest.approx(1 / 12, rel=1e-4)


def test_lai_variable_is_preferred_over_other_cubes():
    """A file holding Fpar and Lai uses Lai, and two unnamed cubes need a choice."""
    cube = xr.DataArray(np.zeros((2, 2, 2)), dims=("time", "lat", "lon"))
    dataset = xr.Dataset({"Fpar_500m": cube, "Lai_500m": cube})

    assert format_lai.find_variable(dataset, None, "lat", "lon").name == "Lai_500m"
    with pytest.raises(ValueError, match="--source-variable"):
        format_lai.find_variable(
            dataset.rename({"Lai_500m": "other"}), None, "lat", "lon"
        )


def test_a_month_without_data_raises_instead_of_lai_zero(tmp_path: Path):
    """Daily LAI without March cannot give a monthly March."""
    times = pd.date_range("2001-01-01", "2001-05-31", freq="D")
    times = times[times.month != 3]
    _write_lai_at(tmp_path / "lai.nc", times)
    _write_dem(tmp_path / "dem.tif")

    with pytest.raises(ValueError, match="no data at 1 output time step"):
        format_lai_netcdf_file(
            tmp_path / "lai.nc",
            tmp_path / "dem.tif",
            tmp_path / "out.nc",
            output_temporal_resolution="monthly",
        )


def test_noleap_calendar_keeps_the_real_months():
    """A noleap October-September year is averaged by its real months."""
    times = xr.date_range(
        "2000-10-01", periods=12, freq="MS", calendar="noleap", use_cftime=True
    )
    data = xr.DataArray(np.arange(12.0), dims=("time",), coords={"time": times})

    monthly = prepare_lai_temporal(data, "long-term-mean-monthly").data

    assert monthly.sel(time=10).item() == 0.0
    assert monthly.sel(time=1).item() == 3.0


def test_packed_codes_outside_valid_range_are_not_lai(tmp_path: Path):
    """MODIS-like fill code 254 with valid_range 0..100 must not become LAI 25.4."""
    values = np.full((12, 3, 2), 1.0)
    values[:, 0, 0] = 25.4
    _write_lai_at(
        tmp_path / "lai.nc",
        pd.date_range("2001-01-01", periods=12, freq="MS"),
        values,
        encoding={"dtype": "int16", "scale_factor": 0.1},
    )
    with netCDF4.Dataset(tmp_path / "lai.nc", "a") as dataset:
        dataset["lai"].setncattr("valid_range", np.array([0, 100], dtype="int16"))
    _write_dem(tmp_path / "dem.tif")

    output = format_lai_netcdf_file(
        tmp_path / "lai.nc", tmp_path / "dem.tif", tmp_path / "out.nc"
    )

    with xr.open_dataset(output) as result:
        assert np.nanmax(result["lai"].values) == pytest.approx(1.0)


def test_projected_axes_without_grid_mapping_need_an_input_crs(tmp_path: Path):
    """x/y in metres without a grid mapping must not be read as degrees."""
    xr.DataArray(
        np.ones((12, 3, 4)),
        dims=("time", "y", "x"),
        coords={
            "time": pd.date_range("2001-01-01", periods=12, freq="MS"),
            "y": 3300000.0 + 5000.0 * np.arange(3),
            "x": 500000.0 + 5000.0 * np.arange(4),
        },
        name="lai",
    ).to_dataset().to_netcdf(tmp_path / "lai.nc")
    _write_dem(tmp_path / "dem.tif")

    with pytest.raises(ValueError, match="--input-crs"):
        format_lai_netcdf_file(
            tmp_path / "lai.nc", tmp_path / "dem.tif", tmp_path / "out.nc"
        )


def test_global_lai_is_cut_to_the_dem_before_aggregation(tmp_path: Path, monkeypatch):
    """Only the source cells around a small DEM reach the time aggregation."""
    xr.DataArray(
        np.ones((12, 180, 360)),
        dims=("time", "lat", "lon"),
        coords={
            "time": pd.date_range("2001-01-01", periods=12, freq="MS"),
            "lat": np.arange(-89.5, 90),
            "lon": np.arange(-179.5, 180),
        },
        name="lai",
    ).to_dataset().to_netcdf(tmp_path / "lai.nc")
    _write_dem(tmp_path / "dem.tif")
    sizes = []

    def record_sizes(data, *args, **kwargs):
        """Record the spatial size of the cube passed to the aggregation."""
        sizes.append((data.sizes["lat"], data.sizes["lon"]))
        return prepare_lai_temporal(data, *args, **kwargs)

    monkeypatch.setattr(format_lai, "prepare_lai_temporal", record_sizes)
    format_lai_netcdf_file(
        tmp_path / "lai.nc", tmp_path / "dem.tif", tmp_path / "out.nc"
    )

    # the 0.5 degree DEM spans 2 x 3 degrees, plus one DEM cell and 2 LAI cells
    assert sizes and sizes[0][0] < 10 and sizes[0][1] < 10
