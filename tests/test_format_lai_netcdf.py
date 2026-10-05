"""Tests for streamed gridded LAI NetCDF formatting."""

from pathlib import Path

import netCDF4
import numpy as np
import pandas as pd
import pytest
import rasterio
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
