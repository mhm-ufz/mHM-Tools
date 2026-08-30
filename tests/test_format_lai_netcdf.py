"""Tests for streamed gridded LAI NetCDF formatting."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import rasterio
import xarray as xr
from pyproj import Transformer

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


def _write_lai(path: Path) -> None:
    times = pd.date_range("2001-01-01", periods=12, freq="MS")
    values = np.arange(12 * 3 * 2, dtype="float64").reshape(12, 3, 2)
    xr.DataArray(
        values,
        dims=("time", "lat", "lon"),
        coords={"time": times, "lat": [30.0, 30.5, 31.0], "lon": [99.5, 101.5]},
        name="lai",
        attrs={"units": "1"},
    ).to_dataset().to_netcdf(path)


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


def test_format_lai_netcdf_uses_exact_dem_grid(tmp_path: Path):
    """The output cube is float64 and exactly matches the DEM matrix."""
    source = tmp_path / "source.nc"
    dem = tmp_path / "dem.tif"
    _write_lai(source)
    _write_dem(dem)

    output = format_lai_netcdf_data(source, dem, tmp_path / "output")

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
        assert np.any(dataset["lai"].values > 0)


def test_cancelled_lai_write_leaves_no_output(tmp_path: Path):
    """Cancellation removes the temporary output."""
    source = tmp_path / "source.nc"
    dem = tmp_path / "dem.tif"
    output = tmp_path / "lai.nc"
    _write_lai(source)
    _write_dem(dem)

    with pytest.raises(RuntimeError, match="cancelled"):
        format_lai_netcdf_file(
            source,
            dem,
            output,
            is_cancelled=lambda: True,
        )
    assert not output.exists()


def test_window_copy_pads_an_expanded_grid_with_zero(tmp_path: Path):
    """Aligned L0 expansion uses valid zero LAI outside the source extent."""
    source = tmp_path / "source.nc"
    dem = tmp_path / "dem.tif"
    staged = tmp_path / "staged.nc"
    _write_lai(source)
    _write_dem(dem)
    format_lai_netcdf_file(source, dem, staged)
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

    with xr.open_dataset(output) as dataset:
        assert dataset["lai"].shape == (12, 6, 8)
        assert np.all(dataset["lai"].values[:, 0, :] == 0)
        assert np.all(dataset["lai"].values[:, -1, :] == 0)
        assert np.all(dataset["lai"].values[:, :, 0] == 0)
        assert np.all(dataset["lai"].values[:, :, -1] == 0)


def test_lai_output_guard_checks_disk_and_optional_budget(monkeypatch, tmp_path):
    """The streamed writer guards disk volume rather than process memory."""
    import shutil

    required = format_lai.lai_grid_byte_size(468, 6120, 13320)
    monkeypatch.delenv(format_lai.LAI_MAX_BYTES_ENV, raising=False)
    monkeypatch.setattr(
        shutil,
        "disk_usage",
        lambda _path: shutil._ntuple_diskusage(0, 0, required),
    )
    assert format_lai.assert_lai_output_fits(468, 6120, 13320, tmp_path) == required

    monkeypatch.setenv(format_lai.LAI_MAX_BYTES_ENV, str(16 * 1024**3))
    with pytest.raises(MemoryError, match="over the"):
        format_lai.assert_lai_output_fits(468, 6120, 13320, tmp_path)
