import numpy as np
import xarray as xr

from mhm_tools.post.gridded_data_evaluation import (
    infer_time_resolution_hours_from_files,
    normalize_time_axis,
    regridd_to_higher_spatial_resolution,
)


def _write_timeseries_nc(path, times):
    ds = xr.Dataset(
        {"v": ("time", np.arange(len(times), dtype=float))}, coords={"time": times}
    )
    ds.to_netcdf(path)


def test_infer_time_resolution_hours_from_files_prefers_intra_file(tmp_path):
    times = np.array(["2024-01-01T00", "2024-01-01T03"], dtype="datetime64[ns]")
    f1 = tmp_path / "a.nc"
    f2 = tmp_path / "b.nc"
    _write_timeseries_nc(f1, times)
    _write_timeseries_nc(f2, times + np.timedelta64(1, "D"))

    inferred = infer_time_resolution_hours_from_files([f1, f2])
    assert inferred == 3.0


def test_infer_time_resolution_hours_from_files_uses_start_times(tmp_path):
    f1 = tmp_path / "a.nc"
    f2 = tmp_path / "b.nc"
    _write_timeseries_nc(f1, np.array(["2024-01-01T00"], dtype="datetime64[ns]"))
    _write_timeseries_nc(f2, np.array(["2024-01-01T06"], dtype="datetime64[ns]"))

    inferred = infer_time_resolution_hours_from_files([f1, f2])
    assert inferred == 6.0


def test_normalize_time_axis_hourly_floor():
    times = np.array(["2024-01-01T00:30", "2024-01-01T01:45"], dtype="datetime64[ns]")
    ds = xr.Dataset({"v": ("time", [1.0, 2.0])}, coords={"time": times})

    out = normalize_time_axis(ds, "H")
    expected = times.astype("datetime64[h]").astype("datetime64[ns]")
    np.testing.assert_array_equal(out.time.values, expected)


def test_normalize_time_axis_daily_floor():
    times = np.array(["2024-01-01T12:30", "2024-01-02T01:45"], dtype="datetime64[ns]")
    ds = xr.Dataset({"v": ("time", [1.0, 2.0])}, coords={"time": times})

    out = normalize_time_axis(ds, "D")
    expected = times.astype("datetime64[D]").astype("datetime64[ns]")
    np.testing.assert_array_equal(out.time.values, expected)


def test_normalize_time_axis_monthly_end_groups_by_month():
    times = np.array(
        ["2024-01-15", "2024-01-20", "2024-02-01", "2024-02-18"],
        dtype="datetime64[ns]",
    )
    ds = xr.Dataset({"v": ("time", [1.0, 2.0, 3.0, 4.0])}, coords={"time": times})

    out = normalize_time_axis(ds, "ME")
    out_times = out.time.values
    months = out_times.astype("datetime64[M]")
    for month in np.unique(months):
        mask = months == month
        assert np.unique(out_times[mask]).size == 1


def test_regridd_to_higher_spatial_resolution_uses_coarse_tolerance():
    lat_fine = np.array([0.49, 0.74, 0.99])
    lon_fine = np.array([0.49, 0.74, 0.99])
    lat_coarse = np.array([0.0, 1.0])
    lon_coarse = np.array([0.0, 1.0])

    ds_fine = xr.Dataset(
        {"v": (("lat", "lon"), np.ones((lat_fine.size, lon_fine.size)))},
        coords={"lat": lat_fine, "lon": lon_fine},
    )
    ds_coarse = xr.Dataset(
        {"v": (("lat", "lon"), np.full((lat_coarse.size, lon_coarse.size), 2.0))},
        coords={"lat": lat_coarse, "lon": lon_coarse},
    )

    out1, out2 = regridd_to_higher_spatial_resolution(ds_fine, ds_coarse)

    assert out2["v"].shape == ds_fine["v"].shape
    assert np.isfinite(out2["v"].values).all()
    print(out2["v"].values)
    assert np.allclose(out2["v"].values, 2.0)
