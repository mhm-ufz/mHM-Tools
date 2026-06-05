import numpy as np
import xarray as xr

from mhm_tools.post.gridded_data_evaluation import (
    compare_input_with_ref,
    crop_datasets_to_spatial_overlap,
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


def test_crop_datasets_to_spatial_overlap_preserves_overlap_and_regrids(caplog):
    lat_input = np.array([0.2, 0.4, 0.6])
    lon_input = np.array([0.2, 0.4, 0.6])
    lat_ref = np.array([0.2, 0.6, 1.0])
    lon_ref = np.array([0.2, 0.6, 1.0])

    input_ds = xr.Dataset(
        {"mean": (("lat", "lon"), np.ones((lat_input.size, lon_input.size)))},
        coords={"lat": lat_input, "lon": lon_input},
    )
    ref_ds = xr.Dataset(
        {"mean": (("lat", "lon"), np.full((lat_ref.size, lon_ref.size), 2.0))},
        coords={"lat": lat_ref, "lon": lon_ref},
    )

    caplog.set_level("INFO")
    cropped_input, cropped_ref = crop_datasets_to_spatial_overlap(
        input_ds, ref_ds, input_name="input", ref_name="ref"
    )

    assert np.array_equal(cropped_input["lat"].values, lat_input)
    assert np.array_equal(cropped_input["lon"].values, lon_input)
    assert np.array_equal(cropped_ref["lat"].values, np.array([0.2, 0.6]))
    assert np.array_equal(cropped_ref["lon"].values, np.array([0.2, 0.6]))
    assert "Input spatial extent is a subset of reference" in caplog.text

    out_input, out_ref = regridd_to_higher_spatial_resolution(cropped_input, cropped_ref)
    assert out_input["mean"].shape == out_ref["mean"].shape
    assert out_input["mean"].shape == cropped_input["mean"].shape
    assert np.allclose(out_ref["mean"].values, 2.0)


def test_compare_input_with_ref_keeps_rel_fields_as_dataarrays(monkeypatch, tmp_path):
    lat = np.array([51.0, 50.0])
    lon = np.array([10.0, 11.0])
    month = np.arange(1, 13)

    input_stats = xr.Dataset(
        {
            "mean": (("lat", "lon"), np.array([[2.0, 4.0], [6.0, 8.0]])),
            "std": (("lat", "lon"), np.array([[1.0, 2.0], [3.0, 4.0]])),
            "clim": (("month", "lat", "lon"), np.ones((12, 2, 2))),
        },
        coords={"month": month, "lat": lat, "lon": lon},
    )
    ref_stats = xr.Dataset(
        {
            "mean": (("lat", "lon"), np.array([[1.0, 0.0], [3.0, 4.0]])),
            "std": (("lat", "lon"), np.array([[2.0, 0.0], [1.5, 2.0]])),
            "clim": (("month", "lat", "lon"), np.ones((12, 2, 2))),
        },
        coords={"month": month, "lat": lat, "lon": lon},
    )

    call_count = {"n": 0}

    def fake_get_stats(**_kwargs):
        call_count["n"] += 1
        return input_stats if call_count["n"] == 1 else ref_stats

    captured = {}

    def fake_write_xarray_to_file(ds, file_path):
        captured["ds"] = ds.copy(deep=True)
        captured["file_path"] = file_path

    monkeypatch.setattr(
        "mhm_tools.post.gridded_data_evaluation.get_stats",
        fake_get_stats,
    )
    monkeypatch.setattr(
        "mhm_tools.post.gridded_data_evaluation.regridd_to_higher_spatial_resolution",
        lambda ds_input, ds_ref: (ds_input, ds_ref),
    )
    monkeypatch.setattr(
        "mhm_tools.post.gridded_data_evaluation.write_xarray_to_file",
        fake_write_xarray_to_file,
    )

    compare_input_with_ref(
        input_path=tmp_path / "input",
        input_var="var",
        output_path=tmp_path,
        ref_path=tmp_path / "ref",
        ref_var="var",
        input_name="in",
        ref_name="ref",
        plot=False,
        global_climate=True,
    )

    assert "ds" in captured
    out = captured["ds"]
    assert out["rel_mean"].dims == ("lat", "lon")
    assert out["rel_std"].dims == ("lat", "lon")
    assert np.isnan(out["rel_mean"].values[0, 1])
    assert np.isnan(out["rel_std"].values[0, 1])
    assert not np.isinf(out["rel_mean"].values).any()
    assert not np.isinf(out["rel_std"].values).any()
