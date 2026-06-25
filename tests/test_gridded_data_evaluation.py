import numpy as np
import pytest
import xarray as xr

from mhm_tools.common.logger import configure_mhm_tools_logger
from mhm_tools.post.gridded_data_evaluation import (
    apply_spatial_mask,
    compare_input_with_ref,
    crop_datasets_to_spatial_overlap,
    infer_time_resolution_hours_from_files,
    normalize_time_axis,
    regridd_to_higher_spatial_resolution,
)

# TODO: add a setup fixture to configure the logger to ERROR level to avoid cluttering test output with INFO logs also set propagate to True


@pytest.fixture(autouse=True, scope="session")
def _configure_test_logging():
    """Configure mhm_tools logging for the test session.

    Sets the package logger to ERROR and enables propagation so pytest's
    caplog captures log records without cluttering test output.
    """
    # Only enable propagation so pytest's caplog can capture package logs.
    configure_mhm_tools_logger(propagate=True)
    yield


@pytest.fixture(autouse=True, scope="session")
def _configure_test_logging():
    """Configure mhm_tools logging for the test session.

    Sets the package logger to ERROR and enables propagation so pytest's
    caplog captures log records without cluttering test output.
    """
    # Only enable propagation so pytest's caplog can capture package logs.
    configure_mhm_tools_logger(propagate=True)
    yield


def _write_timeseries_nc(path, times):
    ds = xr.Dataset(
        {"v": ("time", np.arange(len(times), dtype=float))}, coords={"time": times}
    )
    ds.to_netcdf(path)


def _stats_dataset(lat, lon):
    values = np.ones((len(lat), len(lon)), dtype=float)
    clim = np.ones((12, len(lat), len(lon)), dtype=float)
    return xr.Dataset(
        {
            "mean": (("lat", "lon"), values.copy()),
            "std": (("lat", "lon"), values.copy()),
            "clim": (("month", "lat", "lon"), clim),
        },
        coords={"month": np.arange(1, 13), "lat": lat, "lon": lon},
    )


def test_apply_spatial_mask_selects_matching_resolution_mask_variable(caplog):
    lat = np.array([3.0, 2.0, 1.0])
    lon = np.array([0.0, 1.0, 2.0])
    stats = _stats_dataset(lat, lon)
    wrong_fine_mask = np.full((6, 6), np.nan)
    matching_mask = np.ones((3, 3), dtype=float)
    matching_mask[1, 1] = 0.0
    mask_ds = xr.Dataset(
        {
            "mask": (("lat", "lon"), wrong_fine_mask),
            "mask_l2": (("lat_l2", "lon_l2"), matching_mask),
        },
        coords={
            "lat": np.array([3.25, 2.75, 2.25, 1.75, 1.25, 0.75]),
            "lon": np.array([-0.25, 0.25, 0.75, 1.25, 1.75, 2.25]),
            "lat_l2": lat,
            "lon_l2": lon,
        },
    )

    caplog.set_level("INFO")
    masked = apply_spatial_mask(stats, mask_ds)

    assert "Selected mask variable 'mask_l2'" in caplog.text
    assert masked["mean"].sel(lat=3.0, lon=0.0).item() == 1.0
    assert np.isnan(masked["mean"].sel(lat=2.0, lon=1.0).item())
    assert np.isfinite(masked["clim"].values).any()


def test_apply_spatial_mask_uses_finest_finer_mask_when_no_exact_match():
    lat = np.array([2.0, 1.0])
    lon = np.array([0.0, 1.0])
    stats = _stats_dataset(lat, lon)
    coarse_mask = np.zeros((2, 2), dtype=float)
    fine_mask = np.ones((4, 4), dtype=float)
    mask_ds = xr.Dataset(
        {
            "coarse_mask": (("lat_l2", "lon_l2"), coarse_mask),
            "fine_mask": (("lat", "lon"), fine_mask),
        },
        coords={
            "lat_l2": np.array([2.5, 0.5]),
            "lon_l2": np.array([0.5, 2.5]),
            "lat": np.array([2.25, 1.75, 1.25, 0.75]),
            "lon": np.array([-0.25, 0.25, 0.75, 1.25]),
        },
    )

    masked = apply_spatial_mask(stats, mask_ds)

    assert np.isfinite(masked["mean"].values).all()
    assert np.isfinite(masked["clim"].values).all()


def test_apply_spatial_mask_respects_explicit_mask_var():
    lat = np.array([3.0, 2.0, 1.0])
    lon = np.array([0.0, 1.0, 2.0])
    stats = _stats_dataset(lat, lon)
    mask_ds = xr.Dataset(
        {
            "mask": (("lat", "lon"), np.zeros((3, 3), dtype=float)),
            "mask_l2": (("lat_l2", "lon_l2"), np.ones((3, 3), dtype=float)),
        },
        coords={"lat": lat, "lon": lon, "lat_l2": lat, "lon_l2": lon},
    )

    masked = apply_spatial_mask(stats, mask_ds, mask_var="mask_l2")

    assert np.isfinite(masked["mean"].values).all()


def test_apply_spatial_mask_keeps_single_valid_crop_row():
    lat = np.array([52.15, 52.05, 51.95, 51.85])
    lon = np.array([8.05, 8.15, 8.25, 8.35, 8.45, 8.55, 8.65, 8.75])
    stats = _stats_dataset(lat, lon)
    mask = np.full((4, 8), np.nan)
    mask[1, 2:4] = 1.0
    mask_da = xr.DataArray(mask, coords={"lat": lat, "lon": lon}, dims=("lat", "lon"))

    masked = apply_spatial_mask(stats, mask_da)

    assert masked.sizes["lat"] == 1
    assert masked.sizes["lon"] == 2
    assert np.isfinite(masked["mean"].values).any()


def test_apply_spatial_mask_fails_when_only_coarser_masks_exist():
    lat = np.array([2.0, 1.0])
    lon = np.array([0.0, 1.0])
    stats = _stats_dataset(lat, lon)
    mask_ds = xr.Dataset(
        {"coarse_mask": (("lat_l2", "lon_l2"), np.ones((2, 2), dtype=float))},
        coords={
            "lat_l2": np.array([2.5, 0.5]),
            "lon_l2": np.array([0.5, 2.5]),
        },
    )

    with pytest.raises(ValueError, match="only coarser mask resolutions"):
        apply_spatial_mask(stats, mask_ds)


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
    cropped_input, cropped_ref = crop_datasets_to_spatial_overlap(input_ds, ref_ds)

    assert np.array_equal(cropped_input["lat"].values, lat_input)
    assert np.array_equal(cropped_input["lon"].values, lon_input)
    assert np.array_equal(cropped_ref["lat"].values, np.array([0.2, 0.6]))
    assert np.array_equal(cropped_ref["lon"].values, np.array([0.2, 0.6]))
    assert "Input spatial extent is a subset of reference" in caplog.text

    out_input, out_ref = regridd_to_higher_spatial_resolution(
        cropped_input, cropped_ref
    )
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
