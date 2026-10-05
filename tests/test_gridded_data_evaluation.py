import numpy as np
import pandas as pd
import pytest
import xarray as xr

from mhm_tools.common.logger import configure_mhm_tools_logger
from mhm_tools.common.time_utils import (
    normalize_time_axis,
    resample_to_target_freq,
)
from mhm_tools.common.xarray_utils import get_ds_extend
from mhm_tools.post.gridded_data_evaluation import (
    apply_spatial_mask,
    combine_results,
    compare_input_with_ref,
    create_period_mean_series,
    crop_datasets_to_spatial_overlap,
    get_file_stats,
    get_stats,
    get_stats_one_pass,
    get_stats_one_pass_subset,
    infer_time_resolution_hours_from_files,
    regridd_to_higher_spatial_resolution,
    resample_to_target_freq,
)


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


def test_get_file_stats_generates_bounds_for_single_point_slice():
    """A coordinate_slice narrowing to one point must not lose bound info.

    Regression test: generate_bounds needs len > 1 lon/lat to derive a
    cell width by diffing, so once coordinate_slice crops to a single
    point, get_file_stats must derive the resolution from the still
    multi-point ds_in itself (before cropping) and use it to build real
    lat_bnds/lon_bnds on the output - not attempt to diff the now-single-
    point cropped coordinates.
    """
    lat = np.array([2.0, 1.0, 0.0])
    lon = np.array([0.0, 1.0, 2.0])
    time = np.array(["2000-01-15", "2000-02-15", "2000-03-15"], dtype="datetime64[ns]")
    data = np.ones((len(time), len(lat), len(lon)), dtype=float)
    ds_in = xr.Dataset(
        {"v": (("time", "lat", "lon"), data)},
        coords={"time": time, "lat": lat, "lon": lon},
    )

    output = get_file_stats(
        ds_in,
        "v",
        coordinate_slice={"lat": slice(1.0, 1.0), "lon": slice(1.0, 1.0)},
    )

    assert output.sizes["lat"] == 1
    assert output.sizes["lon"] == 1
    assert "spatial_resolution" not in output.attrs
    assert "lat_bnds" in output.coords
    assert "lon_bnds" in output.coords
    assert output["lat"].attrs.get("bounds") == "lat_bnds"
    assert output["lon"].attrs.get("bounds") == "lon_bnds"
    assert np.abs(np.diff(output["lat_bnds"].values, axis=-1)) == pytest.approx(1.0)
    assert np.abs(np.diff(output["lon_bnds"].values, axis=-1)) == pytest.approx(1.0)


def test_get_file_stats_normalizes_latitude_longitude_dimension_names():
    """Regression test for inputs whose real spatial dimensions are named
    'latitude'/'longitude' (CF-compliant, as used by mHM's own pre.nc/pet.nc
    forcing files) rather than 'lat'/'lon'.

    get_file_stats used to build clim/std/mean still dimensioned by the
    alias while attaching a separate, disconnected 'lat'/'lon' coordinate
    built from the same values, so the resulting dataset carried an orphan
    'lat'/'lon' unrelated to clim/std/mean's actual dimensions.
    """
    latitude = np.array([2.0, 1.0, 0.0])
    longitude = np.array([0.0, 1.0, 2.0])
    time = np.array(["2000-01-15", "2000-02-15", "2000-03-15"], dtype="datetime64[ns]")
    data = np.ones((len(time), len(latitude), len(longitude)), dtype=float)
    ds_in = xr.Dataset(
        {"v": (("time", "latitude", "longitude"), data)},
        coords={"time": time, "latitude": latitude, "longitude": longitude},
    )

    output = get_file_stats(ds_in, "v")

    assert "latitude" not in output.dims
    assert "longitude" not in output.dims
    assert output["clim"].dims == ("month", "lat", "lon")
    assert output["clim"].sizes["lat"] == len(latitude)
    assert output["clim"].sizes["lon"] == len(longitude)
    assert output.sizes["lat"] == len(latitude)
    assert output.sizes["lon"] == len(longitude)


def test_get_stats_one_pass_generates_bounds_for_single_point_slice(tmp_path):
    lat = np.array([2.0, 1.0, 0.0])
    lon = np.array([0.0, 1.0, 2.0])
    for month in (1, 2):
        time = np.array([f"2000-{month:02d}-15"], dtype="datetime64[ns]")
        data = np.ones((1, len(lat), len(lon)), dtype=float)
        ds = xr.Dataset(
            {"v": (("time", "lat", "lon"), data)},
            coords={"time": time, "lat": lat, "lon": lon},
        )
        ds.to_netcdf(tmp_path / f"file_{month:02d}.nc")

    output = get_stats_one_pass(
        path=tmp_path,
        var="v",
        coordinate_slice={"lat": slice(1.0, 1.0), "lon": slice(1.0, 1.0)},
    )

    assert output.sizes["lat"] == 1
    assert output.sizes["lon"] == 1
    assert "spatial_resolution" not in output.attrs
    assert "lat_bnds" in output.coords
    assert "lon_bnds" in output.coords
    assert output["lat"].attrs.get("bounds") == "lat_bnds"
    assert output["lon"].attrs.get("bounds") == "lon_bnds"
    assert np.abs(np.diff(output["lat_bnds"].values, axis=-1)) == pytest.approx(1.0)
    assert np.abs(np.diff(output["lon_bnds"].values, axis=-1)) == pytest.approx(1.0)


def test_get_stats_direct_open_narrows_to_single_point_without_raising(tmp_path):
    """End-to-end: coordinate_slice narrows a pre-built stats file to one
    point; get_ds_extend must still resolve the file's extent afterward
    instead of raising the "no valid lon or lat coordinates" ValueError.
    """
    lat = np.array([2.0, 1.0, 0.0])
    lon = np.array([0.0, 1.0, 2.0])
    stats = _stats_dataset(lat, lon)
    path = tmp_path / "stats.nc"
    stats.to_netcdf(path)

    output = get_stats(
        path=path,
        var=None,
        factor=1,
        coordinate_slice={"lat": slice(1.0, 1.0), "lon": slice(1.0, 1.0)},
        n_bootstrap_years=None,
        ncpus=1,
        output_file=None,
    )

    assert output.sizes["lat"] == 1
    assert output.sizes["lon"] == 1
    assert "lat_bnds" in output.coords
    assert "lon_bnds" in output.coords
    assert output["lat"].attrs.get("bounds") == "lat_bnds"
    assert output["lon"].attrs.get("bounds") == "lon_bnds"

    lon_min, lon_max, lat_min, lat_max = get_ds_extend(output)
    assert lon_min <= 1.0 <= lon_max
    assert lat_min <= 1.0 <= lat_max


def test_get_stats_masks_and_crops_latitude_longitude_input(tmp_path):
    """End-to-end regression test for gridded-data-evaluation against mHM's
    own pre.nc/pet.nc meteo forcing files, whose real spatial dimensions
    are named 'latitude'/'longitude' rather than 'lat'/'lon'.

    Before the fix, get_file_stats left clim/std/mean dimensioned by the
    alias while apply_spatial_mask's masking/cropping only ever touched a
    disconnected 'lat'/'lon' coordinate: the spatial mask silently had no
    effect on the actual statistics, and forcing the (correctly cropped)
    coordinate onto the (never cropped) data later crashed with
    'xarray.core.coordinates.CoordinateValidationError: conflicting sizes
    for dimension lat'.
    """
    latitude = np.array([2.0, 1.0, 0.0])
    longitude = np.array([0.0, 1.0, 2.0])
    time = np.array(["2000-01-15", "2000-02-15", "2000-03-15"], dtype="datetime64[ns]")
    data = np.ones((len(time), len(latitude), len(longitude)), dtype=float)
    ds_in = xr.Dataset(
        {"v": (("time", "latitude", "longitude"), data)},
        coords={"time": time, "latitude": latitude, "longitude": longitude},
    )
    path = tmp_path / "input.nc"
    ds_in.to_netcdf(path)

    # A basin mask file with the conventional 'lat'/'lon' naming, valid at
    # only a single cell (lat=0.0, lon=2.0).
    mask = np.zeros((3, 3), dtype=float)
    mask[2, 2] = 1.0
    mask_da = xr.DataArray(
        mask, coords={"lat": latitude, "lon": longitude}, dims=("lat", "lon")
    )

    output = get_stats(
        path=path,
        var="v",
        factor=1,
        coordinate_slice=None,
        n_bootstrap_years=None,
        ncpus=1,
        output_file=None,
        mask_da=mask_da,
    )

    assert output.sizes["lat"] == 1
    assert output.sizes["lon"] == 1
    assert output["clim"].sizes["lat"] == 1
    assert output["clim"].sizes["lon"] == 1
    assert output["lat"].item() == pytest.approx(0.0)
    assert output["lon"].item() == pytest.approx(2.0)
    assert np.isfinite(output["mean"].values).all()


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

    def fake_write_xarray_to_file(ds, file_path, **kwargs):
        captured["ds"] = ds.copy(deep=True)
        captured["file_path"] = file_path
        captured["kwargs"] = kwargs

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


def test_resample_to_target_freq_ignores_other_dims_during_align():
    """Aligning resampled input/ref on time must not also join on lat/lon.

    Regression test: resample_to_target_freq used to finish with
    xr.align(ds_input, ds_ref, join="inner") with no `exclude`, which joins
    on every dimension the two objects share - not just time. lat/lon are
    already matched positionally by the spatial cropping/regridding that
    runs before this function, but their coordinate *labels* can still
    differ by floating-point noise (e.g. a source file storing lat/lon as
    float32, like mHM's NetCDF output, versus float64 elsewhere: 68.15
    stored as float32 and read back as float64 becomes 68.1500015258789,
    not 68.15). An unrestricted inner join treats those as non-matching
    labels and collapses lat/lon to a near-empty intersection, silently
    wiping out all data even though both inputs were perfectly valid.
    """
    lat_nominal = [68.15, 68.05, 67.95, 67.85]
    lon_nominal = [27.65, 27.75]

    # Simulate the real-world mismatch: one side's coordinates round-tripped
    # through float32, the other kept as float64, for the same nominal grid.
    lat_input = np.array(lat_nominal, dtype=np.float32).astype(np.float64)
    lon_input = np.array(lon_nominal, dtype=np.float32).astype(np.float64)
    lat_ref = np.array(lat_nominal, dtype=np.float64)
    lon_ref = np.array(lon_nominal, dtype=np.float64)
    assert not np.array_equal(lat_input, lat_ref)
    assert not np.array_equal(lon_input, lon_ref)

    time_input = pd.date_range("2021-01-01", "2021-02-28T23:00:00", freq="h")
    input_da = xr.DataArray(
        np.ones((time_input.size, len(lat_input), len(lon_input))),
        dims=("time", "lat", "lon"),
        coords={"time": time_input, "lat": lat_input, "lon": lon_input},
        name="time_series",
    )

    # Mid-month anchored, like FLUXCOM's monthly timestamps, already at the
    # target ("ME") cadence so only `input_da` needs resampling.
    time_ref = pd.to_datetime(["2021-01-16", "2021-02-16"])
    ref_da = xr.DataArray(
        np.full((2, len(lat_ref), len(lon_ref)), 2.0),
        dims=("time", "lat", "lon"),
        coords={"time": time_ref, "lat": lat_ref, "lon": lon_ref},
        name="time_series",
    )

    resampled_input, resampled_ref = resample_to_target_freq(input_da, ref_da, "ME")

    assert resampled_input.sizes["time"] == 2
    assert resampled_input.sizes["lat"] == len(lat_nominal)
    assert resampled_input.sizes["lon"] == len(lon_nominal)
    assert resampled_ref.sizes["lat"] == len(lat_nominal)
    assert resampled_ref.sizes["lon"] == len(lon_nominal)
    assert np.isfinite(resampled_input.values).all()
    assert np.isfinite(resampled_ref.values).all()


def _period_subset(period_sums, period_counts, shape=(2, 2)):
    """Build one `get_stats_one_pass_subset` result around given buckets."""
    return (
        np.zeros(shape),
        np.zeros(shape),
        1,
        np.zeros((12, *shape)),
        np.zeros((12, *shape), dtype=np.uint32),
        period_sums,
        period_counts,
    )


def _write_daily_year(path, year, var, lat, lon, values, nan_cell=None):
    """Write one file of daily steps for `year`, optionally holing one cell."""
    time = pd.date_range(f"{year}-01-01", f"{year}-12-31", freq="D")
    data = np.full((time.size, len(lat), len(lon)), values, dtype=np.float32)
    if nan_cell is not None:
        data[::2, nan_cell[0], nan_cell[1]] = np.nan
    ds = xr.Dataset(
        {var: (("time", "lat", "lon"), data)},
        coords={"time": time, "lat": lat, "lon": lon},
    )
    ds.to_netcdf(path / f"{var}_{year}_daily.nc")


def test_create_period_mean_series_divides_each_bucket_by_its_count():
    """Each bucket becomes sum/count, and an empty bucket becomes NaN."""
    period_sums = {
        (2000, 1): np.array([[6.0, 9.0], [0.0, 4.0]], dtype=np.float32),
        (2000, 2): np.array([[2.0, 0.0], [8.0, 0.0]], dtype=np.float32),
    }
    period_counts = {
        (2000, 1): np.array([[3, 3], [0, 2]], dtype=np.uint16),
        (2000, 2): np.array([[1, 0], [4, 0]], dtype=np.uint16),
    }

    series = create_period_mean_series(
        period_sums, period_counts, [0.0, 1.0], [0.0, 1.0], "v", units="mm/day"
    )["time_series"]

    assert series.dtype == np.float32
    np.testing.assert_array_equal(
        series.values[0], np.array([[2.0, 3.0], [np.nan, 2.0]], dtype=np.float32)
    )
    np.testing.assert_array_equal(
        series.values[1], np.array([[2.0, np.nan], [2.0, np.nan]], dtype=np.float32)
    )
    assert series.attrs["units"] == "mm/day"


def test_combine_results_merges_buckets_split_across_subsets():
    """A calendar month present in two subsets is summed element-wise."""
    shared = (2000, 5)
    first = _period_subset(
        {shared: np.full((2, 2), 3.0, dtype=np.float32)},
        {shared: np.full((2, 2), 2, dtype=np.uint16)},
    )
    second = _period_subset(
        {
            shared: np.full((2, 2), 4.0, dtype=np.float32),
            (2000, 6): np.full((2, 2), 1.0, dtype=np.float32),
        },
        {
            shared: np.full((2, 2), 5, dtype=np.uint16),
            (2000, 6): np.full((2, 2), 1, dtype=np.uint16),
        },
    )

    *_, period_sums, period_counts = combine_results([first, second])

    assert sorted(period_sums) == [(2000, 5), (2000, 6)]
    np.testing.assert_array_equal(period_sums[shared], np.full((2, 2), 7.0))
    np.testing.assert_array_equal(period_counts[shared], np.full((2, 2), 7))
    np.testing.assert_array_equal(period_sums[(2000, 6)], np.full((2, 2), 1.0))


def test_combine_results_consumes_subset_results():
    """The subsets are released during the merge, not held alongside it."""
    subsets = [
        _period_subset(
            {(2000, month): np.ones((2, 2), dtype=np.float32)},
            {(2000, month): np.ones((2, 2), dtype=np.uint16)},
        )
        for month in (1, 2)
    ]
    kept_sums, kept_counts = subsets[0][5], subsets[0][6]

    combine_results(subsets)

    assert subsets == []
    assert kept_sums == {}
    assert kept_counts == {}


def test_create_period_mean_series_empties_its_input_buckets():
    """Each bucket is dropped once divided, so the record is never held twice."""
    period_sums = {(2000, 1): np.ones((2, 2), dtype=np.float32)}
    period_counts = {(2000, 1): np.ones((2, 2), dtype=np.uint16)}

    create_period_mean_series(period_sums, period_counts, [0.0, 1.0], [0.0, 1.0], "v")

    assert period_sums == {}
    assert period_counts == {}


def test_period_buckets_use_compact_dtypes(tmp_path):
    """The streamed accumulators stay compact, which bounds the peak memory.

    `monthly_counts` spans the whole record, so it needs uint32: uint16 would
    wrap at 65535, which 15-minute input over 44 years exceeds. A period bucket
    only ever holds one month, so uint16 is safe there.
    """
    lat, lon = np.array([1.0, 0.0]), np.array([0.0, 1.0])
    _write_daily_year(tmp_path, 2000, "v", lat, lon, 2.0)

    (
        _,
        _,
        _,
        monthly_sums,
        monthly_counts,
        period_sums,
        period_counts,
    ) = get_stats_one_pass_subset(
        sorted(tmp_path.glob("*.nc")), "v", with_period_series=True
    )

    assert monthly_counts.dtype == np.uint32
    assert monthly_sums.dtype == np.float64
    assert all(values.dtype == np.float32 for values in period_sums.values())
    assert all(values.dtype == np.uint16 for values in period_counts.values())
    assert np.iinfo(np.uint16).max < 44 * 12 * 31 * 96 // 12


@pytest.mark.parametrize("ncpus", [1, 2])
def test_get_stats_one_pass_period_series_matches_monthly_mean(tmp_path, ncpus):
    """The streamed series equals a direct monthly resample, serial or parallel."""
    lat, lon = np.array([1.0, 0.0]), np.array([0.0, 1.0])
    for year, value in ((2000, 2.0), (2001, 5.0)):
        _write_daily_year(tmp_path, year, "v", lat, lon, value, nan_cell=(0, 1))

    output = get_stats_one_pass(
        path=tmp_path,
        var="v",
        ncpus=ncpus,
        available_years=[2000, 2001],
        with_period_series=True,
    )

    with xr.open_mfdataset(sorted(tmp_path.glob("*.nc"))) as ds:
        expected = ds["v"].resample(time="MS").mean(skipna=True).compute()

    series = output["time_series"].transpose("time", "lat", "lon")
    assert series.sizes["time"] == 24
    np.testing.assert_array_equal(series["time"].values, expected["time"].values)
    np.testing.assert_allclose(
        series.values, expected.values.astype(np.float32), rtol=1e-6
    )
