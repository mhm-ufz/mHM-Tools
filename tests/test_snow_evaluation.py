"""Tests for the snow cover evaluation."""

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from mhm_tools.common.logger import configure_mhm_tools_logger
from mhm_tools.common.time_utils import (
    normalize_time_axis,
    resample_to_target_freq,
)
from mhm_tools.common.xarray_utils import (
    aggregate_to_target_grid,
    regrid_to_coarser_grid,
)
from mhm_tools.post.snow_evaluation import (
    MISSING_VALUE,
    NO_SNOW_VALUE,
    SNOW_VALUE,
    calculate_classification_accuracy,
    calculate_snow_covered_cell_percentage,
    calculate_snow_season_metrics,
    create_region_outputs,
    create_snow_cover_flag,
    create_step_bounds_in_days,
    create_year_windows,
    crop_to_region,
    get_snow_alias,
    get_target_frequency,
    resample_snow_to_target_frequency,
    select_hemisphere,
    select_regions,
    snow_evaluation,
)


@pytest.fixture(autouse=True, scope="session")
def _configure_test_logging():
    """Configure mhm_tools logging so caplog can capture package logs."""
    configure_mhm_tools_logger(propagate=True)


def _snow_field(values, times, lat=(45.0, 46.0), lon=(8.0, 9.0)):
    """Build a (time, lat, lon) DataArray from a nested value list."""
    return xr.DataArray(
        np.asarray(values, dtype=float),
        dims=("time", "lat", "lon"),
        coords={"time": times, "lat": list(lat), "lon": list(lon)},
    )


def _single_cell(values, times):
    """Build a one-cell (time, lat, lon) DataArray from a flat value list."""
    return xr.DataArray(
        np.asarray(values, dtype=float).reshape(-1, 1, 1),
        dims=("time", "lat", "lon"),
        coords={"time": times, "lat": [45.0], "lon": [8.0]},
    )


def _seasonal_field(times, lat, lon, seed, offset=0.0):
    """Build a seasonal snow-like field that peaks in the northern winter."""
    rng = np.random.default_rng(seed)
    day_of_year = times.dayofyear.values[:, None, None]
    seasonal = np.cos(2 * np.pi * day_of_year / 365.25) * 30 + offset
    values = seasonal + rng.normal(0, 3, (len(times), len(lat), len(lon)))
    return xr.DataArray(
        np.clip(values, 0, None),
        dims=("time", "lat", "lon"),
        coords={"time": times, "lat": lat, "lon": lon},
        name="swe",
    )


def _write_snow_nc(path, times, lat, lon, seed, offset=0.0):
    """Write a seasonal snow-like NetCDF file and return its path."""
    _seasonal_field(times, lat, lon, seed, offset).to_dataset().to_netcdf(path)
    return path


# ---------------------------------------------------------------------------
# snow flag
# ---------------------------------------------------------------------------
def test_create_snow_cover_flag_thresholds_strictly_above():
    """Only values strictly above the threshold become snow."""
    times = pd.date_range("2001-01-01", periods=1, freq="D")
    raw = _snow_field([[[-1.0, 0.0], [0.5, np.nan]]], times)

    flag = create_snow_cover_flag(raw, 0.0)

    np.testing.assert_array_equal(
        flag.values.ravel(),
        [NO_SNOW_VALUE, NO_SNOW_VALUE, SNOW_VALUE, MISSING_VALUE],
    )
    assert flag.dtype == np.int8
    assert flag.attrs["snow_threshold"] == 0.0
    assert flag.attrs["flag_meanings"] == "no_snow snow"


def test_create_snow_cover_flag_higher_threshold_excludes_small_values():
    """A raised threshold turns small values into no snow."""
    times = pd.date_range("2001-01-01", periods=1, freq="D")
    raw = _snow_field([[[-1.0, 0.0], [0.5, np.nan]]], times)

    flag = create_snow_cover_flag(raw, 0.5)

    # 0.5 is not above the threshold, only the NaN cell stays missing
    np.testing.assert_array_equal(
        flag.values.ravel(),
        [NO_SNOW_VALUE, NO_SNOW_VALUE, NO_SNOW_VALUE, MISSING_VALUE],
    )


# ---------------------------------------------------------------------------
# classification accuracy
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    ("input_values", "ref_values", "expected"),
    [
        ([0] * 10, [0] * 10, 1.0),
        ([1] * 10, [1] * 10, 1.0),
        ([1] * 10, [0] * 10, 0.0),
        ([1, 1, 1, 1, 1, 0, 0, 0, 0, 0], [1, 1, 1, 0, 0, 1, 1, 0, 0, 0], 0.6),
        ([0, 0, 0, 1, 1, 1, 1, 0, 0, 0], [0, 0, 0, 0, 1, 1, 1, 0, 0, 0], 0.9),
    ],
)
def test_calculate_classification_accuracy_counts_matching_steps(
    input_values, ref_values, expected
):
    """CA is the share of steps in which both flags agree."""
    times = pd.date_range("2001-01-01", periods=10, freq="D")

    accuracy = calculate_classification_accuracy(
        _single_cell(input_values, times), _single_cell(ref_values, times)
    )

    assert float(accuracy["classification_accuracy"]) == pytest.approx(expected)
    assert int(accuracy["compared_time_steps"]) == 10


def test_calculate_classification_accuracy_skips_steps_missing_in_either_dataset():
    """A step missing in either dataset leaves n and the numerator."""
    times = pd.date_range("2001-01-01", periods=10, freq="D")
    input_flag = _single_cell([1, 1, 1, 1, 1, 0, 0, 0, 0, 0], times)
    input_flag[:4] = MISSING_VALUE
    ref_flag = _single_cell([1, 1, 1, 0, 0, 1, 1, 0, 0, 0], times)

    accuracy = calculate_classification_accuracy(input_flag, ref_flag)

    # only the last six steps are comparable, three of them agree
    assert int(accuracy["compared_time_steps"]) == 6
    assert float(accuracy["classification_accuracy"]) == pytest.approx(0.5)


def test_calculate_classification_accuracy_is_undefined_without_common_steps():
    """A cell without a comparable step stays undefined."""
    times = pd.date_range("2001-01-01", periods=10, freq="D")
    input_flag = _single_cell([MISSING_VALUE] * 10, times)
    ref_flag = _single_cell([1] * 10, times)

    accuracy = calculate_classification_accuracy(input_flag, ref_flag)

    # NaN, not 0, so "never comparable" cannot read as "never agreed"
    assert np.isnan(float(accuracy["classification_accuracy"]))
    assert np.isnan(float(accuracy["compared_time_steps"]))


# ---------------------------------------------------------------------------
# season metrics
# ---------------------------------------------------------------------------
def _season_metrics_field():
    """Build one snow year with four hand-placed cell behaviours."""
    times = pd.date_range("2001-09-01", "2002-08-31", freq="D")
    values = np.zeros((len(times), 2, 2))

    def between(start, end):
        return (times >= start) & (times <= end)

    values[between("2001-11-01", "2002-03-31"), 0, 0] = 1  # one long spell
    values[between("2001-12-01", "2001-12-10"), 0, 1] = 1  # two spells with a gap
    values[between("2002-02-01", "2002-02-10"), 0, 1] = 1
    values[:, 1, 0] = 0  # valid but never snow covered
    values[:, 1, 1] = MISSING_VALUE  # no valid data at all
    return times, _snow_field(values, times)


def test_calculate_snow_season_metrics_dates_and_duration():
    """First and last snow day and the season span of one spell."""
    times, flag = _season_metrics_field()
    windows = create_year_windows(pd.DatetimeIndex(times), "snow_year", 9, 1.0)

    metrics = calculate_snow_season_metrics(flag, windows, "D").isel(year=0)

    long_spell = metrics.isel(lat=0, lon=0)
    assert str(long_spell["first_snow_date"].values)[:10] == "2001-11-01"
    assert str(long_spell["last_snow_date"].values)[:10] == "2002-03-31"
    # 2001-09-01 is day 0 of the window
    assert float(long_spell["first_snow_day_of_window"]) == 61.0
    assert float(long_spell["last_snow_day_of_window"]) == 211.0
    assert float(long_spell["season_span_days"]) == 151.0
    assert float(long_spell["snow_cover_days"]) == 151.0


def test_calculate_snow_season_metrics_span_spans_gaps_but_days_do_not():
    """The span covers a mid-season gap, the day count does not."""
    times, flag = _season_metrics_field()
    windows = create_year_windows(pd.DatetimeIndex(times), "snow_year", 9, 1.0)

    metrics = calculate_snow_season_metrics(flag, windows, "D").isel(year=0)
    two_spells = metrics.isel(lat=0, lon=1)

    assert float(two_spells["season_span_days"]) == 72.0
    # the two ten day spells, not the 72 days they straddle
    assert float(two_spells["snow_cover_days"]) == 20.0

    span = metrics["season_span_days"].values
    first = metrics["first_snow_day_of_window"].values
    last = metrics["last_snow_day_of_window"].values
    covered = metrics["snow_cover_days"].values
    defined = np.isfinite(span)
    np.testing.assert_allclose(span[defined], (last - first + 1)[defined])
    assert np.all(covered[defined] <= span[defined])


def test_calculate_snow_season_metrics_separates_snow_free_from_missing_cells():
    """A snow free cell counts zero days, a missing cell stays NaN."""
    times, flag = _season_metrics_field()
    windows = create_year_windows(pd.DatetimeIndex(times), "snow_year", 9, 1.0)

    metrics = calculate_snow_season_metrics(flag, windows, "D").isel(year=0)

    snow_free = metrics.isel(lat=1, lon=0)
    assert np.isnan(float(snow_free["season_span_days"]))
    assert np.isnat(snow_free["first_snow_date"].values)
    # a valid cell that never carries snow has zero snow days, not NaN
    assert float(snow_free["snow_cover_days"]) == 0.0

    missing = metrics.isel(lat=1, lon=1)
    assert np.isnan(float(missing["snow_cover_days"]))
    assert np.isnat(missing["first_snow_date"].values)


def test_create_year_windows_snow_year_keeps_one_winter_together():
    """A snow year holds one winter that a calendar year splits."""
    times, flag = _season_metrics_field()
    time_index = pd.DatetimeIndex(times)

    snow_year = create_year_windows(time_index, "snow_year", 9, 1.0)
    calendar_year = create_year_windows(time_index, "calendar_year", 9, 1.0)

    assert [year for year, _, _ in snow_year] == [2001]
    assert [year for year, _, _ in calendar_year] == [2001, 2002]

    snow_days = calculate_snow_season_metrics(flag, snow_year, "D")
    calendar_days = calculate_snow_season_metrics(flag, calendar_year, "D")
    # the same 151 day winter, split across two calendar years
    assert float(snow_days["snow_cover_days"].isel(year=0, lat=0, lon=0)) == 151.0
    per_calendar_year = [
        float(calendar_days["snow_cover_days"].isel(year=index, lat=0, lon=0))
        for index in range(calendar_days.sizes["year"])
    ]
    assert per_calendar_year == [61.0, 90.0]
    assert sum(per_calendar_year) == 151.0


# ---------------------------------------------------------------------------
# calendar normalization
# ---------------------------------------------------------------------------
def _time_series(freq, periods=40, start="2001-01-01"):
    """Build a plain time series DataArray at the given frequency."""
    times = pd.date_range(start, periods=periods, freq=freq)
    return xr.DataArray(
        np.arange(len(times), dtype=float), dims="time", coords={"time": times}
    )


def test_get_target_frequency_never_finer_than_eight_days():
    """The target is the coarser calendar, floored at 8 days."""
    assert get_target_frequency(_time_series("D"), _time_series("D")) == "8D"
    assert get_target_frequency(_time_series("6h"), _time_series("D")) == "8D"
    assert get_target_frequency(_time_series("D"), _time_series("8D")) == "8D"
    # a coarser calendar still wins over the eight day floor
    assert get_target_frequency(_time_series("D"), _time_series("ME")) == "ME"


def test_get_snow_alias_names_an_eight_day_step():
    """An 8 day step gets its own alias instead of an hour count."""
    # timedelta_to_alias only names daily, weekly and monthly calendars
    assert get_snow_alias(_time_series("8D")) == (192, "8D")
    assert get_snow_alias(_time_series("D")) == (24, "D")


def test_create_step_bounds_in_days_matches_the_aggregated_days():
    """An 8 day step covers exactly the days it was aggregated from."""
    times = pd.date_range("2001-01-01", "2001-12-31", freq="D")
    values = xr.DataArray(
        np.arange(len(times), dtype=float), dims="time", coords={"time": times}
    )
    resampled = normalize_time_axis(
        values.resample(time="8D", origin="epoch").mean(), "8D", anchor="period_start"
    )
    labels = pd.DatetimeIndex(resampled.time.values)
    window_start = pd.Timestamp("2001-01-01")

    starts, ends = create_step_bounds_in_days(
        labels,
        "8D",
        window_start,
        pd.Timestamp("2001-12-31 23:59:59.999999999"),
    )

    index = 3
    step_days = round(ends[index] - starts[index])
    assert step_days == 8
    first_day = window_start + pd.Timedelta(days=starts[index])
    # the value at each day equals its offset from the record start
    expected_mean = np.mean(
        [
            (first_day + pd.Timedelta(days=day) - times[0]).days
            for day in range(step_days)
        ]
    )
    assert float(resampled.isel(time=index)) == pytest.approx(expected_mean)


def test_resample_snow_to_target_frequency_aligns_offset_records():
    """Records starting on different days still share 8 day bins."""
    input_da = _single_cell(
        np.arange(60.0), pd.date_range("2001-01-01", periods=60, freq="D")
    )
    ref_da = _single_cell(
        np.arange(60.0), pd.date_range("2001-01-04", periods=60, freq="D")
    )
    # binning from each record's own start leaves no common eight day bin
    naive_input = pd.DatetimeIndex(input_da.resample(time="8D").mean().time.values)
    naive_ref = pd.DatetimeIndex(ref_da.resample(time="8D").mean().time.values)
    assert len(naive_input.intersection(naive_ref)) == 0

    resampled_input, resampled_ref = resample_snow_to_target_frequency(
        input_da, ref_da, "8D"
    )

    assert resampled_input.sizes["time"] > 0
    np.testing.assert_array_equal(
        resampled_input.time.values, resampled_ref.time.values
    )
    spacing = np.diff(pd.DatetimeIndex(resampled_input.time.values))
    assert set(spacing / np.timedelta64(1, "D")) == {8.0}


# ---------------------------------------------------------------------------
# grid normalization
# ---------------------------------------------------------------------------
def _fine_field():
    """Build a 4x4 quarter degree field holding 0..15."""
    times = pd.date_range("2001-01-01", periods=1, freq="D")
    fine_lat = np.array([45.125, 45.375, 45.625, 45.875])
    fine_lon = np.array([8.125, 8.375, 8.625, 8.875])
    return xr.DataArray(
        np.arange(16, dtype=float).reshape(1, 4, 4),
        dims=("time", "lat", "lon"),
        coords={"time": times, "lat": fine_lat, "lon": fine_lon},
    )


def test_aggregate_to_target_grid_averages_integer_blocks():
    """An integer factor grid is aggregated by block mean."""
    fine = _fine_field()
    target_lat = np.array([45.25, 45.75])
    target_lon = np.array([8.25, 8.75])

    aggregated = aggregate_to_target_grid(fine, target_lat, target_lon)

    assert aggregated.shape == (1, 2, 2)
    np.testing.assert_allclose(aggregated.values.ravel(), [2.5, 4.5, 10.5, 12.5])
    np.testing.assert_allclose(aggregated["lat"].values, target_lat)


def test_aggregate_to_target_grid_bins_non_integer_factors():
    """A non integer factor falls back to binning the cells."""
    fine = _fine_field()
    # 0.3 degree cells are no integer multiple of 0.25, so binning takes over
    target_lat = np.array([45.2, 45.5, 45.8])
    target_lon = np.array([8.2, 8.5, 8.8])

    aggregated = aggregate_to_target_grid(fine, target_lat, target_lon)

    assert aggregated.shape == (1, 3, 3)
    np.testing.assert_allclose(aggregated["lat"].values, target_lat)
    assert np.isfinite(aggregated.values).all()


def test_regrid_to_coarser_grid_moves_the_finer_dataset():
    """The finer dataset ends up on the coarser grid."""
    fine = _fine_field()
    coarse = xr.DataArray(
        np.zeros((1, 2, 2)),
        dims=("time", "lat", "lon"),
        coords={
            "time": fine.time,
            "lat": np.array([45.25, 45.75]),
            "lon": np.array([8.25, 8.75]),
        },
    )

    regridded_fine, regridded_coarse = regrid_to_coarser_grid(fine, coarse)

    assert regridded_fine.shape == regridded_coarse.shape == (1, 2, 2)


# ---------------------------------------------------------------------------
# coverage share and hemispheres
# ---------------------------------------------------------------------------
def test_calculate_snow_covered_cell_percentage_ignores_missing_cells():
    """The share counts only cells that hold valid data."""
    times = pd.date_range("2001-01-01", periods=3, freq="D")
    values = np.zeros((3, 2, 2))
    values[0] = [[1, 1], [0, 0]]
    values[1] = [[1, 1], [1, 0]]
    values[2] = MISSING_VALUE
    values[2, 0, 0] = 1
    flag = _snow_field(values, times, lat=(-1.0, 1.0))

    percentage = calculate_snow_covered_cell_percentage(flag)

    # 2 of 4 valid, 3 of 4 valid, then 1 of the single valid cell
    np.testing.assert_allclose(percentage.values, [50.0, 75.0, 100.0])
    assert percentage.attrs["units"] == "%"


def test_select_hemisphere_splits_at_the_equator():
    """Cells are split at the equator, one side may stay empty."""
    times = pd.date_range("2001-01-01", periods=1, freq="D")
    flag = _snow_field([[[1, 1], [1, 1]]], times, lat=(-1.0, 1.0))

    northern = select_hemisphere(flag, northern=True)
    southern = select_hemisphere(flag, northern=False)

    np.testing.assert_allclose(northern["lat"].values, [1.0])
    np.testing.assert_allclose(southern["lat"].values, [-1.0])
    # a domain on one side of the equator leaves the other selection empty
    northern_only = flag.isel(lat=[1])
    assert select_hemisphere(northern_only, northern=False).sizes["lat"] == 0


# ---------------------------------------------------------------------------
# regions
# ---------------------------------------------------------------------------
def test_select_regions_resolves_names_case_insensitively():
    """Region names are resolved from 'all', 'none' or a list."""
    assert len(select_regions("all")) == 6
    assert select_regions("none") == []
    assert select_regions(" europe , Africa ") == ["Europe", "Africa"]


def test_select_regions_rejects_an_unknown_name():
    """An unknown region name raises instead of being skipped."""
    with pytest.raises(ValueError, match="Unknown region"):
        select_regions("Atlantis")


def test_crop_to_region_returns_no_cells_outside_the_bounds():
    """Cropping to a region outside the domain leaves no cells."""
    times = pd.date_range("2001-01-01", periods=1, freq="D")
    alpine = _snow_field([[[1, 1], [1, 1]]], times, lat=(45.0, 46.0), lon=(8.0, 9.0))

    assert crop_to_region(alpine, "Europe").sizes["lat"] > 0
    assert crop_to_region(alpine, "South America").sizes["lon"] == 0


def test_create_region_outputs_skips_empty_regions_and_sanitizes_names(tmp_path):
    """Empty regions are skipped and a slash never becomes a folder."""
    times = pd.date_range("2001-01-01", periods=12, freq="D")
    lat = np.array([40.0, 41.0])
    lon = np.array([-100.0, -99.0])
    values = np.zeros((len(times), 2, 2))
    values[: len(times) // 2] = 1
    input_flag = _snow_field(values, times, lat=lat, lon=lon)
    ref_flag = input_flag.copy()

    written = create_region_outputs(
        input_flag,
        ref_flag,
        tmp_path,
        ["North/Central America", "South America"],
        input_name="model",
        ref_name="obs",
        target_freq="D",
        write_gif=False,
    )

    assert not any("South America" in key for key in written)
    assert any("North/Central America" in key for key in written)
    # the slash in the region name must not become a directory
    for file_path in written.values():
        assert file_path.parent == tmp_path
        assert "_region_North_Central_America" in file_path.name


# ---------------------------------------------------------------------------
# end to end
# ---------------------------------------------------------------------------
def test_snow_evaluation_writes_its_outputs(tmp_path):
    """The evaluation writes its files and keeps the metrics consistent."""
    times = pd.date_range("2001-01-01", "2002-12-31", freq="D")
    lat = np.arange(45.25, 47.0, 0.5)
    lon = np.arange(8.25, 10.0, 0.5)
    input_file = _write_snow_nc(tmp_path / "input.nc", times, lat, lon, seed=1)
    ref_file = _write_snow_nc(tmp_path / "ref.nc", times, lat, lon, seed=2, offset=3.0)
    output_dir = tmp_path / "out"

    written = snow_evaluation(
        input_path=input_file,
        ref_path=ref_file,
        output_dir=output_dir,
        input_name="model",
        ref_name="obs",
        input_snow_threshold=1.0,
        ref_snow_threshold=1.0,
        regions="none",
        write_gif=False,
    )

    assert written
    for file_path in written.values():
        assert file_path.is_file()
    assert (output_dir / "snow_season_metrics_model.nc").is_file()
    assert (output_dir / "snow_season_metrics_difference_model_minus_obs.nc").is_file()
    assert (output_dir / "classification_accuracy_model_vs_obs.png").is_file()
    assert (output_dir / "snow_cover_percentage_model_vs_obs.png").is_file()

    metrics = xr.open_dataset(output_dir / "snow_season_metrics_model.nc")
    span = metrics["season_span_days"].values
    first = metrics["first_snow_day_of_window"].values
    last = metrics["last_snow_day_of_window"].values
    covered = metrics["snow_cover_days"].values
    defined = np.isfinite(span)
    np.testing.assert_allclose(span[defined], (last - first + 1)[defined])
    assert np.all(covered[defined] <= span[defined] + 1e-9)
    assert metrics.attrs["target_frequency"] == "8D"

    accuracy = xr.open_dataset(output_dir / "classification_accuracy_model_vs_obs.nc")
    values = accuracy["classification_accuracy"].values
    finite = values[np.isfinite(values)]
    assert finite.size
    assert finite.min() >= 0.0
    assert finite.max() <= 1.0

    # both snow cover fields hold nothing but the two flags
    snow_cover = xr.open_dataset(output_dir / "snow_cover_model.nc")["snow_cover"]
    present = snow_cover.values[np.isfinite(snow_cover.values)]
    assert set(np.unique(present)) <= {0.0, 1.0}
    np.testing.assert_array_equal(
        snow_cover.time.values,
        xr.open_dataset(output_dir / "snow_cover_obs.nc")["snow_cover"].time.values,
    )


# ---------------------------------------------------------------------------
# shared helpers keep their previous behaviour
# ---------------------------------------------------------------------------
def test_resample_to_target_freq_defaults_keep_the_previous_anchors():
    """The default resampling anchors are unchanged."""
    times = pd.date_range("2000-01-01", "2000-06-30", freq="D")
    ds = xr.Dataset(
        {"v": ("time", np.arange(len(times), dtype=float))}, coords={"time": times}
    )

    weekly, _ = resample_to_target_freq(ds, ds.copy(), "W")
    monthly, _ = resample_to_target_freq(ds, ds.copy(), "ME")

    # the historic anchors: the Monday after the week, and the month end
    assert pd.Timestamp(weekly.time.values[0]) == pd.Timestamp("2000-01-03")
    assert pd.Timestamp(monthly.time.values[0]) == pd.Timestamp("2000-01-31")
    assert weekly.sizes["time"] == 27
    assert monthly.sizes["time"] == 6


def test_normalize_time_axis_period_start_is_opt_in():
    """Period start anchoring only applies when it is asked for."""
    times = pd.date_range("2000-01-01", "2000-06-30", freq="D")
    ds = xr.Dataset(
        {"v": ("time", np.arange(len(times), dtype=float))}, coords={"time": times}
    )
    weekly = ds.resample(time="W").mean()

    default_anchor = normalize_time_axis(weekly, "W")
    period_start = normalize_time_axis(weekly, "W", anchor="period_start")

    assert pd.Timestamp(default_anchor.time.values[0]) == pd.Timestamp("2000-01-03")
    # the first bin aggregates 1999-12-27 to 2000-01-02
    assert pd.Timestamp(period_start.time.values[0]) == pd.Timestamp("1999-12-27")


def test_resample_to_target_freq_drops_partially_covered_edge_steps():
    """A partly covered edge step is dropped when asked for."""
    input_da = _single_cell(
        np.arange(400.0), pd.date_range("2001-06-15", periods=400, freq="D")
    )
    ref_da = _single_cell(
        np.arange(24.0), pd.date_range("2001-01-31", periods=24, freq="ME")
    )

    kept, _ = resample_to_target_freq(input_da, ref_da, "ME", drop_partial_edges=False)
    trimmed, _ = resample_to_target_freq(
        input_da, ref_da, "ME", drop_partial_edges=True
    )

    # June is only covered from the 15th, so it would average 16 of 30 days
    assert pd.Timestamp(kept.time.values[0]) == pd.Timestamp("2001-06-30")
    assert pd.Timestamp(trimmed.time.values[0]) == pd.Timestamp("2001-07-31")
    assert trimmed.sizes["time"] == kept.sizes["time"] - 2
