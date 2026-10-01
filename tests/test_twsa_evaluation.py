"""Tests for the total water storage anomaly evaluation."""

import unittest

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from mhm_tools._cli._twsa_evaluation import select_metrics
from mhm_tools.common.time_utils import set_time_bounds
from mhm_tools.post.twsa_evaluation import (
    AVAILABLE_METRICS,
    KGE_COMPONENTS,
    resample_twsa_to_reference_calendar,
    twsa_evaluation,
)


def write_storage_record(file_path, times, seed):
    """Write a small total water storage record in mm to a NetCDF file."""
    generator = np.random.default_rng(seed)
    values = 500.0 + 50.0 * generator.normal(size=(len(times), 2, 2))
    record = xr.DataArray(
        values,
        dims=("time", "lat", "lon"),
        coords={"time": times, "lat": [10.0, 11.0], "lon": [20.0, 21.0]},
        name="tws",
        attrs={"units": "mm", "long_name": "total water storage"},
    )
    record.to_dataset().to_netcdf(file_path)


class TestResampleToReferenceCalendar(unittest.TestCase):
    """The reference sets the calendar, through its bounds or the months."""

    LAT = [10.0, 20.0]
    LON = [1.0, 2.0]

    def make_cube(self, times):
        """Build a (time, lat, lon) record with a distinct value per step."""
        values = np.arange(len(times) * 4, dtype=float).reshape(len(times), 2, 2)
        return xr.DataArray(
            values,
            dims=("time", "lat", "lon"),
            coords={"time": times, "lat": self.LAT, "lon": self.LON},
        )

    def make_windowed_ref(self, starts, ends):
        """Build a record stamped on its window centres and carrying its bounds."""
        starts, ends = pd.DatetimeIndex(starts), pd.DatetimeIndex(ends)
        centres = starts + (ends - starts) / 2
        bounds = np.stack([starts.values, ends.values], axis=1)
        return set_time_bounds(self.make_cube(centres), bounds)

    def make_daily_input(self):
        return self.make_cube(pd.date_range("2003-06-01", "2006-06-30", freq="D"))

    def test_calendar_months_without_bounds_stay_on_the_months(self):
        reference = self.make_cube(pd.date_range("2004-01-31", periods=24, freq="ME"))
        averaged, kept_ref = resample_twsa_to_reference_calendar(
            self.make_daily_input(), reference
        )
        self.assertEqual(averaged.sizes["time"], kept_ref.sizes["time"])
        spacing = np.diff(pd.DatetimeIndex(averaged["time"].values))
        self.assertTrue(
            all(27 <= step / np.timedelta64(1, "D") <= 31 for step in spacing)
        )

    def test_twelve_day_steps_without_bounds_stay_twelve_daily(self):
        # the reference sets the calendar, so its steps are neither merged into
        # months nor rejected; their windows come from the even spacing
        reference = self.make_cube(
            pd.date_range("2004-01-06", "2005-12-31", freq="12D")
        )
        averaged, kept_ref = resample_twsa_to_reference_calendar(
            self.make_daily_input(), reference
        )
        self.assertEqual(kept_ref.sizes["time"], reference.sizes["time"])
        self.assertEqual(averaged.sizes["time"], kept_ref.sizes["time"])
        spacing = np.diff(pd.DatetimeIndex(averaged["time"].values))
        np.testing.assert_allclose(spacing / np.timedelta64(1, "D"), 12.0)

    def test_drifting_monthly_steps_without_bounds_keep_their_own_rhythm(self):
        # evenly spaced but not on the calendar, so the windows are derivable
        starts = pd.date_range("2004-01-02", periods=24, freq="30D")
        reference = self.make_cube(starts)
        _, kept_ref = resample_twsa_to_reference_calendar(
            self.make_daily_input(), reference
        )
        self.assertEqual(kept_ref.sizes["time"], reference.sizes["time"])

    def test_a_gap_without_bounds_raises(self):
        # a satellite record missing a month cannot have its windows derived
        times = pd.to_datetime(
            ["2004-10-15", "2004-11-15", "2004-12-15", "2005-02-15", "2005-03-15"]
        )
        with self.assertRaises(ValueError) as raised:
            resample_twsa_to_reference_calendar(
                self.make_daily_input(), self.make_cube(times)
            )
        self.assertIn("bounds", str(raised.exception))

    def test_the_same_gappy_record_works_once_it_states_its_bounds(self):
        starts = pd.to_datetime(
            ["2004-10-01", "2004-11-01", "2004-12-01", "2005-02-01", "2005-03-01"]
        )
        reference = self.make_windowed_ref(starts, starts + pd.Timedelta(days=29))
        _, kept_ref = resample_twsa_to_reference_calendar(
            self.make_daily_input(), reference
        )
        self.assertEqual(kept_ref.sizes["time"], 5)

    def test_shifted_monthly_bounds_keep_every_reference_step(self):
        starts = pd.date_range("2004-01-10", periods=12, freq="MS") + pd.Timedelta(
            days=9
        )
        reference = self.make_windowed_ref(starts, starts + pd.Timedelta(days=29))
        averaged, kept_ref = resample_twsa_to_reference_calendar(
            self.make_daily_input(), reference
        )
        self.assertEqual(kept_ref.sizes["time"], reference.sizes["time"])
        np.testing.assert_array_equal(averaged["time"].values, reference["time"].values)

    def test_nearly_monthly_bounds_keep_every_reference_step(self):
        generator = np.random.default_rng(0)
        starts = [pd.Timestamp("2004-01-02")]
        for _ in range(23):
            starts.append(
                starts[-1] + pd.Timedelta(days=int(generator.integers(28, 32)))
            )
        starts = pd.DatetimeIndex(starts)
        reference = self.make_windowed_ref(starts, starts + pd.Timedelta(days=27))
        _, kept_ref = resample_twsa_to_reference_calendar(
            self.make_daily_input(), reference
        )
        self.assertEqual(kept_ref.sizes["time"], 24)

    def test_twelve_day_bounds_keep_every_reference_step(self):
        starts = pd.date_range("2004-01-01", "2005-12-01", freq="12D")
        reference = self.make_windowed_ref(starts, starts + pd.Timedelta(days=11))
        _, kept_ref = resample_twsa_to_reference_calendar(
            self.make_daily_input(), reference
        )
        self.assertEqual(kept_ref.sizes["time"], reference.sizes["time"])

    def test_bounds_win_over_the_calendar_when_both_would_work(self):
        starts = pd.date_range("2004-01-01", periods=12, freq="MS")
        ends = starts + pd.Timedelta(days=20)
        reference = self.make_windowed_ref(starts, ends)
        averaged, _ = resample_twsa_to_reference_calendar(
            self.make_daily_input(), reference
        )
        daily_input = self.make_daily_input()
        expected = daily_input.sel(time=slice(starts[0], ends[0])).mean("time")
        np.testing.assert_allclose(averaged.isel(time=0).values, expected.values)


def test_select_metrics_resolves_all_to_every_metric():
    """The cli keyword 'all' stands for every available metric."""
    assert select_metrics("all") == list(AVAILABLE_METRICS)


def test_select_metrics_selects_nothing_for_none_and_an_empty_string():
    """Both ways of asking for no metric give an empty selection."""
    assert select_metrics("none") == []
    assert select_metrics("") == []


def test_select_metrics_returns_the_fixed_order():
    """A list is resolved into the order the metrics are declared in."""
    assert select_metrics("rmse, kge") == ["kge", "rmse"]


def test_select_metrics_resolves_names_case_insensitively():
    """A name is matched regardless of how it was typed."""
    assert select_metrics("KGE") == ["kge"]


def test_select_metrics_rejects_an_unknown_name():
    """An unknown metric is refused with the available ones named."""
    with pytest.raises(ValueError, match=", ".join(AVAILABLE_METRICS)):
        select_metrics("spaef")


def test_twsa_evaluation_rejects_an_unknown_metric():
    """The name check fires before an input path is read."""
    with pytest.raises(ValueError, match="Available metrics"):
        twsa_evaluation(
            "missing_input", "missing_ref", "missing_out", metrics=["spaef"]
        )


def test_twsa_evaluation_rejects_a_run_without_metrics_and_fields():
    """No metric and no anomaly fields leaves nothing to calculate."""
    with pytest.raises(ValueError, match="nothing to calculate"):
        twsa_evaluation(
            "missing_input", "missing_ref", "missing_out", metrics=[], write_twsa=False
        )


def test_twsa_evaluation_accepts_one_metric_name_as_a_string():
    """A bare name passes the check, so the run stops on the missing path."""
    with pytest.raises(ValueError, match="does not exist"):
        twsa_evaluation("missing_input", "missing_ref", "missing_out", metrics="kge")


def _write_evaluation_input(work_dir):
    """Write a daily model record and a monthly reference record in mm."""
    input_file = work_dir / "input.nc"
    ref_file = work_dir / "ref.nc"
    write_storage_record(
        input_file, pd.date_range("2003-06-01", "2007-01-31", freq="D"), seed=0
    )
    write_storage_record(
        ref_file, pd.date_range("2004-01-31", periods=36, freq="ME"), seed=1
    )
    return input_file, ref_file


def _evaluate(work_dir, metrics):
    """Run the evaluation into its own output dir and return that dir."""
    input_file, ref_file = _write_evaluation_input(work_dir)
    output_dir = work_dir / f"out_{'_'.join(metrics)}"
    twsa_evaluation(
        input_path=str(input_file),
        ref_path=str(ref_file),
        output_dir=str(output_dir),
        input_name="mhm",
        ref_name="grace",
        input_format="tws",
        ref_format="tws",
        baseline_start_year=2004,
        baseline_end_year=2005,
        min_valid_months=12,
        regions="none",
        metrics=metrics,
        write_twsa=False,
    )
    return output_dir


def _written_metrics(output_dir):
    """Return the variable names the written metrics file holds."""
    with xr.open_dataset(output_dir / "twsa_metrics_mhm_vs_grace.nc") as ds:
        return set(ds.data_vars)


def _plotted_metrics(output_dir):
    """Return the metric names the written maps are named after."""
    return {
        path.name.split("twsa_metric_")[1].split("_mhm_vs_grace")[0]
        for path in output_dir.glob("twsa_metric_*.png")
    }


def test_selecting_only_the_error_writes_and_plots_only_the_error(tmp_path):
    """An error only run still states how many months a cell was scored on."""
    output_dir = _evaluate(tmp_path, ["rmse"])
    assert _written_metrics(output_dir) == {"rmse", "compared_months"}
    assert _plotted_metrics(output_dir) == {"rmse"}


def test_selecting_only_the_kge_leaves_the_error_out(tmp_path):
    """The KGE is written with its components, but only the KGE is plotted."""
    output_dir = _evaluate(tmp_path, ["kge"])
    written = _written_metrics(output_dir)
    assert {"kge", *KGE_COMPONENTS} <= written
    assert "rmse" not in written
    assert _plotted_metrics(output_dir) == {"kge"}


def test_selecting_both_metrics_writes_them_into_one_file(tmp_path):
    """Both metrics share the metrics file and get a map each."""
    output_dir = _evaluate(tmp_path, ["kge", "rmse"])
    written = _written_metrics(output_dir)
    assert {"kge", *KGE_COMPONENTS, "rmse", "compared_months"} <= written
    assert _plotted_metrics(output_dir) == {"kge", "rmse"}


if __name__ == "__main__":
    unittest.main()
