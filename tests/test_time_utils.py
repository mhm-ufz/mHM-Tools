import unittest

import numpy as np
import pandas as pd
import xarray as xr

from mhm_tools.common.time_utils import (
    get_time_bounds,
    infer_time_bounds,
    resample_to_daily_or_hourly_adaptive,
    resample_to_reference_windows,
    set_time_bounds,
    timedelta_to_alias,
)


class TestTimedeltaToAlias(unittest.TestCase):
    def make_time_da(self, start, periods, step):
        """
        Build a 1-D DataArray with a 'time' coord using fixed timedelta steps.

        start : str or np.datetime64 (e.g. '2021-01-01' or '2021-01-01T00')
        periods : int
        step : str like 'h', '6h', 'D', '7D', '30D' (NO 'W' or 'M')
        """
        # normalize to high precision to avoid odd dtype promotion
        start_ts = np.datetime64(start, "ns")

        # parse step like 'h'/'6h' or 'D'/'7D'/'30D'
        if step.endswith("h"):
            n = int(step[:-1]) if step != "h" else 1
            delta = np.timedelta64(n, "h")
        elif step.endswith("D"):
            n = int(step[:-1]) if step != "D" else 1
            delta = np.timedelta64(n, "D")
        else:
            raise ValueError("step must be 'h', 'Nh', 'D', or 'ND'")

        offsets = np.arange(periods) * delta
        time = start_ts + offsets
        return xr.DataArray(np.zeros(time.size), coords={"time": time}, dims=("time",))

    def test_daily(self):
        da = self.make_time_da("2021-01-01", periods=5, step="D")
        hours, alias = timedelta_to_alias(da)
        self.assertEqual(alias, "D")
        self.assertEqual(hours, 24)

    def test_weekly_every_7_days(self):
        # fixed 7-day step (not weekday-anchored)
        da = self.make_time_da("2021-01-01", periods=4, step="7D")
        hours, alias = timedelta_to_alias(da)
        self.assertEqual(alias, "W")
        self.assertEqual(hours, 24 * 7)

    def test_monthly_like_30d(self):
        # fixed 30-day step (calendar-ish, but deterministic)
        da = self.make_time_da("2021-01-01", periods=3, step="30D")
        hours, alias = timedelta_to_alias(da)
        self.assertEqual(alias, "ME")

    def test_fallback_hours(self):
        # 6-hourly → "<N>H"
        da = self.make_time_da("2021-01-01T00", periods=4, step="6h")
        hours, alias = timedelta_to_alias(da)
        self.assertEqual(hours, 6)
        self.assertEqual(alias, "6h")

    def test_raises_with_single_timestamp(self):
        da = self.make_time_da("2021-01-01", periods=1, step="D")
        with self.assertRaises(ValueError):
            timedelta_to_alias(da)


class TestResampleSingleTimestamp(unittest.TestCase):
    def test_dataset_returned_unchanged_when_single_timestamp(self):
        time = np.array([np.datetime64("2020-01-01")])
        data = xr.DataArray(
            np.ones((1, 2, 2)),
            dims=("time", "lat", "lon"),
            coords={"time": time, "lat": [0.0, 1.0], "lon": [10.0, 20.0]},
            name="foo",
            attrs={"units": "degC"},
        )
        ds = data.to_dataset()
        out = resample_to_daily_or_hourly_adaptive(ds, "daily", var="foo")
        self.assertIsInstance(out, xr.Dataset)
        xr.testing.assert_equal(out, ds)


class TimeWindowBase(unittest.TestCase):
    """Build records whose steps follow their own windows or the calendar."""

    LAT = [10.0, 20.0]
    LON = [1.0, 2.0]

    def make_cube(self, times, start=0.0):
        """Build a (time, lat, lon) record with a distinct value per step."""
        values = np.arange(len(times) * 4, dtype=float).reshape(len(times), 2, 2)
        return xr.DataArray(
            values + start,
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


class TestGetAndSetTimeBounds(TimeWindowBase):
    def test_cf_bounds_variable_of_a_dataset_is_read(self):
        times = pd.date_range("2004-01-15", periods=3, freq="ME")
        bounds = np.stack(
            [times.values - np.timedelta64(14, "D"), times.values], axis=1
        )
        ds = xr.Dataset(
            {"tws": ("time", np.zeros(3))},
            coords={"time": times, "time_bounds": (("time", "bnds"), bounds)},
        )
        ds["time"].attrs["bounds"] = "time_bounds"
        self.assertEqual(get_time_bounds(ds).shape, (3, 2))

    def test_bounds_survive_on_a_three_dimensional_data_array(self):
        # a CF bounds variable spans a second dimension that a DataArray of one
        # variable cannot carry, so the two edges travel as 1-D coordinates
        starts = pd.date_range("2004-01-01", periods=3, freq="MS")
        ref = self.make_windowed_ref(starts, starts + pd.Timedelta(days=29))
        self.assertEqual(ref.dims, ("time", "lat", "lon"))
        np.testing.assert_array_equal(
            get_time_bounds(ref)[:, 0], np.asarray(starts.values)
        )

    def test_record_without_bounds_returns_none(self):
        ref = self.make_cube(pd.date_range("2004-01-31", periods=3, freq="ME"))
        self.assertIsNone(get_time_bounds(ref))

    def test_bounds_attribute_naming_a_missing_variable_returns_none(self):
        ref = self.make_cube(pd.date_range("2004-01-31", periods=3, freq="ME"))
        ref.coords["time"].attrs["bounds"] = "not_there"
        self.assertIsNone(get_time_bounds(ref))

    def test_set_time_bounds_without_bounds_is_a_no_op(self):
        ref = self.make_cube(pd.date_range("2004-01-31", periods=3, freq="ME"))
        xr.testing.assert_identical(set_time_bounds(ref, None), ref)


class TestResampleToReferenceWindows(TimeWindowBase):
    def test_shifted_monthly_windows_keep_every_reference_step(self):
        starts = pd.date_range("2004-01-10", periods=12, freq="MS") + pd.Timedelta(
            days=9
        )
        ref = self.make_windowed_ref(starts, starts + pd.Timedelta(days=29))
        averaged, kept_ref = resample_to_reference_windows(self.make_daily_input(), ref)
        self.assertEqual(kept_ref.sizes["time"], ref.sizes["time"])
        np.testing.assert_array_equal(averaged["time"].values, ref["time"].values)

    def test_nearly_monthly_windows_keep_every_reference_step(self):
        generator = np.random.default_rng(0)
        starts = [pd.Timestamp("2004-01-02")]
        for _ in range(23):
            starts.append(
                starts[-1] + pd.Timedelta(days=int(generator.integers(28, 32)))
            )
        starts = pd.DatetimeIndex(starts)
        ref = self.make_windowed_ref(starts, starts + pd.Timedelta(days=27))
        _, kept_ref = resample_to_reference_windows(self.make_daily_input(), ref)
        self.assertEqual(kept_ref.sizes["time"], 24)

    def test_twelve_day_windows_keep_every_reference_step(self):
        starts = pd.date_range("2004-01-01", "2005-12-01", freq="12D")
        ref = self.make_windowed_ref(starts, starts + pd.Timedelta(days=11))
        _, kept_ref = resample_to_reference_windows(self.make_daily_input(), ref)
        self.assertEqual(kept_ref.sizes["time"], ref.sizes["time"])

    def test_two_windows_in_one_month_and_a_gap_are_both_kept(self):
        # the satellite case: neither window is merged, no month is invented
        starts = pd.to_datetime(
            ["2004-11-01", "2004-12-01", "2004-12-17", "2005-02-02"]
        )
        ends = pd.to_datetime(["2004-11-30", "2004-12-16", "2004-12-31", "2005-03-03"])
        ref = self.make_windowed_ref(starts, ends)
        _, kept_ref = resample_to_reference_windows(self.make_daily_input(), ref)
        self.assertEqual(kept_ref.sizes["time"], 4)

    def test_the_input_is_averaged_over_exactly_the_window(self):
        starts = pd.to_datetime(["2004-03-07"])
        ends = pd.to_datetime(["2004-04-05"])
        ref = self.make_windowed_ref(starts, ends)
        daily_input = self.make_daily_input()
        averaged, _ = resample_to_reference_windows(daily_input, ref)
        expected = daily_input.sel(time=slice(starts[0], ends[0])).mean("time")
        np.testing.assert_allclose(averaged.isel(time=0).values, expected.values)

    def test_a_window_the_input_does_not_cover_is_dropped(self):
        starts = pd.to_datetime(["2003-01-01", "2004-03-01"])
        ends = pd.to_datetime(["2003-01-30", "2004-03-30"])
        ref = self.make_windowed_ref(starts, ends)
        averaged, kept_ref = resample_to_reference_windows(self.make_daily_input(), ref)
        self.assertEqual(kept_ref.sizes["time"], 1)
        self.assertEqual(averaged.sizes["time"], 1)

    def test_no_covered_window_raises(self):
        starts = pd.to_datetime(["1990-01-01"])
        ref = self.make_windowed_ref(starts, starts + pd.Timedelta(days=29))
        with self.assertRaises(ValueError):
            resample_to_reference_windows(self.make_daily_input(), ref)

    def test_a_reference_without_bounds_has_its_windows_derived(self):
        ref = self.make_cube(pd.date_range("2004-01-31", periods=12, freq="ME"))
        _, kept_ref = resample_to_reference_windows(self.make_daily_input(), ref)
        self.assertEqual(kept_ref.sizes["time"], ref.sizes["time"])

    def test_an_input_coarser_than_the_windows_raises(self):
        starts = pd.date_range("2004-01-01", "2005-12-01", freq="12D")
        ref = self.make_windowed_ref(starts, starts + pd.Timedelta(days=11))
        monthly_input = self.make_cube(
            pd.date_range("2003-06-30", "2006-06-30", freq="ME")
        )
        with self.assertRaises(ValueError) as raised:
            resample_to_reference_windows(monthly_input, ref)
        self.assertIn("coarser", str(raised.exception))


class TestInferTimeBounds(TimeWindowBase):
    def test_evenly_spaced_stamps_give_windows_of_that_spacing(self):
        ref = self.make_cube(pd.date_range("2004-01-06", periods=10, freq="12D"))
        widths = np.diff(infer_time_bounds(ref), axis=1) / np.timedelta64(1, "D")
        np.testing.assert_allclose(widths.ravel(), 12.0)

    def test_calendar_monthly_stamps_give_the_real_month_lengths(self):
        ref = self.make_cube(pd.date_range("2004-01-31", periods=6, freq="ME"))
        widths = np.diff(infer_time_bounds(ref), axis=1) / np.timedelta64(1, "D")
        self.assertTrue(all(27 <= width <= 32 for width in widths.ravel()))

    def test_the_derived_windows_are_contiguous(self):
        ref = self.make_cube(pd.date_range("2004-01-31", periods=6, freq="ME"))
        bounds = infer_time_bounds(ref)
        np.testing.assert_array_equal(bounds[1:, 0], bounds[:-1, 1])

    def test_a_gap_raises_and_names_it(self):
        # deriving windows across a gap would claim instants never observed
        times = pd.to_datetime(
            ["2004-10-15", "2004-11-15", "2004-12-15", "2005-02-15", "2005-03-15"]
        )
        with self.assertRaises(ValueError) as raised:
            infer_time_bounds(self.make_cube(times))
        message = str(raised.exception)
        self.assertIn("2004-12-15", message)
        self.assertIn("bounds", message)

    def test_a_single_step_raises(self):
        ref = self.make_cube(pd.to_datetime(["2004-01-15"]))
        with self.assertRaises(ValueError):
            infer_time_bounds(ref)
