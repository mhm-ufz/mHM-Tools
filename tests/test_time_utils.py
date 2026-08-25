import unittest

import numpy as np
import xarray as xr

from mhm_tools.common.time_utils import (
    resample_to_daily_or_hourly_adaptive,
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
