import unittest

import numpy as np
import xarray as xr

from mhm_tools.common.time_utils import resample_to_daily_or_hourly_adaptive


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
