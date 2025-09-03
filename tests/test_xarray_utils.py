# tests/test_xarray_utils_unittest.py
import unittest
from pathlib import Path
import numpy as np
import pandas as pd
import xarray as xr

import mhm_tools.common.xarray_utils as utils
from mhm_tools.common.xarray_utils import (
    normalize_lat_lon,
    get_coord_key,
    get_single_data_var,
    induce_data_var_from_file_name,
    timedelta_to_alias,
    get_overlapping_time_slice,
    crop_ds,
)

class XarrayUtilsBase(unittest.TestCase):
    def make_sample_ds(self, lat_key='lat', lon_key='lon'):
        # 3x4 grid, ascending coords
        lat = np.array([10.0, 11.0, 12.0])
        lon = np.array([100.0, 101.0, 102.0, 103.0])
        time = np.array(np.arange("2021-01-01", "2021-01-04", dtype="datetime64[D]"))

        data = xr.DataArray(
            np.random.rand(time.size, lat.size, lon.size),
            dims=("time", lat_key, lon_key),
            coords={"time": time, lat_key: lat, lon_key: lon},
            name="var",
        )
        return data.to_dataset()

    def make_descending_ds(self):
        # descending coords
        lat = np.array([12.0, 11.0, 10.0])
        lon = np.array([103.0, 102.0, 101.0, 100.0])
        time = np.array(np.arange("2021-01-01", "2021-01-04", dtype="datetime64[D]"))
        da = xr.DataArray(
            np.random.rand(time.size, lat.size, lon.size),
            dims=("time", "lat", "lon"),
            coords={"time": time, "lat": lat, "lon": lon},
            name="var",
        )
        return da.to_dataset()

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



class TestNormalizeLatLon(XarrayUtilsBase):
    def test_normalize_lat_lon_renames(self):
        ds = xr.Dataset(
            coords={"latitude": [10, 20], "longitude": [100, 110]},
            data_vars={"z": (("latitude", "longitude"), np.ones((2, 2)))},
        )
        out = normalize_lat_lon(ds, lat="latitude", lon="longitude")
        self.assertIn("lat", out.coords)
        self.assertIn("lon", out.coords)
        self.assertNotIn("latitude", out.coords)
        self.assertNotIn("longitude", out.coords)
        self.assertEqual(out["z"].dims, ("lat", "lon"))


class TestGetCoordKey(XarrayUtilsBase):
    def test_get_coord_key_lat_lon_time_from_names(self):
        ds = self.make_sample_ds()
        self.assertEqual(get_coord_key(ds, lat=True), "lat")
        self.assertEqual(get_coord_key(ds, lon=True), "lon")
        self.assertEqual(get_coord_key(ds, time=True), "time")

    def test_get_coord_key_raises_on_bad_flags(self):
        ds = self.make_sample_ds()
        with self.assertRaises(ValueError):
            get_coord_key(ds, lat=True, lon=True)
        with self.assertRaises(ValueError):
            get_coord_key(ds, lat=False, lon=False, time=False)

    def test_get_coord_key_from_dims_retry(self):
        # dims present, but not as coordinate variables
        arr = xr.DataArray(np.zeros((2, 3)), dims=("y", "x"), name="foo")
        ds = arr.to_dataset()
        self.assertEqual(get_coord_key(ds, lat=True), "y")
        self.assertEqual(get_coord_key(ds, lon=True), "x")

    def test_get_coord_key_no_raise_returns_none(self):
        ds = self.make_sample_ds()
        ds2 = ds.drop_dims("lat")
        self.assertFalse('lat' in ds2.coords)
        self.assertFalse('lat' in ds2.dims)
        lat_coord = get_coord_key(ds2, lat=True, raise_exception=False)
        self.assertIsNone(lat_coord)


class TestGetSingleDataVar(XarrayUtilsBase):
    def test_single_var(self):
        ds = self.make_sample_ds()
        self.assertEqual(get_single_data_var(ds), "var")

    def test_multiple_but_coords_included(self):
        ds = xr.Dataset(
            {
                "lat": ("lat", np.array([10, 11, 12])),
                "lon": ("lon", np.array([100, 101, 102, 103])),
                "target": (("lat", "lon"), np.random.rand(3, 4)),
            }
        )
        self.assertEqual(get_single_data_var(ds), "target")

    def test_multiple_real_vars_returns_none(self):
        ds = xr.Dataset(
            {
                "a": (("lat", "lon"), np.random.rand(2, 2)),
                "b": (("lat", "lon"), np.random.rand(2, 2)),
            },
            coords={"lat": [0, 1], "lon": [0, 1]},
        )
        self.assertIsNone(get_single_data_var(ds))

    def test_no_vars_returns_none(self):
        self.assertIsNone(get_single_data_var(xr.Dataset()))


class TestInduceDataVarFromFileName(XarrayUtilsBase):
    def test_exact_match(self):
        ds = xr.Dataset({"precip": (("lat", "lon"), np.ones((2, 2)))})
        dv = induce_data_var_from_file_name(ds, Path("precip_daily.nc"))
        self.assertEqual(dv, "precip")

    def test_name_contains_dv(self):
        ds = xr.Dataset({"temperature_mean": (("lat", "lon"), np.ones((2, 2)))})
        dv = induce_data_var_from_file_name(ds, Path("temp.nc"))
        self.assertEqual(dv, "temperature_mean")

    def test_no_match(self):
        ds = xr.Dataset({"u": (("lat", "lon"), np.ones((2, 2)))})
        self.assertIsNone(induce_data_var_from_file_name(ds, Path("v_component.nc")))


class TestTimedeltaToAlias(XarrayUtilsBase):
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
        self.assertEqual(alias, "6H")


class TestGetOverlappingTimeSlice(XarrayUtilsBase):
    def test_normal_overlap(self):
        t1 = np.array(np.arange("2021-01-01", "2021-01-07", dtype="datetime64[D]"))
        self.assertEqual(t1[0], np.datetime64("2021-01-01"))
        self.assertEqual(str(t1[-1]), "2021-01-06")
        t2 = np.array(np.arange("2021-01-03", "2021-01-09", dtype="datetime64[D]"))
        self.assertEqual(str(t2[0]), "2021-01-03")
        self.assertEqual(str(t2[-1]), "2021-01-08")
        ds1 = xr.Dataset({"a": ("time", np.ones(t1.size))}, coords={"time": t1})
        ds2 = xr.Dataset({"b": ("time", np.ones(t2.size))}, coords={"time": t2})
        sl = get_overlapping_time_slice(ds1, ds2)
        print(sl)
        self.assertIsInstance(sl, slice)
        self.assertEqual(sl.start, "2021-01-03")
        self.assertEqual(sl.stop, "2021-01-06")

    def test_no_overlap_logs_warning(self):
        t1 = np.array(np.arange("2021-01-01", "2021-01-03", dtype="datetime64[D]"))
        t2 = np.array(np.arange("2021-01-05", "2021-01-07", dtype="datetime64[D]"))
        ds1 = xr.Dataset({"a": ("time", np.ones(t1.size))}, coords={"time": t1})
        ds2 = xr.Dataset({"b": ("time", np.ones(t2.size))}, coords={"time": t2})
        with self.assertLogs(utils.logger.name, level="WARNING") as cm:
            sl = get_overlapping_time_slice(ds1, ds2)
        self.assertTrue(any("not overlapping" in msg for msg in cm.output))
        self.assertIsInstance(sl, slice)
        # function returns a slice even when not overlapping
        self.assertGreaterEqual(sl.start, sl.stop)

    def test_all_nan_raises(self):
        # Note: the implementation path uses a context manager; if it is misused,
        # any Exception is acceptable here.
        t = np.array(np.arange("2021-01-01", "2021-01-04", dtype="datetime64[D]"))
        ds1 = xr.Dataset({"a": ("time", np.array([np.nan, np.nan, np.nan]))}, coords={"time": t})
        ds2 = xr.Dataset({"b": ("time", np.array([np.nan, np.nan, np.nan]))}, coords={"time": t})
        with self.assertRaises(Exception):
            get_overlapping_time_slice(ds1, ds2)


class TestCropDs(XarrayUtilsBase):
    def test_basic(self):
        ds = self.make_sample_ds()
        out = crop_ds(ds, lon_min=101, lon_max=103, lat_min=10.5, lat_max=12.0)
        self.assertEqual(out.dims["lon"], 3)  # 101, 102, 103
        self.assertEqual(out.dims["lat"], 2)  # 11, 12
        self.assertTrue(set(out.lon.values).issubset({101.0, 102.0, 103.0}))
        self.assertTrue(set(out.lat.values).issubset({11.0, 12.0}))

    def test_reversed_inputs(self):
        ds = self.make_sample_ds()
        out = crop_ds(ds, lon_min=103, lon_max=101, lat_min=12.0, lat_max=10.5)
        self.assertEqual(out.dims["lon"], 3)  # 101, 102, 103
        self.assertEqual(out.dims["lat"], 2)  # 11, 12
        self.assertTrue(set(out.lon.values).issubset({101.0, 102.0, 103.0}))
        self.assertTrue(set(out.lat.values).issubset({11.0, 12.0}))

    def test_descending_axes(self):
        ds = self.make_descending_ds()
        out = crop_ds(ds, lon_min=101, lon_max=103, lat_min=10.5, lat_max=12.0)
        self.assertSetEqual(set(np.round(out.lon.values, 6)), {101.0, 102.0, 103.0})
        self.assertSetEqual(set(np.round(out.lat.values, 6)), {11.0, 12.0})

    def test_custom_coord_names(self):
        ds = xr.Dataset(
            coords={"X": [100.0, 101.0, 102.0], "Y": [10.0, 11.0, 12.0]},
            data_vars={"v": (("Y", "X"), np.random.rand(3, 3))},
        )
        out = crop_ds(ds, 100.5, 102.0, 10.5, 12.0, lon_name="X", lat_name="Y")
        self.assertSetEqual(set(np.round(out.X.values, 6)), {101.0, 102.0})
        self.assertSetEqual(set(np.round(out.Y.values, 6)), {11.0, 12.0})


if __name__ == "__main__":
    # Allows running directly: python -m unittest tests/test_xarray_utils_unittest.py
    unittest.main()
