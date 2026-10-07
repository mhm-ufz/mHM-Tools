# tests/test_xarray_utils_unittest.py
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import xarray as xr

import mhm_tools.common.xarray_utils as utils
from mhm_tools.common.netcdf import generate_bounds
from mhm_tools.common.xarray_utils import (
    crop_ds,
    get_coord_key,
    get_overlapping_time_slice,
    get_single_data_var,
    induce_data_var_from_file_name,
    normalize_lat_lon,
    normalize_longitude_range,
    regrid_mask,
    snap_to_target,
)


class XarrayUtilsBase(unittest.TestCase):
    def make_sample_ds(self, lat_key="lat", lon_key="lon"):
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


class TestNormalizeLatLon(XarrayUtilsBase):
    def test_normalize_lat_lon_renames(self):
        ds = xr.Dataset(
            coords={"latitude": [10, 20], "longitude": [100, 110]},
            data_vars={"z": (("latitude", "longitude"), np.ones((2, 2)))},
        )
        out = normalize_lat_lon(ds, lat_key="latitude", lon_key="longitude")
        self.assertIn("lat", out.coords)
        self.assertIn("lon", out.coords)
        self.assertNotIn("latitude", out.coords)
        self.assertNotIn("longitude", out.coords)
        self.assertEqual(out["z"].dims, ("lat", "lon"))

    def test_snap_to_target_renames_and_assigns_exact_coords(self):
        target_lat = np.array([3.0, 2.0, 1.0])
        target_lon = np.array([0.0, 1.0, 2.0])
        da = xr.DataArray(
            np.ones((3, 3)),
            dims=("latitude", "longitude"),
            coords={
                "latitude": target_lat + 1e-4,
                "longitude": target_lon + 1e-4,
            },
            name="mask",
        )

        out = snap_to_target(
            da,
            lat_key="latitude",
            lon_key="longitude",
            target_lat_array=target_lat,
            target_lon_array=target_lon,
            new_lat_key="lat",
            new_lon_key="lon",
        )

        self.assertEqual(out.dims, ("lat", "lon"))
        np.testing.assert_array_equal(out["lat"].values, target_lat)
        np.testing.assert_array_equal(out["lon"].values, target_lon)


class TestNormalizeLatLonCoordinateLayouts(XarrayUtilsBase):
    """Cover the coordinate layouts that leave a dimension without an index."""

    LAT = np.array([40.0, 30.0, 20.0, 10.0])
    LON = np.array([1.0, 2.0, 3.0])

    def make_aliased_coords_ds(self):
        """Dims lat/lon, but the coordinates are named latitude/longitude."""
        return xr.Dataset(
            {"tws": (("lat", "lon"), np.zeros((4, 3)))},
            coords={"latitude": ("lat", self.LAT), "longitude": ("lon", self.LON)},
        )

    def make_coord_data_vars_ds(self):
        """The coordinates are stored as data variables, only the dims point to them."""
        return xr.Dataset(
            {
                "tws": (("lat", "lon"), np.zeros((4, 3))),
                "latitude": ("lat", self.LAT),
                "longitude": ("lon", self.LON),
            }
        )

    def test_aliased_coordinate_becomes_an_indexed_dimension_coordinate(self):
        out = normalize_lat_lon(self.make_aliased_coords_ds())
        self.assertIn("lat", out.coords)
        self.assertNotIn("latitude", out.coords)
        # without an index every label based selection turns positional
        self.assertIn("lat", out.indexes)
        self.assertIn("lon", out.indexes)
        np.testing.assert_array_equal(out["lat"].values, self.LAT)

    def test_coordinate_data_variables_are_promoted_and_indexed(self):
        out = normalize_lat_lon(self.make_coord_data_vars_ds())
        self.assertEqual(tuple(out.data_vars), ("tws",))
        self.assertIn("lat", out.indexes)
        self.assertIn("lon", out.indexes)
        np.testing.assert_array_equal(out["lat"].values, self.LAT)

    def test_label_selection_works_after_normalization(self):
        for ds in (self.make_aliased_coords_ds(), self.make_coord_data_vars_ds()):
            out = normalize_lat_lon(ds)
            # a missing index made this silently positional and raise on a float
            cropped = crop_ds(out["tws"], 1.0, 3.0, 20.0, 40.0)
            self.assertEqual(cropped.sizes["lat"], 3)

    def test_descending_latitude_is_detected_after_normalization(self):
        for ds in (self.make_aliased_coords_ds(), self.make_coord_data_vars_ds()):
            out = normalize_lat_lon(ds)
            # positional indices would compare as ascending and never flip
            self.assertGreater(float(out["lat"][0]), float(out["lat"][-1]))

    def test_already_normalized_dataset_is_untouched(self):
        ds = xr.Dataset(
            {"tws": (("lat", "lon"), np.zeros((4, 3)))},
            coords={"lat": self.LAT, "lon": self.LON},
        )
        out = normalize_lat_lon(ds)
        self.assertEqual(tuple(out.coords), tuple(ds.coords))
        self.assertIn("lat", out.indexes)

    def test_existing_lat_coordinate_is_not_clobbered_by_an_alias(self):
        ds = xr.Dataset(
            {"tws": (("lat", "lon"), np.zeros((4, 3)))},
            coords={
                "lat": self.LAT,
                "lon": self.LON,
                "latitude": ("lat", self.LAT + 100),
            },
        )
        out = normalize_lat_lon(ds)
        np.testing.assert_array_equal(out["lat"].values, self.LAT)
        self.assertIn("latitude", out.coords)

    def test_unrelated_lat_dimension_blocks_the_rename(self):
        ds = xr.Dataset(
            {
                "tws": (("latitude", "lon"), np.zeros((4, 3))),
                "other": (("lat",), np.zeros(2)),
            },
            coords={"latitude": self.LAT, "lon": self.LON, "lat": np.array([0.0, 1.0])},
        )
        out = normalize_lat_lon(ds)
        # renaming latitude onto the unrelated lat dimension would collide
        self.assertIn("latitude", out.coords)
        np.testing.assert_array_equal(out["lat"].values, np.array([0.0, 1.0]))

    def test_two_dimensional_coordinates_are_left_without_an_index(self):
        ds = xr.Dataset(
            {"tws": (("y", "x"), np.zeros((4, 3)))},
            coords={
                "latitude": (("y", "x"), np.zeros((4, 3))),
                "longitude": (("y", "x"), np.zeros((4, 3))),
            },
        )
        out = normalize_lat_lon(ds)
        self.assertNotIn("lat", out.indexes)

    def test_dataset_without_lat_lon_is_returned_unchanged(self):
        ds = xr.Dataset({"tws": (("a", "b"), np.zeros((4, 3)))})
        out = normalize_lat_lon(ds)
        self.assertEqual(tuple(out.sizes), tuple(ds.sizes))

    def test_data_array_input_is_also_normalized(self):
        out = normalize_lat_lon(self.make_aliased_coords_ds()["tws"])
        self.assertIn("lat", out.indexes)


class TestRegridMask(XarrayUtilsBase):
    def test_regrid_mask_snaps_same_resolution_shifted_coordinates(self):
        target_lon = np.array([0.0, 1.0, 2.0])
        target_lat = np.array([3.0, 2.0, 1.0])
        mask = xr.DataArray(
            np.array(
                [
                    [1.0, 0.0, 1.0],
                    [1.0, 1.0, 0.0],
                    [0.0, 1.0, 1.0],
                ]
            ),
            dims=("lat", "lon"),
            coords={
                "lat": target_lat + 1e-4,
                "lon": target_lon + 1e-4,
            },
            name="mask",
        )

        regridded = regrid_mask(
            mask_ds=mask,
            lon_key_mask="lon",
            lat_key_mask="lat",
            target_lon=target_lon,
            target_lat=target_lat,
            lon_key_target="lon",
            lat_key_target="lat",
        )

        self.assertEqual(regridded.dims, ("lat", "lon"))
        np.testing.assert_allclose(regridded["lon"].values, target_lon)
        np.testing.assert_allclose(regridded["lat"].values, target_lat)
        np.testing.assert_array_equal(regridded.values, mask.values)

    def test_regrid_mask_uses_precomputed_resolution_for_single_point_target_axis(
        self,
    ):
        """A target axis narrowed to one point can't derive its own resolution by
        diffing; regrid_mask must use the caller-supplied target_res/mask_res
        instead of blindly re-deriving them from the (possibly length-1) arrays.
        """
        mask_lon = np.array([0.0, 0.5, 1.0, 1.5, 2.0])
        mask_lat = np.array([2.0, 1.5, 1.0, 0.5, 0.0])
        mask = xr.DataArray(
            np.ones((len(mask_lat), len(mask_lon))),
            dims=("lat", "lon"),
            coords={"lat": mask_lat, "lon": mask_lon},
            name="mask",
        )

        target_lon = np.array([1.0])
        target_lat = np.array([2.0, 1.0, 0.0])

        regridded = regrid_mask(
            mask_ds=mask,
            lon_key_mask="lon",
            lat_key_mask="lat",
            target_lon=target_lon,
            target_lat=target_lat,
            lon_key_target="lon",
            lat_key_target="lat",
            target_res=1.0,
            mask_res=0.5,
        )

        self.assertEqual(regridded.shape, (3, 1))
        np.testing.assert_array_equal(regridded.values, np.ones((3, 1)))

    def test_regrid_mask_fallback_derives_resolution_from_other_axis(self):
        """No target_res/mask_res given, and target_lon has only one point -
        regrid_mask's own fallback must derive resolution from target_lat
        instead of blindly indexing the length-1 target_lon, mirroring the
        real crash this was fixed for (a coordinate_slice narrowing only one
        axis to a point).
        """
        mask_lon = np.array([0.0, 1.0, 2.0])
        mask_lat = np.array([2.0, 1.0, 0.0])
        mask = xr.DataArray(
            np.ones((len(mask_lat), len(mask_lon))),
            dims=("lat", "lon"),
            coords={"lat": mask_lat, "lon": mask_lon},
            name="mask",
        )

        target_lon = np.array([1.0])
        target_lat = np.array([2.0, 1.0, 0.0])

        regridded = regrid_mask(
            mask_ds=mask,
            lon_key_mask="lon",
            lat_key_mask="lat",
            target_lon=target_lon,
            target_lat=target_lat,
            lon_key_target="lon",
            lat_key_target="lat",
        )

        self.assertEqual(regridded.shape, (3, 1))
        np.testing.assert_array_equal(regridded.values, np.ones((3, 1)))

    def test_regrid_mask_uses_provided_resolution_over_diffed_value(self):
        """A provided target_res must actually change the regridding window,
        not be silently recomputed - proven here since target_lon/target_lat
        both have a single point, so regrid_mask has no other way to obtain
        a resolution at all.
        """
        mask_lat = np.array([0.0])
        mask_lon = np.array([0.0, 3.0])
        mask = xr.DataArray(
            np.array([[0.0, 1.0]]),
            dims=("lat", "lon"),
            coords={"lat": mask_lat, "lon": mask_lon},
            name="mask",
        )
        target_lat = np.array([0.0])
        target_lon = np.array([0.0])

        narrow = regrid_mask(
            mask_ds=mask,
            lon_key_mask="lon",
            lat_key_mask="lat",
            target_lon=target_lon,
            target_lat=target_lat,
            lon_key_target="lon",
            lat_key_target="lat",
            target_res=1.0,
            mask_res=0.5,
        )
        wide = regrid_mask(
            mask_ds=mask,
            lon_key_mask="lon",
            lat_key_mask="lat",
            target_lon=target_lon,
            target_lat=target_lat,
            lon_key_target="lon",
            lat_key_target="lat",
            target_res=8.0,
            mask_res=0.5,
        )

        np.testing.assert_array_equal(narrow.values, np.zeros((1, 1)))
        np.testing.assert_array_equal(wide.values, np.ones((1, 1)))

    def test_regrid_mask_single_point_target_with_bounds_derived_resolution(self):
        """Resolution can also come from real CF bounds (as produced by
        generate_bounds elsewhere in this bounds-based architecture), not
        just a hand-picked literal or coordinate diffing.
        """
        mask_lon = np.array([0.0, 0.5, 1.0, 1.5, 2.0])
        mask_lat = np.array([2.0, 1.5, 1.0, 0.5, 0.0])
        mask = xr.DataArray(
            np.ones((len(mask_lat), len(mask_lon))),
            dims=("lat", "lon"),
            coords={"lat": mask_lat, "lon": mask_lon},
            name="mask",
        )
        mask_lon_bnds = generate_bounds(
            xr.DataArray(mask_lon, dims="lon", coords={"lon": mask_lon})
        )
        mask_res = float(
            np.abs(mask_lon_bnds.values[0, 1] - mask_lon_bnds.values[0, 0])
        )

        full_target_lat = np.array([2.0, 1.0, 0.0])
        target_lat_bnds = generate_bounds(
            xr.DataArray(full_target_lat, dims="lat", coords={"lat": full_target_lat})
        )
        target_res = float(
            np.abs(target_lat_bnds.values[0, 1] - target_lat_bnds.values[0, 0])
        )

        regridded = regrid_mask(
            mask_ds=mask,
            lon_key_mask="lon",
            lat_key_mask="lat",
            target_lon=np.array([1.0]),
            target_lat=full_target_lat,
            lon_key_target="lon",
            lat_key_target="lat",
            target_res=target_res,
            mask_res=mask_res,
        )

        self.assertEqual(regridded.shape, (3, 1))
        np.testing.assert_array_equal(regridded.values, np.ones((3, 1)))


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
        self.assertFalse("lat" in ds2.coords)
        self.assertFalse("lat" in ds2.dims)
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
        self.assertEqual(sl.start, pd.to_datetime("2021-01-03"))
        # For daily buckets the stop is expanded to the end of that day.
        self.assertEqual(sl.stop.date(), pd.to_datetime("2021-01-06").date())

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
        t = np.array(np.arange("2021-01-01", "2021-01-04", dtype="datetime64[D]"))
        ds1 = xr.Dataset(
            {"a": ("time", np.array([np.nan, np.nan, np.nan]))}, coords={"time": t}
        )
        ds2 = xr.Dataset(
            {"b": ("time", np.array([np.nan, np.nan, np.nan]))}, coords={"time": t}
        )
        with self.assertRaisesRegex(
            ValueError,
            "Cannot determine temporal overlap.*Input valid timesteps: 0; "
            "Reference valid timesteps: 0",
        ):
            get_overlapping_time_slice(ds1, ds2)

    def test_input_all_nan_reports_valid_timestep_counts(self):
        t = np.array(np.arange("2021-01-01", "2021-01-04", dtype="datetime64[D]"))
        ds1 = xr.Dataset(
            {"a": ("time", np.array([np.nan, np.nan, np.nan]))}, coords={"time": t}
        )
        ds2 = xr.Dataset({"b": ("time", np.ones(t.size))}, coords={"time": t})

        with self.assertRaisesRegex(
            ValueError,
            "Input valid timesteps: 0; Reference valid timesteps: 3",
        ):
            get_overlapping_time_slice(ds1, ds2)

    def test_daily_bucket_ignores_intraday_offsets(self):
        sim_time = pd.date_range("1990-01-01 11:00:00", periods=30, freq="D")
        obs_time = pd.date_range("1990-01-01 00:00:00", periods=31, freq="D")
        sim = xr.Dataset(
            {"a": ("time", np.ones(sim_time.size))}, coords={"time": sim_time}
        )
        obs = xr.Dataset(
            {"b": ("time", np.ones(obs_time.size))}, coords={"time": obs_time}
        )

        sl = get_overlapping_time_slice(sim, obs)
        self.assertEqual(sl.start, pd.to_datetime("1990-01-01 00:00:00"))
        self.assertEqual(sl.stop.date(), pd.to_datetime("1990-01-30").date())

    def test_month_bucket_ignores_day_in_month(self):
        sim_time = pd.to_datetime(["1990-01-01", "1990-02-01", "1990-03-01"])
        obs_time = pd.to_datetime(["1990-01-15", "1990-02-15", "1990-03-15"])
        sim = xr.Dataset(
            {"a": ("time", np.ones(sim_time.size))}, coords={"time": sim_time}
        )
        obs = xr.Dataset(
            {"b": ("time", np.ones(obs_time.size))}, coords={"time": obs_time}
        )

        sl = get_overlapping_time_slice(sim, obs)
        self.assertEqual(sl.start, pd.to_datetime("1990-01-01 00:00:00"))
        self.assertEqual(sl.stop.date(), pd.to_datetime("1990-03-31").date())


class TestCropDs(XarrayUtilsBase):
    def test_basic(self):
        ds = self.make_sample_ds()
        out = crop_ds(ds, lon_min=101, lon_max=103, lat_min=10.5, lat_max=12.0)
        self.assertEqual(out.sizes["lon"], 3)  # 101, 102, 103
        self.assertEqual(out.sizes["lat"], 2)  # 11, 12
        self.assertTrue(set(out.lon.values).issubset({101.0, 102.0, 103.0}))
        self.assertTrue(set(out.lat.values).issubset({11.0, 12.0}))

    def test_reversed_inputs(self):
        ds = self.make_sample_ds()
        out = crop_ds(ds, lon_min=103, lon_max=101, lat_min=12.0, lat_max=10.5)
        self.assertEqual(out.sizes["lon"], 3)  # 101, 102, 103
        self.assertEqual(out.sizes["lat"], 2)  # 11, 12
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


def _storage(units=None):
    """Create a one value storage field with the given units."""
    attrs = {"long_name": "storage"} if units is None else {"units": units}
    return xr.DataArray([2.0], dims="cell", attrs=attrs, name="tws")


@pytest.mark.parametrize(
    ("units", "mm"),
    [("cm", 20.0), ("m", 2000.0), ("kg m-2", 2.0), ("kg/m^2", 2.0)],
)
def test_convert_water_storage_to_mm_reads_depth_units(units, mm):
    """Convert depths of water and kg/m2 to mm."""
    converted = utils.convert_water_storage_to_mm(_storage(units))
    np.testing.assert_allclose(converted.values, mm)
    assert converted.attrs["units"] == "mm"


def test_convert_water_storage_to_mm_prefers_the_scale_factor():
    """Use an explicit scale factor instead of the units attribute."""
    converted = utils.convert_water_storage_to_mm(_storage("m"), scale_factor=10)
    np.testing.assert_allclose(converted.values, 20.0)


def test_convert_water_storage_to_mm_falls_back_to_the_encoding_units():
    """Read the units from the encoding when the attribute is gone."""
    storage = _storage()
    storage.encoding["units"] = "cm"
    np.testing.assert_allclose(utils.convert_water_storage_to_mm(storage).values, 20.0)


def test_convert_water_storage_to_mm_rejects_unknown_units():
    """Refuse a storage whose units are no depth of water."""
    with pytest.raises(ValueError, match="unrecognized unit"):
        utils.convert_water_storage_to_mm(_storage("m3"))


class TestNormalizeLongitudeRange(unittest.TestCase):
    @staticmethod
    def make_ds(lon):
        """Create a two-row dataset whose values hold the index of their column.

        Args:
            lon (numpy.ndarray): Longitude axis.

        Returns:
            Dataset with one lat/lon data variable.
        """
        values = np.tile(np.arange(lon.size, dtype=float), (2, 1))
        return xr.Dataset(
            {"var": (("lat", "lon"), values)}, coords={"lat": [1.0, 0.0], "lon": lon}
        )

    def test_projected_x_axis_in_metres_is_left_unchanged(self):
        """Keep values and order of an x axis in metres, which holds no degrees."""
        ds = self.make_ds(4_000_100.0 + 200.0 * np.arange(50))
        xr.testing.assert_identical(normalize_longitude_range(ds), ds)

    def test_axis_reaching_below_minus_180_is_left_unchanged(self):
        """Keep a projected axis around 0 m that reaches below -180."""
        ds = self.make_ds(np.linspace(-500_000.0, 500_000.0, 11))
        xr.testing.assert_identical(normalize_longitude_range(ds), ds)

    def test_global_0_to_360_axis_is_still_rotated(self):
        """Rotate a global 0 to 360 axis onto -180 to 180 with its values."""
        ds = self.make_ds(np.arange(0.5, 360.0, 1.0))
        rotated = normalize_longitude_range(ds)
        np.testing.assert_array_equal(
            rotated["lon"].values, np.arange(-179.5, 180.0, 1.0)
        )
        # the values move with their column, 180.5 E becomes -179.5
        np.testing.assert_array_equal(
            rotated["var"].sel(lon=-179.5).values, ds["var"].sel(lon=180.5).values
        )


if __name__ == "__main__":
    # Allows running directly: python -m unittest tests/test_xarray_utils_unittest.py
    unittest.main()
