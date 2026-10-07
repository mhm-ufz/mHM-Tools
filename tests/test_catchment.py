import csv
import tempfile
import unittest
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import xarray as xr
from shapely.geometry import box

import mhm_tools.common.utils
from mhm_tools import __version__
from mhm_tools.common.file_handler import get_xarray_ds_from_file
from mhm_tools.common.provenance import CREATED_ATTR, HISTORY_ATTR, VERSION_ATTR
from mhm_tools.common.utils import (
    distance_100m_units,
    find_best_gauge_location_by_area,
    get_candidate_search_window,
)
from mhm_tools.common.xarray_utils import get_coord_key
from mhm_tools.pre import catchment

HERE = Path(__file__).parent


class TestCatchment(unittest.TestCase):
    def _require_geospatial(self):
        try:
            import geopandas  # noqa: F401
            import rasterio  # noqa: F401
        except Exception as exc:
            self.skipTest(f"Geospatial dependencies missing: {exc}")

    def _make_small_catchment(self):
        lon = np.array([0, 1, 2, 3, 4], dtype=float)
        lat = np.array([4, 3, 2, 1, 0], dtype=float)
        data = np.zeros((len(lat), len(lon)), dtype=float)
        ds = xr.Dataset(
            {"dem": (["lat", "lon"], data)},
            coords={"lon": lon, "lat": lat},
        )
        return catchment.Catchment(
            ds,
            "dem",
            var="dem",
            ftype="ldd",
            transform=(1.0, 0.0, 0.0, 0.0, 1.0, 0.0),
            latlon=True,
        )

    def _make_small_catchment(self):
        lon = np.array([0, 1, 2, 3, 4], dtype=float)
        lat = np.array([4, 3, 2, 1, 0], dtype=float)
        data = np.zeros((len(lat), len(lon)), dtype=float)
        ds = xr.Dataset(
            {"dem": (["lat", "lon"], data)},
            coords={"lon": lon, "lat": lat},
        )
        return catchment.Catchment(
            ds,
            "dem",
            var="dem",
            ftype="ldd",
            transform=(1.0, 0.0, 0.0, 0.0, 1.0, 0.0),
            latlon=True,
        )

    def setUp(self):
        lon = np.linspace(-180, 180, 360)
        lat = np.linspace(90, -90, 180)
        data = np.random.rand(180, 360)
        self.ds = xr.Dataset(
            {
                "dem": (["lat", "lon"], data),
                "flwdir": (["lat", "lon"], np.random.randint(1, 9, size=data.shape)),
            },
            coords={
                "lon": lon,
                "lat": lat,
            },
        )
        self.var_name = "dem"
        self.ftype = "ldd"
        self.transform = (0.05, 0.0, -180, 0, 0.05, -90)
        self.out_var_name = None
        self.latlon = True
        self._tmpdir = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmpdir.cleanup)
        self.tmp_path = Path(self._tmpdir.name)
        self.output_path = self.tmp_path / "files"
        self.output_path.mkdir(parents=True, exist_ok=True)
        super().setUp()

    def test_initialization(self):
        c = catchment.Catchment(
            self.ds,
            self.var_name,
            var="dem",
            ftype=self.ftype,
            transform=self.transform,
            out_var_name=self.out_var_name,
            latlon=self.latlon,
        )
        self.assertIsNotNone(c)
        self.assertIs(c.ds, self.ds)

    def test_modify_data(self):
        c = catchment.Catchment(
            self.ds,
            self.var_name,
            var="dem",
            ftype=self.ftype,
            transform=self.transform,
            out_var_name=self.out_var_name,
            latlon=self.latlon,
            do_shift=True,
        )
        modified_data = c._modify_data(self.ds[self.var_name])
        self.assertEqual(modified_data.shape, self.ds[self.var_name].shape)

    def test_add_fdir(self):
        c = catchment.Catchment(
            self.ds,
            "flwdir",
            var="fdir",
            ftype=self.ftype,
            transform=self.transform,
            out_var_name=self.out_var_name,
            latlon=self.latlon,
        )
        self.assertIsNotNone(c._fdir)

    def test_add_dem(self):
        c = catchment.Catchment(
            self.ds,
            self.var_name,
            var="dem",
            ftype=self.ftype,
            transform=self.transform,
            out_var_name=self.out_var_name,
            latlon=self.latlon,
        )
        self.assertIsNotNone(c.elevtn)
        self.assertIsNotNone(c._fdir)

    def test_get_basins(self):
        c = catchment.Catchment(
            self.ds,
            self.var_name,
            var="dem",
            ftype=self.ftype,
            transform=self.transform,
            out_var_name=self.out_var_name,
            latlon=self.latlon,
        )
        c.get_basins()
        self.assertIsNotNone(c.basin)

    def test_write(self):
        output_var_names = ["hydro1.nc", "hydro2.nc"]

        catchments = [
            catchment.Catchment(
                self.ds,
                self.var_name,
                var="dem",
                ftype=self.ftype,
                transform=self.transform,
                out_var_name=output_var_names[0],
                latlon=self.latlon,
            ),
            catchment.Catchment(
                self.ds,
                self.var_name,
                var="dem",
                ftype=self.ftype,
                transform=self.transform,
                out_var_name=output_var_names[1],
                latlon=self.latlon,
                do_shift=True,
            ),
        ]

        for c, out_var_name in zip(catchments, output_var_names):
            c.get_basins()
            c.get_facc()
            c.get_grid_area()
            c.get_upstream_area()
            c.write(self.output_path, single_file=True)
            output_file = self.output_path / out_var_name
            self.assertTrue(output_file.exists(), f"Failed to create {out_var_name}")

    def test_write_mask_file_hard_links_land_mask_to_mask(self):
        """land_mask and mask must be readable independently without duplicating data.

        Regression test: write_mask_file used to write the same mask array
        under both the 'land_mask' and 'mask' NetCDF variable names, which
        are pure synonyms consumed by different downstream tools, silently
        duplicating the mask data on disk. It should now write the array
        once (as 'mask') and add 'land_mask' as an HDF5 hard link to that
        same on-disk object, so both names remain fully independently
        readable while sharing one copy of the data.
        """
        import h5py

        c = catchment.Catchment(
            self.ds,
            self.var_name,
            var="dem",
            ftype=self.ftype,
            transform=self.transform,
            out_var_name=self.out_var_name,
            latlon=self.latlon,
        )
        lat = np.array([2.0, 1.0, 0.0])
        lon = np.array([0.0, 1.0])
        basin = xr.DataArray(
            np.array([[1, 0], [1, 1], [0, 0]]),
            coords={"lat": lat, "lon": lon},
            dims=["lat", "lon"],
        )
        fake_ds = xr.Dataset({"basin": basin}, coords={"lat": lat, "lon": lon})
        mask_file = self.tmp_path / "mask.nc"

        c.write_mask_file(fake_ds, mask_file)

        # mask.nc encodes 0 as the NetCDF _FillValue (NC_ENCODE_MASK), so
        # outside-catchment cells round-trip as NaN, not 0.
        expected = np.where(basin.values > 0, 1.0, np.nan)
        with get_xarray_ds_from_file(mask_file) as out:
            self.assertIn("land_mask", out.data_vars)
            self.assertIn("mask", out.data_vars)
            np.testing.assert_array_equal(out["mask"].values, expected)
            np.testing.assert_array_equal(out["land_mask"].values, out["mask"].values)

        with h5py.File(mask_file, "r") as f:
            self.assertIsInstance(f.get("land_mask", getlink=True), h5py.HardLink)
            addr_mask = h5py.h5o.get_info(f["mask"].id).addr
            addr_land_mask = h5py.h5o.get_info(f["land_mask"].id).addr
            self.assertEqual(
                addr_mask,
                addr_land_mask,
                "land_mask should be a hard link to the same on-disk object as "
                "mask, not a separately stored duplicate.",
            )

    def test_write_gauge_info_csv(self):
        out_dir = self.tmp_path / "gauge_csv"
        out_dir.mkdir(parents=True, exist_ok=True)
        rows = [
            {
                "id": 101,
                "lon": 10.0,
                "lat": 50.0,
                "lon_old": 10.1,
                "lat_old": 50.1,
                "distance": 1.25,
                "area": 1000.0,
                "old_area": 980.0,
                "area_error": 0.02,
                "score": 0.45,
                "shape_error": 0.25,
                "used_method": "area-basinex",
            }
        ]
        gauge = catchment.Gauge(gauge_id=101, lat=50.0, lon=10.0, area=1000.0)
        gauge.update(
            lon=10.1,
            lat=50.1,
            distance_error=1.25,
            area=980.0,
            area_error=0.02,
            score=0.12,
            shape_error=0.34,
            method="shape-area",
        )
        catchment.write_gauges_to_csv([gauge], out_dir, "gauges_info.csv")

        out_file = out_dir / "gauges_info.csv"
        self.assertTrue(out_file.exists())
        with out_file.open("r", encoding="utf-8", newline="") as csv_file:
            reader = csv.DictReader(csv_file)
            written_rows = list(reader)
            self.assertEqual(reader.fieldnames, list(catchment.GAUGE_INFO_COLUMNS))
        self.assertEqual(len(written_rows), 1)
        self.assertEqual(written_rows[0]["id"], "101")
        self.assertAlmostEqual(float(written_rows[0]["distance"]), 1.25)
        self.assertAlmostEqual(float(written_rows[0]["score"]), 0.12)
        self.assertAlmostEqual(float(written_rows[0]["shape_error"]), 0.34)
        self.assertEqual(written_rows[0]["method"], "shape-area")

        catchment.write_gauges_to_csv(rows, out_dir, "gauges_info_dict.csv")
        dict_out_file = out_dir / "gauges_info_dict.csv"
        with dict_out_file.open("r", encoding="utf-8", newline="") as csv_file:
            dict_rows = list(csv.DictReader(csv_file))
        self.assertEqual(len(dict_rows), 1)
        self.assertAlmostEqual(float(dict_rows[0]["score"]), 0.45)
        self.assertAlmostEqual(float(dict_rows[0]["shape_error"]), 0.25)
        self.assertEqual(dict_rows[0]["method"], "area-basinex")

        catchment.write_gauges_to_nc([gauge], out_dir, "gauges_info.nc")
        nc_file = out_dir / "gauges_info.nc"
        self.assertTrue(nc_file.exists())
        with xr.open_dataset(nc_file) as nc_ds:
            if __version__ != "not_available":
                self.assertEqual(nc_ds.attrs[VERSION_ATTR], __version__)
            self.assertIn(CREATED_ATTR, nc_ds.attrs)
            self.assertIn("mhm-tools command:", nc_ds.attrs[HISTORY_ATTR])

    def test_write_gauge_info_csv_no_rows(self):
        out_dir = self.tmp_path / "gauge_csv_empty"
        out_dir.mkdir(parents=True, exist_ok=True)
        catchment.write_gauges_to_csv([], out_dir, "gauges_info.csv")
        self.assertFalse((out_dir / "gauges_info.csv").exists())

    def test_write_basin_shape_outputs_files(self):
        self._require_geospatial()
        c = self._make_small_catchment()
        c.catchment_mask = np.zeros((5, 5), dtype=bool)
        c.catchment_mask[2:4, 2:4] = True
        shapes_dir = self.tmp_path / "shapes"
        c.write_basin_shape(shapes_dir, gauge_id=123)
        self.assertTrue((shapes_dir / "basin_123.shp").exists())

    def test_merge_catchment(self):
        self.test_write()  # Ensure files are written first

        path1 = self.output_path / "hydro1.nc"
        path2 = self.output_path / "hydro2.nc"
        out_path = self.output_path / "hydro_merged_03min.nc"

        self.assertTrue(path1.is_file(), "hydro1.nc does not exist.")
        with xr.open_dataset(path2, engine="netcdf4") as ds1:
            lat_key = get_coord_key(ds1, lat=True, raise_exception=False)
            lon_key = get_coord_key(ds1, lon=True, raise_exception=False)
        self.assertTrue(path2.is_file(), "hydro2.nc does not exist.")

        catchment.merge_catchment(path1, path2, out_path)
        self.assertTrue(out_path.is_file())

    def test_resolution_l2_file_resolution_matches(self):
        lon = np.array([0.0, 0.5, 1.0])
        lat = np.array([1.0, 0.5, 0.0])
        ds = xr.Dataset(
            {"dummy": (["lat", "lon"], np.zeros((len(lat), len(lon))))},
            coords={"lon": lon, "lat": lat},
        )
        l2_path = self.tmp_path / "l2_res_match.nc"
        ds.to_netcdf(l2_path)

        res = mhm_tools.common.utils.Resolution(l2=0.5, l2_file=l2_path)
        self.assertAlmostEqual(res.l2, 0.5, places=9)

    def test_resolution_l2_file_resolution_within_tolerance(self):
        lon = np.array([0.0, 0.5, 1.0])
        lat = np.array([1.0, 0.5, 0.0])
        ds = xr.Dataset(
            {"dummy": (["lat", "lon"], np.zeros((len(lat), len(lon))))},
            coords={"lon": lon, "lat": lat},
        )
        l2_path = self.tmp_path / "l2_res_within_tol.nc"
        ds.to_netcdf(l2_path)

        res = mhm_tools.common.utils.Resolution(l2=0.5000005, l2_file=l2_path)
        self.assertAlmostEqual(res.l2, 0.5, places=6)

    def test_resolution_l2_file_resolution_mismatch_raises(self):
        lon = np.array([0.0, 0.5, 1.0])
        lat = np.array([1.0, 0.5, 0.0])
        ds = xr.Dataset(
            {"dummy": (["lat", "lon"], np.zeros((len(lat), len(lon))))},
            coords={"lon": lon, "lat": lat},
        )
        l2_path = self.tmp_path / "l2_res_mismatch.nc"
        ds.to_netcdf(l2_path)

        with self.assertRaises(ValueError):
            mhm_tools.common.utils.Resolution(l2=0.500002, l2_file=l2_path)

    def test_find_best_gauge_location_best_candidate(self):
        c = self._make_small_catchment()
        upstream_area = np.zeros((5, 5), dtype=float)
        upstream_area[2, 1] = 100.0
        upstream_area[2, 3] = 105.0
        best_coord_basinex, error_basinex, distance_basinex = (
            find_best_gauge_location_by_area(
                ds=c.ds,
                upstream_area=upstream_area,
                gauge_coords=(2.0, 2.0),
                ref_catchment_area=103.0,
                resolutions=c.resolutions,
                max_distance_cells=4,
                max_error=0.05,
                method="basinex",
                raise_on_fallback=True,
            )
        )
        best_coord_burek, error_burek, distance_burek = (
            find_best_gauge_location_by_area(
                ds=c.ds,
                upstream_area=upstream_area,
                gauge_coords=(2.0, 2.0),
                ref_catchment_area=103.0,
                resolutions=c.resolutions,
                max_distance_cells=4,
                max_error=0.05,
                method="burek",
                raise_on_fallback=True,
            )
        )
        self.assertEqual(best_coord_basinex, (2, 3))
        self.assertLessEqual(error_basinex, 0.05)
        self.assertEqual(best_coord_burek, (2, 3))
        self.assertLessEqual(error_burek, 0.05)

        c = self._make_small_catchment()
        upstream_area = np.zeros((5, 5), dtype=float)
        upstream_area[2, 1] = 100.0
        upstream_area[0, 2] = 98.2
        upstream_area[3, 2] = 102.0
        best_coord_basinex, error_basinex, distance_basinex = (
            find_best_gauge_location_by_area(
                ds=c.ds,
                upstream_area=upstream_area,
                gauge_coords=(2.0, 2.0),
                ref_catchment_area=100.0,
                resolutions=c.resolutions,
                max_distance_cells=4,
                max_error=0.05,
                method="basinex",
                raise_on_fallback=True,
            )
        )
        best_coord_burek, error_burek, distance_burek = (
            find_best_gauge_location_by_area(
                ds=c.ds,
                upstream_area=upstream_area,
                gauge_coords=(2.0, 2.0),
                ref_catchment_area=100.0,
                resolutions=c.resolutions,
                max_distance_cells=4,
                max_error=0.05,
                method="burek",
                raise_on_fallback=True,
            )
        )
        self.assertEqual(best_coord_basinex, best_coord_burek)
        self.assertAlmostEqual(error_basinex, error_burek)
        self.assertEqual(best_coord_basinex, (2, 1))
        self.assertAlmostEqual(error_basinex, 0.0)
        self.assertEqual(best_coord_burek, (2, 1))
        self.assertAlmostEqual(error_burek, 0.0)

    def test_area_delimiter_can_be_disabled(self):
        """Select the best candidate inside the radius despite its area error."""
        c = self._make_small_catchment()
        upstream_area = np.full((5, 5), np.nan)
        upstream_area[2, 2] = 50.0
        upstream_area[2, 3] = 80.0

        with self.assertRaises(ValueError):
            find_best_gauge_location_by_area(
                ds=c.ds,
                upstream_area=upstream_area,
                gauge_coords=(2.0, 2.0),
                ref_catchment_area=100.0,
                resolutions=c.resolutions,
                max_distance_m=1.5,
                max_error=0.1,
                raise_on_fallback=True,
            )

        best_coord, error, _ = find_best_gauge_location_by_area(
            ds=c.ds,
            upstream_area=upstream_area,
            gauge_coords=(2.0, 2.0),
            ref_catchment_area=100.0,
            resolutions=c.resolutions,
            max_distance_m=1.5,
            max_error=0.1,
            use_max_error=False,
            raise_on_fallback=True,
        )

        self.assertEqual(best_coord, (2, 3))
        self.assertAlmostEqual(error, 0.2)

    def test_max_distance_m_uses_strict_radial_mask(self):
        """Exclude square-window corners beyond the meter radius."""
        c = self._make_small_catchment()
        upstream_area = np.full((5, 5), np.nan)
        upstream_area[1, 1] = 100.0
        upstream_area[1, 2] = 90.0

        best_coord, _, _ = find_best_gauge_location_by_area(
            ds=c.ds,
            upstream_area=upstream_area,
            gauge_coords=(2.0, 2.0),
            ref_catchment_area=100.0,
            resolutions=c.resolutions,
            max_distance_m=1.1,
            use_max_error=False,
            raise_on_fallback=True,
        )

        self.assertEqual(best_coord, (1, 2))

    def test_latlon_max_distance_m_is_coordinate_aware(self):
        """Apply a radial meter limit to a latitude/longitude grid."""
        coordinates = np.array([-0.01, 0.0, 0.01])
        _, _, _, _, distance_mask, distances_m = get_candidate_search_window(
            lat_values=coordinates,
            lon_values=coordinates,
            gauge_row=1,
            gauge_col=1,
            max_distance_m=1200,
            latlon=True,
        )

        self.assertTrue(distance_mask[0, 1])
        self.assertFalse(distance_mask[0, 0])
        self.assertGreater(distances_m[0, 0], 1200)

    def test_candidate_distance_arguments_are_validated(self):
        """Reject conflicting and negative candidate distance limits."""
        coordinates = np.arange(3, dtype=float)
        with self.assertRaises(ValueError):
            get_candidate_search_window(
                coordinates,
                coordinates,
                1,
                1,
                max_distance_cells=1,
                max_distance_m=1,
            )
        with self.assertRaises(ValueError):
            get_candidate_search_window(
                coordinates, coordinates, 1, 1, max_distance_m=-1
            )

    def test_zero_meter_distance_selects_only_gauge_cell(self):
        """Restrict a zero-meter candidate search to the gauge cell."""
        coordinates = np.arange(3, dtype=float)

        _, _, _, _, distance_mask, distances_m = get_candidate_search_window(
            coordinates, coordinates, 1, 1, max_distance_m=0
        )

        self.assertEqual(distance_mask.shape, (1, 1))
        self.assertTrue(distance_mask[0, 0])
        self.assertEqual(distances_m[0, 0], 0)

    def test_distance_100m_units_3_arcsec(self):
        res = 1.0 / 1200.0  # 3 arc sec in degrees
        lon = np.array([0.0, res, 2 * res], dtype=float)
        lat = np.array([res, 0.0, -res], dtype=float)
        ds = xr.Dataset(
            {"dem": (["lat", "lon"], np.zeros((len(lat), len(lon))))},
            coords={"lon": lon, "lat": lat},
        )
        c = catchment.Catchment(
            ds,
            "dem",
            var="dem",
            ftype="ldd",
            transform=(res, 0.0, 0.0, 0.0, res, 0.0),
            latlon=True,
        )
        d = distance_100m_units(1, 0, l0_resolution=c.resolutions.l0, lat_deg=0.0)
        self.assertGreater(d, 0.90)
        self.assertLess(d, 0.95)

    def test_distance_100m_units_scales_with_resolution(self):
        res_small = 1.0 / 1200.0  # 3 arc sec
        res_large = 1.0 / 600.0  # 6 arc sec
        lon_small = np.array([0.0, res_small, 2 * res_small], dtype=float)
        lat_small = np.array([res_small, 0.0, -res_small], dtype=float)
        lon_large = np.array([0.0, res_large, 2 * res_large], dtype=float)
        lat_large = np.array([res_large, 0.0, -res_large], dtype=float)

        ds_small = xr.Dataset(
            {"dem": (["lat", "lon"], np.zeros((len(lat_small), len(lon_small))))},
            coords={"lon": lon_small, "lat": lat_small},
        )
        ds_large = xr.Dataset(
            {"dem": (["lat", "lon"], np.zeros((len(lat_large), len(lon_large))))},
            coords={"lon": lon_large, "lat": lat_large},
        )

        c_small = catchment.Catchment(
            ds_small,
            "dem",
            var="dem",
            ftype="ldd",
            transform=(res_small, 0.0, 0.0, 0.0, res_small, 0.0),
            latlon=True,
        )
        c_large = catchment.Catchment(
            ds_large,
            "dem",
            var="dem",
            ftype="ldd",
            transform=(res_large, 0.0, 0.0, 0.0, res_large, 0.0),
            latlon=True,
        )

        d_small = distance_100m_units(
            1, 0, l0_resolution=c_small.resolutions.l0, lat_deg=0.0
        )
        d_large = distance_100m_units(
            1, 0, l0_resolution=c_large.resolutions.l0, lat_deg=0.0
        )
        self.assertAlmostEqual(d_large / d_small, 2.0, places=2)

    def test_cut_to_filled_area_l2_alignment_mismatch_raises(self):
        lon = np.arange(0.5, 64.5, 1.0)
        lat = np.arange(63.5, -0.5, -1.0)
        data = np.zeros((len(lat), len(lon)), dtype=float)
        ds = xr.Dataset(
            {"dem": (["lat", "lon"], data)},
            coords={
                "lon": lon,
                "lat": lat,
            },
        )

        l2_lon = np.arange(2.0, 62.0 + 0.1, 4.0)
        l2_lat = np.arange(62.0, 2.0 - 0.1, -4.0)
        l2_ds = xr.Dataset(
            {"dummy": (["lat", "lon"], np.zeros((len(l2_lat), len(l2_lon))))},
            coords={
                "lon": l2_lon,
                "lat": l2_lat,
            },
        )
        l2_path = self.tmp_path / "l2_alignment.nc"
        l2_ds.to_netcdf(l2_path)

        resolutions = mhm_tools.common.utils.Resolution(
            l1=32,
            l11=32,
            l2=32,
        )
        transform = catchment.get_transformation_matrix_nc(ds, "dem")
        c = catchment.Catchment(
            ds,
            "dem",
            var="dem",
            ftype="ldd",
            transform=transform,
            latlon=True,
            resolutions=resolutions,
        )
        c.resolutions.l2_file = (
            l2_path  # manually set l2_file to not trigger alignment check
        )
        mask = np.zeros((len(lat), len(lon)), dtype=bool)
        mask[10:21, 10:21] = True
        c.catchment_mask = mask

        with self.assertRaises(AssertionError) as ctx:
            c.cut_to_filled_area(raise_on_l2_alignment_mismatch=True)
        self.assertIn("not divisible by factor=32", str(ctx.exception))

    def test_cut_to_filled_area_l2_alignment_matches_factor(self):
        lon = np.arange(0.5, 64.5, 1.0)
        lat = np.arange(63.5, -0.5, -1.0)
        data = np.zeros((len(lat), len(lon)), dtype=float)
        ds = xr.Dataset(
            {"dem": (["lat", "lon"], data)},
            coords={
                "lon": lon,
                "lat": lat,
            },
        )

        l2_lon = np.array([16.0, 48.0])
        l2_lat = np.array([48.0, 16.0])
        l2_ds = xr.Dataset(
            {"dummy": (["lat", "lon"], np.zeros((len(l2_lat), len(l2_lon))))},
            coords={
                "lon": l2_lon,
                "lat": l2_lat,
            },
        )
        l2_path = self.tmp_path / "l2_alignment_ok.nc"
        l2_ds.to_netcdf(l2_path)

        resolutions = mhm_tools.common.utils.Resolution(
            l1=32,
            l11=32,
            l2=32,
            l2_file=l2_path,
        )
        transform = catchment.get_transformation_matrix_nc(ds, "dem")
        c = catchment.Catchment(
            ds,
            "dem",
            var="dem",
            ftype="ldd",
            transform=transform,
            latlon=True,
            resolutions=resolutions,
        )
        mask = np.zeros((len(lat), len(lon)), dtype=bool)
        mask[10:21, 10:21] = True
        c.catchment_mask = mask

        lat_slice_idx, lon_slice_idx = c.cut_to_filled_area()
        n_lat = lat_slice_idx.stop - lat_slice_idx.start
        n_lon = lon_slice_idx.stop - lon_slice_idx.start
        self.assertEqual(n_lat % 32, 0)
        self.assertEqual(n_lon % 32, 0)

    # ------------------------------------------------------------------
    # Optional integration tests for delineation using a real flow-direction
    # file and gauge coordinates. These tests are skipped unless you provide
    # the path to your fdir file and the gauge coordinates below.
    # To run, set the variables FDIR_PATH, GAUGE_LAT, GAUGE_LON (and
    # optionally REF_AREA) to appropriate values.
    # ------------------------------------------------------------------

    # Replace these placeholders with your real test inputs before running.
    FDIR_PATH = Path(HERE, "files", "test_create_catchment", "fdir.nc")
    FDIR_VAR = "fdir"  # change to the variable name in your file if different
    GAUGE_LAT = [49.292013, 48.445978]
    GAUGE_LON = [8.679113, 8.70628]
    REF_AREA = [113.33, 1123.61]
    GAUGE_IDS = [101, 102]

    def test_delineate_basin_without_ref(self):
        """Integration-style test: delineate a basin using a Gauge object without a reference area."""
        if (
            not self.FDIR_PATH.exists()
            or self.GAUGE_LAT is None
            or self.GAUGE_LON is None
        ):
            self.skipTest(
                "Set FDIR_PATH, GAUGE_LAT and GAUGE_LON in this test file to run this integration test."
            )

        ds = get_xarray_ds_from_file(str(self.FDIR_PATH))
        var_name = (
            self.FDIR_VAR if self.FDIR_VAR in ds.data_vars else list(ds.data_vars)[0]
        )

        # get coordinate arrays and convert lat/lon to nearest indices
        lat_key = get_coord_key(ds, lat=True, raise_exception=False)
        lon_key = get_coord_key(ds, lon=True, raise_exception=False)

        # only test on basin 2 because basin 1 can not be resolved without ref area
        c = catchment.Catchment(
            ds,
            var_name,
            var="fdir",
            ftype="d8",
            transform=self.transform,
            latlon=True,
        )
        gauge = catchment.Gauge(
            gauge_id=202,
            lat=float(self.GAUGE_LAT[1]),
            lon=float(self.GAUGE_LON[1]),
            area=None,
        )
        c.delineate_basin(gauge, raise_on_sanity_check=False)

        self.assertIsNotNone(c.basin)
        self.assertTrue(
            np.any(c.catchment_mask),
            "No catchment cells found for provided gauge coordinates",
        )

        # compute area of resulting catchment using create_cell_area
        c.compute_cell_area()
        cell_area = c.cell_area
        area_km2 = float(np.sum(cell_area[c.catchment_mask]))
        self.assertGreater(area_km2, 0.0)
        rel_diff = abs(area_km2 - float(self.REF_AREA[1])) / float(self.REF_AREA[1])
        self.assertLessEqual(
            rel_diff,
            0.05,
            f"Delineated area {area_km2} differs more than 5% from REF_AREA {self.REF_AREA[1]}",
        )
        print(
            f"No ref Delineated area: {area_km2} km², Reference area: {self.REF_AREA[1]} km², Relative difference: {rel_diff*100:.2f}%"
        )

    def test_delineate_basin_with_ref(self):
        """Integration-style test: delineate a basin with Gauge-provided reference area."""
        if (
            not self.FDIR_PATH.exists()
            or self.GAUGE_LAT is None
            or self.GAUGE_LON is None
            or self.REF_AREA is None
        ):
            self.skipTest(
                "Set FDIR_PATH, GAUGE_LAT, GAUGE_LON and REF_AREA in this test file to run this integration test."
            )

        ds = get_xarray_ds_from_file(str(self.FDIR_PATH))
        var_name = (
            self.FDIR_VAR if self.FDIR_VAR in ds.data_vars else list(ds.data_vars)[0]
        )

        lat_key = get_coord_key(ds, lat=True, raise_exception=False)
        lon_key = get_coord_key(ds, lon=True, raise_exception=False)

        for lat, lon, ref_area in zip(self.GAUGE_LAT, self.GAUGE_LON, self.REF_AREA):
            c = catchment.Catchment(
                ds,
                var_name,
                var="fdir",
                ftype="d8",
                transform=self.transform,
                latlon=True,
            )
            gauge = catchment.Gauge(
                gauge_id=101,
                lat=float(lat),
                lon=float(lon),
                area=float(ref_area),
            )
            c.delineate_basin(
                gauge=gauge,
                max_distance_cells=10,
                max_error=0.05,
                raise_on_sanity_check=True,
            )

            self.assertIsNotNone(c.basin)
            self.assertTrue(
                np.any(c.catchment_mask),
                "No catchment cells found for provided gauge coordinates and ref area",
            )

            c.compute_cell_area()
            cell_area = c.cell_area
            area_km2 = float(np.sum(cell_area[c.catchment_mask]))

            # check that computed area is reasonably close to the reference (5% tolerance)
            rel_diff = abs(area_km2 - float(ref_area)) / float(ref_area)
            self.assertLessEqual(
                rel_diff,
                0.05,
                f"Delineated area {area_km2} differs more than 5% from REF_AREA {ref_area}",
            )
            print(
                f"With ref Delineated area: {area_km2} km², Reference area: {ref_area} km², Relative difference: {rel_diff*100:.2f}%"
            )

    def test_multiple_gauges_idgauges_consistency(self):
        """Compare idgauges from individual gauges vs. combined multi-gauge output."""
        if (
            not self.FDIR_PATH.exists()
            or self.GAUGE_LAT is None
            or self.GAUGE_LON is None
            or self.REF_AREA is None
        ):
            self.skipTest(
                "Set FDIR_PATH, GAUGE_LAT, GAUGE_LON and REF_AREA in this test file to run this integration test."
            )

        gauge_coords = [
            (float(self.GAUGE_LAT[0]), float(self.GAUGE_LON[0])),
            (float(self.GAUGE_LAT[1]), float(self.GAUGE_LON[1])),
        ]
        gauge_areas = [float(self.REF_AREA[0]), float(self.REF_AREA[1])]

        with get_xarray_ds_from_file(str(self.FDIR_PATH)) as ds:
            var_name = (
                self.FDIR_VAR
                if self.FDIR_VAR in ds.data_vars
                else list(ds.data_vars)[0]
            )

        snapped = []
        resolutions = mhm_tools.common.utils.Resolution(l0=0.001953125, l1=1 / 32)
        for idx, (lat, lon) in enumerate(gauge_coords):
            out_dir = self.tmp_path / f"single_{idx}"
            out_dir.mkdir(parents=True, exist_ok=True)
            with get_xarray_ds_from_file(
                str(self.FDIR_PATH),
                var_name=var_name,
                normalize_latlon_coords=True,
                force_decending_y=True,
            ) as ds:
                transform = catchment.get_transformation_matrix_nc(ds, var_name)
                catchment.create_catchment(
                    input_file=str(self.FDIR_PATH),
                    output_path=str(out_dir),
                    var_name=var_name,
                    var="fdir",
                    ftype="d8",
                    gauge_coords=(lat, lon),
                    gauge_ids=[self.GAUGE_IDS[idx]],
                    ref_catchment_area=gauge_areas[idx],
                    max_distance_cells=10,
                    max_error=0.25,
                    latlon=True,
                    frame=1,
                    resolutions=resolutions,
                    raise_on_fallback=True,
                )
            id_path = out_dir / "idgauges.nc"
            self.assertTrue(id_path.is_file())
            with xr.open_dataset(id_path) as id_ds:
                id_da = id_ds["idgauges"]
                matches = np.argwhere(id_da.values == self.GAUGE_IDS[idx])
                self.assertEqual(matches.shape[0], 1)
                row, col = matches[0]
                lat_val = float(id_da["lat"].values[row])
                lon_val = float(id_da["lon"].values[col])
            snapped.append((lat_val, lon_val, self.GAUGE_IDS[idx]))

        combined_dir = self.tmp_path / "combined"
        combined_dir.mkdir(parents=True, exist_ok=True)
        catchment.create_catchment(
            input_file=str(self.FDIR_PATH),
            output_path=str(combined_dir),
            var_name=var_name,
            var="fdir",
            ftype="d8",
            gauge_coords=gauge_coords,
            gauge_ids=self.GAUGE_IDS,
            ref_catchment_area=gauge_areas,
            max_distance_cells=10,
            max_error=0.25,
            latlon=True,
            frame=1,
            resolutions=resolutions,
            raise_on_fallback=True,
        )
        combined_path = combined_dir / "idgauges.nc"
        self.assertTrue(combined_path.is_file())
        with xr.open_dataset(combined_path) as combined_ds:
            combined_da = combined_ds["idgauges"]
            for lat_val, lon_val, gid in snapped:
                combined_val = combined_da.sel(
                    lat=lat_val, lon=lon_val, method="nearest"
                ).item()
                self.assertEqual(int(combined_val), gid)

    def test_parallel_matches_sequential_multi_gauge(self):
        """Parallel output should match sequential output for multiple gauges."""
        if (
            not self.FDIR_PATH.exists()
            or self.GAUGE_LAT is None
            or self.GAUGE_LON is None
        ):
            self.skipTest(
                "Set FDIR_PATH, GAUGE_LAT and GAUGE_LON in this test file to run this integration test."
            )

        gauge_coords = [
            (float(self.GAUGE_LAT[0]), float(self.GAUGE_LON[0])),
            (float(self.GAUGE_LAT[1]), float(self.GAUGE_LON[1])),
        ]

        with get_xarray_ds_from_file(str(self.FDIR_PATH)) as ds:
            var_name = (
                self.FDIR_VAR
                if self.FDIR_VAR in ds.data_vars
                else list(ds.data_vars)[0]
            )

        resolutions = mhm_tools.common.utils.Resolution(l0=0.001953125, l1=1 / 32)

        seq_dir = self.tmp_path / "seq"
        par_dir = self.tmp_path / "par"
        seq_dir.mkdir(parents=True, exist_ok=True)
        par_dir.mkdir(parents=True, exist_ok=True)

        catchment.create_catchment(
            input_file=str(self.FDIR_PATH),
            output_path=str(seq_dir),
            var_name=var_name,
            var="fdir",
            ftype="d8",
            gauge_coords=gauge_coords,
            gauge_ids=self.GAUGE_IDS,
            latlon=True,
            frame=1,
            resolutions=resolutions,
            ncpus=1,
            raise_on_fallback=True,
        )
        catchment.create_catchment(
            input_file=str(self.FDIR_PATH),
            output_path=str(par_dir),
            var_name=var_name,
            var="fdir",
            ftype="d8",
            gauge_coords=gauge_coords,
            gauge_ids=self.GAUGE_IDS,
            latlon=True,
            frame=1,
            resolutions=resolutions,
            ncpus=2,
            raise_on_fallback=True,
        )

        seq_path = seq_dir / "idgauges.nc"
        par_path = par_dir / "idgauges.nc"
        self.assertTrue(seq_path.is_file())
        self.assertTrue(par_path.is_file())

        with xr.open_dataset(seq_path) as seq_ds, xr.open_dataset(par_path) as par_ds:
            np.testing.assert_array_equal(
                seq_ds["idgauges"].values, par_ds["idgauges"].values
            )

    def test_shape_based_create_catchment_matches_reference_basin(self):
        """Shape-only delineation should reproduce the basin used to create the shape."""
        self._require_geospatial()
        if (
            not self.FDIR_PATH.exists()
            or self.GAUGE_LAT is None
            or self.GAUGE_LON is None
            or self.REF_AREA is None
        ):
            self.skipTest(
                "Set FDIR_PATH, GAUGE_LAT, GAUGE_LON and REF_AREA in this test file to run this integration test."
            )

        with get_xarray_ds_from_file(str(self.FDIR_PATH)) as ds:
            var_name = (
                self.FDIR_VAR
                if self.FDIR_VAR in ds.data_vars
                else list(ds.data_vars)[0]
            )

        gauge_id = self.GAUGE_IDS[0]
        gauge_coords = (float(self.GAUGE_LAT[0]), float(self.GAUGE_LON[0]))
        ref_area = float(self.REF_AREA[0])
        common_kwargs = {
            "input_file": str(self.FDIR_PATH),
            "var_name": var_name,
            "var": "fdir",
            "ftype": "d8",
            "gauge_ids": gauge_id,
            "max_distance_cells": 10,
            "max_error": 0.25,
            "latlon": True,
            "frame": 0,
            "output_vars": ["basin"],
            "raise_on_fallback": True,
        }

        reference_dir = self.tmp_path / "shape_reference"
        shape_with_coords_dir = self.tmp_path / "shape_with_coords"
        shape_only_dir = self.tmp_path / "shape_only"

        catchment.create_catchment(
            output_path=reference_dir,
            gauge_coords=gauge_coords,
            ref_catchment_area=ref_area,
            **common_kwargs,
        )
        shape_folder = reference_dir / "shapes"
        self.assertTrue(
            (shape_folder / f"basin_{gauge_id}.shp").is_file(),
            "Reference catchment shapefile was not written.",
        )

        catchment.create_catchment(
            output_path=shape_with_coords_dir,
            gauge_coords=gauge_coords,
            shape_folder=shape_folder,
            **common_kwargs,
        )
        catchment.create_catchment(
            output_path=shape_only_dir,
            gauge_coords=None,
            shape_folder=shape_folder,
            **common_kwargs,
        )

        reference_mask = self._read_basin_mask(reference_dir / "basin_ids.nc")
        shape_with_coords_mask = self._read_basin_mask(
            shape_with_coords_dir / "basin_ids.nc"
        )
        shape_only_mask = self._read_basin_mask(shape_only_dir / "basin_ids.nc")

        self._assert_same_basin_mask(reference_mask, shape_with_coords_mask)
        self._assert_same_basin_mask(reference_mask, shape_only_mask)
        self._assert_same_gauge_cell(
            reference_dir / "idgauges.nc",
            shape_with_coords_dir / "idgauges.nc",
            gauge_id,
        )
        self._assert_same_gauge_cell(
            reference_dir / "idgauges.nc",
            shape_only_dir / "idgauges.nc",
            gauge_id,
        )

    def _read_basin_mask(self, path):
        self.assertTrue(path.is_file(), f"Missing basin output: {path}")
        with xr.open_dataset(path) as ds:
            return (ds["basin"] > 0).load()

    def _assert_same_basin_mask(self, expected, actual):
        expected_aligned, actual_aligned = xr.align(
            expected, actual, join="outer", fill_value=False
        )
        np.testing.assert_array_equal(
            expected_aligned.values,
            actual_aligned.values,
            "Delineated basin mask does not match the reference shape basin.",
        )

    def _assert_same_gauge_cell(self, expected_path, actual_path, gauge_id):
        self.assertTrue(
            expected_path.is_file(), f"Missing gauge output: {expected_path}"
        )
        self.assertTrue(actual_path.is_file(), f"Missing gauge output: {actual_path}")
        with xr.open_dataset(expected_path) as expected_ds, xr.open_dataset(
            actual_path
        ) as actual_ds:
            expected_cells = np.argwhere(expected_ds["idgauges"].values == gauge_id)
            actual_cells = np.argwhere(actual_ds["idgauges"].values == gauge_id)
            self.assertEqual(expected_cells.shape[0], 1)
            self.assertEqual(actual_cells.shape[0], 1)
            expected_row, expected_col = expected_cells[0]
            actual_row, actual_col = actual_cells[0]
            expected_lat = float(expected_ds["idgauges"]["lat"].values[expected_row])
            expected_lon = float(expected_ds["idgauges"]["lon"].values[expected_col])
            actual_lat = float(actual_ds["idgauges"]["lat"].values[actual_row])
            actual_lon = float(actual_ds["idgauges"]["lon"].values[actual_col])
            self.assertEqual((expected_lat, expected_lon), (actual_lat, actual_lon))

    def test_shape_comparison_l0(self):
        self._require_geospatial()
        if (
            not self.FDIR_PATH.exists()
            or self.GAUGE_LAT is None
            or self.GAUGE_LON is None
            or self.REF_AREA is None
        ):
            self.skipTest(
                "Set FDIR_PATH, GAUGE_LAT, GAUGE_LON and REF_AREA in this test file to run this integration test."
            )

        with get_xarray_ds_from_file(str(self.FDIR_PATH)) as ds:
            var_name = (
                self.FDIR_VAR
                if self.FDIR_VAR in ds.data_vars
                else list(ds.data_vars)[0]
            )
            transform = catchment.get_transformation_matrix_nc(ds, var_name)
            c = catchment.Catchment(
                ds,
                var_name,
                var="fdir",
                ftype="d8",
                transform=transform,
                latlon=True,
            )
            gauge = catchment.Gauge(
                gauge_id=101,
                lat=float(self.GAUGE_LAT[0]),
                lon=float(self.GAUGE_LON[0]),
                area=float(self.REF_AREA[0]),
            )
            c.delineate_basin(
                gauge, raise_on_sanity_check=True, max_distance_cells=10, max_error=0.25
            )
            self.assertIsNotNone(c.catchment_mask)
            l0_shape = catchment._vectorize_mask_to_gdf(
                c.catchment_mask, c.transform, catchment._shape_crs(c.latlon)
            )
            upstream_area = c.calc_upstream_area()
            result = c.find_best_gauge_location_shape(
                upstream_area=upstream_area,
                gauge_coords=(self.GAUGE_LAT[0], self.GAUGE_LON[0]),
                ref_catchment_area=self.REF_AREA[0],
                shape_folder=None,
                gauge_id=None,
                reference_shape_gdf=l0_shape,
                max_distance_cells=10,
            )
            self.assertIsNotNone(result)

            (
                candidate_idx,
                _area_error,
                _distance_error,
                score,
                shape_error,
                method,
            ) = result
            self.assertTrue(np.isfinite(score))
            self.assertTrue(np.isfinite(shape_error))
            self.assertGreaterEqual(shape_error, 0.0)
            self.assertLessEqual(shape_error, 1.0)
            self.assertEqual(method, "shape_iou")
            linear = np.ravel_multi_index(candidate_idx, c._fdir.shape)
            basin = c._fdir.basins(idxs=np.array([linear], dtype=np.int64))
            candidate_mask = basin > 0
            candidate_gdf = catchment._vectorize_mask_to_gdf(
                candidate_mask, c.transform, catchment._shape_crs(c.latlon)
            )
            iou = catchment._shape_iou(l0_shape, candidate_gdf)
            self.assertGreaterEqual(iou, 0.7)

    def test_shape_area_matches_delineated_area(self):
        self._require_geospatial()
        if (
            not self.FDIR_PATH.exists()
            or self.GAUGE_LAT is None
            or self.GAUGE_LON is None
            or self.REF_AREA is None
        ):
            self.skipTest(
                "Set FDIR_PATH, GAUGE_LAT, GAUGE_LON and REF_AREA in this test file to run this integration test."
            )

        import geopandas as gpd

        with get_xarray_ds_from_file(str(self.FDIR_PATH)) as ds:
            var_name = (
                self.FDIR_VAR
                if self.FDIR_VAR in ds.data_vars
                else list(ds.data_vars)[0]
            )
            transform = catchment.get_transformation_matrix_nc(ds, var_name)
            c = catchment.Catchment(
                ds,
                var_name,
                var="fdir",
                ftype="d8",
                transform=transform,
                latlon=True,
            )
            gauge = catchment.Gauge(
                gauge_id=101,
                lat=float(self.GAUGE_LAT[0]),
                lon=float(self.GAUGE_LON[0]),
                area=float(self.REF_AREA[0]),
            )
            c.delineate_basin(
                gauge,
                max_distance_cells=10,
                max_error=0.25,
                raise_on_sanity_check=True,
            )
            self.assertIsNotNone(c.catchment_mask)
            self.assertTrue(np.any(c.catchment_mask))

            c.compute_cell_area()
            delineated_area = float(np.sum(c.cell_area[c.catchment_mask]))

            shapes_dir = self.tmp_path / "shape_area"
            c.write_basin_shape(shapes_dir, gauge_id=gauge.gauge_id)
            basin_shape = gpd.read_file(shapes_dir / f"basin_{gauge.gauge_id}.shp")
            shape_area = c.calculate_shape_area_on_current_grid(
                basin_shape,
                shape_label="written basin shape",
            )

            self.assertIsNotNone(shape_area)
            self.assertGreater(shape_area, 0.0)
            self.assertAlmostEqual(shape_area, delineated_area, places=6)

    def test_shape_correction_after_upscale(self):
        self._require_geospatial()
        if (
            not self.FDIR_PATH.exists()
            or self.GAUGE_LAT is None
            or self.GAUGE_LON is None
            or self.REF_AREA is None
        ):
            self.skipTest(
                "Set FDIR_PATH, GAUGE_LAT, GAUGE_LON and REF_AREA in this test file to run this integration test."
            )

        with get_xarray_ds_from_file(str(self.FDIR_PATH)) as ds:
            var_name = (
                self.FDIR_VAR
                if self.FDIR_VAR in ds.data_vars
                else list(ds.data_vars)[0]
            )
            transform = catchment.get_transformation_matrix_nc(ds, var_name)
            lon_vals = ds.lon.data
            l0_res = abs(lon_vals[1] - lon_vals[0])
            resolutions = mhm_tools.common.utils.Resolution(l1=l0_res * 2)
            c = catchment.Catchment(
                ds,
                var_name,
                var="fdir",
                ftype="d8",
                transform=transform,
                latlon=True,
                resolutions=resolutions,
                upscale=True,
            )
            gauge = catchment.Gauge(
                gauge_id=101,
                lat=float(self.GAUGE_LAT[0]),
                lon=float(self.GAUGE_LON[0]),
                area=float(self.REF_AREA[0]),
            )
            c.delineate_basin(
                gauge, max_distance_cells=10, max_error=0.25, raise_on_sanity_check=True
            )
            l0_shape = catchment._vectorize_mask_to_gdf(
                c.catchment_mask, c.transform, catchment._shape_crs(c.latlon)
            )
            c.upscale("fdir")
            corrected_coords = c.correct_gauge_location_l1_from_shape(
                l0_shape,
                (self.GAUGE_LAT[0], self.GAUGE_LON[0]),
                max_distance_cells=10,
                max_error=0.3,
                reference_upstream_area=self.REF_AREA[0],
            )
            self.assertIsNotNone(corrected_coords)

            lon_l1, lat_l1 = c._coords_l1()
            gauge_row = int(np.abs(lat_l1 - float(self.GAUGE_LAT[0])).argmin())
            gauge_col = int(np.abs(lon_l1 - float(self.GAUGE_LON[0])).argmin())
            naive_linear = np.ravel_multi_index((gauge_row, gauge_col), c._fdir.shape)
            naive_basin = c._fdir.basins(idxs=np.array([naive_linear], dtype=np.int64))
            naive_gdf = catchment._vectorize_mask_to_gdf(
                naive_basin > 0, c._fdir.transform, catchment._shape_crs(c.latlon)
            )
            naive_iou = catchment._shape_iou(l0_shape, naive_gdf)

            corr_row = int(np.abs(lat_l1 - float(corrected_coords[0])).argmin())
            corr_col = int(np.abs(lon_l1 - float(corrected_coords[1])).argmin())
            corr_linear = np.ravel_multi_index((corr_row, corr_col), c._fdir.shape)
            corr_basin = c._fdir.basins(idxs=np.array([corr_linear], dtype=np.int64))
            corr_gdf = catchment._vectorize_mask_to_gdf(
                corr_basin > 0, c._fdir.transform, catchment._shape_crs(c.latlon)
            )
            corr_iou = catchment._shape_iou(l0_shape, corr_gdf)

            self.assertGreaterEqual(
                corr_iou,
                naive_iou,
                "L1 shape correction did not improve shape agreement.",
            )

    def test_merge_catchment_only_replaces_dateline_basins(self):
        lat = np.array([1.0, 0.0])
        lon = np.array([-179.0, -177.0, -100.0, 0.0, 100.0, 177.0, 179.0])
        basin1 = np.array(
            [
                [1, 1, 2, 2, 2, 3, 3],
                [1, 1, 2, 2, 2, 3, 3],
            ]
        )
        basin2 = basin1 + 10
        ds1 = xr.Dataset(
            {"basin": (["lat", "lon"], basin1)},
            coords={"lat": lat, "lon": lon},
        )
        ds2 = xr.Dataset(
            {"basin": (["lat", "lon"], basin2)},
            coords={"lat": lat, "lon": lon + 180},
        )
        path1 = self.tmp_path / "merge_hydro1.nc"
        path2 = self.tmp_path / "merge_hydro2.nc"
        out_path = self.tmp_path / "merge_out.nc"
        ds1.to_netcdf(path1)
        ds2.to_netcdf(path2)

        catchment.merge_catchment(path1, path2, out_path)

        with xr.open_dataset(out_path) as merged:
            self.assertEqual(int(merged["basin"].sel(lat=1.0, lon=0.0)), 2)
            self.assertGreater(int(merged["basin"].sel(lat=1.0, lon=-179.0)), 3)
            self.assertGreater(int(merged["basin"].sel(lat=1.0, lon=179.0)), 3)

    def _make_narrow_basin_catchment(self, nlat=120, nlon=60):
        """Build a catchment whose L2 grid is only one cell wide.

        The L0 window is a whole number of L2 cells tall but a single one
        wide, which is what a basin smaller than one coarse cell produces.
        """
        l0, l2 = 1 / 600, 0.1
        lat = 48.209166 - np.arange(nlat) * l0
        lon = 9.58 + np.arange(nlon) * l0
        basin = np.zeros((nlat, nlon), dtype=np.uint32)
        basin[10:110, 5:55] = 1
        dem_ds = xr.Dataset(
            {"dem": (["lat", "lon"], np.zeros((nlat, nlon)))},
            coords={"lat": lat, "lon": lon},
        )
        basin_ds = xr.Dataset(
            {"basin": (["lat", "lon"], basin)}, coords={"lat": lat, "lon": lon}
        )
        catchment_obj = catchment.Catchment(
            dem_ds,
            "dem",
            var="dem",
            ftype="ldd",
            transform=(l0, 0.0, lon[0], 0.0, -l0, lat[0]),
            resolutions=catchment.Resolution(l0=l0, l1=l2, l11=l2, l2=l2),
            latlon=True,
        )
        return catchment_obj, basin_ds, l0, l2

    def test_write_mask_file_generates_bounds_for_single_cell_l2_domain(self):
        """A basin narrower than one L2 cell still gets CF bounds.

        The upscaled grid then has a single longitude value, which has no
        neighbour to derive a cell width from, so the L2 resolution is used.
        """
        catchment_obj, basin_ds, l0, l2 = self._make_narrow_basin_catchment()
        mask_file = self.tmp_path / "mask_single_cell.nc"

        catchment_obj.write_mask_file(basin_ds, mask_file)

        with xr.open_dataset(mask_file) as mask_ds:
            self.assertEqual(mask_ds["mask_l2"].shape, (2, 1))
            self.assertIn("lon_l2_bnds", mask_ds.coords)
            lon_bnds = mask_ds["lon_l2_bnds"].values
            self.assertAlmostEqual(abs(lon_bnds[0, 1] - lon_bnds[0, 0]), l2)
            self.assertEqual(mask_ds["lon_l2"].attrs["bounds"], "lon_l2_bnds")
            # the fine coordinates keep the width derived from their own
            # spacing rather than the resolution handed in for the coarse one
            fine_bnds = mask_ds["lon_bnds"].values
            self.assertAlmostEqual(abs(fine_bnds[0, 1] - fine_bnds[0, 0]), l0)

    def test_write_mask_file_bounds_unchanged_for_multi_cell_domain(self):
        """Handing in a resolution does not override real coordinate spacing."""
        catchment_obj, basin_ds, _, l2 = self._make_narrow_basin_catchment(nlon=180)
        mask_file = self.tmp_path / "mask_multi_cell.nc"

        catchment_obj.write_mask_file(basin_ds, mask_file)

        with xr.open_dataset(mask_file) as mask_ds:
            self.assertEqual(mask_ds["mask_l2"].shape, (2, 3))
            lon_bnds = mask_ds["lon_l2_bnds"].values
            widths = np.abs(lon_bnds[:, 1] - lon_bnds[:, 0])
            np.testing.assert_allclose(widths, l2, rtol=1e-6)

    def _make_upscaled_catchment(self, l2, nlat, nlon, l1=0.1):
        """Build a catchment whose mask is already upscaled to L1.

        Mirrors the state `upscale` leaves behind, where the working grid is
        L1 rather than L0 and the mask still has to reach L2.
        """
        l0 = 1 / 600
        lat = 48.2 - np.arange(nlat) * l1
        lon = 9.6 + np.arange(nlon) * l1
        dem_ds = xr.Dataset(
            {"dem": (["lat", "lon"], np.zeros((nlat, nlon)))},
            coords={"lat": lat, "lon": lon},
        )
        basin_ds = xr.Dataset(
            {"basin": (["lat", "lon"], np.ones((nlat, nlon), dtype=np.uint32))},
            coords={"lat": lat, "lon": lon},
        )
        catchment_obj = catchment.Catchment(
            dem_ds,
            "dem",
            var="dem",
            ftype="ldd",
            transform=(l1, 0.0, lon[0], 0.0, -l1, lat[0]),
            resolutions=catchment.Resolution(l0=l0, l1=l1, l11=l1, l2=l2),
            latlon=True,
        )
        catchment_obj.do_upscale = True
        catchment_obj.is_upscaled = True
        catchment_obj.upscaled_resolution = l1
        return catchment_obj, basin_ds

    def test_upscale_mask_with_correct_coords_accepts_an_explicit_factor(self):
        """An explicit factor still reports the resolution it upscales to."""
        catchment_obj, basin_ds = self._make_upscaled_catchment(0.5, 10, 10)
        mask_da = basin_ds["basin"].astype("int8")

        upscaled = catchment_obj.upscale_mask_with_correct_coords(mask_da, factor=5)

        self.assertEqual(upscaled.shape, (2, 2))
        self.assertAlmostEqual(abs(float(upscaled["lon"][1] - upscaled["lon"][0])), 0.5)

    def test_write_mask_file_coarsens_an_upscaled_mask_to_l2(self):
        """An already upscaled mask still reaches L2 when L1 and L2 differ.

        The coordinates and their bounds have to describe the same grid; an
        L1 mask carrying the L2 name would leave them contradicting.
        """
        catchment_obj, basin_ds = self._make_upscaled_catchment(0.5, 10, 10)
        mask_file = self.tmp_path / "mask_upscaled_to_l2.nc"

        catchment_obj.write_mask_file(basin_ds, mask_file)

        with xr.open_dataset(mask_file) as mask_ds:
            self.assertEqual(mask_ds["mask_l2"].shape, (2, 2))
            spacing = abs(float(mask_ds["lon_l2"][1] - mask_ds["lon_l2"][0]))
            self.assertAlmostEqual(spacing, 0.5)
            bnds = mask_ds["lon_l2_bnds"].values
            self.assertAlmostEqual(abs(bnds[0, 1] - bnds[0, 0]), 0.5)

    def test_write_mask_file_keeps_a_single_cell_domain_needing_no_coarsening(self):
        """A one cell wide mask survives when L1 already equals L2.

        There is nothing to coarsen at a factor of one, so the mask is written
        as it stands instead of being run through the cell edge arithmetic.
        """
        catchment_obj, basin_ds = self._make_upscaled_catchment(0.1, 3, 1)
        mask_file = self.tmp_path / "mask_single_cell_factor_one.nc"

        catchment_obj.write_mask_file(basin_ds, mask_file)

        with xr.open_dataset(mask_file) as mask_ds:
            self.assertEqual(mask_ds["mask_l2"].shape, (3, 1))
            bnds = mask_ds["lon_l2_bnds"].values
            self.assertAlmostEqual(abs(bnds[0, 1] - bnds[0, 0]), 0.1)

    def test_write_mask_file_reports_a_domain_too_narrow_to_coarsen(self):
        """A mask thinner than one L2 cell is refused with a readable message.

        Coarsening trims the axis to nothing, so the message has to name the
        axis and both resolutions rather than surfacing an index error.
        """
        catchment_obj, basin_ds = self._make_upscaled_catchment(0.5, 10, 1)
        mask_file = self.tmp_path / "mask_too_narrow.nc"

        with self.assertRaises(ValueError) as ctx:
            catchment_obj.write_mask_file(basin_ds, mask_file)

        message = str(ctx.exception).lower()
        self.assertIn("lon", message)
        self.assertIn("0.5", message)
        self.assertIn("0.1", message)


class TestProjectedCatchment(unittest.TestCase):
    """Delineation on projected grids in metres, laid out like an EPSG:3035 fdir."""

    CELL_SIZE_M = 200.0
    CELL_AREA_KM2 = (CELL_SIZE_M / 1000) ** 2
    GAUGE_ID = 6335020

    def setUp(self):
        """Create a temporary output folder for every test."""
        self._tmpdir = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmpdir.cleanup)
        self.tmp_path = Path(self._tmpdir.name)

    @staticmethod
    def create_axes(
        row_count, col_count, row_step_m=CELL_SIZE_M, col_step_m=CELL_SIZE_M
    ):
        """Create descending y and ascending x cell centres in metres.

        Args:
            row_count (int): Number of rows.
            col_count (int): Number of columns.
            row_step_m (float): Cell height in m.
            col_step_m (float): Cell width in m.

        Returns:
            Tuple of the y and the x cell centres.
        """
        y = 3_000_000.0 + row_step_m * (np.arange(row_count)[::-1] + 0.5)
        x = 4_000_000.0 + col_step_m * (np.arange(col_count) + 0.5)
        return y, x

    @staticmethod
    def create_row_rivers(row_count, col_count):
        """Create D8 flow directions in which every row drains east off the grid.

        Args:
            row_count (int): Number of rows.
            col_count (int): Number of columns.

        Returns:
            Array of D8 codes, a cell in column c drains c + 1 cells.
        """
        return np.ones((row_count, col_count), dtype=np.int16)

    @staticmethod
    def create_river_tree(row_count, col_count):
        """Create D8 flow directions draining the whole grid into its middle row.

        The middle row flows east off the grid, the rows above it flow south and
        the rows below it north, so a middle row cell in column c drains every
        cell of the columns 0 to c.

        Args:
            row_count (int): Number of rows, odd so the middle row is centred.
            col_count (int): Number of columns.

        Returns:
            Array of D8 codes.
        """
        middle_row = row_count // 2
        fdir = np.full((row_count, col_count), 4, dtype=np.int16)
        fdir[middle_row + 1 :] = 64
        fdir[middle_row] = 1
        return fdir

    @staticmethod
    def create_projected_catchment(fdir, y, x):
        """Create a Catchment on a projected grid held in memory.

        Args:
            fdir (numpy.ndarray): D8 flow directions.
            y (numpy.ndarray): Descending y cell centres in m.
            x (numpy.ndarray): Ascending x cell centres in m.

        Returns:
            The Catchment, with latlon set to False.
        """
        ds = xr.Dataset({"fdir": (("lat", "lon"), fdir)}, coords={"lat": y, "lon": x})
        return catchment.Catchment(
            ds,
            "fdir",
            var="fdir",
            ftype="d8",
            transform=catchment.get_transformation_matrix_nc(ds, "fdir"),
            out_var_name="basin_ids.nc",
            latlon=False,
        )

    def write_projected_fdir_netcdf(self, fdir, y, x):
        """Write flow directions like a projected fdir NetCDF with 2D lat/lon.

        Args:
            fdir (numpy.ndarray): D8 flow directions.
            y (numpy.ndarray): Descending y cell centres in m.
            x (numpy.ndarray): Ascending x cell centres in m.

        Returns:
            Path of the written file.
        """
        lon, lat = np.meshgrid(
            np.linspace(10.0, 10.1, x.size), np.linspace(50.1, 50.0, y.size)
        )
        ds = xr.Dataset(
            {"fdir": (("y", "x"), fdir)},
            coords={
                "x": ("x", x, {"axis": "X", "units": "m"}),
                "y": ("y", y, {"axis": "Y", "units": "m"}),
                "lon": (("y", "x"), lon, {"units": "degrees_east"}),
                "lat": (("y", "x"), lat, {"units": "degrees_north"}),
            },
        )
        fdir_file = self.tmp_path / "fdir.nc"
        ds.to_netcdf(fdir_file, encoding={"fdir": {"_FillValue": -9999}})
        return fdir_file

    @staticmethod
    def create_catchment_kwargs(fdir_file, output_dir):
        """Create the create_catchment arguments shared by the projected runs.

        Args:
            fdir_file (Path): Flow direction file.
            output_dir (Path): Output folder of the run.

        Returns:
            Dictionary of keyword arguments.
        """
        return {
            "input_file": fdir_file,
            "output_path": output_dir,
            "var_name": "fdir",
            "var": "fdir",
            "ftype": "d8",
            "latlon": False,
            "max_distance_m": 300.0,
            "max_error": 0.1,
            "frame": 0,
            "output_vars": ["basin"],
            "raise_on_fallback": True,
        }

    def test_projected_cell_area_is_the_product_of_the_axis_steps(self):
        """Give every projected cell the area of its axis steps, in grid order."""
        y, x = self.create_axes(5, 8, row_step_m=100.0)
        projected_catchment = self.create_projected_catchment(
            self.create_row_rivers(5, 8), y, x
        )
        projected_catchment.compute_cell_area()
        self.assertEqual(projected_catchment.cell_area.shape, (5, 8))
        np.testing.assert_allclose(projected_catchment.cell_area, 0.1 * 0.2)

    def test_projected_upstream_area_counts_cells_in_km2(self):
        """Sum the projected cell areas in km2 along rows draining east."""
        y, x = self.create_axes(5, 8)
        projected_catchment = self.create_projected_catchment(
            self.create_row_rivers(5, 8), y, x
        )
        upstream_area = projected_catchment.calc_upstream_area()
        np.testing.assert_allclose(upstream_area[:, 0], self.CELL_AREA_KM2)
        np.testing.assert_allclose(upstream_area[:, -1], 8 * self.CELL_AREA_KM2)

    def test_write_keeps_the_full_extent_of_a_projected_grid(self):
        """Write every row of a projected grid instead of cropping to 84 N to 56 S."""
        y, x = self.create_axes(5, 8)
        projected_catchment = self.create_projected_catchment(
            self.create_row_rivers(5, 8), y, x
        )
        projected_catchment.get_basins()
        projected_catchment.write(
            self.tmp_path, single_file=True, frame=0, variables=["basin"]
        )
        with xr.open_dataset(self.tmp_path / "basin_ids.nc") as basin_ds:
            np.testing.assert_array_equal(basin_ds["lat"].values, y)
            np.testing.assert_array_equal(basin_ds["lon"].values, x)

    def test_multi_gauge_run_drops_an_unmatched_gauge_with_a_warning(self):
        """Drop a gauge without matching outlet and still delineate the other."""
        y, x = self.create_axes(5, 8)
        fdir_file = self.write_projected_fdir_netcdf(self.create_row_rivers(5, 8), y, x)
        output_dir = self.tmp_path / "multi_gauge"
        unmatched_gauge_id = self.GAUGE_ID + 10
        with self.assertLogs("mhm_tools.pre.catchment", level="WARNING") as logs:
            catchment.create_catchment(
                gauge_coords=[(float(y[1]), float(x[5])), (float(y[3]), float(x[5]))],
                gauge_ids=[self.GAUGE_ID, unmatched_gauge_id],
                # the first gauge drains six cells, no cell near the second 100 km2
                ref_catchment_area=[6 * self.CELL_AREA_KM2, 100.0],
                **self.create_catchment_kwargs(fdir_file, output_dir),
            )
        self.assertTrue(
            any(f"Dropping gauge {unmatched_gauge_id}" in line for line in logs.output)
        )
        gauges_info = pd.read_csv(output_dir / "gauges_info.csv")
        self.assertEqual(gauges_info["id"].tolist(), [self.GAUGE_ID])
        with xr.open_dataset(output_dir / "basin_ids.nc") as basin_ds:
            self.assertEqual((basin_ds.sizes["lat"], basin_ds.sizes["lon"]), (5, 8))

    def test_projected_netcdf_gauge_is_moved_onto_the_river_by_area(self):
        """Move a gauge beside the river onto the river cell draining its area."""
        y, x = self.create_axes(7, 9)
        fdir_file = self.write_projected_fdir_netcdf(self.create_river_tree(7, 9), y, x)
        output_dir = self.tmp_path / "by_area"
        # the river cell in column 4 drains the columns 0 to 4 of all 7 rows
        river_area_km2 = 5 * 7 * self.CELL_AREA_KM2
        catchment.create_catchment(
            gauge_coords=(float(y[2]), float(x[4])),
            gauge_ids=self.GAUGE_ID,
            ref_catchment_area=river_area_km2,
            **self.create_catchment_kwargs(fdir_file, output_dir),
        )
        gauge_info = pd.read_csv(output_dir / "gauges_info.csv").iloc[0]
        self.assertEqual((gauge_info["lat"], gauge_info["lon"]), (y[3], x[4]))
        self.assertAlmostEqual(gauge_info["distance"], 0.2)
        self.assertAlmostEqual(gauge_info["area"], river_area_km2)
        self.assertEqual(gauge_info["method"], "area_basinex")
        with xr.open_dataset(output_dir / "basin_ids.nc") as basin_ds:
            np.testing.assert_array_equal(basin_ds["lon"].values, x[:5])
            self.assertEqual(int((basin_ds["basin"] > 0).sum()), 5 * 7)

    def test_projected_netcdf_gauge_is_moved_onto_the_river_by_shape(self):
        """Move a gauge onto the river cell whose catchment is its reference shape."""
        y, x = self.create_axes(7, 9)
        fdir_file = self.write_projected_fdir_netcdf(self.create_river_tree(7, 9), y, x)
        shape_dir = self.tmp_path / "reference_shapes"
        shape_dir.mkdir()
        # the catchment of the river cell in column 4, the columns 0 to 4
        reference_shape = box(4_000_000.0, 3_000_000.0, 4_001_000.0, 3_001_400.0)
        gpd.GeoDataFrame(geometry=[reference_shape], crs="EPSG:3035").to_file(
            shape_dir / f"basin_{self.GAUGE_ID}.shp"
        )
        output_dir = self.tmp_path / "by_shape"
        catchment.create_catchment(
            gauge_coords=(float(y[2]), float(x[4])),
            gauge_ids=self.GAUGE_ID,
            shape_folder=shape_dir,
            **self.create_catchment_kwargs(fdir_file, output_dir),
        )
        gauge_info = pd.read_csv(output_dir / "gauges_info.csv").iloc[0]
        self.assertEqual((gauge_info["lat"], gauge_info["lon"]), (y[3], x[4]))
        self.assertEqual(gauge_info["method"], "shape_iou")
        self.assertAlmostEqual(gauge_info["shape_error"], 0.0)
        self.assertAlmostEqual(gauge_info["distance"], 0.2)
