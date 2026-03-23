import tempfile
import unittest
from pathlib import Path

import numpy as np
import xarray as xr

from mhm_tools.common.file_handler import get_xarray_ds_from_file
from mhm_tools.common.xarray_utils import get_coord_key
from mhm_tools.pre import catchment

HERE = Path(__file__).parent


class TestCatchment(unittest.TestCase):
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

    def test_merge_catchment(self):
        self.test_write()  # Ensure files are written first

        path1 = self.output_path / "hydro1.nc"
        path2 = self.output_path / "hydro2.nc"
        out_path = self.output_path / "hydro_merged_03min.nc"

        self.assertTrue(path1.is_file(), "hydro1.nc does not exist.")
        with xr.open_dataset(path2, engine="netcdf4") as ds1:
            lat_key = get_coord_key(ds1, lat=True, raise_exception=False)
            lon_key = get_coord_key(ds1, lon=True, raise_exception=False)
            # print(ds1[lat_key].shape)
            # print(ds1[lon_key].shape)
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

        res = catchment.Resolution(l2=0.5, l2_file=l2_path)
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

        res = catchment.Resolution(l2=0.5000005, l2_file=l2_path)
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
            catchment.Resolution(l2=0.500002, l2_file=l2_path)

    def test_find_best_gauge_location_best_candidate(self):
        c = self._make_small_catchment()
        upstream_area = np.zeros((5, 5), dtype=float)
        upstream_area[2, 1] = 100.0
        upstream_area[2, 3] = 105.0
        best_coord_basinex, error_basinex = c.find_best_gauge_location(
            upstream_area,
            gauge_coords=(2.0, 2.0),
            ref_catchment_area=103.0,
            max_distance_cells=4,
            max_error=0.05,
            method="basinex",
            raise_on_fallback=True,
        )
        best_coord_burek, error_burek = c.find_best_gauge_location(
            upstream_area,
            gauge_coords=(2.0, 2.0),
            ref_catchment_area=103.0,
            max_distance_cells=4,
            max_error=0.05,
            method="burek",
            raise_on_fallback=True,
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
        best_coord_basinex, error_basinex = c.find_best_gauge_location(
            upstream_area,
            gauge_coords=(2.0, 2.0),
            ref_catchment_area=100.0,
            max_distance_cells=4,
            max_error=0.05,
            method="basinex",
            raise_on_fallback=True,
        )
        best_coord_burek, error_burek = c.find_best_gauge_location(
            upstream_area,
            gauge_coords=(2.0, 2.0),
            ref_catchment_area=100.0,
            max_distance_cells=4,
            max_error=0.05,
            method="burek",
            raise_on_fallback=True,
        )
        self.assertEqual(best_coord_basinex, best_coord_burek)
        self.assertAlmostEqual(error_basinex, error_burek)
        self.assertEqual(best_coord_basinex, (2, 1))
        self.assertAlmostEqual(error_basinex, 0.0)
        self.assertEqual(best_coord_burek, (2, 1))
        self.assertAlmostEqual(error_burek, 0.0)

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
        d = c._distance_100m_units(1, 0, lat_deg=0.0)
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

        d_small = c_small._distance_100m_units(1, 0, lat_deg=0.0)
        d_large = c_large._distance_100m_units(1, 0, lat_deg=0.0)
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

        resolutions = catchment.Resolution(
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

        resolutions = catchment.Resolution(
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
    REF_AREA = [113.33, 1123.61]  # optional reference area in km2, e.g. 25400

    def test_delineate_basin_without_ref(self):
        """Integration-style test: delineate a basin using gauge coords without providing a ref area."""
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
        c.delineate_basin(
            (self.GAUGE_LAT[1], self.GAUGE_LON[1]), raise_on_sanity_check=False
        )

        self.assertIsNotNone(c.basin)
        self.assertTrue(
            np.any(c.catchment_mask),
            "No catchment cells found for provided gauge coordinates",
        )

        # compute area of resulting catchment using create_cell_area
        cell_area = catchment.create_cell_area(
            ds, lat_name=lat_key, lon_name=lon_key
        ).data
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
        """Integration-style test: delineate a basin with an explicit reference area and check closeness."""
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
            c.delineate_basin(
                (lat, lon),
                ref_catchment_area=float(ref_area),
                max_distance_cells=10,
                max_error=0.05,
                raise_on_sanity_check=False,
            )

            self.assertIsNotNone(c.basin)
            self.assertTrue(
                np.any(c.catchment_mask),
                "No catchment cells found for provided gauge coordinates and ref area",
            )

            cell_area = catchment.create_cell_area(
                ds, lat_name=lat_key, lon_name=lon_key
            ).data
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
        ):
            self.skipTest(
                "Set FDIR_PATH, GAUGE_LAT and GAUGE_LON in this test file to run this integration test."
            )

        gauge_ids = [101, 202]
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

        snapped = []
        resolutions = catchment.Resolution(l0=0.001953125, l1=1 / 32)
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
                    gauge_ids=[gauge_ids[idx]],
                    latlon=True,
                    frame=1,
                    resolutions=resolutions,
                )
            id_path = out_dir / "idgauges.nc"
            self.assertTrue(id_path.is_file())
            with xr.open_dataset(id_path) as id_ds:
                id_da = id_ds["idgauges"]
                matches = np.argwhere(id_da.values == gauge_ids[idx])
                self.assertEqual(matches.shape[0], 1)
                row, col = matches[0]
                lat_val = float(id_da["lat"].values[row])
                lon_val = float(id_da["lon"].values[col])
            snapped.append((lat_val, lon_val, gauge_ids[idx]))

        combined_dir = self.tmp_path / "combined"
        combined_dir.mkdir(parents=True, exist_ok=True)
        catchment.create_catchment(
            input_file=str(self.FDIR_PATH),
            output_path=str(combined_dir),
            var_name=var_name,
            var="fdir",
            ftype="d8",
            gauge_coords=gauge_coords,
            gauge_ids=gauge_ids,
            latlon=True,
            frame=1,
            resolutions=resolutions,
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

        gauge_ids = [101, 202]
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

        resolutions = catchment.Resolution(l0=0.001953125, l1=1 / 32)

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
            gauge_ids=gauge_ids,
            latlon=True,
            frame=1,
            resolutions=resolutions,
            ncpus=1,
        )
        catchment.create_catchment(
            input_file=str(self.FDIR_PATH),
            output_path=str(par_dir),
            var_name=var_name,
            var="fdir",
            ftype="d8",
            gauge_coords=gauge_coords,
            gauge_ids=gauge_ids,
            latlon=True,
            frame=1,
            resolutions=resolutions,
            ncpus=2,
        )

        seq_path = seq_dir / "idgauges.nc"
        par_path = par_dir / "idgauges.nc"
        self.assertTrue(seq_path.is_file())
        self.assertTrue(par_path.is_file())

        with xr.open_dataset(seq_path) as seq_ds, xr.open_dataset(par_path) as par_ds:
            np.testing.assert_array_equal(
                seq_ds["idgauges"].values, par_ds["idgauges"].values
            )
