import unittest
from pathlib import Path

import numpy as np
import xarray as xr

from mhm_tools.common.file_handler import get_xarray_ds_from_file
from mhm_tools.common.xarray_utils import get_coord_key
from mhm_tools.pre import catchment

HERE = Path(__file__).parent


class TestCatchment(unittest.TestCase):
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
        self.assertEqual(c.ds, self.ds)

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

        output_path = Path(HERE, "files")
        output_path.mkdir(parents=True, exist_ok=True)

        for c, out_var_name in zip(catchments, output_var_names):
            c.get_basins()
            c.get_facc()
            c.get_grid_area()
            c.get_upstream_area()
            c.write(output_path, single_file=True)
            output_file = output_path / out_var_name
            self.assertTrue(output_file.exists(), f"Failed to create {out_var_name}")

    def test_merge_catchment(self):
        self.test_write()  # Ensure files are written first

        path1 = Path(HERE, "files/hydro1.nc")
        path2 = Path(HERE, "files/hydro2.nc")
        out_path = Path(HERE, "files/hydro_merged_03min.nc")

        self.assertTrue(path1.is_file(), "hydro1.nc does not exist.")
        with xr.open_dataset(path2, engine="netcdf4") as ds1:
            lat_key = get_coord_key(ds1, lat=True, raise_exception=False)
            lon_key = get_coord_key(ds1, lon=True, raise_exception=False)
            # print(ds1[lat_key].shape)
            # print(ds1[lon_key].shape)
        self.assertTrue(path2.is_file(), "hydro2.nc does not exist.")

        catchment.merge_catchment(path1, path2, out_path)
        self.assertTrue(out_path.is_file())



    def tearDown(self):
        files_to_remove = ["hydro1.nc", "hydro2.nc", "hydro_merged_03min.nc"]
        for filename in files_to_remove:
            file_path = Path(HERE, "files", filename)
            try:
                if file_path.exists():
                    file_path.unlink()
            except Exception as e:
                print(f"Error removing file {file_path}: {e}")

    # ------------------------------------------------------------------
    # Optional integration tests for delineation using a real flow-direction
    # file and gauge coordinates. These tests are skipped unless you provide
    # the path to your fdir file and the gauge coordinates below.
    # To run, set the variables FDIR_PATH, GAUGE_LAT, GAUGE_LON (and
    # optionally REF_AREA) to appropriate values.
    # ------------------------------------------------------------------

    # Replace these placeholders with your real test inputs before running.
    FDIR_PATH = Path(HERE, "files", "test_create_catchment",  "fdir.nc")
    FDIR_VAR = "fdir"  # change to the variable name in your file if different
    # GAUGE_LAT = [48.445978]
    # GAUGE_LON = [8.70628]
    # REF_AREA = [1123.61]  # optional reference area in km2, e.g. 25400

    GAUGE_LAT = [49.292013, 48.445978]
    GAUGE_LON = [8.679113, 8.70628]
    REF_AREA = [113.33, 1123.61]  # optional reference area in km2, e.g. 25400

    def test_delineate_basin_without_ref(self):
        """Integration-style test: delineate a basin using gauge coords without providing a ref area."""
        if not self.FDIR_PATH.exists() or self.GAUGE_LAT is None or self.GAUGE_LON is None:
            self.skipTest("Set FDIR_PATH, GAUGE_LAT and GAUGE_LON in this test file to run this integration test.")

        ds = get_xarray_ds_from_file(str(self.FDIR_PATH))
        var_name = self.FDIR_VAR if self.FDIR_VAR in ds.data_vars else list(ds.data_vars)[0]

        # get coordinate arrays and convert lat/lon to nearest indices
        lat_key = get_coord_key(ds, lat=True, raise_exception=False)
        lon_key = get_coord_key(ds, lon=True, raise_exception=False)

        for lat, lon, ref_area in zip(self.GAUGE_LAT, self.GAUGE_LON, self.REF_AREA):
            c = catchment.Catchment(ds, var_name, var="fdir", ftype="d8", transform=self.transform, latlon=True)
            c.delineate_basin((lat, lon))

            self.assertIsNotNone(c.basin)
            self.assertTrue(np.any(c.catchment_mask), "No catchment cells found for provided gauge coordinates")

            # compute area of resulting catchment using create_cell_area
            cell_area = catchment.create_cell_area(ds, lat_name=lat_key, lon_name=lon_key).data
            area_km2 = float(np.sum(cell_area[c.catchment_mask]))
            self.assertGreater(area_km2, 0.0)
            rel_diff = abs(area_km2 - float(ref_area)) / float(ref_area)
            self.assertLessEqual(rel_diff, 0.05, f"Delineated area {area_km2} differs more than 5% from REF_AREA {ref_area}")
            print(f"No ref Delineated area: {area_km2} km², Reference area: {ref_area} km², Relative difference: {rel_diff*100:.2f}%")


    def test_delineate_basin_with_ref(self):
        """Integration-style test: delineate a basin with an explicit reference area and check closeness."""
        if not self.FDIR_PATH.exists() or self.GAUGE_LAT is None or self.GAUGE_LON is None or self.REF_AREA is None:
            self.skipTest("Set FDIR_PATH, GAUGE_LAT, GAUGE_LON and REF_AREA in this test file to run this integration test.")

        ds = get_xarray_ds_from_file(str(self.FDIR_PATH))
        var_name = self.FDIR_VAR if self.FDIR_VAR in ds.data_vars else list(ds.data_vars)[0]

        lat_key = get_coord_key(ds, lat=True, raise_exception=False)
        lon_key = get_coord_key(ds, lon=True, raise_exception=False)

        for lat, lon, ref_area in zip(self.GAUGE_LAT, self.GAUGE_LON, self.REF_AREA):
            c = catchment.Catchment(ds, var_name, var="fdir", ftype="d8", transform=self.transform, latlon=True)
            c.delineate_basin((lat, lon), ref_catchment_area=float(ref_area), max_distance_cells=10, max_error=0.05)

            self.assertIsNotNone(c.basin)
            self.assertTrue(np.any(c.catchment_mask), "No catchment cells found for provided gauge coordinates and ref area")

            cell_area = catchment.create_cell_area(ds, lat_name=lat_key, lon_name=lon_key).data
            area_km2 = float(np.sum(cell_area[c.catchment_mask]))

            # check that computed area is reasonably close to the reference (5% tolerance)
            rel_diff = abs(area_km2 - float(ref_area)) / float(ref_area)
            self.assertLessEqual(rel_diff, 0.05, f"Delineated area {area_km2} differs more than 5% from REF_AREA {ref_area}")
            print(f"With ref Delineated area: {area_km2} km², Reference area: {ref_area} km², Relative difference: {rel_diff*100:.2f}%")

if __name__ == "__main__":
    unittest.main()
