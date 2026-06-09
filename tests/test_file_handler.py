import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import xarray as xr

import mhm_tools.common.file_handler as fh
from mhm_tools.common.provenance import CREATED_ATTR, HISTORY_ATTR, VERSION_ATTR

try: 
    from mhm_tools._version import __version__
except ImportError:
    __version__ = "not_available"

def _engine_available(engine: str) -> bool:
    if engine != "h5netcdf":
        return True
    try:
        ds = xr.Dataset({"var": (("y", "x"), np.zeros((1, 1), dtype=np.float32))})
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "probe.nc"
            ds.to_netcdf(path, engine=engine)
            xr.open_dataset(path, engine=engine).close()
        return True
    except Exception:
        return False


class BaseDatasetMixin:
    def make_simple_ds(self, dtype=np.float32):
        # 3 x 4 grid, lat descending
        lat = np.array([52.0, 51.0, 50.0])
        lon = np.array([10.0, 11.0, 12.0, 13.0])
        data = np.arange(lat.size * lon.size, dtype=dtype).reshape(lat.size, lon.size)
        return xr.Dataset(
            {"var": (("lat", "lon"), data)}, coords={"lat": lat, "lon": lon}
        )

    def make_ds_with_time(self, lat_coord="lat", lon_coord="lon"):
        base = self.make_simple_ds()
        time = np.array(["2000-01-01", "2000-01-02"], dtype="datetime64[ns]")
        data3d = np.stack([base["var"].values, base["var"].values + 100], axis=0)
        return xr.Dataset(
            {"var": (("time", lat_coord, lon_coord), data3d)},
            coords={"time": time, lat_coord: base["lat"], lon_coord: base["lon"]},
        )


class TestCreateHeader(unittest.TestCase, BaseDatasetMixin):

    def test_create_header_returns_dir_without_write(self):
        for dtype in [np.float32, np.int32, np.uint16]:
            with self.subTest(dtype=dtype):
                ds = self.make_simple_ds(dtype=dtype)
                header_dict = fh.create_header(ds, no_data_value="-9999")
                self.assertEqual(header_dict["ncols"], ds.sizes["lon"])
                self.assertEqual(header_dict["nrows"], ds.sizes["lat"])
                self.assertAlmostEqual(float(header_dict["cellsize"]), 1.0)
                self.assertAlmostEqual(
                    float(header_dict["xllcorner"]), ds["lon"].values.min() - 0.5
                )
                self.assertAlmostEqual(
                    float(header_dict["yllcorner"]), ds["lat"].values.min() - 0.5
                )
                if dtype in [np.int32, np.uint16]:
                    self.assertEqual(header_dict["nodata_value"], -9999)
                else:
                    self.assertEqual(header_dict["nodata_value"], -9999.0)

    def test_create_header_writes_file(self):
        ds = self.make_simple_ds()
        with tempfile.TemporaryDirectory() as td:
            expected_path = Path(td) / "header.txt"
            self.assertFalse(expected_path.is_file())
            header_dict = fh.create_header(ds, output_path=Path(td), no_data_value=2245)
            self.assertTrue(expected_path.is_file())
            self.assertEqual(header_dict["ncols"], ds.sizes["lon"])
            self.assertEqual(header_dict["nrows"], ds.sizes["lat"])
            self.assertEqual(header_dict["nodata_value"], 2245)


class TestChunkHelpers(unittest.TestCase, BaseDatasetMixin):
    def test_chunk_dataset_space_only_with_time(self):
        ds = self.make_ds_with_time()
        for mem_gib in [0.1, 0.5, 1.0, 4.0]:
            with self.subTest(mem_gib=mem_gib):
                chunks = fh.chunk_dataset_space_only(ds, mem_gib)
                self.assertEqual(chunks.get("time"), -1)
                self.assertTrue(1 <= chunks["lat"] <= ds.sizes["lat"])
                self.assertTrue(1 <= chunks["lon"] <= ds.sizes["lon"])

    def test_chunk_dataset_space_only_no_time(self):
        ds = self.make_simple_ds()
        chunks = fh.chunk_dataset_space_only(ds, 1.0)
        self.assertNotIn("time", chunks)
        self.assertTrue(1 <= chunks["lat"] <= ds.sizes["lat"])
        self.assertTrue(1 <= chunks["lon"] <= ds.sizes["lon"])

    def test_chunk_dataset_space_and_time(self):
        ds = self.make_ds_with_time()
        for mem_gib in [0.1, 0.5, 1.0, 2.0]:
            with self.subTest(mem_gib=mem_gib):
                chunks = fh.chunk_dataset_space_and_time(ds, mem_gib)
                self.assertIn("time", chunks)
                self.assertGreaterEqual(chunks["time"], 1)
                self.assertTrue(1 <= chunks["lat"] <= ds.sizes["lat"])
                self.assertTrue(1 <= chunks["lon"] <= ds.sizes["lon"])

    def test_chunk_dataset_switches_on_enum(self):
        ds = self.make_ds_with_time()
        flags = {"space": False, "space_time": False}

        def _space_only(_ds, _mem):
            flags["space"] = True
            return {"lat": 2, "lon": 2, "time": -1}

        def _space_time(_ds, _mem):
            flags["space_time"] = True
            return {"lat": 1, "lon": 2, "time": 1}

        with patch.object(
            fh, "chunk_dataset_space_only", side_effect=_space_only
        ), patch.object(fh, "chunk_dataset_space_and_time", side_effect=_space_time):

            out_space = fh.chunk_dataset(ds, fh.ChunkType.SPACE, 1.0)
            self.assertTrue(flags["space"])
            self.assertFalse(flags["space_time"])
            self.assertIn("lat", out_space.chunks)
            self.assertIn("time", out_space.chunks)

            flags = {"space": False, "space_time": False}
            out_time = fh.chunk_dataset(ds, fh.ChunkType.TIME, 1.0)
            self.assertTrue(flags["space_time"])
            self.assertFalse(flags["space"])
            self.assertIn("time", out_time.chunks)


class TestAsciiReadWrite(unittest.TestCase, BaseDatasetMixin):

    def test_write_and_read_ascii_roundtrip(self):
        ds = self.make_simple_ds()
        ds["var"].attrs["nodata_value"] = -9999

        with tempfile.TemporaryDirectory() as td:
            asc_path = Path(td) / "grid.asc"
            fh.write_xarray_to_ascii(ds, asc_path, data_var="var")
            self.assertTrue(asc_path.is_file())

            # header sanity
            text = asc_path.read_text().splitlines()
            print(text)
            self.assertTrue(any(line.startswith("ncols") for line in text))
            self.assertTrue(any(line.startswith("nrows") for line in text))
            self.assertTrue(any(line.startswith("cellsize") for line in text))

            # read back via helper
            ds_back = fh.read_ascii_to_xarray(asc_path, var_name="var")
            self.assertIn("var", ds_back.data_vars)
            self.assertIn("lat", ds_back.coords)
            self.assertIn("lon", ds_back.coords)
            self.assertEqual(ds_back["var"].shape, ds["var"].shape)
            self.assertEqual(ds_back["var"].mean(), ds["var"].mean())
            self.assertEqual(ds_back["var"].max(), ds["var"].max())
            self.assertEqual(ds_back["var"].min(), ds["var"].min())

    # def test_write_xarray_to_ascii_formats_strings(self):
    #     lat = np.array([2, 1])
    #     lon = np.array([0, 1, 2])
    #     data = np.array([["a", "b", "c"], ["d", "e", "f"]], dtype="U1")
    #     ds = xr.Dataset(
    #         {"cats": (("lat", "lon"), data)}, coords={"lat": lat, "lon": lon}
    #     )

    #     with tempfile.TemporaryDirectory() as td:
    #         asc_path = Path(td) / "strings.asc"
    #         fh.write_xarray_to_ascii(ds, asc_path, data_var="cats")
    #         s = asc_path.read_text()
    #         self.assertIn("a b c", s)
    def test_get_xarray_ds_from_file_ascii_preserves_dtype_kind(self):
        lat = np.array([52.0, 51.0, 50.0])
        lon = np.array([10.0, 11.0, 12.0, 13.0])

        int_data = np.arange(lat.size * lon.size, dtype=np.int32).reshape(
            lat.size, lon.size
        )
        float_data = np.arange(lat.size * lon.size, dtype=np.float64).reshape(
            lat.size, lon.size
        )

        int_ds = xr.Dataset(
            {"var": (("lat", "lon"), int_data)}, coords={"lat": lat, "lon": lon}
        )
        float_ds = xr.Dataset(
            {"var": (("lat", "lon"), float_data)}, coords={"lat": lat, "lon": lon}
        )

        with tempfile.TemporaryDirectory() as td:
            int_path = Path(td) / "int.asc"
            float_path = Path(td) / "float.asc"
            fh.write_xarray_to_ascii(int_ds, int_path, data_var="var")
            fh.write_xarray_to_ascii(float_ds, float_path, data_var="var")

            int_out = fh.get_xarray_ds_from_file(int_path, var_name="var")
            float_out = fh.get_xarray_ds_from_file(float_path, var_name="var")

            self.assertTrue(
                np.issubdtype(int_out["var"].dtype, np.integer),
                f"Expected integer dtype, got {int_out['var'].dtype}",
            )
            self.assertTrue(
                np.issubdtype(float_out["var"].dtype, np.floating),
                f"Expected float dtype, got {float_out['var'].dtype}",
            )

    def test_write_xarray_to_ascii_multiple_vars_without_data_var_returns_none(self):
        lat = np.array([1, 0])
        lon = np.array([0, 1, 2])
        ds = xr.Dataset(
            {
                "a": (("lat", "lon"), np.ones((2, 3))),
                "b": (("lat", "lon"), np.zeros((2, 3))),
            },
            coords={"lat": lat, "lon": lon},
        )
        with tempfile.TemporaryDirectory() as td:
            asc_path = Path(td) / "multi.asc"
            self.assertFalse(asc_path.exists())
            result = fh.write_xarray_to_ascii(ds, asc_path, data_var=None)
            self.assertIsNone(result)
            self.assertFalse(asc_path.exists())


class TestWriteXarrayToFile(unittest.TestCase, BaseDatasetMixin):
    def test_write_xarray_to_file_nc(self):
        ds = self.make_simple_ds()
        ds.attrs[HISTORY_ATTR] = "previous processing step"
        engines = [e for e in ["netcdf4", "h5netcdf"] if _engine_available(e)]
        if not engines:
            self.skipTest("No NetCDF engine available for writing.")
        for engine in engines:
            with tempfile.TemporaryDirectory() as td:
                nc_path = Path(td) / "ds.nc"
                fh.write_xarray_to_file(ds, nc_path, engine=engine)
                self.assertTrue(nc_path.is_file())
                with xr.open_dataset(nc_path, engine=engine) as back:
                    self.assertIn("var", back)
                    if __version__ != "not_available":
                        self.assertEqual(back.attrs[VERSION_ATTR], __version__)
                    self.assertIn(CREATED_ATTR, back.attrs)
                    self.assertIn("previous processing step", back.attrs[HISTORY_ATTR])
                    self.assertIn("mhm-tools command:", back.attrs[HISTORY_ATTR])

    def test_write_xarray_to_file_asc(self):
        ds = self.make_simple_ds()
        with tempfile.TemporaryDirectory() as td:
            asc_path = Path(td) / "out.asc"
            fh.write_xarray_to_file(ds, asc_path, var_name="var")
            self.assertTrue(asc_path.is_file())
            text = asc_path.read_text(encoding="utf-8")
            self.assertNotIn(VERSION_ATTR, text)
            self.assertNotIn(CREATED_ATTR, text)
            back = fh.read_ascii_to_xarray(asc_path, var_name="var")
            self.assertIn("var", back)

    def test_write_xarray_to_file_unsupported_suffix_raises(self):
        ds = self.make_simple_ds()
        with tempfile.TemporaryDirectory() as td:
            out = Path(td) / "file.xyz"
            with self.assertRaises(NotImplementedError):
                fh.write_xarray_to_file(ds, out)

    def test_set_grid_ignores_incompatible_coords(self):
        time = np.array(["2017-01-01", "2017-01-02"], dtype="datetime64[ns]")
        da = xr.DataArray(
            np.random.rand(2, 2, 2),
            dims=("time", "lat", "lon"),
            coords={
                "time": ("time", time, {"bounds": "time_bnds"}),
                "lat": ("lat", [10.0, 11.0]),
                "lon": ("lon", [100.0, 101.0]),
            },
            name="v",
        )
        ds = da.to_dataset()
        ds["time_bnds"] = xr.DataArray(
            np.stack([time - np.timedelta64(1, "D"), time], axis=1),
            dims=("time", "bnds"),
            coords={"time": ds["time"]},
        )

        grid = fh.get_grid(ds)
        new_data = np.zeros((2, 2, 2))
        out = fh.set_grid(new_data, grid, "pet")

        self.assertEqual(out["pet"].dims, ("time", "lat", "lon"))
        self.assertIn("time_bnds", out.variables)


class TestGetXarrayDsFromFile(unittest.TestCase, BaseDatasetMixin):
    def test_get_xarray_ds_from_file_asci(self):
        lat = np.array([50.0, 51.0, 52.0])
        lon = np.array([10.0, 11.0, 12.0, 13.0])
        data = np.arange(lat.size * lon.size, dtype=np.float32).reshape(
            lat.size, lon.size
        )
        asc_lat_ds = xr.Dataset(
            {"var": (("lat", "lon"), data)}, coords={"lat": lat, "lon": lon}
        )

        with tempfile.TemporaryDirectory() as td:
            nc_path = Path(td) / "asc.nc"
            asc_lat_ds.to_netcdf(nc_path)

            with patch.object(fh, "read_dataset", return_value=asc_lat_ds):
                out = fh.get_xarray_ds_from_file(nc_path, force_ascending_y=True)
                self.assertGreater(out["lat"].values[-1], out["lat"].values[0])
                out.close()
                out = fh.get_xarray_ds_from_file(nc_path, force_decending_y=True)
                self.assertGreater(out["lat"].values[0], out["lat"].values[-1])
                out.close()

    def test_get_xarray_ds_from_file_nc_flow(self):
        ds_with_time = self.make_ds_with_time(lon_coord="x", lat_coord="y")
        engines = [e for e in ["netcdf4", "h5netcdf"] if _engine_available(e)]
        if not engines:
            self.skipTest("No NetCDF engine available for writing.")
        for engine in engines:
            with tempfile.TemporaryDirectory() as td:
                nc_path = Path(td) / "stub.nc"
                ds_with_time.to_netcdf(nc_path, engine=engine)

                with patch.object(
                    fh, "read_dataset", return_value=ds_with_time
                ), patch.object(
                    fh,
                    "chunk_dataset",
                    side_effect=lambda ds, ctype, mem: ds.chunk(
                        {"time": -1, "lat": 2, "lon": 2}
                    ),
                ):

                    out = fh.get_xarray_ds_from_file(
                        nc_path,
                        var_name=None,
                        chunking=True,
                        available_mem_gib=1.0,
                        chunk_type=fh.ChunkType.SPACE,
                        use_mfdataset=False,
                        engine=engine,
                        normalize_latlon_coords=True,
                        force_decending_y=True,
                    )
                    self.assertIn("lat", out.coords)
                    self.assertIn("lon", out.coords)
                    self.assertIn("time", out.sizes)
                    self.assertIsNotNone(out.chunks)

    def test_get_xarray_ds_from_file_nc_flow_cross_engine(self):
        ds_with_time = self.make_ds_with_time(lon_coord="x", lat_coord="y")
        engines = [e for e in ["netcdf4", "h5netcdf"] if _engine_available(e)]
        if len(engines) < 2:
            self.skipTest("Cross-engine test requires both netcdf4 and h5netcdf.")
        for i, j in [(0, 1), (1, 0)]:
            with tempfile.TemporaryDirectory() as td:
                nc_path = Path(td) / "stub.nc"
                ds_with_time.to_netcdf(nc_path, engine=engines[i])

                with patch.object(
                    fh, "read_dataset", return_value=ds_with_time
                ), patch.object(
                    fh,
                    "chunk_dataset",
                    side_effect=lambda ds, ctype, mem: ds.chunk(
                        {"time": -1, "lat": 2, "lon": 2}
                    ),
                ):

                    out = fh.get_xarray_ds_from_file(
                        nc_path,
                        var_name=None,
                        chunking=True,
                        available_mem_gib=1.0,
                        chunk_type=fh.ChunkType.SPACE,
                        use_mfdataset=False,
                        engine=engines[j],
                        normalize_latlon_coords=True,
                        force_decending_y=True,
                    )
                    self.assertIn("lat", out.coords)
                    self.assertIn("lon", out.coords)
                    self.assertIn("time", out.sizes)
                    self.assertIsNotNone(out.chunks)

    def test_get_xarray_ds_from_file_asc_flow(self):
        ds = self.make_simple_ds()
        with tempfile.TemporaryDirectory() as td:
            asc_path = Path(td) / "in.asc"
            fh.write_xarray_to_ascii(ds, asc_path, data_var="var")

            out = fh.get_xarray_ds_from_file(
                asc_path,
                var_name="var",
                chunking=False,
                available_mem_gib=None,
                chunk_type=fh.ChunkType.SPACE,
                normalize_latlon_coords=False,
                force_decending_y=True,
            )
            self.assertIn("var", out)
            self.assertIn("lon", out.coords)
            self.assertIn("lat", out.coords)
            for var in out.data_vars:
                self.assertIsNone(out[var].chunks)

    def test_get_xarray_ds_from_file_nonexistent_raises(self):
        with tempfile.TemporaryDirectory() as td:
            bad = Path(td) / "nope.nc"
            with self.assertRaises(ValueError):
                fh.get_xarray_ds_from_file(bad)

    def test_get_xarray_ds_from_file_unsupported_suffix_raises(self):
        with tempfile.TemporaryDirectory() as td:
            weird = Path(td) / "strange.xyz"
            weird.write_text("placeholder")
            with self.assertRaises(NotImplementedError):
                fh.get_xarray_ds_from_file(weird)

    def test_get_xarray_ds_from_file_force_y_order(self):
        # Build ds with ascending lat
        lat = np.array([52.0, 51.0, 50.0])
        lon = np.array([10.0, 11.0, 12.0, 13.0])
        data = np.arange(lat.size * lon.size, dtype=np.float32).reshape(
            lat.size, lon.size
        )
        asc_lat_ds = xr.Dataset(
            {"var": (("lat", "lon"), data)}, coords={"lat": lat, "lon": lon}
        )

        with tempfile.TemporaryDirectory() as td:
            nc_path = Path(td) / "asc.nc"
            asc_lat_ds.to_netcdf(nc_path)

            with patch.object(fh, "read_dataset", return_value=asc_lat_ds):
                out = fh.get_xarray_ds_from_file(nc_path, force_ascending_y=True)
                self.assertGreater(out["lat"].values[-1], out["lat"].values[0])
                out.close()
                out = fh.get_xarray_ds_from_file(nc_path, force_decending_y=True)
                self.assertGreater(out["lat"].values[0], out["lat"].values[-1])
                out.close()


class TestGetDatasetFromPath(unittest.TestCase, BaseDatasetMixin):
    def test_get_dataset_from_path_file(self):
        ds = self.make_simple_ds()
        with tempfile.TemporaryDirectory() as td:
            nc_path = Path(td) / "single.nc"
            ds.to_netcdf(nc_path)

            out = fh.get_dataset_from_path(nc_path)
            self.assertIn("lat", out.coords)
            self.assertIn("lon", out.coords)
            self.assertIn("var", out.data_vars)
            self.assertIsNone(out["var"].chunks)

    def test_get_dataset_from_path_directory(self):
        lat = np.array([50.0, 51.0, 52.0])
        lon = np.array([10.0, 11.0, 12.0, 13.0])
        data = np.arange(lat.size * lon.size, dtype=np.float32).reshape(
            lat.size, lon.size
        )
        asc_lat_ds = xr.Dataset(
            {"var": (("lat", "lon"), data)}, coords={"lat": lat, "lon": lon}
        )

        with tempfile.TemporaryDirectory() as td:
            base = Path(td)
            year_dir = base / "2000"
            year_dir.mkdir(parents=True, exist_ok=True)
            file_list = []
            for name in ["a.nc", "b.nc"]:
                p = year_dir / name
                p.write_text("stub")
                file_list.append(p)

            with patch.object(fh, "read_dataset", return_value=asc_lat_ds):
                out = fh.get_dataset_from_path(base, force_decending_y=True)
                self.assertGreater(out["lat"].values[0], out["lat"].values[-1])
                self.assertIn("var", out.data_vars)

    def test_get_dataset_from_path_directory_empty_raises(self):
        with tempfile.TemporaryDirectory() as td:
            base = Path(td)
            with self.assertRaises(ValueError):
                fh.get_dataset_from_path(base)

    def test_get_dataset_from_path_wildcard(self):
        ds = self.make_simple_ds()
        with tempfile.TemporaryDirectory() as td:
            base = Path(td)
            for name in ["a.nc", "b.nc"]:
                ds.to_netcdf(base / name)

            with patch.object(fh, "read_dataset", return_value=ds) as mocked:
                out = fh.get_dataset_from_path(str(base / "*.nc"))
                self.assertIn("var", out.data_vars)
                self.assertTrue(mocked.called)


class TestCropAndCoords(unittest.TestCase, BaseDatasetMixin):
    # def setUp(self):
    #     self.patcher_coord = patch.object(fh, "get_coord_key", side_effect=fh.get_coord_key)
    #     self.patcher_coord.start()

    # def tearDown(self):
    #     self.patcher_coord.stop()

    def test_crop_file_by_mask(self):
        ds = self.make_ds_with_time()
        mask_lat = np.array([51.5, 50.5])  # max=51.5, min=50.5
        mask_lon = np.array([10.5, 11.5])  # min=10.5, max=11.5
        mask_ds = xr.Dataset(coords={"lat": mask_lat, "lon": mask_lon})

        with patch.object(fh, "get_xarray_ds_from_file", return_value=mask_ds):
            cropped = fh.crop_file_by_mask(ds, mask_file="irrelevant.nc")
            self.assertEqual(cropped.sizes["lat"], 1)
            self.assertEqual(cropped.sizes["lon"], 1)
            self.assertTrue(np.isclose(cropped["lon"].values, 11.0))
            self.assertTrue(np.isclose(cropped["lat"].values, 51.0))

    def test_get_coord_values(self):
        ds = self.make_simple_ds()
        lats = fh.get_coord_values(ds, lat=True)
        lons = fh.get_coord_values(ds, lon=True)
        np.testing.assert_allclose(lats, ds["lat"].values)
        np.testing.assert_allclose(lons, ds["lon"].values)


if __name__ == "__main__":
    unittest.main()
