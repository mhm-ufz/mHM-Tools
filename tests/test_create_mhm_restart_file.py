import logging
import shutil
import unittest
from pathlib import Path

import mhm_tools as mt
import xarray as xr

HERE = Path(__file__).parent
TMP = HERE / "tmp"
TMP.mkdir(parents=True, exist_ok=True)


class TestCreateRestart(unittest.TestCase):
    def setUp(self):
        # setup read_morph_files test
        self.morph_dir = TMP / "morph"
        self.morph_dir.mkdir(parents=True, exist_ok=True)
        # create empty files in morph directory
        self.file_names = [
            "lai",
            "land_cover_1990",
            "land_cover_2020",
            "sand_content",
            "BLDFIE",
            "CLYPPT",
            "geology",
            "slope",
            "aspect",
            "SNDPPT",
        ]
        for n in self.file_names:
            (self.morph_dir / f"{n}.nc").touch()

    def tearDown(self) -> None:
        # self.morph_dir.rmdir()
        shutil.rmtree(TMP)
        return super().tearDown()

    def test_read_morph_files(self):
        mf = mt.pre.MorphFiles(self.morph_dir)
        assert mf.aspect == self.morph_dir / "aspect.nc"
        assert mf.slope == self.morph_dir / "slope.nc"
        assert mf.geology == self.morph_dir / "geology.nc"
        assert mf.clay_content == self.morph_dir / "CLYPPT.nc"
        assert mf.sand_content == self.morph_dir / "sand_content.nc"
        assert mf.bulk_density == self.morph_dir / "BLDFIE.nc"
        assert mf.land_cover == [
            self.morph_dir / "land_cover_1990.nc",
            self.morph_dir / "land_cover_2020.nc",
        ]
        assert mf.get_file("lai") == self.morph_dir / "lai.nc"

    def test_read_latlon(self):
        p = Path(HERE / "files" / "test_create_restart" / "latlon_0p0625.nc")
        d = mt.pre.Grid(file_path=self.morph_dir, latlon_file=p)
        assert d.l0.lon_min - 11.500977 < 1e-6
        assert d.l0.lon_max - 12.999023 < 1e-6
        assert d.l0.lat_max - 50.249023 < 1e-6
        assert d.l0.lat_min - 49.000977 < 1e-6
        assert d.l0.resolution - 0.001953125 < 1e-6
        assert d.l1.lon_min - 11.53125 < 1e-6
        assert d.l1.lon_max - 12.96875 < 1e-6
        assert d.l1.lat_max - 50.21875 < 1e-6
        assert d.l1.lat_min - 49.03125 < 1e-6
        assert d.l1.resolution - 0.0625 < 1e-6

    def test_split_grid(self):
        morph = Path(HERE / "files" / "test_create_restart")
        lon_min_target_grid = -10
        lon_max_target_grid = 10
        lat_min_target_grid = -10
        lat_max_target_grid = 10
        l0_resolution = 0.002
        l1_resolution = 0.1
        increment_l1 = 20  # thats 2 degree
        m = mt.pre.MHMRestartFile(
            input_file_path=morph,
            output_path=TMP,
            nml_template=morph / "mpr_mhm_template.nml",
            lon_min_target_grid=lon_min_target_grid,
            lon_max_target_grid=lon_max_target_grid,
            lat_min_target_grid=lat_min_target_grid,
            lat_max_target_grid=lat_max_target_grid,
            l0_resolution=l0_resolution,
            l1_resolution=l1_resolution,
            increment_l1=increment_l1,
            log_level=logging.ERROR,
        )
        # test setup successful
        assert m.grid.l0.lon_min - -10 < 1e-3
        assert m.grid.l0.lon_max - 10 < 1e-3
        assert m.grid.l0.lat_max - 10 < 1e-3
        assert m.grid.l0.lat_min - -10 < 1e-3
        assert m.grid.l0.resolution - 0.002 < 1e-6
        assert m.grid.l1.lon_min - -10 < 1e-3
        assert m.grid.l1.lon_max - 10 < 1e-3
        assert m.grid.l1.lat_max - 10 < 1e-3
        assert m.grid.l1.lat_min - -10 < 1e-3
        assert m.grid.l1.resolution - 0.1 < 1e-6
        assert m.grid.morph_files.geology == morph / "geology_0.002.nc"

        # split grid
        m._split_grid()
        assert len(m.subgrids) == 100

        for sd in reversed(m.subgrids):
            # print(sd.morph_files.geology, flush=True)
            with xr.open_dataset(sd.morph_files.geology) as ds:
                assert (
                    abs(float(ds["longitude"].min()) - sd.l0.lon_min)
                    - sd.l0.resolution / 2
                    < 1e-6
                )  # difference is half the resolution because the xarray grid provides the center of the cell
                assert (
                    abs(ds["longitude"].max() - sd.l0.lon_max) - sd.l0.resolution / 2
                    < 1e-6
                )
                assert (
                    abs(ds["latitude"].min() - sd.l0.lat_min) - sd.l0.resolution / 2
                    < 1e-6
                )
                assert (
                    abs(ds["latitude"].max() - sd.l0.lat_max) - sd.l0.resolution / 2
                    < 1e-6
                )
            assert sd.l1.get_n_lon() == 20
            assert sd.l1.get_n_lat() == 20
            assert sd.l0.get_n_lon() == 1000
            assert sd.l0.get_n_lat() == 1000

    def test_write_namelists(self):
        pass


if __name__ == "__main__":
    unittest.main()
