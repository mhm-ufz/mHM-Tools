import unittest
from pathlib import Path
import shutil

import xarray as xr

import mhm_tools as mt

HERE = Path(__file__).parent
TMP = HERE / "tmp"
TMP.mkdir(parents=True, exist_ok=True)


class TestCreateRestart(unittest.TestCase):
    def setUp(self):
        # setup read_morph_files test
        self.morph_dir = TMP / "morph"
        self.morph_dir.mkdir(parents=True, exist_ok=True)
        # create empty files in morph directory
        self.file_names = ["lai", "land_cover_1990", "land_cover_2020", 'sand_content', 'BLDFIE', 'CLYPPT', 'geology', 'slope', 'aspect', 'SNDPPT']
        for n in self.file_names:
            (self.morph_dir / f"{n}.nc").touch()
    
    def tearDown(self) -> None:
        # self.morph_dir.rmdir()
        shutil.rmtree(TMP)
        return super().tearDown()

    def test_read_morph_files(self):
        mf = mt.pre.MorphFiles(self.morph_dir)
        self.assertTrue(mf.aspect == self.morph_dir / "aspect.nc")
        self.assertTrue(mf.slope == self.morph_dir / "slope.nc")
        self.assertTrue(mf.geology == self.morph_dir / "geology.nc")
        self.assertTrue(mf.clay_content == self.morph_dir / "CLYPPT.nc")
        self.assertTrue(mf.sand_content == self.morph_dir / "sand_content.nc")
        self.assertTrue(mf.bulk_density == self.morph_dir / "BLDFIE.nc")
        self.assertTrue(mf.land_cover == [self.morph_dir / "land_cover_1990.nc", self.morph_dir / "land_cover_2020.nc"])
        self.assertTrue(mf.get_file('lai') == self.morph_dir / "lai.nc")
    
    def test_read_latlon(self):
        p = Path(HERE / "files" / "test_create_restart" / "latlon_0p0625.nc")
        d = mt.pre.Domain(file_path=self.morph_dir, latlon_file=p)
        self.assertTrue(d.l0.lon_min - 11.500977 < 1e-6)
        self.assertTrue(d.l0.lon_max  -  12.999023  < 1e-6)
        self.assertTrue(d.l0.lat_max  -  50.249023 < 1e-6)
        self.assertTrue(d.l0.lat_min  -  49.000977 < 1e-6)
        self.assertTrue(d.l0.resolution  -  0.001953125 < 1e-6)
        self.assertTrue(d.l1.lon_min  -  11.53125 < 1e-6)
        self.assertTrue(d.l1.lon_max  -  12.96875 < 1e-6)
        self.assertTrue(d.l1.lat_max  -  50.21875 < 1e-6)
        self.assertTrue(d.l1.lat_min  -  49.03125 < 1e-6)
        self.assertTrue(d.l1.resolution  -  0.0625 < 1e-6)

    
    def test_split_domain(self):
        pass
        morph = Path(HERE / "files" / "test_create_restart")
        lon_min_target_grid=-10
        lon_max_target_grid=10
        lat_min_target_grid=-10
        lat_max_target_grid=10
        l0_resolution=0.002
        l1_resolution=0.1
        increment_l1 = 20 # thats 2 degree
        m = mt.pre.MHMRestartFile(input_file_path=morph, output_path=TMP, nml_template=morph / 'mpr_mhm_template.nml', lon_min_target_grid=lon_min_target_grid, lon_max_target_grid=lon_max_target_grid, lat_min_target_grid=lat_min_target_grid, lat_max_target_grid=lat_max_target_grid, l0_resolution=l0_resolution, l1_resolution=l1_resolution
                                  , increment_l1=increment_l1)
        # test setup successful
        self.assertTrue(m.domain.l0.lon_min - -10 < 1e-6)
        self.assertTrue(m.domain.l0.lon_max  -  10  < 1e-6)
        self.assertTrue(m.domain.l0.lat_max  -  10 < 1e-6)
        self.assertTrue(m.domain.l0.lat_min  -  -10 < 1e-6)
        self.assertTrue(m.domain.l0.resolution  -  0.002 < 1e-6)
        self.assertTrue(m.domain.l1.lon_min  -  -10 < 1e-6)
        self.assertTrue(m.domain.l1.lon_max  -  10 < 1e-6)
        self.assertTrue(m.domain.l1.lat_max  -  10 < 1e-6)
        self.assertTrue(m.domain.l1.lat_min  -  -10 < 1e-6)
        self.assertTrue(m.domain.l1.resolution  -  0.1 < 1e-6)
        self.assertTrue(m.domain.morph_files.geology == morph / "geology_0.002.nc")

        print(m.domain.morph_files.get_files_as_list(), flush=True)
        # split domain
        m._split_domain()
        self.assertEqual(len(m.subdomains), 100)
        
        for sd in m.subdomains:
            # print(sd.morph_files.geology, flush=True)
            with xr.open_dataset(sd.morph_files.geology) as ds:
                self.assertEqual(ds['longitude'].min(), sd.l0.lon_min)
                self.assertEqual(ds['longitude'].max(), sd.l0.lon_max)
                self.assertEqual(ds['latitude'].min(), sd.l0.lat_min)
                self.assertEqual(ds['latitude'].max(), sd.l0.lat_max)
                        

    def test_write_namelists(self):
        pass
    

    # def test_latlon_from_file(self):
    #     ll = mt.pre.LatLon()
    #     self.assertTrue(not ll.is_fully_defined()


if __name__ == "__main__":
    unittest.main()
