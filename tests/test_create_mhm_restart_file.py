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
        assert mf.aspect == self.morph_dir / "aspect.nc"
        assert mf.slope == self.morph_dir / "slope.nc"
        assert mf.geology == self.morph_dir / "geology.nc"
        assert mf.clay_content == self.morph_dir / "CLYPPT.nc"
        assert mf.sand_content == self.morph_dir / "sand_content.nc"
        assert mf.bulk_density == self.morph_dir / "BLDFIE.nc"
        assert mf.land_cover == [self.morph_dir / "land_cover_1990.nc", self.morph_dir / "land_cover_2020.nc"]
        assert mf.get_file('lai') == self.morph_dir / "lai.nc"
    
    def test_read_latlon(self):
        p = Path(HERE / "files" / "test_create_restart" / "latlon_0p0625.nc")
        d = mt.pre.Domain(file_path=self.morph_dir, latlon_file=p)
        assert d.l0.lon_min - 11.500977 < 1e-6
        assert d.l0.lon_max  -  12.999023  < 1e-6
        assert d.l0.lat_max  -  50.249023 < 1e-6
        assert d.l0.lat_min  -  49.000977 < 1e-6
        assert d.l0.resolution  -  0.001953125 < 1e-6
        assert d.l1.lon_min  -  11.53125 < 1e-6
        assert d.l1.lon_max  -  12.96875 < 1e-6
        assert d.l1.lat_max  -  50.21875 < 1e-6
        assert d.l1.lat_min  -  49.03125 < 1e-6
        assert d.l1.resolution  -  0.0625 < 1e-6

    
    def test_split_domain(self):
        morph = Path(HERE / "files" / "test_create_restart")

        m = mt.pre.MHMRestartFile(input_file_path=morph, latlon_file=morph / 'latlon_0p0625.nc', output_path=TMP, nml_template=morph / 'mpr_mhm_template.nml')
        # test setup successful
        assert m.domain.l0.lon_min - 11.500977 < 1e-6
        assert m.domain.l0.lon_max  -  12.999023  < 1e-6
        assert m.domain.morph_files.lai == morph / "lai.nc"

        # split domain
        m._split_domain()
        print(len(m.subdomains))

    def test_write_namelists(self):
        pass
    

    # def test_latlon_from_file(self):
    #     ll = mt.pre.LatLon()
    #     assert not ll.is_fully_defined()


if __name__ == "__main__":
    unittest.main()
