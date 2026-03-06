from pathlib import Path

import pytest

import mhm_tools as mt
from mhm_tools.common.file_handler import get_xarray_ds_from_file

HERE = Path(__file__).parent


class TestLatLon:
    @pytest.fixture(autouse=True)
    def _tmp_paths(self, tmp_path):
        self.latlon_file = tmp_path / "latlon.nc"
        self.header_l0 = tmp_path / "header_l0.asc"
        self.header_l1 = tmp_path / "header_l1.asc"
        self.header_l11 = tmp_path / "header_l11.asc"
        self.header_l2 = tmp_path / "header_l2.asc"
        self.l0_header = {
            "ncols": 288,
            "nrows": 432,
            "xllcorner": 3973369,
            "yllcorner": 2735847,
            "cellsize": 500,
            "nodata_value": -9999,
        }

    def test_latlon(self):
        mt.pre.create_latlon(
            out_file=self.latlon_file,
            level0=self.l0_header,
            level1=12000,
            level11=6000,
            level2=24000,
            crs="epsg:3035",
            add_bounds=True,
            write_header_l0=self.header_l0,
            write_header_l1=self.header_l1,
            write_header_l11=self.header_l11,
            write_header_l2=self.header_l2,
        )
        header_l0 = mt.common.read_header(self.header_l0)
        header_l1 = mt.common.read_header(self.header_l1)
        header_l11 = mt.common.read_header(self.header_l11)
        header_l2 = mt.common.read_header(self.header_l2)

        assert header_l0["nrows"] == 432
        assert header_l0["ncols"] == 288
        assert header_l1["nrows"] == 18
        assert header_l1["ncols"] == 12
        assert header_l11["nrows"] == 36
        assert header_l11["ncols"] == 24
        assert header_l2["nrows"] == 9
        assert header_l2["ncols"] == 6

        ds = get_xarray_ds_from_file(self.latlon_file)
        assert len(ds["yc_l0"]) == 432
        assert len(ds["xc_l0"]) == 288
        assert len(ds["yc_l1"]) == 18
        assert len(ds["xc_l1"]) == 12
        assert len(ds["yc_l11"]) == 36
        assert len(ds["xc_l11"]) == 24
        ds.close()
