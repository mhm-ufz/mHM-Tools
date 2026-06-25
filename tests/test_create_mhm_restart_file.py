from pathlib import Path

import numpy as np
import pytest
import xarray as xr

import mhm_tools as mt
from mhm_tools.common.file_handler import get_xarray_ds_from_file
from mhm_tools.common.logger import configure_mhm_tools_logger
from mhm_tools.pre.create_mhm_restart_file import Grid, LatLon, MPRRunner

HERE = Path(__file__).parent


class TestCreateRestart:
    @pytest.fixture(autouse=True)
    def _tmp_dirs(self, tmp_path):
        # setup read_morph_files test
        configure_mhm_tools_logger("ERROR")
        self.tmp_path = tmp_path
        self.tmp_work = tmp_path / "work"
        self.tmp_out = tmp_path / "out"
        self.tmp_work.mkdir(parents=True, exist_ok=True)
        self.tmp_out.mkdir(parents=True, exist_ok=True)
        self.morph_dir = tmp_path / "morph"
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

    def _write_split_fixture(
        self,
        lon_min,
        lon_max,
        lat_min,
        lat_max,
        l0_resolution=0.02,
        l1_resolution=0.1,
    ):
        morph = self.tmp_path / "split_morph"
        morph.mkdir(parents=True, exist_ok=True)

        lon_l0 = _cell_centers(lon_min, lon_max, l0_resolution)
        lat_l0 = _cell_centers(lat_min, lat_max, l0_resolution)[::-1]
        geology = xr.Dataset(
            data_vars={
                "geology": (
                    ("latitude", "longitude"),
                    np.ones((lat_l0.size, lon_l0.size), dtype=np.float32),
                )
            },
            coords={"latitude": lat_l0, "longitude": lon_l0},
        )
        geology.to_netcdf(morph / f"geology_{l0_resolution:g}.nc", engine="netcdf4")

        lon_l1 = _cell_centers(lon_min, lon_max, l1_resolution)
        lat_l1 = _cell_centers(lat_min, lat_max, l1_resolution)[::-1]
        land_mask = xr.Dataset(
            data_vars={
                "land_mask": (
                    ("lat", "lon"),
                    np.ones((lat_l1.size, lon_l1.size), dtype=np.float32),
                )
            },
            coords={"lat": lat_l1, "lon": lon_l1},
        )
        land_mask.to_netcdf(morph / "land_mask.nc", engine="netcdf4")
        return morph

    def test_read_morph_files(self):
        mf = mt.pre.MorphFiles(self.morph_dir)
        assert mf.aspect == self.morph_dir / "aspect.nc"
        assert mf.slope == self.morph_dir / "slope.nc"
        assert mf.geology == self.morph_dir / "geology.nc"
        assert mf.clay_content == self.morph_dir / "CLYPPT.nc"
        assert mf.sand_content == self.morph_dir / "sand_content.nc"
        assert mf.bulk_density == self.morph_dir / "BLDFIE.nc"
        assert set(mf.land_cover) == {
            self.morph_dir / "land_cover_1990.nc",
            self.morph_dir / "land_cover_2020.nc",
        }
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
        lon_min_target_grid = -0.2
        lon_max_target_grid = 0.2
        lat_min_target_grid = -0.2
        lat_max_target_grid = 0.2
        l0_resolution = 0.02
        l1_resolution = 0.1
        increment_l1 = 2
        morph = self._write_split_fixture(
            lon_min_target_grid,
            lon_max_target_grid,
            lat_min_target_grid,
            lat_max_target_grid,
            l0_resolution=l0_resolution,
            l1_resolution=l1_resolution,
        )
        land_mask_file = morph / "land_mask.nc"
        mpr_runner = MPRRunner(
            "path_to_mpr_exe"
        )  # dummy path because it is not used in the test
        l0 = LatLon(
            lon_min=lon_min_target_grid,
            lon_max=lon_max_target_grid,
            lat_min=lat_min_target_grid,
            lat_max=lat_max_target_grid,
            resolution=l0_resolution,
        )
        l1 = LatLon(
            lon_min=lon_min_target_grid,
            lon_max=lon_max_target_grid,
            lat_min=lat_min_target_grid,
            lat_max=lat_max_target_grid,
            resolution=l1_resolution,
        )
        grid = Grid(
            file_path=morph,
            name="whole grid",
            latlon_file=None,
            l0=l0,
            l1=l1,
            land_mask_file=land_mask_file,
        )
        m = mt.pre.MHMRestartFile(
            grid=grid,
            output_path=self.tmp_out,
            nml_template=morph / "mpr_mhm_template.nml",
            increment_l1=increment_l1,
            mpr=mpr_runner,
            run_on_whole_domain=False,
            use_split_grids=False,
            clean_temp_files=True,
            ncpus=1,
            merge=True,
            merge_only=False,
        )
        # test setup successful
        assert m.grid.l0.lon_min - lon_min_target_grid < 1e-3
        assert m.grid.l0.lon_max - lon_max_target_grid < 1e-3
        assert m.grid.l0.lat_max - lat_max_target_grid < 1e-3
        assert m.grid.l0.lat_min - lat_min_target_grid < 1e-3
        assert m.grid.l0.resolution - l0_resolution < 1e-6
        assert m.grid.l1.lon_min - lon_min_target_grid < 1e-3
        assert m.grid.l1.lon_max - lon_max_target_grid < 1e-3
        assert m.grid.l1.lat_max - lat_max_target_grid < 1e-3
        assert m.grid.l1.lat_min - lat_min_target_grid < 1e-3
        assert m.grid.l1.resolution - 0.1 < 1e-6
        assert m.grid.morph_files.geology == morph / "geology_0.02.nc"

        # split grid
        m._split_grid()
        assert len(m.subgrids) == 4

        for sd in reversed(m.subgrids):
            # print(sd.morph_files.geology, flush=True)
            with get_xarray_ds_from_file(sd.morph_files.geology) as ds:
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
            assert sd.l1.get_n_lon() == 2
            assert sd.l1.get_n_lat() == 2
            assert sd.l0.get_n_lon() == 10
            assert sd.l0.get_n_lat() == 10

    def test_not_perfect_grid(self):
        # use an increment that does not fit the grid
        lon_min_target_grid = -0.2
        lon_max_target_grid = 0.1
        lat_min_target_grid = -0.2
        lat_max_target_grid = 0.1
        l0_resolution = 0.02
        l1_resolution = 0.1
        increment_l1 = 2
        morph = self._write_split_fixture(
            lon_min_target_grid,
            lon_max_target_grid,
            lat_min_target_grid,
            lat_max_target_grid,
            l0_resolution=l0_resolution,
            l1_resolution=l1_resolution,
        )
        land_mask_file = morph / "land_mask.nc"
        mpr_runner = MPRRunner(
            "path_to_mpr_exe"
        )  # dummy path because it is not used in the test
        l0 = LatLon(
            lon_min=lon_min_target_grid,
            lon_max=lon_max_target_grid,
            lat_min=lat_min_target_grid,
            lat_max=lat_max_target_grid,
            resolution=l0_resolution,
        )
        l1 = LatLon(
            lon_min=lon_min_target_grid,
            lon_max=lon_max_target_grid,
            lat_min=lat_min_target_grid,
            lat_max=lat_max_target_grid,
            resolution=l1_resolution,
        )
        grid = Grid(
            file_path=morph,
            name="whole grid",
            latlon_file=None,
            l0=l0,
            l1=l1,
            land_mask_file=land_mask_file,
        )
        m = mt.pre.MHMRestartFile(
            grid=grid,
            output_path=self.tmp_out,
            work_path=self.tmp_work,
            nml_template=morph / "mpr_mhm_template.nml",
            increment_l1=increment_l1,
            mpr=mpr_runner,
            run_on_whole_domain=False,
            use_split_grids=False,
            clean_temp_files=True,
            ncpus=1,
            merge=True,
            merge_only=False,
        )
        # test setup successful
        m._split_grid()
        assert len(m.subgrids) == 4
        for sd in reversed(m.subgrids):
            with get_xarray_ds_from_file(sd.morph_files.geology) as ds:
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

            _, i_str, j_str = sd.name.split("_")
            i = int(i_str)
            j = int(j_str)
            if i == 1 and j != 1:
                assert sd.l1.get_n_lon() == 1
                assert sd.l1.get_n_lat() == 2
                assert sd.l0.get_n_lon() == 5
                assert sd.l0.get_n_lat() == 10
            elif i != 1 and j == 1:
                assert sd.l1.get_n_lon() == 2
                assert sd.l1.get_n_lat() == 1
                assert sd.l0.get_n_lon() == 10
                assert sd.l0.get_n_lat() == 5
            elif i == 1 and j == 1:
                assert sd.l1.get_n_lon() == 1
                assert sd.l1.get_n_lat() == 1
                assert sd.l0.get_n_lon() == 5
                assert sd.l0.get_n_lat() == 5
            else:
                assert sd.l1.get_n_lon() == 2
                assert sd.l1.get_n_lat() == 2
                assert sd.l0.get_n_lon() == 10
                assert sd.l0.get_n_lat() == 10

    def test_write_namelists(self):
        pass


def _cell_centers(lower, upper, resolution):
    n_cells = int(round((upper - lower) / resolution))
    return lower + resolution / 2 + np.arange(n_cells) * resolution
