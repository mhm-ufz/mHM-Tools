"""Tests for mhm_tools.common.utils.cut_to_filled_area."""

import tempfile
import unittest
from pathlib import Path

import numpy as np
import xarray as xr

from mhm_tools.common.resolution_handler import Resolution
from mhm_tools.common.utils import align_bounds_to_l2, cut_to_filled_area


def _make_ds(n_rows, n_cols):
    """Build a minimal dataset with lon/lat coordinates of the given shape."""
    return xr.Dataset(
        coords={
            "lat": np.arange(n_rows, dtype=float),
            "lon": np.arange(n_cols, dtype=float),
        }
    )


class TestCutToFilledArea(unittest.TestCase):
    def test_single_cell_returns_length_one_slice(self):
        """Regression test: a single filled cell used to collapse to an empty slice."""
        mask = np.zeros((2, 5), dtype=bool)
        mask[0, 3] = True
        ds = _make_ds(2, 5)

        lat_slice, lon_slice = cut_to_filled_area(
            ds=ds, resolutions=Resolution(l0=1.0), catchment_mask=mask
        )

        self.assertEqual(lat_slice, slice(0, 1))
        self.assertEqual(lon_slice, slice(3, 4))

    def test_multi_cell_includes_last_row_and_col(self):
        """The last filled row/column must be included, not silently dropped."""
        mask = np.zeros((10, 10), dtype=bool)
        mask[2:6, 3:7] = True  # rows 2-5, cols 3-6 inclusive
        ds = _make_ds(10, 10)

        lat_slice, lon_slice = cut_to_filled_area(
            ds=ds, resolutions=Resolution(l0=1.0), catchment_mask=mask
        )

        self.assertEqual(lat_slice, slice(2, 6))
        self.assertEqual(lon_slice, slice(3, 7))

    def test_filled_area_at_array_edge(self):
        """A filled area touching the array's last row/column must be kept."""
        mask = np.zeros((10, 10), dtype=bool)
        mask[8:10, 8:10] = True
        ds = _make_ds(10, 10)

        lat_slice, lon_slice = cut_to_filled_area(
            ds=ds, resolutions=Resolution(l0=1.0), catchment_mask=mask
        )

        self.assertEqual(lat_slice, slice(8, 10))
        self.assertEqual(lon_slice, slice(8, 10))

    def test_buffer_expands_and_clips_to_bounds(self):
        """The buffer must expand the window and clip at both array bounds."""
        mask = np.zeros((10, 10), dtype=bool)
        mask[1, 8] = True
        ds = _make_ds(10, 10)

        lat_slice, lon_slice = cut_to_filled_area(
            ds=ds,
            resolutions=Resolution(l0=1.0),
            catchment_mask=mask,
            buffer=3,
        )

        # Row buffer would reach -2:5; clipped at the lower bound to 0:5.
        self.assertEqual(lat_slice, slice(0, 5))
        # Column buffer would reach 5:12; clipped at the upper bound to 5:10.
        self.assertEqual(lon_slice, slice(5, 10))

    def test_upscaling_rounds_up_without_dropping_edge_cell(self):
        """Ceiling-rounded upscaling bounds must not be clamped below the edge cell."""
        mask = np.zeros((9, 9), dtype=bool)
        mask[8, 8] = True  # last row and column, under a factor-3 upscale
        ds = _make_ds(9, 9)

        lat_slice, lon_slice = cut_to_filled_area(
            ds=ds,
            resolutions=Resolution(l0=1.0, l2=3.0),
            catchment_mask=mask,
        )

        self.assertEqual(lat_slice, slice(6, 9))
        self.assertEqual(lon_slice, slice(6, 9))

    def test_raises_for_none_mask(self):
        ds = _make_ds(2, 2)
        with self.assertRaises(ValueError):
            cut_to_filled_area(
                ds=ds, resolutions=Resolution(l0=1.0), catchment_mask=None
            )

    def test_raises_for_empty_mask(self):
        ds = _make_ds(2, 2)
        mask = np.zeros((2, 2), dtype=bool)
        with self.assertRaises(ValueError):
            cut_to_filled_area(
                ds=ds, resolutions=Resolution(l0=1.0), catchment_mask=mask
            )


L0_RES = 1 / 600
L2_RES = 0.1
FACTOR = 60
# the fdir window is cut to the shapefile bounds, so its corner sits off the
# 0.1 degree meteo grid; every index below is counted from these edges
L0_LAT_TOP_EDGE = 47.575
L0_LON_LEFT_EDGE = 9.595


def _make_l0_ds(n_rows, n_cols):
    """Build an L0 dataset whose grid corner is off the L2 grid."""
    lat = L0_LAT_TOP_EDGE - (np.arange(n_rows) + 0.5) * L0_RES
    lon = L0_LON_LEFT_EDGE + (np.arange(n_cols) + 0.5) * L0_RES
    return xr.Dataset(coords={"lat": lat, "lon": lon})


def _write_l2_file(path):
    """Write a meteo file on the standard 0.1 degree grid."""
    lat = np.arange(47.55, 47.14, -L2_RES)
    lon = np.arange(9.65, 10.06, L2_RES)
    ds = xr.Dataset(
        data_vars={"pre": (("lat", "lon"), np.ones((lat.size, lon.size)))},
        coords={"lat": lat, "lon": lon},
    )
    ds.to_netcdf(path)
    return path


class TestAlignBoundsToL2(unittest.TestCase):
    """Bounds derived from an L0 grid that is offset from the meteo grid."""

    @classmethod
    def setUpClass(cls):
        cls._tmp_dir = tempfile.TemporaryDirectory()
        cls.l2_file = _write_l2_file(Path(cls._tmp_dir.name) / "pre.nc")

    @classmethod
    def tearDownClass(cls):
        cls._tmp_dir.cleanup()

    def _resolutions(self):
        return Resolution(
            l0=L0_RES, l1=L2_RES, l11=L2_RES, l2=L2_RES, l2_file=self.l2_file
        )

    def _catchment_mask(self, n_rows, n_cols):
        # a catchment spanning lat 47.35..47.45 and lon 9.65..9.85, so the
        # aligned domain has to be lat 47.3..47.5 and lon 9.6..9.9
        mask = np.zeros((n_rows, n_cols), dtype=bool)
        mask[75:135, 33:153] = True
        return mask

    def test_aligned_window_is_divisible_by_the_upscaling_factor(self):
        """The aligned L0 window must hold whole L2 cells."""
        ds = _make_l0_ds(240, 240)

        min_row, max_row, min_col, max_col = align_bounds_to_l2(
            ds, self._resolutions(), 75, 134, 33, 152
        )

        self.assertEqual((max_row + 1 - min_row) % FACTOR, 0)
        self.assertEqual((max_col + 1 - min_col) % FACTOR, 0)

    def test_cropped_window_edges_land_on_the_l2_grid(self):
        """The cropped domain must start and end on 0.1 degree lines."""
        n_rows, n_cols = 240, 240
        ds = _make_l0_ds(n_rows, n_cols)
        mask = self._catchment_mask(n_rows, n_cols)

        lat_slice, lon_slice = cut_to_filled_area(
            ds=ds, resolutions=self._resolutions(), catchment_mask=mask
        )

        lat_top = L0_LAT_TOP_EDGE - lat_slice.start * L0_RES
        lat_bottom = L0_LAT_TOP_EDGE - lat_slice.stop * L0_RES
        lon_left = L0_LON_LEFT_EDGE + lon_slice.start * L0_RES
        lon_right = L0_LON_LEFT_EDGE + lon_slice.stop * L0_RES

        for name, edge in [
            ("lat_top", lat_top),
            ("lat_bottom", lat_bottom),
            ("lon_left", lon_left),
            ("lon_right", lon_right),
        ]:
            with self.subTest(edge=name):
                self.assertAlmostEqual(
                    edge / L2_RES,
                    round(edge / L2_RES),
                    places=6,
                    msg=f"{name}={edge} is not on the {L2_RES} grid",
                )
