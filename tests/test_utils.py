"""Tests for mhm_tools.common.utils.cut_to_filled_area."""

import unittest

import numpy as np
import xarray as xr

from mhm_tools.common.resolution_handler import Resolution
from mhm_tools.common.utils import cut_to_filled_area


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
