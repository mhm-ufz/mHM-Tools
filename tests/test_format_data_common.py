"""Tests for the shared helpers behind the format-data workflows."""

import numpy as np
import pytest

from mhm_tools.common.format_data import fill_grid_nodata


def test_fill_grid_nodata_fills_in_place_outside_the_mask():
    """A raster block is filled in place and masked cells keep the fill value."""
    values = np.array(
        [[1.0, 2.0, np.nan], [4.0, 5.0, np.nan]],
        dtype=np.float64,
    )
    mask = np.array([[False, False, False], [False, False, True]])

    filled = fill_grid_nodata(
        values,
        x=np.array([0.0, 1.0, 2.0]),
        y=np.array([1.0, 0.0]),
        mask=mask,
        fill_value=-9999.0,
        name="clay",
    )

    assert filled == 1
    np.testing.assert_array_equal(
        values,
        [[1.0, 2.0, 2.0], [4.0, 5.0, -9999.0]],
    )


def test_fill_grid_nodata_uses_a_marker_on_integer_classes():
    """Integer class rasters are filled through their missing-value marker."""
    values = np.array([[1, 2, -9999, -9999]], dtype=np.int32)

    filled = fill_grid_nodata(
        values,
        missing_value=-9999.0,
        fill_value=-9999,
        name="land_cover",
    )

    assert filled == 2
    np.testing.assert_array_equal(values, [[1, 2, 2, 2]])


def test_fill_grid_nodata_uses_coordinates_instead_of_cell_indices():
    """Non-square cells select the nearest neighbour in map units."""
    values = np.array([[1.0, np.nan], [3.0, 4.0]], dtype=np.float64)

    filled = fill_grid_nodata(
        values,
        # Rows are ten times further apart than columns.
        x=np.array([0.0, 1.0]),
        y=np.array([10.0, 0.0]),
    )

    assert filled == 1
    np.testing.assert_array_equal(values, [[1.0, 1.0], [3.0, 4.0]])


def test_fill_grid_nodata_rejects_non_two_dimensional_input():
    """Only two-dimensional raster blocks are accepted."""
    with pytest.raises(ValueError, match="two-dimensional"):
        fill_grid_nodata(np.zeros((2, 2, 2)))
