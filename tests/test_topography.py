"""Tests for the slope and aspect of an elevation grid."""

import numpy as np
import pytest
from rasterio.transform import from_origin

from mhm_tools.common.constants import NO_DATA
from mhm_tools.common.topography import calculate_aspect, calculate_slope


def _create_plane(rows, cols, east_rise, north_rise):
    """Create a north-up planar elevation grid of cells 1 m wide.

    Args:
        rows (int): Number of rows.
        cols (int): Number of columns.
        east_rise (float): Rise per cell towards the east.
        north_rise (float): Rise per cell towards the north.

    Returns
    -------
        numpy.ndarray: Elevation, row 0 is the northernmost.
    """
    east = np.arange(cols, dtype="float64")
    north = np.arange(rows, dtype="float64")[::-1]
    return 100.0 + east_rise * east[None, :] + north_rise * north[:, None]


@pytest.mark.parametrize(
    ("east_rise", "north_rise", "expected"),
    [
        (-1.0, 0.0, 90.0),  # ground falls towards the east
        (0.0, 1.0, 180.0),
        (1.0, 0.0, 270.0),
        (0.0, -1.0, 0.0),
        (-1.0, -1.0, 45.0),
    ],
)
def test_aspect_is_the_compass_bearing_of_the_steepest_descent(
    east_rise, north_rise, expected
):
    """Give the bearing of the steepest descent, 0 for north, clockwise."""
    plane = _create_plane(5, 5, east_rise, north_rise)
    aspect = calculate_aspect(plane, from_origin(0.0, 5.0, 1.0, 1.0))
    assert aspect[2, 2] == pytest.approx(expected)


def test_aspect_of_flat_ground_is_90_by_default():
    """Give flat cells 90 degrees, neutral in the PET aspect correction of mHM."""
    aspect = calculate_aspect(np.full((3, 3), 12.0), from_origin(0.0, 3.0, 1.0, 1.0))
    np.testing.assert_array_equal(aspect, 90.0)


@pytest.mark.parametrize("missing", [NO_DATA, np.nan])
def test_aspect_next_to_missing_cells_stays_valid(missing):
    """Use the centre value for a missing neighbour and keep missing cells missing."""
    plane = _create_plane(5, 5, -1.0, 0.0)
    plane[2, 3] = missing
    aspect = calculate_aspect(plane, from_origin(0.0, 5.0, 1.0, 1.0))
    assert aspect[2, 2] == pytest.approx(90.0)
    assert aspect[2, 3] == NO_DATA


def test_aspect_on_a_geographic_grid_is_measured_in_metres():
    """Correct the bearing for the shorter degrees of longitude at 60 degrees north."""
    from pyflwdir.gis_utils import degree_metres_x, degree_metres_y

    cellsize = 0.001
    # the centre row of five lies at 60 degrees north
    transform = from_origin(10.0, 60.0 + 2.5 * cellsize, cellsize, cellsize)
    # falls 1 m per metre towards the east and towards the north
    plane = _create_plane(
        5,
        5,
        -cellsize * degree_metres_x(60.0),
        -cellsize * degree_metres_y(60.0),
    )
    assert calculate_aspect(plane, transform, latlon=True)[2, 2] == pytest.approx(45.0)
    # per degree instead of per metre the bearing is pulled towards the north
    assert calculate_aspect(plane, transform)[2, 2] == pytest.approx(26.6, abs=0.1)


def test_slope_is_the_pyflwdir_slope_in_percent():
    """Give the pyflwdir slope in percent and keep missing cells missing."""
    import pyflwdir

    transform = from_origin(0.0, 500.0, 100.0, 100.0)
    plane = _create_plane(5, 5, 10.0, 0.0)
    plane[0, 0] = NO_DATA
    slope = calculate_slope(plane, transform)

    expected = pyflwdir.dem.slope(
        plane, nodata=NO_DATA, transform=np.asarray(tuple(transform), dtype="float64")
    )
    np.testing.assert_allclose(slope[1:, 1:], 100.0 * expected[1:, 1:])
    # 10 m rise per 100 m cell is 10 percent
    assert slope[2, 2] == pytest.approx(10.0)
    assert slope[0, 0] == NO_DATA
