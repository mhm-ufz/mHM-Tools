"""Calculate slope and aspect of an elevation grid.

Both use Horn's 3x3 gradient. A neighbour that is missing, because it is nodata
or lies beyond the grid edge, takes the value of the centre cell, as in the
slope of :mod:`pyflwdir` and of the merit-masker tools. On geographic grids the
gradient is converted to metres with the degree lengths pyflwdir uses, so slope
and aspect describe the same surface.

Authors
-------
- Simon Lüdke
"""

import numpy as np

from mhm_tools.common.constants import NO_DATA


def calculate_slope(elevation, transform, latlon=False, nodata=NO_DATA):
    """Calculate the slope of an elevation grid with Horn's 3x3 gradient.

    Args:
        elevation (numpy.ndarray): North-up elevation; NaN and ``nodata`` mark
            missing cells.
        transform (affine.Affine): Transform of the elevation grid.
        latlon (bool): Whether the grid is geographic, so the cell size in
            degrees is converted to metres per row.
        nodata (float): Value of missing cells in the input and the output.

    Returns
    -------
        numpy.ndarray: Slope in percent, ``nodata`` where the elevation is missing.
    """
    import pyflwdir

    values = np.where(np.isfinite(elevation), elevation, nodata).astype("float64")
    # pyflwdir reads the transform as a plain array (affine 3 is no tuple anymore)
    transform_array = np.asarray(tuple(transform), dtype=np.float64)
    slope = pyflwdir.dem.slope(
        values, nodata=nodata, latlon=latlon, transform=transform_array
    )
    slope *= 100.0
    slope = slope.astype("float64")
    slope[values == nodata] = nodata
    return slope


def calculate_aspect(
    elevation, transform, latlon=False, nodata=NO_DATA, missing_aspect=90.0
):
    """Calculate the aspect of an elevation grid with Horn's 3x3 gradient.

    The aspect is the compass bearing of the steepest descent, 0 for north,
    increasing clockwise. Flat cells have no aspect and get ``missing_aspect``.

    Args:
        elevation (numpy.ndarray): North-up elevation; NaN and ``nodata`` mark
            missing cells.
        transform (affine.Affine): Transform of the elevation grid.
        latlon (bool): Whether the grid is geographic, so the cell size in
            degrees is converted to metres per row.
        nodata (float): Value of missing cells in the input and the output.
        missing_aspect (float): Aspect of flat cells. 90 gives them the factor
            halfway between north- and south-facing cells in the PET aspect
            correction of mHM at its default threshold of 180. NaN or
            ``nodata`` leaves them missing.

    Returns
    -------
        numpy.ndarray: Aspect in degrees in [0, 360), ``nodata`` where the
        elevation is missing and on flat cells left missing.
    """
    values = np.where(np.isfinite(elevation), elevation, nodata).astype("float64")
    invalid = values == nodata
    centre = np.where(invalid, np.nan, values)
    padded = np.pad(centre, 1, constant_values=np.nan)
    rows, cols = centre.shape
    # sums of the east/west and north/south neighbours, kept apart so that
    # equal neighbours cancel exactly and flat cells are detected as flat
    east_west = np.zeros((2, rows, cols))
    north_south = np.zeros((2, rows, cols))
    for row_offset in (-1, 0, 1):
        for col_offset in (-1, 0, 1):
            window = padded[
                1 + row_offset : 1 + row_offset + rows,
                1 + col_offset : 1 + col_offset + cols,
            ]
            neighbour = np.where(np.isnan(window), centre, window)
            # Horn weights the side neighbours twice as heavily as the corners
            if col_offset:
                east_west[int(col_offset < 0)] += (2 - abs(row_offset)) * neighbour
            if row_offset:
                north_south[int(row_offset > 0)] += (2 - abs(col_offset)) * neighbour
    del padded
    gradient_east = east_west[0] - east_west[1]
    gradient_north = north_south[0] - north_south[1]
    if latlon:
        from pyflwdir.gis_utils import degree_metres_x, degree_metres_y

        row_latitude = transform.f + (np.arange(rows) + 0.5) * transform.e
        metres_east = abs(transform.a) * degree_metres_x(row_latitude)[:, None]
        metres_north = abs(transform.e) * degree_metres_y(row_latitude)[:, None]
    else:
        metres_east = abs(transform.a)
        metres_north = abs(transform.e)
    gradient_east /= 8.0 * metres_east
    gradient_north /= 8.0 * metres_north
    flat = (gradient_east == 0) & (gradient_north == 0)
    aspect = np.degrees(np.arctan2(-gradient_east, -gradient_north)) % 360.0
    aspect[flat] = nodata if np.isnan(missing_aspect) else missing_aspect
    aspect[invalid] = nodata
    return aspect
