"""Tests for the shared helpers behind the categorical format commands."""

import logging

import numpy as np
import pandas as pd
import pytest
import rioxarray  # noqa: F401
import xarray as xr

from mhm_tools.common.categorical_raster import (
    _reclassify,
    fill_grid_nodata,
    prepare_categorical_data,
)
from mhm_tools.pre.format_lc_data import _lc_resampling

GLOBCOVER_MAPPING = {11.0: 3, 40.0: 1, 190.0: 2, 210.0: 3}


def create_category_raster(values, nodata):
    """Create a category raster that declares its nodata value.

    Args:
        values (numpy.ndarray): Two-dimensional raster categories.
        nodata (float): Value stored as the raster's ``_FillValue``.

    Returns:
        xarray.DataArray: The categories on ``y``/``x`` dimensions.
    """
    return xr.DataArray(
        values, dims=("y", "x"), name="categories", attrs={"_FillValue": nodata}
    )


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


def test_reclassify_maps_uint8_categories_and_nodata():
    """Map every listed uint8 category and write the raster nodata as -9999."""
    values = np.array([[11, 40, 190], [210, 0, 40]], dtype=np.uint8)

    classes = _reclassify(create_category_raster(values, 0), GLOBCOVER_MAPPING)

    assert classes.dtype == np.int32
    np.testing.assert_array_equal(classes, [[3, 1, 2], [3, -9999, 1]])


def test_reclassify_maps_negative_int16_categories():
    """Map signed int16 categories, including negative keys and nodata."""
    values = np.array([[-5, 7, -1], [300, -5, 7]], dtype=np.int16)

    classes = _reclassify(
        create_category_raster(values, -1), {-5.0: 1, 7.0: 2, 300.0: 3, -1.0: 3}
    )

    np.testing.assert_array_equal(classes, [[1, 2, -9999], [3, 1, 2]])


def test_reclassify_logs_unmatched_uint8_categories(caplog, monkeypatch):
    """Write unlisted categories as -9999 and log their cell count and values."""
    values = np.array([[11, 230, 230], [14, 40, 0]], dtype=np.uint8)
    # CLI tests configure the shared logger with propagate=False, hiding it from
    # caplog; monkeypatch restores the setting after this test.
    monkeypatch.setattr(logging.getLogger("mhm_tools"), "propagate", True)

    with caplog.at_level(logging.WARNING, logger="mhm_tools.common.categorical_raster"):
        classes = _reclassify(create_category_raster(values, 0), GLOBCOVER_MAPPING)

    np.testing.assert_array_equal(classes, [[3, -9999, -9999], [-9999, 1, -9999]])
    assert (
        "3 valid raster cells in 2 categories were not present in the lookup "
        "table and were written as -9999: 14, 230"
    ) in caplog.text


def test_reclassify_rejects_uint8_raster_without_a_matching_category():
    """Refuse a uint8 raster none of whose valid categories is listed."""
    values = np.array([[14, 230], [0, 14]], dtype=np.uint8)

    with pytest.raises(ValueError, match="No valid raster category matched"):
        _reclassify(create_category_raster(values, 0), GLOBCOVER_MAPPING)


def test_reclassify_maps_float_categories_with_nan():
    """Map float categories and write NaN and the nodata value as -9999."""
    values = np.array([[11.0, np.nan], [-1.0, 190.0]], dtype=np.float32)

    classes = _reclassify(create_category_raster(values, -1.0), GLOBCOVER_MAPPING)

    np.testing.assert_array_equal(classes, [[3, -9999], [-9999, 2]])


def test_reclassify_uint8_lookup_array_matches_the_per_category_loop():
    """Give the same classes for uint8 input as the loop gives for int32 input."""
    values = np.random.default_rng(42).integers(0, 256, (64, 80), dtype=np.uint8)
    mapping = {float(category): int(category) % 3 + 1 for category in range(1, 250)}

    lookup_classes = _reclassify(create_category_raster(values, 0), mapping)
    loop_classes = _reclassify(
        create_category_raster(values.astype(np.int32), 0), mapping
    )

    np.testing.assert_array_equal(lookup_classes, loop_classes)


IDENTITY_LOOKUP = pd.DataFrame({"code": range(1, 10), "class": range(1, 10)})


def create_utm_grid(values, x_origin=1000.0):
    """Create a north-up EPSG:32632 grid of 100 m cells.

    Args:
        values (numpy.ndarray): Two-dimensional cell values, row 0 northernmost.
        x_origin (float): Western edge of the grid in metres.

    Returns:
        xarray.DataArray: The values on ``y``/``x`` dimensions with a CRS.
    """
    rows, cols = values.shape
    return xr.DataArray(
        values,
        dims=("y", "x"),
        coords={
            "y": 2000.0 - 100.0 * (np.arange(rows) + 0.5),
            "x": x_origin + 100.0 * (np.arange(cols) + 0.5),
        },
    ).rio.write_crs("EPSG:32632")


def write_class_raster(path, x_origin=1000.0):
    """Write a 3x3 GeoTIFF of the classes 1..9 and return its path."""
    classes = np.arange(1, 10, dtype=np.int32).reshape(3, 3)
    create_utm_grid(classes, x_origin).rio.write_nodata(0).rio.to_raster(path)
    return path


def prepare_identity_classes(input_file, reference, resampling="nearest"):
    """Run prepare_categorical_data with the identity lookup."""
    return prepare_categorical_data(
        input_file,
        reference,
        IDENTITY_LOOKUP,
        "code",
        "class",
        variable_name="classes",
        resampling=resampling,
    )


def test_prepare_categorical_data_keeps_orientation_of_x_y_ordered_reference(
    tmp_path,
):
    """A reference stored as (x, y) gives the same classes as one stored as (y, x)."""
    input_file = write_class_raster(tmp_path / "classes.tif")
    reference = create_utm_grid(np.ones((3, 3)))

    expected = prepare_identity_classes(input_file, reference)["classes"]
    output = prepare_identity_classes(input_file, reference.transpose("x", "y"))

    np.testing.assert_array_equal(
        output["classes"].transpose("y", "x").values, expected.values
    )


def test_class_rasters_reject_resampling_that_invents_classes(tmp_path):
    """Bilinear resampling would blend the classes 1 and 9 into new ones."""
    input_file = write_class_raster(tmp_path / "classes.tif")
    reference = create_utm_grid(np.ones((3, 3)))

    with pytest.raises(ValueError, match="invent classes"):
        prepare_identity_classes(input_file, reference, resampling="bilinear")
    with pytest.raises(ValueError, match="auto, nearest or mode"):
        _lc_resampling("bilinear")


def test_prepare_categorical_data_rejects_input_outside_the_reference(tmp_path):
    """Classes that miss the reference domain, e.g. after a wrong CRS, raise."""
    input_file = write_class_raster(tmp_path / "classes.tif", x_origin=500000.0)
    reference = create_utm_grid(np.ones((3, 3)))

    with pytest.raises(ValueError, match="No class"):
        prepare_identity_classes(input_file, reference)
