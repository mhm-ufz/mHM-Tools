import logging

import numpy as np
import pytest
import xarray as xr

from mhm_tools.common.resolution_handler import (
    Resolution,
    calculate_coordinate_resolution,
    get_file_res,
)

RESOLUTIONS = [0.5, 1 / 120, 1 / 240]


@pytest.fixture(autouse=True)
def _propagate_package_logs():
    """Let pytest's caplog capture the package logs of every test in this module.

    Other test modules configure the package logger globally, which can disable
    propagation or raise its level, so it is restored around each test.
    """
    package_logger = logging.getLogger("mhm_tools")
    propagate, level = package_logger.propagate, package_logger.level
    package_logger.propagate = True
    package_logger.setLevel(logging.WARNING)
    yield
    package_logger.propagate = propagate
    package_logger.setLevel(level)


def make_global_lon(res, dtype=float, decimals=None):
    """Build a global longitude axis with the given resolution.

    Args:
        res (float): Coordinate spacing in degree.
        dtype: Numpy dtype to store the values in.
        decimals (int): Number of decimals to round the values to.

    Returns
    -------
        The longitude values as numpy array.
    """
    lon = -180 + res * np.arange(int(360 / res))
    if decimals is not None:
        lon = np.round(lon, decimals)
    return lon.astype(dtype)


def has_warning(caplog, text="not evenly spaced"):
    """Return whether a warning containing the given text was logged.

    Args:
        caplog: Pytest caplog fixture holding the captured records.
        text (str): Text the warning message has to contain.

    Returns
    -------
        True if a matching warning was logged.
    """
    return any(
        record.levelno == logging.WARNING and text in record.message
        for record in caplog.records
    )


def make_ds(lon, lat, resolution_attr=None, as_dataarray=False):
    """Build a mask dataset from the given coordinates.

    Args:
        lon: Longitude values.
        lat: Latitude values.
        resolution_attr (float): Value for the "spatial_resolution" attribute.
        as_dataarray (bool): Return the mask DataArray instead of the Dataset.

    Returns
    -------
        The dataset or its mask DataArray.
    """
    attrs = {} if resolution_attr is None else {"spatial_resolution": resolution_attr}
    ds = xr.Dataset(
        data_vars={"mask": (("lat", "lon"), np.ones((lat.size, lon.size)))},
        coords={"lon": lon, "lat": lat},
        attrs=attrs,
    )
    if as_dataarray:
        return ds["mask"].assign_attrs(attrs)
    return ds


@pytest.mark.parametrize("res", RESOLUTIONS)
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("descending", [False, True])
def test_calculate_coordinate_resolution_regular_grids(res, dtype, descending, caplog):
    # the average step averages out the float32 noise of the single steps
    lon = make_global_lon(res, dtype=dtype)
    if descending:
        lon = lon[::-1].copy()

    with caplog.at_level(logging.WARNING):
        result = calculate_coordinate_resolution(lon)

    assert result == pytest.approx(res, abs=1e-9)
    assert not has_warning(caplog)


@pytest.mark.parametrize("res", RESOLUTIONS)
@pytest.mark.parametrize("decimals", [6, 5, 4, 3])
def test_calculate_coordinate_resolution_rounded_coordinates(res, decimals, caplog):
    # rounded coordinates are noisy but not defective, so no warning is expected
    lon = make_global_lon(res, decimals=decimals)

    with caplog.at_level(logging.WARNING):
        result = calculate_coordinate_resolution(lon)

    assert result == pytest.approx(res, abs=1e-5)
    assert not has_warning(caplog)


def make_dateline_wrapped_lon():
    """Return a 0.5 degree longitude axis that wraps at the dateline."""
    lon = np.arange(0, 360, 0.5)
    lon[lon >= 180] -= 360
    return lon


def make_lon_with_nan():
    """Return a 0.25 degree longitude axis with a single NaN value."""
    lon = np.arange(0, 50, 0.25)
    lon[17] = np.nan
    return lon


@pytest.mark.parametrize(
    "lon",
    [
        make_dateline_wrapped_lon(),
        np.delete(np.arange(0, 100, 0.5), 40),
        np.delete(np.arange(0, 100, 0.5), [40, 90]),
        np.array([0.0, 0.5, 0.0, 0.5]),
        make_lon_with_nan(),
    ],
    ids=[
        "dateline_wrap",
        "one_missing_cell",
        "two_missing_cells",
        "non_monotonic",
        "nan_value",
    ],
)
def test_calculate_coordinate_resolution_warns_on_irregular_coordinates(lon, caplog):
    # the average step is unusable here, the median step still recovers 0.5/0.25
    expected = 0.25 if np.isnan(lon).any() else 0.5

    with caplog.at_level(logging.WARNING):
        result = calculate_coordinate_resolution(lon)

    assert result == pytest.approx(expected, abs=1e-9)
    assert has_warning(caplog)


def test_calculate_coordinate_resolution_two_point_coordinate(caplog):
    with caplog.at_level(logging.WARNING):
        result = calculate_coordinate_resolution(np.array([10.0, 10.5]))

    assert result == pytest.approx(0.5)
    assert not has_warning(caplog)


@pytest.mark.parametrize(
    "coord",
    [np.array([1.0]), np.zeros((3, 4))],
    ids=["single_cell", "two_dimensional"],
)
def test_calculate_coordinate_resolution_raises_for_invalid_shape(coord):
    with pytest.raises(ValueError, match="Cannot determine the resolution"):
        calculate_coordinate_resolution(coord)


def test_get_file_res_uses_median_for_irregular_coordinates():
    # a single missing cell would pull the average step to 0.5025
    lon = xr.DataArray(np.delete(np.arange(0, 100, 0.5), 40), dims="lon")

    assert get_file_res(lon=lon) == pytest.approx(0.5, abs=1e-9)


def test_get_file_res_raises_value_error_for_single_point():
    # _coord_bound_resolution relies on this ValueError to fall back to bounds
    lon = xr.DataArray(np.array([5.0]), dims="lon")

    with pytest.raises(ValueError, match="Cannot determine file resolution"):
        get_file_res(lon=lon, lat=None)


def test_get_file_res_snaps_float32_fine_grid_to_provided_resolution():
    # single float32 steps of a 1/120 deg axis exceed the 1e-5 snap tolerance
    res = 1 / 120
    lon = xr.DataArray(make_global_lon(res, dtype=np.float32), dims="lon")

    result = get_file_res(lon=lon, resolutions=Resolution(l0=res))

    assert result == res


def test_get_file_res_derives_resolution_from_dataset():
    ds = make_ds(np.arange(0, 10, 0.5), np.arange(40, 45, 0.5))

    assert get_file_res(ds=ds) == pytest.approx(0.5, abs=1e-9)


def test_get_file_res_uses_matching_resolution_attribute(caplog):
    # the attribute is the clean value, so it wins over the float32 measurement
    res = 1 / 120
    ds = make_ds(
        make_global_lon(res, dtype=np.float32),
        np.arange(40, 45, res, dtype=np.float32),
        resolution_attr=res,
    )

    with caplog.at_level(logging.WARNING):
        result = get_file_res(ds=ds)

    assert result == res
    assert not has_warning(caplog, "differs from the resolution")


def test_get_file_res_warns_when_resolution_attribute_disagrees(caplog):
    ds = make_ds(np.arange(0, 10, 0.5), np.arange(40, 45, 0.5), resolution_attr=1.0)

    with caplog.at_level(logging.WARNING):
        result = get_file_res(ds=ds)

    # the coordinates are the ground truth, the attribute is only reported
    assert result == pytest.approx(0.5, abs=1e-9)
    assert has_warning(caplog, "differs from the resolution")


def test_get_file_res_trusts_resolution_attribute_of_single_cell_dataset(caplog):
    res = 1 / 240
    ds = make_ds(np.array([4.0]), np.array([-60.0]), resolution_attr=res)

    with caplog.at_level(logging.WARNING):
        result = get_file_res(ds=ds)

    assert result == res
    assert not has_warning(caplog, "differs from the resolution")


def test_get_file_res_accepts_a_dataarray():
    da = make_ds(np.arange(0, 10, 0.25), np.arange(40, 45, 0.25), as_dataarray=True)

    assert get_file_res(ds=da) == pytest.approx(0.25, abs=1e-9)


def test_get_file_res_snaps_dataset_resolution_to_provided_resolutions():
    res = 1 / 120
    ds = make_ds(
        make_global_lon(res, dtype=np.float32),
        np.arange(40, 45, res, dtype=np.float32),
    )

    assert get_file_res(ds=ds, resolutions=Resolution(l0=res)) == res


def test_get_file_res_returns_positive_resolution_for_descending_dataset():
    ds = make_ds(np.arange(10, 0, -0.5), np.arange(45, 40, -0.5))

    assert get_file_res(ds=ds) == pytest.approx(0.5, abs=1e-9)


def test_get_file_res_falls_back_to_latitude_of_dataset():
    # a single longitude cell must not stop the latitude from being measured
    ds = make_ds(np.array([4.0]), np.arange(40, 45, 0.25))

    assert get_file_res(ds=ds) == pytest.approx(0.25, abs=1e-9)


def test_get_file_res_returns_nan_for_single_cell_dataset_without_attribute(caplog):
    ds = make_ds(np.array([4.0]), np.array([-60.0]))

    with caplog.at_level(logging.WARNING):
        result = get_file_res(ds=ds, raise_exception=False)

    assert np.isnan(result)
    assert has_warning(caplog, "using NaN")


def test_get_file_res_returns_nan_for_too_small_coordinates():
    lon = xr.DataArray(np.array([5.0]), dims="lon")

    assert np.isnan(get_file_res(lon=lon, raise_exception=False))


def test_get_file_res_raises_for_single_cell_dataset_by_default():
    ds = make_ds(np.array([4.0]), np.array([-60.0]))

    with pytest.raises(ValueError, match="Cannot determine file resolution"):
        get_file_res(ds=ds)
