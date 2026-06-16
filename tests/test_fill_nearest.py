import logging

import numpy as np
import pytest
import xarray as xr

from mhm_tools.common.logger import configure_mhm_tools_logger
from mhm_tools.pre.fill_nearest import fill_dataarray_with_nearest


@pytest.fixture(autouse=True, scope="session")
def _configure_test_logging():
    """Configure mhm_tools logging for the test session.

    Sets the package logger to ERROR and enables propagation so pytest's
    caplog captures log records without cluttering test output.
    """
    # Only enable propagation so pytest's caplog can capture package logs.
    configure_mhm_tools_logger(propagate=True)
    yield


def test_fill_dataarray_with_nearest_handles_no_valid_source_cells(caplog):
    data = xr.DataArray(
        np.full((2, 2, 2), -9999.0),
        dims=("time", "lat", "lon"),
        coords={"time": [0, 1], "lat": [0.0, 1.0], "lon": [0.0, 1.0]},
        name="pre",
    )

    with caplog.at_level(logging.WARNING):
        filled = fill_dataarray_with_nearest(
            data,
            missing_value=-9999.0,
            fill_value=-9999.0,
            source_file="meteo/pre.nc",
        )

    assert filled == 0
    assert np.all(data.values == -9999.0)
    assert "no valid source cells are available" in caplog.text
    assert "meteo/pre.nc" in caplog.text


def test_fill_dataarray_with_nearest_reports_filled_cells():
    data = xr.DataArray(
        np.array(
            [
                [[1.0, -9999.0], [3.0, 4.0]],
                [[2.0, -9999.0], [4.0, 5.0]],
            ]
        ),
        dims=("time", "lat", "lon"),
        coords={"time": [0, 1], "lat": [0.0, 1.0], "lon": [0.0, 1.0]},
        name="pre",
    )

    filled = fill_dataarray_with_nearest(
        data,
        missing_value=-9999.0,
        fill_value=-9999.0,
        source_file="meteo/pre.nc",
    )

    assert filled == 1
    assert np.all(data.sel(lon=1.0, lat=0.0).values == np.array([1.0, 2.0]))


def test_fill_dataarray_with_nearest_respects_mask():
    data = xr.DataArray(
        np.array(
            [
                [[1.0, -9999.0, -9999.0], [4.0, 5.0, 6.0]],
                [[2.0, -9999.0, -9999.0], [5.0, 6.0, 7.0]],
            ]
        ),
        dims=("time", "lat", "lon"),
        coords={"time": [0, 1], "lat": [0.0, 1.0], "lon": [0.0, 1.0, 2.0]},
        name="pre",
    )
    mask = np.array([[False, False, True], [False, False, False]])

    filled = fill_dataarray_with_nearest(
        data,
        missing_value=-9999.0,
        mask=mask,
        fill_value=-1.0,
        source_file="meteo/pre.nc",
    )

    assert filled == 1
    assert np.all(data.sel(lon=1.0, lat=0.0).values == np.array([1.0, 2.0]))
    assert np.all(data.sel(lon=2.0, lat=0.0).values == np.array([-1.0, -1.0]))
