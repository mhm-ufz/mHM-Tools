"""Tests for the discharge unit handling of the discharge evaluation."""

import logging

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from mhm_tools.common.logger import configure_mhm_tools_logger
from mhm_tools.post.discharge_evaluation import convert_discharge_to_observed_units


@pytest.fixture(autouse=True, scope="session")
def _configure_test_logging():
    """Let caplog capture the package logs."""
    configure_mhm_tools_logger(propagate=True)


def _discharge(values, units):
    """Create a daily discharge record with the given units."""
    time = pd.date_range("1990-01-01", periods=len(values), freq="D")
    attrs = {"long_name": "discharge"}
    if units is not None:
        attrs["units"] = units
    return xr.DataArray(
        np.asarray(values, dtype=float),
        dims="time",
        coords={"time": time},
        attrs=attrs,
    )


def test_convert_discharge_to_observed_units_converts_the_simulation():
    """Convert a simulation in m3 d-1 to the observed m3 s-1."""
    simulated, observed = convert_discharge_to_observed_units(
        _discharge([86400.0, 172800.0], "m3 d-1"), _discharge([1.0, 2.0], "m3 s-1")
    )
    np.testing.assert_allclose(simulated.values, [1.0, 2.0])
    assert simulated.attrs == {"long_name": "discharge", "units": "m3 s-1"}
    assert observed.attrs["units"] == "m3 s-1"


def test_convert_discharge_to_observed_units_assumes_m3_s_without_units(caplog):
    """Take a record without units as m3/s and warn about it."""
    with caplog.at_level(logging.WARNING):
        simulated, _ = convert_discharge_to_observed_units(
            _discharge([1.0], None), _discharge([1.0], "m3 s-1")
        )
    assert simulated.attrs["units"] == "m3 s-1"
    assert "has no units" in caplog.text


def test_convert_discharge_to_observed_units_rejects_a_depth_rate():
    """Refuse to compare a depth rate with a volume rate."""
    with pytest.raises(ValueError, match="catchment area"):
        convert_discharge_to_observed_units(
            _discharge([1.0], "mm d-1"), _discharge([1.0], "m3 s-1")
        )
