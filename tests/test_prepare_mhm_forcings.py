"""Tests for the unit conversion of the mHM forcing preparation."""

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from mhm_tools.pre.prepare_mhm_forcings import convert_units


def _forcing(units, freq="D", value=1.0):
    """Create a short forcing record with the given units and time step."""
    time = pd.date_range("2000-01-01", periods=4, freq=freq)
    return xr.DataArray(
        np.full(4, value),
        dims="time",
        coords={"time": time},
        attrs={"units": units, "long_name": "forcing"},
        name="forcing",
    )


@pytest.mark.parametrize(
    ("units", "freq", "factor"),
    [
        ("kg m-2", "D", 1.0),
        ("m", "D", 1000.0),
        ("mm s-1", "D", 86400.0),
        ("kg m-2 s-1", "h", 3600.0),
        ("mm d-1", "h", 1 / 24),
    ],
)
def test_convert_units_turns_precipitation_into_mm(units, freq, factor):
    """Turn precipitation amounts and rates into mm per time step."""
    converted, _ = convert_units(_forcing(units, freq), "forcing")
    np.testing.assert_allclose(converted.values, factor)
    assert converted.attrs["units"] == "mm"


@pytest.mark.parametrize(
    ("units", "value", "celsius"),
    [("K", 273.15, 0.0), ("degF", 212.0, 100.0)],
)
def test_convert_units_turns_temperature_into_celsius(units, value, celsius):
    """Turn Kelvin and Fahrenheit into degrees Celsius."""
    converted, _ = convert_units(_forcing(units, value=value), "forcing")
    np.testing.assert_allclose(converted.values, celsius)
    assert converted.attrs["units"] == "degC"


def test_convert_units_keeps_attributes_and_returns_the_encoding():
    """Keep the other attributes and return the fill value encoding."""
    converted, encoding = convert_units(_forcing("kg m-2 s-1"), "forcing")
    assert converted.attrs["long_name"] == "forcing"
    assert encoding == {"_FillValue": -9999.0, "missing_value": -9999.0}


@pytest.mark.parametrize("units", [None, "m3 s-1"])
def test_convert_units_rejects_missing_or_unknown_units(units):
    """Refuse a forcing without units or with units that are no forcing."""
    forcing = _forcing("mm")
    forcing.attrs["units"] = units
    with pytest.raises(ValueError, match="units"):
        convert_units(forcing, "forcing")
