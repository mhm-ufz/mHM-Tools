"""Tests for the shared unit converter and its time and DataArray helpers."""

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from mhm_tools.common.constants import SECONDS_PER_YEAR
from mhm_tools.common.time_utils import calculate_median_time_step_seconds
from mhm_tools.common.units import (
    calculate_amount_factor,
    calculate_conversion_factor,
    convert_rate_time_unit,
    convert_temperature_to_celsius,
    split_units,
)
from mhm_tools.common.xarray_utils import convert_dataarray_units

DAY = 86400.0
HOUR = 3600.0


@pytest.mark.parametrize(
    ("units", "kind", "amount_factor", "time_seconds"),
    [
        ("m3 s-1", "volume", 1.0, 1.0),
        ("m3/s", "volume", 1.0, 1.0),
        ("m³/s", "volume", 1.0, 1.0),
        ("m^3/s", "volume", 1.0, 1.0),
        ("m**3 s**-1", "volume", 1.0, 1.0),
        ("m3.s-1", "volume", 1.0, 1.0),
        ("m3 d-1", "volume", 1.0, DAY),
        ("l/s", "volume", 1e-3, 1.0),
        ("km3/day", "volume", 1e9, DAY),
        ("mm/d", "depth", 1e-3, DAY),
        ("mm day-1", "depth", 1e-3, DAY),
        ("mm/days", "depth", 1e-3, DAY),
        ("mm h-1", "depth", 1e-3, HOUR),
        ("mm/hr", "depth", 1e-3, HOUR),
        ("kg m-2 s-1", "depth", 1e-3, 1.0),
        ("kg/m2/s", "depth", 1e-3, 1.0),
        ("kg m⁻² s⁻¹", "depth", 1e-3, 1.0),
        ("mm/month", "depth", 1e-3, SECONDS_PER_YEAR / 12),
        ("mm/year", "depth", 1e-3, SECONDS_PER_YEAR),
        ("mm", "depth", 1e-3, None),
        ("kg m-2", "depth", 1e-3, None),
        ("cm of water", "depth", 1e-2, None),
        ("mm water equivalent", "depth", 1e-3, None),
        ("m", "depth", 1.0, None),
    ],
)
def test_split_units_reads_each_spelling(units, kind, amount_factor, time_seconds):
    """Read volume and depth amounts and rates in their common spellings."""
    parts = split_units(units)
    assert parts.kind == kind
    assert parts.amount_factor == pytest.approx(amount_factor)
    if time_seconds is None:
        assert parts.time_seconds is None
    else:
        assert parts.time_seconds == pytest.approx(time_seconds)


@pytest.mark.parametrize("units", ["W m-2", "m3 s-2", "s-1", "kelvin", ""])
def test_split_units_rejects_unknown_units(units):
    """Refuse units that are no amount or amount per single time unit."""
    with pytest.raises(ValueError, match=r"units|amount"):
        split_units(units)


@pytest.mark.parametrize(
    ("from_units", "to_units", "factor"),
    [
        ("m3 d-1", "m3 s-1", 1 / DAY),
        ("kg m-2 s-1", "mm d-1", DAY),
        ("mm h-1", "mm d-1", 24.0),
        ("l s-1", "m3 s-1", 1e-3),
        ("mm", "cm", 0.1),
        ("mm/month", "mm/year", 12.0),
    ],
)
def test_calculate_conversion_factor_between_units(from_units, to_units, factor):
    """Convert between amounts and between rates of the same kind."""
    assert calculate_conversion_factor(from_units, to_units) == pytest.approx(factor)


@pytest.mark.parametrize(
    ("from_units", "to_units"),
    [("m3 s-1", "mm d-1"), ("mm", "mm d-1")],
)
def test_calculate_conversion_factor_rejects_other_kinds(from_units, to_units):
    """Refuse volume to depth and amount to rate conversions."""
    with pytest.raises(ValueError, match="Cannot convert"):
        calculate_conversion_factor(from_units, to_units)


@pytest.mark.parametrize(
    ("rate_units", "step_seconds", "to_units", "area_km2", "factor"),
    [
        ("m3 s-1", DAY, "m3", None, DAY),
        ("mm s-1", DAY, "mm", None, DAY),
        ("mm d-1", HOUR, "mm", None, 1 / 24),
        ("mm d-1", DAY, "m3", 100.0, 1e5),
    ],
)
def test_calculate_amount_factor_turns_rates_into_amounts(
    rate_units, step_seconds, to_units, area_km2, factor
):
    """Turn a rate over one time step into an amount, a depth via the area."""
    assert calculate_amount_factor(
        rate_units, step_seconds, to_units, area_km2
    ) == pytest.approx(factor)


def test_calculate_amount_factor_without_area_is_nan():
    """Give NaN for a depth rate turned into a volume without an area."""
    assert np.isnan(calculate_amount_factor("mm d-1", DAY, "m3"))


@pytest.mark.parametrize(
    ("units", "time_unit", "factor", "new_units"),
    [
        ("mm d-1", "h", 1 / 24, "mm h-1"),
        ("mm/month", "d", 12 * DAY / SECONDS_PER_YEAR, "mm d-1"),
    ],
)
def test_convert_rate_time_unit_rescales_the_rate(units, time_unit, factor, new_units):
    """Express a rate per another time unit and rewrite its units."""
    converted_factor, converted_units = convert_rate_time_unit(units, time_unit)
    assert converted_factor == pytest.approx(factor)
    assert converted_units == new_units


@pytest.mark.parametrize(
    ("value", "units", "celsius"),
    [
        (273.15, "K", 0.0),
        (212.0, "degF", 100.0),
        (5.0, "degC", 5.0),
        (5.0, "degree_Celsius", 5.0),
    ],
)
def test_convert_temperature_to_celsius(value, units, celsius):
    """Convert Kelvin and Fahrenheit and keep Celsius."""
    assert convert_temperature_to_celsius(value, units) == pytest.approx(celsius)


def test_convert_temperature_to_celsius_rejects_unknown_units():
    """Refuse a unit that is no temperature."""
    with pytest.raises(ValueError, match="temperature units"):
        convert_temperature_to_celsius(5.0, "mm")


@pytest.mark.parametrize(
    ("time_values", "step_seconds"),
    [
        (pd.date_range("2000-01-01", periods=5, freq="D").values, DAY),
        (pd.date_range("2000-01-01", periods=5, freq="h").values, HOUR),
        (
            np.array(
                ["2000-01-01", "2000-01-02", "2000-01-05", "2000-01-06"],
                dtype="datetime64[ns]",
            ),
            DAY,
        ),
    ],
)
def test_calculate_median_time_step_seconds(time_values, step_seconds):
    """Take the median step, so a single gap does not change it."""
    assert calculate_median_time_step_seconds(time_values) == pytest.approx(
        step_seconds
    )


def test_calculate_median_time_step_seconds_of_one_value_is_none():
    """Give None when a single time value has no step."""
    time_values = pd.date_range("2000-01-01", periods=1).values
    assert calculate_median_time_step_seconds(time_values) is None


def test_convert_dataarray_units_scales_and_keeps_attributes():
    """Scale the values, set the units and keep the other attributes."""
    da = xr.DataArray(
        [DAY, 2 * DAY], dims="time", attrs={"units": "m3 d-1", "long_name": "Q"}
    )
    converted = convert_dataarray_units(da, "m3 s-1")
    np.testing.assert_allclose(converted.values, [1.0, 2.0])
    assert converted.attrs == {"units": "m3 s-1", "long_name": "Q"}
    assert da.attrs["units"] == "m3 d-1"
