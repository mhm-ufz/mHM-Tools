"""Tests for the unit conversion of the mHM run overview."""

import pytest

from mhm_tools.common.constants import SECONDS_PER_YEAR
from mhm_tools.post.mhm_run_overview import _convert_stats_to_target_time_unit


def _stats(unit):
    """Create a stats row with the given unit."""
    return {"min_value": 24.0, "mean_value": 48.0, "max_value": 72.0, "unit": unit}


def test_convert_stats_to_target_time_unit_uses_hours_for_hourly_input():
    """Express daily rates per hour for an hourly setup instead of per second."""
    converted = _convert_stats_to_target_time_unit(_stats("mm d-1"), 3600.0)
    assert converted["unit"] == "mm h-1"
    assert converted["mean_value"] == pytest.approx(2.0)


def test_convert_stats_to_target_time_unit_uses_a_twelfth_year_as_month():
    """Take a month as 1/12 of a 365.25 day year."""
    converted = _convert_stats_to_target_time_unit(_stats("mm d-1"), 30 * 86400.0)
    assert converted["unit"] == "mm month-1"
    assert converted["mean_value"] == pytest.approx(
        48.0 * SECONDS_PER_YEAR / 12 / 86400
    )


def test_convert_stats_to_target_time_unit_keeps_units_without_time():
    """Leave stats of a unit without a time denominator unchanged."""
    stats = _stats("W m-2")
    assert _convert_stats_to_target_time_unit(stats, 3600.0) == stats
