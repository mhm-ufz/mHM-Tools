"""Tests for resolving the log level from the command line options."""

import pytest

from mhm_tools.common.logger import get_log_level


@pytest.mark.parametrize("count_verbose", [1, 2, 3])
def test_get_log_level_clamps_verbose_to_debug(count_verbose):
    """Stay at DEBUG however often -v is repeated."""
    assert get_log_level(count_verbose=count_verbose) == (10, None)


@pytest.mark.parametrize("count_quiet", [2, 3, 4])
def test_get_log_level_clamps_quiet_to_error(count_quiet):
    """Stay at ERROR however often -q is repeated."""
    assert get_log_level(count_quiet=count_quiet) == (40, None)


@pytest.mark.parametrize("level", ["CRITICAL", "critical"])
def test_get_log_level_maps_critical_to_error(level):
    """Fall back from CRITICAL to ERROR without an error message."""
    assert get_log_level(level) == (40, None)
