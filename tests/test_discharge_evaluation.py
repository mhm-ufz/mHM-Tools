"""Tests for the discharge evaluation: unit handling and gauge-to-node matching."""

import logging

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from mhm_tools.common.logger import configure_mhm_tools_logger
from mhm_tools.post.discharge_evaluation import (
    convert_discharge_to_observed_units,
    get_sim_data_for_gauges_from_nodes,
)


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


def _node_output(station_ids=None):
    """Create mRM node output of four nodes along x, each holding its index.

    Parameters
    ----------
    station_ids : sequence, optional
        Station id of every node, None for output without station ids.

    Returns
    -------
    xr.Dataset
        Node output with ``discharge(station, time)`` and node ``x``/``y``.
    """
    node_count, step_count = 4, 3
    dataset = xr.Dataset(
        {
            "discharge": (
                ("station", "time"),
                np.repeat(np.arange(node_count, dtype=float), step_count).reshape(
                    node_count, step_count
                ),
            ),
            "x": ("station", np.arange(node_count) * 10.0),
            "y": ("station", np.zeros(node_count)),
        },
        coords={"time": pd.date_range("1990-01-01", periods=step_count, freq="D")},
    )
    if station_ids is not None:
        dataset = dataset.assign_coords(station=np.asarray(station_ids))
    return dataset


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


def test_gauges_are_matched_to_nodes_by_station_id_before_coordinates():
    """Match by station id even where the gauge coordinates point at other nodes."""
    sim_sel, matched_x, matched_y, ids = get_sim_data_for_gauges_from_nodes(
        _node_output(station_ids=[101, 102, 103, 104]),
        "discharge",
        x_new=[0.0, 30.0],
        y_new=[5.0, 5.0],
        gauge_ids=[103, 101],
    )
    np.testing.assert_array_equal(ids, [103, 101])
    np.testing.assert_array_equal(sim_sel.sel(id=103).values, [2.0, 2.0, 2.0])
    np.testing.assert_array_equal(sim_sel.sel(id=101).values, [0.0, 0.0, 0.0])
    # the gauge coordinates from the gauges file are kept
    np.testing.assert_array_equal(matched_x, [0.0, 30.0])
    np.testing.assert_array_equal(matched_y, [5.0, 5.0])


def test_gauges_without_a_node_station_id_are_dropped_with_a_warning(caplog):
    """Drop a gauge whose id no node carries and match the others."""
    with caplog.at_level(logging.WARNING):
        sim_sel, _, _, ids = get_sim_data_for_gauges_from_nodes(
            _node_output(station_ids=[101, 102, 103, 104]),
            "discharge",
            x_new=[10.0, 20.0],
            y_new=[0.0, 0.0],
            gauge_ids=[102, 999],
        )
    np.testing.assert_array_equal(ids, [102])
    np.testing.assert_array_equal(sim_sel.sel(id=102).values, [1.0, 1.0, 1.0])
    assert "Dropping 1 gauge(s)" in caplog.text


def test_no_matching_station_id_is_refused():
    """Refuse node output whose station ids match none of the gauges."""
    with pytest.raises(ValueError, match="None of the 2 gauge ids"):
        get_sim_data_for_gauges_from_nodes(
            _node_output(station_ids=[101, 102, 103, 104]),
            "discharge",
            x_new=[0.0, 10.0],
            y_new=[0.0, 0.0],
            gauge_ids=[998, 999],
        )


def test_node_fill_values_are_ignored_and_the_first_repeated_id_is_used(caplog):
    """Skip nodes without an id and use the first of nodes sharing one."""
    with caplog.at_level(logging.WARNING):
        sim_sel, _, _, ids = get_sim_data_for_gauges_from_nodes(
            _node_output(station_ids=[np.nan, 102.0, 102.0, 104.0]),
            "discharge",
            x_new=[0.0, 0.0],
            y_new=[0.0, 0.0],
            gauge_ids=[102, 104],
        )
    np.testing.assert_array_equal(ids, [102, 104])
    np.testing.assert_array_equal(sim_sel.sel(id=102).values, [1.0, 1.0, 1.0])
    np.testing.assert_array_equal(sim_sel.sel(id=104).values, [3.0, 3.0, 3.0])
    assert "repeat a station id" in caplog.text


def test_gauges_are_matched_by_coordinates_without_node_station_ids():
    """Fall back to the nearest node when the node output holds no station ids."""
    sim_sel, matched_x, _, ids = get_sim_data_for_gauges_from_nodes(
        _node_output(),
        "discharge",
        x_new=[19.0, 1.0],
        y_new=[0.0, 0.0],
        gauge_ids=[7, 8],
    )
    np.testing.assert_array_equal(ids, [7, 8])
    np.testing.assert_array_equal(sim_sel.sel(id=7).values, [2.0, 2.0, 2.0])
    np.testing.assert_array_equal(sim_sel.sel(id=8).values, [0.0, 0.0, 0.0])
    # the coordinate matching reports the coordinates of the matched nodes
    np.testing.assert_array_equal(matched_x, [20.0, 0.0])
