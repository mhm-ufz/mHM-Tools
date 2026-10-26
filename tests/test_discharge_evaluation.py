"""Tests for the discharge evaluation (``mhm_tools.post.discharge_evaluation``).

Unit handling, gauge matching to river nodes and grids, gauge maps, and the
data workflow of ``Q_data_to_xarray`` and ``evaludate_discharge_data``.
"""

import contextlib
import logging
import tempfile
import tracemalloc
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest
import xarray as xr

import mhm_tools.post.discharge_evaluation as discharge_evaluation
from mhm_tools.common.logger import configure_mhm_tools_logger
from mhm_tools.post.discharge_evaluation import (
    convert_discharge_to_observed_units,
    get_sim_data_for_gauges_from_nodes,
    plot_map,
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


@pytest.fixture
def _offline_map_projections(monkeypatch):
    """Record the projection of every saved map and skip the map features.

    Coastlines and borders would be downloaded by cartopy; the tests only need
    the gauges, so ``add_feature`` is a no-op.

    Returns
    -------
    list
        The cartopy projection of every figure saved by ``plot_map``.
    """
    pytest.importorskip("cartopy")
    from cartopy.mpl.geoaxes import GeoAxes
    from matplotlib.figure import Figure

    projections = []
    save_figure = Figure.savefig

    def record_and_save(figure, *args, **kwargs):
        """Record the map projection before saving the figure."""
        projections.append(figure.axes[0].projection)
        return save_figure(figure, *args, **kwargs)

    monkeypatch.setattr(GeoAxes, "add_feature", lambda *args, **kwargs: None)
    monkeypatch.setattr(Figure, "savefig", record_and_save)
    return projections


def test_plot_map_draws_gauges_in_degrees_on_a_plate_carree_map(
    tmp_path, _offline_map_projections
):
    """Write one PlateCarree map per variable for gauges given in degrees."""
    import cartopy.crs as ccrs

    results = pd.DataFrame(
        {
            "id": [1, 2, 3],
            "x": [10.0, 11.0, 12.0],
            "y": [50.0, 51.0, 52.0],
            "kge": [0.1, 0.5, 0.9],
        }
    )

    written = plot_map(results, tmp_path, variables=["kge"])

    assert written == [tmp_path / "map_kge.png"]
    assert written[0].is_file()
    assert isinstance(_offline_map_projections[0], ccrs.PlateCarree)


def _gridded_output(step_count=3):
    """Create gridded output of 3 x 4 cells of 1000 m, each holding 10 * row + column.

    The grid carries 2-D lon/lat coordinates like CF output on a projected grid,
    which are no grid dimensions.

    Returns
    -------
    xr.DataArray
        Discharge ``rdis(time, y, x)`` with y descending.
    """
    y = np.array([3500.0, 2500.0, 1500.0])
    x = np.array([500.0, 1500.0, 2500.0, 3500.0])
    cell_values = 10.0 * np.arange(y.size)[:, None] + np.arange(x.size)[None, :]
    lon, lat = np.meshgrid(x / 1000.0, y / 1000.0)
    return xr.DataArray(
        np.broadcast_to(cell_values, (step_count, y.size, x.size)).copy(),
        dims=("time", "y", "x"),
        coords={
            "time": pd.date_range("1990-01-01", periods=step_count, freq="D"),
            "y": y,
            "x": x,
            "lon": (("y", "x"), lon),
            "lat": (("y", "x"), lat),
        },
        name="rdis",
        attrs={"units": "m3 s-1"},
    )


def test_gridded_gauges_take_the_nearest_cell_and_keep_their_coordinates():
    """Read every gauge from the grid cell it lies in."""
    sim_sel, gauge_x, gauge_y, ids = (
        discharge_evaluation.get_sim_data_for_gauges_from_grid(
            _gridded_output(),
            x_values=[1400.0, 3600.0],
            y_values=[3550.0, 1600.0],
            gauge_ids=[7, 8],
        )
    )
    assert set(sim_sel.dims) == {"time", "id"}
    np.testing.assert_array_equal(ids, [7, 8])
    # gauge 7 lies in row 0, column 1; gauge 8 in row 2, column 3
    np.testing.assert_array_equal(sim_sel.sel(id=7).values, [1.0, 1.0, 1.0])
    np.testing.assert_array_equal(sim_sel.sel(id=8).values, [23.0, 23.0, 23.0])
    np.testing.assert_array_equal(gauge_x, [1400.0, 3600.0])
    np.testing.assert_array_equal(gauge_y, [3550.0, 1600.0])


def test_gridded_gauges_outside_the_grid_are_dropped_with_a_warning(caplog):
    """Drop a gauge beyond the grid instead of reading the edge cell."""
    sim_sel, _, _, ids = discharge_evaluation.get_sim_data_for_gauges_from_grid(
        _gridded_output(),
        x_values=[1500.0, 9000.0],
        y_values=[2500.0, 2500.0],
        gauge_ids=[7, 8],
    )
    np.testing.assert_array_equal(ids, [7])
    np.testing.assert_array_equal(sim_sel.sel(id=7).values, [11.0, 11.0, 11.0])
    assert "outside" in caplog.text


def test_no_gauge_on_the_grid_points_to_the_crs():
    """Refuse gauges that all lie beyond the grid, hinting at their CRS."""
    with pytest.raises(ValueError, match="CRS"):
        discharge_evaluation.get_sim_data_for_gauges_from_grid(
            _gridded_output(), x_values=[1.0e6], y_values=[5.0e6], gauge_ids=[7]
        )


def test_node_output_is_refused_by_the_grid_matching():
    """Refuse node output where gridded output is expected."""
    with pytest.raises(ValueError, match="gridded"):
        discharge_evaluation.get_sim_data_for_gauges_from_grid(
            _node_output(station_ids=[1, 2, 3, 4])["discharge"],
            x_values=[0.0],
            y_values=[0.0],
            gauge_ids=[1],
        )


def test_search_window_holds_the_cells_around_the_gauge_with_nan_beyond_the_grid():
    """Return the 3 x 3 cells around a corner gauge, rows first, NaN off the grid."""
    sim_sel, _, _, _ = discharge_evaluation.get_sim_data_for_gauges_from_grid(
        _gridded_output(),
        x_values=[500.0],
        y_values=[3500.0],
        gauge_ids=[7],
        search_cells=1,
    )
    assert sim_sel.sizes["candidate"] == 9
    window = sim_sel.sel(id=7).isel(time=0).transpose("candidate").values
    nan = np.nan
    np.testing.assert_array_equal(
        window, [nan, nan, nan, nan, 0.0, 1.0, nan, 10.0, 11.0]
    )
    np.testing.assert_array_equal(
        sim_sel["candidate_dy"], [-1, -1, -1, 0, 0, 0, 1, 1, 1]
    )
    np.testing.assert_array_equal(
        sim_sel["candidate_dx"], [-1, 0, 1, -1, 0, 1, -1, 0, 1]
    )


def _candidate_series(series_by_offset, observed_values):
    """Create the candidate series of gauge 7 and its observations.

    Parameters
    ----------
    series_by_offset : dict
        (dy, dx) cell offset -> discharge series of that cell.
    observed_values : sequence
        Observed discharge series of gauge 7.

    Returns
    -------
    tuple
        Candidates ``(time, id, candidate)`` with offsets and cell coordinates
        (cells of 1000 m around x=1000, y=1000), and observations ``(time, id)``.
    """
    offsets = list(series_by_offset)
    time = pd.date_range("1990-01-01", periods=len(observed_values), freq="D")
    data = np.array([series_by_offset[offset] for offset in offsets], dtype=float)
    candidates = xr.DataArray(
        data.T[:, None, :],
        dims=("time", "id", "candidate"),
        coords={
            "time": time,
            "id": [7],
            "candidate_dy": ("candidate", [dy for dy, _ in offsets]),
            "candidate_dx": ("candidate", [dx for _, dx in offsets]),
            "candidate_x": (
                ("id", "candidate"),
                [[1000.0 * (1 + dx) for _, dx in offsets]],
            ),
            "candidate_y": (
                ("id", "candidate"),
                [[1000.0 * (1 - dy) for dy, _ in offsets]],
            ),
        },
        name="rdis",
    )
    observed = xr.DataArray(
        np.asarray(observed_values, dtype=float)[:, None],
        dims=("time", "id"),
        coords={"time": time, "id": [7]},
    )
    return candidates, observed


def test_mean_flow_picks_the_cell_matching_the_observed_mean_on_shared_days():
    """Pick the neighbour whose mean equals the observations on observed days."""
    # on all days the centre (mean 10) would be closer to 5 than the neighbour (36.7)
    candidates, observed = _candidate_series(
        {(0, 0): [10.0, 10.0, 10.0], (0, 1): [5.0, 5.0, 100.0]},
        observed_values=[5.0, 5.0, np.nan],
    )
    sim_sel, cell_x, cell_y = discharge_evaluation.select_cells_by_mean_flow(
        candidates, observed
    )
    np.testing.assert_array_equal(sim_sel.sel(id=7).values, [5.0, 5.0, 100.0])
    np.testing.assert_array_equal(cell_x, [2000.0])
    np.testing.assert_array_equal(cell_y, [1000.0])


def test_mean_flow_keeps_the_centre_cell_on_a_tie():
    """Keep the gauge's own cell when a neighbour matches equally well."""
    candidates, observed = _candidate_series(
        {(0, 1): [4.0, 4.0, 4.0], (0, 0): [6.0, 6.0, 6.0]},
        observed_values=[5.0, 5.0, 5.0],
    )
    sim_sel, cell_x, _ = discharge_evaluation.select_cells_by_mean_flow(
        candidates, observed
    )
    np.testing.assert_array_equal(sim_sel.sel(id=7).values, [6.0, 6.0, 6.0])
    np.testing.assert_array_equal(cell_x, [1000.0])


def test_read_gauge_coordinates_keeps_reference_ids_and_uses_x_y(tmp_path):
    """Read x/y of the gauges that have observations, in file order."""
    gauges_file = tmp_path / "gauges.nc"
    xr.Dataset(
        {
            "station": ("station", [1.0, 2.0, 3.0]),
            "x": ("station", [4000.0, 5000.0, 6000.0]),
            "y": ("station", [3000.0, 3100.0, 3200.0]),
        }
    ).to_netcdf(gauges_file)

    gauge_x, gauge_y, ids = discharge_evaluation.read_gauge_coordinates(
        gauges_file, reference_ids=[2.0, 3.0, 9.0]
    )
    np.testing.assert_array_equal(ids, [2.0, 3.0])
    np.testing.assert_array_equal(gauge_x, [5000.0, 6000.0])
    np.testing.assert_array_equal(gauge_y, [3100.0, 3200.0])


def test_read_gauge_coordinates_prefers_lon_lat(tmp_path):
    """Use lon/lat when the gauges file holds them besides x/y."""
    gauges_file = tmp_path / "gauges.nc"
    xr.Dataset(
        {
            "station": ("station", [1.0, 2.0]),
            "x": ("station", [4000.0, 5000.0]),
            "y": ("station", [3000.0, 3100.0]),
            "lon": ("station", [10.0, 11.0]),
            "lat": ("station", [50.0, 51.0]),
        }
    ).to_netcdf(gauges_file)

    gauge_x, gauge_y, _ = discharge_evaluation.read_gauge_coordinates(
        gauges_file, reference_ids=[1.0, 2.0]
    )
    np.testing.assert_array_equal(gauge_x, [10.0, 11.0])
    np.testing.assert_array_equal(gauge_y, [50.0, 51.0])


def test_read_gauge_coordinates_refuses_gauges_without_observations(tmp_path):
    """Refuse a gauges file sharing no id with the reference."""
    gauges_file = tmp_path / "gauges.nc"
    xr.Dataset(
        {
            "station": ("station", [1.0]),
            "x": ("station", [4000.0]),
            "y": ("station", [3000.0]),
        }
    ).to_netcdf(gauges_file)
    with pytest.raises(ValueError, match="No gauges"):
        discharge_evaluation.read_gauge_coordinates(gauges_file, reference_ids=[5.0])


def _write_gauges_with_lonlat_and_xy(gauges_file):
    """Write two gauges holding lon/lat in degrees and x/y in metres."""
    xr.Dataset(
        {
            "station": ("station", [1.0, 2.0]),
            "x": ("station", [4000.0, 5000.0]),
            "y": ("station", [3000.0, 3100.0]),
            "lon": ("station", [10.0, 11.0]),
            "lat": ("station", [50.0, 51.0]),
        }
    ).to_netcdf(gauges_file)
    return gauges_file


def test_read_gauge_coordinates_uses_xy_on_projected_grid(tmp_path):
    """Use x/y besides lon/lat when the model grid has x/y dimensions."""
    gauges_file = _write_gauges_with_lonlat_and_xy(tmp_path / "gauges.nc")

    gauge_x, gauge_y, _ = discharge_evaluation.read_gauge_coordinates(
        gauges_file, reference_ids=[1.0, 2.0], grid_dim_names=("y", "x")
    )
    np.testing.assert_array_equal(gauge_x, [4000.0, 5000.0])
    np.testing.assert_array_equal(gauge_y, [3000.0, 3100.0])


def test_read_gauge_coordinates_uses_lonlat_on_geographic_grid(tmp_path):
    """Use lon/lat besides x/y when the model grid has lat/lon dimensions."""
    gauges_file = _write_gauges_with_lonlat_and_xy(tmp_path / "gauges.nc")

    gauge_x, gauge_y, _ = discharge_evaluation.read_gauge_coordinates(
        gauges_file, reference_ids=[1.0, 2.0], grid_dim_names=("lat", "lon")
    )
    np.testing.assert_array_equal(gauge_x, [10.0, 11.0])
    np.testing.assert_array_equal(gauge_y, [50.0, 51.0])


@pytest.mark.parametrize("ascending", [True, False])
def test_nearest_grid_indices_finds_cells_and_marks_outside_gauges(ascending):
    """Find the cell of every gauge on either axis order, -1 beyond the grid."""
    axis = np.arange(10) + 0.5
    gauges = np.array([0.2, 3.7, 9.9, 10.6, -0.6, np.nan])
    expected = np.array([0, 3, 9, -1, -1, -1])
    if not ascending:
        axis = axis[::-1]
        expected = np.where(expected >= 0, axis.size - 1 - expected, -1)
    indices = discharge_evaluation._get_nearest_grid_indices(axis, gauges)
    np.testing.assert_array_equal(indices, expected)


def test_nearest_grid_indices_does_not_build_a_gauge_by_cell_matrix():
    """Match many gauges on a long axis without a gauges x cells array."""
    axis = np.arange(50_000) * 0.01
    gauges = np.linspace(0.0, 499.0, 200)
    tracemalloc.start()
    discharge_evaluation._get_nearest_grid_indices(axis, gauges)
    peak_mib = tracemalloc.get_traced_memory()[1] / 2**20
    tracemalloc.stop()
    # a 200 x 50,000 float64 matrix alone takes 76 MiB
    assert peak_mib < 20


@pytest.mark.parametrize("max_distance_cells", [0, -1])
def test_mean_flow_with_zero_distance_is_refused(max_distance_cells):
    """Refuse mean-flow without any neighbour cell to choose from."""
    with pytest.raises(ValueError, match="--gauge-max-distance-cells"):
        discharge_evaluation._resolve_gauge_location_method(
            "mean-flow", max_distance_cells, gauge_coordinate_file="g.nc"
        )


@pytest.mark.parametrize(
    ("method", "max_distance_cells", "sources", "expected"),
    [
        (None, None, {"facc_file": "f.nc"}, ("basinex", 3)),
        ("burek", 5, {"facc_file": "f.nc"}, ("burek", 5)),
        (None, None, {"gauge_coordinate_file": "g.nc"}, ("nearest", 1)),
        ("mean-flow", None, {"gauge_coordinate_file": "g.nc"}, ("mean-flow", 1)),
        ("mean-flow", 2, {"gauge_coordinate_file": "g.nc"}, ("mean-flow", 2)),
        (None, None, {"scc_gauges_file": "s.nc"}, (None, None)),
    ],
)
def test_gauge_location_method_defaults_follow_the_gauge_source(
    method, max_distance_cells, sources, expected
):
    """Default to basinex/3 with a facc file and nearest/1 with gauge coordinates."""
    assert (
        discharge_evaluation._resolve_gauge_location_method(
            method, max_distance_cells, **sources
        )
        == expected
    )


@pytest.mark.parametrize(
    ("method", "sources"),
    [
        ("burek", {"gauge_coordinate_file": "g.nc"}),
        ("basinex", {"scc_gauges_file": "s.nc"}),
        ("nearest", {"facc_file": "f.nc"}),
        ("mean-flow", {}),
        (None, {"facc_file": "f.nc", "gauge_coordinate_file": "g.nc"}),
        (None, {"scc_gauges_file": "s.nc", "gauge_coordinate_file": "g.nc"}),
    ],
)
def test_gauge_location_method_refuses_unfitting_combinations(method, sources):
    """Refuse a method without its gauge source and two competing sources."""
    with pytest.raises(ValueError):
        discharge_evaluation._resolve_gauge_location_method(method, None, **sources)


def test_plot_map_draws_projected_gauges_in_their_crs(
    tmp_path, _offline_map_projections
):
    """Draw gauges given in EPSG:3035 metres on an EPSG:3035 map."""
    from rasterio.crs import CRS

    results = pd.DataFrame(
        {
            "id": [1, 2, 3],
            "x": [4.0e6, 4.1e6, 4.2e6],
            "y": [3.0e6, 3.1e6, 3.2e6],
            "kge": [0.1, 0.5, 0.9],
        }
    )

    written = plot_map(results, tmp_path, variables=["kge"], crs=CRS.from_epsg(3035))

    assert written == [tmp_path / "map_kge.png"]
    assert getattr(_offline_map_projections[0], "epsg_code", None) == 3035


# -----------------------------
# Small helpers to make datasets
# -----------------------------


def make_sim_data(times, var="Qrouted", lat=None, lon=None, values=None):
    """Simulation dataset (to be cropped/select later)."""
    lat = np.array([50.0, 49.9]) if lat is None else np.asarray(lat, float)
    lon = np.array([10.0, 10.1]) if lon is None else np.asarray(lon, float)
    if values is None:
        rng = np.random.RandomState(1)
        values = rng.rand(len(times), len(lat), len(lon))
    da = xr.DataArray(
        values,
        dims=("time", "lat", "lon"),
        coords={"time": times, "lat": lat, "lon": lon},
        name=var,
    )
    ds = da.to_dataset()
    ds.attrs["which"] = "sim"  # marker for timedelta_to_alias mock
    return ds


# -----------------------------
# Leaf-level functions
# -----------------------------
class TestLowLevelHelpers(unittest.TestCase):
    def test_flatten_list(self):
        self.assertEqual(
            discharge_evaluation.flatten_list([1, [2, [3, None], 4], None, 5]),
            [1, 2, 3, 4, 5],
        )

    def test_get_region_from_id_matches_leading_digit(self):
        self.assertEqual(discharge_evaluation.get_region_from_id(1001), "Africa")
        self.assertEqual(discharge_evaluation.get_region_from_id("6042"), "Europe")

    def test_get_region_from_id_unknown_prefix(self):
        self.assertEqual(discharge_evaluation.get_region_from_id(9001), "Unknown")

    def test_gen_list_of_result_dicts_all_nan(self):
        times = pd.date_range("2000-01-01", periods=3, freq="h")
        da = xr.DataArray(
            [np.nan, np.nan, np.nan], dims=("time",), coords={"time": times}
        )
        res = discharge_evaluation.gen_list_of_result_dicts(
            da, id=123, datatype="obs", facc=42.0
        )
        self.assertIsNone(res)

    def test_gen_list_of_result_dicts_ok(self):
        times = pd.date_range("2000-01-01", periods=2, freq="h")
        da = xr.DataArray([1.0, 2.0], dims=("time",), coords={"time": times})
        res = discharge_evaluation.gen_list_of_result_dicts(da, id=7, facc=11.0)
        self.assertEqual(len(res), 2)
        self.assertEqual(res[0]["id"], 7)
        self.assertIn("facc", res[0])
        self.assertEqual(res[1]["year"], 2000)

    def test_get_grdc_for_one_gauge_empty(self):
        da = xr.DataArray([], dims=("time",), coords={"time": []})
        out = discharge_evaluation.get_grdc_for_one_gauge(1, da)
        self.assertIsNone(out)

    def test_get_grdc_for_one_gauge_drops_nans(self):
        times = pd.date_range("2000-01-01", periods=3, freq="D")
        da = xr.DataArray([1.0, np.nan, 2.0], dims=("time",), coords={"time": times})
        out = discharge_evaluation.get_grdc_for_one_gauge(1, da)
        self.assertEqual(out.size, 2)

    def test_filter_ids_by_observed_years_threshold(self):
        ids = np.array([101, 202], dtype=int)
        times = pd.date_range("2000-01-01", "2002-12-01", freq="MS")
        vals = np.full((len(times), len(ids)), np.nan, dtype=float)
        years = times.year
        vals[np.isin(years, [2000, 2001]), 0] = 1.0
        vals[years == 2001, 1] = 1.0
        obs_da = xr.DataArray(
            vals,
            dims=("time", "id"),
            coords={"time": times, "id": ids},
            name="discharge",
        )

        filtered_ids, dropped = discharge_evaluation.filter_ids_by_observed_years(
            valid_ids=ids,
            obs_discharge_data=obs_da,
            min_overlapping_years=2,
        )
        self.assertListEqual(filtered_ids.tolist(), [101])
        self.assertIn((202, 1), dropped)

    def test_filter_ids_by_observed_years_preserves_order(self):
        ids = np.array([30, 10, 20], dtype=int)
        times = pd.date_range("2000-01-01", "2001-12-01", freq="MS")
        vals = np.full((len(times), len(ids)), np.nan, dtype=float)
        years = times.year
        vals[np.isin(years, [2000, 2001]), 0] = 1.0  # id 30 passes
        vals[years == 2001, 1] = 1.0  # id 10 fails
        vals[np.isin(years, [2000, 2001]), 2] = 1.0  # id 20 passes
        obs_da = xr.DataArray(
            vals,
            dims=("time", "id"),
            coords={"time": times, "id": ids},
            name="discharge",
        )

        filtered_ids, _dropped = discharge_evaluation.filter_ids_by_observed_years(
            valid_ids=ids,
            obs_discharge_data=obs_da,
            min_overlapping_years=2,
        )
        self.assertListEqual(filtered_ids.tolist(), [30, 20])


# -----------------------------
# Q_data_to_xarray integration (no plotting)
# -----------------------------
class TestQDataToXarray(unittest.TestCase):
    def setUp(self):
        # 2 gauges in a tiny spatial domain
        self.ids = [101, 202]
        self.xs = [10.0, 10.1]
        self.ys = [50.0, 49.9]
        self.facc = [1000.0, 1500.0]

        # observed hourly, sim 3-hourly (mismatched temporal resolution)
        self.obs_times = pd.date_range("2001-01-01", periods=6, freq="h")
        self.sim_times = pd.date_range("2001-01-01", periods=4, freq="3h")
        # datasets for mocked file reads
        self.obs_ds = make_observed_data(
            self.ids, self.obs_times, self.xs, self.ys, self.facc, var="runoff_mean_mm"
        )
        self.sim_ds = make_sim_data(self.sim_times, var="Qrouted")
        # Add id dimension so downstream selection by id is valid
        if "id" not in self.sim_ds.dims:
            self.sim_ds = self.sim_ds.expand_dims(id=np.array(self.ids, dtype=int))

    def _mock_overlapping_time_slice(self, sim_ds, obs_ds):
        t0 = max(sim_ds.time.min().item(), obs_ds.time.min().item())
        t1 = min(sim_ds.time.max().item(), obs_ds.time.max().item())
        return slice(np.datetime64(t0), np.datetime64(t1))

    def test_q_data_to_xarray_resamples_and_builds_datasets(self):
        with tempfile.TemporaryDirectory() as td:
            td = Path(td)

            def gxarr_side_effect(path, **kwargs):
                path = Path(path)
                if path.name == "obs.nc":
                    return cm_enter(self.obs_ds)
                if path.name == "sim.nc":
                    return cm_enter(self.sim_ds)
                if path.name == "restart.nc":
                    return cm_enter(xr.Dataset())  # dummy restart
                raise FileNotFoundError(path)

            # Patch pieces that hit filesystem or heavy logic
            with patch.object(
                discharge_evaluation, "load_ds", side_effect=gxarr_side_effect
            ), patch.object(
                discharge_evaluation,
                "get_gauge_coords",
                side_effect=[(10.0, 50.0, 1000.0), (10.1, 49.9, 1500.0)],
            ), patch.object(
                discharge_evaluation, "get_sim_data_for_one_gauge"
            ) as mock_sim_for_gauge, patch.object(
                discharge_evaluation, "write_xarray_to_file"
            ) as mock_write:
                # Make get_sim_data_for_one_gauge return per-id DA with time dimension
                def _sim_for_gauge(
                    id, index, sim_data, yarr, xarr, resolution, **kwargs
                ):
                    sim_times = pd.to_datetime(np.asarray(sim_data.time.values))
                    da = xr.DataArray(
                        np.arange(sim_times.size) + index,
                        dims=("time",),
                        coords={"time": sim_times},
                        name="discharge",
                    )
                    return da.expand_dims(dim={"id": [int(id)]})

                mock_sim_for_gauge.side_effect = _sim_for_gauge

                obs_out, sim_out = discharge_evaluation.Q_data_to_xarray(
                    model_data_path="sim.nc",
                    observed_data_path="obs.nc",
                    facc_file="restart.nc",
                    sim_variable="Qrouted",
                    observed_variable="runoff_mean_mm",
                    model_keyword="mrm",
                    saving_path=td,
                    lon_min=9.95,
                    lon_max=10.15,
                    lat_min=49.85,
                    lat_max=50.05,
                    resolution=0.1,
                    n_jobs=1,
                    date_slice=None,
                    overwrite=True,  # force writing, skip file existence branch
                    direct_comparison=True,  # apply overlap slice
                    write_input_data_cache=True,
                )

                # Observed output dataset shape and coords
                self.assertIn("discharge", obs_out)
                self.assertIn("facc", obs_out)
                self.assertListEqual(sorted(list(obs_out.dims)), ["id", "time"])
                self.assertEqual(len(obs_out["id"]), 2)

                # Simulation output dataset
                self.assertIn("discharge", sim_out)
                self.assertIn("facc", sim_out)
                self.assertIn("x", sim_out)
                self.assertIn("y", sim_out)
                self.assertEqual(len(sim_out["id"]), 2)
                self.assertIn("time", sim_out["discharge"].dims)

                # write_xarray_to_file called for obs & sim
                self.assertGreaterEqual(mock_write.call_count, 2)

                # Ensure temporal overlap was respected: times within intersection
                tmin = max(self.obs_times.min(), self.sim_times.min())
                tmax = min(self.obs_times.max(), self.sim_times.max())
                self.assertGreaterEqual(
                    pd.Timestamp(obs_out.time.min().values), pd.Timestamp(tmin)
                )
                self.assertLessEqual(
                    pd.Timestamp(obs_out.time.max().values), pd.Timestamp(tmax)
                )
                self.assertGreaterEqual(
                    pd.Timestamp(sim_out.time.min().values), pd.Timestamp(tmin)
                )
                self.assertLessEqual(
                    pd.Timestamp(sim_out.time.max().values), pd.Timestamp(tmax)
                )


# -----------------------------
# evaluate_grdc_data: direct comparison path
# -----------------------------
class TestEvaluateDirect(unittest.TestCase):
    def test_evaludate_grdc_data_direct_produces_results_csv(self):
        with tempfile.TemporaryDirectory() as td:
            outdir = Path(td)

            # Build very small ready-made datasets that Q_data_to_xarray returns
            ids = [42]
            times = pd.date_range("2005-01-01", periods=3, freq="D")
            obs_da = xr.DataArray(
                [1.0, 2.0, 3.0],
                dims=("time",),
                coords={"time": times},
                name="discharge",
            ).expand_dims(id=[42])
            sim_da = xr.DataArray(
                [1.0, 2.0, 3.0],
                dims=("time",),
                coords={"time": times},
                name="discharge",
            ).expand_dims(id=[42])

            observed_out = xr.Dataset({"discharge": obs_da})
            model_out = xr.Dataset(
                {
                    "discharge": sim_da,
                    "facc": (("id",), np.array([1000.0])),
                    "x": (("id",), np.array([10.0])),
                    "y": (("id",), np.array([50.0])),
                },
                coords={"id": [42]},
            )

            # Q_data_to_xarray returns these directly (bypass all I/O/Parallel)
            with patch.object(
                discharge_evaluation,
                "Q_data_to_xarray",
                return_value=(observed_out, model_out),
            ), patch.object(
                discharge_evaluation,
                "gen_hydrograph_by_data_sets",
                return_value={"id": 42, "alpha": 1.0, "beta": 1.0, "gamma": 0.5},
            ), patch.object(
                discharge_evaluation, "plot_cdf"
            ) as mock_plot, patch.object(
                discharge_evaluation, "plot_map"
            ):

                discharge_evaluation.evaludate_discharge_data(
                    model_data_path="sim.nc",
                    observed_data_path="obs.nc",
                    facc_file="restart.nc",
                    output_path=outdir,
                    n_jobs=1,
                    direct_comparison=True,  # direct path
                    n_boostrap_selections=0,  # disable bootstrap
                    overwrite=True,
                )

                # results.csv exists and has expected content
                results_csv = outdir / "results.csv"
                self.assertTrue(results_csv.is_file())
                df = pd.read_csv(results_csv, index_col=0)
                self.assertIn("alpha", df.columns)
                self.assertEqual(df.loc[0, "id"], 42)
                # plot_cdf not called for small sample sizes
                mock_plot.assert_not_called()


# -----------------------------
# evaluate_grdc_data: bootstrap path
# -----------------------------
class TestEvaluateBootstrap(unittest.TestCase):
    def test_evaludate_grdc_data_bootstrap(self):
        with tempfile.TemporaryDirectory() as td:
            outdir = Path(td)

            # Two years, one id; values: deterministic
            ids = [7]
            t = pd.date_range(
                "2010-01-01", periods=730, freq="D"
            )  # 2 years (non-leap + leap ok)
            obs_vals = np.ones((len(t), 1))
            sim_vals = 2.0 * np.ones((len(t), 1))
            obs_ds = xr.Dataset(
                {"discharge": (("time", "id"), obs_vals)}, coords={"time": t, "id": ids}
            )
            sim_ds = xr.Dataset(
                {
                    "discharge": (("time", "id"), sim_vals),
                    "facc": (("id",), np.array([1234.0])),
                    "x": (("id",), np.array([11.0])),
                    "y": (("id",), np.array([49.0])),
                },
                coords={"time": t, "id": ids},
            )

            with patch.object(
                discharge_evaluation, "Q_data_to_xarray", return_value=(obs_ds, sim_ds)
            ), patch.object(
                discharge_evaluation,
                "gen_hydrograph_by_data_sets",
                return_value={"id": 7, "alpha": 2.0, "beta": 0.0, "gamma": 0.0},
            ), patch.object(
                discharge_evaluation,
                "boostap_statistics",
                return_value={
                    "index": 0,
                    "id": 7,
                    "alpha": 2.0,
                    "beta": 0.0,
                    "gamma": 0.7,
                },
            ), patch.object(
                discharge_evaluation, "plot_cdf"
            ) as mock_plot, patch.object(
                discharge_evaluation, "plot_map"
            ):

                discharge_evaluation.evaludate_discharge_data(
                    model_data_path="sim.nc",
                    observed_data_path="obs.nc",
                    facc_file="restart.nc",
                    output_path=outdir,
                    n_jobs=1,
                    direct_comparison=False,  # take bootstrap branch
                    n_bootstrap_years=1,
                    n_boostrap_selections=3,  # >0 triggers bootstrap
                    overwrite=True,
                )

                # results.csv exists and has bootstrap content (from boostap_statistics)
                results_csv = outdir / "results.csv"
                self.assertTrue(results_csv.is_file())
                df = pd.read_csv(results_csv, index_col=0)
                self.assertIn("gamma", df.columns)
                self.assertTrue((df["id"] == 7).all())
                # plot_cdf not called for small sample sizes
                mock_plot.assert_not_called()


def cm_enter(ds):
    """Build a context manager object returning ds on __enter__()."""
    cm = MagicMock()
    cm.__enter__.return_value = ds
    cm.__exit__.return_value = False
    return cm


def make_observed_data(ids, times, xs, ys, facc, var="runoff_mean_mm", values=None):
    if values is None:
        values = np.arange(len(times) * len(ids)).reshape(len(times), len(ids))
    da = xr.DataArray(
        values,
        dims=("time", "id"),
        coords={"time": times, "id": np.array(ids, dtype=int)},
        name=var,
    )
    ds = xr.Dataset(
        {
            "geo_x": (("id",), np.array(xs, dtype=float)),
            "geo_y": (("id",), np.array(ys, dtype=float)),
            "area": (("id",), np.array(facc, dtype=float)),
            var: da,
        },
        coords={"id": np.array(ids, dtype=int), "time": times},
    )
    return ds


def make_scc_gauges(ids, xs, ys):
    return xr.Dataset(
        {
            "lon": (("station",), np.array(xs, dtype=float)),
            "lat": (("station",), np.array(ys, dtype=float)),
            "station": (("station",), np.array(ids, dtype=int)),
        }
    )


def write_discharge_file(path, gid, times, obs_vals, sim_vals):
    gid_str = f"{int(gid):010d}"
    ds = xr.Dataset(
        {
            f"Qsim_{gid_str}": (("time",), np.array(sim_vals, dtype=float)),
            f"Qobs_{gid_str}": (("time",), np.array(obs_vals, dtype=float)),
        },
        coords={"time": times},
    )
    ds.to_netcdf(path)


def assert_discharge_equal(ds_a, ds_b):
    da_a = ds_a["discharge"].sortby("id")
    da_b = ds_b["discharge"].sortby("id")
    if {"time", "id"}.issubset(set(da_a.dims)):
        da_a = da_a.transpose("time", "id")
    if {"time", "id"}.issubset(set(da_b.dims)):
        da_b = da_b.transpose("time", "id")
    xr.testing.assert_allclose(da_a, da_b)


class TestGRDCValidation2Consistency(unittest.TestCase):
    def test_q_data_to_xarray_consistent_across_inputs(self):
        ids = [1, 2]
        xs = [10.0, 10.1]
        ys = [50.0, 49.9]
        facc = [100.0, 200.0]
        times = pd.date_range("2001-01-01", periods=4, freq="D")
        date_slice = slice(times[1], times[-1])

        obs_vals = np.array([[1.0, 2.0], [1.5, 2.5], [2.0, 3.0], [2.5, 3.5]])
        sim_vals = np.array([[10.0, 20.0], [10.5, 20.5], [11.0, 21.0], [11.5, 21.5]])

        obs_ds = make_observed_data(
            ids, times, xs, ys, facc, var="runoff_mean_mm", values=obs_vals
        )
        sim_ds = xr.Dataset(
            {
                "Qrouted": (
                    ("time", "id"),
                    np.zeros((len(times), len(ids)), dtype=float),
                )
            },
            coords={
                "time": times,
                "id": np.array(ids, dtype=int),
            },
        )
        scc_ds = make_scc_gauges(ids, xs, ys)

        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            discharge_root = td / "discharge_inputs"
            discharge_root.mkdir()
            for gid, i in zip(ids, range(len(ids))):
                sub = discharge_root / f"gauge_{gid}"
                sub.mkdir()
                write_discharge_file(
                    sub / "discharge.nc",
                    gid,
                    times,
                    obs_vals[:, i],
                    sim_vals[:, i],
                )

            def load_ds_discharge(path, file_name=None):
                ds = xr.open_dataset(path)
                return cm_enter(ds)

            with patch.object(
                discharge_evaluation, "load_ds", side_effect=load_ds_discharge
            ), patch.object(discharge_evaluation, "write_xarray_to_file"):
                obs_d, sim_d = discharge_evaluation.Q_data_to_xarray(
                    model_data_path=discharge_root,
                    observed_data_path="obs.nc",
                    sim_variable=None,
                    observed_variable=None,
                    model_keyword="mrm",
                    saving_path=td,
                    model_file_name="discharge.nc",
                    date_slice=date_slice,
                    overwrite=True,
                    direct_comparison=False,
                )

            def load_ds_side_effect(path, file_name=None):
                path = Path(path)
                if path.name == "obs.nc":
                    return cm_enter(obs_ds)
                if path.name == "sim.nc":
                    return cm_enter(sim_ds)
                if path.name == "scc.nc":
                    return cm_enter(scc_ds)
                if path.name == "restart.nc":
                    return cm_enter(xr.Dataset())
                raise FileNotFoundError(path)

            def sim_from_nodes(*args, **kwargs):
                da = xr.DataArray(
                    sim_vals,
                    dims=("time", "id"),
                    coords={"time": times, "id": np.array(ids, dtype=int)},
                    name="discharge",
                )
                da = da.sel(time=date_slice)
                return da, np.array(xs), np.array(ys), np.array(ids, dtype=int)

            def sim_for_gauge(id, index, sim_data, yarr, xarr, resolution, **kwargs):
                da = xr.DataArray(
                    sim_vals[:, index],
                    dims=("time",),
                    coords={"time": times},
                    name="discharge",
                )
                da = da.sel(time=date_slice)
                return da.expand_dims(dim={"id": [int(id)]})

            with patch.object(
                discharge_evaluation, "load_ds", side_effect=load_ds_side_effect
            ), patch.object(
                discharge_evaluation,
                "resample_to_coarser_calendar",
                return_value=(sim_ds, obs_ds),
            ), patch.object(
                discharge_evaluation,
                "get_sim_data_for_gauges_from_nodes",
                side_effect=sim_from_nodes,
            ), patch.object(
                discharge_evaluation, "write_xarray_to_file"
            ):
                obs_s, sim_s = discharge_evaluation.Q_data_to_xarray(
                    model_data_path="sim.nc",
                    observed_data_path="obs.nc",
                    scc_gauges_file="scc.nc",
                    sim_variable="Qrouted",
                    observed_variable="runoff_mean_mm",
                    model_keyword="mrm",
                    saving_path=td,
                    model_file_name="mrm_output.nc",
                    date_slice=date_slice,
                    overwrite=True,
                    direct_comparison=False,
                )

            with patch.object(
                discharge_evaluation, "load_ds", side_effect=load_ds_side_effect
            ), patch.object(
                discharge_evaluation,
                "resample_to_coarser_calendar",
                return_value=(sim_ds, obs_ds),
            ), patch.object(
                discharge_evaluation,
                "get_gauge_coords",
                side_effect=[(xs[0], ys[0], facc[0]), (xs[1], ys[1], facc[1])],
            ), patch.object(
                discharge_evaluation,
                "get_sim_data_for_one_gauge",
                side_effect=sim_for_gauge,
            ), patch.object(
                discharge_evaluation, "write_xarray_to_file"
            ):
                obs_g, sim_g = discharge_evaluation.Q_data_to_xarray(
                    model_data_path="sim.nc",
                    observed_data_path="obs.nc",
                    facc_file="restart.nc",
                    sim_variable="Qrouted",
                    observed_variable="runoff_mean_mm",
                    model_keyword="mrm",
                    saving_path=td,
                    model_file_name="mrm_output.nc",
                    date_slice=date_slice,
                    overwrite=True,
                    direct_comparison=False,
                )

            assert_discharge_equal(obs_d, obs_s)
            assert_discharge_equal(obs_d, obs_g)
            assert_discharge_equal(sim_d, sim_s)
            assert_discharge_equal(sim_d, sim_g)


class TestGaugeLocationSources(unittest.TestCase):
    """Gauge location sources of Q_data_to_xarray: facc file, scc file or none."""

    ids = [1, 2]
    times = pd.date_range("2001-01-01", periods=4, freq="D")
    sim_values = np.array([[10.0, 20.0], [10.5, 20.5], [11.0, 21.0], [11.5, 21.5]])

    def setUp(self):
        """Create the observed and simulated datasets of two gauges."""
        self.observed = make_observed_data(
            self.ids, self.times, [10.0, 10.1], [50.0, 49.9], [100.0, 200.0]
        )
        self.simulated = xr.Dataset(
            {"Qrouted": (("time", "id"), self.sim_values)},
            coords={"time": self.times, "id": np.array(self.ids, dtype=int)},
        )

    def create_sim_discharge(self, gauge_ids):
        """Create the simulated discharge of the given gauges along id.

        Args:
            gauge_ids (array_like): Ids of gauges 1 and/or 2.

        Returns
        -------
            xr.DataArray: Discharge with dims (time, id).
        """
        columns = [self.ids.index(int(gauge_id)) for gauge_id in gauge_ids]
        return xr.DataArray(
            self.sim_values[:, columns],
            dims=("time", "id"),
            coords={"time": self.times, "id": np.asarray(gauge_ids, dtype=int)},
            name="discharge",
        )

    def run_q_data_to_xarray(self, datasets, patches, **kwargs):
        """Run Q_data_to_xarray on mocked files with additional patches.

        Args:
            datasets (dict): Dataset served by load_ds per file name.
            patches (dict): Further attribute name -> replacement on the module.
            **kwargs: Gauge location arguments of Q_data_to_xarray.

        Returns
        -------
            tuple: The observed and simulated datasets.
        """

        def load_ds_side_effect(path, file_name=None):
            return cm_enter(datasets[Path(path).name])

        with tempfile.TemporaryDirectory() as temp_dir, contextlib.ExitStack() as stack:
            stack.enter_context(
                patch.object(
                    discharge_evaluation, "load_ds", side_effect=load_ds_side_effect
                )
            )
            stack.enter_context(
                patch.object(
                    discharge_evaluation,
                    "resample_to_coarser_calendar",
                    return_value=(self.simulated, self.observed),
                )
            )
            stack.enter_context(
                patch.object(discharge_evaluation, "write_xarray_to_file")
            )
            for name, replacement in patches.items():
                stack.enter_context(
                    patch.object(discharge_evaluation, name, replacement)
                )
            return discharge_evaluation.Q_data_to_xarray(
                model_data_path="sim.nc",
                observed_data_path="obs.nc",
                sim_variable="Qrouted",
                observed_variable="runoff_mean_mm",
                model_keyword="mrm",
                saving_path=Path(temp_dir),
                model_file_name="mrm_output.nc",
                date_slice=slice(None, None),
                overwrite=True,
                direct_comparison=False,
                **kwargs,
            )

    def test_facc_file_relocates_gauges_with_basinex_within_three_cells(self):
        """Relocate gauges by flow accumulation with basinex and 3 cells by default."""
        gauge_coords = MagicMock(side_effect=[(10.0, 50.0, 100.0), (10.1, 49.9, 200.0)])

        def sim_for_gauge(id, index, sim_data, yarr, xarr, resolution, **kwargs):
            return self.create_sim_discharge([id])

        self.run_q_data_to_xarray(
            {
                "obs.nc": self.observed,
                "sim.nc": self.simulated,
                "restart.nc": xr.Dataset(),
            },
            {
                "get_gauge_coords": gauge_coords,
                "get_sim_data_for_one_gauge": MagicMock(side_effect=sim_for_gauge),
            },
            facc_file="restart.nc",
        )

        self.assertEqual(gauge_coords.call_count, len(self.ids))
        for call in gauge_coords.call_args_list:
            self.assertEqual(call.kwargs["method"], "basinex")
            self.assertEqual(call.kwargs["max_distance_cells"], 3)

    def test_missing_gauge_location_source_is_refused(self):
        """Refuse a run that has no way to locate the gauges."""
        with self.assertRaisesRegex(ValueError, "Neither"):
            self.run_q_data_to_xarray(
                {"obs.nc": self.observed, "sim.nc": self.simulated}, {}
            )

    def test_scc_gauges_without_lon_lat_use_x_y_of_reference_ids(self):
        """Take gauge x/y from an scc file without lon/lat, for reference ids only."""
        # station 3 has no observations, x/y differ from the GRDC geo_x/geo_y
        scc_gauges = xr.Dataset(
            {
                "station": (("station",), np.array([1, 2, 3])),
                "x": (("station",), np.array([4000.0, 5000.0, 6000.0])),
                "y": (("station",), np.array([3000.0, 3100.0, 3200.0])),
            }
        )
        node_extraction = MagicMock(
            side_effect=lambda **kwargs: (
                self.create_sim_discharge(kwargs["gauge_ids"]),
                kwargs["x_new"],
                kwargs["y_new"],
                kwargs["gauge_ids"],
            )
        )

        self.run_q_data_to_xarray(
            {"obs.nc": self.observed, "sim.nc": self.simulated, "scc.nc": scc_gauges},
            {"get_sim_data_for_gauges_from_nodes": node_extraction},
            scc_gauges_file="scc.nc",
        )

        call_kwargs = node_extraction.call_args.kwargs
        np.testing.assert_array_equal(call_kwargs["gauge_ids"], [1, 2])
        np.testing.assert_array_equal(call_kwargs["x_new"], [4000.0, 5000.0])
        np.testing.assert_array_equal(call_kwargs["y_new"], [3000.0, 3100.0])

    def create_gridded_files(self):
        """Create gridded output of 2 x 2 cells and gauge coordinates on it.

        Cell (row 0, column 0) holds 100, (0, 1) 3, (1, 0) 50 and (1, 1) 4 on
        every day. Gauge 1 lies in cell (0, 0), gauge 2 in cell (1, 1); their
        observed means are 3 and 4.

        Returns
        -------
            dict: Datasets served by load_ds per file name.
        """
        cell_values = np.array([[100.0, 3.0], [50.0, 4.0]])
        gridded = xr.Dataset(
            {
                "Qrouted": (
                    ("time", "y", "x"),
                    np.broadcast_to(cell_values, (len(self.times), 2, 2)).copy(),
                )
            },
            coords={
                "time": self.times,
                "y": np.array([3500.0, 2500.0]),
                "x": np.array([500.0, 1500.0]),
            },
        )
        gauges = xr.Dataset(
            {
                "station": (("station",), np.array([1, 2])),
                "x": (("station",), np.array([500.0, 1500.0])),
                "y": (("station",), np.array([3500.0, 2500.0])),
            }
        )
        return {"obs.nc": self.observed, "sim.nc": gridded, "gauges.nc": gauges}

    def test_gauge_coordinate_file_reads_the_nearest_cells_of_gridded_output(self):
        """Read every gauge from the grid cell its coordinates lie in."""
        _observed, simulated = self.run_q_data_to_xarray(
            self.create_gridded_files(), {}, gauge_coordinate_file="gauges.nc"
        )
        np.testing.assert_array_equal(
            simulated["discharge"].sel(id=1).values, [100.0] * 4
        )
        np.testing.assert_array_equal(
            simulated["discharge"].sel(id=2).values, [4.0] * 4
        )

    def test_mean_flow_reads_the_neighbour_cell_matching_the_observed_mean(self):
        """Move gauge 1 to the neighbour cell whose mean matches its observations."""
        _observed, simulated = self.run_q_data_to_xarray(
            self.create_gridded_files(),
            {},
            gauge_coordinate_file="gauges.nc",
            gauge_location_method="mean-flow",
        )
        np.testing.assert_array_equal(
            simulated["discharge"].sel(id=1).values, [3.0] * 4
        )
        np.testing.assert_array_equal(
            simulated["discharge"].sel(id=2).values, [4.0] * 4
        )
