"""Tests for the discharge evaluation: unit handling and gauge-to-node matching."""

import logging

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
    np.testing.assert_array_equal(sim_sel["candidate_dy"], [-1, -1, -1, 0, 0, 0, 1, 1, 1])
    np.testing.assert_array_equal(sim_sel["candidate_dx"], [-1, 0, 1, -1, 0, 1, -1, 0, 1])


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
            "candidate_x": (("id", "candidate"), [[1000.0 * (1 + dx) for _, dx in offsets]]),
            "candidate_y": (("id", "candidate"), [[1000.0 * (1 - dy) for dy, _ in offsets]]),
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
