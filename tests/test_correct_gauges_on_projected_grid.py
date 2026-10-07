"""Tests for the example script correcting gauges on a projected grid."""

import importlib.util
import logging
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
import xarray as xr
from pyproj import CRS, Transformer
from shapely.geometry import Polygon
from shapely.ops import transform

SCRIPT_FILE = (
    Path(__file__).resolve().parents[1]
    / "examples"
    / "02_correct_gauges_on_projected_grid.py"
)
GRID_CRS = CRS.from_epsg(3035)


@pytest.fixture(scope="module")
def script():
    """Load the example script, whose file name is no valid module name.

    Returns:
        The loaded script module.
    """
    spec = importlib.util.spec_from_file_location(
        "correct_gauges_on_projected_grid", SCRIPT_FILE
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_read_projected_gauges_casts_ids_and_drops_gauges_outside(
    script, tmp_path, caplog
):
    """Project the gauges with integer ids and drop the ones outside the grid."""
    scc_gauges_file = tmp_path / "scc_gauges.nc"
    xr.Dataset(
        {"lon": ("station", [10.0, -60.0]), "lat": ("station", [50.0, -10.0])},
        coords={"station": [6335020.0, 3618000.0]},
    ).to_netcdf(scc_gauges_file)
    gauge_x, gauge_y = Transformer.from_crs(
        "EPSG:4326", GRID_CRS, always_xy=True
    ).transform(10.0, 50.0)
    grid_bounds = (gauge_x - 1000, gauge_y - 1000, gauge_x + 1000, gauge_y + 1000)

    with caplog.at_level(logging.WARNING):
        gauges = script.read_projected_gauges(scc_gauges_file, GRID_CRS, grid_bounds)

    assert gauges["id"].tolist() == [6335020]
    assert gauges["id"].dtype == np.int64
    np.testing.assert_allclose(gauges[["x", "y"]].to_numpy(), [[gauge_x, gauge_y]])
    assert "Dropping gauge 3618000: it lies outside the fdir extent." in caplog.text


def test_write_projected_reference_shapes_round_trips_and_drops_missing(
    script, tmp_path, caplog
):
    """Write a shape in the grid CRS that projects back onto the original."""
    shape_dir = tmp_path / "shapes"
    shape_dir.mkdir()
    # an L shaped catchment of a few km, where any misplaced vertex shows
    catchment_shape = Polygon(
        [
            (10.00, 50.00),
            (10.06, 50.00),
            (10.06, 50.02),
            (10.02, 50.02),
            (10.02, 50.05),
            (10.00, 50.05),
        ]
    )
    shape_file = shape_dir / "basin_6335020.shp"
    gpd.GeoDataFrame(geometry=[catchment_shape], crs="EPSG:4326").to_file(shape_file)
    output_dir = tmp_path / "reference_shapes"

    with caplog.at_level(logging.WARNING):
        gauge_ids = script.write_projected_reference_shapes(
            [6335020, 6335030], shape_dir, GRID_CRS, output_dir
        )

    assert gauge_ids == [6335020]
    assert "Dropping gauge 6335030: no shapefile" in caplog.text
    original_shape = gpd.read_file(shape_file).geometry.iloc[0]
    projected = gpd.read_file(output_dir / shape_file.name)
    assert projected.crs.to_epsg() == 3035
    # normalizing ignores a ring orientation or start vertex changed on writing
    to_grid = Transformer.from_crs("EPSG:4326", GRID_CRS, always_xy=True)
    expected_shape = transform(to_grid.transform, original_shape)
    assert (
        projected.geometry.iloc[0]
        .normalize()
        .equals_exact(expected_shape.normalize(), tolerance=1e-3)
    )
    round_trip_shape = projected.to_crs("EPSG:4326").geometry.iloc[0]
    assert round_trip_shape.normalize().equals_exact(
        original_shape.normalize(), tolerance=1e-8
    )


def test_write_merged_gauges_info_drops_uncorrected_gauges(script, tmp_path, caplog):
    """Merge the per gauge files and drop a gauge left uncorrected."""
    gauge_dirs = []
    for gauge_id, method in ((6335020, "shape_iou"), (6335030, None)):
        gauge_dir = tmp_path / str(gauge_id)
        gauge_dir.mkdir()
        pd.DataFrame(
            {
                "id": [gauge_id],
                "lon": [4_000_900.0],
                "lat": [3_000_700.0],
                "distance": [0.2],
                "method": [method],
            }
        ).to_csv(gauge_dir / "gauges_info.csv", index=False)
        gauge_dirs.append(gauge_dir)
    gauges_info_file = tmp_path / "gauges_info.csv"

    with caplog.at_level(logging.WARNING):
        gauges_info = script.write_merged_gauges_info(gauge_dirs, gauges_info_file)

    assert gauges_info["id"].tolist() == [6335020]
    assert pd.read_csv(gauges_info_file)["id"].tolist() == [6335020]
    assert "Dropping gauge 6335030: create_catchment left it uncorrected." in (
        caplog.text
    )


def test_write_corrected_scc_gauges_file_replaces_lon_lat_by_x_y(script, tmp_path):
    """Keep the matched stations with projected x/y and copy everything else."""
    scc_gauges_file = tmp_path / "scc_gauges.nc"
    xr.Dataset(
        {
            "lon": ("station", [10.0, 11.0, 12.0]),
            "lat": ("station", [50.0, 51.0, 52.0]),
            "area": ("station", [1.4, 2.0, 3.0]),
        },
        coords={"station": [6335020.0, 6335030.0, 6335040.0]},
        attrs={"title": "SCC gauges specification"},
    ).to_netcdf(scc_gauges_file)
    # listed in another order than the file, whose station order is kept
    gauges_info = pd.DataFrame(
        {
            "id": [6335040, 6335020],
            "lon": [4_100_000.0, 4_000_000.0],
            "lat": [3_100_000.0, 3_000_000.0],
        }
    )
    corrected_scc_file = tmp_path / "scc_gauges_epsg3035.nc"

    script.write_corrected_scc_gauges_file(
        scc_gauges_file, gauges_info, GRID_CRS, corrected_scc_file
    )

    with xr.open_dataset(corrected_scc_file) as corrected_ds:
        np.testing.assert_array_equal(
            corrected_ds["station"].values, [6335020.0, 6335040.0]
        )
        assert corrected_ds["station"].dtype == np.float64
        assert "lon" not in corrected_ds.variables
        assert "lat" not in corrected_ds.variables
        np.testing.assert_array_equal(
            corrected_ds["x"].values, [4_000_000.0, 4_100_000.0]
        )
        np.testing.assert_array_equal(
            corrected_ds["y"].values, [3_000_000.0, 3_100_000.0]
        )
        assert corrected_ds["x"].attrs["units"] == "m"
        assert corrected_ds["x"].attrs["grid_mapping"] == "crs"
        assert (
            corrected_ds["crs"].attrs["grid_mapping_name"]
            == "lambert_azimuthal_equal_area"
        )
        np.testing.assert_array_equal(corrected_ds["area"].values, [1.4, 3.0])
        assert corrected_ds.attrs["title"] == "SCC gauges specification"
