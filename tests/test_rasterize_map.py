"""Tests for rasterizing vector maps on a reference DEM grid."""

from pathlib import Path

import geopandas as gpd
import numpy as np
import pytest
import rasterio
from click.testing import CliRunner
from mhm_tools._cli._main import cli
from mhm_tools.pre import rasterize_map as rasterize_map_module
from mhm_tools.pre.rasterize_map import rasterize_map_data
from shapely import geometry as shapely_geometry


def _write_dem(path: Path) -> None:
    transform = rasterio.transform.from_origin(100.0, 220.0, 10.0, 10.0)
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        width=3,
        height=2,
        count=1,
        dtype="float32",
        crs="EPSG:32632",
        transform=transform,
        nodata=-9999.0,
    ) as dataset:
        dataset.write(np.zeros((2, 3), dtype=np.float32), 1)


def _write_vector(path: Path, values) -> None:
    box = shapely_geometry.box
    frame = gpd.GeoDataFrame(
        {"map_code": values},
        geometry=[
            box(100.0, 210.0, 110.0, 220.0),
            box(110.0, 200.0, 130.0, 220.0),
        ],
        crs="EPSG:32632",
    )
    frame.to_file(path, driver="GPKG")


def test_rasterize_map_uses_exact_dem_grid(tmp_path: Path):
    """Output CRS, transform, shape, and extent come exactly from the DEM."""
    dem_file = tmp_path / "dem.tif"
    vector_file = tmp_path / "soil.gpkg"
    output_file = tmp_path / "soil_raw.tif"
    _write_dem(dem_file)
    _write_vector(vector_file, [5, 9])

    result = rasterize_map_data(vector_file, dem_file, output_file, "map_code")

    assert result == output_file
    with rasterio.open(dem_file) as dem, rasterio.open(output_file) as output:
        assert output.crs == dem.crs
        assert output.transform == dem.transform
        assert output.bounds == dem.bounds
        assert (output.height, output.width) == (dem.height, dem.width)
        assert output.count == 1
        assert output.dtypes == ("int32",)
        assert output.nodata == -9999
        np.testing.assert_array_equal(
            output.read(1),
            np.array([[5, 9, 9], [-9999, 9, 9]], dtype=np.int32),
        )


def test_rasterize_map_does_not_prevalidate_geometry(tmp_path: Path):
    """Topologically invalid geometry is passed directly to Rasterio."""
    dem_file = tmp_path / "dem.tif"
    vector_file = tmp_path / "soil.gpkg"
    output_file = tmp_path / "soil_raw.tif"
    _write_dem(dem_file)
    geometry = shapely_geometry.Polygon(
        [
            (100.0, 200.0),
            (130.0, 220.0),
            (130.0, 200.0),
            (100.0, 220.0),
            (100.0, 200.0),
        ]
    )
    assert not geometry.is_valid
    gpd.GeoDataFrame(
        {"map_code": [7]}, geometry=[geometry], crs="EPSG:32632"
    ).to_file(vector_file, driver="GPKG")

    rasterize_map_data(vector_file, dem_file, output_file, "map_code")

    with rasterio.open(output_file) as output:
        assert np.any(output.read(1) == 7)


def test_rasterize_map_rejects_non_integral_mapping_values(tmp_path: Path):
    """Categorical values must be integral before they are burned."""
    dem_file = tmp_path / "dem.tif"
    vector_file = tmp_path / "soil.gpkg"
    _write_dem(dem_file)
    _write_vector(vector_file, [1.5, 2.0])

    with pytest.raises(ValueError, match="integral values"):
        rasterize_map_data(
            vector_file,
            dem_file,
            tmp_path / "soil_raw.tif",
            "map_code",
        )


def test_rasterize_map_applies_lookup_to_text_categories(tmp_path: Path):
    """Lookup-aware rasterization burns SOIL_CLASS for textual vector keys."""
    dem_file = tmp_path / "dem.tif"
    vector_file = tmp_path / "soil.gpkg"
    lookup_file = tmp_path / "lookup.gpkg"
    output_file = tmp_path / "soil_class.tif"
    _write_dem(dem_file)
    _write_vector(vector_file, ["loam", "clay"])
    gpd.GeoDataFrame(
        {
            "Map code": ["loam", None, "clay"],
            "SOIL_CLASS": [4, 4, 8],
        },
        geometry=[
            shapely_geometry.Point(0, 0),
            shapely_geometry.Point(1, 1),
            shapely_geometry.Point(2, 2),
        ],
        crs="EPSG:4326",
    ).to_file(lookup_file, driver="GPKG")

    rasterize_map_data(
        vector_file,
        dem_file,
        output_file,
        "map code",
        lookup_table=lookup_file,
        lookup_mapping_field="map_code",
    )

    with rasterio.open(output_file) as output:
        np.testing.assert_array_equal(
            output.read(1),
            np.array([[4, 8, 8], [-9999, 8, 8]], dtype=np.int32),
        )


def test_rasterize_map_normalizes_integral_text_and_numeric_keys(tmp_path: Path):
    """Numeric-looking text keys retain the matching behavior of pymhm."""
    dem_file = tmp_path / "dem.tif"
    vector_file = tmp_path / "soil.gpkg"
    lookup_file = tmp_path / "lookup.gpkg"
    output_file = tmp_path / "soil_class.tif"
    _write_dem(dem_file)
    _write_vector(vector_file, ["1.0", "2"])
    gpd.GeoDataFrame(
        {"map_code": [1, 2], "SOIL_CLASS": [3, 6]},
        geometry=[
            shapely_geometry.Point(0, 0),
            shapely_geometry.Point(1, 1),
        ],
        crs="EPSG:4326",
    ).to_file(lookup_file, driver="GPKG")

    rasterize_map_data(
        vector_file,
        dem_file,
        output_file,
        "map_code",
        lookup_table=lookup_file,
    )

    with rasterio.open(output_file) as output:
        np.testing.assert_array_equal(
            output.read(1),
            np.array([[3, 6, 6], [-9999, 6, 6]], dtype=np.int32),
        )


def test_rasterize_map_cli_registration_and_short_options(monkeypatch):
    """The grouped command accepts its single-dash option forms."""
    captured = {}

    def fake_rasterize_map_data(**kwargs):
        captured.update(kwargs)

    monkeypatch.setattr(
        rasterize_map_module,
        "rasterize_map_data",
        fake_rasterize_map_data,
    )
    runner = CliRunner()
    result = runner.invoke(
        cli,
        [
            "data-converter",
            "rasterize-map",
            "-i",
            "soil.gpkg",
            "-d",
            "dem.tif",
            "-o",
            "soil.tif",
            "-m",
            "map_code",
        ],
    )

    assert result.exit_code == 0, result.output
    assert captured == {
        "input_file": Path("soil.gpkg"),
        "dem_file": Path("dem.tif"),
        "output_file": Path("soil.tif"),
        "mapping_field": "map_code",
    }
    alias_result = runner.invoke(cli, ["data-converter", "rasterize_map", "--help"])
    assert alias_result.exit_code == 0
