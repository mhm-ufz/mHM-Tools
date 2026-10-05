"""Tests for rasterizing vector maps on a reference DEM grid."""

from pathlib import Path

import geopandas as gpd
import numpy as np
import pytest
import rasterio
import xarray as xr
from click.testing import CliRunner
from shapely import geometry as shapely_geometry

from mhm_tools._cli._main import cli
from mhm_tools.common.constants import NO_DATA
from mhm_tools.common.file_handler import get_raster_data
from mhm_tools.common.netcdf import NetcdfCompression
from mhm_tools.pre import rasterize_map as rasterize_map_module
from mhm_tools.pre.rasterize_map import rasterize_map_data


def _write_dem(path: Path, crs="EPSG:32632") -> None:
    transform = rasterio.transform.from_origin(100.0, 220.0, 10.0, 10.0)
    if path.suffix == ".nc":
        pytest.importorskip("rioxarray")
        dem = xr.DataArray(
            np.zeros((2, 3), dtype=np.float32),
            dims=("y", "x"),
            coords={"x": [105.0, 115.0, 125.0], "y": [215.0, 205.0]},
            name="elevation",
        ).rio.write_transform(transform)
        if crs is not None:
            dem = dem.rio.write_crs(crs)
        dem.to_dataset().to_netcdf(path)
        return

    driver = "AAIGrid" if path.suffix == ".asc" else "GTiff"
    with rasterio.open(
        path,
        "w",
        driver=driver,
        width=3,
        height=2,
        count=1,
        dtype="float32",
        crs=crs,
        transform=transform,
        nodata=-9999.0,
    ) as dataset:
        dataset.write(np.zeros((2, 3), dtype=np.float32), 1)


def _write_vector(path: Path, values, crs="EPSG:32632") -> None:
    box = shapely_geometry.box
    frame = gpd.GeoDataFrame(
        {"map_code": values},
        geometry=[
            box(100.0, 210.0, 110.0, 220.0),
            box(110.0, 200.0, 130.0, 220.0),
        ],
        crs=crs,
    )
    driver = "ESRI Shapefile" if path.suffix == ".shp" else "GPKG"
    frame.to_file(path, driver=driver)


@pytest.mark.parametrize("suffix", [".tif", ".tiff", ".asc", ".nc"])
def test_rasterize_map_uses_exact_dem_grid(tmp_path: Path, suffix: str):
    """Every output format uses the exact DEM grid and categorical values."""
    dem_file = tmp_path / "dem.tif"
    vector_file = tmp_path / "soil.gpkg"
    output_file = tmp_path / f"soil_raw{suffix}"
    _write_dem(dem_file)
    _write_vector(vector_file, [5, 9])

    result = rasterize_map_data(vector_file, dem_file, output_file, "map_code")

    assert result == output_file
    output = get_raster_data(output_file)
    try:
        values = np.where(np.isfinite(output.values), output.values, NO_DATA)
        assert output.rio.crs.to_epsg() == 32632
        assert output.rio.transform().almost_equals(
            rasterio.transform.from_origin(100.0, 220.0, 10.0, 10.0)
        )
        assert output.shape == (2, 3)
        np.testing.assert_array_equal(
            values.astype(np.int32),
            np.array([[5, 9, 9], [-9999, 9, 9]], dtype=np.int32),
        )
    finally:
        output.close()

    if suffix in {".tif", ".tiff"}:
        with rasterio.open(output_file) as raw:
            assert raw.count == 1
            assert raw.dtypes == ("int32",)
            assert raw.nodata == -9999
            assert raw.compression == rasterio.enums.Compression.deflate
    elif suffix == ".asc":
        assert output_file.with_suffix(".prj").is_file()
    else:
        with xr.open_dataset(output_file, decode_cf=False) as raw:
            assert raw["rasterized_map"].dtype == np.dtype("int32")
            assert raw["rasterized_map"].attrs["_FillValue"] == -9999
            assert raw["rasterized_map"].attrs["coordinates"] == "spatial_ref"
            assert raw["spatial_ref"].attrs["grid_mapping_name"]
            np.testing.assert_array_equal(
                raw["rasterized_map"].values,
                [[5, 9, 9], [-9999, 9, 9]],
            )


@pytest.mark.parametrize("suffix", [".asc", ".nc"])
def test_rasterize_map_accepts_ascii_and_netcdf_dem(tmp_path: Path, suffix: str):
    """ASCII and NetCDF DEMs provide the same exact output grid as GeoTIFF."""
    dem_file = tmp_path / f"dem{suffix}"
    vector_file = tmp_path / "soil.gpkg"
    output_file = tmp_path / "soil_raw.tif"
    _write_dem(dem_file)
    _write_vector(vector_file, [5, 9])

    rasterize_map_data(vector_file, dem_file, output_file, "map_code")

    with rasterio.open(output_file) as output:
        assert output.crs == rasterio.crs.CRS.from_epsg(32632)
        assert output.transform == rasterio.transform.from_origin(
            100.0, 220.0, 10.0, 10.0
        )
        assert (output.height, output.width) == (2, 3)


def test_rasterize_map_assigns_explicit_dem_crs(tmp_path: Path):
    """An explicit fallback gives a CRS-less ASCII DEM spatial meaning."""
    dem_file = tmp_path / "dem.asc"
    vector_file = tmp_path / "soil.gpkg"
    output_file = tmp_path / "soil_raw.tif"
    _write_dem(dem_file, crs=None)
    _write_vector(vector_file, [5, 9])

    rasterize_map_data(
        vector_file,
        dem_file,
        output_file,
        "map_code",
        dem_crs="EPSG:32632",
    )

    with rasterio.open(output_file) as output:
        assert output.crs == rasterio.crs.CRS.from_epsg(32632)


def test_rasterize_map_assigns_explicit_vector_crs(tmp_path: Path):
    """An explicit fallback gives a CRS-less vector spatial meaning."""
    dem_file = tmp_path / "dem.tif"
    vector_file = tmp_path / "soil.gpkg"
    output_file = tmp_path / "soil_raw.tif"
    _write_dem(dem_file)
    _write_vector(vector_file, [5, 9], crs=None)

    rasterize_map_data(
        vector_file,
        dem_file,
        output_file,
        "map_code",
        input_crs="EPSG:32632",
    )

    with rasterio.open(output_file) as output:
        assert output.crs == rasterio.crs.CRS.from_epsg(32632)


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
    gpd.GeoDataFrame({"map_code": [7]}, geometry=[geometry], crs="EPSG:32632").to_file(
        vector_file, driver="GPKG"
    )

    rasterize_map_data(vector_file, dem_file, output_file, "map_code")

    with rasterio.open(output_file) as output:
        assert np.any(output.read(1) == 7)


def test_rasterize_map_rejects_non_integral_burn_values(tmp_path: Path):
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


@pytest.mark.parametrize("suffix", [".tif", ".tiff", ".asc", ".nc"])
def test_rasterize_map_applies_lookup_to_text_categories(tmp_path: Path, suffix: str):
    """Every output format burns lookup values for textual vector keys."""
    dem_file = tmp_path / "dem.tif"
    vector_file = tmp_path / "soil.gpkg"
    lookup_file = tmp_path / "lookup.gpkg"
    output_file = tmp_path / f"soil_class{suffix}"
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
        "SOIL_CLASS",
        lookup_table=lookup_file,
        mapping_field="map code",
    )

    output = get_raster_data(output_file)
    try:
        values = np.where(np.isfinite(output.values), output.values, NO_DATA)
        np.testing.assert_array_equal(
            values.astype(np.int32),
            np.array([[4, 8, 8], [-9999, 8, 8]], dtype=np.int32),
        )
    finally:
        output.close()


def test_rasterize_map_rejects_unsupported_output_suffix(tmp_path: Path):
    """Output format is selected only from the supported raster suffixes."""
    dem_file = tmp_path / "dem.tif"
    vector_file = tmp_path / "soil.gpkg"
    _write_dem(dem_file)
    _write_vector(vector_file, [5, 9])

    with pytest.raises(ValueError, match=r"\.asc, \.nc, \.tif, or \.tiff"):
        rasterize_map_data(
            vector_file,
            dem_file,
            tmp_path / "soil.txt",
            "map_code",
        )


def test_rasterize_map_protects_shapefile_prj_from_ascii_output(tmp_path: Path):
    """An ASCII sidecar cannot overwrite a same-stem shapefile CRS."""
    dem_file = tmp_path / "dem.tif"
    vector_file = tmp_path / "soil.shp"
    output_file = tmp_path / "soil.asc"
    _write_dem(dem_file)
    _write_vector(vector_file, [5, 9])
    projection = vector_file.with_suffix(".prj")
    original_projection = projection.read_bytes()

    with pytest.raises(ValueError, match="sidecar"):
        rasterize_map_data(vector_file, dem_file, output_file, "map_code")

    assert projection.read_bytes() == original_projection
    assert not output_file.exists()


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
        "SOIL_CLASS",
        lookup_table=lookup_file,
        mapping_field="map_code",
    )

    with rasterio.open(output_file) as output:
        np.testing.assert_array_equal(
            output.read(1),
            np.array([[3, 6, 6], [-9999, 6, 6]], dtype=np.int32),
        )


def test_rasterize_map_cli_forwards_options(monkeypatch):
    """The grouped command forwards direct, lookup, and CRS options."""
    captured = {}

    def fake_rasterize_map_data(**kwargs):
        captured.update(kwargs)

    monkeypatch.setattr(
        rasterize_map_module,
        "rasterize_map_data",
        fake_rasterize_map_data,
    )
    result = CliRunner().invoke(
        cli,
        [
            "data-converter",
            "rasterize-map",
            "-i",
            "soil.gpkg",
            "-d",
            "dem.asc",
            "-o",
            "soil.tif",
            "-b",
            "SOIL_CLASS",
            "-l",
            "lookup.csv",
            "-m",
            "map_code",
            "-s",
            "EPSG:32631",
            "-r",
            "EPSG:32632",
            "--compression",
            "9",
        ],
    )

    assert result.exit_code == 0, result.output
    assert captured == {
        "input_file": Path("soil.gpkg"),
        "compression": NetcdfCompression(complevel=9, shuffle=None),
        "dem_file": Path("dem.asc"),
        "output_file": Path("soil.tif"),
        "burn_field": "SOIL_CLASS",
        "lookup_table": Path("lookup.csv"),
        "mapping_field": "map_code",
        "input_crs": "EPSG:32631",
        "dem_crs": "EPSG:32632",
    }


@pytest.mark.parametrize(
    "lookup_options",
    [
        ["--lookup-table", "lookup.csv"],
        ["--mapping-field", "map_code"],
    ],
)
def test_rasterize_map_cli_rejects_incomplete_lookup_options(lookup_options):
    """Lookup table and category field must always be supplied together."""
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
            "-b",
            "SOIL_CLASS",
            *lookup_options,
        ],
    )

    assert result.exit_code == 2
    assert (
        "Options --lookup-table and --mapping-field must be provided together."
        in result.output
    )
