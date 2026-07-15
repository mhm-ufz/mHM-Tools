"""Tests for lookup-based soil data formatting."""

from pathlib import Path

import geopandas as gpd
import numpy as np
import pytest
import rasterio
import xarray as xr
from click.testing import CliRunner
from shapely import geometry as shapely_geometry

from mhm_tools import pre
from mhm_tools._cli._main import cli
from mhm_tools.common.file_handler import (
    get_raster_data,
    get_xarray_ds_from_file,
    write_xarray_to_file,
)
from mhm_tools.pre import format_soil as format_soil_module
from mhm_tools.pre.format_soil import format_soil_data, write_soil_classdefinition


def _write_category_raster(path: Path) -> None:
    values = np.array(
        [[10, 20, -9999], [30, 10, 99]],
        dtype=np.int32,
    )
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        width=3,
        height=2,
        count=1,
        dtype="int32",
        crs="EPSG:32632",
        transform=rasterio.transform.from_origin(100.0, 220.0, 10.0, 10.0),
        nodata=-9999,
    ) as dataset:
        dataset.write(values, 1)


def _write_dem(path: Path, *, width: int = 3, height: int = 2, cellsize=10) -> None:
    """Write a DEM grid used as the formatter reference."""
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        width=width,
        height=height,
        count=1,
        dtype="float32",
        crs="EPSG:32632",
        transform=rasterio.transform.from_origin(100.0, 220.0, cellsize, cellsize),
        nodata=-9999,
    ) as dataset:
        dataset.write(np.ones((height, width), dtype=np.float32), 1)


def _convert_raster(source: Path, output: Path) -> None:
    data = get_raster_data(source)
    try:
        write_xarray_to_file(data, output, var_name=data.name, crs=data.rio.crs)
    finally:
        data.close()


def _write_lookup(path: Path) -> None:
    point = shapely_geometry.Point
    table = gpd.GeoDataFrame(
        {
            "*Map-code [id]": [10, None, 20, 30],
            "soil class [id]": [1, 1, 2, 3],
            "HORIZON": [1, 2, 1, 1],
            "UPPER_DEPTH [mm]": [0, 100, 0, 0],
            "LOWER_DEPTH [mm]": [100, 300, 200, 300],
            "CLAY [%]": [20, 22, 30, 40],
            "SAND [%]": [50, 48, 40, 30],
            "SILT [%]": [30, 30, 30, 30],
            "BULK_DENSITY [gcm-3]": [1.2, 1.25, 1.3, 1.4],
        },
        geometry=[point(0, 0), point(1, 1), point(2, 2), point(3, 3)],
        crs="EPSG:4326",
    )
    table.to_file(path, driver="GPKG")


def test_format_soil_data_writes_nc_and_asc(tmp_path: Path):
    """Lookup fields are normalized and the source grid and nodata are kept."""
    input_file = tmp_path / "soil_raw.tif"
    dem_file = tmp_path / "dem.tif"
    lookup_file = tmp_path / "soil_lookup.gpkg"
    _write_category_raster(input_file)
    _write_dem(dem_file)
    _write_lookup(lookup_file)
    expected = np.array(
        [[1, 2, -9999], [3, 1, -9999]],
        dtype=np.int32,
    )

    nc_file = format_soil_data(
        input_file,
        dem_file,
        tmp_path / "nc",
        lookup_file,
        "map code",
        output_type="nc",
    )

    assert nc_file == tmp_path / "nc" / "soil_class.nc"
    assert (tmp_path / "nc" / "soil_classdefinition.txt").read_text() == (
        "nSoil_Types 3\n"
        "SOIL_NR\tHORIZON\tUD[mm]\tLD[mm]\tClay[%]\tSAND[%]\t"
        "Bd[gcm-3]\tSilt[%]\n"
        "1\t1\t0\t100\t20\t50\t1.2\t30\n"
        "1\t2\t100\t300\t22\t48\t1.25\t30\n"
        "2\t1\t0\t200\t30\t40\t1.3\t30\n"
        "3\t1\t0\t300\t40\t30\t1.4\t30\n"
    )
    with xr.open_dataset(nc_file, decode_cf=False) as dataset:
        assert "soil_class" in dataset.data_vars
        assert dataset["soil_class"].dims == ("y", "x")
        assert dataset["soil_class"].dtype == np.dtype("int32")
        assert dataset["soil_class"].attrs["_FillValue"] == -9999
        np.testing.assert_array_equal(dataset["soil_class"].values, expected)
        np.testing.assert_allclose(dataset["x"].values, [105.0, 115.0, 125.0])
        np.testing.assert_allclose(dataset["y"].values, [215.0, 205.0])

    asc_file = format_soil_data(
        input_file,
        dem_file,
        tmp_path / "asc",
        lookup_file,
        "map code",
        output_type="asc",
    )
    assert asc_file == tmp_path / "asc" / "soil_class.asc"
    assert asc_file.with_suffix(".prj").is_file()
    assert (tmp_path / "asc" / "soil_classdefinition.txt").is_file()
    asc_dataset = get_xarray_ds_from_file(asc_file)
    try:
        np.testing.assert_array_equal(asc_dataset["data"].values, expected)
        np.testing.assert_allclose(asc_dataset["lon"].values, [105, 115, 125])
        np.testing.assert_allclose(asc_dataset["lat"].values, [215, 205])
    finally:
        asc_dataset.close()


@pytest.mark.parametrize(
    ("input_suffix", "dem_suffix"),
    [(".asc", ".nc"), (".nc", ".asc")],
)
def test_format_soil_data_accepts_mixed_raster_formats(
    tmp_path: Path, input_suffix: str, dem_suffix: str
):
    """Mixed supported raster formats use the DEM grid and CRS."""
    input_tif = tmp_path / "soil_raw.tif"
    dem_tif = tmp_path / "dem.tif"
    input_file = tmp_path / f"soil_raw{input_suffix}"
    dem_file = tmp_path / f"dem{dem_suffix}"
    lookup_file = tmp_path / "soil_lookup.gpkg"
    _write_category_raster(input_tif)
    _write_dem(dem_tif)
    _convert_raster(input_tif, input_file)
    _convert_raster(dem_tif, dem_file)
    _write_lookup(lookup_file)

    output = format_soil_data(
        input_file,
        dem_file,
        tmp_path / "output",
        lookup_file,
        "map code",
    )

    with xr.open_dataset(output, decode_cf=False) as dataset:
        np.testing.assert_array_equal(
            dataset["soil_class"].values,
            [[1, 2, -9999], [3, 1, -9999]],
        )


def test_format_soil_data_requires_at_least_one_mapping(tmp_path: Path):
    """A lookup that matches no raster category is rejected."""
    input_file = tmp_path / "soil_raw.tif"
    lookup_file = tmp_path / "soil_lookup.gpkg"
    _write_category_raster(input_file)
    point = shapely_geometry.Point
    table = gpd.GeoDataFrame(
        {
            "source": [777],
            "SOIL_CLASS": [1],
            "HORIZON": [1],
            "UPPER_DEPTH": [0],
            "LOWER_DEPTH": [100],
            "CLAY": [20],
            "SAND": [50],
            "SILT": [30],
            "BULK_DENSITY": [1.2],
        },
        geometry=[point(0, 0)],
        crs="EPSG:4326",
    )
    table.to_file(lookup_file, driver="GPKG")

    with pytest.raises(ValueError, match="No valid raster category matched"):
        format_soil_data(
            input_file,
            input_file,
            tmp_path / "output",
            lookup_file,
            "source",
        )


def test_format_soil_cli_short_options_and_pre_exports(monkeypatch):
    """The command is grouped, uses short options, and public APIs are exposed."""
    assert pre.format_soil_data is format_soil_data
    assert pre.write_soil_classdefinition is write_soil_classdefinition
    assert pre.rasterize_map_data.__name__ == "rasterize_map_data"

    captured = {}

    def fake_format_soil_data(**kwargs):
        captured.update(kwargs)

    monkeypatch.setattr(
        format_soil_module,
        "format_soil_data",
        fake_format_soil_data,
    )
    runner = CliRunner()
    result = runner.invoke(
        cli,
        [
            "data-converter",
            "format-soil-data",
            "-i",
            "soil.tif",
            "-d",
            "dem.nc",
            "-o",
            "output",
            "-l",
            "lookup.gpkg",
            "-m",
            "source",
            "-t",
            "asc",
            "-s",
            "EPSG:32632",
            "-r",
            "EPSG:32633",
        ],
    )

    assert result.exit_code == 0, result.output
    assert captured == {
        "input_file": Path("soil.tif"),
        "dem_file": Path("dem.nc"),
        "output_path": Path("output"),
        "lookup_table": Path("lookup.gpkg"),
        "mapping_field": "source",
        "output_type": "asc",
        "input_crs": "EPSG:32632",
        "dem_crs": "EPSG:32633",
    }
    alias_result = runner.invoke(cli, ["data-converter", "format_soil_data", "--help"])
    assert alias_result.exit_code == 0
    assert "soil_classdefinition.txt" in alias_result.output


def test_write_soil_classdefinition_rowwise_exact_output(tmp_path: Path):
    """Row-per-horizon lookup rows are normalized, sorted, and formatted."""
    lookup_file = tmp_path / "rowwise.gpkg"
    table = gpd.GeoDataFrame(
        {
            "*SOIL CLASS [id]": [2, 1, 2],
            "HORIZON": [2, 1, 1],
            "UPPER-DEPTH [mm]": [100, 0, 0],
            "LOWER DEPTH [mm]": [300, 120, 100],
            "Clay [%]": [20.125, 31, 22],
            "SAND [%]": [30, 40.5, 33],
            "Silt [%]": [49.875, 28.5, 45],
            "Bulk Density [gcm-3]": [1.23456789, 1.4, 1.3],
        },
        geometry=[
            shapely_geometry.Point(0, 0),
            shapely_geometry.Point(1, 1),
            shapely_geometry.Point(2, 2),
        ],
        crs="EPSG:4326",
    )
    table.to_file(lookup_file, driver="GPKG")

    output_file = write_soil_classdefinition(
        lookup_file, tmp_path / "definitions" / "soil_classdefinition.txt"
    )

    assert output_file.read_text() == (
        "nSoil_Types 2\n"
        "SOIL_NR\tHORIZON\tUD[mm]\tLD[mm]\tClay[%]\tSAND[%]\t"
        "Bd[gcm-3]\tSilt[%]\n"
        "1\t1\t0\t120\t31\t40.5\t1.4\t28.5\n"
        "2\t1\t0\t100\t22\t33\t1.3\t45\n"
        "2\t2\t100\t300\t20.125\t30\t1.23457\t49.875\n"
    )


def test_write_soil_classdefinition_wide_exact_output(tmp_path: Path):
    """Wide lookup rows are expanded into cumulative horizon depths."""
    lookup_file = tmp_path / "wide.gpkg"
    table = gpd.GeoDataFrame(
        {
            "SOIL_CLASS": [2, 1],
            "HORIZONS": [2, 1],
            "DEPTH1 [mm]": [100, 75],
            "BULK_DENSITY1": [1.2, 1.1],
            "CLAY1": [20, 10],
            "SILT1": [30, 40],
            "SAND1": [50, 50],
            "DEPTH2 [mm]": [250, 0],
            "BULK_DENSITY2": [1.35, 0],
            "CLAY2": [25, 0],
            "SILT2": [35, 0],
            "SAND2": [40, 0],
        }
    )
    table.to_file(lookup_file, driver="GPKG")

    output_file = write_soil_classdefinition(
        lookup_file, tmp_path / "soil_classdefinition.txt"
    )

    assert output_file.read_text() == (
        "nSoil_Types 2\n"
        "SOIL_NR\tHORIZON\tUD[mm]\tLD[mm]\tClay[%]\tSAND[%]\t"
        "Bd[gcm-3]\tSilt[%]\n"
        "1\t1\t0\t75\t10\t50\t1.1\t40\n"
        "2\t1\t0\t100\t20\t50\t1.2\t30\n"
        "2\t2\t100\t250\t25\t40\t1.35\t35\n"
    )


def test_format_soil_data_uses_exact_dem_grid(tmp_path: Path):
    """Input categories are aligned to the exact DEM grid before mapping."""
    input_file = tmp_path / "soil_raw.tif"
    lookup_file = tmp_path / "soil_lookup.gpkg"
    dem_file = tmp_path / "dem.tif"
    _write_category_raster(input_file)
    _write_lookup(lookup_file)
    with rasterio.open(
        dem_file,
        "w",
        driver="GTiff",
        width=6,
        height=4,
        count=1,
        dtype="float32",
        crs="EPSG:32632",
        transform=rasterio.transform.from_origin(100.0, 220.0, 5.0, 5.0),
        nodata=-9999,
    ) as dataset:
        dataset.write(np.ones((4, 6), dtype=np.float32), 1)

    result = format_soil_data(
        input_file,
        dem_file,
        tmp_path / "output",
        lookup_file,
        "map code",
    )

    assert result == tmp_path / "output" / "soil_class.nc"
    assert (tmp_path / "output" / "soil_classdefinition.txt").is_file()
    expected = np.repeat(
        np.repeat(
            np.array([[1, 2, -9999], [3, 1, -9999]], dtype=np.int32),
            2,
            axis=0,
        ),
        2,
        axis=1,
    )
    output = get_raster_data(result)
    reference = get_raster_data(dem_file)
    try:
        assert output.rio.crs == reference.rio.crs
        assert output.rio.transform() == reference.rio.transform()
        assert output.shape == reference.shape
        np.testing.assert_array_equal(output.fillna(-9999).values, expected)
    finally:
        output.close()
        reference.close()


def test_definition_is_validated_before_raster_is_written(tmp_path: Path):
    """An incomplete definition schema cannot leave a new raster behind."""
    input_file = tmp_path / "soil_raw.tif"
    lookup_file = tmp_path / "incomplete.gpkg"
    _write_category_raster(input_file)
    table = gpd.GeoDataFrame(
        {"source": [10], "SOIL_CLASS": [1]},
        geometry=[shapely_geometry.Point(0, 0)],
        crs="EPSG:4326",
    )
    table.to_file(lookup_file, driver="GPKG")
    output_path = tmp_path / "output"

    with pytest.raises(ValueError, match="row-per-horizon layout"):
        format_soil_data(input_file, input_file, output_path, lookup_file, "source")

    assert not (output_path / "soil_class.nc").exists()
    assert not (output_path / "soil_classdefinition.txt").exists()
