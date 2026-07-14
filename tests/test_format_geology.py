"""Tests for lookup-based geology data formatting."""

from pathlib import Path

import geopandas as gpd
import numpy as np
import pytest
import rasterio
import xarray as xr
from click.testing import CliRunner
from mhm_tools import pre
from mhm_tools._cli._main import cli
from mhm_tools.common.file_handler import get_xarray_ds_from_file
from mhm_tools.pre import format_geology as format_geology_module
from mhm_tools.pre.format_geology import (
    format_geology_data,
    write_geology_classdefinition,
)


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


def _write_lookup(path: Path) -> None:
    table = gpd.GeoDataFrame(
        {
            "*Map-code [id]": [10, 20, 30],
            "*Geology Class [id]": [3, 1, 2],
            "Geo Class [count]": [2, 1, 2],
            "Karstic [flag]": ["yes", "0", "TRUE"],
            "Parameter Value [default]": [30, 10, 20],
        }
    )
    table.to_file(path, driver="GPKG")


def _expected_classdefinition() -> str:
    return (
        "nGeo_Formations  3\n"
        "GeoParam(i)   ClassUnit     Karstic      Description\n"
        "         1\t         1              0      GeoUnit-1\n"
        "         2\t         2              1      GeoUnit-2\n"
        "         2\t         3              1      GeoUnit-3\n"
        "!<-END\n"
        "\n"
        "\n"
        "!***********************************\n"
        "! NOTES\n"
        "!***********************************\n"
        "1 = Karstic\n"
        "0 = Non-karstic\n"
        "\n"
        "IMPORTANT ::\n"
        "   Ordering has to be according to the ordering in mhm_parameter.nml\n"
        "   (namelist: geoparameter)\n"
    )


def test_format_geology_data_writes_nc_and_asc(tmp_path: Path):
    """Both output formats preserve the grid, classes, and nodata."""
    input_file = tmp_path / "geology_raw.tif"
    lookup_file = tmp_path / "geology_lookup.gpkg"
    _write_category_raster(input_file)
    _write_lookup(lookup_file)
    expected = np.array(
        [[3, 1, -9999], [2, 3, -9999]],
        dtype=np.int32,
    )

    nc_file = format_geology_data(
        input_file,
        tmp_path / "nc",
        lookup_file,
        "map code",
        output_type="nc",
    )

    assert nc_file == tmp_path / "nc" / "geology_class.nc"
    assert (tmp_path / "nc" / "geology_classdefinition.txt").read_text() == (
        _expected_classdefinition()
    )
    with xr.open_dataset(nc_file, decode_cf=False) as dataset:
        assert "geology_class" in dataset.data_vars
        assert dataset["geology_class"].dims == ("y", "x")
        assert dataset["geology_class"].dtype == np.dtype("int32")
        assert dataset["geology_class"].attrs["_FillValue"] == -9999
        np.testing.assert_array_equal(dataset["geology_class"].values, expected)
        np.testing.assert_allclose(dataset["x"].values, [105.0, 115.0, 125.0])
        np.testing.assert_allclose(dataset["y"].values, [215.0, 205.0])

    asc_file = format_geology_data(
        input_file,
        tmp_path / "asc",
        lookup_file,
        "map code",
        output_type="asc",
    )

    assert asc_file == tmp_path / "asc" / "geology_class.asc"
    assert (tmp_path / "asc" / "geology_classdefinition.txt").read_text() == (
        _expected_classdefinition()
    )
    asc_dataset = get_xarray_ds_from_file(asc_file)
    try:
        np.testing.assert_array_equal(asc_dataset["data"].values, expected)
        np.testing.assert_allclose(asc_dataset["lon"].values, [105, 115, 125])
        np.testing.assert_allclose(asc_dataset["lat"].values, [215, 205])
    finally:
        asc_dataset.close()


def test_write_geology_classdefinition_normalizes_sorts_and_parses_karstic(
    tmp_path: Path,
):
    """Template field variants and supported booleans match pymhm output."""
    lookup_file = tmp_path / "geology_lookup.gpkg"
    _write_lookup(lookup_file)

    output_file = write_geology_classdefinition(
        lookup_file,
        tmp_path / "definitions" / "geology_classdefinition.txt",
    )

    assert output_file.read_text() == _expected_classdefinition()


def test_format_geology_data_requires_at_least_one_mapping(tmp_path: Path):
    """A valid definition that matches no raster category leaves no outputs."""
    input_file = tmp_path / "geology_raw.tif"
    lookup_file = tmp_path / "geology_lookup.gpkg"
    output_path = tmp_path / "output"
    _write_category_raster(input_file)
    table = gpd.GeoDataFrame(
        {
            "source": [777],
            "GEOLOGY_CLASS": [1],
            "GEO_CLASS": [1],
            "KARSTIC": [0],
            "PARAMETER_VALUE": [10],
        }
    )
    table.to_file(lookup_file, driver="GPKG")

    with pytest.raises(ValueError, match="No valid raster category matched"):
        format_geology_data(input_file, output_path, lookup_file, "source")

    assert not (output_path / "geology_class.nc").exists()
    assert not (output_path / "geology_classdefinition.txt").exists()


def test_definition_is_validated_before_geology_raster_is_written(tmp_path: Path):
    """An incomplete definition schema cannot leave a new output behind."""
    input_file = tmp_path / "geology_raw.tif"
    lookup_file = tmp_path / "incomplete.gpkg"
    output_path = tmp_path / "output"
    _write_category_raster(input_file)
    table = gpd.GeoDataFrame(
        {
            "source": [10],
            "GEOLOGY_CLASS": [1],
            "GEO_CLASS": [1],
            "KARSTIC": [0],
        }
    )
    table.to_file(lookup_file, driver="GPKG")

    with pytest.raises(ValueError, match="PARAMETER_VALUE"):
        format_geology_data(input_file, output_path, lookup_file, "source")

    assert not (output_path / "geology_class.nc").exists()
    assert not (output_path / "geology_classdefinition.txt").exists()


def test_format_geology_cli_short_options_alias_and_pre_exports(monkeypatch):
    """The grouped command and lazy public exports expose the new formatter."""
    assert pre.format_geology_data is format_geology_data
    assert pre.write_geology_classdefinition is write_geology_classdefinition

    captured = {}

    def fake_format_geology_data(**kwargs):
        captured.update(kwargs)

    monkeypatch.setattr(
        format_geology_module,
        "format_geology_data",
        fake_format_geology_data,
    )
    runner = CliRunner()
    result = runner.invoke(
        cli,
        [
            "data-converter",
            "format-geology-data",
            "-i",
            "geology.tif",
            "-o",
            "output",
            "-l",
            "lookup.gpkg",
            "-m",
            "source",
            "-t",
            "asc",
        ],
    )

    assert result.exit_code == 0, result.output
    assert captured == {
        "input_file": Path("geology.tif"),
        "output_path": Path("output"),
        "lookup_table": Path("lookup.gpkg"),
        "mapping_field": "source",
        "output_type": "asc",
    }
    alias_result = runner.invoke(
        cli,
        ["data-converter", "format_geology_data", "--help"],
    )
    assert alias_result.exit_code == 0
    assert "geology_classdefinition.txt" in alias_result.output
