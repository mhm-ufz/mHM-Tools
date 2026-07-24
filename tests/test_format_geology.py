"""Tests for lookup-based geology data formatting."""

from pathlib import Path

import geopandas as gpd
import numpy as np
import pytest
import rasterio
import xarray as xr

from mhm_tools import pre
from mhm_tools.common.file_handler import (
    get_raster_data,
    get_xarray_ds_from_file,
    write_xarray_to_file,
)
from mhm_tools.pre.format_geology import (
    format_geology_data,
    write_geology_classdefinition,
)

_CLASS_FIELD = "mapped geology unit"


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
    table = gpd.GeoDataFrame(
        {
            "*Map-code [id]": [10, 20, 30],
            "*Mapped Geology Unit [id]": [3, 1, 2],
            "Geo Class [count]": [2, 1, 2],
            "Karstic [flag]": ["yes", "0", "TRUE"],
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


def test_format_geology_data_writes_nc_asc_and_tif(tmp_path: Path):
    """A custom class field drives every raster and its classdefinition."""
    input_file = tmp_path / "geology_raw.tif"
    dem_file = tmp_path / "dem.tif"
    lookup_file = tmp_path / "geology_lookup.gpkg"
    _write_category_raster(input_file)
    _write_dem(dem_file)
    _write_lookup(lookup_file)
    expected = np.array(
        [[3, 1, -9999], [2, 3, -9999]],
        dtype=np.int32,
    )

    nc_file = format_geology_data(
        input_file,
        dem_file,
        tmp_path / "nc",
        lookup_file,
        "map code",
        _CLASS_FIELD,
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
        dem_file,
        tmp_path / "asc",
        lookup_file,
        "map code",
        _CLASS_FIELD,
        output_type="asc",
    )

    assert asc_file == tmp_path / "asc" / "geology_class.asc"
    assert asc_file.with_suffix(".prj").is_file()
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

    tif_file = format_geology_data(
        input_file,
        dem_file,
        tmp_path / "tif",
        lookup_file,
        "map code",
        _CLASS_FIELD,
        output_type="tif",
    )
    assert tif_file == tmp_path / "tif" / "geology_class.tif"
    assert (tmp_path / "tif" / "geology_classdefinition.txt").read_text() == (
        _expected_classdefinition()
    )
    with rasterio.open(tif_file) as dataset:
        assert dataset.dtypes == ("int32",)
        assert dataset.nodata == -9999
        assert dataset.crs == rasterio.crs.CRS.from_epsg(32632)
        assert dataset.transform == rasterio.transform.from_origin(
            100.0, 220.0, 10.0, 10.0
        )
        np.testing.assert_array_equal(dataset.read(1), expected)


@pytest.mark.parametrize(
    ("input_suffix", "dem_suffix"),
    [(".asc", ".nc"), (".nc", ".asc")],
)
def test_format_geology_data_accepts_mixed_raster_formats(
    tmp_path: Path, input_suffix: str, dem_suffix: str
):
    """Mixed supported raster formats use the DEM grid and CRS."""
    input_tif = tmp_path / "geology_raw.tif"
    dem_tif = tmp_path / "dem.tif"
    input_file = tmp_path / f"geology_raw{input_suffix}"
    dem_file = tmp_path / f"dem{dem_suffix}"
    lookup_file = tmp_path / "geology_lookup.gpkg"
    _write_category_raster(input_tif)
    _write_dem(dem_tif)
    _convert_raster(input_tif, input_file)
    _convert_raster(dem_tif, dem_file)
    _write_lookup(lookup_file)

    output = format_geology_data(
        input_file,
        dem_file,
        tmp_path / "output",
        lookup_file,
        "map code",
        _CLASS_FIELD,
    )

    with xr.open_dataset(output, decode_cf=False) as dataset:
        np.testing.assert_array_equal(
            dataset["geology_class"].values,
            [[3, 1, -9999], [2, 3, -9999]],
        )


def test_write_geology_classdefinition_normalizes_sorts_and_parses_karstic(
    tmp_path: Path,
):
    """Template field variants and supported booleans match pymhm output."""
    lookup_file = tmp_path / "geology_lookup.gpkg"
    _write_lookup(lookup_file)

    output_file = write_geology_classdefinition(
        lookup_file,
        tmp_path / "definitions" / "geology_classdefinition.txt",
        _CLASS_FIELD,
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
        }
    )
    table.to_file(lookup_file, driver="GPKG")

    with pytest.raises(ValueError, match="No valid raster category matched"):
        format_geology_data(
            input_file,
            input_file,
            output_path,
            lookup_file,
            "source",
            "GEOLOGY_CLASS",
        )

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
        }
    )
    table.to_file(lookup_file, driver="GPKG")

    with pytest.raises(ValueError, match="KARSTIC"):
        format_geology_data(
            input_file,
            input_file,
            output_path,
            lookup_file,
            "source",
            "GEOLOGY_CLASS",
        )

    assert not (output_path / "geology_class.nc").exists()
    assert not (output_path / "geology_classdefinition.txt").exists()


def test_pre_exports_geology_formatters():
    """The public pre package exposes the geology formatters."""
    assert pre.format_geology_data is format_geology_data
    assert pre.write_geology_classdefinition is write_geology_classdefinition


def test_format_geology_data_uses_exact_dem_grid(tmp_path: Path):
    """Input categories are aligned to the exact DEM grid before mapping."""
    input_file = tmp_path / "geology_raw.tif"
    dem_file = tmp_path / "dem.tif"
    lookup_file = tmp_path / "geology_lookup.gpkg"
    _write_category_raster(input_file)
    _write_dem(dem_file, width=6, height=4, cellsize=5)
    _write_lookup(lookup_file)

    result = format_geology_data(
        input_file,
        dem_file,
        tmp_path / "output",
        lookup_file,
        "map code",
        _CLASS_FIELD,
    )

    expected = np.repeat(
        np.repeat(
            np.array([[3, 1, -9999], [2, 3, -9999]], dtype=np.int32),
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
