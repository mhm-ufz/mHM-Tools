"""Tests for lookup-based land-cover formatting."""

from pathlib import Path

import geopandas as gpd
import numpy as np
import pytest
import rasterio
import xarray as xr

from mhm_tools import pre
from mhm_tools.pre.format_lc_data import format_lc_data


def _write_raster(path: Path, values, *, cellsize: float) -> None:
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        width=values.shape[1],
        height=values.shape[0],
        count=1,
        dtype=values.dtype,
        crs="EPSG:32632",
        transform=rasterio.transform.from_origin(100.0, 220.0, cellsize, cellsize),
        nodata=-9999,
    ) as dataset:
        dataset.write(values, 1)


def test_format_lc_data_maps_and_aligns_to_dem(tmp_path: Path):
    """Land-cover output uses its fixed filename and NetCDF variable name."""
    input_file = tmp_path / "land_cover.tif"
    dem_file = tmp_path / "dem.tif"
    lookup_file = tmp_path / "lookup.gpkg"
    _write_raster(
        input_file,
        np.array([[10, 20, -9999], [30, 10, 99]], dtype=np.int32),
        cellsize=10,
    )
    _write_raster(dem_file, np.ones((4, 6), dtype=np.float32), cellsize=5)
    gpd.GeoDataFrame({"Grid value": [10, 20, 30], "Numeric class": [1, 2, 3]}).to_file(
        lookup_file, driver="GPKG"
    )

    output = format_lc_data(
        input_file,
        dem_file,
        tmp_path / "output",
        lookup_file,
        "Grid value",
        "Numeric class",
        fill_nodata=False,
    )

    assert output == tmp_path / "output" / "lc.nc"
    assert not list(output.parent.glob("*classdefinition*"))
    expected = np.repeat(
        np.repeat(
            np.array([[1, 2, -9999], [3, 1, -9999]], dtype=np.int32),
            2,
            axis=0,
        ),
        2,
        axis=1,
    )
    with xr.open_dataset(output, decode_cf=False) as dataset:
        assert set(dataset.data_vars) >= {"land_cover"}
        assert dataset["land_cover"].dtype == np.dtype("int32")
        assert dataset["land_cover"].attrs["_FillValue"] == -9999
        np.testing.assert_array_equal(dataset["land_cover"].values, expected)
        np.testing.assert_allclose(dataset["x"].values, np.arange(102.5, 130, 5))
        np.testing.assert_allclose(dataset["y"].values, np.arange(217.5, 200, -5))


def test_format_lc_data_fills_nodata_inside_the_dem_domain(tmp_path: Path):
    """Unmapped and nodata cells default to their nearest classified neighbour."""
    input_file = tmp_path / "land_cover.tif"
    dem_file = tmp_path / "dem.tif"
    lookup_file = tmp_path / "lookup.gpkg"
    _write_raster(
        input_file,
        np.array([[10, 20, -9999, 99]], dtype=np.int32),
        cellsize=10,
    )
    _write_raster(dem_file, np.ones((1, 4), dtype=np.float32), cellsize=10)
    gpd.GeoDataFrame({"Grid value": [10, 20], "Numeric class": [1, 2]}).to_file(
        lookup_file, driver="GPKG"
    )

    output = format_lc_data(
        input_file,
        dem_file,
        tmp_path / "output",
        lookup_file,
        "Grid value",
        "Numeric class",
    )

    with xr.open_dataset(output, decode_cf=False) as dataset:
        np.testing.assert_array_equal(dataset["land_cover"].values, [[1, 2, 2, 2]])


def test_format_lc_data_requires_numeric_classes(tmp_path: Path):
    """Text labels are not guessed or converted to mHM class numbers."""
    input_file = tmp_path / "land_cover.tif"
    lookup_file = tmp_path / "lookup.gpkg"
    _write_raster(input_file, np.array([[10]], dtype=np.int32), cellsize=10)
    gpd.GeoDataFrame({"source": [10], "class": ["Forest"]}).to_file(
        lookup_file, driver="GPKG"
    )

    with pytest.raises(ValueError, match="non-numeric"):
        format_lc_data(
            input_file,
            input_file,
            tmp_path / "output",
            lookup_file,
            "source",
            "class",
        )


def test_pre_exports_format_lc_data():
    """The public pre package exposes the land-cover formatter."""
    assert pre.format_lc_data is format_lc_data
