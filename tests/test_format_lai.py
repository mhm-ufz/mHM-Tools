"""Tests for lookup-based LAI data formatting."""

from pathlib import Path

import numpy as np
import rasterio
import xarray as xr

from mhm_tools import pre
from mhm_tools.pre.format_lai import format_lai_data, write_lai_classdefinition


def _write_raster(path: Path, values, *, dtype: str) -> None:
    values = np.asarray(values, dtype=dtype)
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        width=values.shape[1],
        height=values.shape[0],
        count=1,
        dtype=dtype,
        crs="EPSG:32632",
        transform=rasterio.transform.from_origin(100.0, 220.0, 10.0, 10.0),
        nodata=-9999,
    ) as dataset:
        dataset.write(values, 1)


def _write_lookup(path: Path) -> None:
    months = ",".join(
        (
            "Jan",
            "Feb",
            "Mar",
            "Apr",
            "May",
            "Jun",
            "Jul",
            "Aug",
            "Sep",
            "Oct",
            "Nov",
            "Dec",
        )
    )
    path.write_text(
        f"ID,LAND-USE,{months}, LAI_CLASS\n"
        "10,Forest,1,2,3,4,5,6,7,8,9,10,11,12,2\n"
        "20,Water,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,1\n"
        "30,Forest,1,2,3,4,5,6,7,8,9,10,11,12,2\n",
        encoding="utf-8",
    )


def _expected_definition() -> str:
    return (
        "NoLAIclasses           2\n"
        "ID   LAND-USE                  Jan.   Feb.    Mar.    Apr.    May    "
        "Jun.    Jul.    Aug.    Sep.    Oct.    Nov.    Dec.\n"
        " 1    Water                     0.1     0.1     0.1     0.1     "
        "0.1     0.1     0.1     0.1     0.1     0.1     0.1     0.1\n"
        " 2    Forest                    1       2       3       4       "
        "5       6       7       8       9       10      11      12\n"
    )


def test_format_lai_data_reads_text_lookup_and_writes_outputs(tmp_path: Path):
    """LAI classes use the shared DEM grid and unique class definitions."""
    input_file = tmp_path / "lai_raw.tif"
    dem_file = tmp_path / "dem.tif"
    lookup_file = tmp_path / "lai_lookup.txt"
    _write_raster(input_file, [[10, 20, -9999], [30, 10, 99]], dtype="int32")
    _write_raster(dem_file, np.ones((2, 3)), dtype="float32")
    _write_lookup(lookup_file)

    output = format_lai_data(
        input_file,
        dem_file,
        tmp_path / "output",
        lookup_file,
        "ID",
        "LAI_CLASS",
        fill_nodata=False,
    )

    assert output == tmp_path / "output" / "lai_class.nc"
    definition = tmp_path / "output" / "LAI_classdefinition.txt"
    assert definition.read_text(encoding="utf-8") == _expected_definition()
    with xr.open_dataset(output, decode_cf=False) as dataset:
        np.testing.assert_array_equal(
            dataset["lai_class"].values,
            [[2, 1, -9999], [2, 2, -9999]],
        )
        np.testing.assert_allclose(dataset["x"].values, [105, 115, 125])
        np.testing.assert_allclose(dataset["y"].values, [215, 205])


def test_lai_writer_and_public_exports(tmp_path: Path):
    """The standalone writer and lazy pre exports expose the LAI API."""
    lookup_file = tmp_path / "lai_lookup.txt"
    output_file = tmp_path / "definitions" / "LAI_classdefinition.txt"
    _write_lookup(lookup_file)

    assert (
        write_lai_classdefinition(lookup_file, output_file, "lai class") == output_file
    )
    assert output_file.read_text(encoding="utf-8") == _expected_definition()
    assert pre.format_lai_data is format_lai_data
    assert pre.write_lai_classdefinition is write_lai_classdefinition
