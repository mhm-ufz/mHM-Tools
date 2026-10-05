"""Tests for manifest-based land-cover and soil formatting."""

from pathlib import Path

import geopandas as gpd
import netCDF4
import numpy as np
import pytest
import rasterio
import xarray as xr

from mhm_tools import pre
from mhm_tools.common.netcdf import DEFAULT_COMPLEVEL, NetcdfCompression
from mhm_tools.pre.format_lc_data import format_lc_periods
from mhm_tools.pre.format_soil import _bulk_density_unit, format_soil_horizons


def _write_raster(
    path: Path,
    values,
    *,
    cellsize: float = 1.0,
    crs="EPSG:32632",
    transform=None,
) -> None:
    values = np.asarray(values)
    transform = transform or rasterio.transform.from_origin(
        0.0, values.shape[0] * cellsize, cellsize, cellsize
    )
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        width=values.shape[1],
        height=values.shape[0],
        count=1,
        dtype=values.dtype,
        crs=crs,
        transform=transform,
        nodata=-9999,
    ) as dataset:
        dataset.write(values, 1)


@pytest.mark.parametrize("complevel", [None, 0, 7])
def test_format_lc_periods_maps_before_majority_and_writes_both_formats(
    tmp_path, complevel
):
    """Historical classes are mapped first and retain full datetime bounds."""
    input_path = tmp_path / "land-cover"
    input_path.mkdir()
    first = input_path / "first.tif"
    second = input_path / "second.tif"
    dem = tmp_path / "dem.tif"
    lookup = tmp_path / "lookup.gpkg"

    # Code 30 is the raw majority, while mapped class 1 is the class majority.
    _write_raster(
        first,
        np.array([10] * 5 + [20] * 5 + [30] * 6, dtype=np.int16).reshape(4, 4),
    )
    _write_raster(second, np.full((4, 4), 30, dtype=np.int16))
    _write_raster(dem, np.ones((1, 1), dtype=np.float32), cellsize=4.0)
    gpd.GeoDataFrame({"source": [10, 20, 30], "class": [1, 1, 2]}).to_file(
        lookup, driver="GPKG"
    )
    manifest = input_path / "historical-land-cover.csv"
    manifest.write_text(
        "StartDateTime,EndDateTime,FilePath\n"
        "2000-01-01T00:00:00,2005-07-01T12:00:00,first.tif\n"
        "2005-07-01T12:00:00,2010-01-01T00:00:00,second.tif\n",
        encoding="utf-8",
    )

    ascii_outputs = format_lc_periods(
        manifest,
        dem,
        tmp_path / "ascii",
        lookup,
        "source",
        "class",
        "asc",
    )
    assert [path.name for path in ascii_outputs] == [
        "lc_20000101T000000_20050701T120000.asc",
        "lc_20050701T120000_20100101T000000.asc",
    ]
    with rasterio.open(ascii_outputs[0]) as dataset:
        np.testing.assert_array_equal(dataset.read(1), [[1]])
        assert dataset.transform == rasterio.transform.from_origin(0, 4, 4, 4)
    with rasterio.open(ascii_outputs[1]) as dataset:
        np.testing.assert_array_equal(dataset.read(1), [[2]])

    netcdf_outputs = format_lc_periods(
        manifest,
        dem,
        tmp_path / "netcdf",
        lookup,
        "source",
        "class",
        "nc",
        compression=NetcdfCompression(complevel=complevel, significant_digits=3),
    )
    with netCDF4.Dataset(netcdf_outputs[0]) as raw:
        assert raw["land_cover"].filters()["complevel"] == (
            DEFAULT_COMPLEVEL if complevel is None else complevel
        )
        assert raw["land_cover"].quantization() is None
        assert all(raw[name].quantization() is None for name in raw.variables)
    assert [path.name for path in netcdf_outputs] == ["lc_periods.nc"]
    with xr.open_dataset(netcdf_outputs[0]) as dataset:
        assert dataset["land_cover"].dims == ("time", "y", "x")
        np.testing.assert_array_equal(dataset["land_cover"].values[:, 0, 0], [1, 2])
        np.testing.assert_array_equal(
            dataset["time"].values,
            np.array(
                ["2000-01-01T00:00:00", "2005-07-01T12:00:00"],
                dtype="datetime64[ns]",
            ),
        )
        np.testing.assert_array_equal(
            dataset["time_bnds"].values,
            np.array(
                [
                    ["2000-01-01T00:00:00", "2005-07-01T12:00:00"],
                    ["2005-07-01T12:00:00", "2010-01-01T00:00:00"],
                ],
                dtype="datetime64[ns]",
            ),
        )
        assert dataset["time"].attrs["bounds"] == "time_bnds"
        assert dataset["time"].attrs["standard_name"] == "time"
        assert dataset["land_cover"].attrs["units"] == "1"
        assert dataset.attrs["Conventions"].startswith("CF-")


@pytest.mark.parametrize("output_type", ["asc", "nc"])
def test_format_lc_periods_reprojects_to_dem_crs(tmp_path, output_type):
    """Historical land cover is warped to the exact DEM grid without osgeo."""
    input_path = tmp_path / "land-cover"
    input_path.mkdir()
    source = input_path / "period.tif"
    dem = tmp_path / "dem.tif"
    lookup = tmp_path / "lookup.gpkg"
    _write_raster(source, [[10, 20], [30, 40]], crs="EPSG:4326")
    bounds = rasterio.warp.transform_bounds("EPSG:4326", "EPSG:3857", 0, 0, 2, 2)
    cellsize = (bounds[2] - bounds[0]) / 2
    dem_transform = rasterio.transform.from_origin(
        bounds[0], bounds[3], cellsize, cellsize
    )
    _write_raster(
        dem,
        np.ones((2, 2), dtype=np.float32),
        crs="EPSG:3857",
        transform=dem_transform,
    )
    gpd.GeoDataFrame({"source": [10, 20, 30, 40], "class": [1, 2, 3, 4]}).to_file(
        lookup, driver="GPKG"
    )
    manifest = input_path / "periods.csv"
    manifest.write_text(
        "StartYear,EndYear,FilePath\n2000,2000,period.tif\n",
        encoding="utf-8",
    )

    (output,) = format_lc_periods(
        manifest,
        dem,
        tmp_path / output_type,
        lookup,
        "source",
        "class",
        output_type,
        resampling="nearest",
        fill_nodata=False,
    )

    if output_type == "asc":
        with rasterio.open(output) as dataset:
            assert dataset.crs.to_epsg() == 3857
            assert dataset.transform.almost_equals(dem_transform)
            np.testing.assert_array_equal(dataset.read(1), [[1, 2], [3, 4]])
    else:
        with xr.open_dataset(output, decode_cf=False) as dataset:
            output_crs = rasterio.crs.CRS.from_wkt(dataset["crs"].attrs["spatial_ref"])
            assert output_crs.to_epsg() == 3857
            np.testing.assert_array_equal(
                dataset["land_cover"].values, [[[1, 2], [3, 4]]]
            )


def test_format_lc_periods_rejects_gaps(tmp_path):
    """Datetime periods must form one ordered, continuous time axis."""
    input_path = tmp_path / "land-cover"
    input_path.mkdir()
    _write_raster(input_path / "first.tif", np.ones((1, 1), dtype=np.int16))
    _write_raster(input_path / "second.tif", np.ones((1, 1), dtype=np.int16))
    manifest = input_path / "format-data.csv"
    manifest.write_text(
        "start_date_time,end-date-time,file path\n"
        "2000-01-01,2005-01-01,first.tif\n"
        "2006-01-01,2010-01-01,second.tif\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="ordered, continuous and non-overlapping"):
        format_lc_periods(
            manifest,
            tmp_path / "missing-dem.tif",
            tmp_path / "output",
            tmp_path / "missing-lookup.gpkg",
            "source",
            "class",
        )


def _write_soil_manifest(input_path: Path, name: str = "format-data.txt") -> Path:
    header = (
        "Horizon,Upper Depth,Lower Depth,Clay Layer,Sand Layer,Silt Layer,"
        "Bulk Density Layer,Bulk Density Unit\n"
    )
    rows = (
        "1,0,100,clay1.tif,sand1.tif,silt1.tif,bd1.tif,kg/m3\n"
        "2,100,300,clay2.tif,sand2.tif,silt2.tif,bd2.tif,kg/m3\n"
    )
    manifest = input_path / name
    manifest.write_text(header + rows, encoding="utf-8")
    return manifest


def _write_soil_inputs(input_path: Path) -> None:
    values = {
        "clay1": [[2, 4], [0, 20]],
        "sand1": [[3, 2], [0, 30]],
        "silt1": [[5, 4], [0, 50]],
        "bd1": [[1300, 1400], [1500, 1300]],
        "clay2": [[1, 2], [3, 1]],
        "sand2": [[1, 4], [3, 1]],
        "silt2": [[3, 4], [4, 3]],
        "bd2": [[1500, 1600], [1700, 1500]],
    }
    for name, data in values.items():
        _write_raster(input_path / f"{name}.tif", np.asarray(data, dtype=np.float32))


def test_format_soil_horizons_writes_v5_profiles_and_normalizes_composition(
    tmp_path,
):
    """v5 classes describe full profiles and preserve an invalid component sum."""
    input_path = tmp_path / "soil"
    input_path.mkdir()
    manifest = _write_soil_manifest(input_path, "soil-horizons.txt")
    _write_soil_inputs(input_path)
    dem = tmp_path / "dem.tif"
    _write_raster(dem, np.ones((2, 2), dtype=np.float32))

    raster, definition = format_soil_horizons(manifest, dem, tmp_path / "v5", "asc")

    assert raster.name == "soil_class.asc"
    assert definition.name == "soil_classdefinition.txt"
    with rasterio.open(raster) as dataset:
        np.testing.assert_array_equal(dataset.read(1), [[1, 2], [-9999, 1]])
        assert dataset.transform == rasterio.transform.from_origin(0, 2, 1, 1)
    assert definition.read_text(encoding="utf-8").splitlines() == [
        "nSoil_Types 2",
        "MU_GLOBAL\tHORIZON\tUD[mm]\tLD[mm]\tCLAY[%]\tSAND[%]\tBD[gcm-3]",
        "1\t1\t0\t100\t20\t30\t1.3",
        "1\t2\t100\t300\t20\t20\t1.5",
        "2\t1\t0\t100\t40\t20\t1.4",
        "2\t2\t100\t300\t20\t40\t1.6",
    ]


@pytest.mark.parametrize("complevel", [None, 0, 7])
def test_format_soil_horizons_writes_v6_horizon_classes_and_mode1_lut(
    tmp_path, complevel
):
    """v6 retains per-horizon validity, depth bounds, and a mode-1 LUT."""
    input_path = tmp_path / "soil"
    input_path.mkdir()
    manifest = _write_soil_manifest(input_path)
    _write_soil_inputs(input_path)
    dem = tmp_path / "dem.tif"
    _write_raster(dem, np.ones((2, 2), dtype=np.float32))

    raster, definition = format_soil_horizons(
        manifest,
        dem,
        tmp_path / "v6",
        "nc",
        compression=NetcdfCompression(complevel=complevel, significant_digits=3),
    )

    with netCDF4.Dataset(raster) as raw:
        assert raw["soil_class"].filters()["complevel"] == (
            DEFAULT_COMPLEVEL if complevel is None else complevel
        )
        assert raw["soil_class"].quantization() is None
        assert all(raw[name].quantization() is None for name in raw.variables)
    assert raster.name == "soil_horizon_class.nc"
    assert definition.name == "soil_classdefinition_iFlag_soilDB_1.txt"
    with xr.open_dataset(raster, decode_cf=False) as dataset:
        assert dataset["soil_class"].dims == ("z", "y", "x")
        np.testing.assert_array_equal(
            dataset["soil_class"].values,
            [[[2, 5], [-9999, 2]], [[1, 3], [4, 1]]],
        )
        np.testing.assert_allclose(dataset["z"].values, [100, 300])
        np.testing.assert_allclose(dataset["z_bnds"].values, [[0, 100], [100, 300]])
        assert dataset["z"].attrs["bounds"] == "z_bnds"
        assert dataset["z"].attrs["positive"] == "down"
        assert dataset["soil_class"].attrs["_FillValue"] == -9999
        assert dataset.attrs["Conventions"].startswith("CF-")
    assert definition.read_text(encoding="utf-8").splitlines() == [
        "nSoil_Types 5",
        "ID\tCLAY[%]\tSAND[%]\tBD[gcm-3]",
        "1\t20\t20\t1.5",
        "2\t20\t30\t1.3",
        "3\t20\t40\t1.6",
        "4\t30\t30\t1.7",
        "5\t40\t20\t1.4",
    ]


def _write_gapped_soil_manifest(input_path: Path) -> Path:
    manifest = input_path / "format-data.txt"
    manifest.write_text(
        "Horizon,Upper Depth,Lower Depth,Clay Layer,Sand Layer,Silt Layer,"
        "Bulk Density Layer,Bulk Density Unit\n"
        "1,0,100,clay1.tif,sand1.tif,silt1.tif,bd1.tif,kg/m3\n"
        "2,100,300,clay2.tif,sand2.tif,silt2.tif,bd2.tif,kg/m3\n",
        encoding="utf-8",
    )
    return manifest


def _write_gapped_soil_inputs(input_path: Path) -> None:
    """Write a 1x3 profile whose first horizon misses clay in the last cell."""
    values = {
        "clay1": [[20, 40, -9999]],
        "sand1": [[30, 20, 20]],
        "silt1": [[50, 40, 40]],
        "bd1": [[1300, 1400, 1400]],
        "clay2": [[10, 10, 10]],
        "sand2": [[40, 40, 40]],
        "silt2": [[50, 50, 50]],
        "bd2": [[1500, 1500, 1500]],
    }
    for name, data in values.items():
        _write_raster(input_path / f"{name}.tif", np.asarray(data, dtype=np.float32))


@pytest.mark.parametrize("output_type", ["nc", "asc"])
def test_format_soil_horizons_fills_layer_nodata_from_nearest(tmp_path, output_type):
    """A hole in one input layer no longer drops the cell from the output."""
    input_path = tmp_path / "soil"
    input_path.mkdir()
    manifest = _write_gapped_soil_manifest(input_path)
    _write_gapped_soil_inputs(input_path)
    dem = tmp_path / "dem.tif"
    _write_raster(dem, np.ones((1, 3), dtype=np.float32))

    raster, _ = format_soil_horizons(
        manifest,
        dem,
        tmp_path / "filled",
        output_type,
        resampling="nearest",
    )

    if output_type == "asc":
        with rasterio.open(raster) as dataset:
            # The gap takes clay from its nearest neighbour and shares its class.
            np.testing.assert_array_equal(dataset.read(1), [[1, 2, 2]])
    else:
        with xr.open_dataset(raster, decode_cf=False) as dataset:
            np.testing.assert_array_equal(
                dataset["soil_class"].values,
                [[[2, 3, 3]], [[1, 1, 1]]],
            )


@pytest.mark.parametrize("output_type", ["nc", "asc"])
def test_format_soil_horizons_keeps_layer_nodata_when_filling_is_off(
    tmp_path, output_type
):
    """Opting out of filling keeps the historical hole in the output."""
    input_path = tmp_path / "soil"
    input_path.mkdir()
    manifest = _write_gapped_soil_manifest(input_path)
    _write_gapped_soil_inputs(input_path)
    dem = tmp_path / "dem.tif"
    _write_raster(dem, np.ones((1, 3), dtype=np.float32))

    raster, _ = format_soil_horizons(
        manifest,
        dem,
        tmp_path / "unfilled",
        output_type,
        resampling="nearest",
        fill_nodata=False,
    )

    if output_type == "asc":
        with rasterio.open(raster) as dataset:
            np.testing.assert_array_equal(dataset.read(1), [[1, 2, -9999]])
    else:
        with xr.open_dataset(raster, decode_cf=False) as dataset:
            np.testing.assert_array_equal(
                dataset["soil_class"].values,
                [[[2, 3, -9999]], [[1, 1, 1]]],
            )


@pytest.mark.parametrize("fill_nodata", [True, False])
def test_format_lc_periods_fills_period_nodata_from_nearest(tmp_path, fill_nodata):
    """Land-cover gaps inside the DEM domain follow the fill-nodata switch."""
    input_path = tmp_path / "land-cover"
    input_path.mkdir()
    period = input_path / "first.tif"
    dem = tmp_path / "dem.tif"
    lookup = tmp_path / "lookup.gpkg"
    _write_raster(period, np.array([[10, 20, -9999]], dtype=np.int16))
    _write_raster(dem, np.ones((1, 3), dtype=np.float32))
    gpd.GeoDataFrame({"source": [10, 20], "class": [1, 2]}).to_file(
        lookup, driver="GPKG"
    )
    manifest = input_path / "format-data.csv"
    manifest.write_text(
        "StartYear,EndYear,FilePath\n2000,2004,first.tif\n",
        encoding="utf-8",
    )

    (output,) = format_lc_periods(
        manifest,
        dem,
        tmp_path / f"output-{fill_nodata}",
        lookup,
        "source",
        "class",
        "asc",
        resampling="nearest",
        fill_nodata=fill_nodata,
    )

    expected = [[1, 2, 2]] if fill_nodata else [[1, 2, -9999]]
    with rasterio.open(output) as dataset:
        np.testing.assert_array_equal(dataset.read(1), expected)


@pytest.mark.parametrize(
    ("unit", "canonical", "factor"),
    [
        ("g/cm3", "g/cm3", 1.0),
        ("kg m^-3", "kg/m3", 1.0e-3),
        ("cg/cm³", "cg/cm3", 1.0e-2),
        ("mg/cm3", "mg/cm3", 1.0e-3),
        ("g/dm3", "g/dm3", 1.0e-3),
        ("kg/dm3", "kg/dm3", 1.0),
    ],
)
def test_bulk_density_units_convert_to_g_per_cm3(unit, canonical, factor):
    """Manifest units have explicit, tested conversions to g/cm3."""
    assert _bulk_density_unit(unit) == (canonical, factor)


def test_pre_exports_manifest_formatters():
    """Both manifest-based formatters are public Python APIs."""
    assert pre.format_lc_periods is format_lc_periods
    assert pre.format_soil_horizons is format_soil_horizons
