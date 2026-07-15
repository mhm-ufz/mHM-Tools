"""Tests for shared raster and CRS file handling."""

import numpy as np
import pytest
import rioxarray  # noqa: F401
import xarray as xr
from rasterio.transform import from_origin

from mhm_tools.common.file_handler import (
    MissingCRSError,
    align_raster_to_reference,
    get_raster_data,
    get_xarray_ds_from_file,
    read_ascii_to_xarray,
    resolve_crs,
    write_xarray_to_file,
)


def _raster(*, crs="EPSG:32645", nodata=-9999):
    data = xr.DataArray(
        np.arange(12, dtype=np.int32).reshape(3, 4),
        dims=("y", "x"),
        coords={"y": [250.0, 150.0, 50.0], "x": [50.0, 150.0, 250.0, 350.0]},
        name="classes",
        attrs={"nodata_value": nodata},
    ).rio.set_spatial_dims(x_dim="x", y_dim="y")
    return data.rio.write_crs(crs) if crs else data


def test_geotiff_roundtrip_preserves_grid_crs_and_zero_nodata(tmp_path):
    """GeoTIFF round trips preserve its grid, CRS, values, and zero nodata."""
    data = _raster(nodata=0)
    output = tmp_path / "classes.tif"

    write_xarray_to_file(data, output)

    dataset = get_xarray_ds_from_file(output)
    assert "spatial_ref" in dataset.coords
    result = get_raster_data(output)
    assert result.rio.crs == data.rio.crs
    assert result.rio.transform() == data.rio.transform()
    assert result.rio.nodata == 0
    np.testing.assert_array_equal(result.values, data.values)


def test_geotiff_write_requires_crs_before_overwriting(tmp_path):
    """A missing CRS fails before an existing GeoTIFF is changed."""
    output = tmp_path / "classes.tif"
    output.write_bytes(b"keep")

    with pytest.raises(MissingCRSError):
        write_xarray_to_file(_raster(crs=None), output)

    assert output.read_bytes() == b"keep"


def test_ascii_prj_roundtrip_and_crsless_compatibility(tmp_path):
    """ASCII uses optional PRJ metadata without breaking CRS-less files."""
    output = tmp_path / "classes.asc"
    write_xarray_to_file(_raster(), output)

    assert output.with_suffix(".prj").is_file()
    dataset = read_ascii_to_xarray(output)
    assert resolve_crs(dataset) == _raster().rio.crs
    assert get_raster_data(output).rio.transform() == _raster().rio.transform()

    crsless = tmp_path / "crsless.asc"
    write_xarray_to_file(_raster(crs=None), crsless)
    assert crsless.is_file()
    assert not crsless.with_suffix(".prj").exists()
    with pytest.raises(MissingCRSError):
        get_raster_data(crsless)
    assert get_raster_data(crsless, crs="EPSG:4326").rio.crs.to_epsg() == 4326


@pytest.mark.parametrize("suffix", [".asc", ".tif"])
def test_raster_writers_normalize_spatial_dimension_order(tmp_path, suffix):
    """Raster output is row-major even when input dimensions are X then Y."""
    data = _raster().transpose("x", "y")
    output = tmp_path / f"classes{suffix}"

    write_xarray_to_file(data, output)

    result = get_raster_data(output)
    try:
        assert result.dims == (result.rio.y_dim, result.rio.x_dim)
        np.testing.assert_array_equal(result.values, data.transpose("y", "x").values)
    finally:
        result.close()


def test_ascii_reader_preserves_single_row_grid(tmp_path):
    """NumPy's squeezed one-row result is restored to the declared grid shape."""
    output = tmp_path / "single_row.asc"
    output.write_text(
        "ncols 3\n"
        "nrows 1\n"
        "xllcorner 0\n"
        "yllcorner 0\n"
        "cellsize 1\n"
        "NODATA_value -9999\n"
        "1 2 3\n",
        encoding="utf-8",
    )

    result = read_ascii_to_xarray(output)

    assert result["data"].shape == (1, 3)
    np.testing.assert_array_equal(result["data"].values, [[1, 2, 3]])
    raster = get_raster_data(output, crs="EPSG:32645")
    try:
        assert raster.rio.transform() == from_origin(0, 1, 1, 1)
    finally:
        raster.close()


def test_single_cell_ascii_roundtrip_uses_stored_transform(tmp_path):
    """A 1x1 ASCII grid retains the header cell size when rewritten."""
    source = tmp_path / "single_cell.asc"
    source.write_text(
        "ncols 1\n"
        "nrows 1\n"
        "xllcorner 10\n"
        "yllcorner 20\n"
        "cellsize 5\n"
        "NODATA_value -9999\n"
        "7\n",
        encoding="utf-8",
    )
    data = read_ascii_to_xarray(source)
    output = tmp_path / "single_cell_copy.asc"

    write_xarray_to_file(data, output)

    result = get_raster_data(output, crs="EPSG:32645")
    try:
        assert result.rio.transform() == from_origin(10, 25, 5, 5)
        np.testing.assert_array_equal(result.values, [[7]])
    finally:
        result.close()


def test_ascii_writer_normalizes_ascending_y(tmp_path):
    """CF-style ascending Y coordinates are written north-up."""
    data = _raster().isel(y=slice(None, None, -1))
    output = tmp_path / "ascending_y.asc"

    write_xarray_to_file(data, output)

    result = get_raster_data(output)
    try:
        expected = data.isel(y=slice(None, None, -1))
        np.testing.assert_array_equal(result.values, expected.values)
        np.testing.assert_array_equal(
            result[result.rio.y_dim].values,
            expected[expected.rio.y_dim].values,
        )
    finally:
        result.close()


def test_crsless_ascii_write_removes_uppercase_prj(tmp_path):
    """A stale uppercase CRS sidecar is not retained for CRS-less output."""
    output = tmp_path / "classes.asc"
    uppercase_prj = output.with_suffix(".PRJ")
    uppercase_prj.write_text("EPSG:4326", encoding="utf-8")

    write_xarray_to_file(_raster(crs=None), output)

    assert not uppercase_prj.exists()


def test_ascii_rejects_unequal_cell_sizes(tmp_path):
    """ASCII cannot preserve a grid with separate X and Y resolutions."""
    data = xr.DataArray(
        np.ones((2, 2), dtype=np.int32),
        dims=("y", "x"),
        coords={"y": [3.0, 1.0], "x": [0.5, 1.5]},
        name="classes",
    )

    with pytest.raises(ValueError, match="equal X and Y"):
        write_xarray_to_file(data, tmp_path / "classes.asc")


def test_explicit_crs_conflict_is_rejected(tmp_path):
    """Explicit CRS assignment cannot override embedded metadata."""
    output = tmp_path / "classes.asc"
    write_xarray_to_file(_raster(), output)

    with pytest.raises(ValueError, match="conflicts"):
        get_raster_data(output, crs="EPSG:4326")


def test_cf_netcdf_grid_mapping_is_not_a_payload(tmp_path):
    """A CF grid mapping is retained and excluded from payload selection."""
    output = tmp_path / "classes.nc"
    write_xarray_to_file(_raster(), output, engine="h5netcdf")

    result = get_raster_data(output)
    assert result.name == "classes"
    assert result.rio.crs.to_epsg() == 32645
    assert result.rio.transform() == _raster().rio.transform()
    assert result._close is not None


def test_netcdf_global_crs_is_applied_to_payload(tmp_path):
    """Dataset-level CRS metadata remains available after payload selection."""
    output = tmp_path / "global_crs.nc"
    _raster(crs=None).to_dataset().assign_attrs(crs="EPSG:32645").to_netcdf(output)

    result = get_raster_data(output)
    try:
        assert result.rio.crs.to_epsg() == 32645
    finally:
        result.close()


def test_decoded_integer_netcdf_writes_safe_geotiff(tmp_path):
    """Decoded NaN cells are restored to integer nodata for GeoTIFF output."""
    data = _raster()
    data.values[0, 0] = -9999
    netcdf = tmp_path / "classes.nc"
    output = tmp_path / "classes.tif"
    encoding = {"classes": {"dtype": "int32", "_FillValue": -9999}}
    write_xarray_to_file(data, netcdf, encoding=encoding, engine="h5netcdf")

    decoded = get_raster_data(netcdf)
    try:
        assert np.isnan(decoded.values[0, 0])
        write_xarray_to_file(decoded, output)
    finally:
        decoded.close()

    result = get_raster_data(output)
    assert result.dtype == np.dtype("int32")
    assert result.rio.nodata == -9999
    assert result.values[0, 0] == -9999


def test_geotiff_rejects_uint64_with_negative_nodata(tmp_path):
    """GeoTIFF conversion does not silently lose uint64 category precision."""
    data = _raster().astype(np.uint64)
    data.values[0, 0] = 2**63 + 1
    output = tmp_path / "uint64.tif"

    with pytest.raises(ValueError, match="no safe integer dtype"):
        write_xarray_to_file(data, output)

    assert not output.exists()


def test_align_raster_matches_exact_reference_grid():
    """Nearest-neighbour alignment adopts the complete reference grid."""
    source = _raster()
    reference = xr.DataArray(
        np.zeros((6, 8), dtype=np.float32),
        dims=("lat", "lon"),
        coords={
            "lat": [275.0, 225.0, 175.0, 125.0, 75.0, 25.0],
            "lon": [25.0, 75.0, 125.0, 175.0, 225.0, 275.0, 325.0, 375.0],
        },
        name="dem",
    ).rio.set_spatial_dims(x_dim="lon", y_dim="lat")
    reference = reference.rio.write_crs(source.rio.crs)

    result = align_raster_to_reference(source, reference)

    assert result.dims == reference.dims
    assert result.shape == reference.shape
    assert result.rio.crs == reference.rio.crs
    assert result.rio.transform() == reference.rio.transform()
    np.testing.assert_array_equal(result["lat"], reference["lat"])
    np.testing.assert_array_equal(result["lon"], reference["lon"])


def test_align_raster_exact_grid_uses_reference_dimension_names():
    """An already matching grid still adopts reference dimension names."""
    source = _raster()
    reference = source.rename({"y": "lat", "x": "lon"}).rename("dem")
    reference = reference.rio.set_spatial_dims(x_dim="lon", y_dim="lat")

    result = align_raster_to_reference(source, reference)

    assert result.dims == ("lat", "lon")
    np.testing.assert_array_equal(result.values, source.values)


def test_align_raster_normalizes_source_nodata_on_exact_grid():
    """A no-op grid match still replaces the source nodata sentinel."""
    source = _raster(nodata=-32768)
    source.values[0, 0] = -32768
    reference = source.rename("dem")

    result = align_raster_to_reference(source, reference)

    assert result.values[0, 0] == -9999
    assert result.rio.nodata == -9999


def test_align_raster_respects_spatial_axis_order():
    """Square X/Y-first arrays are not relabelled without transposition."""
    source = xr.DataArray(
        np.array([[1, 2], [3, 4]], dtype=np.int32),
        dims=("x", "y"),
        coords={"x": [0.5, 1.5], "y": [1.5, 0.5]},
        name="classes",
    ).rio.set_spatial_dims(x_dim="x", y_dim="y")
    source = source.rio.write_crs("EPSG:32632")
    reference = xr.DataArray(
        np.zeros((2, 2), dtype=np.float32),
        dims=("y", "x"),
        coords={"x": source.x, "y": source.y},
        name="dem",
    ).rio.write_crs(source.rio.crs)

    result = align_raster_to_reference(source, reference)

    np.testing.assert_array_equal(result.values, source.transpose("y", "x").values)


def test_align_raster_preserves_large_unsigned_categories():
    """Adding negative nodata does not overflow uint32 category values."""
    source = _raster().astype(np.uint32)
    source.values[0, 0] = 4_000_000_000
    reference = source.rename("dem")

    result = align_raster_to_reference(source, reference)

    assert result.dtype == np.dtype("int64")
    assert result.values[0, 0] == 4_000_000_000


def test_align_raster_reprojects_to_reference_crs():
    """Categorical alignment performs a real cross-CRS reprojection."""
    source = xr.DataArray(
        np.array([[1, 2], [3, 4]], dtype=np.int32),
        dims=("lat", "lon"),
        coords={"lat": [50.75, 50.25], "lon": [9.25, 9.75]},
        name="classes",
    ).rio.set_spatial_dims(x_dim="lon", y_dim="lat")
    source = source.rio.write_crs("EPSG:4326")
    reference = source.rio.reproject("EPSG:3857").rename("dem")

    result = align_raster_to_reference(source, reference)

    assert result.rio.crs == reference.rio.crs
    assert result.rio.transform() == reference.rio.transform()
    assert result.shape == reference.shape
