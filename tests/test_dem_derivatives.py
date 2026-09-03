"""Tests for the DEM derivative command."""

import numpy as np
import pytest
import rioxarray  # noqa: F401
import xarray as xr

from mhm_tools.pre.dem_derivatives import (
    DERIVATIVE_ATTRS,
    NO_DATA,
    create_dem_derivatives,
    horn_aspect,
)


def _write_dem(path, values, *, crs="EPSG:32632", cellsize=100.0, nodata=None):
    """Write a small north-up GeoTIFF DEM and return its path."""
    values = np.asarray(values, dtype="float64")
    rows, cols = values.shape
    data = xr.DataArray(
        values,
        dims=("y", "x"),
        coords={
            "y": 2000.0 - cellsize * (np.arange(rows) + 0.5),
            "x": 1000.0 + cellsize * (np.arange(cols) + 0.5),
        },
        name="dem",
    ).rio.set_spatial_dims(x_dim="x", y_dim="y")
    data = data.rio.write_crs(crs)
    if nodata is not None:
        data = data.rio.write_nodata(nodata)
    data.rio.to_raster(path)
    return path


def _slope_dem(rows=5, cols=5):
    """Return a plane dropping to the east, with one pit."""
    values = 50.0 - np.arange(cols, dtype="float64") + np.arange(rows)[:, None]
    values[1, 1] = 30.0
    return values


def test_netcdf_holds_every_derivative_with_grid_and_crs(tmp_path):
    dem = _write_dem(tmp_path / "dem.tif", _slope_dem())
    (output,) = create_dem_derivatives(dem, tmp_path / "out")

    assert output.name == "dem_derivatives.nc"
    with xr.open_dataset(output, decode_coords="all") as ds:
        for name in DERIVATIVE_ATTRS:
            assert ds[name].shape == (5, 5)
        assert {"lat", "lon", "x", "y", "crs"}.issubset(set(ds.variables))
        assert "32632" in ds["crs"].attrs["crs_wkt"] or "UTM zone 32N" in (
            ds["crs"].attrs["crs_wkt"]
        )
        # The pit is raised to its lowest neighbour, the rest is untouched.
        assert ds["dem"].values[1, 1] == pytest.approx(30.0)
        assert ds["dem_filled"].values[1, 1] == pytest.approx(48.0)
        assert ds["dem_filled"].values[1, 0] == pytest.approx(51.0)
        # facc counts cells, so every cell drains at least itself.
        assert ds["facc"].min() >= 1


def test_geographic_dem_writes_lonlat_without_projected_axes(tmp_path):
    dem = _write_dem(tmp_path / "dem.tif", _slope_dem(), crs="EPSG:4326", cellsize=0.01)
    (output,) = create_dem_derivatives(dem, tmp_path / "out.nc")

    assert output.name == "out.nc"
    with xr.open_dataset(output, decode_coords="all") as ds:
        assert ds["x"].attrs["units"] == "degrees"
        # On a geographic grid lon/lat simply restate the axes.
        assert ds["lon"].values[0] == pytest.approx(ds["x"].values)
        assert ds["lat"].values[:, 0] == pytest.approx(ds["y"].values)


@pytest.mark.parametrize("extension", ["asc", "tif"])
def test_per_layer_output_keeps_the_source_grid(tmp_path, extension):
    dem = _write_dem(tmp_path / "dem.tif", _slope_dem())
    written = create_dem_derivatives(dem, tmp_path / "out", extension)

    assert sorted(p.name for p in written) == sorted(
        f"{name}.{extension}" for name in DERIVATIVE_ATTRS
    )
    filled = next(p for p in written if p.stem == "dem_filled")
    with xr.open_dataarray(filled, engine="rasterio").squeeze() as data:
        assert data.shape == (5, 5)
        assert data.rio.transform().a == pytest.approx(100.0)
        assert data.rio.transform().e == pytest.approx(-100.0)
        # Row 0 is the northern row, as it is in the source.
        assert data.values[0] == pytest.approx([50.0, 49.0, 48.0, 47.0, 46.0])


def test_ascii_output_carries_the_grid_header_and_projection(tmp_path):
    dem = _write_dem(tmp_path / "dem.tif", _slope_dem())
    written = create_dem_derivatives(dem, tmp_path / "out", "asc")

    slope = next(p for p in written if p.stem == "slope")
    header = dict(line.split() for line in slope.read_text().splitlines()[:6])
    assert int(header["ncols"]) == 5
    assert int(header["nrows"]) == 5
    assert float(header["xllcorner"]) == pytest.approx(1000.0)
    assert float(header["yllcorner"]) == pytest.approx(1500.0)
    assert float(header["cellsize"]) == pytest.approx(100.0)
    assert float(header["nodata_value"]) == pytest.approx(NO_DATA)
    assert "UTM zone 32N" in slope.with_suffix(".prj").read_text()


def test_nodata_cells_stay_nodata_in_every_derivative(tmp_path):
    values = _slope_dem()
    values[4, 4] = NO_DATA
    dem = _write_dem(tmp_path / "dem.tif", values, nodata=NO_DATA)
    (output,) = create_dem_derivatives(dem, tmp_path / "out")

    # Read raw: CF decoding would turn the declared _FillValue back into NaN.
    with xr.open_dataset(output, mask_and_scale=False) as ds:
        for name in DERIVATIVE_ATTRS:
            assert ds[name].values[4, 4] == pytest.approx(NO_DATA)


def test_unsupported_extension_is_rejected(tmp_path):
    dem = _write_dem(tmp_path / "dem.tif", _slope_dem())
    with pytest.raises(ValueError, match="Unsupported output extension"):
        create_dem_derivatives(dem, tmp_path / "out", "jpg")


def test_dem_without_crs_is_rejected(tmp_path):
    from mhm_tools.common.crs_handler import MissingCRSError

    path = tmp_path / "plain.asc"
    path.write_text(
        "ncols 2\nnrows 2\nxllcorner 0\nyllcorner 0\n"
        "cellsize 1\nnodata_value -9999\n1 2\n3 4\n"
    )
    with pytest.raises(MissingCRSError):
        create_dem_derivatives(path, tmp_path / "out")


@pytest.mark.parametrize(
    ("gradient", "expected"),
    [
        ("east", 90.0),  # ground falls towards +x
        ("south", 180.0),
        ("west", 270.0),
        ("north", 0.0),
    ],
)
def test_horn_aspect_reports_the_downhill_compass_bearing(gradient, expected):
    from rasterio.transform import from_origin

    rows, cols = 5, 5
    axis = np.arange(cols, dtype="float64")
    surfaces = {
        "east": -np.tile(axis, (rows, 1)),
        "west": np.tile(axis, (rows, 1)),
        "south": -np.tile(axis[:rows][:, None], (1, cols)),
        "north": np.tile(axis[:rows][:, None], (1, cols)),
    }
    aspect = horn_aspect(surfaces[gradient], from_origin(0.0, 5.0, 1.0, 1.0))
    assert aspect[2, 2] == pytest.approx(expected)


def test_horn_aspect_reports_nodata_on_flat_ground():
    from rasterio.transform import from_origin

    aspect = horn_aspect(np.full((3, 3), 12.0), from_origin(0.0, 3.0, 1.0, 1.0))
    assert np.all(aspect == NO_DATA)
