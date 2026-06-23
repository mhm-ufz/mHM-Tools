from pathlib import Path

import numpy as np
import xarray as xr

from mhm_tools.common.esri_grid import read_header
from mhm_tools.common.resolution_handler import Resolution
from mhm_tools.common.xarray_utils import get_ds_extend
from mhm_tools.pre.crop_mhm_setup import crop_file, regrid_mask


def test_regrid_mask_snaps_same_resolution_shifted_coordinates():
    target_lon = np.array([0.0, 1.0, 2.0])
    target_lat = np.array([3.0, 2.0, 1.0])
    mask = xr.DataArray(
        np.array(
            [
                [1.0, 0.0, 1.0],
                [1.0, 1.0, 0.0],
                [0.0, 1.0, 1.0],
            ]
        ),
        dims=("lat", "lon"),
        coords={
            "lat": target_lat + 1e-4,
            "lon": target_lon + 1e-4,
        },
        name="mask",
    )

    regridded = regrid_mask(
        mask_ds=mask,
        lon_key_mask="lon",
        lat_key_mask="lat",
        target_lon=target_lon,
        target_lat=target_lat,
        lon_key_target="lon",
        lat_key_target="lat",
    )

    assert regridded.dims == ("lat", "lon")
    assert np.allclose(regridded["lon"].values, target_lon)
    assert np.allclose(regridded["lat"].values, target_lat)
    assert np.array_equal(regridded.values, mask.values)


def test_crop_file_masking_snaps_shifted_mask_coordinates_without_extra_dims(tmp_path):
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"
    input_dir.mkdir()
    output_dir.mkdir()
    input_file = input_dir / "dem.nc"

    lon = np.array([0.0, 1.0, 2.0, 3.0])
    lat = np.array([3.0, 2.0, 1.0, 0.0])
    dem = np.arange(16, dtype=float).reshape(len(lat), len(lon))
    input_ds = xr.Dataset(
        {"dem": (("lat", "lon"), dem)},
        coords={"lat": lat, "lon": lon},
    )
    input_ds.to_netcdf(input_file)

    mask_values = np.ones_like(dem)
    mask_values[1, 1] = 0.0
    mask_ds = xr.Dataset(
        {"mask": (("latitude", "longitude"), mask_values)},
        coords={
            "latitude": lat + 1e-4,
            "longitude": lon + 1e-4,
        },
    )

    crop_file(
        input_file=input_file,
        mask_ds=mask_ds,
        latslice=slice(3.0, 1.0),
        lonslice=slice(0.0, 2.0),
        output_path=output_dir,
        input_path=input_dir,
        overwrite=True,
        available_mem_gib=3,
        resolutions=Resolution(l0=1.0),
    )

    with xr.open_dataset(output_dir / Path(input_file).name) as output_ds:
        output = output_ds.load()

    assert output["dem"].dims == ("lat", "lon")
    assert "latitude" not in output.dims
    assert "longitude" not in output.dims
    assert np.allclose(output["lon"].values, np.array([0.0, 1.0, 2.0]))
    assert np.allclose(output["lat"].values, np.array([3.0, 2.0, 1.0]))
    assert np.isnan(output["dem"].sel(lat=2.0, lon=1.0).item())
    assert output["dem"].sel(lat=3.0, lon=0.0).item() == dem[0, 0]


def test_crop_file_selects_matching_resolution_mask_variable(tmp_path):
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"
    input_dir.mkdir()
    output_dir.mkdir()
    input_file = input_dir / "dem.nc"

    lon = np.array([0.0, 1.0, 2.0, 3.0])
    lat = np.array([3.0, 2.0, 1.0, 0.0])
    dem = np.arange(16, dtype=float).reshape(len(lat), len(lon))
    xr.Dataset(
        {"dem": (("lat", "lon"), dem)},
        coords={"lat": lat, "lon": lon},
    ).to_netcdf(input_file)

    coarse_mask = np.zeros((2, 2), dtype=float)
    matching_mask = np.ones_like(dem)
    matching_mask[1, 1] = 0.0
    mask_ds = xr.Dataset(
        {
            "coarse_mask": (("lat_l2", "lon_l2"), coarse_mask),
            "matching_mask": (("latitude", "longitude"), matching_mask),
        },
        coords={
            "lat_l2": np.array([2.5, 0.5]),
            "lon_l2": np.array([0.5, 2.5]),
            "latitude": lat,
            "longitude": lon,
        },
    )

    crop_file(
        input_file=input_file,
        mask_ds=mask_ds,
        latslice=slice(3.0, 1.0),
        lonslice=slice(0.0, 2.0),
        output_path=output_dir,
        input_path=input_dir,
        overwrite=True,
        available_mem_gib=3,
        resolutions=Resolution(l0=1.0, l2=2.0),
    )

    with xr.open_dataset(output_dir / input_file.name) as output_ds:
        output = output_ds.load()

    assert output["dem"].dims == ("lat", "lon")
    assert np.isnan(output["dem"].sel(lat=2.0, lon=1.0).item())
    assert output["dem"].sel(lat=3.0, lon=0.0).item() == dem[0, 0]
    assert "lat_l2" not in output.dims
    assert "lon_l2" not in output.dims


def test_crop_file_selects_finer_mask_variable_when_no_match_exists(tmp_path):
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"
    input_dir.mkdir()
    output_dir.mkdir()
    input_file = input_dir / "dem.nc"

    lon = np.array([-2, -1, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0])
    lat = np.array([4.0, 3.0, 2.0, 1.0, 0.0, -1.0])
    dem = np.arange(len(lon) * len(lat), dtype=float).reshape(len(lat), len(lon))
    xr.Dataset(
        {"dem": (("lat", "lon"), dem)},
        coords={"lat": lat, "lon": lon},
    ).to_netcdf(input_file)

    coarse_mask = np.zeros((3, 3), dtype=float)
    fine_lon = np.array([-0.25, 0.25, 0.75, 1.25, 1.75, 2.25, 2.75, 3.25])
    fine_lat = np.array([3.25, 2.75, 2.25, 1.75, 1.25, 0.75, 0.25, -0.25])
    fine_mask = np.ones((len(fine_lat), len(fine_lon)), dtype=float)
    mask_ds = xr.Dataset(
        {
            "coarse_mask": (("lat_l2", "lon_l2"), coarse_mask),
            "fine_mask": (("latitude", "longitude"), fine_mask),
        },
        coords={
            "lat_l2": np.array([3.5, 2.5, 0.5]),
            "lon_l2": np.array([0.5, 2.5, 3.5]),
            "latitude": fine_lat,
            "longitude": fine_lon,
        },
    )
    resolutions = Resolution(l0=0.5, l1=1.0, l2=2.0)
    (
        lon_min_target_grid,
        lon_max_target_grid,
        lat_min_target_grid,
        lat_max_target_grid,
    ) = get_ds_extend(mask_ds, "fine_mask", resolutions=resolutions)
    crop_file(
        input_file=input_file,
        mask_ds=mask_ds,
        latslice=slice(lat_max_target_grid, lat_min_target_grid),
        lonslice=slice(lon_min_target_grid, lon_max_target_grid),
        output_path=output_dir,
        input_path=input_dir,
        overwrite=True,
        available_mem_gib=3,
        resolutions=resolutions,
    )

    with xr.open_dataset(output_dir / input_file.name) as output_ds:
        output = output_ds.load()

    assert output["dem"].dims == ("lat", "lon")
    assert np.array_equal(output["dem"].values, dem[1:5, 2:6])
    assert "lat_l2" not in output.dims
    assert "lon_l2" not in output.dims


def test_crop_file_crops_coordinate_based_netcdf(tmp_path):
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"
    input_dir.mkdir()
    output_dir.mkdir()
    input_file = input_dir / "data.nc"

    lon = np.array([0.0, 1.0, 2.0, 3.0])
    lat = np.array([3.0, 2.0, 1.0, 0.0])
    values = np.arange(16, dtype=float).reshape(len(lat), len(lon))
    input_ds = xr.Dataset(
        {"data": (("lat", "lon"), values)},
        coords={"lat": lat, "lon": lon},
    )
    input_ds.to_netcdf(input_file)

    crop_file(
        input_file=input_file,
        mask_ds=None,
        latslice=slice(3.0, 1.0),
        lonslice=slice(1.0, 2.0),
        output_path=output_dir,
        input_path=input_dir,
        overwrite=True,
        available_mem_gib=3,
    )

    with xr.open_dataset(output_dir / input_file.name) as output_ds:
        output = output_ds.load()

    assert output["data"].dims == ("lat", "lon")
    assert np.allclose(output["lat"].values, np.array([3.0, 2.0, 1.0]))
    assert np.allclose(output["lon"].values, np.array([1.0, 2.0]))
    assert np.array_equal(output["data"].values, values[:3, 1:3])


def test_crop_file_crops_dimension_only_netcdf_using_header(tmp_path):
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"
    input_dir.mkdir()
    output_dir.mkdir()
    input_file = input_dir / "pre.nc"
    input_header = input_dir / "header.txt"

    input_header.write_text(
        "\n".join(
            [
                "ncols 4",
                "nrows 4",
                "xllcorner 100.0",
                "yllcorner 200.0",
                "cellsize 10.0",
                "nodata_value -9999.0",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    values = np.arange(16, dtype=float).reshape(4, 4)
    input_ds = xr.Dataset({"pre": (("lat", "lon"), values)})
    input_ds.to_netcdf(input_file)

    crop_file(
        input_file=input_file,
        mask_ds=None,
        latslice=slice(230.0, 210.0),
        lonslice=slice(110.0, 130.0),
        output_path=output_dir,
        input_path=input_dir,
        overwrite=True,
        available_mem_gib=3,
    )

    with xr.open_dataset(output_dir / input_file.name) as output_ds:
        output = output_ds.load()

    output_header = read_header(output_dir / "header.txt")
    assert output["pre"].dims == ("lat", "lon")
    assert output["pre"].shape == (2, 2)
    assert np.array_equal(output["pre"].values, values[1:3, 1:3])
    assert output_header["ncols"] == 2
    assert output_header["nrows"] == 2
    assert output_header["xllcorner"] == 110.0
    assert output_header["yllcorner"] == 210.0
    assert output_header["cellsize"] == 10.0
