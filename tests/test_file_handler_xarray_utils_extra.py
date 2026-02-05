import numpy as np
import xarray as xr

from mhm_tools.common.file_handler import get_xarray_ds_from_file


def _gen_ds(lat, lon):
    data = np.arange(len(lat) * len(lon)).reshape(len(lat), len(lon))
    ds = xr.Dataset(
        {"v": (("lat", "lon"), data)},
        coords={"lat": np.array(lat), "lon": np.array(lon)},
    )
    return ds


def _write_tmp_nc(tmp_path, lat, lon):
    ds = _gen_ds(lat, lon)
    path = tmp_path / "data.nc"
    ds.to_netcdf(path)
    return path


def test_get_xarray_ds_from_file_forces_descending_y(tmp_path):
    path = _write_tmp_nc(tmp_path, lat=[0.0, 1.0, 2.0], lon=[10.0, 11.0])
    ds = get_xarray_ds_from_file(path, force_decending_y=True)
    assert np.allclose(ds["lat"].values, np.array([2.0, 1.0, 0.0]))


def test_get_xarray_ds_from_file_forces_ascending_y(tmp_path):
    path = _write_tmp_nc(tmp_path, lat=[2.0, 1.0, 0.0], lon=[10.0, 11.0])
    ds = get_xarray_ds_from_file(path, force_ascending_y=True)
    assert np.allclose(ds["lat"].values, np.array([0.0, 1.0, 2.0]))


def test_get_xarray_ds_from_file_removes_chunk_encodings(tmp_path):
    ds = _gen_ds(lat=[0.0, 1.0], lon=[10.0, 11.0])
    ds["v"].encoding["chunksizes"] = (1, 1)
    ds["v"].encoding["_ChunkSizes"] = (1, 1)
    ds["v"].encoding["chunks"] = (1, 1)
    path = tmp_path / "data.nc"
    ds.to_netcdf(path)

    out = get_xarray_ds_from_file(path, chunking=False)
    enc = out["v"].encoding
    assert "chunksizes" not in enc
    assert "_ChunkSizes" not in enc
    assert "chunks" not in enc
    assert enc.get("contiguous") is True
