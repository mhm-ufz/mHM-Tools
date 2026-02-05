import numpy as np
import xarray as xr

from mhm_tools.pre.regrid import (
    _build_aligned_coords,
    _check_integer_multiple,
    _parse_res,
    regrid_xarray,
)


def test_parse_res_single_and_pair():
    assert _parse_res("0.1") == (0.1, 0.1)
    assert _parse_res("0.1x0.2") == (0.1, 0.2)
    assert _parse_res("0.3,0.4") == (0.3, 0.4)


def test_check_integer_multiple():
    ok, k = _check_integer_multiple(0.2, 0.05)
    assert ok
    assert k == 4
    ok, _ = _check_integer_multiple(0.2, 0.03)
    assert not ok


def test_build_aligned_coords():
    coords = _build_aligned_coords(0.0, 0.2, 0.1)
    assert np.allclose(coords, np.array([0.0, 0.1, 0.2]))


def test_regrid_xarray_preserves_attrs_and_names():
    lon = np.array([0.0, 0.5, 1.0])
    lat = np.array([10.0, 11.0])
    data = np.arange(6).reshape(2, 3)
    ds = xr.Dataset(
        {"v": (("lat", "lon"), data)},
        coords={"lon": lon, "lat": lat},
        attrs={"source": "test"},
    )
    lon_t = np.array([0.0, 1.0])
    lat_t = np.array([10.0, 11.0])

    out = regrid_xarray(ds, "lon", "lat", lon_t, lat_t, method="nearest")
    assert out["v"].shape == (2, 2)
    assert out.attrs["source"] == "test"
    assert np.allclose(out["lon"].values, lon_t)
    assert np.allclose(out["lat"].values, lat_t)
