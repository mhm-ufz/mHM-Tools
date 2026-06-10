import numpy as np
import xarray as xr

from mhm_tools.common.cli_utils import get_coords_from_mask


def test_get_coords_from_mask_with_bounds(tmp_path):
    # build simple lon/lat coords with explicit bounds variables
    lon = np.array([0.0, 1.0, 2.0])
    lat = np.array([10.0, 11.0])
    lon_bnds = np.column_stack([lon - 0.5, lon + 0.5])
    lat_bnds = np.column_stack([lat - 0.5, lat + 0.5])
    mask_vals = np.ones((lat.size, lon.size), dtype=float)

    ds = xr.Dataset(
        data_vars={"mask": (("lat", "lon"), mask_vals)},
        coords={
            "lon": ("lon", lon, {"bounds": "lon_bnds"}),
            "lat": ("lat", lat, {"bounds": "lat_bnds"}),
            "lon_bnds": (("lon", "bnds"), lon_bnds),
            "lat_bnds": (("lat", "bnds"), lat_bnds),
        },
    )

    fn = tmp_path / "mask.nc"
    ds.to_netcdf(fn)

    lon_min, lon_max, lat_min, lat_max, mask_da = get_coords_from_mask(str(fn))

    assert lon_min == float(lon_bnds.min())
    assert lon_max == float(lon_bnds.max())
    assert lat_min == float(lat_bnds.min())
    assert lat_max == float(lat_bnds.max())

    # returned DataArray should equal the dataset variable
    xr.testing.assert_equal(mask_da, ds["mask"])
