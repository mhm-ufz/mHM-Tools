import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import xarray as xr
from pyproj import CRS, Transformer

from mhm_tools.pre.pet_calc import (
    calculate_pet,
    e_rad_calculator,
    pet_calculator,
    validate_tmin_tmax,
)


def _write_temp_nc(tmp_dir: Path, name: str, var_name: str, data: np.ndarray) -> Path:
    lat = np.array([50.0, 51.0])
    lon = np.array([10.0, 11.0])
    time = np.array(["2020-01-01", "2020-01-02"], dtype="datetime64[ns]")
    ds = xr.Dataset(
        {
            var_name: (
                ("time", "lat", "lon"),
                data.astype(np.float32),
            )
        },
        coords={"time": time, "lat": lat, "lon": lon},
    )
    path = tmp_dir / name
    ds.to_netcdf(path)
    return path


def _write_projected_temp_nc(
    tmp_dir: Path,
    name: str,
    var_name: str,
    data: np.ndarray,
    with_grid_mapping: bool = True,
) -> Path:
    x = np.array([4321000.0, 4326000.0])
    y = np.array([3210000.0, 3215000.0])
    time = np.array(["2020-01-01", "2020-01-02"], dtype="datetime64[ns]")
    attrs = {"grid_mapping": "lambert_azimuthal_equal_area"}
    data_vars = {
        var_name: (
            ("time", "y", "x"),
            data.astype(np.float32),
            attrs if with_grid_mapping else {},
        )
    }
    if with_grid_mapping:
        data_vars["lambert_azimuthal_equal_area"] = (
            (),
            0,
            {
                "grid_mapping_name": "lambert_azimuthal_equal_area",
                "latitude_of_projection_origin": 52.0,
                "longitude_of_projection_origin": 10.0,
                "false_easting": 4321000.0,
                "false_northing": 3210000.0,
                "semi_major_axis": 6378137.0,
                "inverse_flattening": 298.257223563,
            },
        )

    ds = xr.Dataset(
        data_vars,
        coords={
            "time": time,
            "x": (
                "x",
                x,
                {
                    "axis": "X",
                    "standard_name": "projection_x_coordinate",
                    "units": "m",
                },
            ),
            "y": (
                "y",
                y,
                {
                    "axis": "Y",
                    "standard_name": "projection_y_coordinate",
                    "units": "m",
                },
            ),
        },
    )
    path = tmp_dir / name
    ds.to_netcdf(path)
    return path


def _projected_latitudes() -> np.ndarray:
    x = np.array([4321000.0, 4326000.0])
    y = np.array([3210000.0, 3215000.0])
    x_values, y_values = np.meshgrid(x, y)
    crs = CRS.from_cf(
        {
            "grid_mapping_name": "lambert_azimuthal_equal_area",
            "latitude_of_projection_origin": 52.0,
            "longitude_of_projection_origin": 10.0,
            "false_easting": 4321000.0,
            "false_northing": 3210000.0,
            "semi_major_axis": 6378137.0,
            "inverse_flattening": 298.257223563,
        }
    )
    transformer = Transformer.from_crs(crs, CRS.from_epsg(4326), always_xy=True)
    _, lat = transformer.transform(x_values, y_values)
    return lat


class TestPetCalculatorMethods(unittest.TestCase):
    def setUp(self):
        self.time = datetime(2020, 1, 15, tzinfo=timezone.utc)
        lat_deg = np.array([[50.0, 50.0], [51.0, 51.0]])
        self.lat = np.radians(lat_deg)[np.newaxis, :, :]

    def test_oudin(self):
        tavg = np.array([[[10.0, 12.0], [8.0, -6.0]]])
        e_rad = e_rad_calculator(self.time, self.lat)
        expected = (e_rad / (2.26 * 977.0)) * ((tavg + 5.0) / 100.0) * 1000.0
        expected = np.where(tavg < -5.0, 0.0, expected)
        result = pet_calculator(
            tavg=tavg,
            lat=self.lat,
            time=self.time,
            stat_freq="daily",
            method="oudin",
        )
        np.testing.assert_allclose(result, expected, rtol=1e-6, atol=0.0)

    def test_hargreaves_samani(self):
        tmin = np.array([[[5.0, 6.0], [4.0, 3.0]]])
        tmax = np.array([[[15.0, 16.0], [14.0, 13.0]]])
        tavg = (tmin + tmax) / 2.0
        e_rad = e_rad_calculator(self.time, self.lat)
        expected = (
            0.0023
            * (e_rad / (2.26 * 977.0))
            * np.sqrt(np.maximum(tmax - tmin, 0.0))
            * (tavg + 17.8)
            * 1000.0
        )
        result = pet_calculator(
            tavg=tavg,
            tmin=tmin,
            tmax=tmax,
            lat=self.lat,
            time=self.time,
            stat_freq="daily",
            method="hargreaves-samani",
        )
        np.testing.assert_allclose(result, expected, rtol=1e-6, atol=0.0)


class TestPetCalcValidation(unittest.TestCase):
    def test_tmin_tmax_raises_when_tmin_exceeds_tmax(self):
        tmin = xr.DataArray([[2.0, 6.0]])
        tmax = xr.DataArray([[3.0, 5.0]])
        with self.assertRaises(ValueError):
            validate_tmin_tmax(tmin, tmax)

    def test_calculate_pet_missing_tavg_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            out_file = Path(tmp) / "pet.nc"
            with self.assertRaises(ValueError):
                calculate_pet(
                    out_file=str(out_file),
                    tavg_file=None,
                    method="hamon",
                )

    def test_calculate_pet_missing_tmin_tmax_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_dir = Path(tmp)
            tavg_file = _write_temp_nc(tmp_dir, "tavg.nc", "tavg", np.ones((2, 2, 2)))
            out_file = tmp_dir / "pet.nc"
            with self.assertRaises(ValueError):
                calculate_pet(
                    out_file=str(out_file),
                    tavg_file=str(tavg_file),
                    tmin_file=None,
                    tmax_file=None,
                    method="hargreaves-samani",
                )


class TestPetCalcFromFiles(unittest.TestCase):
    def _expected_from_arrays(self, tavg, tmin, tmax, lat_deg, times, method):
        lat_rad = np.radians(lat_deg)
        lat3d = lat_rad[np.newaxis, :, :]
        results = []
        for idx, tval in enumerate(times):
            current_time = datetime.fromtimestamp(int(tval) / 1e9, tz=timezone.utc)
            tavg_slice = tavg[idx : idx + 1, :, :]
            tmin_slice = None if tmin is None else tmin[idx : idx + 1, :, :]
            tmax_slice = None if tmax is None else tmax[idx : idx + 1, :, :]
            kwargs = {
                "tavg": tavg_slice,
                "lat": lat3d,
                "time": current_time,
                "stat_freq": "daily",
                "method": method,
            }
            if tmin_slice is not None:
                kwargs["tmin"] = tmin_slice
            if tmax_slice is not None:
                kwargs["tmax"] = tmax_slice
            results.append(pet_calculator(**kwargs))
        return np.vstack(results)

    def test_oudin_from_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_dir = Path(tmp)
            tavg_data = np.array(
                [
                    [[10.0, 12.0], [8.0, 9.0]],
                    [[11.0, 13.0], [7.0, 10.0]],
                ]
            )
            tavg_file = _write_temp_nc(tmp_dir, "tavg.nc", "tavg", tavg_data)
            out_file = tmp_dir / "pet.nc"
            calculate_pet(
                out_file=str(out_file),
                tavg_file=str(tavg_file),
                method="oudin",
            )
            ds = xr.open_dataset(out_file)
            lat2d = np.repeat(ds.lat.data[:, np.newaxis], len(ds.lon.data), axis=1)
            expected = self._expected_from_arrays(
                tavg_data,
                None,
                None,
                lat2d,
                ds.time.data,
                "oudin",
            )
            np.testing.assert_allclose(ds.pet.data, expected, rtol=1e-6, atol=0.0)

    def test_hargreaves_samani_from_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_dir = Path(tmp)
            tavg_data = np.array(
                [
                    [[10.0, 12.0], [8.0, 9.0]],
                    [[11.0, 13.0], [7.0, 10.0]],
                ]
            )
            tmin_data = tavg_data - 3.0
            tmax_data = tavg_data + 5.0
            tavg_file = _write_temp_nc(tmp_dir, "tavg.nc", "tavg", tavg_data)
            tmin_file = _write_temp_nc(tmp_dir, "tmin.nc", "tmin", tmin_data)
            tmax_file = _write_temp_nc(tmp_dir, "tmax.nc", "tmax", tmax_data)
            out_file = tmp_dir / "pet.nc"
            calculate_pet(
                out_file=str(out_file),
                tavg_file=str(tavg_file),
                tmin_file=str(tmin_file),
                tmax_file=str(tmax_file),
                method="hargreaves-samani",
            )
            ds = xr.open_dataset(out_file)
            lat2d = np.repeat(ds.lat.data[:, np.newaxis], len(ds.lon.data), axis=1)
            expected = self._expected_from_arrays(
                tavg_data,
                tmin_data,
                tmax_data,
                lat2d,
                ds.time.data,
                "hargreaves-samani",
            )
            np.testing.assert_allclose(ds.pet.data, expected, rtol=1e-6, atol=0.0)

    def test_oudin_from_projected_file_without_latitude(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_dir = Path(tmp)
            tavg_data = np.array(
                [
                    [[10.0, 12.0], [8.0, 9.0]],
                    [[11.0, 13.0], [7.0, 10.0]],
                ]
            )
            tavg_file = _write_projected_temp_nc(
                tmp_dir,
                "tavg_projected.nc",
                "tavg",
                tavg_data,
            )
            out_file = tmp_dir / "pet.nc"
            calculate_pet(
                out_file=str(out_file),
                tavg_file=str(tavg_file),
                method="oudin",
            )
            ds = xr.open_dataset(out_file)
            expected = self._expected_from_arrays(
                tavg_data,
                None,
                None,
                _projected_latitudes(),
                ds.time.data,
                "oudin",
            )
            np.testing.assert_allclose(ds.pet.data, expected, rtol=1e-6, atol=0.0)
            self.assertIn("lambert_azimuthal_equal_area", ds)
            self.assertEqual(
                ds.pet.attrs["grid_mapping"],
                "lambert_azimuthal_equal_area",
            )
            self.assertNotIn("latitude", ds)

    def test_projected_file_without_latitude_requires_grid_mapping(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_dir = Path(tmp)
            tavg_file = _write_projected_temp_nc(
                tmp_dir,
                "tavg_projected.nc",
                "tavg",
                np.ones((2, 2, 2)),
                with_grid_mapping=False,
            )
            out_file = tmp_dir / "pet.nc"
            with self.assertRaisesRegex(ValueError, "latitude.*grid_mapping"):
                calculate_pet(
                    out_file=str(out_file),
                    tavg_file=str(tavg_file),
                    method="oudin",
                )
