
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
import logging
from pathlib import Path
from mhm_tools.common.file_handler import get_xarray_ds_from_file, write_xarray_to_file
from mhm_tools.common.logger import ErrorLogger, log_arguments
from mhm_tools.common.xarray_utils import get_coord_key, get_single_data_var, timedelta_to_alias
import xarray as xr 
import numpy as np

logger = logging.getLogger(__name__)


def pet_calculator(
    tavg: np.ndarray,
    lat: np.ndarray,
    time: datetime,
    stat_freq: str,
    l_heat: float = 2.26,
    w_density: float = 977.0,
) -> np.ndarray:
    """
    Calculate potential evapotranspiration (PET) based on Hargreaves & Samani equation (1985).
    """
    e_rad = e_rad_calculator(time, lat)
    pet = (e_rad / (l_heat * w_density)) * ((tavg + 5) / 100)
    pet = pet * 1000
    pet[tavg < -5] = 0
    return pet if stat_freq == "daily" else pet / 24


def e_rad_calculator(time: datetime, lat: np.ndarray) -> np.ndarray:
    """
    Calculate extraterrestrial radiation (MJ/m²/day) for a given day and latitude.
    """
    doy = pd.Timestamp(time).day_of_year - 1
    dist = 1 + (0.033 * np.cos(((2 * np.pi * doy) / 365)))
    dec = np.radians(-23.44 * np.cos(np.radians((360 / 365) * (doy + 10))))
    ang = np.arccos(np.clip(-np.tan(lat) * np.tan(dec), -1, 1))
    e_rad = (ang * np.sin(lat) * np.sin(dec)) + (
        np.cos(lat) * np.cos(dec) * np.sin(ang)
    )
    return 37.5 * dist * e_rad


def _compute_pet(args):
    """
    Worker helper to calculate PET for a single time slice.
    args: tuple(tavg_array, lat_array, time_val, stat_freq)
    """
    tavg_slice, lat_array, time_val, stat_freq = args
    return pet_calculator(tavg_slice, lat_array, time_val, stat_freq)


@log_arguments('INFO')
def calculate_pet(
    tavg_file: str,
    out_file: str,
    lon_number: int =None,
    stat_freq: str= None,
    max_workers: int = None,
) -> None:
    """
    Calculate PET in parallel across time dimension and save to NetCDF.
    """
    # Load dataset
    tavg_file = Path(tavg_file)
    if not tavg_file.is_file():
        raise ValueError('Tavg file is not a file.')
    ds = get_xarray_ds_from_file(tavg_file)
    data_var = get_single_data_var(ds)
    tavg = ds.tavg if 'tavg' in ds else ds[data_var] # (time, lat, lon)
    lat = ds.lat if 'lat' in ds.coords else ds[get_coord_key(ds, lat=True)]
    lon = ds.lon if 'lon' in ds.coords else ds[get_coord_key(ds, lat=True)]

    times = ds.time.values

    if stat_freq is None: 
        hours, time_id = timedelta_to_alias(ds)
        if time_id == 'D':
            stat_freq='daily'
        elif time_id == '1H':
            stat_freq='hourly'
        else: 
            msg = f"Frequency of input dataset is different from hourly or daily it is {time_id}."
            with ErrorLogger(logger):
                raise ValueError(msg)

    # Prepare latitude broadcast
    if lon_number is None: 
        lon_number = len(lon)
     # Prepare latitude broadcast
    lat_rad = np.radians(lat.data)
    lat2d = np.repeat(lat_rad[:, np.newaxis], lon_number, axis=1)
    lat3d = lat2d[np.newaxis, :, :]

    # Build arguments for each time slice
    tasks = []
    for idx, t in enumerate(times):
        # convert timestamp to datetime
        current_time = datetime.fromtimestamp(int(t) / 1e9, tz=timezone.utc)
        tarr = tavg.isel(time=idx).values[np.newaxis, :, :]
        tasks.append((tarr, lat3d, current_time, stat_freq))

    # Compute PET in parallel
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        results = list(executor.map(_compute_pet, tasks))

    # Stack results into array
    pet_data = np.vstack(results)
    pet_data = pet_data.astype(np.float32)

    # Wrap into DataArray
    pet_da = xr.DataArray(
        pet_data,
        coords=[ds.time, ds.lat, ds.lon],
        dims=["time", "lat", "lon"],
        attrs={"units": "mm", "missing_value": -9999.0, "_FillValue": -9999.0},
    )
    pet_ds = xr.Dataset({"pet": pet_da})

    # Output
    write_xarray_to_file(pet_ds, Path(out_file))

    ds.close()

