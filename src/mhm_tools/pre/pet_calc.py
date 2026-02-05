"""PET calculation helpers (Hargreaves & Samani)."""

import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import xarray as xr
from joblib import Parallel, delayed

from mhm_tools.common.file_handler import get_xarray_ds_from_file, write_xarray_to_file
from mhm_tools.common.logger import ErrorLogger, log_arguments
from mhm_tools.common.xarray_utils import (
    get_coord_key,
    get_single_data_var,
    timedelta_to_alias,
)

logger = logging.getLogger(__name__)


def pet_calculator(
    tavg: np.ndarray,
    lat: np.ndarray,
    time: datetime,
    stat_freq: str,
    l_heat: float = 2.26,
    w_density: float = 977.0,
) -> np.ndarray:
    """Calculate PET via Hargreaves & Samani (1985)."""
    e_rad = e_rad_calculator(time, lat)
    pet = (e_rad / (l_heat * w_density)) * ((tavg + 5) / 100)
    pet = pet * 1000
    pet[tavg < -5] = 0
    return pet if stat_freq == "daily" else pet / 24


def e_rad_calculator(time: datetime, lat: np.ndarray) -> np.ndarray:
    """Calculate extraterrestrial radiation (MJ/m²/day) for a given day/lat."""
    doy = pd.Timestamp(time).day_of_year - 1
    dist = 1 + (0.033 * np.cos((2 * np.pi * doy) / 365))
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


@log_arguments("INFO")
def calculate_pet(
    tavg_file: str,
    out_file: str,
    lon_number: Optional[int] = None,
    stat_freq: Optional[str] = None,
    max_workers: int = 1,
) -> None:
    """Calculate PET in parallel across time dimension and save to NetCDF."""
    # Load dataset
    tavg_file = Path(tavg_file)
    if not tavg_file.is_file():
        msg = "Tavg file is not a file."
        with ErrorLogger(logger):
            raise ValueError(msg)
    ds = get_xarray_ds_from_file(tavg_file)
    data_var = get_single_data_var(ds)
    tavg = ds.tavg if "tavg" in ds else ds[data_var]  # (time, lat, lon)
    lat = ds.lat if "lat" in ds.coords else ds[get_coord_key(ds, lat=True)]
    lon = ds.lon if "lon" in ds.coords else ds[get_coord_key(ds, lon=True)]

    times = ds.time.data
    logger.info(f"Creating pet from tavg ds with shape {tavg.shape}")
    if stat_freq is None:
        hours, time_id = timedelta_to_alias(ds)
        if time_id == "D":
            stat_freq = "daily"
        elif time_id in ["1h", "1H"]:
            stat_freq = "hourly"
        else:
            msg = f"Frequency of input dataset is different from hourly or daily it is {time_id}."
            with ErrorLogger(logger):
                raise ValueError(msg)
    logger.info(f"Data frequency is {stat_freq}")
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
        tarr = tavg.isel(time=idx).data[np.newaxis, :, :]
        tasks.append((tarr, lat3d, current_time, stat_freq))

    logger.info(f"Calculating pet in parallel on {max_workers} cores")
    # Compute PET in parallel
    results = Parallel(n_jobs=max_workers, backend="loky")(
        delayed(_compute_pet)(task) for task in tasks
    )
    logger.info(f"results are merged into one array {len(results)}")
    # Stack results into array
    pet_data = np.vstack(results)
    pet_data = pet_data.astype(np.float32)

    # Wrap into DataArray
    pet_da = xr.DataArray(
        pet_data,
        coords=[ds.time, lat, lon],
        dims=["time", "lat", "lon"],
        attrs={"units": "mm", "missing_value": -9999.0, "_FillValue": -9999.0},
    )
    pet_ds = xr.Dataset({"pet": pet_da})
    logger.info("writing output")
    # Output
    write_xarray_to_file(pet_ds, Path(out_file))

    ds.close()
    logger.info("done")
