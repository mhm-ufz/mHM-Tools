"""Calculate PET with multiple different methods."""

import contextlib
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import xarray as xr
from joblib import Parallel, delayed

from mhm_tools.common.file_handler import (
    get_grid,
    get_xarray_ds_from_file,
    set_grid,
    write_xarray_to_file,
)
from mhm_tools.common.logger import ErrorLogger, log_arguments
from mhm_tools.common.xarray_utils import (
    get_coord_key,
    get_single_data_var,
    timedelta_to_alias,
)

logger = logging.getLogger(__name__)


METHODS_REQUIRING_TMAX_TMIN = {
    "hargreaves_samani",
    "hargreaves-samani",
    "HS",
    "baier_robertson",
    "baier-robertson",
}
METHODS_REQUIRING_TAVG = {
    "hargreaves_samani",
    "hargreaves-samani",
    "HS",
    "oudinmcguinness_bordne",
    "mcguinness-bordne",
    "hamon",
    "jensen_haise",
    "jensen-haise",
}


def _daylength_hours(lat_deg: np.ndarray, time: datetime) -> np.ndarray:
    """
    Approximate day length (hours) from latitude + day-of-year.

    Good enough for PET empirical methods (Hamon, Blaney-Criddle, Thornthwaite).
    """
    lat = np.deg2rad(lat_deg)
    doy = time.timetuple().tm_yday
    # solar declination (radians); common approximation
    delta = 0.409 * np.sin(2.0 * np.pi * (doy - 81) / 365.0)
    # sunset hour angle
    cos_omega = -np.tan(lat) * np.tan(delta)
    cos_omega = np.clip(cos_omega, -1.0, 1.0)
    omega = np.arccos(cos_omega)
    # day length in hours
    return 24.0 / np.pi * omega


def _sat_vapor_pressure_kpa(t_c: np.ndarray) -> np.ndarray:
    """Saturation vapor pressure (kPa) for temperature in °C (FAO-56)."""
    return 0.6108 * np.exp((17.27 * t_c) / (t_c + 237.3))


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


def validate_tmin_tmax(tmin, tmax):
    """Test that tmin smaller or equal tmax at every point and time."""
    comparison = np.where(~np.isnan(tmin + tmax), tmin <= tmax, True)
    # comparison = (tmin <= tmax) | np.isnan(tmin) | np.isnan(tmax)
    if not bool(np.all(np.asarray(comparison))):
        msg = "Tmin not allways less or equal to tmax."
        with ErrorLogger(logger):
            raise ValueError(msg)


def _get_latitude_da(ds: xr.Dataset) -> xr.DataArray:
    """Return the latitude coordinate/auxiliary variable from ``ds``."""

    def _is_lat_candidate(name: str, var: xr.DataArray) -> bool:
        std_name = str(var.attrs.get("standard_name", "")).lower()
        return std_name == "latitude" or name.lower() in {"lat", "latitude"}

    for name in list(ds.coords) + list(ds.data_vars):
        var = ds[name]
        if _is_lat_candidate(name, var):
            return var

    lat_key = get_coord_key(ds, lat=True)
    return ds[lat_key]


def pet_calculator(
    # tavg: np.ndarray,
    lat: np.ndarray,
    time: datetime,
    stat_freq: str,
    method: str = "oudin",
    l_heat: float = 2.26,  # latent heat of vaporization (MJ/kg) or compatible with e_rad units
    w_density: float = 977.0,  # water density (kg/m3); used in your original scaling
    **kwargs,
) -> np.ndarray:
    """
    Calculate PET/ET0 using several formulations (as in your figure).

    Required inputs depend on `method`:
      - 'oudin' (default): uses tavg, lat, time (via e_rad_calculator)
      - 'hargreaves_samani': needs tmin, tmax
      - 'mcguinness_bordne': needs Ta (or uses tavg if not given)
      - 'hamon': needs k (optional, default 1.0); uses DL and es computed internally
      - 'baier_robertson': needs tmin, tmax; uses Re (extraterrestrial radiation) from e_rad_calculator
      - 'blaney_criddle': uses DL computed internally
      - 'thornthwaite': needs I and k (heat index + exponent), and typically monthly tavg; uses DL
      - 'jensen_haise': uses Re and Tavg (tavg)
      - 'priestley_taylor': needs delta, rn, g, gamma
      - 'milly_dunne': needs rn, g
      - 'penman_monteith': needs delta, rn, g, gamma, u2, es, ea (or will compute es from tavg if not provided)
      - 'penman_monteith_co2': same as PM plus co2 (ppm)

    Notes
    -----
      - This function assumes your e_rad_calculator(time, lat) returns R_e consistent with l_heat & w_density scaling.
      - Output unit will match your original Oudin output (typically mm/day), then converted to hourly if needed.
    """
    method = method.lower()

    # Common radiation term from your existing pipeline
    # (You already have this function elsewhere.)
    e_rad = e_rad_calculator(time, lat)  # treated as R_e in the figure

    # Daylength for methods that need it
    DL = kwargs.get("DL", _daylength_hours(lat, time))  # hours

    # Convenience
    tavg = kwargs.get("tavg")
    tmin = kwargs.get("tmin")
    tmax = kwargs.get("tmax")

    if method == "oudin":
        pet = (e_rad / (l_heat * w_density)) * ((tavg + 5.0) / 100.0) * 1000.0
        pet = np.where(tavg < -5.0, 0.0, pet)

    elif method in {"hargreaves_samani", "hargreaves-samani", "HS"}:

        tmin = kwargs["tmin"]
        tmax = kwargs["tmax"]
        pet = (
            0.0023
            * (e_rad / (l_heat * w_density))
            * np.sqrt(np.maximum(tmax - tmin, 0.0))
            * (tavg + 17.8)
            * 1000.0
        )

    elif method in {"mcguinness_bordne", "mcguinness-bordne"}:
        # Figure uses (Ta + 5)/68
        pet = 1000.0 * (e_rad / (l_heat * w_density)) * ((tavg + 5.0) / 68.0)

    elif method == "hamon":
        # PET = k * 0.165 * 216.7 * (DL/12) * es/(tavg+273.3)
        k = float(kwargs.get("k", 1.0))
        es = kwargs.get("es", _sat_vapor_pressure_kpa(tavg))
        pet = k * 0.165 * 216.7 * (DL / 12.0) * (es / (tavg + 273.3))

    # elif method in {"baier_robertson", "baier-robertson"}:
    #     tmin = kwargs["tmin"]
    #     tmax = kwargs["tmax"]
    #     pet = 0.157 * tmax + 0.158 * (tmax - tmin) + 0.109 * e_rad - 5.39

    elif method in {"blaney_criddle", "blaney-criddle"}:
        pet = 0.825 * (0.46 * tavg + 8.13) * ((100.0 * DL) / (365.0 * 12.0))

    # elif method == "thornthwaite":
    #     # PET = 16*(DL/360)*((10*tavg)/I)^k
    #     # Typically monthly; you must supply I and k.
    #     annual_heat_index = kwargs.get("I", None)
    #     if annual_heat_index is None:
    #         # alternative https://upcommons.upc.edu/server/api/core/bitstreams/487bda42-f690-4738-bb0c-1ae96b7c1adc/content
    #         mean_monthly_temperature = kwargs.get("mean_monthly_temperature")
    #         if mean_monthly_temperature is None:
    #             if hasattr(tavg, "resample") and "time" in tavg.dims:
    #                 mean_monthly_temperature = tavg.resample(time="MS").mean("time")
    #             else:
    #                 raise ValueError(
    #                     "Thornthwaite requires I or tavg with a time dimension."
    #                 )
    #         monthly_heat_index = np.power(mean_monthly_temperature / 5.0, 1.514)
    #         if isinstance(monthly_heat_index, xr.DataArray):
    #             annual_heat_index = monthly_heat_index.resample(time="Y").mean("time")
    #         else:
    #             annual_heat_index = np.sum(monthly_heat_index, axis=0) / 12
    #     kexp = kwargs["k"]
    #     base = (10.0 * tavg) / annual_heat_index
    #     base = np.where(base > 0, base, 0.0)
    #     pet = 16.0 * (DL / 360.0) * np.power(base, kexp)

    elif method in {"jensen_haise", "jensen-haise"}:
        pet = 1000.0 * (e_rad / (l_heat * w_density)) * (tavg / 40.0)

    # elif method in {"priestley_taylor", "priestley-taylor"}:
    #     # PET = 1.26*\Delta*(Rn-G) / (\lambda*\rho*(\Delta+\gamma))
    #     delta = kwargs["delta"]
    #     rn = kwargs["rn"]
    #     g = kwargs.get("g", 0.0)
    #     gamma = kwargs["gamma"]
    #     pet = (1.26 * delta * (rn - g)) / (l_heat * w_density * (delta + gamma))

    # elif method in {"milly_dunne", "milly-dunne"}:
    #     rn = kwargs["rn"]
    #     g = kwargs.get("g", 0.0)
    #     pet = 0.8 * (rn - g)

    # elif method in {"penman_monteith", "penman-monteith"}:
    #     delta = kwargs["delta"]
    #     rn = kwargs["rn"]
    #     g = kwargs.get("g", 0.0)
    #     gamma = kwargs["gamma"]
    #     u2 = kwargs["u2"]
    #     es = kwargs.get("es", _sat_vapor_pressure_kpa(tavg))
    #     ea = kwargs["ea"]
    #     num = 0.408 * delta * (rn - g) + gamma * (900.0 / (tavg + 273.0)) * u2 * (
    #         es - ea
    #     )
    #     den = delta + gamma * (1.0 + 0.34 * u2)
    #     pet = num / den

    # elif method in {
    #     "penman_monteith_co2",
    #     "penman-monteith[co2]",
    #     "penman_monteith[co2]",
    # }:
    #     delta = kwargs["delta"]
    #     rn = kwargs["rn"]
    #     g = kwargs.get("g", 0.0)
    #     gamma = kwargs["gamma"]
    #     u2 = kwargs["u2"]
    #     co2 = kwargs["co2"]  # ppm
    #     es = kwargs.get("es", _sat_vapor_pressure_kpa(tavg))
    #     ea = kwargs["ea"]
    #     num = 0.408 * delta * (rn - g) + gamma * (900.0 / (Tavg + 273.0)) * u2 * (
    #         es - ea
    #     )
    #     den = delta + gamma * (1.0 + 0.34 * (u2 + 2e-4 * (co2 - 300.0)))
    #     pet = num / den

    else:
        error_msg = f"Unknown method: {method}"
        with ErrorLogger(logger):
            raise ValueError(error_msg)

    # Match your frequency handling: daily stays daily, otherwise convert to hourly
    return np.maximum(pet, 0.0) if stat_freq == "daily" else np.maximum(pet, 0.0) / 24.0


def get_time_and_freq(ds, stat_freq):
    """Return time as array and time frequencs as string."""
    times = ds.time.data
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
    return times, stat_freq


@log_arguments("INFO")
def calculate_pet(
    out_file: str,
    tavg_file: Optional[str] = None,
    tmax_file: Optional[str] = None,
    tmin_file: Optional[str] = None,
    stat_freq: Optional[str] = None,
    method: Optional[str] = "oudin",
    max_workers: int = 1,
) -> None:
    """Calculate PET in parallel across time dimension and save to NetCDF."""
    error_msg = []
    tavg, tmin, tmax = None, None, None
    datasets_to_close = []
    grid_dataset: Optional[xr.Dataset] = None
    grid_dataarray: Optional[xr.DataArray] = None

    if tavg_file is not None:
        tavg_file = Path(tavg_file)
        ds = get_xarray_ds_from_file(tavg_file)
        datasets_to_close.append(ds)
        if "tavg" in ds:
            tavg = ds.tavg
        elif "tas" in ds:
            tavg = ds.tas
        else:  # (time, lat, lon)
            data_var = get_single_data_var(ds)
            tavg = ds[data_var]
        times, stat_freq = get_time_and_freq(ds, stat_freq)
        grid_dataset = ds
        grid_dataarray = tavg
    elif method in METHODS_REQUIRING_TAVG:
        error_msg += [f"Method {method} requires tavg file."]

    if tmax_file is not None and tmin_file is not None:
        tmin_file = Path(tmin_file)
        ds_tmin = get_xarray_ds_from_file(tmin_file)
        datasets_to_close.append(ds_tmin)
        data_var_min = get_single_data_var(ds_tmin, ["tmin", "tasmin"])
        tmin = ds_tmin[data_var_min]

        tmax_file = Path(tmax_file)
        ds_tmax = get_xarray_ds_from_file(tmax_file)
        datasets_to_close.append(ds_tmax)
        data_var_max = get_single_data_var(ds_tmax, ["tmax", "tasmax"])
        tmax = ds_tmax[data_var_max]

        if tavg_file is None:
            times, stat_freq = get_time_and_freq(ds_tmax, stat_freq)
            grid_dataset = ds_tmax
            grid_dataarray = tmax

        if method in METHODS_REQUIRING_TMAX_TMIN:
            validate_tmin_tmax(tmin=tmin, tmax=tmax)

    elif method in METHODS_REQUIRING_TMAX_TMIN:
        error_msg += [f"Method {method} requires tmax and tmin files."]

    if error_msg:
        with ErrorLogger(logger):
            raise ValueError("\n".join(error_msg))

    if grid_dataset is None or grid_dataarray is None:
        msg = "Could not determine reference grid. Provide at least one forcing file."
        with ErrorLogger(logger):
            raise ValueError(msg)

    grid_var_name = grid_dataarray.name or get_single_data_var(grid_dataset)
    grid_definition = get_grid(grid_dataset, grid_var_name)

    lat_da = _get_latitude_da(grid_dataset)
    template = grid_dataarray.isel(time=0, drop=True)
    lat_broadcast = xr.broadcast(lat_da, template)[0]
    lat3d = np.radians(lat_broadcast.data)[np.newaxis, ...]

    logger.info(f"Data frequency is {stat_freq}")

    # Build arguments for each time slice
    tasks = []
    for idx, t in enumerate(times):
        # convert timestamp to datetime
        current_time = datetime.fromtimestamp(int(t) / 1e9, tz=timezone.utc)
        tarr = None
        tmaxarr = None
        tminarr = None
        if tavg is not None:
            tarr = tavg.isel(time=idx).data[np.newaxis, :, :]
        if tmax is not None:
            tmaxarr = tmax.isel(time=idx).data[np.newaxis, :, :]
        if tmin is not None:
            tminarr = tmin.isel(time=idx).data[np.newaxis, :, :]
        tasks.append(
            {
                "tavg": tarr,
                "tmax": tmaxarr,
                "tmin": tminarr,
                "lat": lat3d,
                "time": current_time,
                "stat_freq": stat_freq,
                "method": method,
            }
        )
    logger.info(f"Calculating pet in parallel on {max_workers} cores")
    # Compute PET in parallel
    results = Parallel(n_jobs=max_workers, backend="loky")(
        delayed(pet_calculator)(**task) for task in tasks
    )
    logger.info(f"results are merged into one array {len(results)}")
    # Stack results into array
    pet_data = np.vstack(results)
    pet_data = pet_data.astype(np.float32)

    # Wrap into DataArray
    data_attrs = {"units": "mm", "missing_value": -9999.0, "_FillValue": -9999.0}
    pet_ds = set_grid(pet_data, grid_definition, "pet", data_attrs)
    logger.info("writing output")
    write_xarray_to_file(pet_ds, Path(out_file))

    for dataset in datasets_to_close:
        with contextlib.suppress(AttributeError):
            dataset.close()
    logger.info("done")
