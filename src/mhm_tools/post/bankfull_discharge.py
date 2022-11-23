"""
Purpose:
    Calculate the river discharge at bankfull conditions and the bankfull width.

Author:
    Lennart Schueler
"""

import numpy as np
import xarray as xr

from . import netcdf4 as nc
from .netcdf4 import NcDataset


def find_nearest_idx(array, value):
    """
    Find nearest index.

    Parameters
    ----------
    array : numpy.ndarray
        input array
    value : float
        desired value

    Returns
    -------
    int
        nearest index
    """
    return (np.abs(array - value)).argmin()


def find_nearest(array, value):
    """
    Find nearest value.

    Parameters
    ----------
    array : numpy.ndarray
        input array
    value : float
        desired value

    Returns
    -------
    int
        nearest value
    """
    return array[find_nearest_idx(array, value)]


def read_discharge(filename, var_name="Qrouted"):
    """Reads in the discharge from a previous mHM run.

    Assumes that the time variable is named 'time'.
    Converts the time into an array of datetime objects.

    Args:
        filename (str): name of the mHM output file
        var_name (str): name of the variable to be read in
    Returns:
        t (1d ndarray): array of datetime objects
        Q (3d ndarray): the mHM data
    """
    rootgrp = NcDataset(filename, "r")
    t_nc = rootgrp["time"]
    t = t_nc[:]
    t_units = t_nc.units
    try:
        t_cal = t_nc.calendar
    except AttributeError:
        t_cal = "standard"

    t = nc.num2date(t, units=t_units, calendar=t_cal)
    Q = rootgrp[var_name][:]
    return t, Q


def write_Q_bkfl(Q_bkfl, discharge_filename, ncout_filename, peri_bkfl=False):
    """Copies dims and attrs from given file and writes the bankfull discharge

    Args:
        Q_bkfl (2d ndarray): the bankfull discharge
        discharge_filename (str): the filename of the discharge data
        ncout_filename (str): the output filename
        peri_bkfl (bool): whether to calculate the wetted perimeter derived
            from bankful discharge
    """
    ncin = NcDataset(discharge_filename, "r")
    ncout = NcDataset(ncout_filename, "w")

    dims = nc.getDimensions(ncin)
    variables = nc.getVariables(ncin)

    nc.copyDimensions(ncout, dims)
    nc.copyVariables(ncout, variables, skip="Qrouted")

    Q_bkfl_nc = ncout.createVariable(
        "Q_bkfl", "f8", (dims["northing"].name, dims["easting"].name)
    )
    set_nc_attrs(Q_bkfl_nc, "Discharge at bankfull conditions")
    Q_bkfl_nc[:] = Q_bkfl
    if peri_bkfl:
        P_bkfl_nc = ncout.createVariable(
            "P_bkfl", "f8", (dims["northing"].name, dims["easting"].name)
        )
        set_nc_attrs(P_bkfl_nc, "Perimeter at bankfull conditions", units="m")
        P_bkfl_nc[:] = 4.8 * np.sqrt(Q_bkfl)


def set_nc_attrs(nc_var, long_name, units="m3 s-1"):
    """Set NetCDF attributes."""
    nc_var.setncattr("FillValue", -9999.0)
    nc_var.setncattr("long_name", long_name)
    nc_var.setncattr("units", units)
    nc_var.setncattr("scale_factor", 1.0)
    nc_var.setncattr("missing_value", -9999.0)
    nc_var.setncattr("coordinates", "lat lon")


def calc_monthly_means(t, Q):
    """Calculates the monthly mean of Q

    Args:
        t (1d ndarray): the time
        Q (3d ndarray): the discharge on the mRM grid
    """
    ds = xr.Dataset({"Q": (["time", "y", "x"], Q)}, coords={"time": t})
    # ds_mon = ds.resample('M', dim='time')
    ds_mon = ds.resample(time="1M").mean()
    return ds_mon["Q"]


def calc_Q_bkfl(Q, return_period):
    """Calculates the discharge at bankfull conditions for a single time series

    Args:
        t (1d ndarray): the time
        Q (1d ndarray): the discharge
        return_period (float, opt.): the return period of bankfull conditions
    """
    # exceedance probability
    ex_prob = np.linspace(0, 1, len(Q), endpoint=False)
    # empirical CDF
    Q_sort = np.sort(Q)[::-1]
    # plotting Q_sort against ex_prob gives the exceedance probability
    # pt.plot(Q_sort, ex_prob)
    # X-year flood is defined as a flood which has a
    # 1/x% chance to occur during a year
    idx_bkfl = find_nearest_idx(ex_prob, 1 / return_period)
    return Q_sort[idx_bkfl]


def process_grid(Q, return_period):
    """Calculates the discharge at bankfull conditions for a complete grid

    Args:
        t (1d ndarray): the time
        Q (3d ndarray): the discharge on the mRM grid
        return_period (float, opt.): the return period in years
    """
    Q_bkfl = np.ma.empty(Q.shape[1:], Q.dtype)
    for i in range(Q.shape[1]):
        for j in range(Q.shape[2]):
            if not np.all(Q[:, i, j]):
                Q_bkfl[i, j] = calc_Q_bkfl(Q[:, i, j], return_period)
    return Q_bkfl


def bankfull_discharge(ncin_path, ncout_path, return_period=1.5, peri_bkfl=False):
    """
    Calculate bankfull discharge.

    Parameters
    ----------
    ncin_path : _type_
        _description_
    ncout_path : _type_
        _description_
    return_period : float, optional
        _description_, by default 1.5
    peri_bkfl : bool, optional
        _description_, by default False
    """
    t, Q = read_discharge(ncin_path)
    Q_mon = calc_monthly_means(t, Q)
    Q_bkfl = process_grid(Q_mon, return_period)
    write_Q_bkfl(Q_bkfl, ncin_path, ncout_path, peri_bkfl)
