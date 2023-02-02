"""
Purpose
-------
Calculate the river discharge at bankfull conditions and the bankfull width.

Authors
-------
- Lennart Schüler
- Sebastian Müller
"""

import numpy as np
import xarray as xr


def find_nearest_idx(array, value):
    """Find nearest index.

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


def calc_q_bkfl(q_monthly, return_period):
    """Calculates the discharge at bankfull conditions for a single time series.

    Parameters
    ----------
    q_monthly : arraylike
        discharge time-series
    return_period : float
        The return period of the flood

    Returns
    -------
    numpy.ndarray
        discharge at bankfull conditions
    """
    # exceedance probability
    ex_prob = np.linspace(0, 1, len(q_monthly), endpoint=False)
    # empirical CDF
    Q_sort = np.sort(q_monthly)[::-1]
    # plotting Q_sort against ex_prob gives the exceedance probability
    # pt.plot(Q_sort, ex_prob)
    # X-year flood is defined as a flood which has a
    # 1/x% chance to occur during a year
    idx_bkfl = find_nearest_idx(ex_prob, 1 / return_period)
    return Q_sort[idx_bkfl]


def process_grid(q_monthly, return_period):
    """Calculates the discharge at bankfull conditions for a complete grid.

    Parameters
    ----------
    q_monthly : arraylike
        monthly mean discharge (3d ndarray)
    return_period : float
        The return period of the flood

    Returns
    -------
    numpy.ma.MaskedArray
    """
    q_bkfl = np.ma.empty(q_monthly.shape[1:], q_monthly.dtype)
    for i in range(q_monthly.shape[1]):
        for j in range(q_monthly.shape[2]):
            if not np.all(q_monthly[:, i, j]):
                q_bkfl[i, j] = calc_q_bkfl(q_monthly[:, i, j], return_period)
    return q_bkfl


def gen_bankfull_discharge(ncin_path, ncout_path, return_period=1.5, peri_bkfl=False):
    """Calculate bankfull discharge.

    Parameters
    ----------
    ncin_path : pathlike
        The path of the mRM NetCDF file with the discharge data
    ncout_path : pathlike
        The path of the output NetCDF file
    return_period : float, optional
        The return period of the flood, by default 1.5
    peri_bkfl : bool, optional
        Whether to also estimate the wetted perimeter, by default False
    """

    ds = xr.open_dataset(ncin_path, engine="netcdf4", mask_and_scale=False)
    # bankfull discharge
    q_monthly = ds["Qrouted"].resample(time="1M").mean()
    q_bkfl_data = process_grid(q_monthly, return_period=return_period)
    q_bkfl = q_monthly.isel(time=0, drop=True).copy(data=q_bkfl_data)
    q_bkfl.attrs["long_name"] = "Discharge at bankfull conditions"
    # drop time (and all time dependent variables)
    ds = ds.drop_dims("time")
    ds.encoding.pop("unlimited_dims", None)
    # add new variable
    ds["Q_bkfl"] = q_bkfl
    # perimeter
    if peri_bkfl:
        p_bkfl_data = np.copy(q_bkfl_data)
        p_bkfl_data[q_bkfl_data > 0] = 4.8 * np.sqrt(q_bkfl_data[q_bkfl_data > 0])
        p_bkfl = q_bkfl.copy(data=p_bkfl_data)
        p_bkfl.attrs["long_name"] = "Perimeter at bankfull conditions"
        p_bkfl.attrs["units"] = "m"
        ds["P_bkfl"] = p_bkfl
    # no FillValue for coords and bounds
    encoding = {}
    for v in list(ds.data_vars) + list(ds.coords):
        if not (v in ds.coords or v.endswith("_bnds")):
            continue
        ds[v].attrs.pop("_FillValue", None)
        ds[v].attrs.pop("missing_value", None)
        encoding[v] = {"_FillValue": None}
    # save
    ds.to_netcdf(ncout_path, encoding=encoding)
