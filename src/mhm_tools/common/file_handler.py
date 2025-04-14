"""File handling utils."""

import logging
from pathlib import Path

import numpy as np
import xarray as xr

from mhm_tools.common.logger import ErrorLogger
from mhm_tools.common.xarray_utils import get_coord_key, get_single_data_var

logger = logging.getLogger(__name__)

######
# more on this in the cut_classical_mhm_setups branch There are classes for Morph and meteo data
######


def create_header(ds, output_path=None, no_data_value="-9999", write=True):
    """Write a header file from a dataset."""
    lat_key = get_coord_key(ds, lat=True)
    lon_key = get_coord_key(ds, lon=True)
    x = ds[lon_key].values
    y = ds[lat_key].values
    cellsize = abs(x[1] - x[0])
    xllcorner = np.nanmin(x) - 0.5 * cellsize
    yllcorner = np.nanmin(y) - 0.5 * cellsize

    ncols = len(x)
    nrows = len(y)
    if write:
        header_out_path = output_path / "header.txt"
        header_str = f"""
ncols                {ncols}
nrows                {nrows}
xllcorner            {xllcorner:.6f}
yllcorner            {yllcorner:.6f}
cellsize             {cellsize:.6f}
NODATA_value         {no_data_value}
            """
        logger.info(
            f"Writing header file to {header_out_path} with header str: {header_str}"
        )
        with header_out_path.open("w") as hf:
            hf.write(header_str)
        return header_out_path
    return {
        "ncols": ncols,
        "nrows": nrows,
        "xllcorner": xllcorner,
        "yllcorner": yllcorner,
        "cellsize": cellsize,
        "NODATA_value": no_data_value,
    }


def crop_file_by_mask(ds, mask_file):
    """Crop file by mask."""
    with xr.open_dataset(mask_file) as mask:
        lat_key_mask = get_coord_key(mask, lat=True)
        lon_key_mask = get_coord_key(mask, lon=True)
        lat_key = get_coord_key(ds, lat=True)
        lon_key = get_coord_key(ds, lon=True)
        return ds.sel(
            {
                lat_key: slice(mask[lat_key_mask].max(), mask[lat_key_mask].min()),
                lon_key: slice(mask[lon_key_mask].min(), mask[lon_key_mask].max()),
            }
        )


def get_xarray_ds_from_file(file_path, var_name=None, chunking=False):
    """Read file and return xarray dataset."""
    file_path = Path(file_path)
    logger.info(f"Reading {file_path} to xarray with chunking = {chunking}")
    ds_out = None
    if not file_path.is_file():
        msg = f"File path does not point to an existing file: {file_path}"
        with ErrorLogger(logger):
            raise ValueError(msg)
    if file_path.suffix == ".asc":
        ds_out = read_ascii_to_xarray(filepath=file_path, var_name=var_name, chunking=chunking)
    if file_path.suffix == ".nc":
        if chunking:
            ds_out = xr.open_dataset(file_path, chunks={"time": 1, "lat": 250, "lon": 250})
        ds_out = xr.open_dataset(file_path)
    if ds_out is None:
        msg = f"File types other than asci and netcdf are not implemented. The suffix of the file was: {file_path.suffix}"
        with ErrorLogger(logger):
            raise NotImplementedError()
    # if var_name is None:
    #     data_var = induce_data_var_from_file_name(ds_out, file_path)
    #     print(type(ds_out[data_var].data))
    return ds_out

def write_xarray_to_file(ds, file_path, var_name=None, fmt=None):
    """Write xarray Datasets to file with file type depending on the file suffix."""
    file_path = Path(file_path)
    logger.info(f"Writing file to {file_path}.")
    if file_path.suffix == ".asc":
        return write_xarray_to_ascii(ds, file_path, var_name, fmt)
    if file_path.suffix == ".nc":
        return ds.to_netcdf(file_path)
    msg = f"File types other than asci and netcdf are not implemented. The suffix of the file was: {file_path.suffix}"
    with ErrorLogger(logger):
        raise NotImplementedError(msg)


def write_xarray_to_ascii(dataset, filepath, data_var=None, fmt=None):
    """Take xarray dataset and writes it to an asci file that can by read by mHM."""
    # Extract the data, coordinates, and nodata value from the Dataset
    if data_var is None:
        data_var = get_single_data_var(dataset)
        if data_var is None:
            logger.error(
                f"Data can not be written to {filepath} as the dataset has multiple data_vars which is incompatible with asci."
            )
            return
    data = dataset[data_var]
    lat = dataset["lat"].values
    lon = dataset["lon"].values
    nodata_value = dataset[data_var].attrs.get("nodata_value", -9999)

    # Calculate header information
    nrows, ncols = data.shape
    cellsize = lon[1] - lon[0]  # Assuming uniform spacing in lon
    xllcorner = lon[0] - 0.5 * cellsize
    yllcorner = lat[-1] - 0.5 * cellsize  # lat starts at the top and descends

    # Create the header
    header = (
        f"ncols         {ncols}\n"
        f"nrows         {nrows}\n"
        f"xllcorner     {xllcorner}\n"
        f"yllcorner     {yllcorner}\n"
        f"cellsize      {cellsize}\n"
        f"NODATA_value  {nodata_value}\n"
    )

    # Replace NaN values with nodata_value in data

    if data.dtype.kind in ["i", "u", "f"]:  # i=int, u=unsigned, f=float
        data_type = "num"
        data_to_write = np.where(np.isnan(data.values), nodata_value, data)
    if data.dtype.kind in ["U", "S"]:
        data_type = "str"
        data_to_write = data

    # Write header and data to ASCII file
    with filepath.open("w") as f:
        f.write(header)
        if fmt is not None:
            np.savetxt(f, data_to_write, fmt=fmt)
        elif data_type == "num":
            np.savetxt(f, data_to_write, fmt="%g")
        else:
            np.savetxt(f, data_to_write, fmt="%s")
        logger.info(f"Writting file to {filepath}")


def read_ascii_to_xarray(filepath, var_name=None, chunking=False):
    """Read an mHM readable asci file to an xarray dataset."""
    # Read the header from the file
    with filepath.open("r") as f:
        header = {}
        for i in range(6):
            line = f.readline().strip()
            logger.debug(f"File {filepath.name} line {i}: {line}")
            key, value = line.split()
            header[key.lower()] = float(value) if "." in value else int(value)

        # Extract header information
        ncols = header["ncols"]
        nrows = header["nrows"]
        xllcorner = header["xllcorner"]
        yllcorner = header["yllcorner"]
        cellsize = header["cellsize"]
        nodata_value = header["nodata_value"]

    # Load the data values
    data_values = np.loadtxt(filepath, skiprows=6)

    # Calculate latitude and longitude coordinates
    lon = np.arange(
        xllcorner + cellsize / 2, xllcorner + (ncols + 0.5) * cellsize, cellsize
    )
    lat = np.arange(
        yllcorner + (nrows - 0.5) * cellsize, yllcorner - cellsize / 2, -cellsize
    )
    logger.debug(lon)
    logger.debug(lat)

    # Create DataArray with lat/lon dimensions and nodata value
    name = "data" if var_name is None else var_name
    data_array = xr.DataArray(
        data=data_values,
        dims=["lat", "lon"],
        coords={"lon": ("lon", lon, {"axis": "X"}), "lat": ("lat", lat, {"axis": "Y"})},
        name=name,
        attrs={"nodata_value": nodata_value, "_FillValue": nodata_value},
    )

    # Convert to Dataset
    if chunking: 
        return xr.Dataset({name: data_array}).chunk({"lat": 1000, "lon": 1000})
    return xr.Dataset({name: data_array})


def get_coord_values(ds, lat=False, lon=False):
    """Get latitude or longitude values from DataSet."""
    key = get_coord_key(ds, lat=lat, lon=lon)
    return ds[key].values
