
"""File handling utils."""


from pathlib import Path
import logging
from mhm_tools.common.logger import ErrorLogger
import numpy as np
import xarray as xr

from mhm_tools.common.xarray_utils import get_coord_key


logger = logging.getLogger(__name__)

######
# more on this in the cut_classical_mhm_setups branch There are classes for Morph and meteo data
######

def create_header_file(ds, output_path, no_data_value="-9999"):
    """Write a header file from a dataset."""
    lat_key = get_coord_key(ds, lat=True)
    lon_key = get_coord_key(ds, lon=True)
    x = ds[lon_key].values
    y = np.flip(ds[lat_key].values)
    # x = x[np.where((x >= mask.lon.values[0]) & (x <= mask.lon.values[-1]))]
    # y = y[np.where((y >= mask.lat.values[0]) & (y <= mask.lat.values[-1]))]
    header_path = output_path / "header.txt"
    if not header_path.is_file():
        with header_path.open("w") as hf:
            hf.write(
                f"""
ncols                {len(x)}
nrows                {len(y)}
xllcorner            {np.nanmin(x)}
yllcorner            {np.nanmin(y)}
cellsize             {abs(x[1]-x[0]):.6f}
NODATA_value         {no_data_value}
            """
            )


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

def get_xarray_ds_from_file(file_path):
    """Read file and return xarray dataset."""
    file_path = Path(file_path)
    if not file_path.is_file():
        msg = f"File path does not point to an existing file: {file_path}"
        with ErrorLogger:
            raise ValueError(msg)
    if file_path.suffix == '.asc':
        return read_ascii_to_xarray(filepath=file_path)
    if file_path.suffix == '.nc':
        return xr.open_dataset(file_path)
    msg = f"File types other than asci and netcdf are not implemented. The suffix of the file was: {file_path.suffix}"
    with ErrorLogger:
        raise NotImplementedError()

def write_xarray_to_ascii(dataset, filepath):
    """Take xarray dataset and writes it to an asci file that can by read by mHM."""
    # Extract the data, coordinates, and nodata value from the Dataset
    data = dataset["data"].values
    lat = dataset["lat"].values
    lon = dataset["lon"].values
    nodata_value = dataset["data"].attrs.get("nodata_value", -9999)

    # Calculate header information
    nrows, ncols = data.shape
    cellsize = lon[1] - lon[0]  # Assuming uniform spacing in lon
    xllcorner = lon[0]
    yllcorner = lat[-1]  # lat starts at the top and descends

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
    data_to_write = np.where(np.isnan(data), nodata_value, data)

    # Write header and data to ASCII file
    with filepath.open("w") as f:
        f.write(header)
        np.savetxt(f, data_to_write, fmt="%g")

def read_ascii_to_xarray(filepath):
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
    lon = np.arange(xllcorner, xllcorner + (ncols) * cellsize, cellsize)
    lat = np.arange(
        yllcorner + (nrows - 0.5) * cellsize, yllcorner - cellsize / 2, -cellsize
    )

    # Create DataArray with lat/lon dimensions and nodata value
    data_array = xr.DataArray(
        data=data_values,
        dims=["lat", "lon"],
        coords={"lon": lon, "lat": lat},
        name="data",
        attrs={"nodata_value": nodata_value},
    )

    # Convert to Dataset
    return xr.Dataset({"data": data_array})

