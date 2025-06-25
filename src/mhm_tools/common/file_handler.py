"""File handling utils."""

import logging
from enum import Enum
from pathlib import Path

import numpy as np
import rioxarray
import xarray as xr

from mhm_tools.common.esri_grid import write_header
from mhm_tools.common.logger import ErrorLogger
from mhm_tools.common.netcdf import read_dataset
from mhm_tools.common.xarray_utils import (
    get_coord_key,
    get_single_data_var,
    normalize_lat_lon,
)

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
    header = {
        "ncols": ncols,
        "nrows": nrows,
        "xllcorner": xllcorner,
        "yllcorner": yllcorner,
        "cellsize": cellsize,
        "NODATA_value": no_data_value,
    }
    if write:
        header_out_path = output_path / "header.txt"
        logger.info(f"Writing header file to {header_out_path} with header: {header}")
        write_header(header_out_path, header)
        return header_out_path
    return header


def crop_file_by_mask(ds, mask_file):
    """Crop file by mask."""
    with get_xarray_ds_from_file(mask_file) as mask_ds:
        lat_key_mask = get_coord_key(mask_ds, lat=True)
        lon_key_mask = get_coord_key(mask_ds, lon=True)
        lat_key = get_coord_key(ds, lat=True)
        lon_key = get_coord_key(ds, lon=True)
        return ds.sel(
            {
                lat_key: slice(
                    mask_ds[lat_key_mask].max(), mask_ds[lat_key_mask].min()
                ),
                lon_key: slice(
                    mask_ds[lon_key_mask].min(), mask_ds[lon_key_mask].max()
                ),
            }
        )


def get_chunks_space_only(ds: xr.Dataset, available_mem_gib: float) -> xr.Dataset:
    """
    Chunk only in space (lat/lon), leaving time whole, sized to available memory.

    - Uses 80% of available_mem_gib for a single chunk
    - Computes how many total cells (t × y × x) fit, then allocates all t,
      and splits y/x so that t·y·x·bytes_per_cell ≤ work_bytes.
    - If no time dimension, behaves similarly with t=1.
    """
    logger.info(
        f"Chunking spatial dims to fit ≈{available_mem_gib} GiB (time unchunked)"
    )
    # --- pick one variable to get dtype size ---
    var = next(iter(ds.data_vars.values()))
    dtype_sz = var.dtype.itemsize  # bytes per element

    # --- find coordinate names ---
    lat_key = get_coord_key(ds, lat=True)
    lon_key = get_coord_key(ds, lon=True)
    time_key = get_coord_key(ds, time=True, raise_exception=False)
    ny = ds.sizes[lat_key]
    nx = ds.sizes[lon_key]
    nt = ds.sizes.get(time_key, 1)

    # --- memory budget in bytes (80%) ---
    work_bytes = max(int(0.1 * available_mem_gib * 1024**3), 4 * 1024**2)
    # how many total cells fit
    max_cells = work_bytes // dtype_sz
    # allocate all time steps
    cells_per_slice = max_cells // nt

    # square-ish spatial block
    side = max(1, int(np.sqrt(cells_per_slice)))
    y_chunk = min(ny, side)
    x_chunk = min(nx, side)

    chunks = {lat_key: y_chunk, lon_key: x_chunk}
    if time_key:
        # -1 means “take all” for that dim
        chunks[time_key] = -1

    logger.debug(
        f"Chunk sizes → time: {chunks.get(time_key,'—')}, "
        f"{lat_key}: {y_chunk}, {lon_key}: {x_chunk}"
    )
    return chunks


def get_chunks_space_and_time(ds, available_mem_gib) -> xr.Dataset:
    """
    Chunk dataset adjusting chunk size to avaiable memory.

    Simple heuristic:
      - try to keep time chunks small (1…4)vi
      - make y/x chunks as square as possible
    """
    logger.info(f"Chunking dataset with a max amount of mem of {available_mem_gib}Gb")
    # ---------------- metadata only (cheap) --------------------------------
    var_name = next(iter(ds.data_vars))  # first data variable
    var = ds[var_name]  # an xarray.Variable wrapper
    dtype_sz = var.dtype.itemsize  # bytes per element

    lat_key = get_coord_key(ds, lat=True)
    lon_key = get_coord_key(ds, lon=True)
    time_key = get_coord_key(ds, time=True, raise_exception=False)

    ny = ds.sizes[lat_key]
    nx = ds.sizes[lon_key]
    nt = ds.sizes.get(time_key, 1)

    # ---------------- convert GiB → bytes and keep 80 % ---------------------
    _MIN_BYTES_PER_CHUNK = 4 * 1024**2  # 4MB
    work_bytes = max(int(0.8 * available_mem_gib * 1024**3), _MIN_BYTES_PER_CHUNK)
    max_cells = work_bytes // dtype_sz  # how many array elements fit

    # ---------------- choose chunk sizes -----------------------------------
    t_chunk = min(nt, 4) if time_key else None  # ≤4 along time
    cells_per_t = max_cells // (t_chunk or 1)

    side = max(1, int(np.sqrt(cells_per_t)))  # square-ish y/x chunk
    y_chunk = min(ny, side)
    x_chunk = min(nx, side)

    chunks = {lat_key: y_chunk, lon_key: x_chunk}
    if time_key:
        t_chunk = max_cells // (y_chunk * x_chunk)
        chunks[time_key] = t_chunk
    logger.debug(f"   The chunks used are {chunks}")

    return chunks


class ChunkType(Enum):
    """
    Define Types of chunking.

    SPACE: Only chunking in space. Time dimension is conserved.
    TIME: Chunking predominately in time. If necessary also in space.
    """

    SPACE = 1
    TIME = 2

def get_xarray_ds_from_file(
    file_path,
    var_name=None,
    chunking=False,
    available_mem_gib=None,
    chunk_type=ChunkType.SPACE,
    use_mfdataset=False,
    engine="h5netcdf",
    normalize_latlon_coords=False,
    force_decending_y=True,
    force_ascending_y=False,
):
    """Read file and return xarray dataset."""
    file_path = Path(file_path)
    logger.info(f"Reading {file_path} to xarray with chunking = {chunking}")
    ds_out = None
    if not file_path.is_file():
        msg = f"File path does not point to an existing file: {file_path}"
        with ErrorLogger(logger):
            raise ValueError(msg)
    if file_path.suffix == ".asc":
        ds_out = read_ascii_to_xarray(
            filepath=file_path,
            var_name=var_name,
        )
        chunk_type = ChunkType.SPACE
    if file_path.suffix == ".nc":
        ds_out = read_dataset(
            file_path=file_path,
            use_mfdataset=use_mfdataset,
            engine=engine,
        )

    lat_key = get_coord_key(ds_out, lat=True)
    lon_key = get_coord_key(ds_out, lon=True)
    # force correct order of y coordinate
    if force_decending_y and ds_out[lat_key].values[0] < ds_out[lat_key].values[-1]:
        ds_out = ds_out.sel({lat_key: slice(None, None, -1)})
    if force_ascending_y and ds_out[lat_key].values[0] > ds_out[lat_key].values[-1]:
        ds_out = ds_out.sel({lat_key: slice(None, None, -1)})
    # re-name input coords to lat and lon
    if normalize_latlon_coords:
        ds_out = normalize_lat_lon(ds_out, lat_key, lon_key)

    if chunking and available_mem_gib is not None:
        ds_out = chunk_dataset(ds_out, chunk_type, available_mem_gib)
    if ds_out is None:
        msg = f"File types other than asci and netcdf are not implemented. The suffix of the file was: {file_path.suffix}"
        with ErrorLogger(logger):
            raise NotImplementedError()
    logger.info(f"ds_out: {ds_out}")
    return ds_out


def chunk_dataset(ds, chunk_type, available_mem_gib):
    """Chunk xarray.DataSet depending on chunk_type and available memory."""
    if chunk_type == ChunkType.TIME:
        chunks = get_chunks_space_and_time(ds, available_mem_gib)
    if chunk_type == ChunkType.SPACE:
        chunks = get_chunks_space_only(ds, available_mem_gib)
    return ds.chunk(chunks)


def write_xarray_to_file(ds, file_path, var_name=None, fmt=None, create_folder=True):
    """Write xarray Datasets to file with file type depending on the file suffix."""
    file_path = Path(file_path)
    if create_folder and not file_path.parent.is_dir():
        file_path.parent.mkdir(parents=True)
    logger.info(f"Writing file to {file_path}")
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
    lat_key = get_coord_key(dataset, lat=True)
    lon_key = get_coord_key(dataset, lon=True)
    lat = dataset[lat_key]
    lon = dataset[lon_key]
    nodata_value = dataset[data_var].attrs.get("nodata_value", -9999)

    # Calculate header information
    nrows, ncols = data.shape
    cellsize = lon.data[1] - lon.data[0]  # Assuming uniform spacing in lon
    xllcorner = lon.min() - 0.5 * cellsize
    yllcorner = lat.min() - 0.5 * cellsize  # lat starts at the top and descends

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


def read_ascii_to_xarray(filepath, var_name=None):
    """Read an mHM readable ASCII file to an xarray dataset with axis attributes."""

    # Read the header from the file
    name = "data" if var_name is None else var_name
    da = rioxarray.open_rasterio(filepath, default_name=name)

    # Select the first band
    da = da.sel(band=1, drop=True)

    # Convert to Dataset
    ds = da.to_dataset()

    # Add axis attributes
    if 'x' in ds.coords:
        ds.coords['x'].attrs['axis'] = 'X'
    if 'y' in ds.coords:
        ds.coords['y'].attrs['axis'] = 'Y'

    # Drop spatial_ref if present
    return ds.reset_coords("spatial_ref", drop=True)



def get_coord_values(ds, lat=False, lon=False):
    """Get latitude or longitude values from DataSet."""
    key = get_coord_key(ds, lat=lat, lon=lon)
    return ds[key].values
