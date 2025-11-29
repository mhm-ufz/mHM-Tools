"""File handling utils."""

import contextlib
import logging
from enum import Enum
from pathlib import Path
from textwrap import dedent

import numpy as np
import xarray as xr

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
    """Write a header file from a dataset.

    Takes an xarray Dataset and writes the ASCII header needed for GIS tools.
    """
    lat_key = get_coord_key(ds, lat=True)
    lon_key = get_coord_key(ds, lon=True)
    x = ds[lon_key].values
    y = ds[lat_key].values
    cellsize = abs(x[1] - x[0])
    xllcorner = np.nanmin(x) - 0.5 * cellsize
    yllcorner = np.nanmin(y) - 0.5 * cellsize

    ncols = len(x)
    nrows = len(y)
    header_str = dedent(
        f"""
        ncols                {ncols}
        nrows                {nrows}
        xllcorner            {xllcorner:.6f}
        yllcorner            {yllcorner:.6f}
        cellsize             {cellsize:.6f}
        NODATA_value         {no_data_value}
        """
    )
    if not write:
        return header_str

    if output_path.is_dir():
        header_out_path = output_path / "header.txt"
    elif output_path.is_file():
        header_out_path = output_path
    else:
        msg = "Header output path is neither file nor directory."
        with ErrorLogger(logger):
            raise ValueError(msg)
    logger.info(
        f"Writing header file to {header_out_path} with header str: {header_str}"
    )
    with header_out_path.open("w") as hf:
        hf.write(header_str)
    return header_str


def crop_file_by_mask(ds, mask_file):
    """Crop file by mask."""
    if isinstance(mask_file, xr.Dataset):
        mask = mask_file
    else:
        mask = get_xarray_ds_from_file(mask_file)
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


def chunk_dataset_space_only(
    ds: xr.Dataset, available_mem_gib: float
) -> dict[str, int]:
    """Chunk only in space (lat/lon), leaving time whole, sized to available memory.

    - Uses 80% of available_mem_gib for a single chunk.
    - Computes how many total cells (t * y * x) fit, then allocates all t,
      and splits y/x so that t·y·x·bytes_per_cell ≤ work_bytes.
    - If no time dimension, behaves similarly with t=1.
    """
    logger.info(
        f"Chunking spatial dims to fit ≈{available_mem_gib} GiB (time unchunked)"
    )
    # --- pick one variable to get dtype size ---
    var = get_single_data_var(ds)

    dtype_sz = ds[var].dtype.itemsize  # bytes per element

    # --- find coordinate names ---
    lat_key = get_coord_key(ds, lat=True)
    lon_key = get_coord_key(ds, lon=True)
    time_key = None
    with contextlib.suppress(ValueError):
        time_key = get_coord_key(ds, time=True)

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

    chunks = {lat_key: int(y_chunk), lon_key: int(x_chunk)}
    if time_key:
        # -1 means “take all” for that dim
        chunks[time_key] = -1

    logger.debug(
        f"Chunk sizes → time: {chunks.get(time_key, '—')}, "
        f"{lat_key}: {y_chunk}, {lon_key}: {x_chunk}"
    )
    return chunks


def chunk_dataset_space_and_time(ds, available_mem_gib) -> dict[str, int]:
    """Chunk dataset adjusting chunk size to avaiable memory.

    Simple heuristic:
      - try to keep time chunks small (1…4)vi
      - make y/x chunks as square as possible
    """
    logger.info(
        f"Chunking dataset with a max amount of mem of {available_mem_gib / 1_000_000_000 :.1f}Gb"
    )
    # ---------------- metadata only (cheap) --------------------------------
    var_name = next(iter(ds.data_vars))  # first data variable
    var = ds[var_name]  # an xarray.Variable wrapper
    dtype_sz = var.dtype.itemsize  # bytes per element

    lat_key = get_coord_key(ds, lat=True)
    lon_key = get_coord_key(ds, lon=True)
    time_key = get_coord_key(ds, time=True, raise_exception=False)
    if time_key is None:
        return chunk_dataset_space_only(ds, available_mem_gib)

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

    chunks = {lat_key: int(y_chunk), lon_key: int(x_chunk)}
    if time_key:
        t_chunk = max(1, max_cells // max(1, y_chunk * x_chunk))
        chunks[time_key] = int(t_chunk)
    logger.debug(f"   The chunks used are {chunks}")

    return chunks


class ChunkType(Enum):
    """Define Types of chunking.

    SPACE: Only chunking in space. Time dimension is conserved.
    TIME: Chunking predominately in time. If necessary also in space.
    """

    SPACE = 1
    TIME = 2


def chunk_dataset(ds, chunk_type, available_mem_gib):
    """Chunk xarray.DataSet depending on chunk_type and available memory."""
    if chunk_type == ChunkType.TIME:
        chunks = chunk_dataset_space_and_time(ds, available_mem_gib)
    if chunk_type == ChunkType.SPACE:
        chunks = chunk_dataset_space_only(ds, available_mem_gib)
    return ds.chunk(chunks)


def get_xarray_ds_from_file(
    file_path,
    var_name=None,
    chunking=False,
    available_mem_gib=None,
    chunk_type=ChunkType.SPACE,
    use_mfdataset=False,
    engine="h5netcdf",
    normalize_latlon_coords=False,
    force_decending_y=False,
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
    elif file_path.suffix == ".nc":
        ds_out = read_dataset(
            file_path=file_path,
            use_mfdataset=use_mfdataset,
            engine=engine,
        )
    else:
        msg = f"Reading file types other than asci and netcdf is not implemented. The suffix of the file was: {file_path.suffix}"
        with ErrorLogger(logger):
            raise NotImplementedError(msg)
    lat_key = get_coord_key(ds_out, lat=True, raise_exception=False)
    lon_key = get_coord_key(ds_out, lon=True, raise_exception=False)
    # force correct order of y coordinate
    if lat_key is not None and (
        (force_decending_y and ds_out[lat_key].values[0] < ds_out[lat_key].values[-1])
        or (
            force_ascending_y and ds_out[lat_key].values[0] > ds_out[lat_key].values[-1]
        )
    ):
        ds_out = ds_out.sel({lat_key: slice(None, None, -1)})
    logger.debug(ds_out)
    logger.debug(lat_key)
    logger.debug(lon_key)
    if normalize_latlon_coords:
        # re-name input coords to lat and lon
        ds_out = normalize_lat_lon(ds_out, lat_key, lon_key)

    if lon_key is None and lat_key is None:
        logger.warning("Dataset does not have lon and lat key.")
    elif lon_key is None or lat_key is None:
        logger.error("Dataset has only one of lon at lat keys.")

    if chunking and available_mem_gib is not None:
        ds_out = chunk_dataset(ds_out, chunk_type, available_mem_gib)
    if ds_out is None:
        msg = f"The dataset read from {file_path} is empty."
        with ErrorLogger(logger):
            raise NotImplementedError()
    logger.debug(f"ds_out: {ds_out}")
    return ds_out


def write_xarray_to_file(ds, file_path, var_name=None, fmt=None, create_folder=True):
    """Write xarray Datasets to file with file type depending on the file suffix."""
    file_path = Path(file_path)
    if create_folder and not file_path.parent.is_dir():
        file_path.parent.mkdir(parents=True)
    logger.info(f"Writing file to {file_path}.")
    if file_path.suffix == ".asc":
        return write_xarray_to_ascii(ds, file_path, var_name, fmt)
    if file_path.suffix == ".nc":
        return ds.to_netcdf(file_path)
    msg = f"Writing to file types other than asci and netcdf is not implemented. The suffix of the file was: {file_path.suffix}"
    with ErrorLogger(logger):
        raise NotImplementedError(msg)


def write_xarray_to_ascii(dataset, filepath, data_var=None, fmt=None):
    """Write xarray Dataset to an ASCII file that can be read by mHM."""
    # Extract the data, coordinates, and nodata value from the Dataset
    if data_var is None:
        data_var = get_single_data_var(dataset)
        if data_var is None:
            logger.error(
                f"Data can not be written to {filepath} as the dataset has multiple data_vars which is incompatible with asci or no datavar exists."
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
        data_to_write = np.where(np.isnan(data.values), nodata_value, data)
    else:
        data_to_write = data

    # Write header and data to ASCII file
    with filepath.open("w") as f:
        f.write(header)
        if fmt is not None:
            np.savetxt(f, data_to_write, fmt=fmt)
        elif data.dtype.kind in ["i", "u", "f"]:
            np.savetxt(f, data_to_write, fmt="%g")
        else:
            np.savetxt(f, data_to_write, fmt="%s")
        logger.info(f"Writting file to {filepath}")


def read_ascii_to_xarray(filepath, var_name=None):
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
    da = xr.DataArray(
        data=data_values,
        dims=["lat", "lon"],
        coords={"lon": ("lon", lon, {"axis": "X"}), "lat": ("lat", lat, {"axis": "Y"})},
        name=name,
        attrs={"nodata_value": nodata_value, "_FillValue": nodata_value},
    )

    # Convert to Dataset
    ds = da.to_dataset()

    # Drop spatial_ref if present
    if "spatial_ref" in ds.coords:
        ds = ds.reset_coords("spatial_ref", drop=True)
    return ds


def get_coord_values(ds, lat=False, lon=False):
    """Get latitude or longitude values from DataSet."""
    key = get_coord_key(ds, lat=lat, lon=lon)
    return ds[key].values
