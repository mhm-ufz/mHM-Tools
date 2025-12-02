"""File handling utils."""

import logging
from enum import Enum
from pathlib import Path

import numpy as np
import xarray as xr

from mhm_tools.common.constants import NC_ENCODE_DEFAULTS, NO_DATA
from mhm_tools.common.esri_grid import standardize_header, write_grid, write_header
from mhm_tools.common.logger import ErrorLogger, log_arguments, log_errors
from mhm_tools.common.netcdf import (
    read_dataset,
    sanitize_nc_encoding,
    set_netcdf_encoding,
)
from mhm_tools.common.xarray_utils import (
    get_coord_key,
    get_dtype,
    get_single_data_var,
    normalize_lat_lon,
)

logger = logging.getLogger(__name__)

######
# more on this in the cut_classical_mhm_setups branch There are classes for Morph and meteo data
######


def create_header(ds, output_path=None, no_data_value=None, write=True) -> dict:
    """Write a header file from a dataset.

    Takes an xarray Dataset and writes the ASCII header needed for GIS tools.
    """
    if no_data_value is None:
        no_data_value = NO_DATA
    lat_key = get_coord_key(ds, lat=True)
    lon_key = get_coord_key(ds, lon=True)
    x = ds[lon_key].values
    y = ds[lat_key].values
    cellsize = abs(x[1] - x[0])
    xllcorner = np.nanmin(x) - 0.5 * cellsize
    yllcorner = np.nanmin(y) - 0.5 * cellsize

    ncols = len(x)
    nrows = len(y)
    dtype = get_dtype(ds)
    typ = int if issubclass(np.dtype(dtype).type, np.integer) else float
    header_dict = {
        "ncols": ncols,
        "nrows": nrows,
        "xllcorner": xllcorner,
        "yllcorner": yllcorner,
        "cellsize": cellsize,
        "nodata_value": typ(no_data_value),
    }
    header_dict = standardize_header(header_dict)

    if write:
        if output_path.is_dir():
            header_out_path = output_path / "header.txt"
        elif output_path.is_file():
            header_out_path = output_path
        else:
            msg = "Header output path is neither file nor directory."
            with ErrorLogger(logger):
                raise ValueError(msg)
        header_str = write_header(header_out_path, header_dict, dtype)
        logger.info(
            f"Writing header file to {header_out_path} with header str: {header_str}"
        )
    return header_dict


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
    # with contextlib.suppress(ValueError):
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
        f"Chunking dataset with a max amount of mem of {available_mem_gib:.1f}Gb"
    )
    # ---------------- metadata only (cheap) --------------------------------
    if isinstance(ds, xr.Dataset):
        var_name = next(iter(ds.data_vars))  # first data variable
        var = ds[var_name]  # an xarray.Variable wrapper
    else:
        var = ds
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


@log_arguments()
def chunk_dataset(ds, chunk_type, available_mem_gib):
    """Chunk xarray.DataSet depending on chunk_type and available memory."""
    if chunk_type == ChunkType.TIME:
        chunks = chunk_dataset_space_and_time(ds, available_mem_gib)
    if chunk_type == ChunkType.SPACE:
        chunks = chunk_dataset_space_only(ds, available_mem_gib)
    try:
        return ds.chunk(chunks)
    except Exception as e:
        logger.error(chunks)
        logger.error(ds)
        with ErrorLogger(logger):
            raise e


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
    landcover=False,
    landcover_year_start=None,
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
        if landcover:
            logger.info("Reading ascii landcover file.")
            ds_out = read_ascii_to_xarray(
                filepath=file_path,
                var_name=var_name,
                landcover=landcover,
                landcover_year_start=landcover_year_start,
            )
        else:
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
        ds_out = normalize_lat_lon(ds_out, lat_key, lon_key, raise_exceptions=False)

    if lon_key is None and lat_key is None:
        logger.warning("Dataset does not have lon and lat key.")
    elif lon_key is None or lat_key is None:
        logger.error("Dataset has only one of lon at lat keys.")

    if chunking and available_mem_gib is not None:
        ds_out = chunk_dataset(ds_out, chunk_type, available_mem_gib)
    else:
        # if no chunking remove chunking encoding because this might cause errors while writing
        for name in list(ds_out.variables):
            enc = ds_out.variables[name].encoding
            enc.pop("chunksizes", None)
            enc.pop("_ChunkSizes", None)
            enc.pop("chunks", None)
            enc["contiguous"] = True
    if ds_out is None:
        msg = f"The dataset read from {file_path} is empty."
        with ErrorLogger(logger):
            raise NotImplementedError()
    logger.debug(f"ds_out: {ds_out}")
    return ds_out


def write_xarray_to_file(  # noqa: PLR0912
    ds,
    file_path,
    var_name=None,
    # fmt=None,
    create_folder=True,
    encoding=None,
    engine="netcdf4",
    # available_mem_gib=None,
    # compute_kwargs=None,
):
    """Write xarray Datasets to file with file type depending on the file suffix."""
    file_path = Path(file_path)
    if create_folder and not file_path.parent.is_dir():
        file_path.parent.mkdir(parents=True)
    if file_path.is_file():
        file_path.unlink()
    logger.info(f"Writing file to {file_path}")
    # ds = chunk_if_too_big(ds)
    # ds = ds.chunk({'time': 512, 'lat': 121, 'lon': 131})
    if file_path.suffix == ".asc":
        write_xarray_to_ascii(ds, file_path, var_name)
    elif file_path.suffix == ".nc":
        if isinstance(ds, xr.DataArray):
            var_name = ds.name
            logger.debug(f"var {var_name}")
            ds = ds.to_dataset(name=var_name)
            data_vars = [var_name]
        elif var_name is None:
            data_vars = list(ds.data_vars)
        else:
            data_vars = [var_name]
        logger.debug(f"data vars: {data_vars}")
        if encoding is None:
            encoding = {
                v: {"zlib": True, "complevel": 4, "shuffle": True, **NC_ENCODE_DEFAULTS}
                for v in data_vars
            }
        # if False and available_mem_gib is not None:
        #     # ds = chunk_dataset(ds, ChunkType.TIME, available_mem_gib//50)
        #     dask.config.set(
        #         {"array.slicing.split_large_chunks": True}
        #     )  # also works as context manager
        #     # 3) Limit threads for the local threaded scheduler
        #     pool = ThreadPoolExecutor(max_workers=1)
        #     with dask.config.set(scheduler="threads", pool=pool):
        #         # ds.to_netcdf(file#_path, engine=engine, format='NETCDF4',
        #         # encoding=encoding)
        #         delayed = ds.to_netcdf(
        #             file_path, engine="netcdf4", encoding=encoding, compute=False
        #         )
        #         delayed.compute(
        #             scheduler="threads",
        #             num_workers=1,
        #             **(compute_kwargs or {}),
        #         )
        # else:
        #     ds, encoding = move_reserved_attrs_to_encoding(ds, encoding=encoding)

        logger.info(ds)
        # if var_name is not None:
        # encoding = generate_safe_nc_encoding(ds[var_name])
        try:

            # Ensure variable attrs do not contain encoding-only keys that
            # xarray/netCDF4 will reject when encoding is also provided.
            # Do a forceful, best-effort clean of both attrs and any
            # inconsistent encoding entries to avoid the
            # "failed to prevent overwriting existing key _FillValue" error.
            for name in list(ds.variables):
                try:
                    # operate on the underlying Variable.attrs dict to ensure
                    # we modify in-place for both DataArray and Dataset views
                    vattrs = ds.variables[name].attrs
                    if "_FillValue" in vattrs:
                        del vattrs["_FillValue"]
                    if "missing_value" in vattrs:
                        del vattrs["missing_value"]
                except Exception:
                    # best-effort: ignore failures popping attrs
                    logger.debug(
                        f"Could not pop reserved attrs for variable {name}",
                        exc_info=True,
                    )

                # Also ensure any encoding present is consistent with the variable dtype.
                try:
                    venc = ds.variables[name].encoding
                    if "_FillValue" in venc:
                        # try to cast to the variable dtype, otherwise drop
                        try:
                            venc["_FillValue"] = (
                                np.array(venc["_FillValue"])
                                .astype(ds.variables[name].dtype)
                                .item()
                            )
                        except Exception:
                            venc.pop("_FillValue", None)
                    if "missing_value" in venc:
                        try:
                            venc["missing_value"] = (
                                np.array(venc["missing_value"])
                                .astype(ds.variables[name].dtype)
                                .item()
                            )
                        except Exception:
                            venc.pop("missing_value", None)
                except Exception:
                    logger.debug(
                        f"Could not normalize encoding for variable {name}",
                        exc_info=True,
                    )

            # Move any reserved attrs into encoding (this returns a shallow
            # copy of the dataset with those attrs removed and a per-variable
            # encoding dict). This is more robust than trying to pop attrs
            # in-place because xarray may hold multiple views/copies.
            try:
                ds_clean, moved_encoding = move_reserved_attrs_to_encoding(
                    ds, include_coords=True, encoding_in=encoding or {}
                )
                logger.debug(
                    f"Moved reserved attrs into encoding for variables: {list(moved_encoding.keys())}"
                )
            except Exception:
                # Fall back to the original ds if something goes wrong.
                logger.debug(
                    "move_reserved_attrs_to_encoding failed; falling back",
                    exc_info=True,
                )
                ds_clean = ds
                moved_encoding = encoding or {}

            # Merge moved_encoding (from attrs) with provided encoding, giving
            # precedence to values already present in moved_encoding.
            merged_encoding = {}
            for name in set(
                list(moved_encoding.keys()) + list((encoding or {}).keys())
            ):
                merged_encoding[name] = {}
                # start from provided encoding (lowest precedence)
                if encoding and name in encoding:
                    merged_encoding[name].update(encoding[name])
                # then overlay moved encoding (higher precedence)
                if name in moved_encoding:
                    merged_encoding[name].update(moved_encoding[name])

            encoding = sanitize_nc_encoding(ds_clean, merged_encoding)
            logger.info(f"Using encoding: {encoding}")
            set_netcdf_encoding(ds_clean, encoding)
            ds_clean.to_netcdf(
                file_path, engine=engine, format="NETCDF4", encoding=encoding
            )
        except ValueError:
            logger.error(f"Error while writing to {file_path}")
            logger.error(ds)
            logger.info(f"Trying to write without encoding {encoding}")
            ds.to_netcdf(file_path, engine=engine, format="NETCDF4")
    else:
        msg = f"Writing to file types other than asci and netcdf is not implemented. The suffix of the file was: {file_path.suffix}"
        with ErrorLogger(logger):
            raise NotImplementedError(msg)


def write_xarray_to_ascii(dataset, filepath, data_var=None, nodata_value=None):
    """Write xarray Dataset to an ASCII file that can be read by mHM."""
    # Extract the data, coordinates, and nodata value from the Dataset
    dtype = data.dtype
    if nodata_value is None:
        is_int = issubclass(np.dtype(dtype).type, (np.integer, np.unsignedinteger))
        typ = int if is_int else float
        nodata_value = typ(NO_DATA)
    if data_var is None:
        data_var = get_single_data_var(dataset)
        if data_var is None:
            logger.error(
                f"Data can not be written to {filepath} as the dataset has multiple data_vars which is incompatible with asci or no datavar exists."
            )
            return
    data = dataset[data_var]
    header = create_header(dataset, write=False, no_data_value=nodata_value)
    data_to_write = data
    if isinstance(data_to_write, xr.DataArray):
        data_to_write = data_to_write.data

    if data_to_write.dtype.kind in ["i", "u", "f"]:  # i=int, u=unsigned, f=float
        data_to_write = np.where(np.isnan(data_to_write), nodata_value, data_to_write)

    write_grid(filepath, header, dtype=dtype, data=data_to_write)


def read_ascii_to_xarray(
    filepath,
    var_name=None,
    landcover=False,
    landcover_year_start=None,
):
    """Read an mHM readable asci file to an xarray dataset."""
    # Read the header from the file
    with filepath.open("r") as f:
        header = {}

        for i, line in enumerate(f.readlines()):
            line_striped = line.strip()
            if not line_striped:
                continue
            logger.debug(f"File {filepath.name} {i}: {line_striped}")
            key, value = line_striped.split()
            header[key.lower()] = float(value) if "." in value else int(value)
            if len(header) == 6:
                break
        # Extract header information
        ncols = header["ncols"]
        nrows = header["nrows"]
        xllcorner = header["xllcorner"]
        yllcorner = header["yllcorner"]
        cellsize = header["cellsize"]
        nodata_value = header["nodata_value"]

    # Load the data values
    data_values = np.loadtxt(filepath, skiprows=i + 1)

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
    coords = {"lon": ("lon", lon, {"axis": "X"}), "lat": ("lat", lat, {"axis": "Y"})}

    # If this is a landcover file add a 1-element time coordinate
    if landcover and landcover_year_start is not None:
        start_ts = np.datetime64(f"{landcover_year_start}-01-01", "ns")
        time = np.array([start_ts], dtype="datetime64[ns]")
        coords["time"] = ("time", time)

    # If we added a time coordinate, expand the data to have a leading time dim
    if "time" in coords:
        data_arr = np.expand_dims(data_values, axis=0)  # shape (1, nrows, ncols)
        dims = ["time", "lat", "lon"]
    else:
        data_arr = data_values
        dims = ["lat", "lon"]

    da = xr.DataArray(
        data=data_arr,
        dims=dims,
        coords=coords,
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


RESERVED_FOR_ENCODING = {
    "_FillValue",
    "scale_factor",
    "add_offset",
    "dtype",
    "zlib",
    "complevel",
    "shuffle",
    "fletcher32",
    "contiguous",
    "chunksizes",
    "endian",
    "least_significant_digit",
}


def move_reserved_attrs_to_encoding(
    obj: xr.Dataset | xr.DataArray,
    include_coords: bool = True,
    extra_reserved: set[str] | None = None,
    encoding_in: dict | None = None,
):
    """
    Move reserved serialization keys from .attrs to .encoding and return.

    Return:
      - a (shallow-copied) xarray object with cleaned attrs
      - an `encoding` dict suitable for xarray.to_netcdf(encoding=encoding)

    Parameters
    ----------
    obj : xr.Dataset | xr.DataArray
        The object to clean.
    include_coords : bool, default True
        If True, also process coordinate variables.
    extra_reserved : set[str] | None
        Additional keys to treat as reserved (moved to encoding).
    encoding_in : dict | None
        Initial encoding dict to update. If None, starts empty.

    Returns
    -------
    cleaned_obj : same type as `obj`
    encoding : dict
        Mapping var_name -> per-variable encoding dict.
    """
    reserved = set(RESERVED_FOR_ENCODING)
    if extra_reserved:
        reserved |= set(extra_reserved)

    @log_errors(raise_exceptions=False)
    def _process_var(var):
        # Move reserved keys from attrs -> encoding
        for k in list(var.attrs.keys()):
            if k in reserved:
                var.encoding[k] = var.attrs.pop(k)

    if isinstance(obj, xr.Dataset):
        ds = obj.copy(deep=False)
        names = list(ds.variables) if include_coords else list(ds.data_vars)
        for name in names:
            _process_var(ds.variables[name])

        # Build the global encoding dict (only include non-empty encodings)
        encoding = {
            name: {
                k: v for k, v in ds.variables[name].encoding.items() if k in reserved
            }
            for name in names
            if ds.variables[name].encoding
        }
        for name, env in encoding.items():
            for key, val in encoding_in.items():
                if key not in env:
                    encoding[name][key] = val
        return ds, encoding

    if isinstance(obj, xr.DataArray):  # DataArray
        da = obj.copy()
        _process_var(da)
        names = [da.name] if da.name is not None else ["__dataarray__"]
    # DataArray
    da = obj.copy(deep=False)
    _process_var(da)
    names = [da.name] if da.name is not None else ["__dataarray__"]

    coord_encoding = {}
    if include_coords:
        for cname, cvar in da.coords.variables.items():
            _process_var(cvar)
            if cvar.encoding:
                coord_encoding[cname] = {
                    k: v for k, v in cvar.encoding.items() if k in reserved
                }

    data_encoding = {}
    # Only include the data variable if it has a name (required by xarray)
    if da.name is not None and da.encoding:
        data_encoding[da.name] = {k: v for k, v in da.encoding.items() if k in reserved}

        # Merge data + coords encodings
        encoding = {**coord_encoding, **data_encoding}
        return da, encoding
    msg = f"wrong type {obj}"
    raise ValueError(msg)
    # Merge data + coords encodings
    encoding = {**coord_encoding, **data_encoding}
    return da, encoding
