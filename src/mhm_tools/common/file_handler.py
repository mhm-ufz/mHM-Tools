"""File handling utils."""

import contextlib
import logging
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Dict, Optional, Sequence, Set, Tuple, Union

import numpy as np
import xarray as xr

from mhm_tools.common.constants import NC_ENCODE_DEFAULTS, NO_DATA
from mhm_tools.common.esri_grid import standardize_header, write_grid, write_header
from mhm_tools.common.logger import ErrorLogger, log_arguments, log_errors
from mhm_tools.common.netcdf import (
    generate_bounds,
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


@dataclass
class GridDefinition:
    """Container to preserve a dataset's spatial grid metadata."""

    template: xr.Dataset
    dims: Tuple[str, ...]


def get_grid(
    ds: xr.Dataset, data_vars: Optional[Union[str, Sequence[str]]] = None
) -> GridDefinition:
    """Extract a grid definition from ``ds``.

    Parameters
    ----------
    ds : xr.Dataset
        Dataset that defines the spatial/temporal grid.
    data_vars : str | sequence[str], optional
        Data variables to drop from the returned template. When omitted the
        first data variable is used.
    """
    if data_vars is None:
        var_name = get_single_data_var(ds)
        if var_name is None:
            msg = "Cannot determine data_var to describe grid."
            with ErrorLogger(logger):
                raise ValueError(msg)
        drop_vars = [var_name]
    elif isinstance(data_vars, str):
        drop_vars = [data_vars]
        var_name = data_vars
    else:
        drop_vars = list(data_vars)
        var_name = drop_vars[0]

    if var_name not in ds.data_vars:
        msg = f"Grid descriptor variable {var_name} not present in dataset."
        with ErrorLogger(logger):
            raise ValueError(msg)

    template = ds.drop_vars(drop_vars, errors="ignore")
    dims = tuple(ds[var_name].dims)
    return GridDefinition(template=template, dims=dims)


def set_grid(
    data: np.ndarray,
    grid: GridDefinition,
    var_name: str,
    data_attrs: Optional[Dict[str, Union[str, float, int]]] = None,
) -> xr.Dataset:
    """Attach ``data`` to a preserved grid definition."""
    coords = {}
    for name, coord in grid.template.coords.items():
        # Only include coords compatible with the target variable dims.
        if set(coord.dims).issubset(set(grid.dims)):
            coords[name] = coord
    da = xr.DataArray(
        data,
        coords=coords,
        dims=grid.dims,
        attrs=data_attrs or {},
        name=var_name,
    )
    ds = grid.template.copy(deep=False)
    ds[var_name] = da
    return ds


def create_header(ds, output_path=None, no_data_value=None, cellsize=None) -> dict:
    """Write a header file from a dataset.

    Takes an xarray Dataset and writes the ASCII header needed for GIS tools.
    """
    if no_data_value is None:
        no_data_value = NO_DATA
    lat_key = get_coord_key(ds, lat=True)
    lon_key = get_coord_key(ds, lon=True)
    x = ds[lon_key].data
    y = ds[lat_key].data
    if cellsize is None:
        if len(x) > 1:
            cellsize = abs(x[1] - x[0])
        elif len(y) > 1:
            cellsize = abs(y[1] - y[0])
        else:
            msg = "Cannot determine cellsize from dataset with only one x and one y value. Please provide cellsize as an argument."
            with ErrorLogger(logger):
                raise ValueError(msg)
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

    if output_path is not None:
        if output_path.is_dir():
            header_out_path = output_path / "header.txt"
        elif output_path.is_file():
            header_out_path = output_path
        else:
            msg = f"Header output path {output_path} is neither file nor directory."
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
) -> Dict[str, int]:
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


def chunk_dataset_space_and_time(ds, available_mem_gib) -> Dict[str, int]:
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


def get_xarray_ds_from_file(  # noqa: PLR0912
    file_path,
    var_name=None,
    chunking=False,
    available_mem_gib=None,
    chunk_type=ChunkType.SPACE,
    use_mfdataset=False,
    engine="netcdf4",
    normalize_latlon_coords=False,
    force_decending_y=False,
    force_ascending_y=False,
    landcover=False,
    landcover_year_start=None,
):
    """Read file and return xarray dataset."""
    file_path = Path(file_path)
    logger.debug(f"Reading {file_path} to xarray with chunking = {chunking}")
    ds_out = None
    if not file_path.is_file():
        msg = f"File path does not point to an existing file: {file_path}"
        with ErrorLogger(logger):
            raise ValueError(msg)
    suffix = file_path.suffix.lower()
    if suffix == ".asc":
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
    elif suffix == ".nc":
        ds_out = read_dataset(
            file_path=file_path,
            use_mfdataset=use_mfdataset,
            engine=engine,
        )
    elif suffix in {".tif", ".tiff"}:
        import rioxarray as rxr

        da = rxr.open_rasterio(file_path)
        if var_name is not None:
            da = da.rename(var_name)
        else:
            da.name = "data"
        if "band" in da.dims and da.sizes.get("band", 0) == 1:
            da = da.squeeze("band", drop=True)
        nodata = None
        with contextlib.suppress(Exception):
            nodata = da.rio.nodata
        if nodata is not None:
            da.attrs.setdefault("nodata_value", nodata)
            da.attrs.setdefault("_FillValue", nodata)
        if "x" in da.coords:
            da["x"].attrs.setdefault("axis", "X")
        if "y" in da.coords:
            da["y"].attrs.setdefault("axis", "Y")
        ds_out = da.to_dataset()
        if "spatial_ref" in ds_out.coords:
            ds_out = ds_out.drop_vars("spatial_ref")
    else:
        msg = (
            "Reading file types other than asci, netcdf, and geotiff is not "
            f"implemented. The suffix of the file was: {file_path.suffix}"
        )
        with ErrorLogger(logger):
            raise NotImplementedError(msg)
    lat_key = get_coord_key(ds_out, lat=True, raise_exception=False)
    lon_key = get_coord_key(ds_out, lon=True, raise_exception=False)
    # force correct order of y coordinate
    if lat_key is not None and (
        (force_decending_y and ds_out[lat_key].data[0] < ds_out[lat_key].data[-1])
        or (force_ascending_y and ds_out[lat_key].data[0] > ds_out[lat_key].data[-1])
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


def write_xarray_to_file(  # noqa: PLR0912, PLR0915
    ds,
    file_path,
    var_name=None,
    # fmt=None,
    create_folder=True,
    encoding=None,
    engine="netcdf4",
    # available_mem_gib=None,
    # compute_kwargs=None,
    resolution=None,
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
        write_xarray_to_ascii(ds, file_path, var_name, resolution=resolution)
    elif file_path.suffix == ".nc":
        if isinstance(ds, xr.DataArray):
            var_name = ds.name
            logger.debug(f"var {var_name}")
            ds = ds.to_dataset(name=var_name)
            data_vars = [var_name]
            logger.debug(f"Creating data vars from dataarray name: {data_vars}")
        elif var_name is None:
            logger.info(f"Taking data vars from list of ds.data_vars {ds.data_vars}")
            data_vars = list(ds.data_vars)
        else:
            data_vars = [var_name]
            logger.info(f"Setting data vars as input varname [var_name]: {data_vars}")
        logger.debug(f"data vars: {data_vars}")
        if encoding is None:
            encoding = {
                v: {"zlib": True, "complevel": 4, "shuffle": True, **NC_ENCODE_DEFAULTS}
                for v in data_vars
            }
        else:
            # Ensure encoding only targets data variables (avoid coords/bounds)
            encoding = {k: v for k, v in encoding.items() if k in data_vars}
        # Ensure time and its bounds have consistent encoding to avoid decode issues
        time_key = get_coord_key(ds, time=True, raise_exception=False)
        if time_key is not None:
            time_da = ds[time_key]
            bounds_name = time_da.attrs.get("bounds")
            if bounds_name in ds:
                bnds_da = ds[bounds_name]
                # If bounds are numeric but time is datetime, regenerate bounds
                if np.issubdtype(time_da.dtype, np.datetime64) and not np.issubdtype(
                    bnds_da.dtype, np.datetime64
                ):
                    try:
                        ds = ds.copy(deep=False)
                        ds.coords[bounds_name] = generate_bounds(time_da)
                        bnds_da = ds[bounds_name]
                    except Exception:
                        logger.debug("Could not regenerate time bounds", exc_info=True)
                # If both are numeric but bounds look wildly out of scale, rebuild.
                if np.issubdtype(time_da.dtype, np.number) and np.issubdtype(
                    bnds_da.dtype, np.number
                ):
                    try:
                        tmax = float(np.nanmax(np.abs(time_da.values)))
                        bmax = float(np.nanmax(np.abs(bnds_da.values)))
                        if tmax > 0 and bmax / tmax > 1e6:
                            logger.warning(
                                "Time bounds look out of scale (max=%s vs time max=%s); "
                                "regenerating bounds from time.",
                                bmax,
                                tmax,
                            )
                            ds = ds.copy(deep=False)
                            ds.coords[bounds_name] = generate_bounds(time_da)
                            bnds_da = ds[bounds_name]
                    except Exception:
                        logger.debug(
                            "Could not validate/regenerate numeric time bounds",
                            exc_info=True,
                        )
                if "units" not in time_da.encoding:
                    if "units" in time_da.attrs:
                        time_da.encoding["units"] = time_da.attrs["units"]
                    else:
                        time_da.encoding["units"] = "days since 1970-01-01 00:00:00"
                if "calendar" not in time_da.encoding and "calendar" in time_da.attrs:
                    time_da.encoding["calendar"] = time_da.attrs["calendar"]
                if "units" not in bnds_da.encoding:
                    bnds_da.encoding["units"] = time_da.encoding.get("units")
                if (
                    "calendar" in time_da.encoding
                    and "calendar" not in bnds_da.encoding
                ):
                    bnds_da.encoding["calendar"] = time_da.encoding["calendar"]
                # Avoid units/calendar in bounds attrs; xarray will set these during encoding.
                for key in ("units", "calendar"):
                    if key in bnds_da.attrs:
                        bnds_da.attrs.pop(key, None)
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
                                .astype(get_dtype(ds.variables[name]))
                                .item()
                            )
                        except Exception:
                            venc.pop("_FillValue", None)
                    if "missing_value" in venc:
                        try:
                            venc["missing_value"] = (
                                np.array(venc["missing_value"])
                                .astype(get_dtype(ds.variables[name]))
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

            # Drop encoding for non-data variables and ensure data variables are safe
            merged_encoding = {
                k: v for k, v in merged_encoding.items() if k in data_vars
            }

            # Fill NaNs for integer data vars and cast bools to uint8
            for name in data_vars:
                da = ds_clean[name]
                dtype = get_dtype(da)
                if np.issubdtype(dtype, np.bool_):
                    ds_clean[name] = da.astype("uint8")
                    merged_encoding.pop(name, None)
                    continue
                if np.issubdtype(dtype, np.integer) and np.any(np.isnan(da)):
                    fillv = da.attrs.get("_FillValue", NO_DATA)
                    ds_clean[name] = da.fillna(fillv).astype(dtype)

            encoding = sanitize_nc_encoding(ds_clean, merged_encoding)
            logger.info(f"Using encoding: {encoding}")
            set_netcdf_encoding(ds_clean, encoding)
            # Ensure time/bounds attrs do not contain units/calendar before encode
            time_key_clean = get_coord_key(ds_clean, time=True, raise_exception=False)
            if time_key_clean is not None:
                time_var = ds_clean[time_key_clean]
                bounds_name_clean = time_var.attrs.get("bounds")
                # ensure time encoding has units/calendar
                if "units" not in time_var.encoding:
                    if "units" in time_var.attrs:
                        time_var.encoding["units"] = time_var.attrs["units"]
                    else:
                        time_var.encoding["units"] = "days since 1970-01-01 00:00:00"
                if "calendar" not in time_var.encoding and "calendar" in time_var.attrs:
                    time_var.encoding["calendar"] = time_var.attrs["calendar"]
                # remove units/calendar from time attrs to avoid xarray overwrite error
                for key in ("units", "calendar"):
                    time_var.attrs.pop(key, None)
                if bounds_name_clean in ds_clean:
                    bnds_var = ds_clean[bounds_name_clean]
                    if "units" not in bnds_var.encoding:
                        bnds_var.encoding["units"] = time_var.encoding.get("units")
                    if (
                        "calendar" in time_var.encoding
                        and "calendar" not in bnds_var.encoding
                    ):
                        bnds_var.encoding["calendar"] = time_var.encoding["calendar"]
                    for key in ("units", "calendar"):
                        bnds_var.attrs.pop(key, None)
            ds_clean.to_netcdf(
                file_path, engine=engine, format="NETCDF4", encoding=encoding
            )
        except ValueError:
            logger.error(f"Error while writing to {file_path}")
            logger.error(ds)
            logger.info(f"Trying to write without encoding {encoding}")
            # also scrub time/bounds attrs on fallback
            time_key_raw = get_coord_key(ds, time=True, raise_exception=False)
            if time_key_raw is not None:
                time_var = ds[time_key_raw]
                bounds_name_raw = time_var.attrs.get("bounds")
                for key in ("units", "calendar"):
                    time_var.attrs.pop(key, None)
                if bounds_name_raw in ds:
                    for key in ("units", "calendar"):
                        ds[bounds_name_raw].attrs.pop(key, None)
            ds.to_netcdf(file_path, engine=engine, format="NETCDF4")
    else:
        msg = f"Writing to file types other than asci and netcdf is not implemented. The suffix of the file was: {file_path.suffix}"
        with ErrorLogger(logger):
            raise NotImplementedError(msg)


def write_xarray_to_ascii(
    dataset, filepath, data_var=None, nodata_value=None, resolution=None
):
    """Write xarray Dataset to an ASCII file that can be read by mHM."""
    # check if a data_var can be optained for writing the data
    if data_var is None and isinstance(dataset, xr.Dataset):
        data_var = get_single_data_var(dataset)
        if data_var is None:
            logger.error(
                f"Data can not be written to {filepath} as the dataset has multiple data_vars, which is incompatible with asci or no datavar exists."
            )
            return
    # get the data from the dataset
    data = dataset[data_var] if isinstance(dataset, xr.Dataset) else dataset

    # set the nodata value
    dtype = get_dtype(data)
    if nodata_value is None:
        is_int = issubclass(np.dtype(dtype).type, (np.integer, np.unsignedinteger))
        typ = int if is_int else float
        nodata_value = typ(NO_DATA)

    header = create_header(dataset, no_data_value=nodata_value)
    if resolution is not None:
        header["cellsize"] = resolution

    data_to_write = data
    if isinstance(data_to_write, xr.DataArray):
        data_to_write = data_to_write.data

    if data_to_write.dtype.kind in ["i", "u", "f"]:  # i=int, u=unsigned, f=float
        data_to_write = np.where(np.isnan(data_to_write), nodata_value, data_to_write)

    out_header_str = write_grid(
        file=filepath, header=header, dtype=dtype, data=data_to_write
    )
    logger.info(f"Writting file to {filepath}")
    logger.debug(f"Header written:\n{out_header_str}")


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
    if isinstance(nodata_value, int):
        data_values = data_values.astype(np.int32)

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
    obj: Union[xr.Dataset, xr.DataArray],
    include_coords: bool = True,
    extra_reserved: Optional[Set[str]] = None,
    encoding_in: Optional[dict] = None,
):
    """
    Move reserved serialization keys from .attrs to .encoding and return.

    Return:
      - a (shallow-copied) xarray object with cleaned attrs
      - an `encoding` dict suitable for xarray.to_netcdf(encoding=encoding)

    Parameters
    ----------
    obj : xr.Dataset or xr.DataArray
        The object to clean.
    include_coords : bool, default True
        If True, also process coordinate variables.
    extra_reserved : set[str] or None
        Additional keys to treat as reserved (moved to encoding).
    encoding_in : dict or None
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

    if not isinstance(obj, xr.DataArray):
        msg = f"wrong type {obj}"
        raise ValueError(msg)
    # DataArray
    da = obj.copy(deep=False)
    _process_var(da)

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

    encoding = {**coord_encoding, **data_encoding}
    return da, encoding


def get_dataset_from_path(
    path,
    var_name=None,
    chunking=None,
    available_mem_gib=None,
    chunk_type=ChunkType.SPACE,
    use_mfdataset=False,
    engine="netcdf4",
    normalize_latlon_coords=False,
    force_decending_y=False,
    force_ascending_y=False,
    landcover=False,
    landcover_year_start=None,
    available_mem=None,
    file_name="*.*",
):
    """Load a dataset from a file, directory, or pattern.

    This mirrors ``get_xarray_ds_from_file`` for single-file inputs and
    extends it to directories (multi-file datasets).
    """
    if path is None:
        path_is_none_msg = "Input path is None. Please provide a valid file path, directory path, or glob pattern."
        with ErrorLogger(logger):
            raise ValueError(path_is_none_msg)

    if available_mem_gib is None and available_mem is not None:
        available_mem_gib = available_mem
        if chunking is None:
            chunking = True
    if chunking is None:
        chunking = False

    def _postprocess(ds_out):
        lat_key = get_coord_key(ds_out, lat=True, raise_exception=False)
        lon_key = get_coord_key(ds_out, lon=True, raise_exception=False)
        if lat_key is not None and (
            (force_decending_y and ds_out[lat_key].data[0] < ds_out[lat_key].data[-1])
            or (
                force_ascending_y and ds_out[lat_key].data[0] > ds_out[lat_key].data[-1]
            )
        ):
            ds_out = ds_out.sel({lat_key: slice(None, None, -1)})

        logger.debug(ds_out)
        logger.debug(lat_key)
        logger.debug(lon_key)

        if normalize_latlon_coords:
            ds_out = normalize_lat_lon(ds_out, lat_key, lon_key, raise_exceptions=False)

        if lon_key is None and lat_key is None:
            logger.warning("Dataset does not have lon and lat key.")
        elif lon_key is None or lat_key is None:
            logger.error("Dataset has only one of lon at lat keys.")

        if chunking and available_mem_gib is not None:
            ds_out = chunk_dataset(ds_out, chunk_type, available_mem_gib)
        else:
            for name in list(ds_out.variables):
                enc = ds_out.variables[name].encoding
                enc.pop("chunksizes", None)
                enc.pop("_ChunkSizes", None)
                enc.pop("chunks", None)
                enc["contiguous"] = True

        return ds_out

    def _is_dir_or_list(p):
        if isinstance(p, list):
            return True
        if isinstance(p, str):
            p = Path(p)
        return p.is_dir()

    if _is_dir_or_list(path):
        file_list = []
        if not isinstance(path, list):
            path = Path(path)
            file_list = list(path.rglob(file_name))
        else:
            path = [Path(p) for p in path]
            for p in path:
                if p.is_file():
                    file_list.append(p)
                elif p.is_dir():
                    file_list.extend(list(p.rglob(file_name)))
        if not file_list:
            with ErrorLogger(logger):
                msg = f"No files found in {path}."
                raise ValueError(msg)
        if len(file_list) == 1:
            return get_xarray_ds_from_file(
                file_list[0],
                var_name=var_name,
                chunking=chunking,
                available_mem_gib=available_mem_gib,
                chunk_type=chunk_type,
                use_mfdataset=use_mfdataset,
                engine=engine,
                normalize_latlon_coords=normalize_latlon_coords,
                force_decending_y=force_decending_y,
                force_ascending_y=force_ascending_y,
                landcover=landcover,
                landcover_year_start=landcover_year_start,
            )
        non_nc = [p for p in file_list if Path(p).suffix != ".nc"]
        if non_nc:
            with ErrorLogger(logger):
                msg = "Multi-file loading supports NetCDF files only."
                raise ValueError(msg)
        ds_out = read_dataset(file_list, use_mfdataset=use_mfdataset, engine=engine)
        return _postprocess(ds_out)

    path_in = path
    path = Path(path_in)

    if path.is_file():
        return get_xarray_ds_from_file(
            path,
            var_name=var_name,
            chunking=chunking,
            available_mem_gib=available_mem_gib,
            chunk_type=chunk_type,
            use_mfdataset=use_mfdataset,
            engine=engine,
            normalize_latlon_coords=normalize_latlon_coords,
            force_decending_y=force_decending_y,
            force_ascending_y=force_ascending_y,
            landcover=landcover,
            landcover_year_start=landcover_year_start,
        )

    path_str = str(path_in)
    if any(w in path_str for w in ("*", "?", "[", "]")) and path_str.endswith(".nc"):
        ds_out = read_dataset(path_str, use_mfdataset=use_mfdataset, engine=engine)
        return _postprocess(ds_out)

    with ErrorLogger(logger):
        msg = f"Path {path} does not exist."
        raise ValueError(msg)
