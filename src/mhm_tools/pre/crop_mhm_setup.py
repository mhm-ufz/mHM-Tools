"""Crop an existing mHM setup or setup file to a target domain.

The module crops NetCDF inputs by mask or coordinate bounds, recreates the
input folder structure, rewrites ESRI headers where needed, copies unsupported
files, and can create a matching latlon file for the cropped setup.

Authors
-------
- Simon Lüdke
"""

import logging
import shutil
from pathlib import Path

import numpy as np
import xarray as xr
from joblib import Parallel, delayed

from mhm_tools.common.esri_grid import read_header, write_header
from mhm_tools.common.file_handler import (
    ChunkType,
    create_header,
    get_xarray_ds_from_file,
    write_xarray_to_file,
)
from mhm_tools.common.logger import ErrorLogger, log_arguments
from mhm_tools.common.resolution_handler import Resolution, get_file_res
from mhm_tools.common.xarray_utils import (
    crop_ds,
    get_coord_key,
    get_dtype,
    get_single_data_var,
    induce_data_var_from_file_name,
    normalize_lat_lon,
    snap_to_target,
)
from mhm_tools.pre.latlon import create_latlon

logger = logging.getLogger(__name__)


class LatlonFiles:
    """Files needed for latlon creation."""

    latlon_output_file = None
    dem_output_file = None
    meteo_header_path = None

    def set_latlon_output_file(self, path):
        """Set the output file path for the new latlon file."""
        logger.info(f"Setting latlon_output_file to {path}")
        self.latlon_output_file = path

    def set_dem_output_file(self, path):
        """Set the path where the dem containing L0 information can be found."""
        logger.info(f"Setting dem_output_file to {path}")
        self.dem_output_file = path

    def set_meteo_header_path(self, path):
        """Set the path where the meteo header containing L11 information can be found."""
        logger.info(f"Setting meteo_header_path to {path}")
        self.meteo_header_path = path

    def get_latlon_output_file(self):
        """Get the latlon output file."""
        return self.latlon_output_file

    def get_dem_output_file(self):
        """Get the dem file path."""
        return self.dem_output_file

    def get_meteo_header_path(self):
        """Get the meteo header file path."""
        return self.meteo_header_path

    def are_set(self):
        """Check if all files needed for latlon creation are set."""
        all_set = True
        if self.latlon_output_file is None:
            logger.info("latlon_output_file not set")
            all_set = False
        if self.dem_output_file is None:
            logger.info("dem_output_file not set")
            all_set = False
        return all_set

    def set_by_list_of_objects(self, obj_list):
        """Set all files by a list of LatlonFiles objects that may be empty or contain files."""
        for obj in obj_list:
            latlon_output_file = obj.get_latlon_output_file()
            dem_output_file = obj.get_dem_output_file()
            meteo_header_path = obj.get_meteo_header_path()
            if latlon_output_file is not None:
                self.set_latlon_output_file(latlon_output_file)
            if dem_output_file is not None:
                self.set_dem_output_file(dem_output_file)
            if meteo_header_path is not None:
                self.set_meteo_header_path(meteo_header_path)


def regrid_mask(
    mask_ds,
    lon_key_mask,
    lat_key_mask,
    target_lon,
    target_lat,
    mask_key=None,
    lon_key_target=None,
    lat_key_target=None,
    target_res=None,
    mask_res=None,
):
    """Regrid a xarray mask dataset mask_ds to the resolution of a second dataset ds2."""

    def _select_mask_var(mask_obj):
        if isinstance(mask_obj, xr.DataArray):
            return mask_obj
        if isinstance(mask_obj, xr.Dataset):
            key = mask_key or get_single_data_var(mask_obj)
            if key is None:
                no_key_msg = "Mask dataset has multiple data_vars; provide mask_key."
                with ErrorLogger(logger):
                    raise ValueError(no_key_msg)
            return mask_obj[key]
        wrong_type_msg = f"Unsupported mask type: {type(mask_obj)}"
        with ErrorLogger(logger):
            raise ValueError(wrong_type_msg)

    if lon_key_target is None:
        lon_key_target = lon_key_mask
    if lat_key_target is None:
        lat_key_target = lat_key_mask
    mask_lon = mask_ds[lon_key_mask].data
    mask_lat = mask_ds[lat_key_mask].data
    mask_res = abs(mask_lon[1] - mask_lon[0])
    target_res = abs(target_lon[1] - target_lon[0])
    if (target_res - mask_res) > 1e-5:
        if target_res % mask_res > 1e-5:
            logger.warning(
                f"Target resolution {target_res} is not an integer muptiple of mask resolution {mask_res}. Factor: {target_res / mask_res}"
            )
        results = np.full((len(target_lat), len(target_lon)), 0.0)
        for i, lat in enumerate(target_lat):
            for j, lon in enumerate(target_lon):
                for n, mlat in enumerate(mask_lat):
                    if mlat < (lat - target_res / 2) or mlat > (lat + target_res / 2):
                        continue
                    for m, mlon in enumerate(mask_lon):
                        if mlon < lon - target_res / 2 or mlon > lon + target_res / 2:
                            continue
                        if mask_key is not None:
                            results[i][j] += mask_ds[mask_key].data[n, m]
                        else:
                            results[i][j] += mask_ds.data[n, m]
        results /= np.nanmax(results)
        mask = results > 1e-3
        results[mask] = 1
        results[~mask] = 0
        return xr.DataArray(
            results,
            dims=[lat_key_target, lon_key_target],
            coords={lat_key_target: target_lat, lon_key_target: target_lon},
        )
    if abs(target_res - mask_res) <= 1e-5:
        logger.debug("Target resolution equals mask resolution (within tolerance).")

        try:
            # quick path: if coords are almost equal, reuse data but snap labels
            if (
                len(mask_lon) == len(target_lon)
                and len(mask_lat) == len(target_lat)
                and np.allclose(mask_lon, target_lon, rtol=0, atol=1e-9)
                and np.allclose(mask_lat, target_lat, rtol=0, atol=1e-9)
            ):
                return snap_to_target(
                    _select_mask_var(mask_ds),
                    lat_key=lat_key_mask,
                    lon_key=lon_key_mask,
                    target_lat_array=target_lat,
                    target_lon_array=target_lon,
                    new_lat_key=lat_key_target,
                    new_lon_key=lon_key_target,
                )

            tol = max(mask_res, target_res) * 1e-3  # generous but safe snapping tol
            reindexed = mask_ds.reindex(
                {
                    lat_key_mask: np.asarray(target_lat),
                    lon_key_mask: np.asarray(target_lon),
                },
                method="nearest",
                tolerance=tol,
            )
            min_lon = min(len(mask_lon), len(target_lon))
            min_lat = min(len(mask_lat), len(target_lat))
            logger.debug(
                f"Reindexed mask to target grid with tolerance {tol}; "
                f"delta lon={float(np.nanmax(np.abs(mask_lon[:min_lon] - target_lon[:min_lon]))):.3g}, "
                f"delta lat={float(np.nanmax(np.abs(mask_lat[:min_lat] - target_lat[:min_lat]))):.3g}"
            )
            return snap_to_target(
                _select_mask_var(reindexed),
                lat_key=lat_key_mask,
                lon_key=lon_key_mask,
                target_lat_array=target_lat,
                target_lon_array=target_lon,
                new_lat_key=lat_key_target,
                new_lon_key=lon_key_target,
            )
        except Exception:
            logger.debug(
                "Mask reindex to target grid failed; using original mask", exc_info=True
            )
            return _select_mask_var(mask_ds)
    else:
        msg = "mask coarser than file not yet implemented"
        with ErrorLogger(logger):
            raise Exception(msg)


def crop_file_with_header(ds_in, file_path, output_path, lonslice, latslice):
    """Crop the nc file and create a new header file for the new coordinates."""
    pres = 1e-9
    header = file_path.parent / "header.txt"
    # read header
    header_information = read_header(header)
    logger.info(f"Read in header: {header_information}")
    # obtain data variable from dataset
    data_var = get_single_data_var(ds_in)
    if data_var is None:
        data_var = induce_data_var_from_file_name(ds_in, file_path)
        if data_var is None:
            logger.error(
                f"File {file_path} could not be croped because the data_var could not be determined."
            )
            return None, None
        logger.debug(f"Found data_var={data_var}")
    logger.debug(f"Read in dataset shape: {ds_in[data_var].shape}")
    lon_key = get_coord_key(ds_in, lon=True, raise_exception=True)
    lat_key = get_coord_key(ds_in, lat=True, raise_exception=True)
    # x values
    index_x_min = int(
        (lonslice.start - pres - header_information["xllcorner"])
        / header_information["cellsize"]
        + 0.5
    )
    index_x_max = int(
        (lonslice.stop + pres - header_information["xllcorner"])
        / header_information["cellsize"]
    )
    if latslice.start > latslice.stop:
        ymax = (
            header_information["yllcorner"]
            + header_information["cellsize"] * header_information["nrows"]
        )
        index_y_min = int(
            (ymax - latslice.start - pres) / header_information["cellsize"] + 0.5
        )
        index_y_max = int(
            (ymax - latslice.stop + pres) / header_information["cellsize"]
        )
    else:
        index_y_min = int(
            (latslice.start - pres - header_information["yllcorner"])
            / header_information["cellsize"]
            + 0.5
        )
        index_y_max = int(
            (latslice.stop + pres - header_information["yllcorner"])
            / header_information["cellsize"]
        )
    logger.debug(f"x: {index_x_min}, {index_x_max}")
    logger.debug(f"y: {index_y_min}, {index_y_max}")
    # write header file
    header_out_dir = output_path if output_path.is_dir() else output_path.parent
    header_out_dir.mkdir(parents=True, exist_ok=True)
    header_out_path = header_out_dir / header.name
    xll = header_information["xllcorner"] + header_information["cellsize"] * index_x_min
    yll = header_information["yllcorner"] + header_information["cellsize"] * (
        header_information["nrows"] - index_y_max
    )
    new_header_information = {
        "ncols": index_x_max - index_x_min,
        "nrows": index_y_max - index_y_min,
        "xllcorner": xll,
        "yllcorner": yll,
        "cellsize": header_information["cellsize"],
        "nodata_value": header_information["nodata_value"],
    }
    logger.info(
        f"Writing header file to {header_out_path} with header: {new_header_information}"
    )
    dtype = get_dtype(ds_in)
    write_header(header_out_path, new_header_information, dtype)
    try:
        logger.info("Cropping dataarray...")
        data = ds_in[data_var].isel(
            {
                lat_key: slice(index_y_min, index_y_max),
                lon_key: slice(index_x_min, index_x_max),
            }
        )
        logger.debug(f"Cropped dataarray: {data}")
        logger.info("Creating new dataset...")
        if "time" in ds_in.dims:
            data_array = xr.DataArray(
                data=data,
                dims=["time", lat_key, lon_key],
                coords={
                    "time": ds_in.time,
                    lat_key: data[lat_key],
                    lon_key: data[lon_key],
                },
                name=data_var,
                attrs=data.attrs,
            )
        else:
            data_array = xr.DataArray(
                data=data,
                dims=[lat_key, lon_key],
                coords={
                    lat_key: data[lat_key],
                    lon_key: data[lon_key],
                },
                name=data_var,
                attrs=data.attrs,
            )
        data_array.attrs.update(
            {
                "_FillValue": header_information["nodata_value"],
                "missing_value": header_information["nodata_value"],
            }
        )
        # Convert to Dataset
        ds_out = data_array.to_dataset()
        ds_out.attrs.update(ds_in.attrs)
        logger.debug(f"cropped ds {ds_out}")
        logger.info(f"Shape of cropped ds: {ds_out[data_var].shape}")
        return ds_out, header_out_path
    except IndexError as e:
        with ErrorLogger(logger):
            raise e


@log_arguments()
def call_create_latlon(
    dem_output_file,
    l1_resolution,
    l11_resolution,
    latlon_output_file,
    meteo_header_path,
    crs,
    chunking=True,
    lat_order="decreasing",
):
    """Create lat/lon headers for multiple resolutions and write the latlon file.

    Builds L0, L1, and optionally L11 headers from the DEM and requested
    resolutions, then calls `create_latlon` with the given CRS and meteo header.
    """
    # create new latlon file
    logger.info("Creating new latlon file")
    with get_xarray_ds_from_file(
        dem_output_file,
        chunking=chunking,
        normalize_latlon_coords=True,
        force_decending_y=(lat_order == "decreasing"),
        force_ascending_y=(lat_order == "increasing"),
    ) as ds_dem:
        l0 = create_header(ds_dem)
    logger.debug(f"L0: {l0}")
    l1 = l1_resolution
    l11 = l11_resolution
    logger.debug(latlon_output_file)
    logger.debug(l0)
    logger.debug(l1)
    logger.debug(l11)
    if meteo_header_path is not None:
        meteo_header_path = Path(meteo_header_path)
    logger.debug(meteo_header_path)
    logger.debug(crs)
    Path(latlon_output_file).parent.mkdir(parents=True, exist_ok=True)
    create_latlon(
        out_file=latlon_output_file,
        level0=dem_output_file,
        level1=l1_resolution,
        level11=l11,
        level2=meteo_header_path,
        crs=crs,
    )
    logger.info(f"Latlon file written to {latlon_output_file}")


def crop_file(  # noqa: PLR0912 PLR0915 PLR0913
    input_file,
    mask_ds,
    latslice,
    lonslice,
    output_path,
    input_path,
    overwrite,
    available_mem_gib,
    force_header_creation=False,
    chunking=False,
    output_var=None,
    no_cropping=False,
    lat_order="decreasing",
    output_suffix=None,
    mask_all=False,
    resolutions=None,
):
    """Crops one file by lat and lon slice and may mask it with the mask dataarray."""
    if resolutions is None:
        logger.debug("No resolutions provided.")
        resolutions = Resolution()
    logger.info(f"Cropping the file {input_file}")
    if input_path.is_file():
        output_file = output_path / input_file.name
    else:
        output_file = output_path / input_file.relative_to(input_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    if output_suffix is not None:
        output_file = output_file.with_suffix(output_suffix)
    latlon_files = LatlonFiles()
    if output_file.is_file() and not overwrite:
        logger.info("Target file already exists. Cropping is skipped.")
        return latlon_files
    # 1. latlon file: The latlon file is not croped but its relative location is saved and the latlon file is newly created after all files are croped
    if "latlon" in input_file.name.lower():
        logger.info(
            "Latlon cropping depreciated will implement new latlon creation using the mhm-tools latlon functionality."
        )
        latlon_files.set_latlon_output_file(output_file)
        return latlon_files
    # 2. Restart files are complex and are not yet implemented. mHM restart files can be croped, mRM restart files can't (?).
    if "restart" in input_file.name.lower():
        logger.warning(
            f"Restart file {input_file} could not be copied as that is not yet implemented."
        )
        return latlon_files
    has_header = bool(list(input_file.parent.glob("header.txt")))
    if input_file.suffix in [".asc", ".nc"]:
        try:
            ds = get_xarray_ds_from_file(
                input_file,
                chunking=chunking,
                available_mem_gib=available_mem_gib // 3,
                normalize_latlon_coords=True,
                force_decending_y=(lat_order == "decreasing" and not has_header),
                force_ascending_y=(lat_order == "increasing" and not has_header),
                chunk_type=ChunkType.TIME,
            )
        except ValueError as ve:
            logger.error(
                f"File {input_file} could not be read. It probably does not have the right format."
            )
            raise ve
            # logger.error(ve)
            # return latlon_files
    else:
        # header files are not copied but recreated as they change
        # other txt and markdown files are copied as they nomaly contain description or class definitions but do not change with domain cropping
        if "header" not in input_file.name.lower():
            try:
                shutil.copy(input_file, output_file)
                logger.debug(f"Copied file {input_file.name} to {output_file}")
            except Exception as e:
                logger.error(f"Can't copy {input_file} because of {e}")
        return latlon_files
    logger.debug(f"read in dataset: {ds}")
    # Handling of special cases:
    ds_cropped = None
    if not no_cropping:
        # 3. Files that are in the same folder as a header file. Typical examples are meteo datasets such as temperature or precipitation, here the header is used as they might not have lat, lon coords
        if has_header:
            logger.debug("Cropping and writing new header file...")
            ds_cropped, header_path = crop_file_with_header(
                ds,
                input_file,
                output_file,
                lonslice=lonslice,
                latslice=latslice,
            )
            if not (ds_cropped is None and header_path is None):
                lat_key = get_coord_key(ds_cropped, lat=True)
                lon_key = get_coord_key(ds_cropped, lon=True)
                if input_file.stem in ["pre", "pet", "tavg"]:
                    latlon_files.set_meteo_header_path(header_path)
        # 4. All other netcdf files containing mostly morphological data.
        else:
            lat_key = get_coord_key(ds, lat=True)
            lon_key = get_coord_key(ds, lon=True)

            # extract lon-lat bounds
            lon_start, lon_stop = float(lonslice.start), float(lonslice.stop)
            lat_start, lat_stop = float(latslice.start), float(latslice.stop)

            logger.debug(
                f"Selecting {input_file.name} using lon:{lonslice} and lat:{latslice}"
            )
            ds_cropped = crop_ds(
                ds=ds,
                lon_min=lon_start,
                lon_max=lon_stop,
                lat_min=lat_start,
                lat_max=lat_stop,
                lon_name=lon_key,
                lat_name=lat_key,
            )

    # check for emptiness or insufficient grid points
    if ds_cropped.sizes[lat_key] < 1 or ds_cropped.sizes[lon_key] < 1:
        msg = f"Cropping resulted in insufficient grid size \
            (lat={ds_cropped.sizes[lat_key]}, lon={ds_cropped.sizes[lon_key]})."
        with ErrorLogger(logger):
            raise ValueError(msg)

    # only the dem file or and eventual mHM restart file are masked using the provided mask file
    if "dem" in input_file.name.lower():
        latlon_files.set_dem_output_file(output_file)
    if "dem" in input_file.name.lower() or mask_all:  # or "mhm" in f.name.lower()
        if mask_ds is not None:
            file_res = get_file_res(ds[lon_key], ds[lat_key], resolutions)
            logger.info(f"Preparing mask for masking of {input_file.name.lower()}")
            selected_mask_var = None
            selected_lon_key_mask = None
            selected_lat_key_mask = None
            for var in mask_ds.data_vars:
                mask_da = mask_ds[var]
                # if resolution of mask matches target file use this maks var and do not regrid
                lon_key_mask = get_coord_key(mask_da, lon=True)
                lat_key_mask = get_coord_key(mask_da, lat=True)
                mask_res = get_file_res(
                    lon=mask_ds[lon_key_mask],
                    lat=mask_ds[lat_key_mask],
                    resolutions=resolutions,
                )
                if abs(mask_res - file_res) <= 1e-5:
                    logger.debug(
                        f"Mask resolution {mask_res} matches file resolution {file_res} (within tolerance). Using mask without regridding."
                    )
                    selected_mask_var = var
                    break
                if mask_res < file_res:
                    logger.debug(f"Prelim mask res {mask_res} < file res {file_res}")
                    selected_lon_key_mask = lon_key_mask
                    selected_lat_key_mask = lat_key_mask
                    selected_mask_var = var
            mask_da = normalize_lat_lon(
                mask_ds[selected_mask_var],
                lon_key=selected_lon_key_mask,
                lat_key=selected_lat_key_mask,
                new_lat_key=lat_key,
                new_lon_key=lon_key,
                log_warning=True,
            )
            if "dem" in input_file.name.lower():
                latlon_files.set_dem_output_file(output_file)
            logger.info("Masking dem file")
            logger.debug(f"dem ds before masking: {ds_cropped}")
            mask_regridded = regrid_mask(
                mask_ds=mask_da,
                lon_key_mask=lon_key,
                lat_key_mask=lat_key,
                target_lon=ds_cropped[lon_key].data,
                target_lat=ds_cropped[lat_key].data,
                lon_key_target=lon_key,
                lat_key_target=lat_key,
                target_res=file_res,
                mask_res=mask_res,
            )
            # apply mask only to data variables that share both spatial dims
            for var in ds_cropped.data_vars:
                if {lat_key, lon_key}.issubset(ds_cropped[var].dims):
                    ds_cropped[var] = ds_cropped[var].where(mask_regridded == 1, np.nan)
            logger.debug(f"dem ds after masking: {ds_cropped}")
        else:
            logger.info("Can't mask dem file because no mask was provided.")
    if output_var is not None:
        try:
            data_var = get_single_data_var(ds_cropped)
            ds_cropped = ds_cropped.rename({data_var: output_var})
            logger.info(
                f"Renamed data_var from {data_var} to specified output variable name {output_var}"
            )
        except ValueError:
            logger.warning(
                f"Could not rename data_var to specified output variable name {output_var}"
            )
    # if ds_cropped is three dimensional convert output file to netcdf (if it was asci) and write warning
    logger.debug(f"Writing cropped file with dims {ds_cropped.dims} to {output_file}")
    if (
        ("time" in ds_cropped.dims and ds_cropped.sizes["time"] > 1)
        or (
            "month_of_year" in ds_cropped.dims and ds_cropped.sizes["month_of_year"] > 1
        )
    ) and output_file.suffix == ".asc":
        output_file = output_file.with_suffix(".nc")
        logger.warning(
            "Converting output file to netcdf because it is three dimensional."
        )
    try:
        write_xarray_to_file(
            ds_cropped, output_file  # , available_mem_gib=available_mem_gib
        )
    except Exception as e:
        logger.warning(f"First try writing the file failed: {e}")
        logger.info("Changing datatype to float")
        for var_name in ds_cropped.data_vars:
            ds_cropped[var_name] = ds_cropped[var_name].astype(float)
        write_xarray_to_file(
            ds_cropped, output_file  # , available_mem_gib=available_mem_gib
        )

    logger.info(f"Written to {output_file}")
    if force_header_creation:
        file_res = get_file_res(ds[lon_key], ds[lat_key], resolutions)
        logger.info(
            f"Creating header file for {output_file} with resolution {file_res}"
        )
        xllcorner = None
        yllcorner = None
        try:
            mask_header = create_header(mask_ds["mask"], cellsize=resolutions.l0)
            xllcorner = mask_header["xllcorner"]
            yllcorner = mask_header["yllcorner"]
            logger.debug(
                f"Using xllcorner {xllcorner} and yllcorner {yllcorner} from mask header"
            )
        except Exception as e:
            logger.warning(f"Failed to create header for mask file: {e}")
        create_header(
            ds_cropped,
            output_path=output_file.parent,
            cellsize=file_res,
            xllcorner=xllcorner,
            yllcorner=yllcorner,
        )
    return latlon_files


@log_arguments()
def crop_mhm_setup(  # noqa: PLR0913
    mask_ds,
    output_path,
    input_path,
    overwrite=True,
    resolutions=None,
    lonslice=None,
    latslice=None,
    crs=None,
    n_jobs=1,
    filename="*.*",
    available_mem_gib=5,
    force_header_creation=False,
    chunking=False,
    output_var=None,
    no_cropping=False,
    lat_order="decreasing",
    output_suffix=None,
    mask_all=False,
):
    """Cut out an existing mhm domain setup using a mask file."""
    # check if the input is correct
    output_path = Path(output_path)
    input_path = Path(input_path)
    if resolutions is None:
        logger.debug("No resolutions provided.")
        resolutions = Resolution()
    # recusively get all the files from the input path if it is a dir
    logger.info(
        f"Cropping to: longitude ({lonslice.start}, {lonslice.stop}) and latitude ({latslice.stop}, {latslice.start})"
    )
    files = []
    if input_path.is_dir():
        files.extend(input_path.rglob(filename))
    else:
        files = [input_path]

    # cut and copy each file
    list_latlon_files = Parallel(n_jobs=n_jobs, backend="loky")(
        delayed(crop_file)(
            input_file=f,
            mask_ds=mask_ds,
            latslice=latslice,
            lonslice=lonslice,
            output_path=output_path,
            input_path=input_path,
            overwrite=overwrite,
            available_mem_gib=available_mem_gib,
            force_header_creation=force_header_creation,
            chunking=chunking,
            output_var=output_var,
            no_cropping=no_cropping,
            lat_order=lat_order,
            output_suffix=output_suffix,
            mask_all=mask_all,
            resolutions=resolutions,
        )
        for f in files
    )
    latlon_files = LatlonFiles()
    latlon_files.set_by_list_of_objects(list_latlon_files)
    if (
        resolutions.l1 is not None
        and latlon_files.get_latlon_output_file() is None
        and latlon_files.get_dem_output_file() is not None
    ):
        latlon_files.set_latlon_output_file(output_path / "latlon" / "latlon.nc")
    if resolutions.l1 is not None and latlon_files.are_set():
        logger.info("Creating latlon")
        call_create_latlon(
            latlon_files.dem_output_file,
            resolutions.l1,
            resolutions.l11,
            latlon_files.latlon_output_file,
            latlon_files.meteo_header_path,
            crs,
            chunking=chunking,
            lat_order=lat_order,
        )
