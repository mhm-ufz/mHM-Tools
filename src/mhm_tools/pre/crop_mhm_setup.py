"""Crop an existing mhm-setup by providing a mask file."""

import logging
import shutil
from pathlib import Path

import numpy as np
import xarray as xr
from joblib import Parallel, delayed

from mhm_tools.common.file_handler import (
    create_header,
    get_xarray_ds_from_file,
    write_xarray_to_ascii,
)
from mhm_tools.common.logger import ErrorLogger, log_arguments
from mhm_tools.common.xarray_utils import (
    crop_ds,
    get_coord_key,
    get_single_data_var,
    induce_data_var_from_file_name,
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
        """Check if all three files  are set and if not which one isn't."""
        all_set = True
        if self.latlon_output_file is None:
            logger.info("latlon_output_file not set")
            all_set = False
        if self.dem_output_file is None:
            logger.info("dem_output_file not set")
            all_set = False
        if self.meteo_header_path is None:
            logger.info("meteo_header_path not set")
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
    mask_ds, lon_key_mask, lat_key_mask, target_lon, target_lat, mask_key=None
):
    """Regrid a xarray mask dataset mask_ds to the resolution of a second dataset ds2."""
    mask_lon = mask_ds[lon_key_mask].data
    mask_lat = mask_ds[lat_key_mask].data
    mask_res = mask_lat[0] - mask_lat[1]
    target_res = target_lat[0] - target_lat[1]
    if target_res > mask_res:
        if target_res % mask_res != 0:
            logger.warning(
                f"Target resolution is not an integer muptiple of mask resolution. Factor: {target_res / mask_res}"
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
    elif target_res == mask_res:
        return mask_ds
    else:
        msg = "mask coarser than file not yet implemented"
        with ErrorLogger(logger):
            raise Exception(msg)
    results[mask] = 1
    results[~mask] = 0
    return results


def write_to_file(ds, output_file: Path):
    """Take xarray Dataset and write it to file.

    File type depends on path suffix.
    """
    logger.info(f"Writing to file {output_file}")
    logger.debug(f"Content is: {ds}")
    suffix = output_file.suffix
    if suffix == ".asc":
        write_xarray_to_ascii(ds, output_file)
    elif suffix == ".nc":
        ds.to_netcdf(output_file)


def crop_file_with_header(ds_in, file_path, output_path, lonslice, latslice):
    """Crop the nc file and create a new header file for the new coordinates."""
    pres = 1e-5
    header = file_path.parent / "header.txt"
    data_var = get_single_data_var(ds_in)
    if data_var is None:
        data_var = induce_data_var_from_file_name(ds_in, file_path)
        if data_var is None:
            logger.error(
                f"File {file_path} could not be croped because the data_var could not be determined."
            )
            return None, None
        logger.debug(f"Found data_var={data_var}")
    logger.debug(type(ds_in[data_var].data))
    with header.open("r") as h:
        d = {}
        logger.debug(f"Reading out header.txt file {header}")
        for line in h.readlines():
            if not line.strip():
                continue
            line_content = line.strip().split(" ")
            logger.debug(f"{line_content[0].strip()} = {line_content[-1].strip()}")
            d[line_content[0].strip()] = float(line_content[-1].strip())
        # lon = np.arange(
        #     d["xllcorner"], d["xllcorner"] + d["cellsize"] * d["ncols"], d["cellsize"]
        # )
        # reverse order for lat (TODO: Make this resistant input with south north ordering)
        # lat = np.arange(
        #     d["yllcorner"] + d["cellsize"] * (d["nrows"] - 1),
        #     d["yllcorner"] - d["cellsize"],
        #     -d["cellsize"],
        # )
        logger.info(ds_in[data_var].shape)
        lon_key = get_coord_key(ds_in, lon=True, raise_exception=True)
        lat_key = get_coord_key(ds_in, lat=True, raise_exception=True)
        # x values
        index_x_min = int(
            (lonslice.start - pres - d["xllcorner"]) / d["cellsize"] + 0.5
        )
        index_x_max = int((lonslice.stop + pres - d["xllcorner"]) / d["cellsize"])
        # index_y_min = int((latslice.stop - pres - d["yllcorner"]) / d["cellsize"] + 0.5)
        # index_y_max = int((latslice.start + pres - d["yllcorner"]) / d["cellsize"])
        ymax = d["yllcorner"] + d["cellsize"] * d["nrows"]
        index_y_min = int((ymax - latslice.start - pres) / d["cellsize"] + 0.5)
        index_y_max = int((ymax - latslice.stop + pres) / d["cellsize"])
        logger.debug(f"x: {index_x_min}, {index_x_max}")
        logger.debug(f"y: {index_y_min}, {index_y_max}")
        # write header file
        header_out_path = output_path / header.name
        xll = d["xllcorner"] + d["cellsize"] * index_x_min
        yll = d["yllcorner"] + d["cellsize"] * (d["nrows"] - index_y_max)
        header_str = f"""
ncols                {index_x_max-index_x_min}
nrows                {index_y_max-index_y_min}
xllcorner            {xll}
yllcorner            {yll}
cellsize             {d['cellsize']}
NODATA_value         {d['NODATA_value']}
            """
        logger.info(
            f"Writing header file to {header_out_path} with header str: {header_str}"
        )
        with (header_out_path).open("w") as nh:
            nh.write(header_str)
        try:
            data = ds_in[data_var].isel(
                {
                    lat_key: slice(index_y_min, index_y_max),
                    lon_key: slice(index_x_min, index_x_max),
                }
            )
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
                {"_FillValue": d["NODATA_value"], "missing_value": d["NODATA_value"]}
            )
            # Convert to Dataset
            ds_out = xr.Dataset({data_var: data_array})
            ds_out.attrs.update(ds_in.attrs)
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
):
    """
    Create header dictionaries for the different resolutions and
    call create latlon to create a latlon file for the setup.
    """
    # create new latlon file
    logger.info("Creating new latlon file")
    with get_xarray_ds_from_file(dem_output_file, chunking=True) as ds_dem:
        l0 = create_header(ds_dem, None, write=False)
    logger.debug(f"L0: {l0}")
    l1 = l0.copy()
    l1["cellsize"] = l1_resolution
    l1["ncols"] = int(
        float(l0["cellsize"]) / float(l1["cellsize"]) * int(l0["ncols"]) + 0.5
    )
    l1["nrows"] = int(
        float(l0["cellsize"]) / float(l1["cellsize"]) * int(l0["nrows"]) + 0.5
    )
    l11 = l1.copy()
    if l11_resolution is not None:
        l11["cellsize"] = l11_resolution
        l11["ncols"] = int(
            float(l0["cellsize"]) / float(l11["cellsize"]) * int(l0["ncols"]) + 0.5
        )
        l11["nrows"] = int(
            float(l0["cellsize"]) / float(l11["cellsize"]) * int(l0["nrows"]) + 0.5
        )
    logger.debug(latlon_output_file)
    logger.debug(l0)
    logger.debug(l1)
    logger.debug(l11)
    logger.debug(Path(meteo_header_path))
    logger.debug(crs)
    create_latlon(
        out_file=latlon_output_file,
        level0=l0,
        level1=l1,
        level11=l11,
        level2=Path(meteo_header_path),
        crs=crs,
    )
    logger.info(f"Latlon file written to {latlon_output_file}")


def crop_file(
    input_file,
    mask_da,
    latslice,
    lonslice,
    output_path,
    input_path,
    overwrite,
    available_mem_gib,
):
    """Crops one file by lat and lon slice and may mask it with the mask dataarray."""
    logger.info(f"Cropping the file {input_file}")
    output_file = output_path / input_file.relative_to(input_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    latlon_files = LatlonFiles()
    if output_file.is_file() and not overwrite:
        logger.info("Target file already exists. Cropping is skipped.")
        return latlon_files
    if input_file.suffix in [".asc", ".nc"]:
        try:
            ds = get_xarray_ds_from_file(
                input_file, chunking=True, available_mem_gib=available_mem_gib
            )
        except ValueError:
            logger.error(
                f"File {input_file} could not be read. It probably does not have the right format."
            )
            return latlon_files
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
    # 1. latlon file: The latlon file is not croped but its relative location is saved and the latlon file is newly created after all files are croped
    if "latlon" in input_file.name.lower():
        logger.info(
            "Latlon cropping depreciated will implement new latlon creation using the mhm-tools latlon functionality."
        )
        latlon_files.set_latlon_output_file(output_file)
    # 2. Restart files are complex and are not yet implemented. mHM restart files can be croped, mRM restart files can't (?).
    elif "restart" in input_file.name.lower():
        logger.warning(
            f"Restart file {input_file} could not be copied as that is not yet implemented."
        )
    # 3. Files that are in the same folder as a header file. Typical examples are meteo datasets such as temperature or precipitation
    elif list(input_file.parent.glob("header.txt")):
        logger.debug("Cropping and writing new header file...")
        ds_cropped, header_path = crop_file_with_header(
            ds,
            input_file,
            output_path / input_file.parent.relative_to(input_path),
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
    if ds_cropped.sizes[lat_key] < 2 or ds_cropped.sizes[lon_key] < 2:
        logger.warning(
            "Cropping resulted in insufficient grid size "
            f"(lat={ds_cropped.sizes[lat_key]}, lon={ds_cropped.sizes[lon_key]})."
        )
        return latlon_files

    # only the dem file or and eventual mHM restart file are masked using the provided mask file
    if "dem" in input_file.name.lower():  # or "mhm" in f.name.lower()
        if mask_da is not None:
            lon_key_mask = get_coord_key(mask_da, lon=True)
            lat_key_mask = get_coord_key(mask_da, lat=True)
            latlon_files.set_dem_output_file(output_file)
            logger.info("Masking file")
            mask_regridded = regrid_mask(
                mask_ds=mask_da,
                lon_key_mask=lon_key_mask,
                lat_key_mask=lat_key_mask,
                target_lon=ds_cropped[lon_key].data,
                target_lat=ds_cropped[lat_key].data,
            )
            ds_cropped = ds_cropped.where(mask_regridded == 1, np.nan)
        else:
            logger.info("Can't mask dem file because no mask was provided.")
    try:
        write_to_file(ds_cropped, output_file)
    except Exception as e:
        logger.warning(f"First try writing the file failed: {e}")
        logger.info("Changing datatype to float")
        for var_name in ds_cropped.data_vars:
            ds_cropped[var_name] = ds_cropped[var_name].astype(float)
        write_to_file(ds_cropped, output_file)
    logger.info(f"Written to {output_file}")
    return latlon_files


@log_arguments()
def crop_mhm_setup(
    mask_da,
    output_path,
    input_path,
    overwrite=True,
    l1_resolution=None,
    l11_resolution=None,
    lonslice=None,
    latslice=None,
    crs=None,
    n_jobs=1,
    filename="*.*",
    recursive_depth=5,
    available_mem_gib=5,
):
    """Cut out an existing mhm domain setup using a mask file."""
    # check if the input is correct
    output_path = Path(output_path)
    input_path = Path(input_path)
    # recusively get all the files from the input path if it is a dir
    logger.info(
        f"Cropping to: longitude ({lonslice.start}, {lonslice.stop}) and latitude ({latslice.stop}, {latslice.start})"
    )
    files = []
    if input_path.is_dir():
        for depth in range(recursive_depth):
            files.extend(input_path.glob("*/" * depth + filename))
    else:
        files = [input_path]

    # cut and copy each file
    list_latlon_files = Parallel(n_jobs=n_jobs, backend="loky")(
        delayed(crop_file)(
            input_file=f,
            mask_da=mask_da,
            latslice=latslice,
            lonslice=lonslice,
            output_path=output_path,
            input_path=input_path,
            overwrite=overwrite,
            available_mem_gib=available_mem_gib,
        )
        for f in files
    )
    latlon_files = LatlonFiles()
    latlon_files.set_by_list_of_objects(list_latlon_files)
    if l1_resolution is not None and latlon_files.are_set():
        logger.info("Creating latlon")
        call_create_latlon(
            latlon_files.dem_output_file,
            l1_resolution,
            l11_resolution,
            latlon_files.latlon_output_file,
            latlon_files.meteo_header_path,
            crs,
        )
