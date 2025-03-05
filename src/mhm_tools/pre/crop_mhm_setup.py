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
from mhm_tools.common.xarray_utils import get_coord_key, get_single_data_var, induce_data_var_from_file_name
from mhm_tools.pre.latlon import create_latlon

logger = logging.getLogger(__name__)

class LatlonFiles:
    latlon_output_file=None
    dem_output_file=None
    meteo_header_path=None

    def set_latlon_output_file(self, path):
        logger.info(f'Setting latlon_output_file to {path}')
        self.latlon_output_file = path
    def set_dem_output_file(self, path):
        logger.info(f'Setting dem_output_file to {path}')
        self.dem_output_file = path
    def set_meteo_header_path(self, path):
        logger.info(f'Setting meteo_header_path to {path}')
        self.meteo_header_path = path

    def get_latlon_output_file(self):
        return self.latlon_output_file
    def get_dem_output_file(self):
        return self.dem_output_file
    def get_meteo_header_path(self):
        return self.meteo_header_path

    
    def are_set(self):
        all_set = True
        if self.latlon_output_file is None:
            logger.info(f'latlon_output_file not set')
            all_set = False
        if self.dem_output_file is None:
            logger.info(f'dem_output_file not set')
            all_set = False
        if self.meteo_header_path is None:
            logger.info(f'meteo_header_path not set')
            all_set = False
        return all_set
    
    def set_by_list_of_paths(self, path_list):
        for latlon_output_file, dem_output_file, meteo_header_path in path_list:
            if latlon_output_file is not None:
                self.set_latlon_output_file(latlon_output_file)
            if dem_output_file is not None:
                self.set_dem_output_file(dem_output_file)
            if meteo_header_path is not None:
                self.set_meteo_header_path(meteo_header_path)
        
    def set_by_list_of_objects(self, obj_list):
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

def regrid_mask(mask_ds, ds2, lon_key_mask, lat_key_mask, lonkey2, latkey2, mask_key=None):
    """Regrid a xarray mask dataset mask_ds to the resolution of a second dataset ds2."""
    lon1 = mask_ds[lon_key_mask].data
    lat1 = mask_ds[lat_key_mask].data
    res1 = lat1[0] - lat1[1]
    lon2 = ds2[lonkey2].data
    lat2 = ds2[latkey2].data
    res2 = lat2[0] - lat2[1]
    if res2 > res1:
        results = np.full((len(lat2), len(lon2)), 0.0)
        for i, lat in enumerate(lat2):
            for j, lon in enumerate(lon2):
                for n, lat1 in enumerate(mask_ds[lat_key_mask].data):
                    if lat1 < (lat - res2 / 2) or lat1 > (lat + res2 / 2):
                        continue
                    for m, lon1 in enumerate(mask_ds[lon_key_mask].data):
                        if lon1 < lon - res2 / 2 or lon1 > lon + res2 / 2:
                            continue
                        if mask_key is not None:
                            results[i][j] += mask_ds[mask_key].data[n, m]
                        else:
                            results[i][j] += mask_ds.data[n, m]

        results /= np.nanmax(results)
        mask = results > 1e-3
    elif res2 == res1:
        return mask_ds
    else:
        msg = "mask coarser than file not yet implemented"
        with ErrorLogger(logger):
            raise Exception(msg)
    results[mask] = 1
    results[~mask] = 0
    return results


def write_to_file(ds, output_file: Path):
    """Take xarray Dataset and write it to file. File type depends on path suffix."""
    suffix = output_file.suffix
    if suffix == ".asc":
        write_xarray_to_ascii(ds, output_file)
    elif suffix == ".nc":
        ds.to_netcdf(output_file)


def crop_file_with_header(ds_in, file_path, mask, output_path, lon_key_mask, lat_key_mask):
    """Crop the nc file and create a new header file for the new coordinates."""
    pres = 1e-5
    header = file_path.parent / "header.txt"
    data_var = get_single_data_var(ds_in)
    if data_var is None: 
        data_var = induce_data_var_from_file_name(ds_in, file_path)
        if data_var is None:
            logger.error(f'File {file_path} could not be croped because the data_var could not be determined.')
            return None, None
        logger.debug(f'Found data_var={data_var}')
    with header.open("r") as h:
        d = {}
        logger.debug(f"Reading out header.txt file {header}")
        for line in h.readlines():
            if not line.strip():
                continue
            line_content = line.strip().split(" ")
            logger.debug(f"{line_content[0].strip()} = {line_content[-1].strip()}")
            d[line_content[0].strip()] = float(line_content[-1].strip())
        lon = np.arange(
            d["xllcorner"], d["xllcorner"] + d["cellsize"] * d["ncols"], d["cellsize"]
        )
        # reverse order for lat (TODO: Make this resistant input with south north ordering)
        lat = np.arange(
            d["yllcorner"] + d["cellsize"] * (d["nrows"] - 1),
            d["yllcorner"] - d["cellsize"],
            -d["cellsize"],
        )
        logger.info(ds_in[data_var].shape)
        lon_key = get_coord_key(ds_in, lon=True, raise_exception=True)
        lat_key = get_coord_key(ds_in, lat=True, raise_exception=True)
        # x values
        mask_res = round(mask[lon_key_mask].data[1] - mask[lon_key_mask].data[0], 6)
        x_mask = (lon >= float(mask[lon_key_mask].min()) - mask_res / 2 - pres) & (
            lon < float(mask[lon_key_mask].max()) - mask_res / 2 + pres
        )
        x = np.arange(0, ds_in.sizes[lon_key], 1)
        x_cropped = x[x_mask]
        # y values
        y_mask = (lat >= float(mask[lat_key_mask].min()) - mask_res / 2 - pres) & (
            lat < float(mask[lat_key_mask].max()) - mask_res / 2 + pres
        )
        y = np.arange(ds_in.sizes[lat_key], 0, -1) - 1
        y_cropped = y[y_mask]
        # write header file
        header_out_path = output_path / header.name
        xll = d["xllcorner"] + d["cellsize"] * np.nanmin(x_cropped)
        yll = d["yllcorner"] + d["cellsize"] * np.nanmin(y_cropped)
        ncols = len(x_cropped)
        nrows = len(y_cropped)
        header_str = f"""
ncols                {ncols}
nrows                {nrows}
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
            data = ds_in[data_var]
            data = data[:, y_mask, :]
            data = data[:, :, x_mask]
            data_array = xr.DataArray(
                data=data,
                dims=["time", lat_key, lon_key],
                coords={"time": ds_in.time, lat_key: lat[y_mask], lon_key: lon[x_mask]},
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
    """Create header dictionaries for the different resolutions and call create latlon to create a latlon file for the setup."""
    # create new latlon file
    logger.info("Creating new latlon file")
    with get_xarray_ds_from_file(dem_output_file) as ds_dem:
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

def crop_file(f, mask_da, latslice, lonslice, output_path, input_path, overwrite):
    logger.info(f"Cropping the file {f}")
    output_file = output_path / f.relative_to(input_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    latlon_files = LatlonFiles()
    lon_key_mask = get_coord_key(mask_da, lon=True)
    lat_key_mask = get_coord_key(mask_da, lat=True)
    if output_file.is_file() and not overwrite:
        logger.info("Target file already exists. Cropping is scipped.")
        return latlon_files
    if f.suffix in [".asc", ".nc"]:
        ds = get_xarray_ds_from_file(f)
    else:
        if "header" in f.name.lower():
            # header files are not copied but recreated as they change
            return latlon_files
        # other txt and markdown files are copied as they nomaly contain description or class definitions but do not change with domain cropping
        try:
            shutil.copy(f, output_file)
            logger.debug(f"Copied file {f.name} to {output_file}")
        except Exception as e:
            logger.error(f"Can't copy {f} because of {e}")
        return latlon_files
    logger.debug(f"read in dataset: {ds}")
    # Handling of special cases:
    # 1. latlon file: The latlon file contains coordinates on multiple resolutions that all have to be croped
    if "latlon" in f.name.lower():
        logger.info(
            "Latlon cropping depreciated will implement new latlon creation using the mhm-tools latlon functionality."
        )
        latlon_files.set_latlon_output_file( output_file)
        return latlon_files
    # 2. Restart files are complex and are not yet implemented. mHM restart files can be croped, mRM restart files can't (?).
    if "restart" in f.name.lower():
        logger.warning(
            f"Restart file {f} could not be copied as that is not yet implemented."
        )
        return latlon_files
    # 3. Files that are in the same folder as a header file. Typical examples are meteo datasets such as temperature or precipitation
    if list(f.parent.glob("header.txt")):
        logger.debug("Cropping and writing new header file...")
        ds_croped, header_path = crop_file_with_header(
            ds, f, mask_da, output_path / f.parent.relative_to(input_path), lon_key_mask, lat_key_mask
        )
        if ds_croped is None and header_path is None:
            return latlon_files
        lat_key = get_coord_key(ds_croped, lat=True)
        lon_key = get_coord_key(ds_croped, lon=True)
        if f.stem in ["pre", "pet", "tavg"]:
            latlon_files.set_meteo_header_path(header_path)
    # 4. All other netcdf files containing mostly morphological data.
    else:
        lat_key = get_coord_key(ds, lat=True)
        lon_key = get_coord_key(ds, lon=True)
        logger.debug(
            f"Selecting {f.name} using lon:{lonslice} and lat:{latslice}"
        )
        ds_croped = ds.sel({lon_key: lonslice, lat_key: latslice})
        if ds_croped[lat_key].shape[0] < 2:
            ds_croped = ds.sel(
                {
                    lon_key: lonslice,
                    lat_key: slice(latslice.stop, latslice.start),
                }
            )
    if ds_croped[lat_key].shape[0] < 2 or ds_croped[lon_key].shape[0] < 2:
        logger.warning(
            "Copying of the file is not possible because after cropping the file is empty."
        )
        # logger.debug(f"File lon: {ds[lon_key].data}")
        # logger.debug(f"lon_slice: {lonslice}")
        # logger.debug(f"File lat: {ds[lat_key].data}")
        # logger.debug(f"lat_slice: {latslice}")
        return latlon_files

    # only the dem file or and eventual mHM restart file are masked using the provided mask file
    if "dem" in f.name.lower():  # or "mhm" in f.name.lower()
        latlon_files.set_dem_output_file(output_file)
        logger.info("Masking file")
        mask_regridded = regrid_mask(
            mask_ds=mask_da,
            ds2=ds_croped,
            lon_key_mask=lon_key_mask,
            lat_key_mask=lat_key_mask,
            lonkey2=lon_key,
            latkey2=lat_key,
        )
        ds_croped = ds_croped.where(mask_regridded == 1, np.nan)
    try:
        write_to_file(ds_croped, output_file)
    except Exception as e:
        logger.warning(f"First try writing the file failed: {e}")
        logger.info("Changing datatype to float")
        for var_name in ds_croped.data_vars:
            ds_croped[var_name] = ds_croped[var_name].astype(float)
        write_to_file(ds_croped, output_file)
    logger.info(f"Written to {output_file}")
    return latlon_files


@log_arguments()
def crop_mhm_setup(
    mask_file,
    output_path,
    input_path,
    overwrite=True,
    l1_resolution=None,
    l11_resolution=None,
    crs=None,
    n_jobs=1
):
    """Cut out an existing mhm domain setup using a mask file."""
    # check if the input is correct
    mask_file = Path(mask_file)
    output_path = Path(output_path)
    input_path = Path(input_path)
    error_msg = ""
    if not mask_file.is_file():
        error_msg += "`mask_file` must be a file. \n"
    if not input_path.exists():
        error_msg += "`input_path` must exist. \n"
    if error_msg:
        with ErrorLogger(logger):
            raise ValueError(error_msg)
    # recusively get all the files from the input path if it is a dir
    files = []
    if input_path.is_dir():
        for depth in range(3):  # Depth 0 to 2
            files.extend(input_path.glob("*/" * depth + "*.*"))
    else:
        files = [input_path]
    with xr.open_dataset(mask_file) as mask_ds:
        mask_key = next(
            key for key in ["mask", "land_mask"] if key in mask_ds.data_vars
        )
        mask_da = mask_ds[mask_key].astype(float)
        lon_key_mask = get_coord_key(mask_da, lon=True)
        lat_key_mask = get_coord_key(mask_da, lat=True)
        pres = 1e-5
        mask_res = round(mask_da[lon_key_mask].data[1] - mask_da[lon_key_mask].data[0], 6)
        latslice = slice(mask_da[lat_key_mask].data[-1]-mask_res/2-pres, mask_da[lat_key_mask].data[0]+mask_res/2+pres)
        lonslice = slice(mask_da[lon_key_mask].data[0]-mask_res/2-pres, mask_da[lon_key_mask].data[-1]+mask_res/2+pres)
        logger.info(
            f"Masking with lon {mask_da[lon_key_mask].min().item()} to {mask_da[lon_key_mask].max().item()} and lat: {mask_da[lat_key_mask].min().item()} to {mask_da[lat_key_mask].max().item()}"
        )
        # cut and copy each file
        list_latlon_files = Parallel(n_jobs=n_jobs, backend="loky")(
            delayed(crop_file)(
                f, mask_da, latslice, lonslice, output_path, input_path, overwrite
                )
            for f in files
        )
        latlon_files = LatlonFiles()
        latlon_files.set_by_list_of_objects(list_latlon_files)
        if l1_resolution is not None and latlon_files.are_set():
            logger.info('Creating latlon')
            call_create_latlon(
                latlon_files.dem_output_file,
                l1_resolution,
                l11_resolution,
                latlon_files.latlon_output_file,
                latlon_files.meteo_header_path,
                crs
            )
