"""Crop an existing mhm-setup by providing a mask file."""

import logging
import shutil
from pathlib import Path

import numpy as np
import xarray as xr

from mhm_tools.common.file_handler import (
    create_header,
    get_xarray_ds_from_file,
    write_xarray_to_ascii,
)
from mhm_tools.common.logger import ErrorLogger, log_arguments
from mhm_tools.common.xarray_utils import get_coord_key, get_single_data_var
from mhm_tools.pre.latlon import create_latlon

logger = logging.getLogger(__name__)


def regrid_mask(mask_ds, ds2, lonkey1, latkey1, lonkey2, latkey2, mask_key=None):
    """Regrid a xarray mask dataset mask_ds to the resolution of a second dataset ds2."""
    lon1 = mask_ds[lonkey1].values
    lat1 = mask_ds[latkey1].values
    res1 = lat1[0] - lat1[1]
    lon2 = ds2[lonkey2].values
    lat2 = ds2[latkey2].values
    res2 = lat2[0] - lat2[1]
    if res2 > res1:
        results = np.full((len(lat2), len(lon2)), 0.0)
        for i, lat in enumerate(lat2):
            for j, lon in enumerate(lon2):
                for n, lat1 in enumerate(mask_ds.lat.values):
                    if lat1 < (lat - res2 / 2) or lat1 > (lat + res2 / 2):
                        continue
                    for m, lon1 in enumerate(mask_ds.lon.values):
                        if lon1 < lon - res2 / 2 or lon1 > lon + res2 / 2:
                            continue
                        if mask_key is not None:
                            results[i][j] += mask_ds[mask_key].values[n, m]
                        else:
                            results[i][j] += mask_ds.values[n, m]

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


def crop_file_with_header(ds, file_path, mask, output_path):
    """Crop the nc file and create a new header file for the new coordinates."""
    header = file_path.parent / "header.txt"
    data_var = get_single_data_var(ds)
    with header.open("r") as h:
        d = {}
        logger.debug(f"Reading out header.txt file {header}")
        for line in h.readlines():
            line_content = line.strip().split(" ")
            logger.debug(f"{line_content[0].strip()} = {line_content[-1].strip()}")
            d[line_content[0].strip()] = float(line_content[-1].strip())
        lon = np.arange(
            d["xllcorner"], d["xllcorner"] + d["cellsize"] * d["ncols"], d["cellsize"]
        )
        # reverse order for lat (TODO: Make this resistant input with south north ordering)
        lat = np.arange(
            d["yllcorner"] + d["cellsize"] * (d["nrows"]-1), d["yllcorner"]-d["cellsize"], -d["cellsize"]
        )
        logger.info(ds[data_var].shape)
        lon_key = get_coord_key(ds, lon=True, raise_exception=True)
        lat_key = get_coord_key(ds, lat=True, raise_exception=True)
        # x values
        mask_res = round(mask.lon.values[1] - mask.lon.values[0], 6)
        x_mask = (lon >= float(mask.lon.min()) - mask_res / 2) & (
            lon < float(mask.lon.max()) - mask_res / 2
        )
        x = np.arange(0, ds.sizes[lon_key], 1)
        x_cropped = x[x_mask]
        # y values
        y_mask = (lat >= float(mask.lat.min()) - mask_res / 2) & (
            lat < float(mask.lat.max()) - mask_res / 2
        )
        y = np.arange(ds.sizes[lat_key], 0, -1) - 1
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
            data = ds[data_var]
            data = data[:, y_mask, :]
            data = data[:, :, x_mask]
            data_array = xr.DataArray(
                data=data,
                dims=["time", lat_key, lon_key],
                coords={"time": ds.time, lat_key: lat[y_mask], lon_key: lon[x_mask]},
                name=data_var,
                attrs={"nodata_value": d["NODATA_value"]},
            )
            # Convert to Dataset
            return xr.Dataset({data_var: data_array}), header_out_path
        except IndexError as e:
            with ErrorLogger(logger):
                raise e


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


@log_arguments()
def crop_mhm_setup(
    mask_file,
    output_path,
    input_path,
    overwrite=True,
    l1_resolution=None,
    l11_resolution=None,
    crs=None,
):
    """Cut out an existing mhm domain setup using a mask file."""
    input_path = Path(input_path)
    output_path = Path(output_path)

    # recusively get all the files from the input path
    files = []
    for depth in range(3):  # Depth 0 to 2
        files.extend(input_path.glob("*/" * depth + "*.*"))

    with xr.open_dataset(mask_file) as mask_ds:
        mask_key = next(
            key for key in ["mask", "land_mask"] if key in mask_ds.data_vars
        )
        mask_da = mask_ds[mask_key].astype(float)
        latslice = slice(mask_da.lat.values[-1], mask_da.lat.values[0])
        lonslice = slice(mask_da.lon.values[0], mask_da.lon.values[-1])
        latlon_output_file = None
        dem_output_file = None
        meteo_header_path = None
        logger.info(
            f"Masking with lon {mask_da.lon.min()} to {mask_da.lon.max()} and lat: {mask_da.lat.min()} to {mask_da.lat.max()}"
        )
        # cut and copy each file
        for f in files:
            logger.info(f"Cropping the file {f}")
            output_file = output_path / f.relative_to(input_path)
            output_file.parent.mkdir(parents=True, exist_ok=True)
            if output_file.is_file() and not overwrite:
                logger.info("Target file already exists. Cropping is scipped.")
                continue

            if f.suffix in [".asc", ".nc"]:
                ds = get_xarray_ds_from_file(f)
            else:
                if "header" in f.name.lower():
                    # header files are not copied but recreated as they change
                    continue
                # other txt and markdown files are copied as they nomaly contain description or class definitions but do not change with domain cropping
                shutil.copy(f, output_file)
                logger.debug(f"Copied file {f.name} to {output_file}")
                continue
            logger.debug(f"read in dataset: {ds}")
            # Handling of special cases:
            # 1. latlon file: The latlon file contains coordinates on multiple resolutions that all have to be croped
            if "latlon" in f.name.lower():
                logger.info(
                    "Latlon cropping depreciated will implement new latlon creation using the mhm-tools latlon functionality."
                )
                latlon_output_file = output_file
                continue
            # 2. Restart files are complex and are not yet implemented. mHM restart files can be croped, mRM restart files can't (?).
            if "restart" in f.name.lower():
                logger.warning(
                    f"Restart file {f} could not be copied as that is not yet implemented."
                )
                continue
            # 3. Files that are in the same folder as a header file. Typical examples are meteo datasets such as temperature or precipitation
            if list(f.parent.glob("header.txt")):
                logger.debug("Cropping and writing new header file...")
                ds_croped, header_path = crop_file_with_header(
                    ds, f, mask_da, output_path / f.parent.relative_to(input_path)
                )
                lat_key = get_coord_key(ds_croped, lat=True)
                lon_key = get_coord_key(ds_croped, lon=True)
                if f.stem in ["pre", "pet", "tavg"]:
                    meteo_header_path = header_path
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
                logger.debug(f"File lon: {ds[lon_key].values}")
                logger.debug(f"lon_slice: {lonslice}")
                logger.debug(f"File lat: {ds[lat_key].values}")
                logger.debug(f"lat_slice: {latslice}")
                continue

            # only the dem file or and eventual mHM restart file are masked using the provided mask file
            if "dem" in f.name.lower():  # or "mhm" in f.name.lower()
                dem_output_file = output_file
                logger.info("Masking file")
                mask_regridded = regrid_mask(
                    mask_ds=mask_da,
                    ds2=ds_croped,
                    lonkey1="lon",
                    latkey1="lat",
                    lonkey2=lon_key,
                    latkey2=lat_key,
                )
                ds_croped = ds_croped.where(mask_regridded == 1, np.nan)
            try:
                write_to_file(ds_croped, output_file)
            except Exception:
                logger.warning("First try writing the file failed: {e}")
                logger.info("Changing datatype to float")
                for var_name in ds.data_vars:
                    ds_croped[var_name] = ds_croped[var_name].astype(float)
                write_to_file(ds_croped, output_file)
            logger.info(f"Written to {output_file}")

        if l1_resolution is not None and dem_output_file is not None:
            call_create_latlon(
                dem_output_file,
                l1_resolution,
                l11_resolution,
                latlon_output_file,
                meteo_header_path,
                crs,
            )
