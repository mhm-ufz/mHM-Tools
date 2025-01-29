"""Crop an existing mhm-setup by providing a mask file."""

import logging
import shutil
from pathlib import Path

from mhm_tools.common.file_handler import read_ascii_to_xarray
from mhm_tools.common.file_handler import write_xarray_to_ascii
import numpy as np
import xarray as xr

from mhm_tools.common.logger import ErrorLogger, log_arguments
from mhm_tools.common.xarray_utils import get_coord_key

logger = logging.getLogger(__name__)


def regrid_mask(mask_ds, mask_key, ds2, lonkey1, latkey1, lonkey2, latkey2):
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
                        results[i][j] += mask_ds[mask_key].values[n, m]
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

def crop_file_with_header(ds, file_path, mask, output_path, lon_key, lat_key):
    """Crop the nc file and create a new header file for the new coordinates."""
    header = file_path.parent / "header.txt"
    with header.open("r") as h:
        d = {}
        for line in h.readlines():
            line_content = line.strip().split(" ")
            d[line_content[0].strip()] = float(line_content[-1].strip())
        lon = np.arange(
            d["xllcorner"], d["xllcorner"] + d["cellsize"] * d["ncols"], d["cellsize"]
        )
        lat = np.arange(
            d["yllcorner"], d["yllcorner"] + d["cellsize"] * d["nrows"], d["cellsize"]
        )
        x = ds[lon_key].values
        y = np.flip(ds[lat_key].values)
        x = x[np.where((lon >= mask.lon.values[0]) & (lon <= mask.lon.values[-1]))]
        y = y[np.where((lat >= mask.lat.values[0]) & (lat <= mask.lat.values[-1]))]
        if not (output_path / header.name).is_file():
            xll = d["xllcorner"] + d["cellsize"] * np.nanmin(x)
            yll = d["yllcorner"] + d["cellsize"] * np.nanmin(y)
            ncols = len(x)
            nrows = len(y)
            with (output_path / header.name).open("w") as nh:
                nh.write(
                    f"""
ncols                {ncols}
nrows                {nrows}
xllcorner            {xll}
yllcorner            {yll}
cellsize             {d['cellsize']}
NODATA_value         {d['NODATA_value']}
                """
                )
        try:
            lat_key = get_coord_key(ds, lat=True)
            lon_key = get_coord_key(ds, lon=True)
            slice_dict = {lon_key: slice(x[0], x[-1]), lat_key: slice(y[0], y[-1])}
            logger.debug(f"Selecting {file_path.name} using {slice_dict}")
            ds = ds.sel({lon_key: slice(x[0], x[-1]), lat_key: slice(y[-1], y[0])})
        except IndexError as e:
            with ErrorLogger(logger):
                raise e


@log_arguments()
def crop_mhm_setup(mask_file, output_path, input_path):
    """Cut out an existing mhm domain setup using a mask file."""
    input_path = Path(input_path)
    output_path = Path(output_path)

    # recusively get all the files from the input path
    files = []
    for depth in range(3):  # Depth 0 to 2
        files.extend(input_path.glob("*/" * depth + "*.*"))

    with xr.open_dataset(mask_file) as mask:
        mask_key = [key for key in ['mask', 'land_mask'] if key in mask.data_vars]
        mask[mask_key] = mask[mask_key].astype(float)
        latslice = slice(mask.lat.values[-1], mask.lat.values[0])
        lonslice = slice(mask.lon.values[0], mask.lon.values[-1])
        # cut and copy each file
        for f in files:
            logger.debug(str(f))
            output_path.mkdir(parents=True, exist_ok=True)
            output_file = output_path / f.name
            if output_file.is_file():
                continue
            if f.suffix == ".asc":
                ds = read_ascii_to_xarray(f)
            elif f.suffix == ".nc":
                ds = xr.open_dataset(f)
            else:
                if "header" in f.name.lower():
                    # header files are not copied but recreated as they change
                    continue
                # other txt and markdown files are copied as they nomaly contain description or class definitions but do not change with domain cropping
                shutil.copy(f, output_file)
                logger.debug(f"Copied file {f.name} to {output_file}")
                continue

            # Handling of special cases:
            # 1. latlon file: The latlon file contains coordinates on multiple resolutions that all have to be croped
            if "latlon" in f.name.lower():
                for r in ["_l0", "", "_l11"]:
                    lat_key = "yc" + r
                    lon_key = "xc" + r
                    ds = ds.sel({lon_key: lonslice, lat_key: latslice})
            # 2. Restart files are complex and are not yet implemented. mHM restart files can be croped, mRM restart files can't (?).
            elif "restart" in f.name.lower():
                continue
            # 3. Files that are in the same folder as a header file. Typical examples are meteo datasets such as temperature or precipitation
            elif list(f.parent.glob("header.txt")):
                crop_file_with_header(ds, f, mask, output_path, lon_key, lat_key)
            # 4. All other netcdf files containing mostly morphological data.
            else:
                lat_key = get_coord_key(ds, lat=True)
                lon_key = get_coord_key(ds, lon=True)
                logger.debug(
                    f"Selecting {f.name} using lon:{lonslice} and lat:{latslice}"
                )
                ds_new = ds.sel({lon_key: lonslice, lat_key: latslice})
                if ds_new.lat.shape[0] < 2:
                    ds_new = ds.sel(
                        {
                            lon_key: lonslice,
                            lat_key: slice(latslice.stop, latslice.start),
                        }
                    )
                ds = ds_new
                logger.debug(ds)
            if ds[lat_key].shape[0] < 2 or ds[lon_key].shape[0] < 2:
                logger.debug("NOT POSSIBLE AS THERE IS NO SELECTION")
                continue

            # only the dem file or and eventual mHM restart file are masked using the provided mask file
            if "mhm" in f.name.lower() or "dem" in f.name.lower():
                mask_regridded = regrid_mask(
                    mask_ds=mask,
                    mask_key=mask_key,
                    ds2=ds,
                    lonkey1="lon",
                    latkey1="lat",
                    lonkey2=lon_key,
                    latkey2=lat_key,
                )
                # Debug: print out the regridded mask in a way that makes the catchmentshape visible for small catchments
                for line in mask_regridded:
                    logger.debug(line)
                ds = ds.where(mask_regridded == 1)
            try:
                write_to_file(ds, output_file)
            except Exception as e:
                logger.warning(e)
                for var_name in ds.data_vars:
                    ds[var_name] = ds[var_name].astype(float)
                write_to_file(ds, output_file)
            logger.info(f"Written to {output_file}")
