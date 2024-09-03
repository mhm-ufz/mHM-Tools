"""
Create the catchment file for mRM.

Authors
-------
- Simon Lüdke
Based on a script by
-------
- Robert Schweppe
- Matthias Kelbling
"""

from mhm_tools.common.logger import logger
from pathlib import Path

import matplotlib as mpl
import numpy as np
import xarray as xr
from scipy.interpolate import NearestNDInterpolator


# GLOBAL VARIABLES
# Coordinate arrays for the shape of Greenland in lons, lats
REF_FILE_ENCODING = {
    "elevtn": {"dtype": "float32", "_FillValue": -9999.0, "zlib": True, "complevel": 4},
    "basin": {"dtype": "int32", "_FillValue": -9999, "zlib": True, "complevel": 4},
    "flwdir": {"dtype": "int32", "_FillValue": -9999, "zlib": True, "complevel": 4},
    "upgrid": {"dtype": "int32", "_FillValue": -9999, "zlib": True, "complevel": 4},
    "uparea_grid": {
        "dtype": "float32",
        "_FillValue": -9999.0,
        "zlib": True,
        "complevel": 4,
    },
    "lat": {"dtype": "float32", "_FillValue": -9999.0, "zlib": True, "complevel": 4},
    "lon": {"dtype": "float32", "_FillValue": -9999.0, "zlib": True, "complevel": 4},
}
COMPRESSION_DICT = {"zlib": True, "complevel": 4}
GREENLAND_COORDS = np.array(
    [
        [-43.65, -17.25, -7.05, -56.05, -60.05, -69.85, -73.65, -71.85, -46.65],
        [58.25, 69.85, 84.65, 85.85, 82.65, 79.45, 79.05, 75.25, 57.65],
    ]
).T
FILL_VALUE = -9999

# FUNCTIONS


class CreateSubdomainMasks:
    """
    A class for creating subdomain masks based on input data.

    Parameters
    ----------
    output_dir : str
        The directory path where the output files will be saved.
    output_file_name : str
        The name of the output file.
    basin_id_file : str
        The file name of the reference basin IDs.
    basin_clusters : str
        The file name of the basin cluster IDs.
    land_mask : str
        The file name of the land mask and grid of target resolution.

    Raises
    ------
    ValueError
        If the input path is not a directory.

    """

    def __init__(
        self,
        output_dir,
        output_file_name,
        basin_id_file,
        basin_clusters,
        land_mask,
    ):
        # unique basins ids, need to be in variable 'basin'
        self.ref_file = basin_id_file

        # clustered basins ids, need to be in variable 'mask', can be any resolution
        self.pgb_file = basin_clusters

        # land mask and grid of target resolution, need to be in integer variable 'land_mask'
        self.land_file = land_mask

        self.output_dir = Path(output_dir)

        if self.output_dir.is_absolute():
            out_dir_path = self.output_dir / output_file_name
        else:
            out_dir_path = output_dir
        if not out_dir_path.is_dir():
            out_dir_path.mkdir(parents=True)
        self.out_file_name = str(out_dir_path / Path(output_file_name).stem)

    @staticmethod
    def read_var(fname, var_name):
        """Read a variable from a netcdf file."""
        logger.info(f"reading variable {var_name} from {fname}")
        with xr.open_dataset(fname) as ds:
            return ds[var_name]

    @staticmethod
    def get_mask_from_polygon(arr, vertices):
        """Create a mask on an array with lon and lat attributes for given list of vertices."""
        polygon = mpl.path.Path(vertices)
        # mask out only the values in arr that fall within bbox of polygon, convert them to points
        bbox = polygon.get_extents()
        bbox_lon_mask = (bbox.xmin < arr.lon) & (bbox.xmax > arr.lon)
        bbox_lat_mask = (bbox.ymin < arr.lat) & (bbox.ymax > arr.lat)
        lon2d, lat2d = np.meshgrid(arr.lon[bbox_lon_mask], arr.lat[bbox_lat_mask])
        points = np.hstack((lon2d.reshape(-1, 1), lat2d.reshape(-1, 1)))
        # mask out the values
        bbox_mask = polygon.contains_points(points).reshape(
            int(bbox_lat_mask.sum()), int(bbox_lon_mask.sum())
        )

        # global mask, set to False
        mask = np.zeros_like(arr.data, dtype=bool)
        # insert the local mask into the global one
        mask[np.ix_(bbox_lat_mask, bbox_lon_mask)] = bbox_mask
        return mask

    def create_subdomains(self):
        """
        Create subdomain masks based on the input data.

        Returns
        -------
        None

        """
        new_ids = self.read_var(fname=self.ref_file, var_name="basin")
        orig_ids = self.read_var(fname=self.pgb_file, var_name="mask")
        land_mask = self.read_var(fname=self.land_file, var_name="land_mask").astype(
            bool
        )
        ds_ref_file = xr.open_dataset(self.ref_file).sel(
            lat=land_mask.lat, lon=land_mask.lon, method="nearest"
        )
        for coord in ["latitude", "longitude"]:
            if coord in ds_ref_file:
                ds_ref_file = ds_ref_file.drop(coord)

        file_basins_remapped = self.output_dir / "unique_basin_ids_03min_agg54classes.nc"
        if not file_basins_remapped.is_file():
            # map the 53 subbasins from PGB reference onto target grid
            logger.info(
                "remapping orignal subbasins to target grid and adding Greenland id"
            )
            orig_remapped = orig_ids.sel(
                lat=land_mask.lat, lon=land_mask.lon, method="nearest"
            )
            # add one additional subbasin for greenland
            greenland_mask = self.get_mask_from_polygon(orig_remapped, GREENLAND_COORDS)
            orig_remapped.values[greenland_mask] = 54

            # for all subbasins where the land_mask is nan, also set nan
            logger.info("remapping new subbasins to target grid")
            new_ids_remapped = new_ids.sel(
                lat=land_mask.lat, lon=land_mask.lon, method="nearest"
            )
            new_ids_remapped.values[np.isnan(land_mask)] = np.nan

            # for all original subbasins that do not cover new subbasins, interpolate
            # https://stackoverflow.com/questions/68197762/fill-nan-with-nearest-neighbor-in-numpy-array
            # in contrast to previous version, we interpolate the whole globe to ensure we do not use a nan as fill value
            logger.info("interpolating missing values in original subbasins")
            mask_fill = np.where(~np.isnan(orig_remapped.values))
            interp = NearestNDInterpolator(
                np.transpose(mask_fill), orig_remapped.values[mask_fill]
            )
            filled_data = interp(*np.indices(orig_remapped.values.shape))

            # 2nd step --- set exactly one subdomain mask for each river
            logger.info(
                "assigning each new subbasin to class defined in original subbasins"
            )
            basin_ids = np.unique(
                new_ids_remapped.values[~np.isnan(new_ids_remapped.values)]
            )
            for basin_id in basin_ids:
                # get mask of all cells for this river
                river_mask = new_ids_remapped.values == basin_id
                # get all ids of the original basins for this id
                sel_ids, sel_counts = np.unique(
                    filled_data[river_mask], return_counts=True
                )
                # select the most commonly occurring one
                new_ids_remapped.values[river_mask] = sel_ids[sel_counts.argmax()]

            # 3rd step --- write masks
            # read land_mask to intersect with subdomain_mask
            new_ids_remapped.to_netcdf(
                file_basins_remapped,
                encoding={
                    new_ids_remapped.name: {"_FillValue": FILL_VALUE, "dtype": "int16"}
                },
            )
        else:
            logger.info("using cached remapped basin ids")
            new_ids_remapped = xr.open_dataarray(file_basins_remapped)
        basin_ids = np.unique(
            new_ids_remapped.values[~np.isnan(new_ids_remapped.values)]
        )

        logger.info("writing output files")
        for i, basin_id in enumerate(basin_ids, 1):

            logger.info(f"processing subdomain {i}")

            sub_mask = new_ids_remapped.values == basin_id

            fname = self.out_file_name + f"_{i:02}.nc"
            ds_sub_ref_file = ds_ref_file.copy()
            for data_var in ds_sub_ref_file.data_vars:
                ds_sub_ref_file[data_var].values[~sub_mask] = np.nan

            ds_sub_ref_file.to_netcdf(fname, encoding=REF_FILE_ENCODING)


def create_subdomain_masks(
    output_dir, output_file_name, basin_id_file, basin_clusters, land_mask
):
    """
    Create subdomain masks based on the provided input parameters.

    Args:
        output_dir (str): The directory where the output files will be saved.
        output_file_name (str): The name of the output file.
        basin_id_file (str): The file containing the basin IDs.
        basin_clusters (str): The file containing the basin clusters.
        land_mask (str): The land mask file.

    Returns
    -------
        None
    """
    csm = CreateSubdomainMasks(
        output_dir=output_dir,
        output_file_name=output_file_name,
        basin_id_file=basin_id_file,
        basin_clusters=basin_clusters,
        land_mask=land_mask,
    )
    csm.create_subdomains()
