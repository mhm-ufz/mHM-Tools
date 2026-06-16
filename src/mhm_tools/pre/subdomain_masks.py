"""Create subdomain masks for parallel mRM routing domains.

The module splits a basin ID river network into independent subdomains from a
cluster definition, fills gaps by nearest-neighbor assignment, and writes mask
files that can be used to run regional parts of a larger routing network.

Authors
-------
- Simon Lüdke
Based on a script by
-------
- Robert Schweppe
- Matthias Kelbling
"""

import logging
from pathlib import Path

import matplotlib as mpl
import numpy as np
import xarray as xr
from scipy.interpolate import NearestNDInterpolator

from mhm_tools.common.file_handler import get_xarray_ds_from_file, write_xarray_to_file
from mhm_tools.common.logger import ErrorLogger, log_arguments

logger = logging.getLogger(__name__)

# GLOBAL VARIABLES
# Coordinate arrays for the shape of Greenland in lons, lats
REF_FILE_ENCODING = {
    "elevtn": {"dtype": "float32", "_FillValue": -9999.0, "zlib": True, "complevel": 4},
    "basin": {"dtype": "int32", "_FillValue": -9999, "zlib": True, "complevel": 4},
    "flwdir": {"dtype": "int32", "_FillValue": -9999, "zlib": True, "complevel": 4},
    # "upgrid": {"dtype": "int32", "_FillValue": -9999, "zlib": True, "complevel": 4},
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
    """A class for creating subdomain masks based on input data.

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
        land_mask_variable="land_mask",
    ):
        # unique basins ids, need to be in variable 'basin'
        self.ref_file = basin_id_file

        # clustered basins ids, need to be in variable 'mask', can be any resolution
        self.pgb_file = basin_clusters

        # land mask and grid of target resolution, need to be in integer variable 'land_mask'
        self.land_file = land_mask

        self.output_dir = Path(output_dir)
        self.land_mask_variable = land_mask_variable
        out_dir_path = self.output_dir
        if not out_dir_path.is_dir():
            out_dir_path.mkdir(parents=True)
        self.out_file_name = str(out_dir_path / Path(output_file_name).stem)

    @staticmethod
    def read_var(fname, var_name):
        """Read a variable from a netcdf file."""
        logger.info(f"reading variable {var_name} from {fname}")
        with get_xarray_ds_from_file(fname) as ds:
            return ds[var_name]

    @staticmethod
    def get_mask_from_polygon(arr, vertices):
        """Create a boolean mask for points inside a polygon.

        The input `arr` is a 2D array with `lat` and `lon` coordinates; the mask is
        True for cells whose (lon, lat) fall inside the polygon defined by
        `vertices`.

        Parameters
        ----------
        arr : xarray.DataArray
            2D data array with coordinates `lat` and `lon`.
        vertices : sequence[tuple[float, float]]
            Polygon vertices as (lon, lat) pairs.

        Returns
        -------
        numpy.ndarray
            Boolean mask with the same shape as `arr`, True inside the polygon.
        """
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
        """Create subdomain masks based on the input data.

        Returns
        -------
        None
        """
        logger.info("Global domain selected. Creating subdomains...")
        if self.pgb_file is None:
            msg = "Basin cluser file not provided even tho the input is global."
            with ErrorLogger(logger):
                raise ValueError(msg)
        new_ids = self.read_var(fname=self.ref_file, var_name="basin")
        orig_ids = self.read_var(fname=self.pgb_file, var_name="mask")
        land_mask = self.read_var(
            fname=self.land_file, var_name=self.land_mask_variable
        ).astype(bool)
        ds_ref_file = get_xarray_ds_from_file(self.ref_file).sel(
            lat=land_mask.lat, lon=land_mask.lon, method="nearest"
        )
        for coord in ["latitude", "longitude"]:
            if coord in ds_ref_file:
                ds_ref_file = ds_ref_file.drop(coord)

        file_basins_remapped = (
            self.output_dir / "unique_basin_ids_03min_agg54classes.nc"
        )
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
            write_xarray_to_file(
                ds=new_ids_remapped,
                file_path=file_basins_remapped,
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

            sub_mask = (new_ids_remapped.values == basin_id) & ~np.isnan(land_mask)

            fname = self.out_file_name + f"_{i:02}.nc"
            ds_sub_ref_file = ds_ref_file.copy()
            for data_var in ds_sub_ref_file.data_vars:
                ds_sub_ref_file[data_var].values[~sub_mask] = np.nan
            write_xarray_to_file(
                ds=ds_sub_ref_file, file_path=fname, encoding=REF_FILE_ENCODING
            )
            logger.info(f"Wrote to {fname}")

    def use_land_mask(self, lat, lon):
        """Reencode and mask the input files."""
        logger.info("Non global file selected. Only reencoding and masking the input.")
        res = lon[1] - lon[0]
        logger.debug(f"lat={slice(lat.values[0], lat.values[-1])}")
        logger.debug(f"lon={slice(lon.values[0], lon.values[-1])}")
        # Read and slice the land mask
        land_mask = self.read_var(fname=self.land_file, var_name="land_mask").astype(
            bool
        )
        land_mask = land_mask.sel(
            lat=slice(lat[0], lat[-1] - res), lon=slice(lon[0] - res, lon[-1])
        )

        # Read and slice the reference file based on the land mask coordinates
        logger.info(f"Reading {self.ref_file}")
        ds_ref_file = get_xarray_ds_from_file(self.ref_file).sel(
            lat=land_mask.lat, lon=land_mask.lon, method="nearest"
        )
        # Drop any redundant coordinates
        ds_ref_file = ds_ref_file.drop_vars(["latitude", "longitude"], errors="ignore")

        # Apply the land mask to all variables in the dataset
        logger.info("Applying land mask to the dataset")
        logger.debug(f"land_mask {land_mask}")
        logger.debug(f"lon land_mask: {land_mask.lon}")
        logger.debug(f"ds_ref {ds_ref_file}")
        ds_sub_ref_file = ds_ref_file.copy()
        # first process fdir
        logger.info("processing fdir")
        data_var_values = ds_sub_ref_file["flwdir"].values
        data_var_values[np.isnan(land_mask)] = np.nan
        ds_sub_ref_file["flwdir"].values = data_var_values

        for data_var in ds_sub_ref_file.data_vars:
            if data_var == "flwdir":
                continue
            logger.info(f"processing {data_var}")
            data_var_values = ds_sub_ref_file[data_var].values
            logger.debug(f"lon {data_var}: {ds_sub_ref_file[data_var].lon}")
            # replace all values where land mask is nan with nan
            ds_sub_ref_file[data_var] = ds_sub_ref_file[data_var].where(
                land_mask != 0, np.nan
            )

        # Write the output to a netCDF file
        fname = self.out_file_name + ".nc"
        logger.info(f"Writing to {fname}")
        write_xarray_to_file(
            ds=ds_sub_ref_file, file_path=fname, encoding=REF_FILE_ENCODING
        )


@log_arguments()
def create_subdomain_masks(
    output_dir,
    output_file_name,
    basin_id_file,
    basin_clusters,
    land_mask,
    land_mask_variable,
):
    """Create subdomain masks based on the provided input parameters.

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
        land_mask_variable=land_mask_variable,
    )
    with get_xarray_ds_from_file(basin_id_file) as ds:
        lat = ds.lat
        lon = ds.lon
        # if input is not global only create a file else create all subdomains
        if (
            np.max(lon) - np.min(lon) != 360
            and (np.max(lat) - np.min(lat) < 130 or np.max(lat) - np.min(lat) > 180)
        ) or basin_clusters is None:
            csm.use_land_mask(lat, lon)
        else:
            csm.create_subdomains()
