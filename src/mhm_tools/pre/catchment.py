r"""
Create the catchment file for mRM.

Authors
-------
- Robert Schweppe
- Matthias Kelbling
- Jeisson Leal
- Simon Lüdke
"""

import logging
import pathlib as pl

import numpy as np
import pyflwdir
import xarray as xr
from scipy.ndimage import binary_dilation

from mhm_tools.common.logger import ErrorLogger, log_arguments

logger = logging.getLogger(__name__)


# GLOBAL VARIABLES
FDIR_FILLVALUE = {"d8": 247, "ldd": 255}
FDIR_SINKVALUE = {"d8": 0, "ldd": 5}
FACC_FILLVALUE = 0
FILLVALUE = -9999
# use d8 for basinex, ldd for mRM version in Ulysses
OUTPUT_FTYPE = "ldd"
CUTOFF_THRESHOLD = 170
# FUNCTIONS


# CLASSES
class Catchment:
    """Catchment class deliniating catchmetns with pyflowdir."""

    def __init__(
        self,
        ds,
        var_name,
        var="data",
        ftype=None,
        transform=None,
        out_var_name=None,
        do_shift=False,
        **kwargs,
    ):
        self.flwdir = None
        self.basin = None
        self.upgrid = None
        self.uparea_grid = None
        self.grdare = None
        self.elevtn = None
        self._fdir = None
        self.ftype = ftype
        self.catchment_mask = None
        self.out_var_name = (
            out_var_name if out_var_name is not None else f"{var_name}.nc"
        )
        self.VARIABLES = {
            "flwdir": {
                "title": f"flow direction ({self.ftype.upper()})",
                "_FillValue": FDIR_FILLVALUE[self.ftype],
                "units": "-",
            },
            "basin": {
                "title": "basin Id",
                "_FillValue": 0,
                "units": "-",
            },
            "uparea_grid": {
                "title": "accumulated data values along the flow directions",
                "_FillValue": FACC_FILLVALUE,
                "units": "m2",
            },
            "grdare": {
                "title": "rectangular grid area",
                "_FillValue": FILLVALUE,
                "units": "m2",
            },
            "elevtn": {
                "title": "outlet pixel elevation",
                "_FillValue": float(FILLVALUE),
                "units": "m",
            },
        }
        if not isinstance(self.out_var_name, str):
            self.out_var_name = f"{var_name}.nc"
        self.do_shift = do_shift
        self.ds = ds
        self.transform = transform

        data = self._modify_data(self.ds[var_name])

        if self.do_shift and self.is_data_global:
            transform = list(self.transform)
            transform[2] = 0
            self.transform = tuple(transform)

        if var == "fdir":
            self.add_fdir(data=data.data, ftype=ftype, **kwargs)
        elif var == "dem":
            self.add_dem(data=data, **kwargs)
        else:
            with ErrorLogger(logger):
                raise NotImplementedError

    @property
    def is_data_global(self):
        """Check if the longitude data is global."""
        return (
            "lon" in self.ds.coords
            and self.ds.lon.min() < (CUTOFF_THRESHOLD * -1)
            and self.ds.lon.max() > CUTOFF_THRESHOLD
        )

    def _modify_data(self, data):
        # correct circumspanning data
        if self.do_shift and self.is_data_global:
            return data.roll(lon=int(len(self.ds.lon) / 2), roll_coords=True)
        return data

    def _revert_data(self, data):
        # correct circumspanning data
        if self.do_shift and self.is_data_global:
            return np.roll(data, int(len(self.ds.lon) / 2), axis=1)
        return data

    def add_dem(self, data, **kwargs):
        """Init the FlwdirRaster class from dem."""
        # perform checks
        # self.input_ds = fill_nan_with_neighbors(self.input_ds)
        self.elevtn = self.input_ds.data
        if self._fdir is None:
            # Create a flow direction object
            logger.info(f"add_dem: kwargs: {kwargs}")
            self._fdir = pyflwdir.from_dem(
                data=self.elevtn,
                nodata=np.nan,
                transform=self.transform,
                latlon=True,
            )
            self.get_fdir()

    def add_fdir(self, data, ftype, **kwargs):
        """Init the FlwdirRaster class from fdir."""
        # perform check
        if self._fdir is None:
            mask = np.isnan(data)
            if mask.any():
                data[mask] = FDIR_FILLVALUE[self.ftype]
            data = data.astype(np.uint8)
            self._fdir = pyflwdir.from_array(data=data, ftype=self.ftype, **kwargs)
        self.get_fdir()

    def delineate_basin(self, gauge_coords, stream_order=4):
        """Deliniate the basin for a given lat and lon."""
        logger.info(f"Deliniating basin for gauge coordinates {gauge_coords}")
        gauge_coords = (gauge_coords[0], gauge_coords[1])
        self.basin = self._fdir.basins(
            xy=gauge_coords, streams=self._fdir.stream_order() >= stream_order
        )
        self.catchment_mask = self.basin > 0
        if np.all(~self.catchment_mask):
            if stream_order > 1:
                self.delineate_basin(
                    (gauge_coords[0], gauge_coords[1]), stream_order=stream_order - 1
                )
            logger.error("No catchment found for the given coordinates")
        if not np.any(np.isnan(self.basin)):
            self.basin[np.where(~self.catchment_mask)] = self.VARIABLES["basin"][
                "_FillValue"
            ]

    def upscale(self, var):
        """Upscale flow direction to taget_resolution if that is int multipe of data resolution."""
        input_res = round(abs(self.ds.lon.data[1] - self.ds.lon.data[0]), 6)
        if (
            int(self.target_resolution / input_res + 0.5)
            - (self.target_resolution / input_res)
            < 1e6
        ):
            factor = int(self.target_resolution / input_res + 0.5)
        else:
            not_int_multiple_msg = f"Upscaling only works if L1 resolution is integer muplipe of L0 resolution but L1 = {self.taget_resolution / input_res:.4f} * L0"
            raise ValueError(not_int_multiple_msg)
        if factor == 1:
            return
        self.get_upstream_area()
        logger.info(
            f"Upscaling flow direction from {input_res} to {self.target_resolution} with the fator {factor}."
        )
        self._fdir, index = self._fdir.upscale(factor, method="ihu", uparea=self.upgrid)
        self.get_fdir()

        if var == "dem":
            lat_size, lon_size = self.input_ds.shape
            # Ensure the dimensions are evenly divisible by the factor
            if lat_size % factor != 0 or lon_size % factor != 0:
                raise ValueError(
                    f"Data dimensions must be divisible by the upscaling factor of {factor}. Lat ({lat_size}/{factor})={lat_size/factor:.2f}; Lon ({lon_size}/{factor})={lon_size/factor:.2f}"
                )

            # Reshape and aggregate data
            reshaped = self.input_ds.values.reshape(
                lat_size // factor, factor, lon_size // factor, factor
            )
            aggregated = reshaped.mean(axis=(1, 3))  # Conservative mean over each block
            # Create new DataArray
            self.elevtn = aggregated

    def get_basins(self):
        """Perform the calculation of the catchment ids."""
        self.basin = self._fdir.basins()

    def get_fdir(self):
        """Perform the calculation of the flow direction."""
        self.flwdir = self._fdir.to_array(ftype=self.ftype or OUTPUT_FTYPE)

    def get_upstream_area(self):
        """Perform the calculation of the upstream catchment area."""
        self.upgrid = self._fdir.upstream_area(unit="km2").astype(int)

    def get_grid_area(self):
        """Perform the calculation of the catchment area."""
        self.grdare = self._fdir.area.astype(int)

    def get_facc(self):
        """Get the flow accumulation area."""
        data = np.ones_like(self.flwdir).astype(np.uint32)
        data[~self._fdir.mask.reshape(data.shape)] = 0
        self.uparea_grid = self._fdir.accuflux(data, nodata=0)

    @staticmethod
    def create_frame(ds, frame=0, frame_value=0):
        """If a frame is used this frame is set to no data values as a frame"""
        logger.info(f"Creating a frame of {frame} cells around the domain.")
        if frame > 0:
            for var in ds.data_vars:
                data = ds.variables[var].data[:]
                # set bounds to -9999.
                data[:frame, :] = frame_value
                data[-frame:, :] = frame_value
                data[:, :frame] = frame_value
                data[:, -frame:] = frame_value
                ds.variables[var].data[:] = data
        return ds

    def fill_adjacent_missing_with_sink(self, da, fill_value):
        """
        Replace all missing values adjacent to non-missing values with 0 in an xarray Dataset.

        Parameters
        ----------
            da (xr.Dataset): Input dataset.

        Returns
        -------
            xr.Dataset: Dataset with adjacent missing values replaced with 0.
        """
        da_filled = da.copy()

        # Mask of missing values
        missing_mask = np.isnan(da)

        # Mask of non-missing values
        non_missing_mask = ~missing_mask

        # Dilate the non-missing mask to include adjacent cells
        adjacent_mask = binary_dilation(
            non_missing_mask, structure=np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]])
        )

        # Identify adjacent missing values
        adjacent_missing = adjacent_mask & missing_mask

        # Replace adjacent missing values with 0
        da_filled = xr.where(adjacent_missing, fill_value, da)

        return da_filled

    def write(
        self,
        out_path,
        single_file=True,
        format="nc",
        cellsize=None,
        cut_by_basin=False,
        mask_file=None,
        frame=1,
        buffer=0,
    ):
        """Write the produced data to one or multiple files."""
        data_vars = {}
        out_path = pl.Path(out_path)
        if not out_path.is_dir():
            out_path.mkdir(parents=True, exist_ok=True)
        if cut_by_basin:
            lat_slice, lon_slice = self.cut_to_filled_area(buffer)
        else:
            lat_slice, lon_slice = slice(84, -56), slice(None)

        for var_name in self.VARIABLES:
            data = getattr(self, var_name)
            if cut_by_basin:
                data[~self.catchment_mask] = self.VARIABLES[var_name]["_FillValue"]
            if data is None:
                continue
            lon = self.ds.lon.data
            lat = self.ds.lat.data
            if self.target_resolution is not None:
                input_res = round(abs(lon[1] - lon[0]), 6)
                res = self.target_resolution
                lon = np.arange(
                    lon.min() - input_res / 2 + res / 2, lon.max() + res / 2, res
                )
                lat = np.arange(
                    lat.max() + input_res / 2 + res / 2, lat.min() + res / 2, -res
                )
            logger.info(
                f"Shape of lon {np.shape(lon)}, lat {np.shape(lat)}, data {np.shape(data)}"
            )
            logger.info(f"lon_min {np.min(lon):.3f}, lon_max {np.max(lon):.3f}")
            logger.info(f"lon_min {-13.1}, lon_max {10.1}")
            logger.info(f"{var_name} - mean {np.nanmean(data)}, max {np.nanmax(data)}")
            logger.info(f"# values > 1e6 {np.sum(data > 1e6)}")
            data_var = xr.Dataset(
                {var_name: (["lat", "lon"], self._revert_data(data))},
                coords={
                    "lon": lon,  # [slice(3555, 3565)],
                    "lat": lat,  # [slice(860, 870)],
                },
            )

            if single_file:
                data_vars[var_name] = data_var
            else:
                # set some attributes
                for coord in data_var.coords:
                    data_var[coord].attrs = self.ds[coord].attrs
                data_var.attrs = {
                    "title": self.VARIABLES[var_name]["title"],
                    "units": self.VARIABLES[var_name]["units"],
                    "creator": "Department of Computational Hydrosystems",
                    "institution": "Helmholtz Centre for Environmental Research - UFZ",
                }
                fname = out_path / f"{var_name}.{format}"
                if format == "nc":
                    data_var.to_netcdf(
                        fname,
                        encoding={
                            var_name: {
                                "dtype": data.dtype,
                                "_FillValue": self.VARIABLES[var_name]["_FillValue"],
                            }
                        },
                    )
                elif format == "asc":
                    cellsize = cellsize or abs(
                        float(data_var["lon"][1] - data_var["lon"][0])
                    )
                    is_ascending = bool(data_var["lat"][0] < data_var["lat"][-1])
                    with fname.open("w") as file_object:
                        file_object.write(f"ncols {data_var[var_name].shape[1]}\n")
                        file_object.write(f"nrows {data_var[var_name].shape[0]}\n")
                        file_object.write(
                            f"xllcorner {float(data_var['lon'][0] - cellsize / 2)}\n"
                        )
                        if is_ascending:
                            file_object.write(
                                f"yllcorner {float(data_var['lat'][0] - cellsize / 2)}\n"
                            )
                        else:
                            file_object.write(
                                f"yllcorner {float(data_var['lat'][-1] - cellsize / 2)}\n"
                            )
                        file_object.write(f"cellsize {cellsize}\n")
                        file_object.write(
                            f"NODATA_value {self.VARIABLES[var_name]['_FillValue']}\n"
                        )
                        if is_ascending:
                            vals = data_var[var_name].values[::-1, :]
                        else:
                            vals = data_var[var_name].values
                        np.savetxt(file_object, vals, delimiter=" ", fmt="%s")
                else:
                    with ErrorLogger(logger):
                        msg = f'Format "{format}" unknown, use one of ["nc", "asc"]'
                        raise Exception(msg)
        if single_file:
            logger.info("Write to single file.")
            ds = xr.merge(data_vars.values())
            # set some attributes
            for coord in ds.coords:
                ds[coord].attrs = self.ds[coord].attrs
            ds.attrs = {
                "title": "Hydrologic information",
                "creator": "Department of Computational Hydrosystems",
                "institution": "Helmholtz Centre for Environmental Research - UFZ",
            }
            for var_name in ds.data_vars:
                ds[var_name].attrs = {
                    "long_name": self.VARIABLES[var_name]["title"],
                    "standard_name": self.VARIABLES[var_name]["title"],
                    "units": self.VARIABLES[var_name]["units"],
                }

            logger.debug(f"lat_slice: {lat_slice}, lon_slice: {lon_slice}")
            logger.debug(f"ds: {ds}")
            mask = ds.basin > 0
            ds = self.create_frame(ds, frame, FDIR_SINKVALUE[self.ftype])
            # For the flow dir map fill masked cells adjecent to filled cells with sink instead of missing value
            # fdir_filled = self.fill_adjacent_missing_with_sink(ds['flwdir'], FDIR_SINKVALUE[self.ftype])
            # ds['flwdir'].data[:] = fdir_filled.data[:]
            ds.to_netcdf(
                out_path / self.out_var_name,
                encoding={
                    var_name: {
                        "dtype": ds[var_name].dtype,
                        "_FillValue": self.VARIABLES[var_name]["_FillValue"],
                    }
                    for var_name in ds.data_vars
                },
            )
            logger.info(f"Basin Id has been written to {out_path / self.out_var_name}")
            # use basin_id to create a mask file
            if mask_file is not None:
                # name the variable mask
                mask_file = pl.Path(mask_file)
                mask = xr.Dataset({"mask": mask}, coords={"lon": ds.lon, "lat": ds.lat})
                mask.to_netcdf(mask_file)
                logger.info(f"Mask file has been written to {mask_file}")

    def cut_to_filled_area(self, buffer=0):
        """Create lat and lon slices to cut the data to the filled area."""
        # Find the non-zero elements
        cols = np.any(
            self.catchment_mask, axis=0
        )  # Boolean array for columns with any filled cells
        rows = np.any(
            self.catchment_mask, axis=1
        )  # Boolean array for rows with any filled cells

        # Get the indices of the non-zero rows and columns
        min_row, max_row = np.where(rows)[0][[0, -1]]
        min_col, max_col = np.where(cols)[0][[0, -1]]
        # Add a buffer of one cell
        min_row = min_row - buffer if min_row > 0 else min_row
        min_col = min_col - buffer if min_col > 0 else min_col
        max_row = (
            max_row + buffer if max_row < self.catchment_mask.shape[0] else max_row
        )
        max_col = (
            max_col + buffer if max_col < self.catchment_mask.shape[1] else max_col
        )

        # Slice the array to extract the filled part
        lon_min, lon_max = np.round(self.ds.lon.values[min_col], 3), np.round(
            self.ds.lon.values[max_col], 3
        )
        lat_min, lat_max = np.round(self.ds.lat.values[max_row], 3), np.round(
            self.ds.lat.values[min_row], 3
        )
        lat_slice = slice(lat_max, lat_min)
        lon_slice = slice(lon_min, lon_max)
        logger.info(f"lat_slice: {lat_slice}, lon_slice: {lon_slice}")
        return lat_slice, lon_slice


def merge_catchment(path1, path2, out_path):
    """Merge the rolled and non-rolled file."""
    # read the rolled and non-rolled files
    ds1 = xr.open_dataset(path1, engine="netcdf4")
    ds2 = xr.open_dataset(path2, engine="netcdf4")

    # select all the basins in the border area
    mask_ids = np.unique(
        ds1["basin"].where(
            (ds1.lon > CUTOFF_THRESHOLD) | (ds1.lon < (CUTOFF_THRESHOLD * -1))
        )
    )
    # get a mask of all the border area basins
    mask = ds1["basin"].isin(mask_ids)
    # modify the ids to avoid overlaps
    ds2["basin"] = ds2["basin"] + 200000

    # in the border area, use the rolled data, else the original
    merged = xr.where(mask, ds2.reindex_like(ds1, method="nearest"), ds1)
    merged.to_netcdf(out_path)


def get_transformation_matrix_nc(ds, var_name):
    """Get Transformation Matrix from input file dimensions and resolution."""
    da = ds[var_name]

    # Get attributes for geotransformation
    lat = da.coords["lat"].values  # Assuming 'lat' and 'lon' are dimensions
    lon = da.coords["lon"].values
    logger.info(f"lat: {lat[0]} | {lat[-1]}")
    logger.info(f"lon: {lon[0]} | {lon[-1]}")

    # Assuming uniform spacing, calculate resolution
    lat_res = abs(lat[1] - lat[0]) if len(lat) > 1 else 0.0
    lon_res = abs(lon[1] - lon[0]) if len(lon) > 1 else 0.0
    lon_res, lat_res = np.round(lon_res, decimals=5), np.round(lat_res, decimals=5)
    logger.info(f"lat_res {lat_res}; lon_res {lon_res}")

    # Get the corner coordinate of the dataset
    x_min, y_max = np.round(lon.min(), decimals=5), np.round(lat.max(), decimals=5)
    return (
        np.float64(lon_res),
        np.float64(0.0),
        np.float64(x_min - lon_res / 2),
        np.float64(0.0),
        np.float64(-lat_res),
        np.float64(y_max + lat_res / 2),
    )


@log_arguments()
def create_catchment(
    input_file,
    output_path,
    var_name,
    var,
    ftype,
    gauge_coords=None,
    coordinate_slices=None,
    mask_file=None,
    target_resolution=None,
    frame=1,
):
    """Create file containing catchment ids, flowdirection and upstream area from dem or flow direction."""
    logger.info(
        f"Creating catchment file for {var_name} using {var} and {ftype} from {input_file}"
    )

    if var not in {"fdir", "dem"}:
        with ErrorLogger(logger):
            msg = f"Unexpected value for var={var}, must be 'fdir' or 'dem'"
            raise ValueError(msg)
    ds = xr.open_dataset(pl.Path(input_file))

    # transform
    transform = get_transformation_matrix_nc(ds, var_name)

    logger.info(transform)
    latlon = True

    if gauge_coords is None and coordinate_slices is None:
        temp_file1 = "hydro1.nc"
        global_catchments = Catchment(
            ds=ds,
            var_name=var_name,
            var=var,
            ftype=ftype,
            transform=transform,
            latlon=latlon,
            out_var_name=temp_file1,
            do_shift=False,
        )
        # create a shifted version of the catchment to avoid border effects
        temp_file2 = "hydro2.nc"
        global_catchments_shifted = Catchment(
            ds=ds,
            var_name=var_name,
            var=var,
            ftype=ftype,
            transform=transform,
            latlon=latlon,
            out_var_name=temp_file2,
            do_shift=True,
            target_resolution=target_resolution,
        )
        catchments = [global_catchments, global_catchments_shifted]

        for c in catchments:
            c.get_basins()
            c.get_facc()
            c.get_grid_area()
            # c.get_upstream_area()
            c.write(output_path, single_file=True, frame=frame)
        # add paths to the temp files
        temp_file1 = pl.Path(output_path, "hydro1.nc")
        temp_file2 = pl.Path(output_path, "hydro2.nc")
        logger.info("Merging catchment files")
        merge_catchment(
            temp_file1,
            temp_file2,
            pl.Path(output_path, "basin_ids.nc"),
        )
        # remove the temporary files
        temp_file1.unlink()
        temp_file2.unlink()
    elif coordinate_slices is not None:
        ds = ds.sel(lat=coordinate_slices["lat"], lon=coordinate_slices["lon"])
        logger.info(transform)
        c = Catchment(
            ds=ds,
            var_name=var_name,
            var=var,
            ftype=ftype,
            transform=transform,
            latlon=latlon,
            out_var_name="basin_ids.nc",
            do_shift=False,
            target_resolution=target_resolution,
        )
        c.get_basins()
        c.get_facc()
        c.get_grid_area()
        c.write(output_path, single_file=True, mask_file=mask_file, frame=frame)
    else:
        logger.info(f"Creating catchment for gauge coordinates {gauge_coords}")
        c = Catchment(
            ds=ds,
            var_name=var_name,
            var=var,
            ftype=ftype,
            transform=transform,
            latlon=latlon,
            out_var_name="basin_ids.nc",
            do_shift=False,
            target_resolution=target_resolution,
        )
        c.delineate_basin(gauge_coords)
        c.get_facc()
        c.get_grid_area()
        c.write(
            output_path,
            single_file=True,
            cut_by_basin=True,
            mask_file=mask_file,
            frame=frame,
            buffer=frame,
        )
