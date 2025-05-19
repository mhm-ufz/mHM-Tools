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

from mhm_tools.common.file_handler import get_xarray_ds_from_file
from mhm_tools.common.logger import ErrorLogger, log_arguments

logger = logging.getLogger(__name__)


# GLOBAL VARIABLES
FDIR_FILLVALUE = {"d8": 247, "ldd": 255}
FDIR_SINKVALUE = {"d8": 0, "ldd": 5}
FACC_FILLVALUE = 0
FILLVALUE = -9999
# use d8 for basinex, ldd for mRM version in Ulysses
OUTPUT_FTYPE = "ldd"
CUTOFF_THRESHOLD = 175
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
        l1_resolution=None,
        upscale=False,
        latlon=True,
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
        self.l1_resolution = l1_resolution
        self.do_upscale = upscale
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
        logger.debug(f"self.ds: {self.ds}")
        self.transform = transform

        data = self._modify_data(self.ds[var_name])

        if self.do_shift:
            transform = list(self.transform)
            transform[2] = 0
            self.transform = tuple(transform)

        self.input_da = data

        self.input_da = data

        if var == "fdir":
            if "nodata_value" in self.input_da.attrs:
                old_no_data_val = self.input_da.attrs["nodata_value"]
            elif "_FillValue" in self.input_da.attrs:
                old_no_data_val = self.input_da.attrs["_FillValue"]
            elif "missing_value" in self.input_da.attrs:
                old_no_data_val = self.input_da.attrs["missing_value"]
            else:
                old_no_data_val = np.nan
            self.input_da.attrs["_FillValue"] = FDIR_FILLVALUE[ftype]
            self.input_da.attrs["nodata_value"] = FDIR_FILLVALUE[ftype]
            self.input_da = self.input_da.where(
                (ds[var_name] != old_no_data_val) & ~np.isnan(ds[var_name]),
                FDIR_FILLVALUE[ftype],
            )
            logger.debug(self.input_da)
            self.add_fdir(latlon=latlon)
        elif var == "dem":
            self.add_dem(latlon=latlon)
        else:
            with ErrorLogger(logger):
                raise NotImplementedError

    def _modify_data(self, data):
        # correct circumspanning data
        if self.do_shift:
            return data.roll(lon=int(len(self.ds.lon) / 2), roll_coords=True)
        return data

    def _revert_data(self, data):
        # correct circumspanning data
        if self.do_shift:
            return np.roll(data, int(len(self.ds.lon) / 2), axis=1)
        return data

    def add_dem(self, latlon):
        """Init the FlwdirRaster class from dem."""
        # perform checks
        # self.input_ds = fill_nan_with_neighbors(self.input_ds)
        self.elevtn = self.input_da.data
        if self._fdir is None:
            # Create a flow direction object
            logger.info("add_dem")
            self._fdir = pyflwdir.from_dem(
                data=self.elevtn,
                nodata=np.nan,
                transform=self.transform,
                latlon=latlon,
            )
            self.get_fdir()

    def add_fdir(self, latlon):
        """Init the FlwdirRaster class from fdir."""
        # perform check
        data = self.input_da.data
        if self._fdir is None:
            data = data.astype(np.uint8)
            self._fdir = pyflwdir.from_array(
                data=data, ftype=self.ftype, transform=self.transform, latlon=latlon
            )
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

    def get_upscaling_factor(self):
        """Create upscaling factor."""
        input_res = round(abs(self.ds.lon.data[1] - self.ds.lon.data[0]), 6)
        if (
            int(self.l1_resolution / input_res + 0.5) - (self.l1_resolution / input_res)
            < 1e6
        ):
            return int(self.l1_resolution / input_res + 0.5)
        not_int_multiple_msg = f"Upscaling only works if L1 resolution is integer muplipe of L0 resolution but L1 = {self.l1_resolution / input_res:.4f} * L0"
        raise ValueError(not_int_multiple_msg)

    def upscale(self, var):
        """Upscale flow direction to l1_resolution if that is int multipe of data resolution."""
        factor = self.get_upscaling_factor()

        if factor == 1:
            self.get_facc()
            return
        # if we upscale the do_upscale flag should be true
        self.do_upscale = True
        logger.info(
            f"Upscaling flow direction to {self.l1_resolution} with the fator {factor}."
        )
        fdir_upscaled, upscaling_indices = self._fdir.upscale(factor, method="ihu")

        subareas = self._fdir.ucat_area(idxs_out=upscaling_indices, unit="km2")[1]
        uparea1 = fdir_upscaled.accuflux(subareas)

        flwerr = self._fdir.upscale_error(fdir_upscaled, upscaling_indices)
        percentage_error = np.sum(flwerr == 0) / np.sum(flwerr != 255) * 100
        logger.info(f"upscaling error in {percentage_error:.2f}% of cells")
        logger.debug(f"Upscaled form {self._fdir.shape} to {fdir_upscaled.shape}")
        self._fdir = fdir_upscaled
        self.get_fdir()
        self.uparea_grid = uparea1  # replaces self.get_facc

        if var == "dem":
            lat_size, lon_size = self.input_da.shape
            # Ensure the dimensions are evenly divisible by the factor
            if lat_size % factor != 0 or lon_size % factor != 0:
                msg = f"Data dimensions must be divisible by the upscaling factor of {factor}. Lat ({lat_size}/{factor})={lat_size/factor:.2f}; Lon ({lon_size}/{factor})={lon_size/factor:.2f}"
                with ErrorLogger(logger):
                    raise ValueError(msg)

            # Reshape and aggregate data
            reshaped = self.input_da.values.reshape(
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
        logger.debug("Get flwdir as array.")
        self.flwdir = self._fdir.to_array(ftype=self.ftype or OUTPUT_FTYPE)

    def get_upstream_area(self):
        """Perform the calculation of the upstream catchment area."""
        self.upgrid = self._fdir.upstream_area(unit="km2").astype(int)

    def get_grid_area(self):
        """Perform the calculation of the catchment area."""
        self.grdare = self._fdir.area.astype(int)

    def get_facc(self):
        """Get the flow accumulation area."""
        logger.info("Calculate flow accumulation...")
        data = np.ones_like(self.flwdir).astype(np.uint32)
        data[~self._fdir.mask.reshape(data.shape)] = 0
        self.uparea_grid = self._fdir.accuflux(data, nodata=0)

    @staticmethod
    def create_frame(ds, frame=0, frame_value=0):
        """If a frame is used this frame is set to no data values as a frame."""
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

    def fill_adjacent_missing_with_sink(self, da, fill_value, sink_value):
        """
        Replace all missing values adjacent to non-missing values with 0 in an xarray Dataset.

        Parameters
        ----------
            da (xr.Dataset): Input dataset.

        Returns
        -------
            xr.Dataset: Dataset with adjacent missing values replaced with 0.
        """
        # Mask of missing values
        missing_mask = da == fill_value

        # Mask of non-missing values
        non_missing_mask = ~missing_mask

        # Dilate the non-missing mask to include adjacent cells
        adjacent_mask = binary_dilation(
            non_missing_mask, structure=np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]])
        )

        # Identify adjacent missing values
        adjacent_missing = adjacent_mask & missing_mask

        # Replace adjacent missing values with 0
        return xr.where(adjacent_missing, sink_value, da)

    @log_arguments()
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
            data_var = self.processing_data_variable(
                var_name, cut_by_basin, lat_slice, lon_slice
            )
            if data_var is None:
                continue
            if single_file:
                data_vars[var_name] = data_var
            else:
                self.write_single_variable_file(
                    data_var, var_name, out_path, cellsize, format
                )
        if single_file:
            ds = self.write_basin_id_file(data_vars, frame, out_path)
            # use basin_id to create a mask file
            self.write_mask_file(ds, mask_file)

    def write_single_variable_file(
        self, data_var, var_name, out_path, cellsize, format
    ):
        """Write a single data variable to a specified file path."""
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
                        "dtype": data_var[var_name].dtype,
                        "_FillValue": self.VARIABLES[var_name]["_FillValue"],
                    }
                },
            )
        elif format == "asc":
            cellsize = cellsize or abs(float(data_var["lon"][1] - data_var["lon"][0]))
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

    def processing_data_variable(self, var_name, cut_by_basin, lat_slice, lon_slice):
        """Process data variable, masking it and croping it spatial dimensions."""
        logger.info(f"Processing {var_name}")
        data = getattr(self, var_name)
        if data is None:
            logger.warning(f"No data for {var_name}")
            return None
        if cut_by_basin:
            data[~self.catchment_mask] = self.VARIABLES[var_name]["_FillValue"]
        if data is None:
            logger.warning(f"No data for {var_name}")
            return None
        lon = self.ds.lon.data
        lat = self.ds.lat.data
        if self.l1_resolution is not None:
            input_res = round(abs(lon[1] - lon[0]), 9)
            if input_res != self.l1_resolution and self.do_upscale:
                logger.debug(
                    f"Creating lon and lat arrays from l1_resolution {self.l1_resolution}"
                )
                lon = np.arange(
                    lon.min() - input_res / 2 + self.l1_resolution / 2,
                    lon.max() + self.l1_resolution / 2,
                    self.l1_resolution,
                )
                lat = np.arange(
                    lat.max() + input_res / 2 - self.l1_resolution / 2,
                    lat.min() - self.l1_resolution / 2,
                    -self.l1_resolution,
                )
        logger.debug(
            f"lon_min {np.min(lon):.3f}, lon_max {np.max(lon):.3f}, resulution: {self.l1_resolution}"
        )
        logger.debug(f"{var_name} - mean {np.nanmean(data)}, max {np.nanmax(data)}")
        logger.debug(f"Shape {data.shape},  lon {len(lon)}, lat {len(lat)}")
        data_var = xr.Dataset(
            {var_name: (["lat", "lon"], self._revert_data(data))},
            coords={
                "lon": lon,  # [slice(3555, 3565)],
                "lat": lat,  # [slice(860, 870)],
            },
        )
        logger.info(f"Cutting {var_name} data to correct spatial dimensions")
        data_var = data_var.sel(lat=lat_slice, lon=lon_slice)
        logger.debug(data_var)
        return data_var

    def write_basin_id_file(self, data_vars, frame, out_path):
        """Write the basin_id file to specified path and set a sink value frame if specified."""
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

        # logger.debug(f"lat_slice: {lat_slice}, lon_slice: {lon_slice}")
        logger.debug(f"ds: {ds}")
        ds = self.create_frame(ds, frame, FDIR_SINKVALUE[self.ftype])
        # For the flow dir map fill masked cells adjecent to filled cells with sink instead of missing value
        # fdir_filled = self.fill_adjacent_missing_with_sink(
        #     ds["flwdir"], FDIR_FILLVALUE[self.ftype], FDIR_SINKVALUE[self.ftype]
        # )
        # ds["flwdir"].data[:] = fdir_filled.data[:]
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
        return ds

    def write_mask_file(self, ds, mask_file):
        """Write basin mask to specified path."""
        if mask_file is not None:
            # name the variable mask
            mask = ds.basin > 0
            mask_file = pl.Path(mask_file)
            mask_da = xr.DataArray(
                mask, coords={"lat": ds.lat, "lon": ds.lon}, dims=["lat", "lon"]
            )
            mask_ds = xr.Dataset(
                {"land_mask": mask_da, "mask": mask_da},
                coords={"lon": ds.lon, "lat": ds.lat},
            )
            mask_ds.to_netcdf(mask_file)
            logger.info(f"Mask file has been written to {mask_file}")
        else:
            logger.info("No mask file path specified.")

    def cut_to_filled_area(self, buffer=0):
        """Create lat and lon slices to cut the data to the filled area."""
        logger.info("Cutting to filled area.")
        # Find the non-zero elements
        cols = np.any(
            self.catchment_mask, axis=0
        )  # Boolean array for columns with any filled cells
        rows = np.any(
            self.catchment_mask, axis=1
        )  # Boolean array for rows with any filled cells

        logger.info(
            f"shape {np.shape(self.catchment_mask)}  cols: {len(cols)}, rows: {len(rows)}"
        )
        logger.info(f"lon {len(self.ds.lon.values)}  lat: {len(self.ds.lat.values)}")

        # Get the indices of the non-zero rows and columns
        min_row, max_row = np.where(rows)[0][[0, -1]]
        min_col, max_col = np.where(cols)[0][[0, -1]]

        if buffer > 0:
            # Add a buffer of one cell
            logger.info(f"Using a min buffer of {buffer}")
            min_row = min_row - buffer if min_row > 0 else min_row
            min_col = min_col - buffer if min_col > 0 else min_col
            max_row = (
                max_row + buffer if max_row < self.catchment_mask.shape[0] else max_row
            )
            max_col = (
                max_col + buffer if max_col < self.catchment_mask.shape[1] else max_col
            )
        logger.info(
            f"min row: {min_row} max row: {max_row} min_col: {min_col}, max_col: {max_col}"
        )
        if self.l1_resolution is not None:
            factor = self.get_upscaling_factor()
            if factor != 1:
                logger.info(
                    f"Regridding to fit coarse grid with res {self.l1_resolution} (factor {factor})"
                )
                min_row = min_row // factor * factor
                min_col = min_col // factor * factor
                # Calculating max_row/col it needs:
                #  +1 to include the whole last coarse grid cell -1 to not get one cell from the next block
                max_row = (max_row // factor + 1) * factor - 1
                max_col = (max_col // factor + 1) * factor - 1
                if max_col >= len(cols):
                    logger.warning("While regridding max_cols was larger than col-size")
                    max_col = len(cols) - 1
                if max_row >= len(rows):
                    logger.warning("While regridding max_rows was larger than row-size")
                    max_row = len(rows) - 1
        logger.info(
            f"min row: {min_row} max row: {max_row} min_col: {min_col}, max_col: {max_col}"
        )

        # Slice the array to extract the filled part
        lon_min, lon_max = (
            np.round(self.ds.lon.values[min_col], 8),
            np.round(self.ds.lon.values[max_col], 8),
        )
        lat_min, lat_max = (
            np.round(self.ds.lat.values[max_row], 8),
            np.round(self.ds.lat.values[min_row], 8),
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
            (ds1.lon.max() > CUTOFF_THRESHOLD)
            | (ds1.lon.min() < (CUTOFF_THRESHOLD * -1))
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
    logger.info(f"lat: {lat.max()} | {lat.min()}")
    logger.info(f"lon: {lon.min()} | {lon.max()}")

    # Assuming uniform spacing, calculate resolution
    lat_res = abs(lat[1] - lat[0]) if len(lat) > 1 else 0.0
    lon_res = abs(lon[1] - lon[0]) if len(lon) > 1 else 0.0
    # logger.info(f"lat_res {lat_res}; lon_res {lon_res}")

    # Get the corner coordinate of the dataset
    x_min, y_max = lon.min(), lat.max()
    return (
        np.float64(lon_res),
        np.float64(0.0),
        np.float64(x_min - lon_res / 2),
        np.float64(0.0),
        np.float64(-lat_res),
        np.float64(y_max + lat_res / 2),
    )


def is_data_global(ds, coordinate_slice):
    """Check if the longitude data is global."""
    if coordinate_slice is not None:
        ds_sliced = ds.sel(lon=coordinate_slice["lon"])
    else:
        ds_sliced = ds
    try:
        return (
            "lon" in ds_sliced.coords
            and ds_sliced.lon.min() < (CUTOFF_THRESHOLD * -1)
            and ds_sliced.lon.max() > CUTOFF_THRESHOLD
        )
    except Exception as e:
        logger.warning(e)
        return False


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
    l1_resolution=None,
    frame=1,
    upscale=False,
    latlon=True,
):
    """Create file containing catchment ids, flowdirection and upstream area from dem or flow direction."""
    logger.info(
        f"Creating catchment file for {var_name} using {var} and {ftype} from {input_file}"
    )

    if var not in {"fdir", "dem"}:
        with ErrorLogger(logger):
            msg = f"Unexpected value for var={var}, must be 'fdir' or 'dem'"
            raise ValueError(msg)

    with get_xarray_ds_from_file(input_file, var_name) as input_ds:
        # transform
        transform = get_transformation_matrix_nc(input_ds, var_name)

        logger.info(transform)

        if gauge_coords is None and is_data_global(input_ds, coordinate_slices):
            logger.info("Creating global basin id file...")
            temp_file1 = "hydro1.nc"
            global_catchments = Catchment(
                ds=input_ds,
                var_name=var_name,
                var=var,
                ftype=ftype,
                transform=transform,
                latlon=latlon,
                out_var_name=temp_file1,
                do_shift=False,
                l1_resolution=l1_resolution,
                upscale=upscale,
            )
            # create a shifted version of the catchment to avoid border effects
            temp_file2 = "hydro2.nc"
            global_catchments_shifted = Catchment(
                ds=input_ds,
                var_name=var_name,
                var=var,
                ftype=ftype,
                transform=transform,
                latlon=latlon,
                out_var_name=temp_file2,
                do_shift=True,
                l1_resolution=l1_resolution,
                upscale=upscale,
            )
            catchments = [global_catchments, global_catchments_shifted]

            for c in catchments:
                if l1_resolution is not None and upscale:
                    c.upscale(var)
                else:
                    c.get_facc()
                c.get_basins()
                c.get_grid_area()
                # c.get_upstream_area()
                c.write(output_path, single_file=True, frame=frame, mask_file=mask_file)
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
            logger.info(f"Creating basin id file for {coordinate_slices}")
            lat_max = coordinate_slices["lat"].start
            lat_min = coordinate_slices["lat"].stop
            lon_min = coordinate_slices["lon"].start
            lon_max = coordinate_slices["lon"].stop
            input_ds_sliced = input_ds.sel(
                lat=slice(lat_max, lat_min), lon=slice(lon_min, lon_max)
            )
            c = Catchment(
                ds=input_ds_sliced,
                var_name=var_name,
                var=var,
                ftype=ftype,
                transform=transform,
                latlon=latlon,
                out_var_name="basin_ids.nc",
                do_shift=False,
                l1_resolution=l1_resolution,
                upscale=upscale,
            )
            if l1_resolution is not None and upscale:
                c.upscale(var)
            else:
                c.get_facc()
            c.get_basins()
            c.get_grid_area()
            c.write(output_path, single_file=True, mask_file=mask_file, frame=frame)
        else:
            logger.info(f"Creating catchment for gauge coordinates {gauge_coords}")
            c = Catchment(
                ds=input_ds,
                var_name=var_name,
                var=var,
                ftype=ftype,
                transform=transform,
                latlon=latlon,
                out_var_name="basin_ids.nc",
                do_shift=False,
                l1_resolution=l1_resolution,
                upscale=upscale,
            )
            c.delineate_basin(gauge_coords)
            if l1_resolution is not None and upscale:
                c.upscale(var)
            else:
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
