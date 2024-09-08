"""
Create the catchment file for mRM

Authors
-------
- Robert Schweppe
- Matthias Kelbling
- Jeisson Leal
"""

import pathlib as pl

import numpy as np
import pyflwdir
import xarray as xr
from mhm_tools.common.logger import logger, set_log_level

# GLOBAL VARIABLES
FDIR_FILLVALUE = {"d8": 247, "ldd": 255}
FACC_FILLVALUE = 0
FILLVALUE = -9999
# use d8 for basinex, ldd for mRM version in Ulysses
OUTPUT_FTYPE = "ldd"
CUTOFF_THRESHOLD = 170
# FUNCTIONS


# CLASSES
class Catchment:
    VARIABLES = {
        "flwdir": {
            "title": f"flow direction ({OUTPUT_FTYPE.upper()})",
            "_FillValue": FDIR_FILLVALUE[OUTPUT_FTYPE],
            "units": "-",
        },
        "basin": {
            "title": "basin Id",
            "_FillValue": 0,
            "units": "-",
        },
        "upgrid": {
            "title": "accumulated data values along the flow directions",
            "_FillValue": FACC_FILLVALUE,
            "units": "-",
        },
        "uparea_grid": {
            "title": "rectangular drainage area",
            "_FillValue": FILLVALUE,
            "units": "km2",
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
        self.catchment_mask=None
        self.out_var_name = out_var_name if out_var_name is not None else f"{var_name}.nc"
        if type(self.out_var_name) is not str:
            self.out_var_name = f"{var_name}.nc"
        self.do_shift = do_shift
        self.ds = ds

        data = self._modify_data(self.ds[var_name])

        if self.do_shift and self.is_data_global:
            transform = list(transform)
            transform[2] = 0
            transform = tuple(transform)

        if var == "fdir":
            self.add_fdir(data=data.data, ftype=ftype, **kwargs)
        elif var == "dem":
            self.add_dem(data=data, **kwargs)
        else:
            raise NotImplementedError

    @property
    def is_data_global(self):
        return (
            "lon" in self.ds.coords
            and self.ds.lon.min() < (CUTOFF_THRESHOLD * -1)
            and self.ds.lon.max() > CUTOFF_THRESHOLD
        )

    def _modify_data(self, data):
        # correct circumspanning data
        if self.do_shift and self.is_data_global:
            tmp = data.roll(lon=int(len(self.ds.lon) / 2), roll_coords=True)
            return tmp
        return data

    def _revert_data(self, data):
        # correct circumspanning data
        if self.do_shift and self.is_data_global:
            return np.roll(data, int(len(self.ds.lon) / 2), axis=1)
        return data

    def add_dem(self, data, **kwargs):
        """
        Inits the FlwdirRaster class from dem.
        """
        # perform checks
        self.elevtn = data.data
        if self._fdir is None:
            # Create a flow direction object
            logger.debug("add_dem: kwargs: ", kwargs)
            self._fdir = pyflwdir.from_dem(data=self.elevtn, nodata=np.nan, transform=(0.05, 0.0, -180, 0, 0.05, -90), latlon=True)
            self.get_fdir()

    def add_fdir(self, data, ftype, **kwargs):
        """
        Inits the FlwdirRaster class from fdir.
        """
        # perform check
        if self._fdir is None:
            mask = np.isnan(data)
            if mask.any():
                data[mask] = FDIR_FILLVALUE[ftype]
            data = data.astype(np.uint8)
            self._fdir = pyflwdir.from_array(data=data, ftype=ftype, **kwargs)
        self.get_fdir()

    def delineate_basin(self, gauge_coords):
        """
        Deliniate the basin for a given lat and lon
        """
        self.basin = self._fdir.basins(xy=gauge_coords, streams=self._fdir.stream_order() >= 4)
        self.catchment_mask = self.basin > 0
        if not np.any(np.isnan(self.basin)):
            self.basin[np.where(~self.catchment_mask)] = self.VARIABLES["basin"]["_FillValue"]
        


    def get_basins(self):
        """
        Performs the calculation of the catchment ids
        """
        self.basin = self._fdir.basins()

    def get_fdir(self, ftype=None):
        """
        Performs the calculation of the flow direction
        """
        self.flwdir = self._fdir.to_array(ftype=ftype or OUTPUT_FTYPE)

    def get_upstream_area(self):
        """
        Performs the calculation of the upstream catchment area
        """
        self.uparea_grid = self._fdir.upstream_area(unit="km2").astype(int)

    def get_grid_area(self):
        """
        Performs the calculation of the catchment area
        """
        self.grdare = self._fdir.area.astype(int)

    def get_facc(self):
        data = np.ones_like(self.flwdir).astype(np.uint32)
        data[~self._fdir.mask.reshape(data.shape)] = 0
        self.upgrid = self._fdir.accuflux(data, nodata=0)

    def write(self, out_path, single_file=True, format="nc", cellsize=None, cut_by_basin=False):
        data_vars = {}
        out_path = pl.Path(out_path)
        data = getattr(self, 'basin')
        if not out_path.is_dir():
            out_path.mkdir(parents=True, exist_ok=True)
        if cut_by_basin:
            lat_slice, lon_slice = self.cut_to_filled_area()
        else:
            lat_slice, lon_slice = slice(-56,84), slice(None)

        for var_name in self.VARIABLES.keys():
            data = getattr(self, var_name)
            if cut_by_basin:
                data[~self.catchment_mask] = self.VARIABLES[var_name]["_FillValue"]
            if data is None:
                continue
            data_var = xr.Dataset(
                {var_name: (["lat", "lon"], self._revert_data(data))},
                coords={
                    "lon": self.ds.lon,  # [slice(3555, 3565)],
                    "lat": self.ds.lat,  # [slice(860, 870)],
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
                    with open(fname, "w") as file_object:
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
                    raise Exception(
                        f'Format "{format}" unknown, use one of ["nc", "asc"]'
                    )
        if single_file:
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

            ds.sel(lat=lat_slice, lon=lon_slice).to_netcdf(
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
    def cut_to_filled_area(self):
        import matplotlib.pyplot as plt
         # Find the non-zero elements
        cols = np.any(self.catchment_mask, axis=0)  # Boolean array for columns with any filled cells
        rows = np.any(self.catchment_mask, axis=1)  # Boolean array for rows with any filled cells

        # Get the indices of the non-zero rows and columns
        min_row, max_row = np.where(rows)[0][[0, -1]]
        min_col, max_col = np.where(cols)[0][[0, -1]]
            # Slice the array to extract the filled part
        logger.debug(f"min_row: {min_row}, max_row: {max_row}, min_col: {min_col}, max_col: {max_col}")
        logger.debug(f"shape of catchment_mask: {self.catchment_mask.shape}")
        logger.debug(f"shape of lat and lon: {self.ds.lat.shape}, {self.ds.lon.shape}")
        lat_slice = slice(self.ds.lat[max_row], self.ds.lat[min_row])
        lon_slice = slice(self.ds.lon[min_col], self.ds.lon[max_col])
        logger.info(f"lat_slice: {lat_slice}, lon_slice: {lon_slice}")
        return lat_slice, lon_slice

# use this code to merge the rolled and non-rolled file
def merge_catchment(path1, path2, out_path):
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



def create_catchment(input_file, output_path, var_name, var, ftype, gauge_coords=None):

    set_log_level("DEBUG")
    logger.info(f"Creating catchment file for {var_name} using {var} and {ftype} from {input_file}")

    if var not in {"fdir", "dem"}:
        raise ValueError(f"Unexpected value for var={var}, must be 'fdir' or 'dem'")

    ds = xr.open_dataset(pl.Path(input_file))
    transform = (0.05, 0.0, -180, 0, 0.05, -90)
    latlon = True

    if gauge_coords is None:
        global_catchments = Catchment(ds=ds, var_name=var_name, var=var, ftype=ftype, transform=transform, latlon=latlon, out_var_name="hydro1.nc", do_shift=False)
        # create a shifted version of the catchment to avoid border effects
        global_catchments_shifted = Catchment(ds=ds, var_name=var_name, var=var, ftype=ftype, transform=transform, latlon=latlon, out_var_name="hydro2.nc", do_shift=True)
        catchments = [
            global_catchments,
            global_catchments_shifted
        ]

        for c in catchments:
            c.get_basins()
            c.get_facc()
            c.get_grid_area()
            c.get_upstream_area()
            c.write(output_path, single_file=True)

        logger.info(f"Merging catchment files")
        merge_catchment(
            pl.Path(output_path, "hydro1.nc"),
            pl.Path(output_path, "hydro2.nc"),
            pl.Path(output_path, "hydro_merged.nc"),
        )
    else:
        logger.info(f"Creating catchment for gauge coordinates {gauge_coords}")
        c = Catchment(ds=ds, var_name=var_name, var=var, ftype=ftype, transform=transform, latlon=latlon, out_var_name="hydro.nc", do_shift=False)
        c.delineate_basin(gauge_coords)
        c.get_facc()
        c.get_grid_area()
        c.get_upstream_area()
        c.write(output_path, single_file=True, cut_by_basin=True)
