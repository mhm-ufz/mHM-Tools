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
from mhm_tools.common.logger import logger

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
            self._fdir = pyflwdir.from_dem(data=self.elevtn, nodata=np.nan, **kwargs)
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

    def delineate_basin(self, lat, lon):
        """
        Deliniate the basin for a given lat and lon
        """
        idx = (lon + 180) / 0.05
        idy = (lat + 90) / 0.05
        id = idy * 7200 + idx
        self.basin = self._fdir.basins(idx=int(id))

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

    def write(self, out_path, single_file=True, format="nc", cellsize=None):
        data_vars = {}
        out_path = pl.Path(out_path)
        if not out_path.is_dir():
            out_path.mkdir(parents=True, exist_ok=True)
        for var_name in self.VARIABLES.keys():
            data = getattr(self, var_name)
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

            ds.sel(lat=slice(84, -56)).to_netcdf(
                out_path / self.out_var_name,
                encoding={
                    var_name: {
                        "dtype": ds[var_name].dtype,
                        "_FillValue": self.VARIABLES[var_name]["_FillValue"],
                    }
                    for var_name in ds.data_vars
                },
            )
    def cut_to_filled_area(self):
        import matplotlib.pyplot as plt
        mask = self.basin
         # Find the non-zero elements
        rows = np.any(mask, axis=1)  # Boolean array for rows with any filled cells
        cols = np.any(mask, axis=0)  # Boolean array for columns with any filled cells

        # Get the indices of the non-zero rows and columns
        min_row, max_row = np.where(rows)[0][[0, -1]]
        min_col, max_col = np.where(cols)[0][[0, -1]]
        logger.info(f"min_row: {min_row}, max_row: {max_row}, min_col: {min_col}, max_col: {max_col}")
        for var_name in self.VARIABLES.keys():
            data = self.VARIABLES[var_name].values
            # Slice the array to extract the filled part
            logger.info(f"Cutting {var_name} to filled area")
            print(data)
            try:
                logger.info(f"Shape of data: {data.shape}")
            except:
                data = data.data
                logger.info(f"Shape of data: {data.shape}")
            filled_part = data[min_row:max_row+1, min_col:max_col+1]
            plt.imshow(filled_part)
            plt.savefig(f"/work/luedke/{var_name}.png")
            plt.close()
            self.VARIABLES[var_name] = filled_part



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

    logger.info(f"Creating catchment file for {var_name} using {var} and {ftype} from {input_file}")

    if var not in {"fdir", "dem"}:
        raise ValueError(f"Unexpected value for var={var}, must be 'fdir' or 'dem'")

    ds = xr.open_dataset(pl.Path(input_file))
    transform = (0.05, 0.0, -180, 0, 0.05, -90)
    latlon = True

    if gauge_coords is None:
        catchments = [
            Catchment(ds, var_name, var, ftype, transform, latlon, "hydro1.nc"),
            Catchment(ds, var_name, var, ftype, transform, latlon, "hydro2.nc", do_shift=True) # create a shifted version of the catchment to avoid border effects
        ]

        for c in catchments:
            c.get_basins()
            c.get_facc()
            c.get_grid_area()
            c.get_upstream_area()
            logger.info(f"Writing catchment file to {output_path}")
            c.write(output_path, single_file=True)

        logger.info(f"Merging catchment files")
        merge_catchment(
            pl.Path(output_path, "hydro1.nc"),
            pl.Path(output_path, "hydro2.nc"),
            pl.Path(output_path, "hydro_merged_03min.nc"),
        )
    else:
        c = Catchment(ds, var_name, var, ftype, transform, latlon, "hydro.nc")
        c.delineate_basin(*gauge_coords)
        c.get_facc()
        c.get_grid_area()
        c.get_upstream_area()
        # logger.info("Cutting to filled area")
        # c.cut_to_filled_area()
        logger.info(f"Writing catchment file to {output_path}")
        fdir = c.flwdir

    print(f"\nNetCDF basins file has been stored! \nSee {output_path}/hydro_merged_03min.nc\n")
