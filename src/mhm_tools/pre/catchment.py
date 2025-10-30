"""Create the catchment file for mRM.

Authors
-------
- Robert Schweppe
- Matthias Kelbling
- Jeisson Leal
- Simon Lüdke
"""

import logging
import pathlib as pl

from mhm_tools.common.netcdf import generate_bounds
from mhm_tools.pre.create_id_gauges import write_gauge_id
import numpy as np
import pyflwdir
import xarray as xr
from scipy.ndimage import binary_dilation

from mhm_tools.common.file_handler import get_xarray_ds_from_file, write_xarray_to_ascii, write_xarray_to_file
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


def create_cell_area(ds, lat_name="lat", lon_name="lon"):
    """Create a cell area data array in km2."""
    logger.info("Create cell area data array.")
    lat = ds[lat_name].data
    lon = ds[lon_name].data
    # calculate cellsize in kilometers
    R = 6371  # radius of the earth in kilometers
    lat_rad = np.deg2rad(lat)
    lon_rad = np.deg2rad(lon)
    dlat = np.abs(np.gradient(lat_rad))
    dlon = np.abs(np.gradient(lon_rad))
    # create 2D arrays for lat and lon
    dlat_2d, dlon_2d = np.meshgrid(dlat, dlon, indexing="ij")
    lat_2d = np.tile(lat_rad[:, np.newaxis], (1, len(lon)))
    # calculate area
    cell_areas = R**2 * dlat_2d * dlon_2d * np.cos(lat_2d)
    cell_area_da = xr.DataArray(
        cell_areas,
        coords={lat_name: lat, lon_name: lon},
        dims=[lat_name, lon_name],
        name="cell_area",
        attrs={
            "title": "cell area",
            "units": "km2",
            "creator": "Department of Computational Hydrosystems",
            "institution": "Helmholtz Centre for Environmental Research - UFZ",
        },
    )
    return cell_area_da


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
        l11_resolution=None,
        l2_resolution=None,
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
        self.gauge_lat = None
        self.gauge_lon = None
        self.ftype = ftype
        self.catchment_mask = None
        self.l1_resolution = l1_resolution
        self.l11_resolution = (
            l11_resolution if l11_resolution is not None else l1_resolution
        )
        self.l2_resolution = (
            l2_resolution if l2_resolution is not None else l1_resolution
        )
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
                "units": "-",
            },
            "upgrid": {
                "title": "upstream area",
                "_FillValue": FACC_FILLVALUE,
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

    def calc_upstream_area(self, cell_area=None):
        """Use pyflwdir to calculate the upstream area from flow direction by providing cell areas."""
        if self._fdir is None:
            logger.error("Flow direction is not initialized.")
            return None
        if cell_area is not None:
            return self._fdir.accuflux(cell_area, nodata=-9999)
        return self._fdir.accuflux(create_cell_area(self.ds).data, nodata=-9999)

    def _coord_to_index(self, lat, lon):
        """Map latitude/longitude or indices to integer grid indices."""
        if "lat" not in self.ds.coords or "lon" not in self.ds.coords:
            with ErrorLogger(logger):
                raise ValueError("Dataset is missing latitude/longitude coordinates.")
        lat_vals = self.ds.lat.data
        lon_vals = self.ds.lon.data

        if isinstance(lat, (int, np.integer)):
            i = int(lat)
            logger.debug(f"Was given latitude index {i} directly. Corresponding lat_value {lat_vals[i]}")
        else:
            if lat < min(lat_vals) or lat > max(lat_vals):
                logger.error(f"Given latitude {lat} is outside dataset bounds ({min(lat_vals)}, {max(lat_vals)}). Clipping to bounds.")
                i = None
            else:
                i = int(np.abs(lat_vals - float(lat)).argmin())
                logger.debug(f"Mapped latitude {float(lat)} to index {i} with lat_value {lat_vals[i]}")

        if isinstance(lon, (int, np.integer)):
            j = int(lon)
            logger.debug(f"Was given longitude index {j} directly. Corresponding lon_value {lon_vals[j]}")
        else:
            if lon < min(lon_vals) or lon > max(lon_vals):
                logger.error(f"Given longitude {lon} is outside dataset bounds ({min(lon_vals)}, {max(lon_vals)}). Clipping to bounds.")
                j = None
            else:
                j = int(np.abs(lon_vals - float(lon)).argmin())
                logger.debug(f"Mapped longitude {float(lon)} to index {j} with lon_value {lon_vals[j]}")
        if i is None or j is None:
            with ErrorLogger(logger):
                raise ValueError("Could not map given coordinates to valid indices within dataset bounds.")
        i = int(np.clip(i, 0, len(lat_vals) - 1))
        j = int(np.clip(j, 0, len(lon_vals) - 1))

        return i, j

    def get_best_coordinate(
        self,
        upstream_area,
        gauge_coords,
        ref_catchment_area,
        max_distance_cells=5,
        max_error=0.25,
        recursion=False,
    ):
        if not recursion:
            max_distance_cells = max_distance_cells // 2

        # Determine whether gauge_coords are lat/lon floats or array indices
        lat_vals = self.ds.lat.data
        lon_vals = self.ds.lon.data
        gi, gj = self._coord_to_index(gauge_coords[0], gauge_coords[1])

        logger.debug(f"Gauge index (row, col): {(gi, gj)}")

        # We will search for candidate outlet cells within a bbox around the gauge
        # (in degrees). These parameters are conservative defaults and can be
        # tuned later or exposed as args.
        max_cells = int(max(0, round(max_distance_cells)))

        # find index window (clamp to domain)
        i_min = max(0, gi - max_cells)
        i_max = min(len(lat_vals) - 1, gi + max_cells)
        j_min = max(0, gj - max_cells)
        j_max = min(len(lon_vals) - 1, gj + max_cells)

        # Ensure min <= max
        if i_min > i_max:
            i_min, i_max = i_max, i_min
        if j_min > j_max:
            j_min, j_max = j_max, j_min

        # Extract subgrid of upstream_area
        sub = upstream_area[i_min : i_max + 1, j_min : j_max + 1]

        # If subgrid is empty fallback to whole domain
        if sub.size == 0:
            logger.warning("Search bbox empty, falling back to full-domain search")
            sub = upstream_area
            i_min, j_min = 0, 0

        # Search for candidate cells whose upstream area matches ref_catchment_area
        size = float(ref_catchment_area)
        error = 0.0
        step = 0.01
        candidates = None
        while error <= max_error and (candidates is None or len(candidates[0]) == 0):
            low = size * (1.0 - error)
            high = size * (1.0 + error)
            candidates = np.where((sub >= low) & (sub <= high))
            if len(candidates[0]) == 0:
                error += step

        best_coord = None
        if len(candidates[0]) > 0:
            # convert sub indices to global indices
            cand_i = candidates[0] + i_min
            cand_j = candidates[1] + j_min
            # choose the candidate nearest to the gauge index
            d2 = (cand_i - gi) ** 2 + (cand_j - gj) ** 2
            k = int(np.argmin(d2))
            best_coord = (int(cand_i[k]), int(cand_j[k]))
            logger.info(
                f"Selected outlet candidate {best_coord} with upstream area {upstream_area[best_coord]} km2 (tolerance {error:.3f})"
            )
        else:
            logger.warning(
                "No candidate found within tolerance; falling back to nearest stream cell by upstream area magnitude."
            )
            if not recursion:
                return self.get_best_coordinate(
                    upstream_area,
                    gauge_coords,
                    ref_catchment_area,
                    max_distance_cells,
                    max_error,
                    recursion=True
                )
            # fallback: pick the cell in bbox with upstream area closest to target
            flat = np.abs(sub - size)
            idx = int(np.argmin(flat))
            ri, rj = np.unravel_index(idx, sub.shape)
            best_coord = (ri + i_min, rj + j_min)
            logger.info(f"The selected outlet candidate is {best_coord} with upstream area {upstream_area[best_coord]} km2 resulting in error {(ref_catchment_area - upstream_area[best_coord]) / ref_catchment_area:.3f}.")

        return best_coord, error

    def delineate_basin(
        self,
        gauge_coords,
        stream_order=4,
        ref_catchment_area=None,
        max_distance_cells=5,
        max_error=0.25,
    ):
        """Delineate the basin for a given lat and lon."""
        logger.info(f"Deliniating basin for gauge coordinates {gauge_coords}")
        gauge_coords = (gauge_coords[0], gauge_coords[1])
        # Target area in km2 we want to match (can be adjusted/replaced by caller later)

        # Compute upstream area (in km2) using accuflux and cell areas
        if ref_catchment_area is not None:
            cell_area = create_cell_area(self.ds).data
            upstream_area = None
            try:
                upstream_area = self.calc_upstream_area(cell_area=cell_area)
            except Exception:
                logger.exception("Failed to compute upstream area (accuflux).")

            if upstream_area is None:
                logger.error(
                    "Could not calculate upstream area. Flow direction may be uninitialized."
                )

            outlet_idx, error = self.get_best_coordinate(
                upstream_area,
                gauge_coords,
                ref_catchment_area,
                max_distance_cells,
                max_error,
            )
            new_lat = float(self.ds.lat.data[outlet_idx[0]])
            new_lon = float(self.ds.lon.data[outlet_idx[1]])
            self.gauge_lat = new_lat
            self.gauge_lon = new_lon
            logger.info(
                f"Moved outlet latitude {float(gauge_coords[0])} to {new_lat} and longitude {float(gauge_coords[1])} to {new_lon}."
            )

            river_mask = (upstream_area > ref_catchment_area * (1 - error - 1e-6)) & (
                upstream_area < ref_catchment_area * (1 + error + 1e-6)
            )
        else:
            logger.warning(
                "No catchment area provided; falling back to original gauge coordinates."
            )
            outlet_idx = self._coord_to_index(gauge_coords[0], gauge_coords[1])
            river_mask = None

        outlet_linear_idx = np.ravel_multi_index(outlet_idx, self._fdir.shape)

        # Now delineate basin using pyflwdir basins()
        streams_mask = self._fdir.stream_order() >= stream_order
        if river_mask is not None:
            streams_mask = np.logical_and(streams_mask, river_mask.astype(bool))
        try:
            self.basin = self._fdir.basins(
                idxs=np.array([outlet_linear_idx], dtype=np.int64),
                streams=streams_mask,
            )
        except Exception as e:
            logger.exception(f"pyflwdir.basins(idxs=...) failed for {outlet_idx}: {e}")
            # try computing all basins and pick id at outlet
            try:
                all_basins = self._fdir.basins()
                basin_id = int(all_basins[outlet_idx])
                self.basin = np.where(all_basins == basin_id, basin_id, 0)
            except Exception as e2:
                logger.exception(f"Fallback basins() also failed: {e2}")
                return
        
        self.catchment_mask = self.basin > 0
        # logging and sanity checks
        mean_cell_area = float(np.mean(cell_area)) if cell_area is not None else np.nan
        unique_vals = np.unique(self.basin[self.catchment_mask])
        cell_count = int(np.sum(self.catchment_mask))
        delineated_area = float(np.sum(cell_area[self.catchment_mask])) if cell_area is not None else np.nan
        uparea_at_outlet = upstream_area[outlet_idx] if upstream_area is not None else np.nan
        area_error = (delineated_area - ref_catchment_area) / ref_catchment_area

        logger.info(
            "Basin unique values: %s | cells in basin: %d | mean cell area: %.6f km2",
            unique_vals,
            cell_count,
            mean_cell_area,
        )
        logger.info(
            "Delineated basin area (sum of cell_area[basin>0]) = %.2f km2; "
            "reference area = %.2f km2; error = %.2f%%",
            delineated_area,
            ref_catchment_area,
            area_error * 100.0,
        )
        logger.info(
            "Upstream area reported at selected outlet cell = %.2f km2. "
            "Difference (sum_cells - upstream_at_outlet) = %.2f km2 (%.2f%%).",
            uparea_at_outlet,
            delineated_area - uparea_at_outlet,
            (delineated_area - uparea_at_outlet) / uparea_at_outlet * 100.0
            if uparea_at_outlet != 0
            else np.nan,
            )
        if abs(area_error) > max_error*2:
            with ErrorLogger(logger):
                msg = f"Delineated basin area ({delineated_area:2f} km2) differs from reference area ({ref_catchment_area:2f} km2) by more than twice the max error {max_error*100:.2f}%. Adjust max_error or max_distance_cells."
                raise ValueError(
                    msg
                )
            # warn if the two area measures disagree substantially
        if not np.isclose(delineated_area, uparea_at_outlet, rtol=0.02, atol=1e-6):
            with ErrorLogger(logger):
                msg = f"Sum of cell areas inside the basin ({delineated_area:2f} km2) differs from " 
                msg += f"upstream area at outlet ({uparea_at_outlet:2f} km2). Investigate flow-direction "
                msg += "masking, nodata handling or area units."
                raise ValueError(
                    msg
                )
        # finalize mask and basin fill values
        
        if np.all(~self.catchment_mask):
            if stream_order > 1:
                # try again with a lower stream_order (legacy behavior)
                self.delineate_basin(
                    gauge_coords,
                    stream_order=stream_order - 1,
                    ref_catchment_area=ref_catchment_area,
                    max_distance_cells=max_distance_cells,
                    max_error=max_error,
                )
                return
            logger.error("No catchment found for the given coordinates")
            return

        # set fillvalue for non-basin cells
        try:
            fillv = self.VARIABLES["basin"]["_FillValue"]
            self.basin = np.where(self.catchment_mask, self.basin, fillv)
        except Exception:
            # best-effort: leave basin as-is
            logger.debug("Could not set basin fill values")

    def get_upscaling_factor(self, max_resolution=True):
        """Create upscaling factor."""
        input_res = round(abs(self.ds.lon.data[1] - self.ds.lon.data[0]), 6)
        upscale_res = self.l1_resolution
        if max_resolution:
            upscale_res = max(
                [
                    res
                    for res in [
                        self.l1_resolution,
                        self.l11_resolution,
                        self.l2_resolution,
                    ]
                    if res is not None
                ],
                default=None,
            )
        if upscale_res is None:
            return 1
        if int(upscale_res / input_res + 0.5) - (upscale_res / input_res) < 1e6:
            return int(upscale_res / input_res + 0.5)
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
                msg = f"Data dimensions must be divisible by the upscaling factor of {factor}. Lat ({lat_size}/{factor})={lat_size / factor:.2f}; Lon ({lon_size}/{factor})={lon_size / factor:.2f}"
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
        # upgrid = self._fdir.upstream_area(unit="km2").astype(int)
        self.upgrid = self.calc_upstream_area().astype(int)

    def get_grid_area(self):
        """Perform the calculation of the catchment area."""
        self.get_upstream_area()
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
        """Replace all missing values adjacent to non-missing values with 0 in an xarray Dataset.

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
        gauge_id=None,
    ):
        """Write the produced data to one or multiple files."""
        data_vars = {}
        out_path = pl.Path(out_path)
        if not out_path.is_dir():
            out_path.mkdir(parents=True, exist_ok=True)
        lat_slice_idx, lon_slice_idx = None, None
        lat_slice, lon_slice = None, None
        if cut_by_basin:
            lat_slice_idx, lon_slice_idx = self.cut_to_filled_area(buffer)
        else:
            lat_slice, lon_slice = slice(84, -56), slice(None)

        for var_name in self.VARIABLES:
            data_var = self.processing_data_variable(
                var_name,
                cut_by_basin,
                lat_slice,
                lon_slice,
                lat_slice_idx,
                lon_slice_idx,
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
            if gauge_id is not None:
                # create empty ds with mask l0 extend and fill the data_var called data with -9999 values
                id_da = xr.DataArray(
                    np.full(ds.basin.shape, -9999, dtype=int), coords={"lat": ds.lat, "lon": ds.lon}, dims=["lat", "lon"]
                )
                id_ds = id_da.to_dataset(name="data")
                id_ds = write_gauge_id(id_ds, gauge_id, self.gauge_lat, self.gauge_lon)
                write_xarray_to_ascii(id_ds, out_path / 'gauges_id.asc', "data", fmt="%.0f")


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
            write_xarray_to_file(
                data_var,
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

    def processing_data_variable(
        self,
        var_name,
        cut_by_basin,
        lat_slice=None,
        lon_slice=None,
        lat_slice_idx=None,
        lon_slice_idx=None,
    ):
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
                    lon.min() - input_res / 2 + self.l2_resolution / 2,
                    lon.max() + self.l2_resolution / 2,
                    self.l2_resolution,
                )
                lat = np.arange(
                    lat.max() + input_res / 2 - self.l2_resolution / 2,
                    lat.min() - self.l2_resolution / 2,
                    -self.l2_resolution,
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
        if lat_slice is not None and lon_slice is not None:
            logger.info(f"Cutting {var_name} data to correct spatial dimensions")
            data_var = data_var.sel(lat=lat_slice, lon=lon_slice)
        elif lat_slice_idx is not None and lon_slice_idx is not None:
            logger.info(f"Cutting {var_name} data to correct spatial dimensions")
            data_var = data_var.isel(lat=lat_slice_idx, lon=lon_slice_idx)
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
        write_xarray_to_file(
            ds,
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

    def _cell_edges(self, centers: np.ndarray) -> np.ndarray:
        """Compute edges (len=N+1) from center coords (len=N) on a regular grid."""
        c = np.asarray(centers)
        d = np.diff(c)
        left = c[0] - 0.5 * d[0]
        right = c[-1] + 0.5 * d[-1]
        mids = (c[:-1] + c[1:]) / 2.0
        return np.concatenate(([left], mids, [right]))

    def _coarse_centers_from_edges(
        self, edges: np.ndarray, k: int, n_blocks: int, ascending: bool
    ) -> np.ndarray:
        """
        Given fine-grid edges, build coarse-grid centers for block size k.
        Ensures coarse edges == fine edges over the cropped window.
        """
        # we assume you've cropped L0 so len(fine_centers) is divisible by k
        # The window's left edge and right edge are edges[0] and edges[k*n_blocks]
        left_edge = edges[0]
        dx_coarse = edges[k] - edges[0]  # = k * dx_fine (works for asc/desc)
        # centers are midpoints of each coarse cell
        n = np.arange(n_blocks)
        centers = left_edge + (n + 0.5) * dx_coarse
        if not ascending:
            centers = centers[::-1]
        return centers

    def upscale_mask_with_correct_coords(
        self,
        da: xr.DataArray,
        factor: int = None,
        lon_name: str = "lon",
        lat_name: str = "lat",
    ) -> xr.DataArray:
        """
        Coarsen a 2D mask-like field by integer factor and assign correct coarse coords
        so that coarse *edges* equal fine *edges* of the cropped window.
        """
        if factor is None:
            factor = self.get_upscaling_factor(max_resolution=True)
        if factor < 1:
            raise ValueError("factor must be >= 1")

        # 1) coarsen over lon/lat windows
        kx = ky = int(factor)
        coarsen_map = {}
        if lon_name in da.dims:
            coarsen_map[lon_name] = kx
        if lat_name in da.dims:
            coarsen_map[lat_name] = ky

        cond = (~xr.apply_ufunc(np.isnan, da)) | (da == 0)
        out = cond.coarsen(coarsen_map, boundary="trim").any().astype("int8")

        # 2) compute correct coarse coordinates from fine edges
        lon_f = da[lon_name].values
        lat_f = da[lat_name].values
        lon_edges = self._cell_edges(lon_f)
        lat_edges = self._cell_edges(lat_f)

        asc_lon = lon_f[0] < lon_f[-1]
        asc_lat = lat_f[0] < lat_f[-1]

        n_lon_blocks = out.sizes.get(lon_name, 1)
        n_lat_blocks = out.sizes.get(lat_name, 1)

        # figure out which portion of edges we used after boundary="trim":
        # Since you cropped L0 to a multiple of factor, the coarsen starts at index 0
        # and uses exactly n_blocks*k cells. So we can take edges[0 : n_blocks*k + 1].
        lon_edges_win = (
            lon_edges[: n_lon_blocks * kx + 1]
            if asc_lon
            else lon_edges[-(n_lon_blocks * kx + 1) :]
        )
        lat_edges_win = (
            lat_edges[: n_lat_blocks * ky + 1]
            if asc_lat
            else lat_edges[-(n_lat_blocks * ky + 1) :]
        )

        lon_coarse = self._coarse_centers_from_edges(
            lon_edges_win, kx, n_lon_blocks, asc_lon
        )
        lat_coarse = self._coarse_centers_from_edges(
            lat_edges_win, ky, n_lat_blocks, asc_lat
        )

        out = out.assign_coords({lon_name: lon_coarse, lat_name: lat_coarse})
        out.name = da.name or "mask_L2"

        # 3) (optional) log edges for verification
        lon_edges_coarse = self._cell_edges(out[lon_name].values)
        lat_edges_coarse = self._cell_edges(out[lat_name].values)
        logger.info(
            f"Coarse lon edges: {lon_edges_coarse[0]:.6f} .. {lon_edges_coarse[-1]:.6f} "
            f"(should equal fine window edges: {lon_edges_win[0]:.6f} .. {lon_edges_win[-1]:.6f})"
        )
        logger.info(
            f"Coarse lat edges: {lat_edges_coarse[0]:.6f} .. {lat_edges_coarse[-1]:.6f} "
            f"(should equal fine window edges: {lat_edges_win[0]:.6f} .. {lat_edges_win[-1]:.6f})"
        )
        return out

    def upscale_mask(
        self, da: xr.DataArray, factor=None, lon_name="lon", lat_name="lat"
    ) -> xr.DataArray:
        """
        Upscale a 2D/3D mask-like field on a regular lon/lat grid by an integer factor.

        L2 cell = 1  if any underlying L0 cell is (not NaN) OR (== 0)
                = 0  otherwise

        Parameters
        ----------
        da : xr.DataArray
            Input with dims including lon_name and lat_name (time/other dims allowed).
        factor : int | tuple[int,int]
            Integer multiple(s): k or (kx, ky). E.g. 5 or (5, 3).
        lon_name, lat_name : str
            Coordinate names.

        Returns
        -------
        xr.DataArray
            Coarsened mask (0/1), aligned to the top-left of each kx×ky block.
        """
        logger.info("Create upscaled mask")
        if factor is None:
            factor = self.get_upscaling_factor(max_resolution=True)
        if isinstance(factor, int):
            kx = ky = int(factor)
        else:
            kx, ky = map(int, factor)

        if kx < 1 or ky < 1:
            raise ValueError("factor must be >= 1")

        # Condition: non-masked (not NaN) OR equals 0
        cond = (~xr.apply_ufunc(np.isnan, da)) | (da == 0)

        # Coarsen over lat/lon windows and take logical OR (any)
        coarsen_map = {
            dim: size
            for dim, size in ((lon_name, kx), (lat_name, ky))
            if dim in cond.dims
        }
        out = cond.coarsen(coarsen_map, boundary="trim").any().astype("int8")

        # Make coordinates every k-th point so grid aligns with the upscaled blocks
        if lon_name in out.dims:
            out = out.assign_coords(
                {lon_name: da[lon_name].isel({lon_name: slice(0, None, kx)})}
            )
        if lat_name in out.dims:
            out = out.assign_coords(
                {lat_name: da[lat_name].isel({lat_name: slice(0, None, ky)})}
            )

        out.name = da.name or "mask_L2"
        out.attrs.update(da.attrs)
        logger.info(
            f"Resolution of upscaled mask: {round(abs(out.lon.data[1] - out.lon.data[0]), 6)}"
        )
        return out

    def write_mask_file(self, ds, mask_file):
        """Write basin mask to specified path."""
        if mask_file is not None:
            logger.info("Writing mask file")
            # name the variable mask
            mask = ds.basin > 0
            mask_file = pl.Path(mask_file)
            mask_da = xr.DataArray(
                mask, coords={"lat": ds.lat, "lon": ds.lon}, dims=["lat", "lon"]
            )
            mask_da["lat"].attrs.update(
                {
                    "units": "degrees_north",
                    "long_name": "latitude",
                    "standard_name": "latitude",
                    "axis": "Y",
                }
            )
            mask_da["lon"].attrs.update(
                {
                    "units": "degrees_east",
                    "long_name": "longitude",
                    "standard_name": "longitude",
                    "axis": "X",
                }
            )
            mask_ds = xr.Dataset(
                {"land_mask": mask_da, "mask": mask_da}
            )
            mask_upscaled = None
            if self.do_upscale:
                mask_upscaled = mask_da
            elif self.l2_resolution is not None:
                mask_upscaled = self.upscale_mask_with_correct_coords(mask_da)
            if mask_upscaled is not None:
                mask_upscaled = mask_upscaled.rename({"lat": "lat_l2", "lon": "lon_l2"})
                mask_ds["land_mask_l2"] = mask_upscaled
            dims = set(mask_ds.dims)
            all_coords = set(mask_ds.coords)
            dim_coords = all_coords & dims  # intersection
            for var in dim_coords:
                bounds_name = f"{var}_bnds"
                mask_ds.coords[bounds_name] = generate_bounds(mask_ds[var])
                mask_ds[var].attrs["bounds"] = bounds_name
            write_xarray_to_file(mask_ds, mask_file)
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
            min_row = max(0, min_row - buffer)
            min_col = max(0, min_col - buffer)
            max_row = min(self.catchment_mask.shape[0] - 1, max_row + buffer)
            max_col = min(self.catchment_mask.shape[1] - 1, max_col + buffer)

            # min_row = min_row - buffer if min_row > 0 else min_row
            # min_col = min_col - buffer if min_col > 0 else min_col
            # max_row = (
            #     max_row + buffer if max_row < self.catchment_mask.shape[0] else max_row
            # )
            # max_col = (
            #     max_col + buffer if max_col < self.catchment_mask.shape[1] else max_col
            # )
        # logger.info(
        #     f"min row: {min_row} max row: {max_row} min_col: {min_col}, max_col: {max_col}"
        # )
        logger.info(
            f"L0 initial window (rows, cols): [{min_row}:{max_row}], [{min_col}:{max_col}]"
        )

        factor = self.get_upscaling_factor(max_resolution=True)
        if factor > 1:
            logger.info(
                f"Regridding to fit coarse grid with res {max([r for r in [self.l1_resolution, self.l11_resolution, self.l2_resolution] if r is not None ])} (factor {factor})"
            )
            min_row = min_row // factor * factor
            min_col = min_col // factor * factor
            # Calculating max_row/col it needs:
            #  +1 to include the whole last coarse grid cell -1 to not get one cell from the next block
            max_row = (max_row // factor + 1) * factor - 1
            max_col = (max_col // factor + 1) * factor - 1
            # clamp
            max_row = min(max_row, self.catchment_mask.shape[0] - 1)
            max_col = min(max_col, self.catchment_mask.shape[1] - 1)
            # if max_col >= len(cols):
            #     logger.warning("While regridding max_cols was larger than col-size")
            #     max_col = len(cols) - 1
            # if max_row >= len(rows):
            #     logger.warning("While regridding max_rows was larger than row-size")
            #     max_row = len(rows) - 1
        logger.info(
            f"min row: {min_row} max row: {max_row} min_col: {min_col}, max_col: {max_col}"
        )
        # build index slice
        lat_slice_idx = slice(min_row, max_row + 1)
        lon_slice_idx = slice(min_col, max_col + 1)

        # Sanity: cropped shape divisible by factor ---
        n_lat = lat_slice_idx.stop - lat_slice_idx.start
        n_lon = lon_slice_idx.stop - lon_slice_idx.start
        if factor > 1:
            if (n_lat % factor) != 0 or (n_lon % factor) != 0:
                raise AssertionError(
                    f"Cropped L0 shape ({n_lat}, {n_lon}) not divisible by factor={factor}"
                )

        # # Slice the array to extract the filled part
        # lon_min, lon_max = (
        #     np.round(self.ds.lon.values[min_col], 8),
        #     np.round(self.ds.lon.values[max_col], 8),
        # )
        # lat_min, lat_max = (
        #     np.round(self.ds.lat.values[max_row], 8),
        #     np.round(self.ds.lat.values[min_row], 8),
        # )
        # lat_slice = slice(lat_max, lat_min)
        # lon_slice = slice(lon_min, lon_max)
        logger.info(f"lat_slice: {lat_slice_idx}, lon_slice: {lon_slice_idx}")
        return lat_slice_idx, lon_slice_idx


def merge_catchment(path1, path2, out_path):
    """Merge the rolled and non-rolled file."""
    # read the rolled and non-rolled files
    ds1 = get_xarray_ds_from_file(path1, engine="netcdf4")
    ds2 = get_xarray_ds_from_file(path2, engine="netcdf4")

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
    write_xarray_to_file(merged, out_path)


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
    l11_resolution=None,
    l2_resolution=None,
    frame=1,
    upscale=False,
    latlon=True,
    available_mem=None,
    ref_catchment_area=None,
    max_distance_cells=5,
    max_error=0.1,
    gauge_id=None,
):
    """Create file containing catchment ids, flowdirection and upstream area from dem or flow direction."""
    logger.info(
        f"Creating catchment file for {var_name} using {var} and {ftype} from {input_file}"
    )

    if var not in {"fdir", "dem"}:
        with ErrorLogger(logger):
            msg = f"Unexpected value for var={var}, must be 'fdir' or 'dem'"
            raise ValueError(msg)

    with get_xarray_ds_from_file(
        input_file,
        var_name,
        normalize_latlon_coords=True,
        force_decending_y=True,
        available_mem_gib=available_mem,
    ) as input_ds:
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
                l11_resolution=l11_resolution,
                l2_resolution=l2_resolution,
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
                l11_resolution=l11_resolution,
                l2_resolution=l2_resolution,
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
                c.get_upstream_area()
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
            input_ds_sliced = input_ds.sel(
                lat=coordinate_slices["lat"], lon=coordinate_slices["lon"]
            )
            logger.info(
                f"lat {input_ds_sliced.lat.data[0]}, {input_ds_sliced.lat.data[-1]}"
            )
            logger.info(
                f"lon {input_ds_sliced.lon.data[0]}, {input_ds_sliced.lon.data[-1]}"
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
                l11_resolution=l11_resolution,
                l2_resolution=l2_resolution,
                upscale=upscale,
            )
            if l1_resolution is not None and upscale:
                c.upscale(var)
            else:
                c.get_facc()
            c.get_basins()
            c.get_grid_area()
            c.get_upstream_area()
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
                l11_resolution=l11_resolution,
                l2_resolution=l2_resolution,
                upscale=upscale,
            )
            c.delineate_basin(
                gauge_coords,
                ref_catchment_area=ref_catchment_area,
                max_distance_cells=max_distance_cells,
                max_error=max_error,
            )
            if l1_resolution is not None and upscale:
                c.upscale(var)
            else:
                c.get_facc()
            c.get_grid_area()
            c.get_upstream_area()
            c.write(
                output_path,
                single_file=True,
                cut_by_basin=True,
                mask_file=mask_file,
                frame=frame,
                buffer=frame,
                gauge_id=gauge_id
            )
