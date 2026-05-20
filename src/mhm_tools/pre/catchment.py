"""Create the catchment file for mRM.

Authors
-------
- Robert Schweppe
- Matthias Kelbling
- Jeisson Leal
- Simon Lüdke
"""

import csv
import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pyflwdir
import xarray as xr
from joblib import Parallel, delayed
from scipy.ndimage import binary_dilation

from mhm_tools.common.constants import NC_ENCODE_MASK
from mhm_tools.common.file_handler import (
    get_coord_values,
    get_xarray_ds_from_file,
    write_xarray_to_file,
)
from mhm_tools.common.logger import ErrorLogger, log_arguments
from mhm_tools.common.netcdf import generate_bounds
from mhm_tools.common.utils import (
    Resolution,
    coord_to_index,
    cut_to_filled_area,
    distance_100m_units,
    find_best_gauge_location_by_area,
    get_upscaling_factor,
)
from mhm_tools.common.xarray_utils import get_dtype
from mhm_tools.pre.create_id_gauges import write_gauge_id

logger = logging.getLogger(__name__)


# GLOBAL VARIABLES
FDIR_FILLVALUE = {"d8": 247, "ldd": 255}
FDIR_SINKVALUE = {"d8": 0, "ldd": 5}
FACC_FILLVALUE = 0
FILLVALUE = -9999
OUTPUT_VARIABLES = ("flwdir", "basin", "uparea_grid", "upgrid", "grdare", "elevtn")
GAUGE_INFO_COLUMNS = (
    "id",
    "lon",
    "lat",
    "lon_old",
    "lat_old",
    "distance",
    "area",
    "old_area",
    "area_error",
)
# use d8 for basinex, ldd for mRM version in Ulysses
OUTPUT_FTYPE = "ldd"
CUTOFF_THRESHOLD = 175


def _shape_crs(is_latlon):
    """Return a CRS string for shapefile operations based on lat/lon usage."""
    crs = "EPSG:4326" if is_latlon else None
    logger.debug("Resolved shape CRS to %s", crs)
    return crs


def _find_shape_file(shape_folder, gauge_id):
    """Find a shapefile in a folder that contains the gauge_id in its name."""
    if not shape_folder or gauge_id is None:
        return None
    shape_dir = Path(shape_folder)
    if not shape_dir.is_dir():
        return None
    matching_shapes = sorted(shape_dir.glob(f"*{gauge_id}*.shp"))
    if not matching_shapes:
        return None
    if len(matching_shapes) > 1:
        logger.warning(
            "Multiple shapefiles matched gauge_id %s in %s. Using %s",
            gauge_id,
            shape_dir,
            matching_shapes[0].name,
        )
    logger.debug("Using shapefile %s for gauge_id %s", matching_shapes[0], gauge_id)
    return matching_shapes[0]


def _vectorize_mask_to_gdf(basin_mask, affine_transform, crs, value_name="basin"):
    """Vectorize a basin mask into a GeoDataFrame."""
    try:
        import geopandas as gpd
        from rasterio import features
        from rasterio.transform import Affine
    except Exception as exc:
        error_msg = (
            "geopandas and rasterio are required for basin shapefile operations."
            "Please install them using `conda install geopandas`"
        )
        with ErrorLogger(logger):
            raise ImportError(error_msg) from exc

    if not isinstance(affine_transform, Affine):
        try:
            affine_transform = Affine(*affine_transform)
        except Exception:
            affine_transform = Affine.from_gdal(*affine_transform)

    data = basin_mask.astype(np.uint8)
    # Extract polygons for non-zero cells
    feats_gen = features.shapes(
        data,
        mask=data.astype(bool),
        transform=affine_transform,
        connectivity=8,
    )
    features_list = [
        {"geometry": geom, "properties": {value_name: 1}} for geom, _ in feats_gen
    ]
    if not features_list:
        logger.debug("No vectorizable features found in basin mask.")
        return gpd.GeoDataFrame(
            columns=[value_name, "geometry"], geometry="geometry", crs=crs
        )
    gdf = gpd.GeoDataFrame.from_features(features_list, crs=crs)
    gdf[value_name] = gdf[value_name].astype(np.uint8)
    logger.debug("Vectorized basin mask into %d features.", len(gdf))
    return gdf


def _shape_iou(reference_gdf, candidate_gdf):
    """Compute intersection-over-union for two GeoDataFrames."""
    if (
        reference_gdf is None
        or candidate_gdf is None
        or reference_gdf.empty
        or candidate_gdf.empty
    ):
        return 0.0
    reference_geom = reference_gdf.geometry.unary_union
    candidate_geom = candidate_gdf.geometry.unary_union
    if reference_geom.is_empty or candidate_geom.is_empty:
        return 0.0
    union_area = reference_geom.union(candidate_geom).area
    if union_area == 0:
        return 0.0
    intersection_area = reference_geom.intersection(candidate_geom).area
    iou = float(intersection_area / union_area)
    logger.debug("Computed shape IoU: %.4f", iou)
    return iou


def _coords_from_transform(affine_transform, grid_shape):
    """Derive coordinate arrays from an affine transform and grid shape."""
    try:
        from rasterio.transform import Affine, xy
    except Exception as exc:
        error_msg = "rasterio is required to derive coordinates from transform."
        with ErrorLogger(logger):
            raise ImportError(error_msg) from exc

    if not isinstance(affine_transform, Affine):
        try:
            affine_transform = Affine(*affine_transform)
        except Exception:
            affine_transform = Affine.from_gdal(*affine_transform)
    n_rows, n_cols = grid_shape
    col_indices = np.arange(n_cols)
    row_indices = np.arange(n_rows)
    x_coords = np.array(
        [xy(affine_transform, 0, col, offset="center")[0] for col in col_indices]
    )
    y_coords = np.array(
        [xy(affine_transform, row, 0, offset="center")[1] for row in row_indices]
    )
    logger.debug(
        "Derived coordinate arrays from transform with lengths %d (x) and %d (y).",
        len(x_coords),
        len(y_coords),
    )
    return x_coords, y_coords


def _combine_shape_bounds(bounds_list):
    """Combine multiple (minx, miny, maxx, maxy) tuples into a single bounding box."""
    if not bounds_list:
        return None
    min_x = min(b[0] for b in bounds_list)
    min_y = min(b[1] for b in bounds_list)
    max_x = max(b[2] for b in bounds_list)
    max_y = max(b[3] for b in bounds_list)
    return min_x, min_y, max_x, max_y


def _buffer_bounds(bounds, latlon):
    """Apply a buffer to bounds (degrees if latlon, otherwise meters)."""
    if bounds is None:
        return None
    buffer_value = 1.0 if latlon else 100_000.0
    min_x, min_y, max_x, max_y = bounds
    return (
        min_x - buffer_value,
        min_y - buffer_value,
        max_x + buffer_value,
        max_y + buffer_value,
    )


def _slices_from_bounds(bounds, lat_values):
    """Create lon/lat slices from bounds, respecting latitude ordering."""
    if bounds is None:
        return None
    min_x, min_y, max_x, max_y = bounds
    if lat_values[0] > lat_values[-1]:
        lat_slice = slice(max_y, min_y)
    else:
        lat_slice = slice(min_y, max_y)
    lon_slice = slice(min_x, max_x)
    return {"lat": lat_slice, "lon": lon_slice}


def _shape_bounds_from_folder(shape_folder, gauge_ids, latlon):
    """Compute a buffered bounding box from available gauge shapefiles."""
    if not shape_folder or gauge_ids is None:
        logger.warning(
            f"No shape_folder or gauge_ids provided; cannot compute shape bounds. {shape_folder}, {gauge_ids}"
        )
        return None
    try:
        import geopandas as gpd
    except Exception as exc:
        logger.warning("geopandas missing; cannot use shapefile bounds: %s", exc)
        return None

    gauge_id_list = [gauge_ids] if not isinstance(gauge_ids, list) else gauge_ids

    bounds_list = []
    logger.debug(
        "Looking for shapefiles in %s for gauge_ids: %s", shape_folder, gauge_id_list
    )
    for gauge_id in gauge_id_list:
        logger.info(
            "Looking for shapefile for gauge_id %s in %s", gauge_id, shape_folder
        )
        shape_path = _find_shape_file(shape_folder, gauge_id)
        logger.debug("Found shapefile %s for gauge_id %s", shape_path, gauge_id)
        if shape_path is None:
            continue
        try:
            gdf = gpd.read_file(shape_path)
        except Exception as exc:
            logger.warning("Failed to read shapefile %s: %s", shape_path, exc)
            continue
        bounds_list.append(tuple(gdf.total_bounds))

    combined_bounds = _combine_shape_bounds(bounds_list)
    if combined_bounds is None:
        return None
    buffered_bounds = _buffer_bounds(combined_bounds, latlon)
    logger.info("Using shapefile bounds with buffer for slicing: %s", buffered_bounds)
    return buffered_bounds


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
    return xr.DataArray(
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


def _normalize_output_vars(output_vars):
    if output_vars is None:
        return set(OUTPUT_VARIABLES)
    if isinstance(output_vars, str):
        output_vars = [val.strip() for val in output_vars.split(",") if val.strip()]
    selected = {val.strip() for val in output_vars if str(val).strip()}
    if not selected:
        msg = "output_vars must contain at least one variable name."
        with ErrorLogger(logger):
            raise ValueError(msg)
    unknown = selected - set(OUTPUT_VARIABLES)
    if unknown:
        msg = (
            "Unknown output vars: "
            f"{sorted(unknown)}. Valid options: {', '.join(OUTPUT_VARIABLES)}."
        )
        with ErrorLogger(logger):
            raise ValueError(msg)
    return selected


def _to_numeric_gauge_ids(raw_ids, context="output"):
    """Convert gauge IDs to integers, generating surrogate IDs when needed."""
    if isinstance(raw_ids, (str, bytes)):
        raw_ids = [raw_ids]
    elif not isinstance(raw_ids, list):
        raw_ids = list(raw_ids)
    key_to_numeric = {}
    used_numeric = {}
    surrogate_candidates = []

    for raw in raw_ids:
        key = "" if raw is None else str(raw).strip()
        if key in key_to_numeric:
            continue
        try:
            numeric_id = int(key)
        except (TypeError, ValueError):
            surrogate_candidates.append(key)
            continue
        if numeric_id in used_numeric and used_numeric[numeric_id] != key:
            logger.warning(
                "Gauge id '%s' collides with '%s' after int conversion for %s; using surrogate id.",
                key,
                used_numeric[numeric_id],
                context,
            )
            surrogate_candidates.append(key)
            continue
        key_to_numeric[key] = numeric_id
        used_numeric[numeric_id] = key

    next_surrogate = max(used_numeric) + 1 if used_numeric else 1
    for key in surrogate_candidates:
        if key in key_to_numeric:
            continue
        while next_surrogate in used_numeric:
            next_surrogate += 1
        key_to_numeric[key] = next_surrogate
        used_numeric[next_surrogate] = key
        logger.warning(
            "Gauge id '%s' is non-numeric for %s; using surrogate id %d.",
            key if key else "<empty>",
            context,
            next_surrogate,
        )
        next_surrogate += 1

    return [key_to_numeric["" if raw is None else str(raw).strip()] for raw in raw_ids]


def write_gauges_out(gauges, output_target):
    """Write gauge information to CSV and NetCDF files."""
    write_gauges_to_csv(gauges, output_target.with_suffix(".csv"))
    write_gauges_to_nc(gauges, output_target.with_suffix(".nc"))


def write_gauges_to_csv(gauges, output_target, filename="gauges_info.csv"):
    """Write gauge information to a CSV file.

    Parameters
    ----------
    gauges : Gauge | list[Gauge] | dict | list[dict]
        Gauge objects or dict rows.
    output_target : str | Path
        Either a target file path or a target directory.
    filename : str
        Filename used when output_target is a directory.
    """
    if not gauges:
        logger.warning("No gauges to write to CSV.")
        return
    if isinstance(gauges, (Gauge, dict)):
        gauges = [gauges]

    output_target = Path(output_target)
    output_path = (
        output_target if output_target.suffix else output_target / str(filename)
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    for gauge in gauges:
        if isinstance(gauge, dict):
            row = {
                "id": gauge.get("id", gauge.get("gauge_id")),
                "lon": gauge.get("lon"),
                "lat": gauge.get("lat"),
                "lon_old": gauge.get("lon_old"),
                "lat_old": gauge.get("lat_old"),
                "distance": gauge.get("distance", gauge.get("distance_error", np.nan)),
                "area": gauge.get("area"),
                "old_area": gauge.get("old_area", gauge.get("area_old")),
                "area_error": gauge.get("area_error", gauge.get("error")),
            }
        else:
            row = {
                "id": gauge.gauge_id,
                "lon": gauge.lon,
                "lat": gauge.lat,
                "lon_old": gauge.lon_old,
                "lat_old": gauge.lat_old,
                "distance": gauge.distance_error,
                "area": gauge.area,
                "old_area": gauge.area_old,
                "area_error": gauge.area_error,
            }
        rows.append(row)

    with output_path.open("w", newline="", encoding="utf-8") as fp:
        writer = csv.DictWriter(fp, fieldnames=GAUGE_INFO_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    logger.info("Wrote gauge information for %d gauges to %s", len(rows), output_path)


def write_gauges_to_nc(gauges, output_target, filename="gauges_info.nc"):
    """Write gauge information to a NetCDF file with CF-style station metadata.

    Parameters
    ----------
    gauges : Gauge | list[Gauge] | dict | list[dict]
        Gauge objects or dict rows.
    output_target : str | Path
        Either a target file path or a target directory.
    filename : str
        Filename used when output_target is a directory.
    """
    if not gauges:
        logger.warning("No gauges to write to NetCDF.")
        return
    if isinstance(gauges, (Gauge, dict)):
        gauges = [gauges]

    output_target = Path(output_target)
    output_path = (
        output_target if output_target.suffix else output_target / str(filename)
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)

    station_ids_raw = []
    lons = []
    lats = []
    areas = []
    for gauge in gauges:
        if isinstance(gauge, dict):
            station_id = gauge.get("gauge_id", gauge.get("id"))
            lon = gauge.get("lon")
            lat = gauge.get("lat")
            area = gauge.get("area")
        else:
            station_id = gauge.gauge_id
            lon = gauge.lon
            lat = gauge.lat
            area = gauge.area

        station_ids_raw.append(station_id)
        lons.append(np.nan if lon is None else float(lon))
        lats.append(np.nan if lat is None else float(lat))
        areas.append(np.nan if area is None else float(area))

    if not station_ids_raw:
        logger.warning("No gauge ids available for NetCDF output.")
        return
    station_ids = _to_numeric_gauge_ids(station_ids_raw, context="gauges_info.nc")

    station = np.asarray(station_ids, dtype=np.int64)
    ds = xr.Dataset(
        data_vars={
            "lon": ("station", np.asarray(lons, dtype=np.float64)),
            "lat": ("station", np.asarray(lats, dtype=np.float64)),
            "area": ("station", np.asarray(areas, dtype=np.float64)),
        },
        coords={"station": ("station", station)},
        attrs={
            "title": "SCC gauges specification",
            "Conventions": "CF-1.12",
            "institution": "UFZ",
            "source": "Catchment delineation with mhm-tools",
        },
    )

    ds["station"].attrs.update(
        {
            "long_name": "Station ID",
            "coordinates": "lon lat",
        }
    )
    ds["lon"].attrs.update(
        {
            "long_name": "longitude",
            "standard_name": "longitude",
            "units": "degrees_east",
        }
    )
    ds["lat"].attrs.update(
        {
            "long_name": "latitude",
            "standard_name": "latitude",
            "units": "degrees_north",
        }
    )
    ds["area"].attrs.update(
        {
            "standard_name": "catchment_area",
            "long_name": (
                "catchment area based on correction to Merit Hydro by Peter Burek et al. (2023)"
            ),
            "units": "km2",
        }
    )

    encoding = {
        "station": {"dtype": "int64"},
        "lon": {"dtype": "float64", "_FillValue": np.nan},
        "lat": {"dtype": "float64", "_FillValue": np.nan},
        "area": {"dtype": "float64", "_FillValue": np.nan},
    }
    ds.to_netcdf(output_path, engine="netcdf4", format="NETCDF4", encoding=encoding)
    logger.info(
        "Wrote gauge information for %d gauges to %s", len(station_ids), output_path
    )


# CLASSES


class Gauge:
    """Class to hold gauge information."""

    def __init__(self, gauge_id=None, lat=None, lon=None, area=None, id=None):
        if gauge_id is None:
            gauge_id = id
        self.gauge_id = gauge_id
        # Backward-compatible alias for older call sites using `gauge.id`.
        self.id = gauge_id
        self.lat = lat
        self.lon = lon
        self.area = area
        self.area_old = None
        self.lat_old = None
        self.lon_old = None
        self.distance_error = None
        self.area_error = None

    def update(
        self, area=None, lat=None, lon=None, distance_error=None, area_error=None
    ):
        """Update gauge information, preserving old values."""
        if area is not None:
            self.area_old = self.area
            self.area = area
        if lat is not None:
            self.lat_old = self.lat
            self.lat = lat
        if lon is not None:
            self.lon_old = self.lon
            self.lon = lon
        if distance_error is not None:
            self.distance_error = distance_error
        if area_error is not None:
            self.area_error = area_error


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
        resolutions: Resolution = None,
        upscale=False,
        latlon=True,
        l0_precision: int = 9,
    ):
        self.flwdir = None
        self.basin = None
        self.upgrid = None
        self.uparea_grid = None
        self.grdare = None
        self.elevtn = None
        self.cell_area = None
        self._fdir = None
        self.gauge_ids = []
        self.gauge_lats = []
        self.gauge_lons = []
        self.ftype = ftype
        self.catchment_mask = None
        self.resolutions = resolutions if resolutions is not None else Resolution()
        if self.resolutions.l0 is None:
            self.resolutions.l0 = round(
                abs(ds.lon.data[1] - ds.lon.data[0]), l0_precision
            )
        self.upscaled_resolution = self.resolutions.l0
        self.do_upscale = upscale
        self.is_upscaled = False
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
        self.latlon = latlon
        self.latlon = latlon
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
                (data != old_no_data_val) & ~np.isnan(data),
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

    def get_current_coordinates(self):
        """Build L1 coordinate arrays based on the dataset extent and L1 resolution."""
        lon_coords = self.ds.lon.data
        lat_coords = self.ds.lat.data
        input_resolution = self.resolutions.l0
        if (
            self.resolutions.l1 is not None
            and input_resolution != self.resolutions.l1
            and self.do_upscale
            and self.is_upscaled
        ):
            # Rebuild coordinates for the coarser L1 grid
            lon_coords = np.arange(
                lon_coords.min() - input_resolution / 2 + self.resolutions.l1 / 2,
                lon_coords.max() + self.resolutions.l1 / 2,
                self.resolutions.l1,
            )
            lat_coords = np.arange(
                lat_coords.max() + input_resolution / 2 - self.resolutions.l1 / 2,
                lat_coords.min() - self.resolutions.l1 / 2,
                -self.resolutions.l1,
            )
        logger.debug(
            "Computed L1 coords: lon=%d, lat=%d", len(lon_coords), len(lat_coords)
        )
        return lon_coords, lat_coords

    def compute_cell_area(self):
        """Create a cell area data array in km2."""
        logger.info("Create cell area data array.")
        lon, lat = self.get_current_coordinates()
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
        self.cell_area = cell_areas

    def calc_upstream_area(self):
        """Use pyflwdir to calculate the upstream area from flow direction by providing cell areas."""
        if self._fdir is None:
            logger.error("Flow direction is not initialized.")
            return None
        if self.cell_area is None:
            self.compute_cell_area()
        if self.cell_area.shape != self._fdir.shape:
            msg = (
                f"cell_area shape {self.cell_area.shape} does not match "
                f"flow-direction shape {self._fdir.shape}."
            )
            with ErrorLogger(logger):
                raise ValueError(msg)
        return self._fdir.accuflux(self.cell_area, nodata=-9999)

    def _coord_to_index(self, lat, lon, lat_vals=None, lon_vals=None):
        """Map latitude/longitude or indices to integer grid indices."""
        if "lat" not in self.ds.coords or "lon" not in self.ds.coords:
            msg = "Dataset is missing latitude/longitude coordinates."
            with ErrorLogger(logger):
                raise ValueError(msg)
        lat_vals = self.ds.lat.data if lat_vals is None else lat_vals
        lon_vals = self.ds.lon.data if lon_vals is None else lon_vals

        if isinstance(lat, (int, np.integer)):
            i = int(lat)
            logger.debug(
                f"Was given latitude index {i} directly. Corresponding lat_value {lat_vals[i]}"
            )
        elif lat < min(lat_vals) or lat > max(lat_vals):
            logger.error(
                f"Given latitude {lat} is outside dataset bounds ({min(lat_vals)}, {max(lat_vals)}). Clipping to bounds."
            )
            i = None
        else:
            i = int(np.abs(lat_vals - float(lat)).argmin())
            logger.debug(
                f"Mapped latitude {float(lat)} to index {i} with lat_value {lat_vals[i]}"
            )

        if isinstance(lon, (int, np.integer)):
            j = int(lon)
            logger.debug(
                f"Was given longitude index {j} directly. Corresponding lon_value {lon_vals[j]}"
            )
        elif lon < min(lon_vals) or lon > max(lon_vals):
            logger.error(
                f"Given longitude {lon} is outside dataset bounds ({min(lon_vals)}, {max(lon_vals)}). Clipping to bounds."
            )
            j = None
        else:
            j = int(np.abs(lon_vals - float(lon)).argmin())
            logger.debug(
                f"Mapped longitude {float(lon)} to index {j} with lon_value {lon_vals[j]}"
            )
        if i is None or j is None:
            msg = (
                "Could not map given coordinates to valid indices within "
                "dataset bounds."
            )
            with ErrorLogger(logger):
                raise ValueError(msg)
        i = int(np.clip(i, 0, len(lat_vals) - 1))
        j = int(np.clip(j, 0, len(lon_vals) - 1))

        return i, j

    def _coords_l1(self):
        """Build L1 coordinate arrays based on the dataset extent and L1 resolution."""
        lon_coords = self.ds.lon.data
        lat_coords = self.ds.lat.data
        input_resolution = self.resolutions.l0
        if (
            self.resolutions.l1 is not None
            and input_resolution != self.resolutions.l1
            and self.do_upscale
        ):
            # Rebuild coordinates for the coarser L1 grid
            lon_coords = np.arange(
                lon_coords.min() - input_resolution / 2 + self.resolutions.l1 / 2,
                lon_coords.max() + self.resolutions.l1 / 2,
                self.resolutions.l1,
            )
            lat_coords = np.arange(
                lat_coords.max() + input_resolution / 2 - self.resolutions.l1 / 2,
                lat_coords.min() - self.resolutions.l1 / 2,
                -self.resolutions.l1,
            )
        logger.debug(
            "Computed L1 coords: lon=%d, lat=%d", len(lon_coords), len(lat_coords)
        )
        return lon_coords, lat_coords

    def update_gauge_coords(self, gauge_id, gauge_lat, gauge_lon):
        """Update stored gauge coordinates for a gauge_id or add it if missing."""
        if gauge_id is None:
            return
        if gauge_id in self.gauge_ids:
            gauge_index = self.gauge_ids.index(gauge_id)
            self.gauge_lats[gauge_index] = gauge_lat
            self.gauge_lons[gauge_index] = gauge_lon
            logger.debug("Updated gauge %s to %s/%s", gauge_id, gauge_lat, gauge_lon)
        else:
            self.gauge_ids.append(gauge_id)
            self.gauge_lats.append(gauge_lat)
            self.gauge_lons.append(gauge_lon)
            logger.debug("Added gauge %s at %s/%s", gauge_id, gauge_lat, gauge_lon)

    def correct_gauge_location_l1_from_shape(
        self,
        l0_shape_gdf,
        gauge_coords,
        max_distance_cells=5,
        ref_catchment_area=None,
        reference_upstream_area=None,
    ):
        """Correct gauge coordinates at L1 using L0 shape similarity and upstream area."""
        if ref_catchment_area is None and reference_upstream_area is not None:
            ref_catchment_area = reference_upstream_area
        if l0_shape_gdf is None or l0_shape_gdf.empty:
            return None
        if self._fdir is None:
            return None
        lon_coords, lat_coords = self._coords_l1()
        if (
            len(lat_coords) != self._fdir.shape[0]
            or len(lon_coords) != self._fdir.shape[1]
        ):
            lon_coords, lat_coords = _coords_from_transform(
                getattr(self._fdir, "transform", self.transform), self._fdir.shape
            )

        l1_upstream_area = self.uparea_grid
        if l1_upstream_area is None:
            logger.warning(
                "L1 upstream area grid missing; cannot apply shape correction."
            )
            return None
        l1_result = self.find_best_gauge_location_shape(
            upstream_area=l1_upstream_area,
            gauge_coords=gauge_coords,
            ref_catchment_area=ref_catchment_area,
            shape_folder=None,
            gauge_id=None,
            max_distance_cells=max_distance_cells,
            max_error=1.0,
            reference_shape_gdf=l0_shape_gdf,
            lat_values=lat_coords,
            lon_values=lon_coords,
        )
        if l1_result is None:
            logger.warning("No L1 candidate basins matched the L0 shape.")
            return None

        def _iou_for_index(row_idx, col_idx):
            try:
                linear_idx = np.ravel_multi_index((row_idx, col_idx), self._fdir.shape)
                basin = self._fdir.basins(idxs=np.array([linear_idx], dtype=np.int64))
                candidate_mask = basin > 0
                gdf = _vectorize_mask_to_gdf(
                    candidate_mask,
                    getattr(self._fdir, "transform", self.transform),
                    _shape_crs(self.latlon),
                )
                return _shape_iou(l0_shape_gdf, gdf)
            except Exception as exc:
                logger.debug(
                    "Could not compute L1 shape IoU for outlet candidate: %s", exc
                )
                return np.nan

        best_candidate_index, _, _ = l1_result
        candidate_iou = _iou_for_index(
            int(best_candidate_index[0]), int(best_candidate_index[1])
        )

        gauge_row = int(np.abs(lat_coords - float(gauge_coords[0])).argmin())
        gauge_col = int(np.abs(lon_coords - float(gauge_coords[1])).argmin())
        naive_iou = _iou_for_index(gauge_row, gauge_col)

        if (
            np.isfinite(candidate_iou)
            and np.isfinite(naive_iou)
            and candidate_iou < naive_iou
        ):
            logger.info(
                "L1 shape candidate IoU %.4f is worse than naive IoU %.4f; keeping naive L1 gauge location.",
                candidate_iou,
                naive_iou,
            )
            return float(lat_coords[gauge_row]), float(lon_coords[gauge_col])

        new_lat = float(lat_coords[int(best_candidate_index[0])])
        new_lon = float(lon_coords[int(best_candidate_index[1])])
        logger.info(
            "L1 shape-based correction selected %.4f/%.4f using L0 shape reference.",
            new_lat,
            new_lon,
        )
        return new_lat, new_lon

    def find_best_gauge_location_shape(  # noqa: PLR0915
        self,
        upstream_area,
        gauge_coords,
        ref_catchment_area,
        shape_folder,
        gauge_id,
        max_distance_cells=2,
        max_error=0.25,
        reference_shape_gdf=None,
        lat_values=None,
        lon_values=None,
        limit_by_error=False,
    ):
        """Find best gauge location using shape similarity."""
        if upstream_area is None:
            logger.warning("Upstream area grid missing for shape-based correction.")
            return None
        if reference_shape_gdf is None:
            shape_path = _find_shape_file(shape_folder, gauge_id)
            if shape_path is None:
                logger.debug("No reference shapefile found for gauge_id %s", gauge_id)
                return None
            try:
                import geopandas as gpd
            except Exception as exc:
                error_msg = "geopandas is required for shape-based gauge correction."
                with ErrorLogger(logger):
                    raise ImportError(error_msg) from exc
            reference_shape = gpd.read_file(shape_path)
            crs = _shape_crs(self.latlon)
            if crs is not None:
                if reference_shape.crs is None:
                    reference_shape = reference_shape.set_crs(crs)
                elif str(reference_shape.crs) != str(crs):
                    reference_shape = reference_shape.to_crs(crs)
            logger.debug("Loaded reference shape %s", shape_path)
            shape_label = shape_path.name
        else:
            reference_shape = reference_shape_gdf
            shape_label = "reference shape"
            crs = reference_shape.crs if hasattr(reference_shape, "crs") else None

        if lat_values is None or lon_values is None:
            lat_values = self.ds.lat.data
            lon_values = self.ds.lon.data

        gauge_row, gauge_col = self._coord_to_index(
            gauge_coords[0],
            gauge_coords[1],
            lat_vals=lat_values,
            lon_vals=lon_values,
        )
        max_cells = (
            int(max(0, round(max_distance_cells)))
            if max_distance_cells is not None
            else 0
        )
        row_min = max(0, gauge_row - max_cells)
        row_max = min(len(lat_values) - 1, gauge_row + max_cells)
        col_min = max(0, gauge_col - max_cells)
        col_max = min(len(lon_values) - 1, gauge_col + max_cells)
        if row_min > row_max:
            row_min, row_max = row_max, row_min
        if col_min > col_max:
            col_min, col_max = col_max, col_min

        # Limit candidate search to a local neighborhood around the gauge
        sub = upstream_area[row_min : row_max + 1, col_min : col_max + 1]
        if ref_catchment_area is not None and limit_by_error:
            candidate_indices = np.where(
                (sub >= ref_catchment_area * (1 - max_error))
                & (sub <= ref_catchment_area * (1 + max_error))
            )
            if candidate_indices[0].size == 0:
                logger.warning(
                    "No candidates within area error bounds; expanding search to all finite upstream area values."
                )
                candidate_indices = np.where(np.isfinite(sub))
        else:
            candidate_indices = np.where(np.isfinite(sub))

        logger.debug(
            "Shape-based candidate count: %d",
            len(candidate_indices[0]),
        )

        best_candidate_index = None
        best_candidate_score = np.inf
        best_candidate_shape_iou = 0.0
        best_candidate_upstream_area = None
        best_candidate_distance_100m = None
        logger.debug(
            f"Using resolution {self.upscaled_resolution} for shape-based distance scoring."
        )
        lat_deg = float(lat_values[gauge_row]) if self.latlon else None
        for cand_row, cand_col in zip(candidate_indices[0], candidate_indices[1]):
            row_idx = int(cand_row + row_min)
            col_idx = int(cand_col + col_min)
            linear = np.ravel_multi_index((row_idx, col_idx), self._fdir.shape)
            basin = self._fdir.basins(
                idxs=np.array([linear], dtype=np.int64),
                # streams=streams_mask,
            )
            basin_mask = basin > 0
            if not np.any(basin_mask):
                continue
            gdf = _vectorize_mask_to_gdf(
                basin_mask, self.transform, crs, value_name="basin"
            )
            shape_overlap_ratio = _shape_iou(reference_shape, gdf)
            upstream_value = upstream_area[row_idx, col_idx]
            distance_100m = distance_100m_units(
                row_idx - gauge_row,
                col_idx - gauge_col,
                l0_resolution=self.upscaled_resolution,
                lat_deg=lat_deg,
            )
            if ref_catchment_area:
                upstream_area_ratio = (
                    min(upstream_value, ref_catchment_area)
                    / max(upstream_value, ref_catchment_area)
                    if upstream_value and ref_catchment_area
                    else 0.0
                )
                score = np.hypot(
                    1 - upstream_area_ratio, 1 - shape_overlap_ratio
                )  # sqrt(x1**2 + x2**2)
            else:
                score = 1 - shape_overlap_ratio
            if (
                best_candidate_index is None
                or score < best_candidate_score
                or (
                    np.isclose(score, best_candidate_score)
                    and best_candidate_distance_100m is not None
                    and distance_100m < best_candidate_distance_100m
                )
            ):
                best_candidate_index = (row_idx, col_idx)
                best_candidate_score = score
                best_candidate_shape_iou = shape_overlap_ratio
                best_candidate_upstream_area = upstream_value
                best_candidate_distance_100m = distance_100m

        if best_candidate_index is None:
            logger.warning("No suitable candidate found for shape-based correction.")
            return None
        distance = distance_100m_units(
            best_candidate_index[0] - gauge_row,
            best_candidate_index[1] - gauge_col,
            l0_resolution=self.upscaled_resolution,
            lat_deg=lat_deg,
        )
        area_error = (
            abs(1 - best_candidate_upstream_area / ref_catchment_area)
            if ref_catchment_area and best_candidate_upstream_area
            else 0.0
        )
        logger.info(
            "Shape-based gauge correction used %s with IoU %.3f and area error %.3f.",
            shape_label,
            best_candidate_shape_iou,
            area_error,
        )
        return best_candidate_index, area_error, distance

    def get_best_gauge_coordinate(
        self,
        upstream_area,
        gauge_coords,
        ref_catchment_area,
        max_distance_cells,
        max_error,
        method,
        shape_folder=None,
        gauge_id=None,
        raise_on_fallback=True,
    ):
        """Get best gauge coordinates given target catchment area."""
        shape_result = None
        if shape_folder and gauge_id is not None:
            try:
                shape_result = self.find_best_gauge_location_shape(
                    upstream_area,
                    gauge_coords,
                    ref_catchment_area,
                    shape_folder,
                    gauge_id,
                    max_distance_cells=max_distance_cells,
                    max_error=max_error,
                    active_resolution=self.resolutions.l0,
                )
            except Exception as exc:
                logger.warning(
                    "Shape-based gauge correction failed for %s: %s",
                    gauge_id,
                    exc,
                )
                shape_result = None

        if shape_result is not None:
            outlet_idx, error, distance_error = shape_result
            new_lat = float(self.ds.lat.data[outlet_idx[0]])
            new_lon = float(self.ds.lon.data[outlet_idx[1]])
            gauge_lat = new_lat
            gauge_lon = new_lon
            logger.info(
                "Moved outlet to %s/%s using shape similarity (distance %.2fkm).",
                new_lat,
                new_lon,
                distance_error / 10,
            )
        elif ref_catchment_area is not None:
            if method != "all":
                outlet_idx, error, distance_error = find_best_gauge_location_by_area(
                    ds=self.ds,
                    upstream_area=upstream_area,
                    gauge_coords=gauge_coords,
                    ref_catchment_area=ref_catchment_area,
                    resolutions=self.resolutions,
                    max_distance_cells=max_distance_cells,
                    max_error=max_error,
                    method=method,
                    raise_on_fallback=raise_on_fallback,
                )
            else:
                outlet_idx_bx, error_bx, distance_error_bx = (
                    find_best_gauge_location_by_area(
                        ds=self.ds,
                        upstream_area=upstream_area,
                        gauge_coords=gauge_coords,
                        ref_catchment_area=ref_catchment_area,
                        resolutions=self.resolutions,
                        max_distance_cells=max_distance_cells,
                        max_error=max_error,
                        method="basinex",
                        raise_on_fallback=False,
                    )
                )
                outlet_idx_bu, error_bu, distance_error_bu = (
                    find_best_gauge_location_by_area(
                        ds=self.ds,
                        upstream_area=upstream_area,
                        gauge_coords=gauge_coords,
                        ref_catchment_area=ref_catchment_area,
                        resolultion=self.resolutions,
                        max_distance_cells=max_distance_cells,
                        max_error=max_error,
                        method="burek",
                        raise_on_fallback=False,
                    )
                )
                logger.info("Results of basin correction:")
                logger.info(f"Burek: lat: {float(self.ds.lat.data[outlet_idx_bu[0]])}")
                logger.info(f"Burek: lon: {float(self.ds.lon.data[outlet_idx_bu[1]])}")
                logger.info(f"Burek: error {error_bu}")
                logger.info(f"Burek: distance change {distance_error_bu/10}km")
            if method in ("all", "basinex"):
                if method == "basinex":
                    outlet_idx_bx = outlet_idx
                    error_bx = error
                    distance_error_bx = distance_error
                logger.info(
                    f"BasinEx: lat: {float(self.ds.lat.data[outlet_idx_bx[0]])}"
                )
                logger.info(
                    f"BasinEx: lon: {float(self.ds.lon.data[outlet_idx_bx[1]])}"
                )
                logger.info(f"BasinEx: error {error_bx}")
                logger.info(f"BasinEx: distance change {distance_error_bx/10}km")
            if method == "all":
                if error_bx < error_bu:
                    logger.info("using BasinEx location")
                    outlet_idx = outlet_idx_bx
                    error = error_bx
                    distance_error = distance_error_bx
                else:
                    logger.info("Using Burek location")
                    outlet_idx = outlet_idx_bu
                    error = error_bu
                    distance_error = distance_error_bu
            new_lat = float(self.ds.lat.data[outlet_idx[0]])
            new_lon = float(self.ds.lon.data[outlet_idx[1]])
            gauge_lat = new_lat
            gauge_lon = new_lon
            logger.info(
                f"Moved outlet {distance_error/10:.2}km shifting latitude {float(gauge_coords[0])} to {new_lat} and longitude {float(gauge_coords[1])} to {new_lon}."
            )
        else:
            logger.warning(
                "No catchment area provided; falling back to original gauge coordinates."
            )
            outlet_idx = coord_to_index(self.ds, gauge_coords[0], gauge_coords[1])
            gauge_lat = float(gauge_coords[0])
            gauge_lon = float(gauge_coords[1])
            error = None
            distance_error = 0.0
        return outlet_idx, error, gauge_lat, gauge_lon, distance_error / 10

    def delineation_sanity_check(
        self,
        catchment_mask,
        basin,
        uparea_at_outlet,
        ref_catchment_area,
        max_error,
        raise_on_sanity_check,
        gauge_id,
    ):
        """Perform sanity checks on the delineated basin."""
        try:
            mean_cell_area = (
                float(np.mean(self.cell_area[catchment_mask]))
                if self.cell_area is not None
                else np.nan
            )
            unique_vals = np.unique(basin[catchment_mask])
            cell_count = int(np.sum(catchment_mask))
            delineated_area = (
                float(np.sum(self.cell_area[catchment_mask]))
                if self.cell_area is not None
                else np.nan
            )

            logger.info(
                "Basin unique values: %s | cells in basin: %d | mean cell area: %.6f km2",
                unique_vals,
                cell_count,
                mean_cell_area,
            )

            logger.info(
                "Upstream area reported at selected outlet cell = %.2f km2. "
                "Difference (sum_cells - upstream_at_outlet) = %.2f km2 (%.2f%%).",
                uparea_at_outlet,
                delineated_area - uparea_at_outlet,
                (
                    (delineated_area - uparea_at_outlet) / uparea_at_outlet * 100.0
                    if uparea_at_outlet != 0
                    else np.nan
                ),
            )
            if ref_catchment_area is not None:
                area_error = (delineated_area - ref_catchment_area) / ref_catchment_area
                logger.info(
                    "Delineated basin area (sum of cell_area[basin>0]) = %.2f km2; "
                    "reference area = %.2f km2; error = %.2f%%",
                    delineated_area,
                    ref_catchment_area,
                    area_error * 100.0,
                )
                if abs(area_error) > max_error * 2:
                    with ErrorLogger(logger):
                        msg = f"Delineated basin area ({delineated_area:2f} km2) differs from reference area ({ref_catchment_area:2f} km2) by more than twice the max error {max_error*100:.2f}%. Adjust max_error or max_distance_cells."
                        if raise_on_sanity_check:
                            raise ValueError(msg)
                        logger.warning(msg)
                        return True
                    # warn if the two area measures disagree substantially
            else:
                logger.warning(
                    "No reference catchment area provided; skipping area consistency check."
                )
            if not np.isclose(delineated_area, uparea_at_outlet, rtol=0.02, atol=1e-6):
                with ErrorLogger(logger):
                    msg = f"Gauge ID {gauge_id}: " if gauge_id is not None else ""
                    msg += f"Sum of cell areas inside the basin ({delineated_area:2f} km2) differs from "
                    msg += f"upstream area at outlet ({uparea_at_outlet:2f} km2). Investigate flow-direction "
                    msg += "masking, nodata handling or area units."
                    if raise_on_sanity_check:
                        raise ValueError(msg)
                    logger.warning(msg)
                    return True
            return False
        except Exception as e:
            logger.exception(f"Sanity check failed: {e}")
            if raise_on_sanity_check:
                raise e
            return True

    def delineate_basin(
        self,
        gauge,
        stream_order=4,
        max_distance_cells=5,
        max_error=0.25,
        raise_on_sanity_check=True,
        upstream_area=None,
        mask_catchment: Optional[bool] = True,
        save_coords: Optional[bool] = True,
        gauge_opti_method: Optional[str] = "basinex",
        shape_folder: Optional[str] = None,
        raise_on_fallback: Optional[bool] = True,
    ):
        """Delineate the basin for a given lat and lon."""
        # Target area in km2 we want to match (can be adjusted/replaced by caller later)
        ref_catchment_area = gauge.area
        gauge_coords = (gauge.lat, gauge.lon)
        gauge_id = getattr(gauge, "gauge_id", getattr(gauge, "id", None))
        # Compute upstream area (in km2) using accuflux and cell areas
        if self.cell_area is None:
            self.compute_cell_area()
        if upstream_area is None:
            try:
                upstream_area = self.calc_upstream_area()
            except Exception:
                logger.exception("Failed to compute upstream area (accuflux).")
            if upstream_area is None:
                msg = "Could not calculate upstream area. Flow direction may be uninitialized."
                with ErrorLogger(logger):
                    raise ValueError(msg)

        gauge_str = f" (ID: {gauge_id})" if gauge_id is not None else ""
        logger.info(
            f"Delineating basin {gauge_str} for gauge coordinates {gauge_coords} and reference catchment area {ref_catchment_area} km2."
        )
        outlet_idx, error, gauge_lat, gauge_lon, distance_error = (
            self.get_best_gauge_coordinate(
                upstream_area=upstream_area,
                gauge_coords=gauge_coords,
                ref_catchment_area=ref_catchment_area,
                max_distance_cells=max_distance_cells,
                max_error=max_error,
                method=gauge_opti_method,
                shape_folder=shape_folder,
                gauge_id=gauge_id,
                raise_on_fallback=raise_on_fallback,
            )
        )
        outlet_linear_idx = np.ravel_multi_index(outlet_idx, self._fdir.shape)

        if ref_catchment_area is not None and error is not None:
            streams_mask = (upstream_area > ref_catchment_area * (1 - error - 1e-6)) & (
                upstream_area < ref_catchment_area * (1 + error + 1e-6)
            ).astype(bool)
        else:
            streams_mask = self._fdir.stream_order() >= stream_order
        try:
            basin = self._fdir.basins(
                idxs=np.array([outlet_linear_idx], dtype=np.int64),
                streams=streams_mask,
            )
        except Exception as e:
            logger.exception(f"pyflwdir.basins(idxs=...) failed for {outlet_idx}: {e}")
            try:
                all_basins = self._fdir.basins()
                basin_id = int(all_basins[outlet_idx])
                basin = np.where(all_basins == basin_id, basin_id, 0)
            except Exception as e2:
                logger.exception(f"Fallback basins() also failed: {e2}")
                return gauge

        catchment_mask = basin > 0
        logger.debug(
            f"mask statistics: min={basin.min()}, max={basin.max()}, all true? {np.all(catchment_mask)}"
        )

        uparea_at_outlet = (
            upstream_area[outlet_idx] if upstream_area is not None else np.nan
        )
        failed_sanity_check = self.delineation_sanity_check(
            catchment_mask,
            basin,
            uparea_at_outlet,
            ref_catchment_area,
            max_error,
            raise_on_sanity_check,
            gauge_id,
        )

        if np.all(catchment_mask):
            if stream_order > 1 and ref_catchment_area is None:
                logger.info("Trying again with stream_order %d", stream_order - 1)
                return self.delineate_basin(
                    gauge,
                    stream_order=stream_order - 1,
                    max_distance_cells=max_distance_cells,
                    max_error=max_error,
                    raise_on_sanity_check=raise_on_sanity_check,
                    upstream_area=upstream_area,
                    mask_catchment=mask_catchment,
                    save_coords=save_coords,
                    gauge_opti_method=gauge_opti_method,
                    shape_folder=shape_folder,
                    raise_on_fallback=raise_on_fallback,
                )
            logger.error("No catchment found for the given coordinates")
            return gauge

        if mask_catchment:
            self.catchment_mask = catchment_mask
            self.basin = basin
            try:
                fillv = self.VARIABLES["basin"]["_FillValue"]
                self.basin = np.where(self.catchment_mask, self.basin, fillv)
            except Exception:
                logger.debug("Could not set basin fill values")

        if not failed_sanity_check:
            if save_coords:
                self.save_coords(gauge_id, gauge_lat, gauge_lon)

            gauge.update(
                lat=gauge_lat,
                lon=gauge_lon,
                area=uparea_at_outlet,
                distance_error=distance_error,
                area_error=error,
            )
        return gauge

    def save_coords(self, gauge_id, gauge_lat, gauge_lon):
        """Save gauge coordinates."""
        if gauge_id is not None:
            self.gauge_ids.append(gauge_id)
        self.gauge_lats.append(gauge_lat)
        self.gauge_lons.append(gauge_lon)

    def write_basin_shape(self, out_dir, gauge_id=None, basin_mask=None):
        """Write a basin shapefile from a basin mask."""
        if basin_mask is None:
            basin_mask = self.catchment_mask
        if basin_mask is None or not np.any(basin_mask):
            return
        try:
            gdf = _vectorize_mask_to_gdf(
                basin_mask, self.transform, _shape_crs(self.latlon), value_name="basin"
            )
        except Exception as exc:
            logger.warning("Could not write basin shapefile: %s", exc)
            return
        output_dir = Path(out_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        suffix = f"_{gauge_id}" if gauge_id is not None else ""
        out_path = output_dir / f"basin{suffix}.shp"
        try:
            gdf.to_file(out_path)
            logger.info("Wrote basin shapefile to %s", out_path)
        except Exception as exc:
            logger.warning("Failed to write basin shapefile %s: %s", out_path, exc)

    def get_upscaling_factor(self, max_resolution=False, l1=False, l2=True):
        """Create upscaling factor."""
        return get_upscaling_factor(
            self.resolutions, max_resolution=max_resolution, l1=l1, l2=l2
        )

    def upscale(self, var):
        """Upscale flow direction to l1_resolution if that is int multipe of data resolution."""
        factor, upscaled_resolution = get_upscaling_factor(self.resolutions, l1=True)

        if factor == 1:
            self.get_facc()
            return
        # if we upscale the do_upscale flag should be true
        self.do_upscale = True
        logger.info(
            f"Upscaling flow direction to {upscaled_resolution} with the fator {factor}."
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
        self.cell_area = (
            None  # reset cell area to be recalculated at new resolution when needed
        )
        self.is_upscaled = True
        self.upscaled_resolution = upscaled_resolution
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
        variables=None,
    ):
        """Write the produced data to one or multiple files."""
        data_vars = {}
        out_path = Path(out_path)
        if not out_path.is_dir():
            out_path.mkdir(parents=True, exist_ok=True)
        selected_vars = _normalize_output_vars(variables)
        lat_slice_idx, lon_slice_idx = None, None
        lat_slice, lon_slice = None, None
        if cut_by_basin:
            lat_slice_idx, lon_slice_idx = cut_to_filled_area(
                ds=self.ds,
                resolutions=self.resolutions,
                catchment_mask=self.catchment_mask,
                buffer=buffer,
            )
        else:
            lat_slice, lon_slice = slice(84, -56), slice(None)

        for var_name in (v for v in self.VARIABLES if v in selected_vars):
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
            if not data_vars:
                msg = "No data variables available to write."
                with ErrorLogger(logger):
                    raise ValueError(msg)
            ds = self.write_basin_id_file(data_vars, frame, out_path)
            # use basin_id to create a mask file
            if "basin" in ds.data_vars:
                self.write_mask_file(ds, mask_file)
                if self.gauge_ids:
                    logger.info("Writing gauges file.")
                    # create empty ds with mask l0 extend and fill the data_var called data with -9999 values
                    id_da = xr.DataArray(
                        np.full(ds.basin.shape, -9999, dtype=int),
                        coords={"lat": ds.lat, "lon": ds.lon},
                        dims=["lat", "lon"],
                    )
                    id_ds = id_da.to_dataset(name="idgauges")
                    gauge_ids_numeric = _to_numeric_gauge_ids(
                        self.gauge_ids,
                        context="idgauges",
                    )
                    id_ds = write_gauge_id(
                        ds=id_ds,
                        id=gauge_ids_numeric,
                        lat=self.gauge_lats,
                        lon=self.gauge_lons,
                        data_var="idgauges",
                    )
                    write_xarray_to_file(id_ds, out_path / "idgauges.nc", "idgauges")
                    write_xarray_to_file(
                        id_ds,
                        out_path / "idgauges.asc",
                        "idgauges",
                        resolution=self.upscaled_resolution,
                    )
                else:
                    logger.info("No gauges to write, skipping gauges file.")
            else:
                logger.info("No basin variable written; skipping mask/gauge files.")

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
                        "dtype": get_dtype(data_var[var_name]),
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
                    f"nodata_value {self.VARIABLES[var_name]['_FillValue']}\n"
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
        lon, lat = self._coords_l1()
        logger.debug(
            f"lon_min {np.min(lon):.3f}, lon_max {np.max(lon):.3f}, resulution: {self.resolutions.l1}"
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
        logger.debug(f"data_vars: {data_vars}")
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
                    "dtype": get_dtype(ds[var_name]),
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
        if not ascending and centers[0] < centers[-1]:
            centers = centers[::-1]
        return centers

    def upscale_mask_with_correct_coords(
        self,
        da: xr.DataArray,
        factor: Optional[int] = None,
        lon_name: str = "lon",
        lat_name: str = "lat",
    ) -> xr.DataArray:
        """
        Coarsen a 2D mask-like field by integer factor and assign correct coarse coords.

        so that coarse *edges* equal fine *edges* of the cropped window.
        """
        if factor is None:
            factor, upscaled_resolution = get_upscaling_factor(
                self.resolutions, l2=True
            )
        if factor < 1:
            msg = "factor must be >= 1"
            with ErrorLogger(logger):
                raise ValueError(msg)

        logger.info(f"Upscaling mask with factor {factor} to {upscaled_resolution}.")

        # 1) coarsen over lon/lat windows
        kx = ky = int(factor)
        coarsen_map = {}
        if lon_name in da.dims:
            coarsen_map[lon_name] = kx
        if lat_name in da.dims:
            coarsen_map[lat_name] = ky

        # Treat only explicit mask=1 (or True) as land; ignore fill values and NaNs.
        if da.dtype == bool:
            cond = da
        else:
            fillv = da.attrs.get("_FillValue", da.encoding.get("_FillValue"))
            da_clean = da
            if fillv is not None:
                da_clean = da.where(da != fillv)
            cond = da_clean == 1
        out = cond.coarsen(coarsen_map, boundary="trim").any().astype("int8")

        # 2) compute correct coarse coordinates from fine edges
        lon_f = da[lon_name].values
        lat_f = da[lat_name].values

        lon_edges = self._cell_edges(lon_f)
        lat_edges = self._cell_edges(lat_f)

        asc_lon = lon_f[0] < lon_f[-1]
        asc_lat = lat_f[0] < lat_f[-1]
        logger.debug(f"asc_lon: {asc_lon}, asc_lat: {asc_lat}")

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
        asc_lat_coarse = lat_coarse[0] < lat_coarse[-1]
        if asc_lat != asc_lat_coarse:
            logger.warning(
                "Coarse lat coordinate ascending order does not match fine grid; check calculations."
            )

        out = out.assign_coords({lon_name: lon_coarse, lat_name: lat_coarse})
        out.name = "mask_L2"

        # 3) (optional) log edges for verification
        try:
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
        except IndexError:
            logger.debug("Could not log coarse edges for verification.")
            logger.debug(f"lon_coarse: {out[lon_name].values}")
            logger.debug(f"lat_coarse: {out[lat_name].values}")
        return out

    def write_mask_file(self, ds, mask_file):
        """Write basin mask to specified path."""
        if mask_file is not None:
            logger.info("Writing mask file")
            # name the variable mask
            mask = np.where(
                ds.basin > 0, 1, 0
            )  # if self.catchment_mask is None else self.catchment_mask
            mask_file = Path(mask_file)
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
            logger.debug(
                f"Created mask dataarray with shape {mask_da.shape} and stats min {mask_da.min().item()}, max {mask_da.max().item()}"
            )
            mask_ds = xr.Dataset({"land_mask": mask_da, "mask": mask_da})
            mask_upscaled = None
            if self.do_upscale:
                mask_upscaled = mask_da
            elif self.resolutions.l2 is not None:
                mask_upscaled = self.upscale_mask_with_correct_coords(mask_da)

            if mask_upscaled is not None:
                mask_upscaled = mask_upscaled.rename({"lat": "lat_l2", "lon": "lon_l2"})
                mask_ds["mask_l2"] = mask_upscaled
            dims = set(mask_ds.dims)
            all_coords = set(mask_ds.coords)
            dim_coords = all_coords & dims  # intersection
            for var in dim_coords:
                bounds_name = f"{var}_bnds"
                try:
                    mask_ds.coords[bounds_name] = generate_bounds(mask_ds[var])
                    mask_ds[var].attrs["bounds"] = bounds_name
                except IndexError:
                    logger.info(f"Could not generate bounds for coord {var}")
            encoding = {
                v: {"zlib": True, "complevel": 4, "shuffle": True, **NC_ENCODE_MASK}
                for v in mask_ds.data_vars
            }
            write_xarray_to_file(mask_ds, mask_file, encoding=encoding)
            logger.info(f"Mask file has been written to {mask_file}")
        else:
            logger.info("No mask file path specified.")

    def cut_to_filled_area(
        self, buffer=0, repeat=False, raise_on_l2_alignment_mismatch=False
    ):
        """Create lat and lon slices to cut the data to the filled area."""
        return cut_to_filled_area(
            ds=self.ds,
            resolutions=self.resolutions,
            catchment_mask=self.catchment_mask,
            buffer=buffer,
            repeat=repeat,
            raise_on_l2_alignment_mismatch=raise_on_l2_alignment_mismatch,
        )


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
    ds2["basin"] = ds2["basin"] + ds1["basin"].max().item() + 1

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


def _is_list_of_float_tuples(a):
    return isinstance(a, list) and all(
        isinstance(x, tuple) and len(x) == 2 and all(isinstance(v, float) for v in x)
        for x in a
    )


@log_arguments()
def create_catchment(  # noqa: PLR0913, PLR0912, PLR0915
    input_file,
    output_path,
    var_name,
    var,
    ftype,
    gauge_coords=None,
    coordinate_slices=None,
    mask_file=None,
    resolutions: Resolution = None,
    frame=1,
    upscale=False,
    latlon=True,
    available_mem=None,
    ref_catchment_area=None,
    max_distance_cells=5,
    max_error=0.1,
    gauge_ids=None,
    ncpus=1,
    output_vars=None,
    gauge_opti_method="basinex",
    shape_folder=None,
    gauge_info_file="gauges_info",
    raise_on_fallback=True,
):
    """Create file containing catchment ids, flowdirection and upstream area from dem or flow direction."""
    logger.info(
        f"Creating catchment file for {var_name} using {var} and {ftype} from {input_file}"
    )
    output_path = Path(output_path)
    if _is_list_of_float_tuples(gauge_coords) and len(gauge_coords) == 1:
        gauge_coords = gauge_coords[0]
    if isinstance(ref_catchment_area, list) and len(ref_catchment_area) == 1:
        ref_catchment_area = ref_catchment_area[0]
        if isinstance(ref_catchment_area, list):
            if len(ref_catchment_area) != 1:
                msg = "If gauge_coords is a list of one tuple, ref_catchment_area (if provided) must be a single value or a list of one value."
                raise ValueError(msg)
            ref_catchment_area = ref_catchment_area[0]
    if resolutions is None:
        resolutions = Resolution()
    if var not in {"fdir", "dem"}:
        with ErrorLogger(logger):
            msg = f"Unexpected value for var={var}, must be 'fdir' or 'dem'"
            raise ValueError(msg)
    output_vars = _normalize_output_vars(output_vars)
    chunking = available_mem is not None
    with get_xarray_ds_from_file(
        input_file,
        var_name,
        normalize_latlon_coords=True,
        force_decending_y=True,
        available_mem_gib=available_mem,
        chunking=chunking,
    ) as input_ds:
        # transform
        transform = get_transformation_matrix_nc(input_ds, var_name)

        logger.info(transform)
        if coordinate_slices is None:
            if shape_folder:
                bounds = _shape_bounds_from_folder(
                    shape_folder, gauge_ids, latlon=latlon
                )
                logger.debug(f"Extracted bounds from shape_folder: {bounds}")
                if bounds is not None:
                    slices = _slices_from_bounds(bounds, input_ds.lat.data)
                    logger.debug(f"Calculated slices from bounds: {slices}")
                    if slices is not None:
                        coordinate_slices = slices
                        logger.info(
                            f"Updated coordinate_slices based on shape_folder: {coordinate_slices}"
                        )
            else:
                logger.debug("No shape_folder provided, using no coordinate slices.")
                coordinate_slices = {"lat": slice(None, None), "lon": slice(None, None)}
        logger.debug(f"Using coordinate_slices: {coordinate_slices}")

        def _compute_requested_outputs(catchment):
            needs_uparea_grid = "uparea_grid" in output_vars
            needs_upgrid = "upgrid" in output_vars
            needs_grdare = "grdare" in output_vars
            needs_basin = "basin" in output_vars

            if resolutions.l1 is not None and upscale:
                catchment.upscale(var)
            elif needs_uparea_grid:
                catchment.get_facc()

            if needs_basin and catchment.basin is None:
                catchment.get_basins()

            if needs_grdare:
                catchment.get_grid_area()
            elif needs_upgrid:
                catchment.get_upstream_area()

        if gauge_coords is None and is_data_global(input_ds, coordinate_slices):
            logger.info("Creating global basin id file...")
            if "basin" in output_vars:
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
                    resolutions=resolutions,
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
                    resolutions=resolutions,
                    upscale=upscale,
                )
                catchments = [global_catchments, global_catchments_shifted]

                for c in catchments:
                    _compute_requested_outputs(c)
                    c.write(
                        output_path,
                        single_file=True,
                        frame=frame,
                        mask_file=mask_file,
                        variables=output_vars,
                    )
                # add paths to the temp files
                temp_file1 = Path(output_path, "hydro1.nc")
                temp_file2 = Path(output_path, "hydro2.nc")
                logger.info("Merging catchment files")
                merge_catchment(
                    temp_file1,
                    temp_file2,
                    Path(output_path, "basin_ids.nc"),
                )
                # remove the temporary files
                temp_file1.unlink()
                temp_file2.unlink()
            else:
                global_catchments = Catchment(
                    ds=input_ds,
                    var_name=var_name,
                    var=var,
                    ftype=ftype,
                    transform=transform,
                    latlon=latlon,
                    out_var_name="basin_ids.nc",
                    do_shift=False,
                    resolutions=resolutions,
                    upscale=upscale,
                )
                _compute_requested_outputs(global_catchments)
                global_catchments.write(
                    output_path,
                    single_file=True,
                    frame=frame,
                    mask_file=mask_file,
                    variables=output_vars,
                )
            return
        input_ds_sliced = input_ds.sel(
            lat=coordinate_slices["lat"], lon=coordinate_slices["lon"]
        )
        logger.info("Cropped input dataset:")
        logger.info(
            f"        lat {input_ds_sliced.lat.data[0]}, {input_ds_sliced.lat.data[-1]}"
        )
        logger.info(
            f"        lon {input_ds_sliced.lon.data[0]}, {input_ds_sliced.lon.data[-1]}"
        )
        if gauge_coords is not None and isinstance(gauge_coords, tuple):
            logger.info(f"Creating catchment for gauge coordinates {gauge_coords}")
            c = Catchment(
                ds=input_ds_sliced,
                var_name=var_name,
                var=var,
                ftype=ftype,
                transform=transform,
                latlon=latlon,
                out_var_name="basin_ids.nc",
                do_shift=False,
                resolutions=resolutions,
                upscale=upscale,
            )
            single_ref_area = (
                ref_catchment_area[0]
                if isinstance(ref_catchment_area, list) and ref_catchment_area
                else ref_catchment_area
            )
            gauge = Gauge(
                gauge_id=gauge_ids if not isinstance(gauge_ids, list) else gauge_ids[0],
                lat=gauge_coords[0],
                lon=gauge_coords[1],
                area=single_ref_area,
            )
            gauge = c.delineate_basin(
                gauge,
                max_distance_cells=max_distance_cells,
                max_error=max_error,
                gauge_opti_method=gauge_opti_method,
                shape_folder=shape_folder,
                raise_on_fallback=raise_on_fallback,
            )
            l0_shape_gdf = None
            if upscale and c.catchment_mask is not None:
                write_gauges_out(
                    gauge, output_path / f"{gauge_info_file}_{resolutions.l0}"
                )
                try:
                    l0_shape_gdf = _vectorize_mask_to_gdf(
                        c.catchment_mask,
                        c.transform,
                        _shape_crs(c.latlon),
                        value_name="basin",
                    )
                except Exception as exc:
                    logger.warning("Could not build L0 basin shape: %s", exc)
            else:
                write_gauges_out(gauge, output_path / gauge_info_file)
            c.write_basin_shape(
                output_path / "shapes",
                gauge_id=gauge_ids if not isinstance(gauge_ids, list) else gauge_ids[0],
            )

            _compute_requested_outputs(c)
            if upscale and c.is_upscaled and l0_shape_gdf is not None and c.gauge_lats:
                logger.info("Applying L1 correction using L0 basin shape.")
                new_coords = c.correct_gauge_location_l1_from_shape(
                    l0_shape_gdf,
                    (c.gauge_lats[-1], c.gauge_lons[-1]),
                    max_distance_cells=max_distance_cells,
                    ref_catchment_area=ref_catchment_area,
                )
                if new_coords is not None:
                    c.update_gauge_coords(
                        gauge_ids if not isinstance(gauge_ids, list) else gauge_ids[0],
                        new_coords[0],
                        new_coords[1],
                    )
                    gauge.update(lat=new_coords[0], lon=new_coords[1])
                write_gauges_out(
                    gauge, output_path / f"{gauge_info_file}_{c.upscaled_resolution}"
                )
            c.write(
                output_path,
                single_file=True,
                cut_by_basin=True,
                mask_file=mask_file,
                frame=frame,
                buffer=frame,
                variables=output_vars,
            )
            return
        logger.info("Creating basin id file for region.")

        c = Catchment(
            ds=input_ds_sliced,
            var_name=var_name,
            var=var,
            ftype=ftype,
            transform=transform,
            latlon=latlon,
            out_var_name="basin_ids.nc",
            do_shift=False,
            resolutions=resolutions,
            upscale=upscale,
        )
        gauge_infos = None
        gauges = []
        if (
            _is_list_of_float_tuples(gauge_coords)
            and isinstance(gauge_ids, list)
            and len(gauge_coords) == len(gauge_ids)
        ):
            logger.info(f"Creating catchments for gauge coordinates {gauge_coords}")
            upstream_area = c.calc_upstream_area()

            lon = get_coord_values(input_ds_sliced, lon=True)
            lat = get_coord_values(input_ds_sliced, lat=True)

            def _process_gauge(i, gc, lon, lat):
                ref_area = (
                    ref_catchment_area[i]
                    if isinstance(ref_catchment_area, list)
                    else ref_catchment_area
                )
                if not (
                    lon.min() <= gc[1] <= lon.max() and lat.min() <= gc[0] <= lat.max()
                ):
                    logger.warning(
                        f"Gauge coordinate {gc} is outside the domain lon: [{lon.min()}, {lon.max()}], lat: [{lat.min()}, {lat.max()}]"
                    )
                    return None

                outlet_idx, error, gauge_lat, gauge_lon, distance_error = (
                    c.get_best_gauge_coordinate(
                        upstream_area=upstream_area,
                        gauge_coords=gc,
                        ref_catchment_area=ref_area,
                        max_distance_cells=max_distance_cells,
                        max_error=max_error,
                        method=gauge_opti_method,
                        shape_folder=shape_folder,
                        gauge_id=gauge_ids[i],
                        raise_on_fallback=raise_on_fallback,
                    )
                )
                return {
                    "gauge_id": gauge_ids[i],
                    "gauge_lat": gauge_lat,
                    "gauge_lon": gauge_lon,
                    "lat_old": gc[0],
                    "lon_old": gc[1],
                    "area_old": ref_area,
                    "outlet_idx": outlet_idx,
                    "error": error,
                    "distance_error": distance_error,
                    "ref_area": ref_area,
                }

            gauge_infos = Parallel(n_jobs=ncpus, prefer="threads")(
                delayed(_process_gauge)(i, gc, lon, lat)
                for i, gc in enumerate(gauge_coords)
            )

            gauge_infos = [gi for gi in gauge_infos if gi is not None]
            if not gauge_infos:
                logger.warning("No valid gauge coordinates found inside domain.")
            else:
                logger.info(
                    f"Found {len(gauge_infos)} valid gauge coordinates inside domain."
                )
                shape_dir = Path(output_path) / "shapes"
                outlet_idxs = [gi["outlet_idx"] for gi in gauge_infos]
                outlet_linear = np.array(
                    [np.ravel_multi_index(idx, c._fdir.shape) for idx in outlet_idxs],
                    dtype=np.int64,
                )
                # combine all river masks
                streams_mask = None
                # logger.info("Creating stream masks for all gauges.")
                # for gi in gauge_infos:
                #     if gi["ref_area"] is None or gi["error"] is None:
                #         continue
                # if streams_mask is None:
                #     low = gi["ref_area"] * (1 - gi["error"] - 1e-6)
                #     high = gi["ref_area"] * (1 + gi["error"] + 1e-6)
                #     streams_mask = np.asarray(
                #         (upstream_area > low) & (upstream_area < high), dtype=bool
                #     )
                # else:
                # low = gi["ref_area"] * (1 - gi["error"] - 1e-6)
                # high = gi["ref_area"] * (1 + gi["error"] + 1e-6)
                # streams_mask |= np.asarray(
                #     (upstream_area > low) & (upstream_area < high), dtype=bool
                # )
                logger.info("Delineating basins for all gauges.")
                # if streams_mask is None or not np.any(streams_mask):
                # logger.warning("No river mask found for any gauge.")
                # streams_mask = np.asarray(c._fdir.stream_order() >= 4, dtype=bool)
                try:
                    basins = c._fdir.basins(
                        idxs=outlet_linear,
                        streams=streams_mask,
                    )
                except Exception as exc:
                    logger.exception("pyflwdir.basins(idxs=...) failed: %s", exc)
                    with ErrorLogger(logger):
                        raise exc
                logger.info("Performing sanity checks for all delineated basins.")
                gauges = []
                for gi in gauge_infos:
                    outlet_idx = gi["outlet_idx"]
                    basin_id = int(basins[outlet_idx])
                    if basin_id == 0:
                        logger.warning(
                            "No basin id found for gauge_id %s at outlet %s",
                            gi["gauge_id"],
                            outlet_idx,
                        )
                        continue
                    catchment_mask = basins == basin_id
                    uparea_at_outlet = (
                        upstream_area[outlet_idx]
                        if upstream_area is not None
                        else np.nan
                    )
                    failed = c.delineation_sanity_check(
                        catchment_mask,
                        basins,
                        uparea_at_outlet,
                        gi["ref_area"],
                        max_error,
                        False,
                        gi["gauge_id"],
                    )
                    if failed:
                        continue
                    if upscale:
                        try:
                            gi["l0_shape_gdf"] = _vectorize_mask_to_gdf(
                                catchment_mask,
                                c.transform,
                                _shape_crs(c.latlon),
                                value_name="basin",
                            )
                        except Exception as exc:
                            logger.warning(
                                "Could not build L0 basin shape for gauge %s: %s",
                                gi["gauge_id"],
                                exc,
                            )
                            gi["l0_shape_gdf"] = None
                    c.save_coords(gi["gauge_id"], gi["gauge_lat"], gi["gauge_lon"])
                    gauge = Gauge(
                        gauge_id=gi["gauge_id"],
                        lat=gi["lat_old"],
                        lon=gi["lon_old"],
                        area=gi["area_old"],
                    )
                    gauge.update(
                        lat=gi["gauge_lat"],
                        lon=gi["gauge_lon"],
                        area=uparea_at_outlet,
                        distance_error=gi["distance_error"],
                        area_error=gi["error"],
                    )
                    gauges.append(gauge)
                    c.write_basin_shape(
                        shape_dir,
                        gauge_id=gi["gauge_id"],
                        basin_mask=catchment_mask,
                    )
                if upscale:
                    write_gauges_out(
                        gauges,
                        output_path / f"{gauge_info_file}_{resolutions.l0}",
                    )
                else:
                    write_gauges_out(gauges, output_path / gauge_info_file)

        _compute_requested_outputs(c)
        if upscale and c.is_upscaled and gauges:
            logger.info("Applying L1 correction for %d gauges.", len(gauge_infos))
            for gauge in gauges:
                gi = next(
                    (
                        item
                        for item in gauge_infos
                        if item["gauge_id"] == gauge.gauge_id
                    ),
                    None,
                )
                if gi is None:
                    continue
                l0_shape_gdf = gi.get("l0_shape_gdf")
                if l0_shape_gdf is None:
                    continue
                new_coords = c.correct_gauge_location_l1_from_shape(
                    l0_shape_gdf,
                    (gauge.lat, gauge.lon),
                    max_distance_cells=max_distance_cells,
                    ref_catchment_area=gi.get("ref_area"),
                )
                if new_coords is not None:
                    c.update_gauge_coords(
                        gi["gauge_id"],
                        new_coords[0],
                        new_coords[1],
                    )
                    gauge.update(lat=new_coords[0], lon=new_coords[1])
            write_gauges_out(
                gauges, output_path / f"{gauge_info_file}_{c.upscaled_resolution}"
            )
        c.write(
            output_path,
            single_file=True,
            mask_file=mask_file,
            frame=frame,
            variables=output_vars,
        )
