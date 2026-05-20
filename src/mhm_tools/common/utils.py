"""Utility helpers."""

import logging
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from mhm_tools.common.file_handler import get_coord_values, get_xarray_ds_from_file
from mhm_tools.common.logger import ErrorLogger
from mhm_tools.common.netcdf import generate_bounds

logger = logging.getLogger(__name__)


def dict_to_multiline_string(d: dict, spacing: int = 12) -> str:
    r"""
    Convert a dictionary into a formatted multiline string.

    Example:
        >>> dict_to_multiline_string({'a': 'b', 'c': 'd'})
        'a           b\nc           d'
    """
    lines = []
    for k, v in d.items():
        lines.append(f"{k!s:<{spacing}}{v}")
    return "\n".join(lines)


def pretty_print_df(df: pd.DataFrame, max_col_width: int = 30) -> None:
    """Pretty-print a DataFrame as an ASCII table with simple truncation.

    Numbers are right-aligned, other columns are left-aligned. Cells longer than
    max_col_width are truncated with an ellipsis.
    """
    if df.empty:
        logger.info("There are no results to display.")
        return

    def is_numeric(col: pd.Series) -> bool:
        return pd.api.types.is_numeric_dtype(col)

    def fmt_cell(val: object, width: int, right: bool) -> str:
        s = ""
        if not pd.isna(val):
            try:
                val = float(val)
                if val < 10:
                    s = f"{val:.1f}"
                elif val < 1:
                    s = f"{val:.2f}"
                elif val < 0.1:
                    s = f"{val:.3f}"
                elif val < 0.01:
                    s = f"{val:.4f}"
                else:
                    s = f"{val:.0f}"
            except ValueError:
                s = str(val)
        else:
            s = "NaN"
        if len(s) > width:
            s = s[: max(1, width - 1)] + "…"
        return s.rjust(width) if right else s.ljust(width)

    headers = list(df.columns)
    widths = []
    aligns_right = []
    for h in headers:
        col = df[h]
        right = is_numeric(col)
        aligns_right.append(right)
        max_len = max(len(str(h)), *(len(str(x)) for x in col.fillna("")))
        widths.append(min(max_col_width, max_len))

    def sep() -> str:
        return "+" + "+".join("-" * (w + 2) for w in widths) + "+"

    # Header
    out_string = "\n"
    out_string += sep() + "\n"
    header_cells = [" " + fmt_cell(h, w, False) + " " for h, w in zip(headers, widths)]
    out_string += "|" + "|".join(header_cells) + "|\n"
    out_string += sep() + "\n"

    # Rows
    for _, row in df.iterrows():
        cells = []
        for h, w, right in zip(headers, widths, aligns_right):
            cells.append(" " + fmt_cell(row[h], w, right) + " ")
        out_string += "|" + "|".join(cells) + "|\n"
    out_string += sep() + "\n"

    logger.info(out_string)


def coord_to_index(ds, lat, lon):
    """Map latitude/longitude or indices to integer grid indices."""
    if "lat" not in ds.coords or "lon" not in ds.coords:
        msg = "Dataset is missing latitude/longitude coordinates."
        with ErrorLogger(logger):
            raise ValueError(msg)
    lat_vals = ds.lat.data
    lon_vals = ds.lon.data

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
        msg = "Could not map given coordinates to valid indices within dataset bounds."
        with ErrorLogger(logger):
            raise ValueError(msg)
    i = int(np.clip(i, 0, len(lat_vals) - 1))
    j = int(np.clip(j, 0, len(lon_vals) - 1))

    return i, j


def distance_100m_units(di, dj, l0_resolution, lat_deg=None, latlon=False):
    """Convert index deltas to distance in ~100 m units using l0_resolution."""
    res = float(abs(l0_resolution))
    if latlon or lat_deg is not None:
        if lat_deg is None:
            lat_deg = 0.0
        # approximate meters per degree
        meters_per_deg_lat = 111_132.92
        dy_m = meters_per_deg_lat * res
        # Not used since burek assumes square cell sizes:
        # lat_rad = np.deg2rad(lat_deg)
        # meters_per_deg_lon = 111_320.0 * np.cos(lat_rad)
        # dx_m = meters_per_deg_lon * res
        dx_m = dy_m
    else:
        # assume resolution already in meters for projected grids
        dy_m = res
        dx_m = res
    return np.sqrt((di * dy_m) ** 2 + (dj * dx_m) ** 2) / 100.0


def find_best_gauge_location_by_area(  # noqa: PLR0915
    ds: xr.Dataset,
    upstream_area,
    gauge_coords,
    ref_catchment_area,
    resolutions,
    max_distance_cells=5,
    max_error=0.25,
    recursion=False,
    method="basinex",
    raise_on_fallback=True,
    latlon=False,
):
    """Find best gauge location given reference gauge location, refernce cathcment area and allowed area and value deviation."""
    # Determine whether gauge_coords are lat/lon floats or array indices
    lat_vals = ds.lat.data
    lon_vals = ds.lon.data
    gi, gj = coord_to_index(ds, gauge_coords[0], gauge_coords[1])

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

    # Extract subgrid around the gauge
    sub = upstream_area[i_min : i_max + 1, j_min : j_max + 1]

    # If subgrid is empty fallback to whole domain
    if sub.size == 0:
        logger.warning("Search bbox empty, falling back to full-domain search")
        sub = upstream_area
        i_min, j_min = 0, 0
    lat_deg = float(ds.lat.data[gi]) if latlon else None
    size = float(ref_catchment_area)
    if size <= 0:
        msg = f"Reference catchment area must be > 0 for burek, got {size}."
        with ErrorLogger(logger):
            raise ValueError(msg)
    # Search for candidate cells whose upstream area matches ref_catchment_area
    if method == "basinex":
        logger.info("Correcting gauge location using basinex method")
        # based on implementation in basinex https://git.ufz.de/schaefed/basin-extractor/-/blame/master/lib/gauges.py?ref_type=heads#L42
        if size <= 0:
            msg = f"Reference catchment area must be > 0 for basinex, got {size}."
            with ErrorLogger(logger):
                raise ValueError(msg)

        sub_error = np.abs(sub - size) / size
        finite_mask = np.isfinite(sub_error)
        within_tol = finite_mask & (sub_error <= max_error)
        if np.any(within_tol):
            min_error = float(np.min(sub_error[within_tol]))
            candidates = np.where(within_tol & np.isclose(sub_error, min_error))
        else:
            candidates = (np.array([], dtype=int), np.array([], dtype=int))

        best_coord = None
        if len(candidates[0]) > 0:
            # convert sub indices to global indices
            cand_i = candidates[0] + i_min
            cand_j = candidates[1] + j_min
            # choose the candidate nearest to the gauge index
            d2 = (cand_i - gi) ** 2 + (cand_j - gj) ** 2
            k = int(np.argmin(d2))
            # if there is more than one candidate with the same distance, choose the one with the smallest area error
            if np.sum(d2 == d2[k]) > 1:
                logger.warning(
                    "Multiple candidates with the same distance to gauge found. Choosing the one with smallest area error."
                )
                error_cand = np.abs(sub[candidates] - size)
                k = int(np.argmin(error_cand))
                if np.sum(error_cand == error_cand[k]) > 1:
                    logger.warning(
                        "Multiple candidates with the same area error and same distance to gauge found. The selected candidate can not be uniquely identified."
                    )
            best_coord = (int(cand_i[k]), int(cand_j[k]))
            error = sub_error[candidates[0][k], candidates[1][k]]
            logger.info(
                f"Selected outlet candidate {best_coord} with upstream area {upstream_area[best_coord]} km2 (tolerance {error:.3f})"
            )
            distanance_100m = distance_100m_units(
                cand_i[k] - gi,
                cand_j[k] - gj,
                l0_resolution=resolutions.l0,
                lat_deg=lat_deg,
                latlon=latlon,
            )
            return (
                best_coord,
                np.abs(1 - sub[candidates[0][k], candidates[1][k]] / size),
                distanance_100m,
            )
    elif method == "burek":
        logger.info("Correcting gauge location using burek method")
        # based on Burek et. al. 2023 https://essd.copernicus.org/articles/15/5617/2023/
        # based on Lehner 2012 Derivation of watershed boundaries for GRDC gauging stations based on the HydroSHEDS drainage network - Technical Report prepared for the GRDC
        # implemented https://github.com/iiasa/CWATM_grdc_calibration_stations/blob/78979cbac8f8685d8dbc5330dba6f40a929716f4/scripts/1_findMeritcoord.py#L335
        ratio = np.full(sub.shape, np.nan, dtype=float)
        valid_sub = np.isfinite(sub) & (sub > 0)
        ratio[valid_sub] = np.where(
            sub[valid_sub] > size, size / sub[valid_sub], sub[valid_sub] / size
        )
        candidates_error = 1 - ratio
        candidates_indices = np.where(
            np.isfinite(candidates_error) & (candidates_error <= max_error)
        )
        if len(candidates_indices[0]) > 0:
            cand_i = candidates_indices[0] + i_min
            cand_j = candidates_indices[1] + j_min
            di = cand_i - gi
            dj = cand_j - gj
            candidates_distance = distance_100m_units(
                di, dj, l0_resolution=resolutions.l0, lat_deg=lat_deg
            )
            # change error to percent
            candidates_error = (
                100 * candidates_error[candidates_indices]
            )  # 100 * np.abs(1 - ups[y, x] / upsreal)
            burek_metric = candidates_error + 2 * candidates_distance
            k = int(np.argmin(burek_metric))
            if np.sum(burek_metric == burek_metric[k]) > 1:
                logger.warning(
                    "Multiple candidates with the same Burek metric found. The selected candidate can not be uniquely identified."
                )
            best_coord = (int(cand_i[k]), int(cand_j[k]))
            error = candidates_error[k]
            return best_coord, error / 100, candidates_distance[k]
    else:
        msg = f"Unknown method: {method}. Valid options are 'basinex' and 'burek'."
        with ErrorLogger(logger):
            raise ValueError(msg)

    if not recursion:
        logger.warning(
            f"No suitable outlet candidate found within {max_distance_cells} cells and {max_error*100:.2f}% area error. Retrying with doubled search radius."
        )
        return find_best_gauge_location_by_area(
            ds,
            upstream_area,
            gauge_coords,
            ref_catchment_area,
            resolutions,
            max_distance_cells=max_distance_cells * 2,
            max_error=max_error,
            recursion=True,
            method=method,
            raise_on_fallback=raise_on_fallback,
            latlon=latlon,
        )
    if raise_on_fallback:
        msg = (
            f"No suitable outlet candidate found within {max_distance_cells} cells and {max_error*100:.2f}% area error. "
            "Consider increasing max_distance_cells or max_error."
        )
        with ErrorLogger(logger):
            raise ValueError(msg)

    # fallback: pick the cell in bbox with upstream area closest to target
    flat = np.abs(sub - size)
    idx = int(np.argmin(flat))
    ri, rj = np.unravel_index(idx, sub.shape)
    best_coord = (ri + i_min, rj + j_min)
    logger.info(
        f"The selected outlet candidate is {best_coord} with upstream area {upstream_area[best_coord]} km2 resulting in error {(ref_catchment_area - upstream_area[best_coord]) / ref_catchment_area:.3f}."
    )
    distance_100m = distance_100m_units(
        best_coord[0] - gi,
        best_coord[1] - gj,
        l0_resolution=resolutions.l0,
        lat_deg=lat_deg,
    )
    return best_coord, abs(upstream_area[best_coord] - size) / size, distance_100m


class Resolution:
    """Class to hold resolution information."""

    def __init__(
        self,
        l1=None,
        l11=None,
        l2=None,
        l2_file=None,
        l0=None,
        l0_resolution=None,
        l1_resolution=None,
        l11_resolution=None,
        l2_resolution=None,
        raise_on_missmatch=True,
    ):
        """Initialize the Resolution class."""
        self.l0 = l0 if l0 is not None else l0_resolution
        self.l1 = l1 if l1 is not None else l1_resolution
        self.l11 = l11 if l11 is not None else l11_resolution
        self.l2 = l2 if l2 is not None else l2_resolution
        self.l2_file = l2_file
        if self.l2_file is not None:
            self.l2_file = Path(self.l2_file)
            if self.l2_file.is_dir():
                # get the first netcdf file in the directory
                nc_files = list(self.l2_file.rglob("*.nc"))
                if len(nc_files) == 0:
                    with ErrorLogger(logger):
                        msg = f"No netcdf files found in {self.l2_file}."
                        raise FileNotFoundError(msg)
                self.l2_file = nc_files[0]
            elif not self.l2_file.is_file():
                with ErrorLogger(logger):
                    msg = f"L2 file {self.l2_file} not found."
                    raise FileNotFoundError(msg)
            if self.l2_file.suffix == ".nc":
                with get_xarray_ds_from_file(self.l2_file) as ds:
                    lon = get_coord_values(ds, lon=True)
                    file_res = round(abs(lon[1] - lon[0]), 9)
                    if self.l2 is not None and abs(file_res - self.l2) > 1e-6:
                        msg = f"Provided l2_resolution {self.l2} differs from resolution derived from file {file_res}. Either provide the correct l2_resolution or remove it to use the resolution derived from the file."
                        if raise_on_missmatch:
                            with ErrorLogger(logger):
                                raise ValueError(msg)
                        logger.warning(msg)
                        self.l2 = file_res
                    elif self.l2 is None:
                        logger.info(
                            f"Derived l2_resolution {file_res} from {self.l2_file}"
                        )
                        self.l2 = file_res
            else:
                logger.error(
                    f"Unsupported file format for l2_file: {self.l2_file.suffix}"
                )
                self.l2_file = None

        self.l11 = self.l11 if self.l11 is not None else self.l1
        self.l2 = self.l2 if self.l2 is not None else self.l1

    @property
    def l0_resolution(self):
        """Backward-compatible alias for l0."""
        return self.l0

    @l0_resolution.setter
    def l0_resolution(self, value):
        self.l0 = value

    @property
    def l1_resolution(self):
        """Backward-compatible alias for l1."""
        return self.l1

    @l1_resolution.setter
    def l1_resolution(self, value):
        self.l1 = value

    @property
    def l11_resolution(self):
        """Backward-compatible alias for l11."""
        return self.l11

    @l11_resolution.setter
    def l11_resolution(self, value):
        self.l11 = value

    @property
    def l2_resolution(self):
        """Backward-compatible alias for l2."""
        return self.l2

    @l2_resolution.setter
    def l2_resolution(self, value):
        self.l2 = value

    def get_max_resolution(self):
        """Get the maximum resolution."""
        return max(
            r
            for r in [
                self.l1,
                self.l11,
                self.l2,
            ]
            if r is not None
        )


def align_bounds_to_l2(ds, resolutions, min_row, max_row, min_col, max_col):
    """Align the given bounds to the L2 grid."""
    if resolutions.l2_file is None:
        return min_row, max_row, min_col, max_col

    with get_xarray_ds_from_file(
        resolutions.l2_file,
        force_decending_y=True,
        normalize_latlon_coords=True,
    ) as ds_l2_file:
        l2_lon = get_coord_values(ds_l2_file, lon=True)
        l2_lat = get_coord_values(ds_l2_file, lat=True)
        if l2_lon is None or l2_lat is None:
            logger.warning(
                f"Could not get lon/lat from L2 file {resolutions.l2_file}, using raw values."
            )

    lon = ds.lon.values
    lat = ds.lat.values
    tol = resolutions.l0 / 2 + 1e-9

    # get lower and upper edges of mask lon/lat
    lon_bounds = generate_bounds(
        xr.DataArray(lon, dims=["lon"], coords={"lon": lon})
    ).values
    lat_bounds = generate_bounds(
        xr.DataArray(lat, dims=["lat"], coords={"lat": lat})
    ).values
    lon_lower_edges = np.minimum(lon_bounds[:, 0], lon_bounds[:, 1])
    lon_upper_edges = np.maximum(lon_bounds[:, 0], lon_bounds[:, 1])
    lat_lower_edges = np.minimum(lat_bounds[:, 0], lat_bounds[:, 1])
    lat_upper_edges = np.maximum(lat_bounds[:, 0], lat_bounds[:, 1])

    cur_lon_min = min(lon_lower_edges[min_col], lon_lower_edges[max_col])
    cur_lon_max = max(lon_upper_edges[min_col], lon_upper_edges[max_col])
    cur_lat_min = min(lat_lower_edges[min_row], lat_lower_edges[max_row])
    cur_lat_max = max(lat_upper_edges[min_row], lat_upper_edges[max_row])

    def _bound_to_grid(values, lower, upper):
        bounds = generate_bounds(
            xr.DataArray(values, dims=["coord"], coords={"coord": values})
        ).values
        lower_edges = np.minimum(bounds[:, 0], bounds[:, 1])
        upper_edges = np.maximum(bounds[:, 0], bounds[:, 1])
        lower_vals = lower_edges[lower_edges <= lower]
        upper_vals = upper_edges[upper_edges >= upper]
        lower_target = lower_vals.max() if lower_vals.size else lower_edges.min()
        upper_target = upper_vals.min() if upper_vals.size else upper_edges.max()
        logger.debug(
            '_bound_to_grid: "values" min: %.6f, max: %.6f',
            lower_target,
            upper_target,
        )
        return lower_target, upper_target

    l2_lon_min, l2_lon_max = _bound_to_grid(l2_lon, cur_lon_min, cur_lon_max)
    l2_lat_min, l2_lat_max = _bound_to_grid(l2_lat, cur_lat_min, cur_lat_max)

    def _idx_for(coordinate_values, target_values, name):
        asc_factor = 1 if coordinate_values[1] > coordinate_values[0] else -1
        target = target_values + resolutions.l0 / 2 * asc_factor
        idx = int(np.argmin(np.abs(coordinate_values - target)))
        logger.debug(
            f"_idx_for: {name} target: {target}, L0 coord: {coordinate_values[idx]}, idx: {idx}"
        )
        if not np.isclose(coordinate_values[idx], target, atol=tol):
            logger.warning(
                f"L2 {name} bound {target} not aligned with L0 grid; using {coordinate_values[idx]}"
            )
        return idx

    lon_min_idx = _idx_for(lon, l2_lon_min, "lon-min")
    lon_max_idx = _idx_for(lon, l2_lon_max, "lon-max")
    lat_min_idx = _idx_for(lat, l2_lat_min, "lat-min")
    lat_max_idx = _idx_for(lat, l2_lat_max, "lat-max")

    min_col = min(lon_min_idx, lon_max_idx)
    max_col = max(lon_min_idx, lon_max_idx)
    min_row = min(lat_min_idx, lat_max_idx)
    max_row = max(lat_min_idx, lat_max_idx)
    return min_row, max_row, min_col, max_col


# FUNCTIONS


def get_upscaling_factor(resolutions, max_resolution=False, l1=False, l2=True):
    """Compute integer upscaling factor from a Resolution-like object."""
    input_res = resolutions.l0
    if input_res is None:
        msg = "L0 resolution is required to compute upscaling factor."
        with ErrorLogger(logger):
            raise ValueError(msg)
    if l1:
        upscale_res = resolutions.l1
    elif l2:
        upscale_res = resolutions.l2
    else:
        msg = "Either l1 or l2 must be True."
        with ErrorLogger(logger):
            raise ValueError(msg)
    if max_resolution:
        upscale_res = resolutions.get_max_resolution()
    if upscale_res is None:
        return 1, input_res
    logger.debug(
        f"Computing upscaling factor for input_res {input_res} and upscale_res {upscale_res}"
    )
    ratio = upscale_res / input_res
    ratio_round = int(ratio + 0.5)
    if abs(ratio_round - ratio) < 1e-6:
        return ratio_round, upscale_res
    logger.debug(
        f"Computed upscaling ratio: {ratio:.8f}, rounded: {ratio_round} with difference {abs(ratio_round - ratio):.8f}"
    )
    msg = (
        "Upscaling only works if target resolution is integer multiple of L0 "
        f"but target/L0 = {ratio:.4f}"
    )
    with ErrorLogger(logger):
        raise ValueError(msg)


def cut_to_filled_area(
    ds,
    resolutions: Resolution,
    catchment_mask,
    buffer=0,
    repeat=False,
    raise_on_l2_alignment_mismatch=False,
):
    """Create index slices that crop to filled mask cells, with optional coarse alignment."""
    logger.info("Cutting to filled area.")
    if catchment_mask is None:
        msg = "catchment_mask is None."
        with ErrorLogger(logger):
            raise ValueError(msg)
    if not np.any(catchment_mask):
        msg = "catchment_mask has no filled cells."
        with ErrorLogger(logger):
            raise ValueError(msg)

    cols = np.any(catchment_mask, axis=0)
    rows = np.any(catchment_mask, axis=1)
    logger.info(
        f"shape {np.shape(catchment_mask)}  cols: {len(cols)}, rows: {len(rows)}"
    )
    logger.info(f"lon {len(ds.lon.values)}  lat: {len(ds.lat.values)}")

    min_row, max_row = np.where(rows)[0][[0, -1]]
    min_col, max_col = np.where(cols)[0][[0, -1]]

    if buffer > 0:
        logger.info(f"Using a min buffer of {buffer}")
        min_row = max(0, min_row - buffer)
        min_col = max(0, min_col - buffer)
        max_row = min(catchment_mask.shape[0] - 1, max_row + buffer)
        max_col = min(catchment_mask.shape[1] - 1, max_col + buffer)
    logger.info(
        f"L0 initial window (rows, cols): [{min_row}:{max_row}], [{min_col}:{max_col}]"
    )

    factor, upscale_resolution = get_upscaling_factor(resolutions, l2=True)
    if factor > 1:
        logger.info(
            f"Regridding to fit coarse grid with res {upscale_resolution} (factor {factor})"
        )
        if resolutions.l2_file is not None and not repeat:
            logger.debug(f"Aligning to L2 grid from file {resolutions.l2_file}")
            min_row, max_row, min_col, max_col = align_bounds_to_l2(
                ds, resolutions, min_row, max_row, min_col, max_col
            )
        else:
            min_row = min_row // factor * factor
            min_col = min_col // factor * factor
            max_row = (max_row // factor + 1) * factor
            max_col = (max_col // factor + 1) * factor
        min_row = max(min_row, 0)
        min_col = max(min_col, 0)
        max_row = min(max_row, catchment_mask.shape[0] - 1)
        max_col = min(max_col, catchment_mask.shape[1] - 1)
        logger.info(
            f"After shifting to L2 grid (rows, cols): [{min_row}:{max_row}], [{min_col}:{max_col}]"
        )

    lat_slice_idx = slice(min_row, max_row)
    lon_slice_idx = slice(min_col, max_col)
    n_lat = lat_slice_idx.stop - lat_slice_idx.start
    n_lon = lon_slice_idx.stop - lon_slice_idx.start
    if factor > 1 and ((n_lat % factor) != 0 or (n_lon % factor) != 0):
        msg = f"Cropped L0 shape ({n_lat}, {n_lon}) not divisible by factor={factor}"
        if (
            not repeat
            and resolutions.l2_file is not None
            and not raise_on_l2_alignment_mismatch
        ):
            logger.warning(
                f"{msg} after aligning to L2 grid; check l2 file and alignment calculations."
            )
            return cut_to_filled_area(
                ds=ds,
                resolutions=resolutions,
                catchment_mask=catchment_mask,
                buffer=buffer,
                repeat=True,
                raise_on_l2_alignment_mismatch=raise_on_l2_alignment_mismatch,
            )
        with ErrorLogger(logger):
            raise AssertionError(msg)

    logger.info(f"lat_slice: {lat_slice_idx}, lon_slice: {lon_slice_idx}")
    return lat_slice_idx, lon_slice_idx
