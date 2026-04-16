"""Utility helpers."""

import logging

import pandas as pd
from mhm_tools.common.logger import ErrorLogger
import numpy as np
import xarray as xr

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
        msg = (
            "Could not map given coordinates to valid indices within "
            "dataset bounds."
        )
        with ErrorLogger(logger):
            raise ValueError(msg)
    i = int(np.clip(i, 0, len(lat_vals) - 1))
    j = int(np.clip(j, 0, len(lon_vals) - 1))

    return i, j

def distance_100m_units(di, dj, l0_resolution, lat_deg=None, latlon=False):
        """Convert index deltas to distance in ~100 m units using l0_resolution."""
        res = float(abs(l0_resolution))
        if latlon:
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
    

def find_best_gauge_location(  # noqa: PLR0915
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

    # Extract subgrid of upstream_area
    sub = upstream_area[i_min : i_max + 1, j_min : j_max + 1]

    # If subgrid is empty fallback to whole domain
    if sub.size == 0:
        logger.warning("Search bbox empty, falling back to full-domain search")
        sub = upstream_area
        i_min, j_min = 0, 0
    lat_deg = float(ds.lat.data[gi]) if latlon else None
    # Search for candidate cells whose upstream area matches ref_catchment_area
    if method == "basinex":
        logger.info("Correcting gauge location using basinex method")
        # based on implementation in basinex https://git.ufz.de/schaefed/basin-extractor/-/blame/master/lib/gauges.py?ref_type=heads#L42
        size = float(ref_catchment_area)
        error = 0.0
        step = 0.01
        candidates = None
        while error <= max_error and (
            candidates is None or len(candidates[0]) == 0
        ):
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
            logger.info(
                f"Selected outlet candidate {best_coord} with upstream area {upstream_area[best_coord]} km2 (tolerance {error:.3f})"
            )
            distanance_100m = distance_100m_units(
                cand_i[k] - gi, cand_j[k] - gj, l0_resolution=resolutions.l0_resolution, lat_deg=lat_deg, latlon=latlon
            )
            return (
                best_coord,
                np.abs(1 - sub[candidates[0][k], candidates[1][k]] / size),
                distanance_100m,
            )
    elif method == "burek":
        logger.info("Correcting gauge location using burek method")
        # based on Burek et. al. 2023 https://essd.copernicus.org/articles/15/5617/2023/
        # implemented https://github.com/iiasa/CWATM_grdc_calibration_stations/blob/78979cbac8f8685d8dbc5330dba6f40a929716f4/scripts/1_findMeritcoord.py#L335
        size = float(ref_catchment_area)
        low = size * (1.0 - max_error)
        high = size * (1.0 + max_error)
        candidates_indices = np.where((sub >= low) & (sub <= high))
        if len(candidates_indices[0]) > 0:
            cand_i = candidates_indices[0] + i_min
            cand_j = candidates_indices[1] + j_min
            di = cand_i - gi
            dj = cand_j - gj
            candidates_distance = distance_100m_units(
                di,
                dj,
                l0_resolution=resolutions.l0_resolution,
                lat_deg=lat_deg,
                latlon=latlon,
            )
            candidates_error = 100 * np.abs(
                1 - sub[candidates_indices] / size
            )  # 100 * np.abs(1 - ups[y, x] / upsreal)
            if not recursion:
                burek_metric = candidates_error + 2 * candidates_distance
            else:
                # in recursive step we double the max distance burek proposes to use distance and error with equal weight
                burek_metric = candidates_error + candidates_distance
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

    if raise_on_fallback:
        msg = (
            f"No suitable outlet candidate found within {max_distance_cells} cells and {max_error*100:.2f}% area error. "
            "Consider increasing max_distance_cells or max_error."
        )
        with ErrorLogger(logger):
            raise ValueError(msg)
    if not recursion:
        logger.warning(
            "No candidate found within initial tolerance; trying again with doubled radius."
        )
        logger.warning(
            f"Radius: {max_distance_cells*2} cells, error tolerance: {max_error*100:.2f}%."
        )
        return find_best_gauge_location(
            ds,
            upstream_area,
            gauge_coords,
            ref_catchment_area,
            resolutions,
            max_distance_cells * 2,
            max_error,
            recursion=True,
            method=method,
            raise_on_fallback=False,
            latlon=latlon,
        )
    logger.warning(
        "No candidate found within tolerance; doubling search radius again and doubling error tolerance."
    )
    logger.warning(
        f"Radius: {max_distance_cells*2} cells, error tolerance: {max_error*200:.2f}%."
    )
    return find_best_gauge_location(
        ds,
        upstream_area,
        gauge_coords,
        ref_catchment_area,
        resolutions,
        max_distance_cells * 2,
        max_error * 2,
        recursion=True,
        method=method,
        raise_on_fallback=True,
        latlon=latlon,
    )
