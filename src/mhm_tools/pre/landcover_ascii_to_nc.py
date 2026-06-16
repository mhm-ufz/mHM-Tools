"""Convert mHM landcover ASCII maps to CF-compliant NetCDF files.

The module reads land-cover periods from an mHM namelist, converts each ASCII
grid to NetCDF, adds time metadata for the configured validity period, and
writes files suitable for mHM v6 setup workflows.

Authors
-------
- Jeisson Leal
"""

import logging
import re
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import xarray as xr

from mhm_tools.common.file_handler import (
    get_xarray_ds_from_file,
    write_xarray_to_file,
)
from mhm_tools.common.logger import ErrorLogger

logger = logging.getLogger(__name__)


def parse_nml_for_landcover(
    nml_text: str,
) -> Tuple[Dict[int, str], Dict[int, dict], Dict[int, str], int | None]:
    """
    Extract landcover-relevant information from an mHM namelist.

    Returns
    -------
    dir_map : dict[int, str]
        idx -> dir_LCover(idx)
    entries : dict[int, dict]
        idx -> {"year_start": int|None, "year_end": int|None}
    fname_map : dict[int, str]
        idx -> LCoverfName(idx)
    n_domains : int | None
        nDomains value if found, else None

    Also validates that successive [year_start, year_end] blocks are
    continuous (no gaps) and non-overlapping. Raises ValueError if invalid.
    """
    # dir_LCover(i) = "path"
    dir_regex = re.compile(r"dir_LCover\(\s*(\d+)\s*\)\s*=\s*['\"]([^'\"]+)['\"]")
    dir_matches = [(int(m.group(1)), m.group(2)) for m in dir_regex.finditer(nml_text)]
    dir_map = dict(dir_matches)
    if not dir_map:
        logger.error("No dir_LCover entries found in namelist file")

    # LCoverfName(i) = 'filename.asc'
    fname_regex = re.compile(r"LCoverfName\(\s*(\d+)\s*\)\s*=\s*['\"]([^'\"]+)['\"]")
    fname_matches = [
        (int(m.group(1)), m.group(2)) for m in fname_regex.finditer(nml_text)
    ]
    fname_map = dict(fname_matches)

    # LCoverYearStart(i) = YYYY
    # LCoverYearEnd(i)   = YYYY
    start_regex = re.compile(r"LCoverYearStart\(\s*(\d+)\s*\)\s*=\s*(\d+)")
    end_regex = re.compile(r"LCoverYearEnd\(\s*(\d+)\s*\)\s*=\s*(\d+)")
    start_map = {
        int(m.group(1)): int(m.group(2)) for m in start_regex.finditer(nml_text)
    }
    end_map = {int(m.group(1)): int(m.group(2)) for m in end_regex.finditer(nml_text)}

    if not start_map:
        logger.error("No LCoverYearStart entries found in namelist file")
    if not end_map:
        logger.error("No LCoverYearEnd entries found in namelist file")

    # nDomains = <int>
    nd_regex = re.compile(r"nDomains\s*=\s*(\d+)")
    nd_match = nd_regex.search(nml_text)
    n_domains = int(nd_match.group(1)) if nd_match else None
    if n_domains is None:
        logger.error("nDomains not found in namelist file")

    # Build per-index entry containing years
    entries: Dict[int, dict] = {}
    for idx in sorted(fname_map.keys()):
        entries[idx] = {
            "year_start": start_map.get(idx),
            "year_end": end_map.get(idx),
        }
        if entries[idx]["year_start"] is None:
            logger.error(
                f"Missing LCoverYearStart({idx}) for landcover file {fname_map[idx]}"
            )
        if entries[idx]["year_end"] is None:
            logger.error(
                f"Missing LCoverYearEnd({idx}) for landcover file {fname_map[idx]}"
            )

    # Validate continuity and non-overlap of the configured blocks
    ranges = [
        (v["year_start"], v["year_end"], idx)
        for idx, v in entries.items()
        if v.get("year_start") is not None and v.get("year_end") is not None
    ]

    if ranges:
        # sort by start year
        ranges.sort(key=lambda x: x[0])

        prev_start, prev_end, prev_idx = ranges[0]
        for start, end, idx in ranges[1:]:
            # gap? (e.g. prev ends 1990 and next starts 1995)
            if start > prev_end + 1:
                missing_start = prev_end + 1
                missing_end = start - 1
                msg = (
                    "Landcover coverage incomplete: missing years "
                    f"{missing_start}..{missing_end} between entries "
                    f"{prev_idx} ({prev_start}-{prev_end}) and "
                    f"{idx} ({start}-{end})"
                )
                with ErrorLogger(logger):
                    raise ValueError(msg)

            # overlap? (e.g. prev 1981-1990, next 1989-2000)
            if start <= prev_end:
                msg = (
                    "Landcover coverage overlapping: entry "
                    f"{idx} starts at {start} which is <= previous "
                    f"end {prev_end} (entry {prev_idx})"
                )
                with ErrorLogger(logger):
                    raise ValueError(msg)

            prev_start, prev_end, prev_idx = start, end, idx

    return dir_map, entries, fname_map, n_domains


def add_time_bounds_cf(
    ds: xr.Dataset,
    input_infos: Dict[int, dict],
) -> xr.Dataset:
    """
    Take a dataset with a datetime64 'time' coordinate.

    (e.g. [1981-01-01, 1991-01-01])
    and return a new dataset with:

    - time: float64 days since first timestep
    - time_bnds(time, nv=2): start/end bounds for each interval in same units
    - CF-ish metadata:
        time.standard_name = "time"
        time.bounds        = "time_bnds"
        time.encoding["units"]      = "days since <ref>"
        time_bnds.encoding["units"] = same string

    The last bound upper edge is (max(year_end)+1)-01-01T00:00:00.
    """
    # sort/normalize to datetime64[ns]
    time_vals = np.sort(ds["time"].values.astype("datetime64[ns]"))

    # compute the final (last) interval end bound:
    # max(year_end) + 1, Jan-01 of that next year
    last_year_end = max(info["year_end"] for info in input_infos.values())
    final_bound_dt = np.datetime64(f"{last_year_end + 1}-01-01T00:00:00").astype(
        "datetime64[ns]"
    )

    # build upper bounds:
    # for each time[i], upper bound is time[i+1], except the last which goes to final_bound_dt
    upper_bounds = np.array(
        [*list(time_vals[1:]), final_bound_dt],
        dtype="datetime64[ns]",
    )

    # time_bnds_dt shape: (ntime, 2) = [start, end)
    time_bnds_dt = np.column_stack([time_vals, upper_bounds])

    # reference time = first timestep
    ref_dt = time_vals[0]

    # convert datetimes to float days since ref_dt
    def days_since_ref(arr: np.ndarray) -> np.ndarray:
        return ((arr - ref_dt) / np.timedelta64(1, "D")).astype("float64")

    time_numeric = days_since_ref(time_vals)
    time_bnds_numeric = days_since_ref(time_bnds_dt)

    # build CF-style unit string
    ref_dt_str = pd.Timestamp(ref_dt).strftime("%Y-%m-%d %H:%M:%S")
    units_str = f"days since {ref_dt_str}"

    # construct new time coordinate (numeric)
    time_coord = xr.DataArray(
        data=time_numeric,
        dims=("time",),
        name="time",
        attrs={
            "standard_name": "time",
            "bounds": "time_bnds",
        },
    )
    # store units/calendar in encoding, not attrs (prevents xarray writer clash)
    time_coord.encoding["units"] = units_str
    time_coord.encoding["calendar"] = "standard"

    # construct time_bnds variable (numeric)
    time_bnds_var = xr.DataArray(
        data=time_bnds_numeric,
        dims=("time", "nv"),
        name="time_bnds",
        attrs={
            "long_name": "time interval endpoints",
            "comment": "column 0 = interval start, column 1 = interval end",
        },
    )
    time_bnds_var.encoding["units"] = units_str
    time_bnds_var.encoding["calendar"] = "standard"

    # attach to new dataset
    ds_new = ds.assign_coords(time=time_coord)
    ds_new["time_bnds"] = time_bnds_var

    return ds_new


def _build_input_infos(
    nml_path: Path,
    dir_map: Dict[int, str],
    entries: Dict[int, dict],
    fname_map: Dict[int, str],
    n_domains: int | None,
) -> Dict[int, dict]:
    """
    For each landcover block index, build.

    {
        idx: {
            "path": absolute Path to ASCII file,
            "filename": "lc_1981.asc",
            "year_start": 1981,
            "year_end": 1990,
        },
        ...
    }
    """
    input_infos: Dict[int, dict] = {}

    for idx in sorted(entries.keys()):
        info = entries[idx]
        filename = fname_map.get(idx)

        # pick directory for this index
        chosen_dir = None
        if dir_map:
            if n_domains == 1 and 1 in dir_map:
                # special case: single-domain namelist -> always use dir_LCover(1)
                chosen_dir = dir_map[1]
            # prefer dir_LCover(idx), else dir_LCover(1), else first available
            elif idx in dir_map:
                chosen_dir = dir_map[idx]
            elif 1 in dir_map:
                chosen_dir = dir_map[1]
            else:
                chosen_dir = next(iter(dir_map.values()))

        # resolve relative dirs against the namelist location
        if chosen_dir:
            cand = Path(chosen_dir)
            if cand.is_absolute():
                base_dir = cand.resolve()
            else:
                base_dir = (nml_path.parent / cand).resolve()
            full_path = (base_dir / filename).resolve()
        else:
            full_path = (nml_path.parent / filename).resolve()

        input_infos[idx] = {
            "path": full_path,
            "filename": filename,
            "year_start": info.get("year_start"),
            "year_end": info.get("year_end"),
        }

    return input_infos


def convert_lc_ascii_to_nc(
    input_nml: str | Path,
    output: str | Path,
    var_name: str | None = "land_cover",
    normalize_latlon: bool = False,
) -> None:
    """
    Workflow for converting landcover ASCII files to NetCDF.

    1. Parse mHM namelist for landcover blocks (paths, years).
    2. Read each ASCII landcover file into xarray.
    3. Concatenate along new 'time' dimension.
    4. Add CF-style numeric time + time_bnds.
    5. Write NetCDF.

    Notes
    -----
    - Supports any number of ASCII tiles (1, 2, ...).
    - The last time bound ends at (max(year_end)+1)-01-01.
    - By default the data variable is called 'land_cover'. You can override
      that with `var_name=...`.
    """
    # read & parse namelist
    nml_path = Path(input_nml)
    if not nml_path.exists():
        msg = f"NML file not found: {nml_path}"
        with ErrorLogger(logger):
            raise FileNotFoundError(msg)

    nml_text = nml_path.read_text()
    dir_map, entries, fname_map, n_domains = parse_nml_for_landcover(nml_text)

    if not entries:
        msg = "No LCoverfName entries found in the provided .nml file"
        with ErrorLogger(logger):
            raise ValueError(msg)

    # resolve actual input files and their metadata
    input_infos = _build_input_infos(
        nml_path=nml_path,
        dir_map=dir_map,
        entries=entries,
        fname_map=fname_map,
        n_domains=n_domains,
    )

    # load each ASCII file into an xarray Dataset
    datasets = []
    for idx in sorted(input_infos.keys()):
        info = input_infos[idx]
        ds_single = get_xarray_ds_from_file(
            info["path"],
            var_name=var_name,
            normalize_latlon_coords=normalize_latlon,
            landcover=True,
            landcover_year_start=info["year_start"],
        )
        datasets.append(ds_single)

    # merge along time
    if len(datasets) == 1:
        merged = datasets[0]
    else:
        merged = xr.concat(datasets, dim="time").sortby("time")

    # add CF-style time + time_bnds
    merged = add_time_bounds_cf(merged, input_infos)

    # write result to NetCDF
    write_xarray_to_file(merged, Path(output))
