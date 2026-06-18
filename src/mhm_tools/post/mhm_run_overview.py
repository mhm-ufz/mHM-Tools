"""Build run-overview statistics from mHM namelist-linked NetCDF files.

This module reads an mHM namelist, resolves referenced input directories/files,
finds output `mHM_Fluxes_States.nc` files from `dir_Out(*)`, computes summary
statistics per variable, and writes a CSV overview table.

The written table has columns:
- `file_name`
- `variable`
- `vartype` (`input` or `output`)
- `min_value`
- `mean_value`
- `max_value`
- `unit`
- `file_path`

Rows are ordered with all input variables first, then output variables.
Within each type, rows are sorted by `file_name` and then `variable`.

If multiple domains are configured using indexed paths (for example
`dir_out(1)`, `dir_out(2)`), one CSV table per domain is written.

Authors
-------
- Simon Lüdke
"""

import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Sequence

import numpy as np
import pandas as pd
import xarray as xr

from mhm_tools.common.file_handler import get_xarray_ds_from_file
from mhm_tools.common.utils import pretty_print_df

logger = logging.getLogger(__name__)

_ASSIGNMENT_RE = re.compile(
    r"^\s*(?P<key>[A-Za-z][A-Za-z0-9_]*)\s*"
    r"(?:\(\s*(?P<index>[^)]*)\s*\))?\s*=\s*(?P<value>.+?)\s*$"
)
_STRING_RE = re.compile(r"""['"]([^'"]*)['"]""")
_NDOMAINS_RE = re.compile(r"^\s*ndomains\s*=\s*(\d+)\b", re.IGNORECASE)

_OUTPUT_FILENAME = ["mHM_Fluxes_States.nc", "discharge.nc"]
_EXCLUDED_INPUT_DIR_KEYS = {
    "dir_out",
    "dir_total_runoff",
}
_TIME_REFERENCE_DIR_KEYS = {
    "dir_precipitation",
    "dir_temperature",
    "dir_referenceet",
    "dir_mintemperature",
    "dir_maxtemperature",
    "dir_netradiation",
    "dir_absvappressure",
    "dir_windspeed",
    "dir_radiation",
}
_TABLE_COLUMNS = [
    "file_name",
    "variable",
    "vartype",
    "min_value",
    "mean_value",
    "max_value",
    "unit",
    "file_path",
]
_VARTYPE_ORDER = {"input": 0, "output": 1}
_SECONDS_PER_TIME_UNIT = {
    "s": 1.0,
    "d": 86400.0,
    "month": 30.4 * 86400.0,
    "y": 12.0 * 30.4 * 86400.0,
    "year": 12.0 * 30.4 * 86400.0,
}


@dataclass(frozen=True)
class NmlStringAssignment:
    """Container for one string assignment parsed from a namelist line."""

    key: str
    index: Optional[str]
    value: str


def _strip_namelist_comment(line: str) -> str:
    """Remove Fortran-style comments while respecting quoted strings."""
    in_single = False
    in_double = False
    for pos, char in enumerate(line):
        if char == "'" and not in_double:
            in_single = not in_single
        elif char == '"' and not in_single:
            in_double = not in_double
        elif char == "!" and not in_single and not in_double:
            return line[:pos]
    return line


def parse_namelist_string_assignments(nml_path: Path) -> List[NmlStringAssignment]:
    """Parse string assignments from a namelist file."""
    assignments: List[NmlStringAssignment] = []
    for raw_line in nml_path.read_text(encoding="utf-8").splitlines():
        line = _strip_namelist_comment(raw_line).strip()
        if not line:
            continue
        match = _ASSIGNMENT_RE.match(line)
        if match is None:
            continue
        value_match = _STRING_RE.search(match.group("value"))
        if value_match is None:
            continue
        assignments.append(
            NmlStringAssignment(
                key=match.group("key").lower(),
                index=match.group("index"),
                value=value_match.group(1).strip(),
            )
        )
    return assignments


def resolve_namelist_path(
    path_str: str, nml_path: Path, base_path: Optional[Path] = None
) -> Path:
    """Resolve a path from namelist context to an absolute path."""
    path = Path(path_str).expanduser()
    if path.is_absolute():
        return path
    anchor = base_path if base_path is not None else nml_path.parent
    return (anchor / path).resolve()


def _unique_paths(paths: Iterable[Path]) -> List[Path]:
    """Return de-duplicated paths with stable order."""
    unique: List[Path] = []
    seen = set()
    for path in paths:
        key = str(path)
        if key in seen:
            continue
        seen.add(key)
        unique.append(path)
    return unique


def _parse_domain_index(index: Optional[str]) -> Optional[int]:
    """Parse a single integer domain index from assignment index text."""
    if index is None:
        return None
    stripped = index.strip()
    if stripped.isdigit():
        return int(stripped)
    return None


def _get_domain_indices(assignments: Sequence[NmlStringAssignment]) -> List[int]:
    """Extract sorted unique domain indices from `dir_out(<i>)` entries."""
    indices = {
        parsed
        for assignment in assignments
        if assignment.key == "dir_out"
        for parsed in [_parse_domain_index(assignment.index)]
        if parsed is not None
    }
    return sorted(indices)


def _parse_n_domains(nml_path: Path) -> Optional[int]:
    """Parse `nDomains` from namelist.

    Returns `None` if no value is found.
    """
    for raw_line in nml_path.read_text(encoding="utf-8").splitlines():
        line = _strip_namelist_comment(raw_line).strip()
        if not line:
            continue
        match = _NDOMAINS_RE.match(line)
        if match is not None:
            return int(match.group(1))
    return None


def collect_input_netcdf_files(
    assignments: Sequence[NmlStringAssignment],
    nml_path: Path,
    base_path: Optional[Path],
    recursive: bool,
    domain_idx: Optional[int] = None,
) -> List[Path]:
    """Collect input NetCDF files from namelist entries."""
    resolved_files: List[Path] = []
    resolved_dirs: List[Path] = []

    for assignment in assignments:
        assignment_domain = _parse_domain_index(assignment.index)
        if domain_idx is not None and assignment_domain not in (None, domain_idx):
            continue
        if (
            assignment.key.startswith("dir_")
            and assignment.key not in _EXCLUDED_INPUT_DIR_KEYS
        ):
            directory = resolve_namelist_path(assignment.value, nml_path, base_path)
            if directory.is_dir():
                resolved_dirs.append(directory)

        if (
            assignment.value.lower().endswith(".nc")
            and "restartout" not in assignment.key
            and "latlon" not in assignment.key
        ):
            file_path = resolve_namelist_path(assignment.value, nml_path, base_path)
            if file_path.is_file():
                resolved_files.append(file_path.resolve())

    for directory in _unique_paths(resolved_dirs):
        iterator = directory.rglob("*.nc") if recursive else directory.glob("*.nc")
        for nc_file in sorted(iterator):
            if nc_file.is_file():
                resolved_files.append(nc_file.resolve())

    return _unique_paths(resolved_files)


def collect_output_flux_files(
    assignments: Sequence[NmlStringAssignment],
    nml_path: Path,
    base_path: Optional[Path],
    domain_idx: Optional[int] = None,
) -> List[Path]:
    """Collect `mHM_Fluxes_States.nc` files from `dir_Out(*)` entries."""
    output_files: List[Path] = []
    for assignment in assignments:
        if assignment.key != "dir_out":
            continue
        assignment_domain = _parse_domain_index(assignment.index)
        if domain_idx is not None and assignment_domain != domain_idx:
            continue
        out_dir = resolve_namelist_path(assignment.value, nml_path, base_path)
        found_in_dir = False
        for filename in _OUTPUT_FILENAME:
            candidate = out_dir / filename
            if candidate.is_file():
                output_files.append(candidate.resolve())
                found_in_dir = True
                continue
            recursive_candidates = sorted(
                p.resolve() for p in out_dir.rglob(filename) if p.is_file()
            )
            if recursive_candidates:
                output_files.extend(recursive_candidates)
                found_in_dir = True
        if not found_in_dir:
            logger.warning(
                f"No known output files ({', '.join(_OUTPUT_FILENAME)}) found in "
                f"{out_dir}"
            )
    return _unique_paths(output_files)


def collect_input_time_reference_files(
    assignments: Sequence[NmlStringAssignment],
    nml_path: Path,
    base_path: Optional[Path],
    recursive: bool,
    domain_idx: Optional[int] = None,
) -> List[Path]:
    """Collect forcing files used to infer input temporal resolution.

    Selection is based on namelist directory keys instead of filename patterns.
    """
    resolved_files: List[Path] = []
    for assignment in assignments:
        if assignment.key not in _TIME_REFERENCE_DIR_KEYS:
            continue
        assignment_domain = _parse_domain_index(assignment.index)
        if domain_idx is not None and assignment_domain not in (None, domain_idx):
            continue
        directory = resolve_namelist_path(assignment.value, nml_path, base_path)
        if not directory.is_dir():
            continue
        iterator = directory.rglob("*.nc") if recursive else directory.glob("*.nc")
        for nc_file in sorted(iterator):
            if nc_file.is_file():
                resolved_files.append(nc_file.resolve())
    return _unique_paths(resolved_files)


def guess_time_dim(da: xr.DataArray) -> Optional[str]:
    """Guess the time dimension of a DataArray."""
    for dim in da.dims:
        if dim.lower() == "time":
            return dim
    for dim in da.dims:
        coord = da.coords.get(dim)
        if coord is None:
            continue
        try:
            if np.issubdtype(coord.dtype, np.datetime64):
                return dim
        except TypeError:
            pass
        standard_name = str(coord.attrs.get("standard_name", "")).lower()
        if standard_name == "time":
            return dim
    return None


def _to_float(value) -> float:
    """Convert scalar values to Python float."""
    try:
        return float(value)
    except Exception:
        return float("nan")


def _infer_time_resolution_seconds(dataset_paths: Sequence[Path]) -> Optional[float]:
    """Infer input temporal resolution in seconds from time coordinates."""
    deltas_seconds: List[float] = []
    for dataset_path in dataset_paths:
        with get_xarray_ds_from_file(dataset_path) as ds:
            for da in ds.data_vars.values():
                time_dim = guess_time_dim(da)
                if time_dim is None:
                    continue
                coord = da.coords.get(time_dim)
                if coord is None or coord.size < 2:
                    continue
                try:
                    if not np.issubdtype(coord.dtype, np.datetime64):
                        continue
                except TypeError:
                    continue
                values = coord.values.astype("datetime64[s]").astype("int64")
                diffs = np.diff(values).astype(float)
                positive_diffs = diffs[diffs > 0]
                if positive_diffs.size > 0:
                    deltas_seconds.extend(positive_diffs.tolist())
    if not deltas_seconds:
        return None
    return float(np.median(np.asarray(deltas_seconds)))


def _closest_time_unit_code(seconds: float) -> str:
    """Return closest supported time unit code for given seconds."""
    return min(
        _SECONDS_PER_TIME_UNIT,
        key=lambda code: abs(seconds - _SECONDS_PER_TIME_UNIT[code]),
    )


def _find_rate_time_unit(unit: str) -> Optional[dict]:
    """Find denominator time unit markers like `/s` or `d-1`."""
    if not unit:
        return None
    token_map = {
        "s": ["seconds", "second", "secs", "sec", "s"],
        "d": ["days", "day", "d"],
        "month": ["months", "month", "mons", "mon"],
        "y": ["years", "year", "yrs", "yr", "y"],
    }
    for code, tokens in token_map.items():
        alt = "|".join(sorted(tokens, key=len, reverse=True))
        slash_pattern = re.compile(rf"/\s*({alt})\b", re.IGNORECASE)
        slash_match = slash_pattern.search(unit)
        if slash_match is not None:
            return {
                "code": code,
                "style": "slash",
                "start": slash_match.start(),
                "end": slash_match.end(),
                "text": slash_match.group(0),
            }

        exp_pattern = re.compile(
            rf"(?<![A-Za-z0-9_])({alt})\s*\^?\s*-\s*1\b",
            re.IGNORECASE,
        )
        exp_match = exp_pattern.search(unit)
        if exp_match is not None:
            return {
                "code": code,
                "style": "exp",
                "start": exp_match.start(),
                "end": exp_match.end(),
                "text": exp_match.group(0),
            }
    return None


def _rewrite_unit_to_target_time(unit: str, target_code: str) -> str:
    """Rewrite denominator time unit in `unit` to target code."""
    found = _find_rate_time_unit(unit)
    if found is None:
        return unit
    source_text = found["text"]
    if found["style"] == "slash":
        replacement = f"/{target_code}"
    else:
        replacement = f"{target_code}-1"
        if "^" in source_text:
            replacement = f"{target_code}^-1"
    return f"{unit[:found['start']]}{replacement}{unit[found['end']:]}"


def _convert_stats_to_target_time_unit(
    stats: dict,
    target_time_seconds: float,
) -> dict:
    """Convert rate-like stats to a target temporal resolution."""
    unit = str(stats.get("unit", "")).strip()
    found = _find_rate_time_unit(unit)
    if found is None:
        return stats
    source_code = found["code"]
    source_seconds = _SECONDS_PER_TIME_UNIT[source_code]
    target_code = _closest_time_unit_code(target_time_seconds)
    target_seconds = _SECONDS_PER_TIME_UNIT[target_code]
    factor = target_seconds / source_seconds
    converted = dict(stats)
    converted["min_value"] = _to_float(converted["min_value"]) * factor
    converted["mean_value"] = _to_float(converted["mean_value"]) * factor
    converted["max_value"] = _to_float(converted["max_value"]) * factor
    converted["unit"] = _rewrite_unit_to_target_time(unit, target_code)
    return converted


def compute_variable_stats(
    da: xr.DataArray, temporal_mean: bool = False
) -> Optional[dict]:
    """Compute min/mean/max for a variable.

    If a time dimension exists, statistics are computed on the temporal mean.
    Non-numeric variables are skipped.
    """
    if not np.issubdtype(da.dtype, np.number):
        return None

    time_dim = guess_time_dim(da)
    if da.ndim == 3 and time_dim is not None and temporal_mean:
        stats_da = da.mean(dim=time_dim, skipna=True)
    else:
        stats_da = da
    min_value = _to_float(stats_da.min(skipna=True).values)
    mean_value = _to_float(stats_da.mean(skipna=True).values)
    max_value = _to_float(stats_da.max(skipna=True).values)
    unit = da.attrs.get("units", "")
    unit_clean = str(unit).strip() if unit is not None else ""

    return {
        "min_value": min_value,
        "mean_value": mean_value,
        "max_value": max_value,
        "unit": unit_clean,
    }


def collect_stats_rows(
    dataset_paths: Sequence[Path],
    vartype: str,
    temporal_mean: bool = False,
    convert_units: bool = False,
    target_time_seconds: Optional[float] = None,
) -> List[dict]:
    """Collect table rows for all numeric variables in given datasets."""
    rows: List[dict] = []

    def _is_coordinate_like_variable(
        da: xr.DataArray, var_name: Optional[str] = None
    ) -> bool:
        keys = ["lon", "lat", "latitude", "longitude", "easting", "northing", "_bnds"]
        for key in keys:
            if key in da.name or (key in var_name if var_name else False):
                return True
        return False

    for dataset_path in dataset_paths:
        with get_xarray_ds_from_file(dataset_path) as ds:
            for var_name, da in ds.data_vars.items():
                if _is_coordinate_like_variable(da):
                    logger.debug(
                        f"Skipping coordinate-like variable {dataset_path.name}:"
                        f"{var_name}."
                    )
                    continue
                stats = compute_variable_stats(da, temporal_mean=temporal_mean)
                if stats is None:
                    logger.debug(
                        f"Skipping {dataset_path.name}:{var_name} because dtype "
                        f"{da.dtype} is non-numeric."
                    )
                    continue
                if (
                    convert_units
                    and vartype == "output"
                    and target_time_seconds is not None
                ):
                    stats = _convert_stats_to_target_time_unit(
                        stats=stats,
                        target_time_seconds=target_time_seconds,
                    )
                rows.append(
                    {
                        "file_name": dataset_path.name,
                        "variable": var_name,
                        "vartype": vartype,
                        "file_path": str(dataset_path),
                        "min_value": f"{_to_float(stats['min_value']):.8g}",
                        "mean_value": f"{_to_float(stats['mean_value']):.8g}",
                        "max_value": f"{_to_float(stats['max_value']):.8g}",
                        "unit": stats.get("unit", ""),
                    }
                )
    return rows


def _resolve_output_csv_path(output_dir: Path) -> Path:
    """Resolve CSV output path.

    If `output_dir` ends with `.csv`, it is used as file path.
    Otherwise a default file `variable_summary.csv` is created in that directory.
    """
    if output_dir.suffix.lower() == ".csv":
        output_dir.parent.mkdir(parents=True, exist_ok=True)
        return output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir / "variable_summary.csv"


def _resolve_output_csv_path_for_domain(output_dir: Path, domain_idx: int) -> Path:
    """Resolve per-domain output CSV path."""
    if output_dir.suffix.lower() == ".csv":
        output_dir.parent.mkdir(parents=True, exist_ok=True)
        return output_dir.with_name(f"{output_dir.stem}_domain{domain_idx}.csv")
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir / f"variable_summary_domain{domain_idx}.csv"


def _sort_overview_table(table: pd.DataFrame) -> pd.DataFrame:
    """Sort table by vartype, file_name, and variable.

    Sorting rules:
    1. all `input` rows first, then `output`
    2. alphabetical by `file_name`
    3. alphabetical by `variable`
    """
    if table.empty:
        return table

    table_sorted = table.copy()
    table_sorted["_vartype_order"] = (
        table_sorted["vartype"].map(_VARTYPE_ORDER).fillna(99)
    )
    table_sorted = table_sorted.sort_values(
        by=["_vartype_order", "file_name", "variable"],
        kind="mergesort",
    ).drop(columns=["_vartype_order"])
    return table_sorted.reset_index(drop=True)


def _drop_duplicate_rows(table: pd.DataFrame) -> pd.DataFrame:
    """Remove duplicate rows from the overview table."""
    if table.empty:
        return table
    before = len(table)
    deduped = table.drop_duplicates(ignore_index=True)
    removed = before - len(deduped)
    if removed > 0:
        logger.info(f"Removed {removed} duplicate rows from overview table.")
    return deduped


def create_mhm_run_overview(
    namelist_file: Path,
    base_path: Optional[Path],
    output_dir: Path,
    recursive_input_search: bool = False,
    temporal_mean: bool = False,
    convert_units: bool = False,
) -> None:
    """Create a CSV overview table for input/output NetCDF variables."""
    assignments = parse_namelist_string_assignments(namelist_file)
    configured_domain_indices = _get_domain_indices(assignments)
    n_domains = _parse_n_domains(namelist_file)
    if n_domains is not None and n_domains > 0:
        domain_indices = list(range(1, n_domains + 1))
    else:
        domain_indices = configured_domain_indices

    if n_domains is not None:
        logger.info(
            f"Using nDomains={n_domains} to select active domains: {domain_indices}"
        )

    def _build_table_for_domain(domain_idx: Optional[int]) -> pd.DataFrame:
        input_files = collect_input_netcdf_files(
            assignments=assignments,
            nml_path=namelist_file,
            base_path=base_path,
            recursive=recursive_input_search,
            domain_idx=domain_idx,
        )
        output_files = collect_output_flux_files(
            assignments=assignments,
            nml_path=namelist_file,
            base_path=base_path,
            domain_idx=domain_idx,
        )

        if domain_idx is None:
            logger.info(f"Found {len(input_files)} input NetCDF files.")
            logger.info(f"Found {len(output_files)} output flux files.")
        else:
            logger.info(
                f"Domain {domain_idx}: found {len(input_files)} input NetCDF files "
                f"and {len(output_files)} output flux files."
            )

        rows_input = collect_stats_rows(
            dataset_paths=input_files,
            vartype="input",
            temporal_mean=temporal_mean,
            convert_units=False,
            target_time_seconds=None,
        )
        input_time_reference_files = collect_input_time_reference_files(
            assignments=assignments,
            nml_path=namelist_file,
            base_path=base_path,
            recursive=recursive_input_search,
            domain_idx=domain_idx,
        )
        if not input_time_reference_files:
            input_time_reference_files = input_files
        target_time_seconds = _infer_time_resolution_seconds(input_time_reference_files)
        if convert_units and target_time_seconds is not None:
            logger.info(
                f"Inferred input temporal resolution: {target_time_seconds:.3f} "
                f"seconds (closest: {_closest_time_unit_code(target_time_seconds)})."
            )
        if convert_units and target_time_seconds is None:
            logger.warning(
                "Could not infer input temporal resolution; output units will not be converted."
            )
        rows_output = collect_stats_rows(
            dataset_paths=output_files,
            vartype="output",
            temporal_mean=temporal_mean,
            convert_units=convert_units,
            target_time_seconds=target_time_seconds,
        )

        table_local = pd.DataFrame(rows_input + rows_output, columns=_TABLE_COLUMNS)
        return _sort_overview_table(_drop_duplicate_rows(table_local))

    if len(domain_indices) > 1:
        for domain_idx in domain_indices:
            table = _build_table_for_domain(domain_idx=domain_idx)
            output_csv = _resolve_output_csv_path_for_domain(output_dir, domain_idx)
            pretty_print_df(table, 70)
            table.to_csv(output_csv, index=False)
            logger.info(f"Domain {domain_idx}: wrote {len(table)} rows to {output_csv}")
        return

    if len(domain_indices) == 1:
        table = _build_table_for_domain(domain_idx=domain_indices[0])
    else:
        table = _build_table_for_domain(domain_idx=None)
    output_csv = _resolve_output_csv_path(output_dir)
    pretty_print_df(table, 70)
    table.to_csv(output_csv, index=False)
    logger.info(f"Wrote {len(table)} rows to {output_csv}")
