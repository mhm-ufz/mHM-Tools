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
- `file_path`

Rows are ordered with all input variables first, then output variables.
Within each type, rows are sorted by `file_name` and then `variable`.
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

_OUTPUT_FILENAME = ["mHM_Fluxes_States.nc", "discharge.nc"]
_EXCLUDED_INPUT_DIR_KEYS = {
    "dir_out",
    "dir_total_runoff",
}
_TABLE_COLUMNS = [
    "file_name",
    "variable",
    "vartype",
    "min_value",
    "mean_value",
    "max_value",
    "file_path",
]
_VARTYPE_ORDER = {"input": 0, "output": 1}


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


def collect_input_netcdf_files(
    assignments: Sequence[NmlStringAssignment],
    nml_path: Path,
    base_path: Optional[Path],
    recursive: bool,
) -> List[Path]:
    """Collect input NetCDF files from namelist entries."""
    resolved_files: List[Path] = []
    resolved_dirs: List[Path] = []

    for assignment in assignments:
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
) -> List[Path]:
    """Collect `mHM_Fluxes_States.nc` files from `dir_Out(*)` entries."""
    output_files: List[Path] = []
    for assignment in assignments:
        if assignment.key != "dir_out":
            continue
        out_dir = resolve_namelist_path(assignment.value, nml_path, base_path)
        for filename in _OUTPUT_FILENAME:
            candidate = out_dir / filename
            if candidate.is_file():
                output_files.append(candidate.resolve())
            else:
                logger.warning("Output file not found: %s", candidate)
    return _unique_paths(output_files)


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

    return {
        "min_value": min_value,
        "mean_value": mean_value,
        "max_value": max_value,
    }


def collect_stats_rows(
    dataset_paths: Sequence[Path], vartype: str, temporal_mean: bool = False
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
                        "Skipping coordinate-like variable %s:%s.",
                        dataset_path.name,
                        var_name,
                    )
                    continue
                stats = compute_variable_stats(da, temporal_mean=temporal_mean)
                if stats is None:
                    logger.debug(
                        "Skipping %s:%s because dtype %s is non-numeric.",
                        dataset_path.name,
                        var_name,
                        da.dtype,
                    )
                    continue
                rows.append(
                    {
                        "file_name": dataset_path.name,
                        "variable": var_name,
                        "vartype": vartype,
                        "file_path": str(dataset_path),
                        **stats,
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
        logger.info("Removed %d duplicate rows from overview table.", removed)
    return deduped


def create_mhm_run_overview(
    namelist_file: Path,
    base_path: Optional[Path],
    output_dir: Path,
    recursive_input_search: bool = False,
    temporal_mean: bool = False,
) -> None:
    """Create a CSV overview table for input/output NetCDF variables."""
    assignments = parse_namelist_string_assignments(namelist_file)

    input_files = collect_input_netcdf_files(
        assignments=assignments,
        nml_path=namelist_file,
        base_path=base_path,
        recursive=recursive_input_search,
    )
    output_files = collect_output_flux_files(
        assignments=assignments,
        nml_path=namelist_file,
        base_path=base_path,
    )

    logger.info("Found %d input NetCDF files.", len(input_files))
    logger.info("Found %d output flux files.", len(output_files))

    rows_input = collect_stats_rows(
        dataset_paths=input_files, vartype="input", temporal_mean=temporal_mean
    )
    rows_output = collect_stats_rows(
        dataset_paths=output_files, vartype="output", temporal_mean=temporal_mean
    )

    # Keep requested ordering: input rows first, then output rows.
    table = pd.DataFrame(rows_input + rows_output, columns=_TABLE_COLUMNS)
    table = _drop_duplicate_rows(table)
    table = _sort_overview_table(table)

    output_csv = _resolve_output_csv_path(output_dir)
    pretty_print_df(table, 70)
    table.to_csv(output_csv, index=False)
    logger.info("Wrote %d rows to %s", len(table), output_csv)
