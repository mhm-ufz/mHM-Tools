"""
Compute long-term means for NetCDF forcing data.

This script is designed for climate/hydrology forcing datasets that span
multiple timesteps (e.g., hourly, daily, monthly, yearly). It supports both
"intensive" variables (averaged over time, e.g., temperature) and "extensive"
variables (summed over time, e.g., precipitation).

The workflow includes:
- optional cropping to a spatial domain,
- optional temporal aggregation (hourly → daily, daily → monthly, etc.),
- merging multiple input files along the time dimension,
- computing the long-term mean (with optional masking thresholds),
- saving the result as a NetCDF file.

Intermediate files can be kept or automatically removed depending on options.

Authors
-------
- Jeisson Leal
"""

import logging
import re
import uuid
from pathlib import Path
from typing import Optional

try:
    from cdo import Cdo

    cdo = Cdo()
except Exception:
    cdo = None

from mhm_tools.common.logger import ErrorLogger
from mhm_tools.pre.crop_mhm_setup import crop_file

logger = logging.getLogger(__name__)

# Shared mapping for CDO operations
OP_MAP = {
    "hourly": ("hourmean", "hoursum"),
    "daily": ("daymean", "daysum"),
    "monthly": ("monmean", "monsum"),
    "yearly": ("yearmean", "yearsum"),
}


def _summarize_inputs_for_log(files: list[Path]) -> str:
    """Return a compact summary for logging a file collection."""
    if not files:
        return "no files"
    years = []
    year_re = re.compile(r"^(\d{4})")
    for file in files:
        match = year_re.match(file.name)
        if match:
            years.append(int(match.group(1)))
    if years:
        return (
            f"{len(files)} files (years {min(years)}-{max(years)}), "
            f"sample: {files[0].name} ... {files[-1].name}"
        )
    return f"{len(files)} files, sample: {files[0].name} ... {files[-1].name}"


def aggregate_files(
    input_file: Path,
    output_path: Path,
    aggregation_type: str = "intensive",
    long_term_mean_type: str = "monthly",
) -> None:
    """Aggregate a single NetCDF file into a coarser temporal resolution using CDO."""
    if aggregation_type not in ("intensive", "extensive"):
        msg = (
            f"aggregation_type must be 'intensive' or 'extensive', "
            f"got {aggregation_type!r}"
        )
        raise ValueError(msg)
    if long_term_mean_type not in OP_MAP:
        msg = (
            f"Unsupported long_term_mean_type {long_term_mean_type!r}; "
            f"choose from {list(OP_MAP)}"
        )
        raise ValueError(msg)

    mean_op, sum_op = OP_MAP[long_term_mean_type]
    cdo_op = mean_op if aggregation_type == "intensive" else sum_op
    output_path.mkdir(parents=True, exist_ok=True)

    if not input_file.exists():
        msg = f"Input file not found: {input_file}"
        raise FileNotFoundError(msg)
    out_filename = f"{cdo_op}_{input_file.name}"
    out_file = output_path / out_filename

    cdo = Cdo()
    try:
        # Per-file aggregation can be numerous; keep this at debug level.
        logger.debug(f"Running CDO {cdo_op} on {input_file} → {out_file}")
        getattr(cdo, cdo_op)(input=str(input_file), output=str(out_file))
    except Exception as e:
        msg = f"CDO operation '{cdo_op}' failed on file '{input_file}': {e}"
        with ErrorLogger(logger):
            raise RuntimeError(msg) from e


def cal_long_term_mean(  # noqa: PLR0912, PLR0915
    in_dir: str,
    in_file: str,
    out_dir: str,
    long_term_mean_type: str = "monthly",
    aggregation_type: str = "intensive",
    keep_temporal_files: bool = False,
    out_file: Optional[str] = None,
    crop: bool = False,
    lon_min: Optional[float] = None,
    lon_max: Optional[float] = None,
    lat_min: Optional[float] = None,
    lat_max: Optional[float] = None,
    aggregate: bool = False,
    lower_threshold: Optional[str] = None,
) -> None:
    """Compute long-term means for NetCDF forcing data.

    Optionally perform temporal aggregation and then merge, or merge raw inputs first.
    If only one file matches `in_file`, skip the CDO mergetime step.
    """
    p_in_dir = Path(in_dir)
    files = sorted(p_in_dir.glob(in_file))
    if not files:
        msg = f"No files match pattern {p_in_dir / in_file}"
        with ErrorLogger(logger):
            raise FileNotFoundError(msg)

    # Track optional temporary directories for safe cleanup.
    crop_dir = None
    aggregation_dir = None

    # If cropping is requested, verify bounds
    if crop and None in (lon_min, lon_max, lat_min, lat_max):
        msg = "All lon/lat bounds must be provided when --crop is used."
        with ErrorLogger(logger):
            raise ValueError(msg)

    # Prepare output directory
    p_out_dir = Path(out_dir)
    p_out_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Matched {_summarize_inputs_for_log(files)} using pattern '{in_file}'")

    # If cropping, produce cropped files into a subfolder and use that as input
    if crop:
        # Unique temp folder avoids collisions across concurrent/repeated runs.
        crop_dir = p_out_dir / f"cropped_files_{uuid.uuid4().hex}"
        crop_dir.mkdir(parents=True, exist_ok=True)
        # Note: flipping lat_min/lat_max because slice(lat_max, lat_min) often needed
        lonslice = slice(lon_min, lon_max)
        latslice = slice(lat_max, lat_min)
        for fpath in files:
            crop_file(
                input_path=p_in_dir,
                input_file=fpath,
                output_path=crop_dir,
                mask_da=False,
                overwrite=True,
                available_mem_gib=None,
                latslice=latslice,
                lonslice=lonslice,
            )
        # p_input_dir = crop_dir
        # After cropping, map to flattened cropped outputs by file name.
        files = sorted((crop_dir / fpath.name) for fpath in files)
    # else:
    # p_input_dir = p_in_dir

    cdo = Cdo()

    # If aggregation is requested, aggregate each file individually first
    if aggregate:
        if long_term_mean_type not in OP_MAP:
            msg = (
                f"Unsupported long_term_mean_type {long_term_mean_type!r}; "
                f"choose from {list(OP_MAP)}"
            )
            raise ValueError(msg)
        if aggregation_type not in ("intensive", "extensive"):
            msg = (
                f"aggregation_type must be 'intensive' or 'extensive', "
                f"got {aggregation_type!r}"
            )
            raise ValueError(msg)

        mean_op, sum_op = OP_MAP[long_term_mean_type]
        cdo_op = mean_op if aggregation_type == "intensive" else sum_op

        # Unique temp folder avoids collisions across concurrent/repeated runs.
        aggregation_dir = p_out_dir / f"aggregated_files_{uuid.uuid4().hex}"
        aggregation_dir.mkdir(parents=True, exist_ok=True)
        logger.info(
            f"Aggregating {_summarize_inputs_for_log(files)} with CDO operator '{cdo_op}'."
        )

        # Run `aggregate_files` on each matched input
        for fpath in files:
            aggregate_files(
                input_file=fpath,
                output_path=aggregation_dir,
                aggregation_type=aggregation_type,
                long_term_mean_type=long_term_mean_type,
            )

        merge_candidates = sorted(aggregation_dir.glob(f"{cdo_op}_*.nc"))

    else:
        # No aggregation: merge (or skip merging) the already matched raw inputs
        merge_candidates = files

    if not merge_candidates:
        msg = f"No files available for merging from pattern {p_in_dir / in_file}"
        with ErrorLogger(logger):
            raise FileNotFoundError(msg)

    # Determine whether to run `mergetime` or skip if there's only one file
    if len(merge_candidates) == 1:
        # Only a single file => skip `mergetime`
        single_file = merge_candidates[0]
        tmp_merge = single_file
        logger.info(f"Single file '{single_file.name}' found; skipping mergetime.")
    else:
        # Multiple files => run `mergetime` to concatenate along time dimension.
        # Use a unique temp name so concurrent/repeated runs in the same output
        # folder never collide.
        tmp_merge = p_out_dir / f"mergetime_{uuid.uuid4().hex}.nc"
        try:
            input_list_str = " ".join(str(p) for p in merge_candidates)
            logger.info(
                f"Running CDO mergetime on {_summarize_inputs_for_log(merge_candidates)} → {tmp_merge}"
            )
            cdo.mergetime(input=input_list_str, output=str(tmp_merge))
        except Exception as e:
            msg = f"CDO mergetime failed: {e}"
            with ErrorLogger(logger):
                raise RuntimeError(msg) from e

    # Finally, compute the time mean on tmp_merge
    final_name = out_file or "long_term_mean.nc"
    final_path = p_out_dir / final_name
    try:
        if lower_threshold is not None:
            logger.info(f"{lower_threshold=} selected.")
            logger.info(f"Calculating CDO timmean above {lower_threshold=}.")
            cdo.timmean(
                input=f"-setrtomiss,-1e20,{lower_threshold} {tmp_merge}",
                output=str(final_path),
            )
        else:
            logger.info(f"Running CDO timmean on {tmp_merge} → {final_path}")
            cdo.timmean(input=str(tmp_merge), output=str(final_path))
    except Exception as e:
        msg = f"CDO timmean failed: {e}"
        with ErrorLogger(logger):
            raise RuntimeError(msg) from e

    # Remove intermediate files if requested
    if not keep_temporal_files:
        # Remove cropped files
        if crop and crop_dir is not None:
            for f in crop_dir.glob("*"):
                f.unlink()
            crop_dir.rmdir()

        # Remove aggregated files
        if aggregate and aggregation_dir is not None:
            for f in aggregation_dir.glob("*"):
                f.unlink()
            aggregation_dir.rmdir()

        # If mergetime was run, remove its output
        if isinstance(tmp_merge, Path) and tmp_merge.name.startswith("mergetime"):
            tmp_merge.unlink()
