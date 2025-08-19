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
from pathlib import Path
from typing import Optional

from cdo import Cdo

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


def aggregate_files(
    input_path: Path,
    file_name: str,
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

    input_file = input_path / file_name
    if not input_file.exists():
        msg = f"Input file not found: {input_file}"
        raise FileNotFoundError(msg)
    out_filename = f"{cdo_op}_{file_name}"
    out_file = output_path / out_filename

    cdo = Cdo()
    try:
        logger.info(f"Running CDO {cdo_op} on {input_file} → {out_file}")
        getattr(cdo, cdo_op)(input=str(input_file), output=str(out_file))
    except Exception as e:
        msg = f"CDO operation '{cdo_op}' failed on file '{input_file}': {e}"
        with ErrorLogger(logger):
            raise RuntimeError(msg) from e


def cal_long_term_mean(
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

    # If cropping is requested, verify bounds
    if crop and None in (lon_min, lon_max, lat_min, lat_max):
        msg = "All lon/lat bounds must be provided when --crop is used."
        with ErrorLogger(logger):
            raise ValueError(msg)

    # Prepare output directory
    p_out_dir = Path(out_dir)
    p_out_dir.mkdir(parents=True, exist_ok=True)

    # If cropping, produce cropped files into a subfolder and use that as input
    if crop:
        crop_dir = p_out_dir / "cropped_files"
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
        p_input_dir = crop_dir
        # After cropping, redefine `files` to be the cropped set
        files = sorted(Path(crop_dir).glob(Path(in_file).name))
    else:
        p_input_dir = p_in_dir

    cdo = Cdo()
    pattern_name = Path(in_file).name

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

        files_to_aggregate = sorted(p_input_dir.glob(in_file))
        aggregation_dir = p_out_dir / "aggregated_files"
        aggregation_dir.mkdir(parents=True, exist_ok=True)

        # Run `aggregate_files` on each matched input
        for fpath in files_to_aggregate:
            aggregate_files(
                input_path=p_input_dir,
                file_name=fpath.name,
                output_path=aggregation_dir,
                aggregation_type=aggregation_type,
                long_term_mean_type=long_term_mean_type,
            )

        # Re-define the list of files that will eventually be merged:
        merged_pattern_dir = aggregation_dir
        merged_pattern_name = f"{cdo_op}_{pattern_name}"
        merge_candidates = sorted(merged_pattern_dir.glob(merged_pattern_name))

    else:
        # No aggregation: merge (or skip merging) the raw inputs
        merged_pattern_dir = p_input_dir
        merged_pattern_name = pattern_name
        merge_candidates = sorted(merged_pattern_dir.glob(merged_pattern_name))

    # Determine whether to run `mergetime` or skip if there's only one file
    if len(merge_candidates) == 1:
        # Only a single file => skip `mergetime`
        single_file = merge_candidates[0]
        tmp_merge = single_file
        logger.info(f"Single file '{single_file.name}' found; skipping mergetime.")
    else:
        # Multiple files => run `mergetime` to concatenate along time dimension
        tmp_merge = p_out_dir / f"mergetime_{pattern_name}"
        try:
            input_list_str = " ".join(str(p) for p in merge_candidates)
            logger.info(f"Running CDO mergetime on {input_list_str} → {tmp_merge}")
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
            tmp_masked = p_out_dir / f"above_threshold_{pattern_name}"
            # Mask all values below lower_threshold by setting them missing:
            cdo.setrtomiss(
                f"-1e20,{lower_threshold}", input=str(tmp_merge), output=str(tmp_masked)
            )
            cdo.timmean(input=str(tmp_masked), output=str(final_path))
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
        if crop:
            for f in (p_out_dir / "cropped_files").glob("*"):
                f.unlink()
            (p_out_dir / "cropped_files").rmdir()

        # Remove aggregated files
        if aggregate:
            for f in aggregation_dir.glob("*"):
                f.unlink()
            aggregation_dir.rmdir()

        # If mergetime was run, remove its output
        if isinstance(tmp_merge, Path) and tmp_merge.name.startswith("mergetime_"):
            tmp_merge.unlink()
