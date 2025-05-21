"""Calculate long term means for input NetCDF files given.

Authors
-------
- Jeisson Leal (refactored to remove var_name dependency)
"""

import glob
import logging
import os
from pathlib import Path
from typing import Optional

from cdo import Cdo

from mhm_tools.common.logger import ErrorLogger
from mhm_tools.pre.crop_mhm_setup import crop_file

logger = logging.getLogger(__name__)

# shared mapping for CDO operations
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
    """Aggregate a single NetCDF file in `input_path` named `file_name` into a
    coarser temporal resolution using CDO, writing to `output_path`."""
    if aggregation_type not in ("intensive", "extensive"):
        raise ValueError(
            f"aggregation_type must be 'intensive' or 'extensive', got {aggregation_type!r}"
        )
    if long_term_mean_type not in OP_MAP:
        raise ValueError(
            f"Unsupported long_term_mean_type {long_term_mean_type!r}; choose from {list(OP_MAP)}"
        )

    mean_op, sum_op = OP_MAP[long_term_mean_type]
    cdo_op = mean_op if aggregation_type == "intensive" else sum_op
    output_path.mkdir(parents=True, exist_ok=True)

    input_file = input_path / file_name
    if not input_file.exists():
        raise FileNotFoundError(f"Input file not found: {input_file}")
    out_filename = f"{cdo_op}_{file_name}"
    out_file = output_path / out_filename

    cdo = Cdo()
    try:
        logger.info(f"Running CDO {cdo_op} on {input_file} → {out_file}")
        getattr(cdo, cdo_op)(input=str(input_file), output=str(out_file))
    except Exception as e:
        with ErrorLogger(logger):
            raise RuntimeError(
                f"CDO operation '{cdo_op}' failed on file '{input_file}': {e}"
            ) from e


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
) -> None:
    """Compute long-term means for NetCDF forcing data.

    Optionally perform temporal aggregation and then merge, or merge raw
    inputs first.
    """
    pattern = os.path.join(in_dir, in_file)
    files = sorted(glob.glob(pattern))
    if not files:
        with ErrorLogger(logger):
            raise FileNotFoundError(f"No files match pattern {pattern}")
    if crop and None in (lon_min, lon_max, lat_min, lat_max):
        with ErrorLogger(logger):
            raise ValueError("All lon/lat bounds must be provided when crop=True.")

    os.makedirs(out_dir, exist_ok=True)

    # spatial crop
    if crop:
        crop_dir = os.path.join(out_dir, "cropped_files")
        os.makedirs(crop_dir, exist_ok=True)
        lonslice = slice(lon_min, lon_max)
        latslice = slice(lat_max, lat_min)
        p_input_dir = Path(in_dir)
        p_crop_dir = Path(crop_dir)
        for fpath in files:
            crop_file(
                input_path=p_input_dir,
                input_file=Path(fpath),
                output_path=p_crop_dir,
                mask_da=False,
                overwrite=True,
                available_mem_gib=None,
                latslice=latslice,
                lonslice=lonslice,
            )
        p_input_dir = p_crop_dir
    else:
        p_input_dir = Path(in_dir)

    cdo = Cdo()

    # choose merge pattern based on aggregation flag
    if aggregate:
        if long_term_mean_type not in OP_MAP:
            raise ValueError(
                f"Unsupported long_term_mean_type {long_term_mean_type!r}; choose from {list(OP_MAP)}"
            )
        if aggregation_type not in ("intensive", "extensive"):
            raise ValueError(
                f"aggregation_type must be 'intensive' or 'extensive', got {aggregation_type!r}"
            )
        mean_op, sum_op = OP_MAP[long_term_mean_type]
        cdo_op = mean_op if aggregation_type == "intensive" else sum_op

        agg_pattern = os.path.join(p_input_dir, in_file)
        files_to_aggregate = sorted(glob.glob(agg_pattern))
        aggregation_dir = os.path.join(out_dir, "aggregated_files")
        os.makedirs(aggregation_dir, exist_ok=True)
        for fn in files_to_aggregate:
            aggregate_files(
                input_path=p_input_dir,
                file_name=Path(fn).name,
                output_path=Path(aggregation_dir),
                aggregation_type=aggregation_type,
                long_term_mean_type=long_term_mean_type,
            )
        merge_pattern = os.path.join(aggregation_dir, f"{cdo_op}_{Path(in_file).name}")
    else:
        merge_pattern = os.path.join(p_input_dir, Path(in_file).name)

    # perform mergetime
    tmp_merge = os.path.join(out_dir, "mergetime.nc")
    try:
        logger.info(f"Running CDO mergetime on {merge_pattern} → {tmp_merge}")
        cdo.mergetime(input=merge_pattern, output=tmp_merge)
    except Exception as e:
        with ErrorLogger(logger):
            raise RuntimeError(f"CDO mergetime failed: {e}") from e

    # compute final time mean
    final_name = out_file or "long_term_mean.nc"
    final_path = os.path.join(out_dir, final_name)
    try:
        logger.info(f"Running CDO timmean on {tmp_merge} → {final_path}")
        cdo.timmean(input=str(tmp_merge), output=str(final_path))
    except Exception as e:
        with ErrorLogger(logger):
            raise RuntimeError(f"CDO timmean failed: {e}") from e

    # cleanup
    if not keep_temporal_files:
        if crop:
            for f in Path(out_dir, "cropped_files").glob("*"):
                f.unlink()
            Path(out_dir, "cropped_files").rmdir()
        if aggregate:
            for f in Path(aggregation_dir).glob("*"):
                f.unlink()
            Path(aggregation_dir).rmdir()
        Path(tmp_merge).unlink()
