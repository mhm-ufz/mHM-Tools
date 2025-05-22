"""
Calculate long-term means for input NetCDF files given.

This function reads one or more CF-compliant NetCDF datasets and returns
the time-mean field as an xarray.DataArray.

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
    """Aggregate a single NetCDF file in `input_path` named `file_name` into a
    coarser temporal resolution using CDO, writing to `output_path`.
    """
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
) -> None:
    """Compute long-term means for NetCDF forcing data.

    Optionally perform temporal aggregation and then merge, or merge raw inputs first.
    """
    p_in_dir = Path(in_dir)
    files = sorted(p_in_dir.glob(in_file))
    if not files:
        msg = f"No files match pattern {p_in_dir / in_file}"
        with ErrorLogger(logger):
            raise FileNotFoundError(msg)
    if crop and None in (lon_min, lon_max, lat_min, lat_max):
        msg = "All lon/lat bounds must be provided when crop=True."
        with ErrorLogger(logger):
            raise ValueError(msg)

    p_out_dir = Path(out_dir)
    p_out_dir.mkdir(parents=True, exist_ok=True)

    if crop:
        crop_dir = p_out_dir / "cropped_files"
        crop_dir.mkdir(parents=True, exist_ok=True)
        lonslice = slice(lon_min, lon_max)
        latslice = slice(lat_max, lat_min)
        p_crop_dir = Path(crop_dir)
        for fpath in files:
            crop_file(
                input_path=p_in_dir,
                input_file=fpath,
                output_path=p_crop_dir,
                mask_da=False,
                overwrite=True,
                available_mem_gib=None,
                latslice=latslice,
                lonslice=lonslice,
            )
        p_input_dir = p_crop_dir
    else:
        p_input_dir = p_in_dir

    cdo = Cdo()

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
        for fpath in files_to_aggregate:
            aggregate_files(
                input_path=p_input_dir,
                file_name=fpath.name,
                output_path=aggregation_dir,
                aggregation_type=aggregation_type,
                long_term_mean_type=long_term_mean_type,
            )
        merge_pattern = str(aggregation_dir / f"{cdo_op}_{Path(in_file).name}")
    else:
        merge_pattern = str(p_input_dir / Path(in_file).name)

    tmp_merge = p_out_dir / "mergetime.nc"
    try:
        logger.info(f"Running CDO mergetime on {merge_pattern} → {tmp_merge}")
        cdo.mergetime(input=merge_pattern, output=str(tmp_merge))
    except Exception as e:
        msg = f"CDO mergetime failed: {e}"
        with ErrorLogger(logger):
            raise RuntimeError(msg) from e

    final_name = out_file or "long_term_mean.nc"
    final_path = p_out_dir / final_name
    try:
        logger.info(f"Running CDO timmean on {tmp_merge} → {final_path}")
        cdo.timmean(input=str(tmp_merge), output=str(final_path))
    except Exception as e:
        msg = f"CDO timmean failed: {e}"
        with ErrorLogger(logger):
            raise RuntimeError(msg) from e

    if not keep_temporal_files:
        if crop:
            for f in (p_out_dir / "cropped_files").glob("*"):
                f.unlink()
            (p_out_dir / "cropped_files").rmdir()
        if aggregate:
            for f in aggregation_dir.glob("*"):
                f.unlink()
            aggregation_dir.rmdir()
        tmp_merge.unlink()
