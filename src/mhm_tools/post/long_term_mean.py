"""
Calculate lon term means for input NetCDF files given

Authors
-------
- Jeisson Leal
"""
import logging
import glob
import os
from pathlib import Path
from typing import Optional

from mhm_tools.common.logger import ErrorLogger, log_arguments
from mhm_tools.common.file_handler import get_xarray_ds_from_file
from mhm_tools.pre.crop_mhm_setup import crop_file

logger = logging.getLogger(__name__)

def cal_long_term_mean(
    in_dir: str,
    in_file: str,
    out_dir: str,
    var_name: str,
    long_term_mean_type: str = "monthly",
    aggregation_type: str = "intensive",
    keep_temporal_files: bool = False,
    out_file: Optional[str] = None,
    crop: bool = False,
    lon_min: Optional[float] = None,
    lon_max: Optional[float] = None,
    lat_min: Optional[float] = None,
    lat_max: Optional[float] = None,
) -> None:
    """
    Compute long-term means (hourly, daily, monthly, yearly) for NetCDF forcing data.

    Parameters
    ----------
    in_dir : str
        Directory containing input NetCDF files.
    in_file : str
        Filename or glob pattern for input files.
    out_dir : str
        Directory to write output file.
    var_name : str
        Variable name: 2t, tp, or tprate.
    long_term_mean_type : str
        One of 'hourly', 'daily', 'monthly', 'yearly'. Defaults to 'monthly'.
    aggregation_type : str
        'intensive' for mean, 'extensive' for sum. Defaults to 'intensive'.
    keep_temporal_files : bool
        Whether to retain intermediate temporal files. Defaults to False.
    out_file : Optional[str]
        Output filename. Defaults to 'long_term_mean_<var_name>.nc'.
    crop : bool
        Whether to crop spatially.
    lon_min, lon_max, lat_min, lat_max : Optional[float]
        Geographic bounds for cropping.
    """
    # build glob pattern and list files
    pattern = os.path.join(in_dir, in_file)
    files = sorted(glob.glob(pattern))
    if not files:
        with ErrorLogger(logger):
            raise FileNotFoundError(f"No files match pattern {pattern}")
    if None in (lon_min, lon_max, lat_min, lat_max):
        with ErrorLogger(logger):
            msg = "All lon/lat bounds must be provided when crop=True."
            raise ValueError(msg)

    # ensure output directory exists
    os.makedirs(out_dir, exist_ok=True)

    # Crop spatially first if requested
    if crop:
        # make a subdirectory for the cropped files
        crop_dir = os.path.join(out_dir, "cropped_files")
        os.makedirs(crop_dir, exist_ok=True)

        # pre-build slice objects
        lonslice = slice(lon_min, lon_max, None)
        latslice = slice(lat_max, lat_min, None)

        # wrap dirs in Path so crop_file can use .relative_to()
        p_input_dir = Path(in_dir)
        p_crop_dir  = Path(crop_dir)

        for file_name in files:
            # wrap each filename in a Path as well
            p_file = Path(file_name)
            crop_file(
                input_path=p_input_dir,
                input_file=p_file,
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

    # Select aggregator
    if aggregation_type == "intensive":
        agg_method = 'mean'
    elif aggregation_type == "extensive":
        agg_method = 'sum'
    else:
        with ErrorLogger(logger):
            raise ValueError(f"Invalid aggregation_type: {aggregation_type}")

    # Compute long-term mean
    if long_term_mean_type == "hourly":
        result = getattr(da.groupby("time.hour"), agg_method)()
    elif long_term_mean_type == "daily":
        result = getattr(da.groupby("time.dayofyear"), agg_method)()
    elif long_term_mean_type == "monthly":
        result = getattr(da.groupby("time.month"), agg_method)()
    elif long_term_mean_type == "yearly":
        result = getattr(da, agg_method)(dim="time")
    else:
        with ErrorLogger(logger):
            raise ValueError(f"Invalid long_term_mean_type: {long_term_mean_type}")

