"""Create id gauges file."""

import logging
from pathlib import Path

import numpy as np

from mhm_tools.common.file_handler import get_xarray_ds_from_file, write_xarray_to_ascii
from mhm_tools.common.logger import log_arguments

logger = logging.getLogger(__name__)


@log_arguments()
def create_id_gauges(id, lon, lat, file, out_path, file_is_idgauges=False):
    """Create id gauges file."""
    file = Path(file)
    out_path = Path(out_path)
    with get_xarray_ds_from_file(file) as ds:
        data_name = next(iter(ds.keys()))
        if "nodata_value" in ds[data_name].attrs:
            missing_value = ds[data_name].attrs["nodata_value"]
        else:
            missing_value = ds[data_name].encoding.get("_FillValue", np.nan)
        if not file_is_idgauges:
            for var_name in ds.data_vars:
                # Set every element of this variable to missing_value:
                ds[var_name].values[:] = missing_value
                contains_value = False
        else:
            contains_value = bool(ds[data_name] == float(id)).any()
        if not contains_value:
            ds.loc[ds.sel(lon=lon, lat=lat, method="nearest").coords] = id
            write_xarray_to_ascii(ds, out_path, data_name, fmt="%.0f")
        else:
            logger.info("Id {id} is already in {file}.")
