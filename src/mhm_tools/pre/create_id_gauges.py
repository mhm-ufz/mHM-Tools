"""Create mRM idgauges files from gauge coordinates.

The module places one or more gauge IDs on a target grid, optionally snapping
the location to the maximum flow-accumulation cell near the provided
coordinates, and writes the result as an ASCII idgauges map.

Authors
-------
- Simon Lüdke
"""

import logging
from pathlib import Path

import xarray as xr

from mhm_tools.common.constants import NO_DATA
from mhm_tools.common.file_handler import get_xarray_ds_from_file, write_xarray_to_ascii
from mhm_tools.common.logger import ErrorLogger, log_arguments
from mhm_tools.common.xarray_utils import get_coord_key, get_single_data_var

logger = logging.getLogger(__name__)


def write_gauge_id(
    ds, id, lat, lon, facc_file=None, data_var=None
):  # , threshold=None, facc_value=None ):
    """Set gauge_id in ds."""
    if facc_file:
        with get_xarray_ds_from_file(file_path=facc_file) as ds_facc:
            lon_key = get_coord_key(ds_facc, lon=True)
            lat_key = get_coord_key(ds_facc, lat=True)
            if "uparea_grid" in ds_facc:
                var = ds_facc["uparea_grid"]
            elif "facc" in ds_facc:
                var = ds_facc["facc"]
            else:
                msg = "Neither uparea_grid nor facc in dataset datavars."
                with ErrorLogger(logger):
                    raise ValueError(msg)
            var_max = var.where(var == var.max(), drop=True)
            # Extract x and y coordinate values
            y_coord = var_max[lat_key].values[0]
            x_coord = var_max[lon_key].values[0]
        logger.info(f"Provided coords {lon} / {lat}")
        logger.info(f"Max facc coords {x_coord} / {y_coord}")
        lat = y_coord
        lon = x_coord

    lon_key = get_coord_key(ds, lon=True)
    lat_key = get_coord_key(ds, lat=True)
    if isinstance(ds, xr.Dataset):
        if data_var is None:
            data_var = get_single_data_var(ds)
            if data_var is None:
                msg = "Dataset has multiple data_vars which is incompatible."
                with ErrorLogger(logger):
                    raise ValueError(msg)
        ds[data_var].loc[
            ds.sel({lat_key: lat, lon_key: lon}, method="nearest").coords
        ] = id
    else:
        ds.loc[ds.sel({lat_key: lat, lon_key: lon}, method="nearest").coords] = id
        if data_var is None:
            data_var = ds.name if ds.name else "data"
        ds = ds.to_dataset(name=data_var)
    return ds.astype({data_var: int})


@log_arguments()
def create_id_gauges(
    id, lon, lat, file, out_path, file_is_idgauges=False, facc_file=None
):
    """Create id gauges file."""
    file = Path(file)
    out_path = Path(out_path)
    with get_xarray_ds_from_file(file) as ds:
        data_name = next(iter(ds.keys()))
        # if "nodata_value" in ds[data_name].attrs:
        #     missing_value = ds[data_name].attrs["nodata_value"]
        # else:
        #     missing_value = ds[data_name].encoding.get("_FillValue", int(NO_DATA))
        missing_value = int(NO_DATA)
        logger.info(f"Missing values is {missing_value}")
        if not file_is_idgauges:
            for var_name in ds.data_vars:
                # Set every element of this variable to missing_value:
                ds[var_name].values[:] = missing_value
            contains_value = False
        else:
            contains_value = bool(ds[data_name] == float(id)).any()
        if not contains_value:
            ds_with_id = write_gauge_id(
                ds, id, lat, lon, facc_file, data_var=data_name
            )  # , threshold, facc_value)
            write_xarray_to_ascii(ds_with_id, out_path, data_name)  # , fmt="%.0f")
        else:
            logger.info("Id {id} is already in {file}.")
