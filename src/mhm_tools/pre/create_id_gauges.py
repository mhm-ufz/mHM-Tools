

from pathlib import Path
from mhm_tools.common.file_handler import get_xarray_ds_from_file, write_xarray_to_ascii
from mhm_tools.common.xarray_utils import get_single_data_var
import numpy as np
import logging
logger = logging.getLogger(__name__)

def create_id_gauges(id, lon, lat, file, out_path, file_is_idgauges=False):
    file = Path(file)
    out_path = Path(out_path)
    with get_xarray_ds_from_file(file) as ds:
        if 'nodata_value' in ds.attrs:
            missing_value = ds.attrs['nodata_value']
        else:
            first_var = list(ds.data_vars)[0]
            missing_value = ds[first_var].encoding.get('_FillValue', np.nan)
        if not file_is_idgauges:
            for var_name in ds.data_vars:
                # Set every element of this variable to missing_value:
                ds[var_name].values[:] = missing_value
                contains_value = False
        else:               
            contains_value = bool(ds[data_var] == float(id)).any()
        data_var = get_single_data_var(ds)
        if not contains_value:
            nearest_lon = ds['lon'].sel(lon=lon, method='nearest')#, tolerance=tolerance)
            # Find the nearest actual lat in the dataset
            nearest_lat = ds['lat'].sel(lat=lat, method='nearest')#, tolerance=tolerance)
            
            # Convert from DataArray to scalar for indexing
            nearest_lon_val = nearest_lon.item()  # or float(nearest_lon.values)
            nearest_lat_val = nearest_lat.item()
            # TODO: Compare flow acc at this coordinate with known flow accumulation by using the basin_ids.nc file
            logger.info(f"{nearest_lat_val}, {nearest_lon_val}")
            logger.info(id)
            
            # ds[data_var] = ds[data_var].astype(int)
            ds[data_var].loc[dict(lon=nearest_lon_val, lat=nearest_lat_val)] = id
            logger.info(ds[data_var].loc[dict(lon=nearest_lon_val, lat=nearest_lat_val)])
            write_xarray_to_ascii(ds, out_path, data_var, fmt="%.0f")
        else:
            logger.info('Id {id} is already in {file}.')

    
        