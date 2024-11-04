import argparse
import numpy as np


def parse_coords(coords_str):
    # Split the input string by comma and convert each part to a float
    try:
        lat, lon = map(float, coords_str.split(","))
        return lat, lon
    except ValueError:
        raise argparse.ArgumentTypeError(
            "Coordinates must be two comma-separated floats."
        )

def get_coords_from_mask(mask):
    """Get the coordinates from a mask file.

    Parameters
    ----------
    mask : str
        path to the mask file

    Returns
    -------
    tuple
        tuple containing the coordinates
    """
    import xarray as xr

    mask = xr.open_dataset(mask)
    lon_min_target_grid = mask.lon.values[0]
    lon_max_target_grid = mask.lon.values[-1]
    lat_min_target_grid = mask.lat.values[0]
    lat_max_target_grid = mask.lat.values[-1]
    
    # change values from center cell to corner values
    resolution = mask.lon.values[1] - mask.lon.values[0]
    lon_min_target_grid -= resolution/2
    lon_max_target_grid += resolution/2
    lat_min_target_grid -= resolution/2
    lat_max_target_grid += resolution/2

    # round values to get rid of inprecission
    lon_min_target_grid = np.round(lon_min_target_grid, 6)
    lon_max_target_grid = np.round(lon_max_target_grid, 6)
    lat_min_target_grid = np.round(lat_min_target_grid, 6)
    lat_max_target_grid = np.round(lat_max_target_grid, 6)

    if lat_min_target_grid > lat_max_target_grid:
        lat_min_target_grid, lat_max_target_grid = lat_max_target_grid, lat_min_target_grid
    if lon_min_target_grid > lon_max_target_grid:
        lon_min_target_grid, lon_max_target_grid = lon_max_target_grid, lon_min_target_grid
    mask = mask.mask
    return (
        lon_min_target_grid,
        lon_max_target_grid,
        lat_min_target_grid,
        lat_max_target_grid,
        mask,
    )

def get_coords(lonlatbox, mask_file, lon_min=None, lon_max=None, lat_min=None, lat_max=None, raise_exeption=True):
    mask = None
    if lonlatbox is not None:
        lonlatbox = lonlatbox.split(",")
        lon_min_target_grid = float(lonlatbox[0])
        lon_max_target_grid = float(lonlatbox[1])
        lat_min_target_grid = float(lonlatbox[2])
        lat_max_target_grid = float(lonlatbox[3])
    elif mask_file is not None:
        (
            lon_min_target_grid,
            lon_max_target_grid,
            lat_min_target_grid,
            lat_max_target_grid,
            mask,
        ) = get_coords_from_mask(mask_file)
    elif not (
        lon_min is None
        or lon_max is None
        or lat_min is None
        or lat_max is None
    ):
        lon_min_target_grid = lon_min
        lon_max_target_grid = lon_max
        lat_min_target_grid = lat_min
        lat_max_target_grid = lat_max
    else:
        if raise_exeption:
            raise ValueError(
                "Either all coordinat bounds and resolutions or --mask_file must be provided"
            )
        else: 
            return None, None, None, None, None
    return lon_min_target_grid, lon_max_target_grid, lat_min_target_grid, lat_max_target_grid, mask