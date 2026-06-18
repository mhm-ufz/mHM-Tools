"""
Create an ASCII header.txt from a NetCDF or ASCII input file.

Authors
-------
- Simon Lüdke
"""

from pathlib import Path


def add_args(parser):
    """Add CLI arguments for the create_header subcommand."""
    required = parser.add_argument_group("required arguments")
    optional = parser.add_argument_group("optional arguments")
    required.add_argument(
        "-i",
        "--input-file",
        required=True,
        help="Path to input file (.nc or .asc).",
    )
    optional.add_argument(
        "-o",
        "--output-dir",
        required=False,
        default=".",
        help="Directory where header.txt is written. Default: current directory.",
    )
    optional.add_argument(
        "-m",
        "--mask-file",
        required=False,
        default=None,
        help="Mask file that can help provide better matching l0_coords",
    )
    optional.add_argument(
        "--mask-var",
        required=False,
        default="mask",
        help="Variable to be used from the mask file. If file only contains one variable that is used automatically.",
    )
    optional.add_argument(
        "--resolution",
        required=False,
        default=None,
        help="Optional resolution to avoid floating point errors.",
    )


def run(args):
    """Create header.txt from an input dataset."""
    import logging

    from mhm_tools.common.file_handler import create_header, get_xarray_ds_from_file
    from mhm_tools.common.resolution_handler import Resolution, get_file_res
    from mhm_tools.common.xarray_utils import get_coord_key, get_single_data_var

    logger = logging.getLogger(__name__)

    input_file = Path(args.input_file)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    def _get_mask_corners(mask_file, mask_var, resolutions):
        xllcorner = None
        yllcorner = None
        if mask_file is None:
            return xllcorner, yllcorner
        try:
            with get_xarray_ds_from_file(mask_file) as mask_ds:
                if mask_var is None or mask_var not in mask_ds:
                    mask_var = get_single_data_var(mask_ds)
                mask_da = mask_ds[mask_var]
                mask_lon = get_coord_key(mask_da, lon=True)
                mask_lat = get_coord_key(mask_da, lat=True)
                file_res = get_file_res(
                    mask_ds[mask_lon], mask_ds[mask_lat], resolutions=resolutions
                )
                mask_header = create_header(mask_ds[mask_var], cellsize=file_res)
                xllcorner = mask_header["xllcorner"]
                yllcorner = mask_header["yllcorner"]
                logger.debug(
                    f"Using xllcorner {xllcorner} and yllcorner {yllcorner} from mask header"
                )
        except Exception as e:
            logger.warning(f"Failed to create header for mask file: {e}")
        return xllcorner, yllcorner

    with get_xarray_ds_from_file(input_file) as ds:
        lon_key = get_coord_key(ds, lon=True)
        lat_key = get_coord_key(ds, lat=True)
        resolutions = Resolution(l1=args.resolution)
        file_res = get_file_res(ds[lon_key], ds[lat_key], resolutions=resolutions)
        xllcorner, yllcorner = _get_mask_corners(
            args.mask_file, mask_var=args.mask_var, resolutions=resolutions
        )
        create_header(
            ds,
            output_path=output_dir,
            cellsize=file_res,
            xllcorner=xllcorner,
            yllcorner=yllcorner,
        )
