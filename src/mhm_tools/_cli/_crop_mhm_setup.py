"""Cut domains out of an existing mHM setup."""

from mhm_tools.common.cli_utils import get_coords
from mhm_tools.common.xarray_utils import get_coord_key
from mhm_tools.pre.crop_mhm_setup import crop_mhm_setup


def add_args(parser):
    """Add cli arguments for the cut_mhm_setupt subcommand.

    Parameters
    ----------
    parser : argparse.ArgumentParser
        the main argument parser
    """
    required_args = parser.add_argument_group("required arguments")
    required_args.add_argument(
        "-m",
        "--mask_file",
        required=True,
        help="The path the the mask file. Mask files can be created using the catchment command with the --mask flag.",
    )
    required_args.add_argument(
        "-i",
        "--input_path",
        required=True,
        help="Path to the directory of the existing mHM setup. \
        Can also be used with a file path to crop a single file.",
    )
    required_args.add_argument(
        "-o",
        "--output_path",
        required=True,
        help="Path of the directory where the new domain setup should be saved.",
    )
    required_args.add_argument(
        "-f",
        "--file_name",
        required=False,
        default="*.*",
        help="Input file name. E.g. '*.nc' to copy only nc files or 'pre*' to copy only precipitation files. If the file has a header in it's folder the header is reproduced regardless of wether nor not it fits the filename.",
    )
    parser.add_argument(
        "--l1_resolution",
        required=False,
        help=("Hydrological resolution. Without it no latlon file can be produced."),
    )
    parser.add_argument(
        "--l11_resolution",
        required=False,
    )
    parser.add_argument(
        "--crs",
        default=None,
        help=(
            "Coordinates reference system (e.g. 'epsg:3035'). Needed to create a new latlon file."
            "If not given, headers will be intncverpreted as given in lat-lon ('epsg:4326')."
        ),
    )
    parser.add_argument(
        "--ncpus",
        required=False,
        default=1,
        help=("Number of cores used for parallelisation."),
    )
    parser.add_argument(
        "--folder_recursion_depth",
        required=False,
        default=5,
        help=("How deep in the folder structure should the file be searched?"),
    )
    parser.add_argument(
        "--lonlatbox",
        required=False,
        default=None,
        help=(
            """coordinates in the form of 'lon_min,lon_max,lat_min,lat_max,resolution_l0'
            required unless --mask_file is provided"""
        ),
    )


def run(args):
    """Cut out a domain setup out of an existing mHM setup..

    Parameters
    ----------
    args : argparse.Namespace
        parsed command line arguments
    """
    (
        lon_min_target_grid,
        lon_max_target_grid,
        lat_min_target_grid,
        lat_max_target_grid,
        mask,
    ) = get_coords(
        lonlatbox=args.lonlatbox,
        mask_file=args.mask_file,
    )
    if mask is not None:
        mask_da = mask.astype(float)
        lon_key_mask = get_coord_key(mask_da, lon=True)
        lat_key_mask = get_coord_key(mask_da, lat=True)
        pres = 1e-5
        l0_resolution = round(
            mask_da[lon_key_mask].data[1] - mask_da[lon_key_mask].data[0], 6
        )
        latslice = slice(
            mask_da[lat_key_mask].data[0] + l0_resolution / 2 + pres,
            mask_da[lat_key_mask].data[-1] - l0_resolution / 2 - pres,
        )
        lonslice = slice(
            mask_da[lon_key_mask].data[0] - l0_resolution / 2 - pres,
            mask_da[lon_key_mask].data[-1] + l0_resolution / 2 + pres,
        )
    elif args.lonlatbox is not None:
        latslice = (lat_max_target_grid, lat_min_target_grid)
        lonslice = (lon_min_target_grid, lon_max_target_grid)
        # l0_resolution = float(args.lonlatbox.split(",")[4])
    crop_mhm_setup(
        mask_da,
        args.output_path,
        args.input_path,
        l1_resolution=args.l1_resolution,
        l11_resolution=args.l11_resolution,
        lonslice=lonslice,
        latslice=latslice,
        crs=args.crs,
        n_jobs=args.ncpus,
        filename=args.file_name,
        recursive_depth=args.folder_recursion_depth,
    )
