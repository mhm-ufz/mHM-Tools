"""Cut domains out of an existing mHM setup."""

import logging

logger = logging.getLogger(__name__)


def add_args(parser):
    """Add cli arguments for the crop_mhm_setup subcommand.

    Parameters
    ----------
    parser : argparse.ArgumentParser
        the main argument parser
    """
    required_args = parser.add_argument_group("required arguments")
    required_args.add_argument(
        "-i",
        "--input-path",
        required=True,
        help="Path to the directory of the existing mHM setup. \
        Can also be used with a file path to crop a single file.",
    )
    required_args.add_argument(
        "-o",
        "--output-path",
        required=True,
        help="Path of the directory where the new domain setup should be saved.",
    )
    optional = parser.add_argument_group("optional arguments")
    flags = parser.add_argument_group("flags")
    optional.add_argument(
        "-f",
        "--file-name",
        required=False,
        default="*.*",
        help="Input file name. E.g. '*.nc' to copy only nc files or 'pre*' to copy only precipitation files. If the file has a header in it's folder the header is reproduced regardless of wether nor not it fits the filename.",
    )
    optional.add_argument(
        "-m",
        "--mask-file",
        required=False,
        help="The path the the mask file. Mask files can be created using the catchment command with the --mask flag.",
    )
    optional.add_argument(
        "--l1-resolution",
        required=False,
        help=("Hydrological resolution. Without it no latlon file can be produced."),
    )
    optional.add_argument(
        "--l11-resolution",
        required=False,
    )
    optional.add_argument(
        "--crs",
        default=None,
        help=(
            "Coordinates reference system (e.g. 'epsg:3035'). Needed to create a new latlon file."
            "If not given, headers will be interpreted as given in lat-lon ('epsg:4326')."
        ),
    )
    optional.add_argument(
        "--ncpus",
        required=False,
        default=1,
        type=int,
        help=("Number of cores used for parallelisation."),
    )
    optional.add_argument(
        "--folder-recursion-depth",
        required=False,
        default=5,
        type=int,
        help=("How deep in the folder structure should the file be searched?"),
    )
    optional.add_argument(
        "--lonlatbox",
        required=False,
        default=None,
        help=(
            """coordinates in the form of 'lon_min,lon_max,lat_min,lat_max,resolution_l0'
            required unless --mask_file is provided"""
        ),
    )
    optional.add_argument(
        "--available-mem",
        required=False,
        default="5",
        help=("""Available memory per cpu in Gb or Mb (default Gb)"""),
    )
    flags.add_argument(
        "--chunking",
        action="store_true",
        help=("""Set if each dataset should be read as a chunked dask array."""),
    )
    optional.add_argument(
        "--lon-min",
        required=False,
        default=None,
        help=("""minimum longitude of the target grid
            required unless --mask_file is provided"""),
    )

    optional.add_argument(
        "--lon-max",
        required=False,
        default=None,
        help=("""maximum longitude of the target grid
            required unless --mask_file is provided"""),
    )

    optional.add_argument(
        "--lat-min",
        required=False,
        default=None,
        help=("""minimum latitude of the target grid
            required unless --mask_file is provided"""),
    )

    optional.add_argument(
        "--lat-max",
        required=False,
        default=None,
        help=("""maximum latitude of the target grid
            required unless --mask_file is provided"""),
    )
    flags.add_argument(
        "--create-header",
        required=False,
        default=False,
        action="store_true",
        help=("""Force creation of header file for all files."""),
    )
    flags.add_argument(
        "--no-cropping",
        required=False,
        default=False,
        action="store_true",
        help=(
            """Do not crop the file but only writes a header. This activate the forced header creation and only header functions."""
        ),
    )
    optional.add_argument(
        "--output-var",
        required=False,
        default=None,
        help=("""Output variable name for single data var"""),
    )
    optional.add_argument(
        "--mask-var", required=False, default="mask", help=("""Mask variable name.""")
    )
    optional.add_argument(
        "--lat-order",
        required=False,
        default="decreasing",
        help=(
            """Direction of the latitude coordinate. Will be forced if input has coordinates."""
        ),
    )
    optional.add_argument(
        "--output-suffix",
        required=False,
        default=None,
        help=("""Suffix added to output file names leading to file type conversion."""),
    )


def run(args):
    """Cut out a domain setup out of an existing mHM setup.

    Parameters
    ----------
    args : argparse.Namespace
        parsed command line arguments
    """
    from mhm_tools.common.cli_utils import get_available_mem_in_unit, get_coords
    from mhm_tools.common.logger import ErrorLogger
    from mhm_tools.pre.crop_mhm_setup import crop_mhm_setup

    (
        lon_min_target_grid,
        lon_max_target_grid,
        lat_min_target_grid,
        lat_max_target_grid,
        mask_da,
    ) = get_coords(
        args.lonlatbox,
        args.mask_file,
        args.lon_min,
        args.lon_max,
        args.lat_min,
        args.lat_max,
        mask_var=args.mask_var,
    )
    if args.lat_order == "decreasing":
        latslice = slice(lat_max_target_grid, lat_min_target_grid)
    elif args.lat_order == "increasing":
        latslice = slice(lat_min_target_grid, lat_max_target_grid)
    else:
        msg = f"Unknown lat_order {args.lat_order!r}. Use 'increasing' or 'decreasing'."
        with ErrorLogger(logger):
            raise ValueError(msg)
    lonslice = slice(lon_min_target_grid, lon_max_target_grid)
    # l0_resolution = float(args.lonlatbox.split(",")[4])
    available_mem = get_available_mem_in_unit(args.available_mem)
    crop_mhm_setup(
        input_path=args.input_path,
        output_path=args.output_path,
        mask_da=mask_da,
        l1_resolution=args.l1_resolution,
        l11_resolution=args.l11_resolution,
        lonslice=lonslice,
        latslice=latslice,
        crs=args.crs,
        n_jobs=args.ncpus,
        filename=args.file_name,
        available_mem_gib=available_mem,
        force_header_creation=args.create_header,
        chunking=args.chunking,
        output_var=args.output_var,
        no_cropping=args.no_cropping,
        lat_order=args.lat_order,
        output_suffix=args.output_suffix,
    )
