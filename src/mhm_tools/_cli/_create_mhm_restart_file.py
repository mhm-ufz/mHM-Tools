"""
Create restart files for the mHM model.

A restart file contains all the static information to run mHM on a specific grid.

"""

from mhm_tools.common.logger import logger, set_log_level
from mhm_tools.pre.create_mhm_restart_file import Grid, LatLon, MPRRunner

from ..pre import MHMRestartFile


def add_args(parser):
    """Add cli arguments for the create_mhm_restart_file subcommand.

    Parameters
    ----------
    parser : argparse.ArgumentParser
        the main argument parser
    """
    required_args = parser.add_argument_group("required arguments")
    required_args.add_argument(
        "-i",
        "--input_dir",
        required=True,
        help=("Path to the input files"),
    )
    required_args.add_argument(
        "-o",
        "--output_dir",
        required=True,
        help=("output directory as path"),
    )
    required_args.add_argument(
        "-n", "--nml_template", required=True, help=("nml_template file for mPR")
    )

    required_args.add_argument(
        "--mpr",
        required=True,
        help=("path to the mPR executable"),
    )

    required_args.add_argument(
        "--l1_resolution",
        required=True,
        help=("""resolution of the mHM target grid in degrees"""),
    )

    parser.add_argument(
        "--coords",
        required=False,
        default=None,
        help=(
            """coordinates in the form of 'lon_min,lon_max,lat_min,lat_max,resolution_l0'
            required unless --mask_file is provided"""
        ),
    )
    parser.add_argument(
        "--lon_min",
        required=False,
        default=None,
        help=(
            """minimum longitude of the target grid
            required unless --mask_file is provided"""
        ),
    )

    parser.add_argument(
        "--lon_max",
        required=False,
        default=None,
        help=(
            """maximum longitude of the target grid
            required unless --mask_file is provided"""
        ),
    )

    parser.add_argument(
        "--lat_min",
        required=False,
        default=None,
        help=(
            """minimum latitude of the target grid
            required unless --mask_file is provided"""
        ),
    )

    parser.add_argument(
        "--lat_max",
        required=False,
        default=None,
        help=(
            """maximum latitude of the target grid
            required unless --mask_file is provided"""
        ),
    )

    parser.add_argument(
        "--l0_resolution",
        required=False,
        default=None,
        help=(
            """resolution of the morphological input data grid in degrees
            required unless --mask_file is provided"""
        ),
    )

    parser.add_argument(
        "--mask_file",
        required=False,
        default=None,
        help=(
            """path to the mask file, a .nc file with a variable 'mask' containing the grid mask at l0 resolution
            required unless --coords is provided"""
        ),
    )

    parser.add_argument(
        "-p",
        "--mpr_params",
        required=False,
        default=None,
        help=("path to the mPR parameters file"),
    )

    parser.add_argument(
        "--l1_increment",
        required=True,
        default=20,
        type=int,
        help=("integer increment for l1 resolution in number of cells"),
    )

    parser.add_argument(
        "--process_domain_as_one",
        dest="run_on_whole_domain",
        action="store_true",
        required=False,
        help=(
            "do not split the grid into subgrids based on the provided basin clusters. Make sure your is not to large to run in one piece"
        ),
    )
    parser.add_argument(
        "--use_split_grids",
        dest="use_split_grids",
        action="store_true",
        required=False,
        help=(
            "ommit the split files step and run MPR on existing split grids. Make sure the files cover the correct domain area"
        ),
    )
    parser.add_argument(
        "--no_merge",
        dest="no_merge",
        action="store_true",
        required=False,
        help=("do not merge the restart files after splitting the grid"),
    )

    parser.add_argument(
        "--merge_only",
        dest="merge_only",
        action="store_true",
        required=False,
        help=(
            "only merge the allready exisiting restart files, do not create new ones; opposite of --no_merge"
        ),
    )
    parser.add_argument(
        "-c",
        "--clean-up",
        dest="clean_up",
        action="store_true",
        required=False,
        help=("delete the temporary files created during the process"),
    )

    parser.add_argument(
        "--log_level",
        default="INFO",
        help=("Set the logging level"),
    )

    parser.add_argument(
        "--ncpus",
        default=1,
        type=int,
        help=("Number of CPUs to use"),
    )
    parser.add_argument(
        "--mpr_packages",
        default=None,
        help=("Packages to load using module load before running mPR"),
    )
    parser.add_argument(
        "--land_mask_file",
        default=None,
        help=("Land mask file to use for l1 resolution"),
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
    l0_resolution = mask.lon.values[1] - mask.lon.values[0]
    mask = mask.mask
    return (
        lon_min_target_grid,
        lon_max_target_grid,
        lat_min_target_grid,
        lat_max_target_grid,
        l0_resolution,
        mask,
    )


def run(args):
    """Create the catchment file.

    Parameters
    ----------
    args : argparse.Namespace
        parsed command line arguments
    """
    set_log_level(args.log_level)
    l1_resolution = float(args.l1_resolution)
    if args.coords is not None:
        coords = args.coords.split(",")
        lon_min_target_grid = float(coords[0])
        lon_max_target_grid = float(coords[1])
        lat_min_target_grid = float(coords[2])
        lat_max_target_grid = float(coords[3])
        l0_resolution = float(coords[4])
    else:
        lon_min_target_grid = args.lon_min
        lon_max_target_grid = args.lon_max
        lat_min_target_grid = args.lat_min
        lat_max_target_grid = args.lat_max
        l0_resolution = args.l0_resolution
    mask = None
    if args.mask_file is not None:
        (
            lon_min_target_grid,
            lon_max_target_grid,
            lat_min_target_grid,
            lat_max_target_grid,
            l0_resolution,
            mask,
        ) = get_coords_from_mask(args.mask_file)
    elif (
        lon_min_target_grid is None
        or lon_max_target_grid is None
        or lat_min_target_grid is None
        or lat_max_target_grid is None
        or l0_resolution is None
    ):
        raise ValueError(
            "Either all coordinat bounds and resolutions or --mask_file must be provided"
        )
    
    logger.info(f"Creating restart file for grid with the following coordinates:")
    logger.info(f"lon_min: {lon_min_target_grid}")
    logger.info(f"lon_max: {lon_max_target_grid}")
    logger.info(f"lat_min: {lat_min_target_grid}")
    logger.info(f"lat_max: {lat_max_target_grid}")
    logger.info(f"l0_resolution: {l0_resolution}")
    logger.info(f"l1_resolution: {l1_resolution}")
    
    if args.land_mask_file is None and args.no_merge is not None:
        raise ValueError(
            "You need to provide a land mask file at L1 resolution if you want to merge the restart files"
        )

    l0 = LatLon(
        lon_min=float(lon_min_target_grid),
        lon_max=float(lon_max_target_grid),
        lat_min=float(lat_min_target_grid),
        lat_max=float(lat_max_target_grid),
        resolution=float(l0_resolution),
        mask=mask,
    )
    l1 = LatLon(
        lon_min=float(lon_min_target_grid),
        lon_max=float(lon_max_target_grid),
        lat_min=float(lat_min_target_grid),
        lat_max=float(lat_max_target_grid),
        resolution=l1_resolution,
    )

    grid = Grid(
        file_path=args.input_dir,
        name="whole grid",
        latlon_file=None,
        l0=l0,
        l1=l1,
        land_mask_file=args.land_mask_file,
    )
    restart_creator = MHMRestartFile(
        grid=grid,
        output_path=args.output_dir,
        nml_template=args.nml_template,
        increment_l1=args.l1_increment,
        mpr=MPRRunner(
            mpr_executable=args.mpr,
            mpr_packages=args.mpr_packages,
            mpr_parameter_file=args.mpr_params,
        ),
        run_on_whole_domain=args.run_on_whole_domain,
        use_split_grids=args.use_split_grids,
        clean_temp_files=args.clean_up,
        ncpus=args.ncpus,
        merge=not args.no_merge,
        merge_only=args.merge_only,
    )
    restart_creator.create_restart_file()
