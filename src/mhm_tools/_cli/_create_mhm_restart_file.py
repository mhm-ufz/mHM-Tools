"""
Create restart files for the mHM model.

A restart file contains all the static information to run mHM on a specific domain.

"""

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
        "--coords",
        required=True,
        help=(
            "coordinates in the form of 'lon_min,lon_max,lat_min,lat_max,resolution_l0,resolution_l1'"
        ),
    )

    required_args.add_argument(
        "--mpr",
        required=True,
        help=(
            "path to the mPR executable"
        ),
    )

    parser.add_argument(
        "--l1_increment",
        required=True,
        default=20,
        help=("increment for l1 resolution in number of cells"),
    )

    parser.add_argument(
        "-s",
        "--split",
        dest="split",
        action="store_true",
        required=False,
        help=(
            "split the domain into subdomains based on the provided basin clusters if your domain is to large to run in one piece"
        ),
    )


def run(args):
    """Create the catchment file.

    Parameters
    ----------
    args : argparse.Namespace
        parsed command line arguments
    """
    coords = args.coords.split(",")
    lon_min_target_grid = float(coords[0])
    lon_max_target_grid = float(coords[1])
    lat_min_target_grid = float(coords[2])
    lat_max_target_grid = float(coords[3])
    l0_resolution = float(coords[4])
    l1_resolution = float(coords[5])
    restart_creator = MHMRestartFile(
        input_file_path=args.input_dir,
        output_path=args.output_dir,
        nml_template=args.nml_template,
        lon_min_target_grid=lon_min_target_grid,
        lon_max_target_grid=lon_max_target_grid,
        lat_min_target_grid=lat_min_target_grid,
        lat_max_target_grid=lat_max_target_grid,
        l0_resolution=l0_resolution,
        l1_resolution=l1_resolution,
        increment_l1=args.l1_increment,
        mpr_executable=args.mpr,
    )
    restart_creator.create_restart_file()
