"""Create restart files for the mHM model.

A restart file contains all the static information to run mHM on a
specific grid.
"""

import logging

logger = logging.getLogger(__name__)


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
        "--input-dir",
        required=True,
        help=("Path to the input files"),
    )
    required_args.add_argument(
        "-o",
        "--output-dir",
        required=True,
        help=("output directory as path"),
    )
    optional = parser.add_argument_group("optional arguments")
    flags = parser.add_argument_group("flags")
    optional.add_argument(
        "-w",
        "--work-dir",
        required=False,
        default=None,
        help=("work directory as path"),
    )
    required_args.add_argument(
        "-n", "--nml-template", required=True, help=("nml_template file for mPR")
    )

    required_args.add_argument(
        "--mpr",
        required=True,
        help=("path to the mPR executable"),
    )

    required_args.add_argument(
        "--l1-resolution",
        required=True,
        help=("""resolution of the mHM target grid in degrees"""),
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

    optional.add_argument(
        "--l0-resolution",
        required=False,
        default=None,
        help=("""resolution of the morphological input data grid in degrees
            required unless --mask_file is provided"""),
    )

    optional.add_argument(
        "--mask-file",
        required=False,
        default=None,
        help=(
            """path to the mask file, a .nc file with a variable 'mask' containing the grid mask at l0 resolution
            required unless --lonlatbox is provided"""
        ),
    )

    optional.add_argument(
        "-p",
        "--mpr-params",
        required=False,
        default=None,
        help=("path to the mPR parameters file"),
    )

    optional.add_argument(
        "--l1-increment",
        required=True,
        default=20,
        type=int,
        help=("integer increment for l1 resolution in number of cells"),
    )

    flags.add_argument(
        "--process-domain-as-one",
        dest="run_on_whole_domain",
        action="store_true",
        required=False,
        help=(
            "do not split the grid into subgrids based on the provided basin clusters. Make sure your is not to large to run in one piece"
        ),
    )
    flags.add_argument(
        "--use-split-grids",
        dest="use_split_grids",
        action="store_true",
        required=False,
        help=(
            "ommit the split files step and run MPR on existing split grids. Make sure the files cover the correct domain area"
        ),
    )
    flags.add_argument(
        "--no-merge",
        dest="no_merge",
        action="store_true",
        required=False,
        help=("do not merge the restart files after splitting the grid"),
    )

    flags.add_argument(
        "--merge-only",
        dest="merge_only",
        action="store_true",
        required=False,
        help=(
            "only merge the allready exisiting restart files, do not create new ones; opposite of --no_merge"
        ),
    )
    flags.add_argument(
        "-c",
        "--clean-up",
        dest="clean_up",
        action="store_true",
        required=False,
        help=("delete the temporary files created during the process"),
    )
    optional.add_argument(
        "--ncpus",
        default=1,
        type=int,
        help=("Number of CPUs to use"),
    )
    optional.add_argument(
        "--mpr-packages",
        default=None,
        help=("Packages to load using module load before running mPR"),
    )
    optional.add_argument(
        "--land-mask-file",
        default=None,
        help=("Land mask file to use for l1 resolution"),
    )


def run(args):
    """Create mHM restart file(s) for the target grid using the mPR stand alone version.

    Parameters
    ----------
    args : argparse.Namespace
        parsed command line arguments
    """
    from pathlib import Path

    from mhm_tools.common.cli_utils import get_coords
    from mhm_tools.common.logger import ErrorLogger
    from mhm_tools.pre.create_mhm_restart_file import Grid, LatLon, MPRRunner

    from ..pre import MHMRestartFile

    l1_resolution = float(args.l1_resolution)

    if args.lonlatbox is not None:
        l0_resolution = float(args.lonlatbox.split(",")[4])
    elif args.l0_resolution is not None:
        l0_resolution = float(args.l0_resolution)
    else:
        msg = "L0 resolution was not provided."
        raise ValueError(msg)

    (
        lon_min_target_grid,
        lon_max_target_grid,
        lat_min_target_grid,
        lat_max_target_grid,
        mask,
    ) = get_coords(
        args.lonlatbox,
        args.mask_file,
        args.lon_min,
        args.lon_max,
        args.lat_min,
        args.lat_max,
    )

    if args.land_mask_file is None and args.no_merge is not None:
        with ErrorLogger(logger):
            no_land_mask_error = "You need to provide a land mask file at L1 resolution if you want to merge the restart files"
            raise ValueError(no_land_mask_error)

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
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    work_dir = Path(args.work_dir) if args.work_dir is not None else output_dir
    if not input_dir.is_dir():
        msg = f"Input dir {input_dir} is not a directory."
        with ErrorLogger(logger):
            raise ValueError(msg)
    grid = Grid(
        file_path=input_dir,
        name="whole_grid",
        latlon_file=None,
        l0=l0,
        l1=l1,
        land_mask_file=args.land_mask_file,
    )
    grid.migrate_grid_using_systemlink(work_dir / "full_grid")
    restart_creator = MHMRestartFile(
        grid=grid,
        output_path=output_dir,
        work_path=work_dir,
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
