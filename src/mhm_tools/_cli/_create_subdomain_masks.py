"""
Create files containing subsets of the river network based on the provided
input files.

The river network, provied as nc file containing basin ids, is split
into subdomains based on the provided basin clusters. These subdomains
of the global network are independent and can be run in parallel.
"""

from ..pre import create_subdomain_masks


def add_args(parser):
    """Add cli arguments for the create_catchment subcommand.

    Parameters
    ----------
    parser : argparse.ArgumentParser
        the main argument parser
    """
    required_args = parser.add_argument_group("required arguments")
    required_args.add_argument(
        "-o",
        "--output_dir",
        help=("specify an output directory path"),
    )
    required_args.add_argument(
        "-f",
        "--output_file_name",
        help=(
            "stem for the output filename, which is then numbered for the different basin clusters"
        ),
    )
    required_args.add_argument(
        "-b",
        "--basin_ids",
        help=(
            "file containing unique basins ids for all river basins"
            "they need to be in variable 'basin'"
        ),
    )
    required_args.add_argument(
        "-c",
        "--basin_clusters",
        default=None,
        help=(
            "file containing clustered basins ids e.g. of the 53 subbasins from PGB reference"
            "they need to be in variable 'mask' but can have any resolution"
        ),
    )
    required_args.add_argument(
        "-l",
        "--land_mask",
        help=(
            "File containing a mask of all land surfaces"
            "grid of target resolution, need to be in integer variable 'land_mask'"
        ),
    )


def run(args):
    """Create the catchment file.

    Parameters
    ----------
    args : argparse.Namespace
        parsed command line arguments
    """
    create_subdomain_masks(
        output_dir=args.output_dir,
        output_file_name=args.output_file_name,
        basin_id_file=args.basin_ids,
        basin_clusters=args.basin_clusters,
        land_mask=args.land_mask,
    )
