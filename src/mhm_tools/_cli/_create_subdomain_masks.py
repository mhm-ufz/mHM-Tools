"""
Create files containing subsets of the river network based on the provided input files.

The river network, provied as nc file containing basin ids, is split into subdomains based on the provided basin clusters.
These subdomains of the global network are independent and can be run in parallel.
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
        "-i",
        "--input_dir",
        required=True,
        help=("Path to the input files"),
    )
    
    # optional arguments
    parser.add_argument(
        "-o",
        "--output_dir",
        default="subdomain_river_masks_3min",
        help=("specify an output directory path"),
    )
    parser.add_argument(
        "-f",
        "--output_file_name",
        default="river_network_subdomain",
        help=(
            "stem for the output filename, which is then numbered for the different basin clusters"
        ),
    )
    parser.add_argument(
        "-b",
        "--basin_ids",
        default="hydro_merged_03min.nc",
        help=(
            "file containing unique basins ids for all river basins"
            "they need to be in variable 'basin'"
        ),
    )
    parser.add_argument(
        "-c",
        "--basin_clusters",
        default="basin_clusters.nc",
        help=(
            "file containing clustered basins ids e.g. of the 53 subbasins from PGB reference"
            "they need to be in variable 'mask' but can have any resolution"
        ),
    )
    parser.add_argument(
        "-l",
        "--land_mask",
        default="land_mask_remapped_03min.nc",
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
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        output_file_name=args.output_file_name,
        basin_id_file=args.basin_ids,
        basin_clusters=args.basin_clusters,
        land_mask=args.land_mask,
    )
