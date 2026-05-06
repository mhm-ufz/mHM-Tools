"""
Create files containing subsets of the river network based on the provided input files.

The river network, provied as nc file containing basin ids, is split
into subdomains based on the provided basin clusters. These subdomains
of the global network are independent and can be run in parallel.
"""


def add_args(parser):
    """Add CLI arguments for the create_subdomain_masks subcommand.

    Parameters
    ----------
    parser : argparse.ArgumentParser
        the main argument parser
    """
    required_args = parser.add_argument_group("required arguments")
    optional = parser.add_argument_group("optional arguments")
    required_args.add_argument(
        "-o",
        "--output-dir",
        required=True,
        help=("specify an output directory path"),
    )
    required_args.add_argument(
        "-b",
        "--basin-ids",
        required=True,
        help=(
            "file containing unique basin ids for all river basins; "
            "variable 'basin' must exist in file"
        ),
    )
    required_args.add_argument(
        "-c",
        "--basin-clusters",
        default=None,
        required=True,
        help=(
            "file containing clustered basin ids (e.g., PGB reference); "
            "variable 'mask' must exist"
        ),
    )
    required_args.add_argument(
        "-l",
        "--land-mask",
        required=True,
        help=(
            "file containing land surface mask at target resolution default variable name is 'land_mask'"
        ),
    )
    optional.add_argument(
        "--land-mask-variable",
        "--land_mask_var",
        dest="land_mask_variable",
        default="land_mask",
        required=False,
        help=("variable name in the land mask file containing the land surface mask; "),
    )
    optional.add_argument(
        "-f",
        "--output-file-name",
        default="subdomain_masks",
        required=False,
        help=(
            "stem for the output filename, which is then numbered for the different basin clusters"
        ),
    )


def run(args):
    """Create the subdomain mask files.

    Parameters
    ----------
    args : argparse.Namespace
        parsed command line arguments
    """
    from ..pre import create_subdomain_masks

    create_subdomain_masks(
        output_dir=args.output_dir,
        output_file_name=args.output_file_name,
        basin_id_file=args.basin_ids,
        basin_clusters=args.basin_clusters,
        land_mask=args.land_mask,
        land_mask_variable=args.land_mask_variable,
    )
