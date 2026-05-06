"""Link all files in a folder tree to another folder tree creating symlinks for each file."""

import logging
from pathlib import Path

logger = logging.getLogger("mhm_tools.link_folder_tree")


def add_args(parser):
    """Add cli arguments for the link_folder_tree subcommand.

    Parameters
    ----------
    parser : argparse.ArgumentParser
        the main argument parser

    """
    optional = parser.add_argument_group("optional arguments")
    flags = parser.add_argument_group("flags")
    optional.add_argument(
        "-i",
        "--input-dir",
        required=True,
        help="The path to input directory.",
    )
    optional.add_argument(
        "-o", "--output-dir", required=True, help="The name of the output directory."
    )
    flags.add_argument(
        "--overwrite",
        action="store_true",
        required=False,
        help=("overwrite existing symlinks"),
    )
    optional.add_argument(
        "--file-name",
        type=str,
        default="*.*",
        required=False,
        help=("File name pattern to link, default is '*.*'"),
    )


def run(args):
    """Create a symlinked copy of a folder tree.

    Parameters
    ----------
    args : argparse.Namespace
        parsed command line arguments
    """
    from mhm_tools.pre.link_folder_tree import link_folder_tree

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    link_folder_tree(
        input_dir=input_dir,
        output_dir=output_dir,
        overwrite=args.overwrite,
        file_name=args.file_name,
    )
