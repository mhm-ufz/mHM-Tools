"""Link all files in a folder tree to another folder tree creating symlinks for each file."""

import logging
from pathlib import Path

from mhm_tools.pre.link_folder_tree import link_folder_tree

logger = logging.getLogger("mhm_tools.link_folder_tree")


def add_args(parser):
    """Add cli arguments for the link_folder_tree subcommand.

    Parameters
    ----------
    parser : argparse.ArgumentParser
        the main argument parser

    """
    parser.add_argument(
        "-i",
        "--input_dir",
        required=True,
        help="The path to input directory.",
    )
    parser.add_argument(
        "-o", "--output_dir", required=True, help="The name of the output directory."
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        required=False,
        help=("overwrite existing symlinks"),
    )
    parser.add_argument(
        "--file_name",
        type=str,
        default="*.*",
        required=False,
        help=("File name pattern to link, default is '*.*'"),
    )


def run(args):
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    link_folder_tree(
        input_dir=input_dir,
        output_dir=output_dir,
        overwrite=args.overwrite,
        file_name=args.file_name,
    )
