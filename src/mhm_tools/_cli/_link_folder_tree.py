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


def run(args):
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    for file in input_dir.rglob("*.nc"):
        relative_path = file.relative_to(input_dir)
        output_file = output_dir / relative_path
        if output_file.exists() and not args.overwrite:
            continue
        if output_file.exists() and args.overwrite:
            output_file.unlink()
        output_file.parent.mkdir(parents=True, exist_ok=True)
        logger.info(f"Linking {file} to {output_file}")
        output_file.symlink_to(file.resolve())
