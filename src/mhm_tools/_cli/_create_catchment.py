"""
TODO: add description
"""

from pathlib import Path

from ..pre import create_catchment


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
        "--input_file",
        required=True,
        help=("Path to the input file"),
    )
    required_args.add_argument(
        "-o",
        "--output_path",
        required=True,
        help=(
            "Path to the output file. If a single file is written "
            "this is the path to this file, if multiple files are "
            "written, this is the prefix."
        ),
    )
    # optional
    parser.add_argument(
        "-S",
        "--split_file",
        action="store_false",
        default=True,
        help=(
            "Write out multiple files. By default a single file is written "
            "alternative is that the file is split up by variables."
        ),
    )
    parser.add_argument(
        "-C",
        "--creator",
        default=None,
        help=(
            "Level-2 (meteorology) information. Either an ascii (header) file, "
            "a dictionary containing the header information "
            "or a cell-size to determine information from level-0. "
            "Level-2 information wont be written to the latlon file."
        ),
    )


def run(args):
    """Create the catchment file.

    Parameters
    ----------
    args : argparse.Namespace
        parsed command line arguments
    """
    create_catchment(
        input_file=args.input_file,
        output_path=args.output_path,
    )
