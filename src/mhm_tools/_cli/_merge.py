"""Merge multiple NetCDF files into combined output files.

The command searches an input directory for files matching a name pattern and
merges them into a single NetCDF output. It can also preserve the top-level
folder structure and merge files recursively within each subfolder.

Authors
-------
- Simon Lüdke
"""

from pathlib import Path


def add_args(parser):
    """Add cli arguments for the merge subcommand.

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
        "--input-path",
        dest="input_path",
        required=True,
        help="The path to input files.",
    )
    optional.add_argument(
        "--input-name",
        required=False,
        default="*.*",
        help="Input file name. E.g. '*.nc' to merge only nc files or 'pre*' to merge only precipitation files.",
    )
    optional.add_argument(
        "-o", "--output-file", required=True, help="The name of the output file."
    )
    flags.add_argument(
        "-p",
        "--preserve-folders",
        required=False,
        action="store_true",
        help="Preserve the top level folder structure. Recusive merge inside.",
    )
    optional.add_argument("--ncpus", required=False, default=1, help="Number of CPUs.")


def run(args):
    """Merge NetCDF files from a folder into a single output file.

    Parameters
    ----------
    args : argparse.Namespace
        parsed command line arguments
    """
    from mhm_tools.pre.merge import merge_files

    input = Path(args.input_path)
    output = Path(args.output_file)
    merge_files(
        input_path=input,
        input_file_part=args.input_name,
        output=output,
        n_cpus=int(args.ncpus),
        preserve_folders=args.preserve_folders,
    )
