"""Convert merge multiple nc files."""

from pathlib import Path

from mhm_tools.pre.merge import merge_files


def add_args(parser):
    """Add cli arguments for the hydrograph subcommand.

    Parameters
    ----------
    parser : argparse.ArgumentParser
        the main argument parser

    """
    parser.add_argument(
        "-i",
        "--input_path",
        required=True,
        help="The path to input files.",
    )
    parser.add_argument(
        "--input_name",
        required=False,
        default="*.*",
        help="Input file name. E.g. '*.nc' to merge only nc files or 'pre*' to merge only precipitation files.",
    )
    parser.add_argument(
        "-o", "--output_file", required=True, help="The name of the output file."
    )
    parser.add_argument("--n_cpus", required=False, default=1, help="Number of CPUs.")


def run(args):
    input = Path(args.input_path)
    output = Path(args.output_file)
    merge_files(
        input_path=input,
        input_file_part=args.input_name,
        output=output,
        n_cpus=int(args.n_cpus),
    )
