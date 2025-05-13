"""Convert between ascii and netcdf by file suffix."""
from mhm_tools.common.file_handler import write_xarray_to_file, get_xarray_ds_from_file


def add_args(parser):
    """Add cli arguments for the hydrograph subcommand.

    Parameters
    ----------
    parser : argparse.ArgumentParser
        the main argument parser

    """
    parser.add_argument(
            "-i",
            "--input",
            required=True,
            help="The path to input file.",
        )
    parser.add_argument(
        "-o",
        "--output",
        required=True,
        help="The name of the output file."
    )

def run(args):
    ds = get_xarray_ds_from_file(args.input)
    write_xarray_to_file(ds, args.output)
