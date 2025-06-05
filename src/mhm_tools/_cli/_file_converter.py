"""Convert between ascii and netcdf by file suffix."""

from pathlib import Path

from mhm_tools.common.file_handler import get_xarray_ds_from_file, write_xarray_to_file


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
        "-o", "--output", required=True, help="The name of the output file."
    )
    parser.add_argument(
        "-f",
        "--varname_eq_in_filename",
        required=False,
        action="store_true",
        help="The name of the variable is set to the input file name.",
    )
    parser.add_argument(
        "-F",
        "--varname_eq_out_filename",
        required=False,
        action="store_true",
        help="The name of the variable is set to the output file name.",
    )
    parser.add_argument(
        "-v",
        "--varname",
        required=False,
        default=None,
        help="The name of the variable.",
    )


def run(args):
    input = Path(args.input)
    output = Path(args.output)
    var_name = args.varname
    if args.varname_eq_in_filename:
        var_name = input.stem
    elif args.varname_eq_out_filename:
        var_name = output.stem
    ds = get_xarray_ds_from_file(input, var_name=var_name)
    write_xarray_to_file(ds, output, var_name=var_name)
