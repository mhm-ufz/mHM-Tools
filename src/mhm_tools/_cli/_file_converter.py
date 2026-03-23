"""Convert between ASCII, GeoTIFF, and NetCDF by file suffix."""

from pathlib import Path


def add_args(parser):
    """Add cli arguments for the file_converter subcommand.

    Parameters
    ----------
    parser : argparse.ArgumentParser
        the main argument parser

    """
    parser.add_argument(
        "-i",
        "--input",
        required=True,
        help="The path to input file. Can be ASCII, GeoTIFF, or NetCDF. The file type is determined by the file suffix.",
    )
    parser.add_argument(
        "-o", "--output", required=True, help="The name of the output file. Can be ASCII or NetCDF. The file type is determined by the file suffix."
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
    parser.add_argument(
        "--latlon",
        action="store_true",
        required=False,
        help=("use latlon variables"),
    )
    parser.add_argument(
        "--only_header",
        action="store_true",
        required=False,
        help=("Only write header output."),
    )


def run(args):
    """Convert between ASCII and NetCDF based on file suffix.

    Parameters
    ----------
    args : argparse.Namespace
        parsed command line arguments
    """
    from mhm_tools.common.file_handler import (
        create_header,
        get_xarray_ds_from_file,
        write_xarray_to_file,
    )

    input = Path(args.input)
    output = Path(args.output)
    var_name = args.varname
    if args.varname_eq_in_filename:
        var_name = input.stem
    elif args.varname_eq_out_filename:
        var_name = output.stem
    ds = get_xarray_ds_from_file(
        input, var_name=var_name, normalize_latlon_coords=args.latlon
    )
    if args.only_header:
        create_header(ds, output_path=output, no_data_value=None)
    else:
        write_xarray_to_file(ds, output, var_name=var_name)
