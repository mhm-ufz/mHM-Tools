"""CLI for converting mHM landcover ASCII files to CF-compliant NetCDF."""

from __future__ import annotations

from argparse import ArgumentParser, Namespace
from pathlib import Path


def add_args(parser: ArgumentParser) -> None:
    """
    Add CLI arguments for the `landcover_ascii_to_nc` subcommand.

    Parameters
    ----------
    parser : argparse.ArgumentParser
        The subparser for this command.
    """
    parser.add_argument(
        "-i",
        "--input_nml",
        required=True,
        help=(
            "Path to the mHM .nml file that lists landcover scenes "
            "(the namelist containing LCoverfName, LCoverYearStart, etc.)."
        ),
    )
    parser.add_argument(
        "-o",
        "--output",
        required=True,
        help="Output NetCDF file path.",
    )
    parser.add_argument(
        "-f",
        "--varname_eq_in_filename",
        action="store_true",
        help=(
            "Set the output variable name to the stem of the FIRST input "
            "ASCII file (i.e. derived from the namelist entry)."
        ),
    )
    parser.add_argument(
        "-F",
        "--varname_eq_out_filename",
        action="store_true",
        help=(
            "Set the output variable name to the stem of the output file "
            "path provided via --output."
        ),
    )
    parser.add_argument(
        "-v",
        "--varname",
        default=None,
        help=(
            "Name of the variable to write in the NetCDF. "
            "Ignored if -f/--varname_eq_in_filename or "
            "-F/--varname_eq_out_filename is given."
        ),
    )
    parser.add_argument(
        "--normalize_latlon",
        action="store_true",
        help="Normalize coordinate names/ordering to lat/lon.",
    )


def _resolve_var_name(
    args: Namespace,
    input_nml: Path,
    output: Path,
) -> str:
    """
    Decide which variable name to use in the output dataset.

    Priority:
    1. --varname_eq_in_filename
    2. --varname_eq_out_filename
    3. --varname (explicit name)
    4. 'land_cover' (fallback)
    """
    if args.varname_eq_in_filename:
        return input_nml.stem
    if args.varname_eq_out_filename:
        return output.stem
    if args.varname:
        return args.varname
    return "land_cover"


def run(args: Namespace) -> None:
    """
    Execute the conversion: read ASCII landcover grids, build a CF-compliant.

    time axis with bounds, and write NetCDF.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed command line arguments (see add_args()).
    """
    from mhm_tools.pre.landcover_ascii_to_nc import convert_lc_ascii_to_nc

    input_nml = Path(args.input_nml)
    output = Path(args.output)
    var_name = _resolve_var_name(args, input_nml, output)

    convert_lc_ascii_to_nc(
        input_nml=input_nml,
        output=output,
        var_name=var_name,
        normalize_latlon=args.normalize_latlon,
    )
