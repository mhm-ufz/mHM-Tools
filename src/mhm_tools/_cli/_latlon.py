"""
Create the latlon.nc file required for mHM.

The latlon file contains the lat-lon information for 3 levels in mHM:
Level-0 (DEM), Level-1 (hydrology) and Level-11 (routing).
All levels will be checked for compatibility including Level-2 (meteo).
Level-0 can be given as a file or a dictionary containing an ascii grid
header. Other levels can then be given by only a cell-size and will be
determined from Level-0.
"""

import ast
from pathlib import Path

from ..pre import create_latlon


def add_args(parser):
    """Add cli arguments for the bankfull subcommand.

    Parameters
    ----------
    parser : argparse.ArgumentParser
        the main argument parser
    """
    required_args = parser.add_argument_group("required arguments")
    required_args.add_argument(
        "-D",
        "--level0",
        required=True,
        help=(
            "Level-0 (DEM) information. Either an ascii (header) file "
            "or a dictionary containing the header information."
        ),
    )
    required_args.add_argument(
        "-H",
        "--level1",
        required=True,
        help=(
            "Level-1 (hydrology) information. Either an ascii (header) file, "
            "a dictionary containing the header information "
            "or a cell-size to determine information from level-0."
        ),
    )
    # optional
    parser.add_argument(
        "-R",
        "--level11",
        default=None,
        help=(
            "Level-11 (routing) information. Either an ascii (header) file, "
            "a dictionary containing the header information "
            "or a cell-size to determine information from level-0."
        ),
    )
    parser.add_argument(
        "-M",
        "--level2",
        default=None,
        help=(
            "Level-2 (meteorology) information. Either an ascii (header) file, "
            "a dictionary containing the header information "
            "or a cell-size to determine information from level-0. "
            "Level-2 information wont be written to the latlon file."
        ),
    )
    parser.add_argument(
        "-c",
        "--crs",
        default=None,
        help=(
            "Coordinates reference system (e.g. 'epsg:3035'). "
            "If not given, headers will be interpreted as given in lat-lon ('epsg:4326')."
        ),
    )
    parser.add_argument(
        "-d",
        "--dtype",
        default="f4",
        help="Data type for the latlon file and headers.",
    )
    parser.add_argument(
        "-x",
        "--compression",
        type=int,
        choices=range(10),
        default=9,
        help="Compression level for the NetCDF file.",
    )
    parser.add_argument(
        "-b",
        "--add_bounds",
        action="store_true",
        default=False,
        help="Add bounds to the NetCDF axis.",
    )
    parser.add_argument(
        "--h0",
        "--write_header_l0",
        default=None,
        help="Write the level-0 header to a given file path.",
    )
    parser.add_argument(
        "--h1",
        "--write_header_l1",
        default=None,
        help="Write the level-1 header to a given file path.",
    )
    parser.add_argument(
        "--h11",
        "--write_header_l11",
        default=None,
        help="Write the level-11 header to a given file path.",
    )
    parser.add_argument(
        "--h2",
        "--write_header_l2",
        default=None,
        help="Write the level-2 header to a given file path.",
    )
    parser.add_argument(
        "-o",
        "--out_file",
        default="latlon.nc",
        help="The path of the output NetCDF file containing the latlon information.",
    )


def run(args):
    """Create the latlon file.

    Parameters
    ----------
    args : argparse.Namespace
        parsed command line arguments
    """
    create_latlon(
        out_file=args.out_file,
        level0=_eval(args.level0, "level0"),
        level1=_eval(args.level1, "level1"),
        level11=_eval(args.level11, "level11"),
        level2=_eval(args.level2, "level2"),
        write_header_l0=args.h0,
        write_header_l1=args.h1,
        write_header_l11=args.h11,
        write_header_l2=args.h2,
        crs=args.crs,
        dtype=args.dtype,
        compression=args.compression,
        add_bounds=args.add_bounds,
    )


def _eval(string, name):
    if not string:
        return None
    if Path(string).is_file():
        return string
    try:
        py_obj = ast.literal_eval(string)
    except Exception as err:
        msg = (
            f"latlon: '{name}' is not an existing file "
            f"and could not be interpreted otherwise: '{string}'"
        )
        raise ValueError(msg) from err
    else:
        return py_obj
