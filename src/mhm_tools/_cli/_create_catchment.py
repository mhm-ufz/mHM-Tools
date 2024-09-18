"""
TODO: add description
"""

import numpy as np

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
    parser.add_argument(
        "--vn",
        "--varname",
        default="flwdir",
        help=("Name of variable in output file"),
    )
    parser.add_argument(
        "-v",
        "--var",
        default="fdir",
        help=("Input variable, use 'fdir' or 'dem'"),
    )
    parser.add_argument(
        "--ftp",
        "--ftype",
        default="ldd",
        help=("ftype of input variable, use 'nextxy', 'ldd' or 'd8'"),
    )
    parser.add_argument(
        "--gauge_coords",
        default=None,
        help=(
            "Gauge coordinates in the form of 'lat,lon' take care to write --gauge_coords='lat,lon'"
        ),
    )
    parser.add_argument(
        "--latlon_box",
        default=None,
        help=(
            "Latlon box in the form of 'latmin,latmax,lonmin,lonmax' take care to write --latlon_box='latmin,latmax,lonmin,lonmax' and that the catchment you want to create is completely within this box." 
        ),
    )
    parser.add_argument(
        "--mask_file",
        default=None,
        help=("Path where to save the mask file"),
    )
    parser.add_argument("--log_level", default="INFO", type=str, help=("Logging level"))


def run(args):
    """Create the catchment file.

    Parameters
    ----------
    args : argparse.Namespace
        parsed command line arguments
    """
    gauge_coords = None
    latlon_box = None
    if args.gauge_coords is not None:
        if args.latlon_box is not None:
            raise ValueError("You can't use --gauge_coords and --latlon_box at the same time.")
        lat, lon = map(float, args.gauge_coords.split(","))
        gauge_coords = (np.array([lon]), np.array([lat]))
    elif args.latlon_box is not None:
        latmin, latmax, lonmin, lonmax = map(float, args.latlon_box.split(","))
        latlon_box = {'lat_slice': (latmin, latmax), 'lon_slice': (lonmin, lonmax)}
    create_catchment(
        input_file=args.input_file,
        output_path=args.output_path,
        var_name=args.vn,
        var=args.var,
        ftype=args.ftp,
        gauge_coords=gauge_coords,
        latlon_box=latlon_box,
        log_level=args.log_level,
        mask_file=args.mask_file,
    )
