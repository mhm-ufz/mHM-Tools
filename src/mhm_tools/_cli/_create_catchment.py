"""Create basin id file and deliniate catchments."""

import logging

import numpy as np

from mhm_tools.common.logger import ErrorLogger

from ..pre import create_catchment

logger = logging.getLogger(__name__)


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
        "--lonlatbox",
        required=False,
        default=None,
        help=(
            """coordinates in the form of 'lon_min,lon_max,lat_min,lat_max,resolution_l0'"""
        ),
    )
    parser.add_argument(
        "--l1_resolution",
        required=False,
        type=float,
        default=None,
        help=("""Resolution of the mHM target grid."""),
    )
    parser.add_argument(
        "--upscale",
        action="store_true",
        default=False,
        help=("""Upscale to l1_resolution."""),
    )
    parser.add_argument(
        "--mask_file",
        default=None,
        help=("Path where to save the mask file"),
    )
    parser.add_argument(
        "--frame",
        default=0,
        type=int,
        help=(
            "Creates a frame of nonflow cells around the domain to enable non global domains in ulysses mrm which connects the eastern and western boundaries."
        ),
    )


def run(args):
    """Create the catchment file.

    Parameters
    ----------
    args : argparse.Namespace
        parsed command line arguments
    """
    gauge_coords = None
    coordinate_slices = None
    if args.gauge_coords is not None:
        if args.lonlatbox is not None:
            with ErrorLogger(logger):
                msg = "You can't use --gauge_coords and --lonlatbox at the same time."
                raise ValueError(msg)
        lat, lon = map(float, args.gauge_coords.split(","))
        gauge_coords = (np.array([lon]), np.array([lat]))
        logger.info(f"using gauge coordinates {gauge_coords}")
    elif args.lonlatbox is not None:
        lonmin, lonmax, latmin, latmax, resl0 = map(float, args.lonlatbox.split(","))
        coordinate_slices = {"lat": slice(latmax, latmin), "lon": slice(lonmin, lonmax)}
        logger.info(
            f"using lonlatbox with extends: lat=({latmax}, {latmin}); lon=({lonmin}, {lonmax})"
        )
    if args.upscale and not args.l1_resolution:
        msg = "If upscaling is enabled l1_resolution must be provided."
        raise ValueError(msg)
    create_catchment(
        input_file=args.input_file,
        output_path=args.output_path,
        var_name=args.vn,
        var=args.var,
        ftype=args.ftp,
        gauge_coords=gauge_coords,
        coordinate_slices=coordinate_slices,
        mask_file=args.mask_file,
        target_resolution=args.l1_resolution,
        frame=args.frame,
        upscale=args.upscale,
    )
