"""Create basin id file and deliniate catchments."""

import logging

import numpy as np

from mhm_tools.common.cli_utils import get_available_mem_in_unit
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
    optional_args = parser.add_argument_group("optional arguments")
    optional_args.add_argument(
        "-S",
        "--split_file",
        action="store_false",
        default=True,
        help=(
            "Write out multiple files. By default a single file is written "
            "alternative is that the file is split up by variables."
        ),
    )
    optional_args.add_argument(
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
    optional_args.add_argument(
        "--vn",
        "--varname",
        default="flwdir",
        help=("Name of variable in output file"),
    )
    optional_args.add_argument(
        "-v",
        "--var",
        default="fdir",
        help=("Input variable, use 'fdir' or 'dem'"),
    )
    optional_args.add_argument(
        "--ftp",
        "--ftype",
        default="ldd",
        help=("ftype of input variable, use 'nextxy', 'ldd' or 'd8'"),
    )
    optional_args.add_argument(
        "--gauge_coords",
        default=None,
        help=(
            "Gauge coordinates in the form of 'lat,lon' take care to write --gauge_coords='lat,lon'"
        ),
    )
    optional_args.add_argument(
        "--lonlatbox",
        required=False,
        default=None,
        help=(
            """coordinates in the form of 'lon_min,lon_max,lat_min,lat_max,resolution_l0'"""
        ),
    )
    optional_args.add_argument(
        "--l1_resolution",
        required=False,
        type=float,
        default=None,
        help=("""Resolution of the mHM target grid."""),
    )
    optional_args.add_argument(
        "--l11_resolution",
        required=False,
        type=float,
        default=None,
        help=(
            """Resolution of the mRM routing resolution. Only used to extend the grid to cleanly fit this data."""
        ),
    )
    optional_args.add_argument(
        "--l2_resolution",
        required=False,
        type=float,
        default=None,
        help=(
            """Resolution of the mHM meteo input resolution. Only used to extend the grid to cleanly fit this data."""
        ),
    )
    optional_args.add_argument(
        "--upscale",
        action="store_true",
        default=False,
        help=("""Upscale to l1_resolution."""),
    )
    optional_args.add_argument(
        "--coords_are_not_latlon",
        action="store_false",
        default=True,
        help=("""Set this flag if the coordinates are in m not degree."""),
    )
    optional_args.add_argument(
        "--mask_file",
        default=None,
        help=("Path where to save the mask file"),
    )
    optional_args.add_argument(
        "--frame",
        default=0,
        type=int,
        help=(
            "Creates a frame of nonflow cells around the domain to enable non global domains in ulysses mrm which connects the eastern and western boundaries."
        ),
    )
    optional_args.add_argument(
        "--ref_catchment_area",
        default=None,
        type=float,
        help=(
            "Reference catchment area in km^2 used to identify the outlet cell near the gauge coordinates."
        ),
    )
    optional_args.add_argument(
        "--max_distance_cells",
        default=5,
        type=int,
        help=("""Maximum distance in cells to search for the outlet cell."""),
    )
    optional_args.add_argument(
        "--max_error",
        default=0.05,
        type=float,
        help=("""Maximum error allowed when searching for the outlet cell."""),
    )
    optional_args.add_argument(
        "--available_mem",
        required=False,
        type=str,
        default="5Gb",
        help=("""Available memory per cpu in Gb or Mb (default Gb)"""),
    )
    optional_args.add_argument(
        "--gauge_id",
        required=False,
        default=None,
        type=int,
        help="If Gauge id is provided in addition to the other output a id_gauges file is created in the output_folder.",
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
        gauge_coords = (np.array([lat]), np.array([lon]))
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
    available_mem = get_available_mem_in_unit(args.available_mem)
    create_catchment(
        input_file=args.input_file,
        output_path=args.output_path,
        var_name=args.vn,
        var=args.var,
        ftype=args.ftp,
        gauge_coords=gauge_coords,
        coordinate_slices=coordinate_slices,
        mask_file=args.mask_file,
        l1_resolution=args.l1_resolution,
        l11_resolution=args.l11_resolution,
        l2_resolution=args.l2_resolution,
        frame=args.frame,
        upscale=args.upscale,
        latlon=args.coords_are_not_latlon,
        available_mem=available_mem,
        ref_catchment_area=args.ref_catchment_area,
        max_distance_cells=args.max_distance_cells,
        max_error=args.max_error,
        id_gauges=args.gauge_id,
    )
