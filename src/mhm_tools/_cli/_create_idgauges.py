"""Create an id gauges file."""

import logging

from mhm_tools.common.logger import ErrorLogger
from mhm_tools.pre import create_id_gauges

logger = logging.getLogger(__name__)


def add_args(parser):
    """Add cli arguments to create an id_gauges file.

    Parameters
    ----------
    parser : argparse.ArgumentParser
        the main argument parser

    """
    parser.add_argument(
        "-f",
        "--input_file",
        required=True,
        help="The path to input file in L0 resolution.",
    )
    parser.add_argument(
        "-i",
        "--gauge_id",
        required=True,
        type=int,
        help="Gauge id",
    )
    parser.add_argument(
        "-o",
        "--output",
        dest="out_file",
        required=False,
        default="gauges_id.asc",
        help="The name of the output file.",
    )
    parser.add_argument(
        "-u",
        "--facc_file",
        required=False,
        default=None,
        help="The name of the facc file used to adjust get coordinates.",
    )
    parser.add_argument(
        "--gauge_coords",
        default=None,
        help=(
            "Gauge coordinates in the form of 'lat,lon' take care to write --gauge_coords='lat,lon'"
        ),
    )
    parser.add_argument(
        "--lon",
        default=None,
        type=float,
        help=(
            "Gauge coordinates in the form of 'lat,lon' take care to write --gauge_coords='lat,lon'"
        ),
    )
    parser.add_argument(
        "--lat",
        default=None,
        type=float,
        help=(
            "Gauge coordinates in the form of 'lat,lon' take care to write --gauge_coords='lat,lon'"
        ),
    )
    parser.add_argument(
        "--is_id_gauges",
        dest="is_id_gauges",
        action="store_true",
        required=False,
        help="Is the provieded file an id_gauges file of which the existing gauge_ids should be conserved?",
    )


def run(args):
    """
    Create the id gauges file.

    Parameters
    ----------
    args : argparse.Namespace
        parsed command line arguments

    """
    if args.lat and args.lon:
        lat = args.lat
        lon = args.lon
    elif args.gauge_coords:
        lat, lon = map(float, args.gauge_coords.split(","))
    else:
        with ErrorLogger(logger):
            msg = (
                "Coordinates must be provided either as gauge_coords or as lat and lon."
            )
            raise ValueError(msg)

    create_id_gauges(
        file=args.input_file,
        out_path=args.out_file,
        lon=lon,
        lat=lat,
        file_is_idgauges=args.is_id_gauges,
        id=args.gauge_id,
        facc_file=args.facc_file,
    )
