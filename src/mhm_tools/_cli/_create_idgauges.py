"""Create an id gauges file.

Authors
-------
- Simon Lüdke
"""

import logging

logger = logging.getLogger(__name__)


def add_args(parser):
    """Add cli arguments to create an id_gauges file.

    Parameters
    ----------
    parser : argparse.ArgumentParser
        the main argument parser
    """
    optional = parser.add_argument_group("optional arguments")
    flags = parser.add_argument_group("flags")
    optional.add_argument(
        "-f",
        "--input-file",
        required=True,
        help="The path to input file in L0 resolution.",
    )
    optional.add_argument(
        "-i",
        "--gauge-id",
        required=True,
        type=int,
        help="Gauge id",
    )
    optional.add_argument(
        "-o",
        "--output",
        dest="out_file",
        required=False,
        default="idgauges.asc",
        help="The name of the output file.",
    )
    optional.add_argument(
        "-u",
        "--facc-file",
        required=False,
        default=None,
        help="The name of the facc file used to adjust get coordinates.",
    )
    optional.add_argument(
        "--gauge-coords",
        default=None,
        help=(
            "Gauge coordinates in the form of 'lat,lon' take care to write --gauge_coords='lat,lon'"
        ),
    )
    optional.add_argument(
        "--lon",
        default=None,
        type=float,
        help=(
            "Gauge coordinates in the form of 'lat,lon' take care to write --gauge_coords='lat,lon'"
        ),
    )
    optional.add_argument(
        "--lat",
        default=None,
        type=float,
        help=(
            "Gauge coordinates in the form of 'lat,lon' take care to write --gauge_coords='lat,lon'"
        ),
    )
    flags.add_argument(
        "--is-id-gauges",
        dest="is_id_gauges",
        action="store_true",
        required=False,
        help="Is the provieded file an id_gauges file of which the existing gauge_ids should be conserved?",
    )


def run(args):
    """Create the id gauges file.

    Parameters
    ----------
    args : argparse.Namespace
        parsed command line arguments
    """
    from mhm_tools.common.logger import ErrorLogger
    from mhm_tools.pre import create_id_gauges

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
