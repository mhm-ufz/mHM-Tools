"""Calculating pet from tavg."""

import logging

from mhm_tools.pre.pet_calc import calculate_pet

logger = logging.getLogger(__name__)


def add_args(parser):
    """Add CLI arguments for calculate_pet subcommand."""
    # Description and epilog
    parser.description = (
        # "Utility functions for mHM processing netCDF precipitation and temperature forcings, "
        # "including unit conversion, coordinate ordering, setting correct variable name and "
        # "units as well as missing and fill values, and spatial cropping."
    )
    parser.epilog = "See mhm-tools documentation for detailed examples and usage."

    # Required arguments
    required = parser.add_argument_group("required arguments")
    required.add_argument(
        "--tavg",
        required=True,
        help="Temperature average file",
    )
    required.add_argument(
        "-o",
        "--output_file",
        required=True,
        help="Output directory for processed files",
    )
    parser.add_argument(
        "-f",
        "--freq",
        required=False,
        default=None,
        help="Frequency of pet output, daily or hourly.",
    )
    parser.add_argument(
        "--ncpus",
        required=False,
        default=1,
        type=int,
        help=("Number of cores used for parallelisation."),
    )


def run(args):
    """Run script to convert input forcings into the right mHM format.

    Parameters
    ----------
    args : argparse.Namespace
        parsed command line arguments
    """
    calculate_pet(
        tavg_file=args.tavg,
        stat_freq=args.freq,
        out_file=args.output_file,
        max_workers=args.ncpus,
    )
