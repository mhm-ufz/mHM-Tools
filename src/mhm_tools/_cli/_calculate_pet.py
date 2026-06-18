"""Calculating pet.

Contains multiple different pet methods mainly temperature based.

Authors
-------
- Matthias Kelbling
- Simon Lüdke
- Stephan Thober
"""

import logging

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
        "-o",
        "--output-file",
        required=True,
        help="Output directory for processed files",
    )
    optional = parser.add_argument_group("optional arguments")
    optional.add_argument(
        "--tavg",
        required=False,
        default=None,
        help="Temperature average file",
    )
    optional.add_argument(
        "--tmax",
        required=False,
        default=None,
        help="Temperature maximum file",
    )
    optional.add_argument(
        "--tmin",
        required=False,
        default=None,
        help="Temperature minimum file",
    )
    optional.add_argument(
        "-f",
        "--freq",
        required=False,
        default=None,
        help="Frequency of pet output, daily or hourly.",
    )
    optional.add_argument(
        "--method",
        required=False,
        default="oudin",
        help="Method of pet calculation. Currently implemented: hargreaves_samani, oudin, jensen_haise, mcguinness_bordne, hamon, blaney_criddle.",
    )
    optional.add_argument(
        "--ncpus",
        required=False,
        default=1,
        type=int,
        help=("Number of cores used for parallelisation."),
    )


def run(args):
    """Calculate PET from temperature inputs and write output.

    Parameters
    ----------
    args : argparse.Namespace
        parsed command line arguments
    """
    from mhm_tools.pre.pet_calc import calculate_pet

    calculate_pet(
        tavg_file=args.tavg,
        tmax_file=args.tmax,
        tmin_file=args.tmin,
        stat_freq=args.freq,
        out_file=args.output_file,
        max_workers=args.ncpus,
        method=args.method,
    )
