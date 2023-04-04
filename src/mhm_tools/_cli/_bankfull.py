"""Calculate the river discharge at bankfull conditions and the bankfull width."""
from ..post.bankfull_discharge import gen_bankfull_discharge


def add_args(parser):
    """Add cli arguments for the bankfull subcommand.

    Parameters
    ----------
    parser : argparse.ArgumentParser
        the main argument parser
    """
    parser.add_argument(
        "-i",
        "--input",
        dest="in_file",
        required=True,
        help="The path of the mRM NetCDF file with the discharge data",
    )
    parser.add_argument(
        "-o",
        "--output",
        dest="out_file",
        required=True,
        help="The path of the output NetCDF file",
    )
    parser.add_argument(
        "-p",
        "--return_period",
        type=float,
        default=1.5,
        help="The return period of the flood, default: 1.5 years",
    )
    parser.add_argument(
        "-w",
        "--wetted_perimeter",
        action="store_true",
        default=False,
        dest="peri_bkfl",
        help="Additionally estimate the wetted perimeter.",
    )
    parser.add_argument(
        "-v",
        "--var",
        default="Qrouted",
        help="Variable name for routed streamflow in the NetCDF file",
    )


def bankfull(args):
    """Calculate the bankfull discharge

    Parameters
    ----------
    args : argparse.Namespace
        parsed command line arguments
    """
    gen_bankfull_discharge(
        ncin_path=args.in_file,
        ncout_path=args.out_file,
        return_period=args.return_period,
        peri_bkfl=args.peri_bkfl,
        var=args.var,
    )
