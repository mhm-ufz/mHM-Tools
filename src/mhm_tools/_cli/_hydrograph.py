"""Create a hydrograph showing the simulated and observed discharge."""
from ..post.hydrograph import gen_hydrograph


def add_args(parser):
    """Add cli arguments for the bankfull subcommand.

    Parameters
    ----------
    parser : argparse.ArgumentParser
        the main argument parser
    """
    parser.add_argument(
        "-p",
        "--path",
        dest="path",
        required=True,
        help="The path to input mhm output directory"
    )
    parser.add_argument(
        "-o",
        "--output",
        dest="out_path",
        required=False,
        help="The path of the output NetCDF file"
    )
    parser.add_argument(
        "-f",
        "--filename",
        dest="filename",
        required=False,
        default="hydrograph.pdf",
        help="The name of the output file"
    )
    parser.add_argument(
        "-t",
        "--title",
        dest="title",
        required=False,
        default="",
        help="The titel for the hydrograph"
    )
    parser.add_argument(
        "-s",
        "--show",
        dest="show",
        required=False,
        action='store_true',
        help="Show the plots that are created the default is only saving them"
    )
    parser.add_argument(
        "-g",
        "--graphics",
        default=15,
        required=False,
        dest="plots_to_be_created",
        help="plot only the specified hydrograph 1 daily, 2 monthly, 4 yearly, 8 seasonality e.g. " \
               "\n - daily and seasonality -> 9" \
               "\n - daily, monthly, yearly -> 7" \
               "\n - all plots -> 15 (default)"
    )

def hydrograph(args):
    """Calculate the bankfull discharge

    Parameters
    ----------
    args : argparse.Namespace
        parsed command line arguments
    """
    gen_hydrograph(
        input_path=args.path,
        output_path=args.out_path,
        filename=args.filename,
        show=args.show,
        save=True,
        title=args.title,
        plots=args.plots_to_be_created
    )