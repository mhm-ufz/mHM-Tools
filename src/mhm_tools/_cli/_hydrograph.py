"""Create a hydrograph showing the simulated and observed discharge."""

from ..post.hydrograph import Hydrograph


def add_args(parser):
    """Add cli arguments for the hydrograph subcommand.

    Parameters
    ----------
    parser : argparse.ArgumentParser
        the main argument parser

    """
    parser.add_argument(
        "-i",
        "--input",
        dest="in_dir",
        required=True,
        help="The path to input (mhm output directory)",
    )
    parser.add_argument(
        "-o",
        "--output",
        dest="out_file",
        required=False,
        default="hydrograph.pdf",
        help="The name of the output file. By default `hydrograph.png` If it contains no '/' the file is written to "
        "the input path.",
    )
    parser.add_argument(
        "-t",
        "--title",
        dest="title",
        required=False,
        default="",
        help="The title for the hydrograph",
    )
    parser.add_argument(
        "-s",
        "--show",
        dest="show",
        required=False,
        action="store_true",
        help="Show the plots that are created the default is only saving them",
    )
    # parser.add_argument(
    #     "-p",
    #     "--plots",
    #     default=15,
    #     required=False,
    #     dest="plots_to_be_created",
    #     help="specifies which graphics are generated. Sum up the numbers of the plots you want: "
    #     "1 model timestep, 2 yearly, 4 seasonality, 8 scatter e.g. "
    #     "all with out seasonality (advised for performance) = 11 "
    #     "default is all",
    # )
    parser.add_argument(
        "-p",
        "--plots",
        default="tysc",
        required=False,
        dest="plots_to_be_created",
        help="specifies which graphics are generated."
        "t model timestep, y yearly, s seasonality, c scatter e.g. "
        "all with out seasonality (advised for performance) = tyc",
    )
    parser.add_argument(
        "-l",
        "--log_level",
        default="warning",
        required=False,
        dest="log_level",
        help="log level (debug, info, warning) default is warning",
    )
    parser.add_argument(
        "--prec",
        dest="prec",
        required=False,
        default="",
        help="path of the precipiation file",
    )


def run(args):
    """
    Create the hydrograph plots.

    Parameters
    ----------
    args : argparse.Namespace
        parsed command line arguments

    """
    hydro = Hydrograph(args.log_level)
    hydro.gen_hydrograph(
        input_path=args.in_dir,
        output_file=args.out_file,
        show=args.show,
        save=True,
        title=args.title,
        plot_code=args.plots_to_be_created,
        prec_path=args.prec,
    )
