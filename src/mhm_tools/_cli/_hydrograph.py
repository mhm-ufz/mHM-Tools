"""Create hydrograph plots from mHM/mRM discharge output.

The command reads one or more simulation output directories, optionally adds
observed discharge and precipitation, and writes discharge diagnostics such as
time series, yearly summaries, seasonality, flow-duration curves, and scatter
plots.

Authors
-------
- Simon Lüdke
"""


def add_args(parser):
    """Add cli arguments for the hydrograph subcommand.

    Parameters
    ----------
    parser : argparse.ArgumentParser
        the main argument parser
    """
    optional = parser.add_argument_group("optional arguments")
    flags = parser.add_argument_group("flags")
    optional.add_argument(
        "-i",
        "--input-dir",
        "--input",
        dest="in_dir",
        required=True,
        nargs="+",
        help="One or more input paths (mhm output directories)",
    )
    optional.add_argument(
        "-o",
        "--output-file",
        "--output",
        dest="out_file",
        required=False,
        default="hydrograph.pdf",
        help="The name of the output file. By default `hydrograph.png` If it contains no '/' the file is written to "
        "the input path.",
    )
    optional.add_argument(
        "-t",
        "--title",
        dest="title",
        required=False,
        default="",
        help="The title for the hydrograph",
    )
    flags.add_argument(
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
    optional.add_argument(
        "-p",
        "--plots",
        default="tyspc",
        required=False,
        dest="plots_to_be_created",
        help="specifies which graphics are generated."
        "t model timestep, y yearly, s seasonality, p flow duration, c scatter e.g. "
        "all with out seasonality (advised for performance) = typc",
    )
    optional.add_argument(
        "--prec",
        dest="prec",
        required=False,
        default="",
        help="path of the precipiation file",
    )
    optional.add_argument(
        "--name",
        dest="sim_names",
        required=False,
        nargs="+",
        help="Optional simulation name(s). If one name is provided, it replaces 'sim' in the legend. "
        "If multiple input paths are provided and the same number of names are given, each simulation "
        "is plotted with its name.",
    )


def run(args):
    """Create the hydrograph plots.

    Parameters
    ----------
    args : argparse.Namespace
        parsed command line arguments
    """
    from ..post.hydrograph import get_hydrograph_from_path

    get_hydrograph_from_path(
        input_path=args.in_dir,
        output_file=args.out_file,
        show=args.show,
        save=True,
        title=args.title,
        plot_code=args.plots_to_be_created,
        prec_path=args.prec,
        sim_names=args.sim_names,
    )
