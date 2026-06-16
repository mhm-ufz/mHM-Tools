"""
Computes and stores long term mean for a given NetCDF file(s).

Authors
-------
- Jeisson Leal
"""

import argparse


def add_args(parser):
    """Add CLI arguments for the long_term_mean subcommand."""
    parser.description = (
        "Compute long-term means (hourly, daily, monthly, yearly) "
        "for precipitation and temperature forcings from NetCDF files."
    )
    parser.epilog = "See mhm-tools documentation for detailed examples and usage."

    required = parser.add_argument_group("required arguments")
    flags = parser.add_argument_group("flags")
    required.add_argument(
        "-i",
        "--input-dir",
        "--in-dir",
        dest="in_dir",
        required=True,
        help="Input directory containing forcing NetCDF files",
    )
    required.add_argument(
        "-o",
        "--output-dir",
        "--out-dir",
        dest="out_dir",
        required=True,
        help="Output directory for processed files",
    )

    optional = parser.add_argument_group("optional arguments")
    optional.add_argument(
        "-f",
        "--input-name",
        "--in-file",
        dest="in_file",
        required=False,
        default="*.nc",
        help="Input filename or glob pattern.",
    )
    optional.add_argument(
        "--long-term-mean-type",
        choices=["hourly", "daily", "monthly", "yearly"],
        default="monthly",
        help=(
            "Period over which to compute long-term means: hourly, daily, monthly, or yearly. "
            "Default is 'monthly'."
        ),
    )
    optional.add_argument(
        "--aggregation-type",
        choices=["intensive", "extensive"],
        default="intensive",
        help=(
            "Aggregation type: 'intensive' (mean) or 'extensive' (sum). Default is 'intensive'."
        ),
    )
    flags.add_argument(
        "--aggregation",
        dest="aggregate",
        action="store_true",
        help="Perform temporal aggregation before merging.",
    )
    flags.add_argument(
        "--keep-temporal-files",
        action="store_true",
        help="Keep intermediate temporal files generated during processing.",
    )
    optional.add_argument(
        "--output-name",
        "--out-file",
        dest="out_file",
        default="long_term_mean.nc",
        help="Name of the output NetCDF file.",
    )
    flags.add_argument(
        "--crop",
        action="store_true",
        help="Crop the data to the specified geographic bounds.",
    )
    optional.add_argument(
        "--lon-min",
        type=float,
        help="Minimum longitude for cropping.",
    )
    optional.add_argument(
        "--lon-max",
        type=float,
        help="Maximum longitude for cropping.",
    )
    optional.add_argument(
        "--lat-min",
        type=float,
        help="Minimum latitude for cropping.",
    )
    optional.add_argument(
        "--lat-max",
        type=float,
        help="Maximum latitude for cropping.",
    )
    optional.add_argument(
        "--lower-threshold",
        type=float,
        default=None,
        help="If given, calculates the long-term mean for values equal or above lower-threshold.",
    )
    parser.set_defaults(aggregate=False)


def run(args: argparse.Namespace):
    """Run script to compute the long-term mean of the input data.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed command line arguments.
    """
    from mhm_tools.post.long_term_mean import cal_long_term_mean

    if not args.aggregate:
        args.aggregation_type = None
        args.long_term_mean_type = None

    cal_long_term_mean(
        in_dir=args.in_dir,
        in_file=args.in_file,
        out_dir=args.out_dir,
        long_term_mean_type=args.long_term_mean_type,
        aggregation_type=args.aggregation_type,
        keep_temporal_files=args.keep_temporal_files,
        out_file=args.out_file,
        crop=args.crop,
        lon_min=args.lon_min,
        lon_max=args.lon_max,
        lat_min=args.lat_min,
        lat_max=args.lat_max,
        aggregate=args.aggregate,
        lower_threshold=args.lower_threshold,
    )
