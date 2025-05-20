"""Computes and stores long term mean for a given NetCDF file(s)"""

from typing import Optional
from mhm_tools.post.long_term_mean import cal_long_term_mean

def add_args(parser):
    """Add CLI arguments for cal_long_term_mean subcommand.

    Utility function to compute long-term means for a set of NetCDF files.

    Example:
      mhm-tools cal_long_term_mean \
        -i in/data -f input_*.nc -o out/data -v 2t \
        --long-term-mean-type yearly --aggregation-type extensive \
        --keep-temporal-files --out-file annual_mean_2t.nc
    """
    parser.description = (
        "Compute long-term means (hourly, daily, monthly, yearly) "
        "for precipitation and temperature forcings from NetCDF files."
    )
    parser.epilog = "See mhm-tools documentation for detailed examples and usage."

    required = parser.add_argument_group("required arguments")
    required.add_argument(
        "-i", "--in-dir",
        required=True,
        help="Input directory containing forcing NetCDF files",
    )
    required.add_argument(
        "-f", "--in-file",
        required=True,
        help='Input filename or glob pattern (e.g. "data_*.nc")',
    )
    required.add_argument(
        "-o", "--out-dir",
        required=True,
        help="Output directory for processed files",
    )
    required.add_argument(
        "-v", "--var-name",
        required=True,
        help="Variable name.",
    )

    optional = parser.add_argument_group("optional arguments")
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
    optional.add_argument(
        "--keep-temporal-files",
        action="store_true",
        help="Keep intermediate temporal files generated during processing (default: False).",
    )
    optional.add_argument(
        "--out-file",
        help="Name of the output NetCDF file (default: long_term_mean_<var>.nc)",
    )
    optional.add_argument(
        "--crop",
        action="store_true",
        help="Crop the data to the specified geographic bounds",
    )
    optional.add_argument(
        "--lon-min",
        type=float,
        help="Minimum longitude for cropping",
    )
    optional.add_argument(
        "--lon-max",
        type=float,
        help="Maximum longitude for cropping",
    )
    optional.add_argument(
        "--lat-min",
        type=float,
        help="Minimum latitude for cropping",
    )
    optional.add_argument(
        "--lat-max",
        type=float,
        help="Maximum latitude for cropping",
    )


def run(args):
    """
    Run script to compute the long-term mean of the input data.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed command line arguments.
    """
    cal_long_term_mean(
        in_dir=args.in_dir,
        in_file=args.in_file,
        out_dir=args.out_dir,
        var_name=args.var_name,
        long_term_mean_type=args.long_term_mean_type,
        aggregation_type=args.aggregation_type,
        keep_temporal_files=args.keep_temporal_files,
        out_file=args.out_file,
        crop=args.crop,
        lon_min=args.lon_min,
        lon_max=args.lon_max,
        lat_min=args.lat_min,
        lat_max=args.lat_max,
    )