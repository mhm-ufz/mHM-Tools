"""
mHM processing netCDF precipitation and temperature forcings.

Authors
-------
- Jeisson Leal
- Simon Lüdke
"""


def add_args(parser):
    """Add CLI arguments for prepare_mhm_forcings subcommand.

    Utility functions for mHM processing of netCDF precipitation and temperature forcings,
    including unit conversion, coordinate ordering, cropping, and setting correct metadata.

    Example:
      mhm-tools prepare_mhm_forcings \
        -i in/data -f input_*.nc -o out/data -v 2t --out-file processed.nc \
        --crop --lon-min 0 --lon-max 10 --lat-min -5 --lat-max 5
    """
    # Description and epilog
    parser.description = (
        "Utility functions for mHM processing netCDF precipitation and temperature forcings, "
        "including unit conversion, coordinate ordering, setting correct variable name and "
        "units as well as missing and fill values, and spatial cropping."
    )
    parser.epilog = "See mhm-tools documentation for detailed examples and usage."

    # Required arguments
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
        "-f",
        "--input-name",
        "--in-file",
        dest="in_file",
        required=True,
        help='Input filename or glob pattern (e.g. "data_*.nc")',
    )
    required.add_argument(
        "-o",
        "--output-dir",
        "--out-dir",
        dest="out_dir",
        required=True,
        help="Output directory for processed files",
    )
    # Optional arguments
    optional = parser.add_argument_group("optional arguments")
    optional.add_argument(
        "-v",
        "--var",
        required=False,
        help="Variable name to convert: 2t (temperature), tp (total precipitation), tprate (precipitation rate)",
    )

    optional.add_argument(
        "-u",
        "--output-name",
        "--out-file",
        dest="out_file",
        default="*",
        help=(
            "Output filename or pattern. Use '*' to retain input basename; "
            "otherwise literal name for each output file."
        ),
    )
    optional.add_argument(
        "--out-var",
        default=None,
        help="Rename output variable to this name.",
    )
    flags.add_argument(
        "--crop", action="store_true", help="Enable spatial cropping of the dataset"
    )
    optional.add_argument(
        "--lon-min",
        type=float,
        help="Minimum longitude for cropping",
        required=False,
        default=None,
    )
    optional.add_argument(
        "--lon-max",
        type=float,
        help="Maximum longitude for cropping",
        required=False,
        default=None,
    )
    optional.add_argument(
        "--lat-min",
        type=float,
        help="Minimum latitude for cropping",
        required=False,
        default=None,
    )
    optional.add_argument(
        "--lat-max",
        type=float,
        help="Maximum latitude for cropping",
        required=False,
        default=None,
    )
    optional.add_argument(
        "--use-mfdataset",
        type=bool,
        default=False,
        help="Use xarray.open_mfdataset for multi-file datasets",
    )
    optional.add_argument(
        "--target-frequency",
        type=str,
        default=None,
        help="Resample the dataset to this target frequency (hourly, daily).",
    )


def run(args):
    """Run script to convert input forcings into the right mHM format.

    Parameters
    ----------
    args : argparse.Namespace
        parsed command line arguments
    """
    from mhm_tools.pre.prepare_mhm_forcings import prepare_forcings

    prepare_forcings(
        in_dir=args.in_dir,
        in_file=args.in_file,
        out_dir=args.out_dir,
        out_file=args.out_file,
        var=args.var,
        crop=args.crop,
        lon_min=args.lon_min,
        lon_max=args.lon_max,
        lat_min=args.lat_min,
        lat_max=args.lat_max,
        use_mfdataset=args.use_mfdataset,
        target_frequency=args.target_frequency,
        out_var=args.out_var,
    )
