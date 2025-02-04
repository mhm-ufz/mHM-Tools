"""Cut domains out of an existing mHM setup."""

from mhm_tools.pre.crop_mhm_setup import crop_mhm_setup


def add_args(parser):
    """Add cli arguments for the cut_mhm_setupt subcommand.

    Parameters
    ----------
    parser : argparse.ArgumentParser
        the main argument parser
    """
    required_args = parser.add_argument_group("required arguments")
    required_args.add_argument(
        "-m",
        "--mask_file",
        required=True,
        help="The path the the mask file. Mask files can be created using the catchment command with the --mask flag.",
    )
    required_args.add_argument(
        "-i",
        "--input_path",
        required=True,
        help="Path to the existing mHM setup.",
    )
    required_args.add_argument(
        "-o",
        "--output_path",
        required=True,
        help="Path where the new domain setup should be saved.",
    )
    parser.add_argument(
        "--l1_resolution",
        required=False,
        help = ("Hydrological resolution. Without it no latlon file can be produced.")
    )
    parser.add_argument(
        "--l11_resolution",
        required=False,
    )
    parser.add_argument(
        "--crs",
        default=None,
        help=(
            "Coordinates reference system (e.g. 'epsg:3035'). "
            "If not given, headers will be interpreted as given in lat-lon ('epsg:4326')."
        ),
    )


def run(args):
    """Cut out a domain setup out of an existing mHM setup..

    Parameters
    ----------
    args : argparse.Namespace
        parsed command line arguments
    """
    crop_mhm_setup(args.mask_file, args.output_path, args.input_path, l1_resolution=args.l1_resolution, crs=args.crs, l11_resolution=args.l11_resolution)
