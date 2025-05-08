"""Validation of spatially distributed data based on their climatology or timeseries."""

from mhm_tools.common.cli_utils import get_available_mem_in_unit, get_coords
from mhm_tools.post.gridded_data_validation import gridded_data_validation


def add_args(parser):
    """Add cli arguments for the gridded validation.

    Parameters
    ----------
    parser : argparse.ArgumentParser
        the main argument parser
    """
    parser.add_argument(
        "--input_path",
        help="Path to the input file. Or the dictionary containing all folders with input files.",
        required=True,
    )
    parser.add_argument(
        "--input_variable", help="Variable name in the input file.", required=False
    )
    parser.add_argument(
        "--input_name", help="Name of the input dataset.", default=None, required=False
    )
    parser.add_argument(
        "--input_factor",
        help="Unit Conversion factor. e.g. MJ/kg/day:  1 / 2.47 = 0.4",
        default=1,
        required=False,
    )
    parser.add_argument("--output_dir", help="Path for the output dir.", required=True)
    parser.add_argument(
        "--ref_path",
        help="Path to the first reference file. Or the dictionary containing all folders with ref files.",
        default=None,
        required=False,
    )
    parser.add_argument(
        "--ref_name",
        help="Name of the reference dataset.",
        default=None,
        required=False,
    )
    parser.add_argument(
        "--ref_factor",
        help="Unit Conversion factor. e.g. MJ/kg/day:  1 / 2.47 = 0.4",
        default=1,
        required=False,
    )
    parser.add_argument(
        "--ref_variable",
        help="Variable name in the first reference file.",
        default=None,
        required=False,
    )
    parser.add_argument(
        "--only_plot",
        help="Set Flag if existing output file should be used to create plot",
        action="store_true",
        required=False,
    )
    parser.add_argument(
        "--lonlatbox",
        required=False,
        default=None,
        help=("""coordinates in the form of 'lon_min,lon_max,lat_min,lat_max'"""),
    )
    parser.add_argument(
        "--mask_file",
        required=False,
        default=None,
        help=(
            """path to the mask file, a .nc file with a variable 'mask' containing the grid mask at l0 resolution
            required unless --lonlatbox is provided"""
        ),
    )
    parser.add_argument(
        "--ncpus",
        default=1,
        type=int,
        help=("Number of CPUs to use"),
    )
    parser.add_argument(
        "--n_boostrap_years",
        required=False,
        default=None,
        type=int,
        help=("""Number of years to draw for each boostrap experiment"""),
    )
    parser.add_argument(
        "--n_bootstrap_selections",
        required=False,
        default=None,
        type=int,
        help=("Number of boostrap experiments"),
    )
    parser.add_argument(
        "--direct_comparison",
        action="store_true",
        dest="direct_comparison",
        required=False,
        help=("Use no statistics but compare timeseries directly. Needs ref_path."),
    )
    parser.add_argument(
        "--start_date",
        required=False,
        default=None,
        type=str,
        help=("""First year allowed in the analysis."""),
    )
    parser.add_argument(
        "--end_date",
        required=False,
        default=None,
        type=str,
        help=("Lates year that is allowed in the analysis."),
    )
    parser.add_argument(
        "--available_mem",
        required=False,
        default=None,
        help=("""Available memory per cpu in Gb or Mb (default Gb)"""),
    )


def run(args):
    """Calculate the validation.

    Parameters
    ----------
    args : argparse.Namespace
        parsed command line arguments
    """
    lon_min, lon_max, lat_min, lat_max, mask = get_coords(
        args.lonlatbox, args.mask_file, raise_exception=False
    )
    coordinate_slice = None
    if (
        lon_min is not None
        and lon_max is not None
        and lat_min is not None
        and lat_max is not None
    ):
        coordinate_slice = {
            "lat": slice(lat_max, lat_min),
            "lon": slice(lon_min, lon_max),
        }
    year_slice = slice(args.start_date, args.end_date)
    available_mem = get_available_mem_in_unit(args.available_mem)
    gridded_data_validation(
        args.input_path,
        args.input_variable,
        args.output_dir,
        args.ref_path,
        args.ref_variable,
        args.input_name,
        args.ref_name,
        float(args.input_factor),
        float(args.ref_factor),
        args.only_plot,
        coordinate_slice,
        args.ncpus,
        args.n_boostrap_years,
        args.n_bootstrap_selections,
        args.direct_comparison,
        year_slice=year_slice,
        avaiable_mem=available_mem,
    )
