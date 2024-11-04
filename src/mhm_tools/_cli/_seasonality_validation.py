

from mhm_tools.post.seasonality_grid_validation import seasonality_grid_validation


def add_args(parser):
    """Add cli arguments for the seasonality validation.

    Parameters
    ----------
    parser : argparse.ArgumentParser
        the main argument parser
    """
    parser.add_argument("--input_file", help="Path to the input file.", required=True)
    parser.add_argument("--input_var", help="Variable name in the input file.", required=False)
    parser.add_argument("--input_name", help="Name of the input dataset.", default=None, required=False)
    parser.add_argument("--input_factor", help="Unit Conversion factor. e.g. MJ/kg/day:  1 / 2.47 = 0.4", default=1, required=False)
    parser.add_argument("--output_path", help="Path for the output path.", required=True)
    parser.add_argument("--ref_file", help="Path to the first reference file.", default=None, required=False)
    parser.add_argument("--ref_name", help="Name of the reference dataset.", default=None, required=False)
    parser.add_argument("--ref_factor", help="Unit Conversion factor. e.g. MJ/kg/day:  1 / 2.47 = 0.4", default=1, required=False)
    parser.add_argument("--ref_var", help="Variable name in the first reference file.", default=None, required=False)
    parser.add_argument("--only_plot", help="Set Flag if existing output file should be used to create plot", action="store_true", required=False)

def run(args):
    """Calculate the validation.

    Parameters
    ----------
    args : argparse.Namespace
        parsed command line arguments
    """
    seasonality_grid_validation(args.input_file, args.input_var, args.output_path, args.ref_file, args.ref_var, args.input_name, args.ref_name, float(args.input_factor), float(args.ref_factor), args.only_plot)

    
