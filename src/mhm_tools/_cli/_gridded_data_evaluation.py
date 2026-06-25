"""Evaluate gridded model data against gridded reference data.

The tool compares spatial fields and time series within an optional mask or
lon/lat domain, supports unit conversion, temporal resampling, direct or
bootstrap comparisons, and restricted year ranges. It writes evaluation
metrics and plots for bias, seasonality, variability, and related diagnostics.

Authors
-------
- Simon Lüdke
"""

import logging

from mhm_tools.common.logger import ErrorLogger

logger = logging.getLogger(__name__)


def normalize_target_frequency(freq):
    """Validate and normalize target frequency aliases.

    Accepts aliases (H, D, W, ME) or words (hourly, daily, weekly, monthly),
    ignoring case. Returns the canonical pandas alias or None.
    """
    if freq is None:
        return None
    if not isinstance(freq, str):
        error_msg = "target frequency must be a string or None"
        with ErrorLogger(logger):
            raise ValueError(error_msg)
    normalized = freq.strip().lower()
    if not normalized:
        return None

    word_map = {
        "hourly": "H",
        "daily": "D",
        "weekly": "W",
        "monthly": "ME",
    }
    alias_map = {"h": "H", "d": "D", "w": "W", "me": "ME", "M": "ME", "m": "ME"}
    if normalized in word_map:
        return word_map[normalized]
    if normalized in alias_map:
        return alias_map[normalized]

    valid = ", ".join(["H", "D", "W", "ME", "hourly", "daily", "weekly", "monthly"])
    error_msg = f"Invalid target frequency '{freq}'. Valid options: {valid}."
    with ErrorLogger(logger):
        raise ValueError(error_msg)


def add_args(parser):
    """Add cli arguments for the gridded evaluation.

    Parameters
    ----------
    parser : argparse.ArgumentParser
        the main argument parser
    """
    required = parser.add_argument_group("required arguments")
    optional = parser.add_argument_group("optional arguments")

    flags = parser.add_argument_group("flags")
    required.add_argument(
        "--input-path",
        help="Path to the input file. Or the dictionary containing all folders with input files.",
        required=True,
    )
    required.add_argument(
        "--output-dir", help="Path for the output dir.", required=True
    )

    optional.add_argument(
        "--input-variable", help="Variable name in the input file.", required=False
    )
    optional.add_argument(
        "--input-name",
        help="Name of the input dataset.",
        default="input",
        required=False,
    )
    optional.add_argument(
        "--input-factor",
        help="Unit Conversion factor. e.g. MJ/kg/day:  1 / 2.47 = 0.4",
        default=1,
        required=False,
    )
    optional.add_argument(
        "--ref-path",
        help="Path to the first reference file. Or the dictionary containing all folders with ref files.",
        default=None,
        required=False,
    )
    optional.add_argument(
        "--ref-name",
        help="Name of the reference dataset.",
        default="ref",
        required=False,
    )
    optional.add_argument(
        "--ref-factor",
        help="Unit Conversion factor. e.g. MJ/kg/day:  1 / 2.47 = 0.4",
        default=1,
        required=False,
    )
    optional.add_argument(
        "--ref-variable",
        help="Variable name in the first reference file.",
        default=None,
        required=False,
    )
    flags.add_argument(
        "--only-plot",
        help="Set Flag if existing output file should be used to create plot",
        action="store_true",
        required=False,
    )
    optional.add_argument(
        "--lonlatbox",
        required=False,
        default=None,
        help=("""coordinates in the form of 'lon_min,lon_max,lat_min,lat_max'"""),
    )
    optional.add_argument(
        "--mask-file",
        required=False,
        default=None,
        help=(
            """path to the mask file, a .nc file with a variable 'mask' containing the grid mask at l0 resolution
            required unless --lonlatbox is provided"""
        ),
    )
    optional.add_argument(
        "--mask-var",
        required=False,
        default=None,
        help="Variable in --mask-file to use for masking. If omitted, the mask variable is selected by resolution.",
    )
    optional.add_argument(
        "--ncpus",
        default=1,
        type=int,
        help=("Number of CPUs to use"),
    )
    optional.add_argument(
        "--n-boostrap-years",
        required=False,
        default=None,
        type=int,
        help=("""Number of years to draw for each boostrap experiment"""),
    )
    optional.add_argument(
        "--n-bootstrap-selections",
        required=False,
        default=None,
        type=int,
        help=("Number of boostrap experiments"),
    )
    flags.add_argument(
        "--no-direct-comparison",
        action="store_true",
        required=False,
        help=("Use statistics and do not compare timeseries directly. Needs ref_path."),
    )
    optional.add_argument(
        "--start-year",
        required=False,
        default=None,
        type=int,
        help=("""First year allowed in the analysis."""),
    )
    optional.add_argument(
        "--end-year",
        required=False,
        default=None,
        type=int,
        help=("Lates year that is allowed in the analysis."),
    )
    optional.add_argument(
        "--available-mem",
        required=False,
        default=None,
        help=("""Available memory per cpu in Gb or Mb (default Gb)"""),
    )
    optional.add_argument(
        "--input-file-name",
        required=False,
        default="*.nc",
        help="Input file name. E.g. '*.nc' to copy only nc files or 'pre*' to copy only precipitation files. If the file has a header in it's folder the header is reproduced regardless of wether nor not it fits the filename.",
    )
    optional.add_argument(
        "--ref-file-name",
        required=False,
        default="*.nc",
        help="Ref file name. E.g. '*.nc' to copy only nc files or 'pre*' to copy only precipitation files. If the file has a header in it's folder the header is reproduced regardless of wether nor not it fits the filename.",
    )
    optional.add_argument(
        "--lon-min",
        required=False,
        default=None,
        type=float,
        help=("""minimum longitude of the target grid
            required unless --mask_file is provided"""),
    )

    optional.add_argument(
        "--lon-max",
        required=False,
        default=None,
        type=float,
        help=("""maximum longitude of the target grid
            required unless --mask_file is provided"""),
    )

    optional.add_argument(
        "--lat-min",
        required=False,
        default=None,
        type=float,
        help=("""minimum latitude of the target grid
            required unless --mask_file is provided"""),
    )

    optional.add_argument(
        "--lat-max",
        required=False,
        default=None,
        type=float,
        help=("""maximum latitude of the target grid
            required unless --mask_file is provided"""),
    )
    flags.add_argument(
        "--bias-only",
        action="store_true",
        required=False,
        help=("Only compare bias spatially and for the seasonality."),
    )
    flags.add_argument(
        "--global-climate",
        action="store_true",
        required=False,
        help=("Only compare bias and temporal standard deviation (no Spearman)."),
    )

    optional.add_argument(
        "--resample-time-to",
        required=False,
        default=None,
        type=normalize_target_frequency,
        help=(
            "Frequency to resample input and ref dataset to. Options: (H, D, W, ME) or hourly, daily, weekly, monthly"
        ),
    )
    optional.add_argument(
        "--metric",
        required=False,
        default="SPAEF",
        help="Result metric written to results.csv. Accepted values: TSM, SPAEF, ESP, WASPAEF, MSPAEF, all.",
    )


def run(args):
    """Calculate the evaluation metrics and create plots.

    Parameters
    ----------
    args : argparse.Namespace
        parsed command line arguments
    """
    from mhm_tools.common.cli_utils import get_available_mem_in_unit, get_coords
    from mhm_tools.common.file_handler import get_xarray_ds_from_file
    from mhm_tools.post.gridded_data_evaluation import (
        EvalDataset,
        gridded_data_evaluation,
    )

    (
        lon_min,
        lon_max,
        lat_min,
        lat_max,
        mask_da,
    ) = get_coords(
        args.lonlatbox,
        args.mask_file,
        args.lon_min,
        args.lon_max,
        args.lat_min,
        args.lat_max,
        raise_exception=False,
        mask_var=args.mask_var,
    )
    if args.mask_file is not None:
        with get_xarray_ds_from_file(
            args.mask_file, normalize_latlon_coords=True
        ) as mask_ds:
            mask_da = mask_ds.load()
        logger.debug(
            f"Passing mask to gridded_data_evaluation as {type(mask_da).__name__}; "
            f"data_vars={list(mask_da.data_vars) if hasattr(mask_da, 'data_vars') else None}; "
            f"mask_var={args.mask_var}"
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
    year_slice = slice(args.start_year, args.end_year)
    available_mem = get_available_mem_in_unit(args.available_mem)
    input = EvalDataset(
        path=args.input_path,
        name=args.input_name,
        var=args.input_variable,
        factor=float(args.input_factor),
        file_name=args.input_file_name,
    )
    ref = EvalDataset(
        path=args.ref_path,
        name=args.ref_name,
        var=args.ref_variable,
        factor=float(args.ref_factor),
        file_name=args.ref_file_name,
    )
    target_freq = normalize_target_frequency(args.resample_time_to)
    if args.bias_only and args.global_climate:
        error_msg = "Options --bias_only and --global_climate are mutually exclusive."
        with ErrorLogger(logger):
            raise ValueError(error_msg)
    gridded_data_evaluation(
        input=input,
        ref=ref,
        output_path=args.output_dir,
        only_plot=args.only_plot,
        coordinate_slice=coordinate_slice,
        mask_da=mask_da,
        n_cpus=args.ncpus,
        n_bootstrap_years=args.n_boostrap_years,
        n_bootstrap_selections=args.n_bootstrap_selections,
        direct_comp=(
            args.n_bootstrap_selections is None and args.n_boostrap_years is None
        )
        and not (args.global_climate or args.no_direct_comparison),
        year_slice=year_slice,
        avaiable_mem=available_mem,
        bias_only=args.bias_only,
        global_climate=args.global_climate,
        target_time_freq=target_freq,
        mask_var=args.mask_var,
        result_metric=args.metric,
    )
