"""Evaluate simulated snow cover against a gridded snow reference.

The tool normalizes both datasets to a common calendar and grid, reduces them
to a binary snow-cover flag and writes the snow season metrics, the
classification accuracy map, the snow covered share over time and an animated
comparison, for the whole domain and per evaluation region.

Authors
-------
- Simon Lüdke
"""

import logging

from mhm_tools.common.logger import ErrorLogger

logger = logging.getLogger(__name__)


def normalize_snow_target_frequency(freq):
    """Validate and normalize target frequency aliases for the snow evaluation.

    Accepts aliases (D, 8D, W, ME) or words (daily, 8-daily, weekly, monthly),
    ignoring case.

    Parameters
    ----------
    freq : str or None
        Frequency given on the command line.

    Returns
    -------
    str or None
        The canonical pandas alias, or None when nothing was given.
    """
    if freq is None:
        return None
    normalized = str(freq).strip().lower()
    if not normalized:
        return None
    alias_map = {
        "d": "D",
        "daily": "D",
        "8d": "8D",
        "8-daily": "8D",
        "8daily": "8D",
        "w": "W",
        "weekly": "W",
        "me": "ME",
        "m": "ME",
        "monthly": "ME",
    }
    if normalized in alias_map:
        return alias_map[normalized]
    valid = (
        ", ".join(sorted(set(alias_map.values())))
        + " (or daily/8-daily/weekly/monthly)"
    )
    error_msg = f"Invalid target frequency '{freq}'. Valid options: {valid}."
    with ErrorLogger(logger):
        raise ValueError(error_msg)


def add_args(parser):
    """Add cli arguments for the snow evaluation.

    Parameters
    ----------
    parser : argparse.ArgumentParser
        the main argument parser
    """
    from mhm_tools.common.constants import WMO_REGION_BOUNDS

    required = parser.add_argument_group("required arguments")
    optional = parser.add_argument_group("optional arguments")
    flags = parser.add_argument_group("flags")

    required.add_argument(
        "--input-path",
        required=True,
        help="Path to the input file, or a directory searched recursively.",
    )
    required.add_argument(
        "--ref-path",
        required=True,
        help="Path to the reference file, or a directory searched recursively.",
    )
    required.add_argument(
        "--output-dir", required=True, help="Path for the output dir."
    )

    optional.add_argument(
        "--input-file-name",
        default="*.nc",
        help="Glob pattern for the recursive input search.",
    )
    optional.add_argument(
        "--ref-file-name",
        default="*.nc",
        help="Glob pattern for the recursive reference search.",
    )
    optional.add_argument(
        "--input-variable",
        default=None,
        help="Variable name in the input file. Detected when the file has one variable.",
    )
    optional.add_argument(
        "--ref-variable",
        default=None,
        help="Variable name in the reference file. Detected when the file has one variable.",
    )
    optional.add_argument(
        "--input-name",
        default="input",
        help="Name of the input dataset, used in plots and file names.",
    )
    optional.add_argument(
        "--ref-name",
        default="ref",
        help="Name of the reference dataset, used in plots and file names.",
    )
    optional.add_argument(
        "--snow-threshold",
        type=float,
        default=0.0,
        help="Values above this count as snow-covered in both datasets.",
    )
    optional.add_argument(
        "--input-snow-threshold",
        type=float,
        default=None,
        help="Snow threshold for the input only. Defaults to --snow-threshold.",
    )
    optional.add_argument(
        "--ref-snow-threshold",
        type=float,
        default=None,
        help="Snow threshold for the reference only. Defaults to --snow-threshold.",
    )
    optional.add_argument(
        "--target-frequency",
        default=None,
        help=(
            "Force a target frequency (D, 8D, W, ME or daily, 8-daily, weekly, "
            "monthly). By default the coarser of the two calendars is used, but "
            "never finer than 8 day composites."
        ),
    )
    optional.add_argument(
        "--year-mode",
        choices=["snow_year", "calendar_year"],
        default="snow_year",
        help="Yearly window used for the season metrics.",
    )
    optional.add_argument(
        "--snow-year-start-month",
        type=int,
        default=9,
        help="First month of a snow year.",
    )
    optional.add_argument(
        "--accuracy-vmin",
        type=float,
        default=None,
        help=(
            "Lower end of the classification accuracy colour scale. By default it "
            "follows the data; set it to compare several runs on one scale."
        ),
    )
    optional.add_argument(
        "--regions",
        default="all",
        help=(
            "Regions to create extra plots and gifs for: 'all', 'none' or a comma "
            f"separated list of: {', '.join(WMO_REGION_BOUNDS)}."
        ),
    )
    optional.add_argument(
        "--gif-fps", type=int, default=4, help="Playback speed of the gifs."
    )
    optional.add_argument(
        "--max-gif-frames",
        type=int,
        default=0,
        help="Maximum number of gif frames, 0 for all time steps.",
    )
    optional.add_argument(
        "--max-memory-gib",
        type=float,
        default=8.0,
        help="Size limit for loading the snow cover fields into memory.",
    )

    flags.add_argument(
        "--no-write-snow-cover",
        action="store_true",
        help="Skip writing the normalized binary snow cover fields.",
    )
    flags.add_argument(
        "--no-gif", action="store_true", help="Skip the animated comparison."
    )


def run(args):
    """Run the snow cover evaluation."""
    from mhm_tools.post.snow_evaluation import snow_evaluation

    snow_threshold = args.snow_threshold
    written_files = snow_evaluation(
        input_path=args.input_path,
        ref_path=args.ref_path,
        output_dir=args.output_dir,
        input_file_name=args.input_file_name,
        ref_file_name=args.ref_file_name,
        input_var=args.input_variable,
        ref_var=args.ref_variable,
        input_name=args.input_name,
        ref_name=args.ref_name,
        input_snow_threshold=(
            snow_threshold
            if args.input_snow_threshold is None
            else args.input_snow_threshold
        ),
        ref_snow_threshold=(
            snow_threshold
            if args.ref_snow_threshold is None
            else args.ref_snow_threshold
        ),
        target_freq=normalize_snow_target_frequency(args.target_frequency),
        year_mode=args.year_mode,
        snow_year_start_month=args.snow_year_start_month,
        accuracy_vmin=args.accuracy_vmin,
        regions=args.regions,
        write_snow_cover=not args.no_write_snow_cover,
        write_gif=not args.no_gif,
        gif_fps=args.gif_fps,
        max_gif_frames=args.max_gif_frames,
        max_memory_gib=args.max_memory_gib,
    )
    for label, file_path in written_files.items():
        logger.info(f"{label}: {file_path}")
