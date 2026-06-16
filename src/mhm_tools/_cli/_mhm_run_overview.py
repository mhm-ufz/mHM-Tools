#!/usr/bin/env python
"""Create a variable overview for an mHM run.

The tool reads an mHM namelist, resolves configured input and output files, and
writes a CSV summary of variables, file paths, and basic statistics. It can
search input folders recursively, compute temporal means before statistics, and
convert time-based output units to the inferred meteo resolution.

Authors
-------
- Simon Lüdke
"""

import argparse
from pathlib import Path


def add_args(parser: argparse.ArgumentParser):
    """Add CLI arguments for the mhm_run_overview subcommand."""
    required = parser.add_argument_group("required arguments")
    required.add_argument(
        "--namelist",
        type=Path,
        required=True,
        help="Path to mhm.nml",
    )
    required.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help=(
            "Output directory for CSV (default file: variable_summary.csv), "
            "or a direct .csv file path."
        ),
    )

    optional = parser.add_argument_group("optional arguments")
    optional.add_argument(
        "--base-path",
        type=Path,
        default=None,
        help=(
            "Optional base path for relative paths in the namelist. "
            "If omitted, the namelist parent directory is used."
        ),
    )
    flags = parser.add_argument_group("flags")
    flags.add_argument(
        "--recursive-input-search",
        action="store_true",
        help="Recursively search input directories for NetCDF files.",
    )
    flags.add_argument(
        "--temporal-mean",
        action="store_true",
        help="Compute temporal means for variables with a 2 spatial dimensions and a time dimension before computing statistics.",
    )
    flags.add_argument(
        "--convert-units",
        action="store_true",
        help=(
            "Convert output variable units with time denominators (s, d, m, y) "
            "to the inferred temporal resolution of meteo input files with time coordinates. "
            "Month length is assumed as 30.4 days."
        ),
    )


def run(args: argparse.Namespace):
    """Run run-overview statistics export."""
    from mhm_tools.post.mhm_run_overview import create_mhm_run_overview

    create_mhm_run_overview(
        namelist_file=args.namelist.resolve(),
        base_path=args.base_path.resolve() if args.base_path is not None else None,
        output_dir=args.output_dir.resolve(),
        recursive_input_search=args.recursive_input_search,
        temporal_mean=args.temporal_mean,
        convert_units=args.convert_units,
    )
