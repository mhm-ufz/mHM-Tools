"""
Command-line interface for nearest-neighbour NetCDF gap filling.

This interpolator should not be used on global scale as it will be slow.

Authors
- Simon Lüdke
- Sebastian Müller
"""

from pathlib import Path

from mhm_tools.common.cli_utils import (
    add_netcdf_compression_args,
    get_netcdf_compression,
)


def add_args(parser):
    """Register arguments for the ``fill-nearest`` subcommand.

    Parameters
    ----------
    parser : argparse.ArgumentParser
        Subcommand parser to extend with fill-nearest options.
    """
    optional = parser.add_argument_group("optional arguments")
    optional.add_argument(
        "-i",
        "--input-dir",
        default=".",
        help="Directory containing input NetCDF files.",
    )
    optional.add_argument(
        "-f",
        "--input-file-name",
        "--fname",
        "--input-name",
        dest="fname",
        default="precipitation_*.nc",
        help="Input filename pattern, for example 'precipitation_*.nc'.",
    )
    optional.add_argument(
        "-o",
        "--output-dir",
        required=True,
        help="Directory where filled NetCDF files are written.",
    )
    optional.add_argument(
        "-m",
        "--mask-file",
        default=None,
        help="Optional NetCDF file used to derive a fixed output mask.",
    )
    optional.add_argument(
        "--mask-var",
        default=None,
        help="Variable in --mask-file whose NaN cells define the fixed output mask.",
    )
    optional.add_argument(
        "--fill-value",
        default=-9999.0,
        type=float,
        help="Fill value written to cells masked by --mask-file/--mask-var.",
    )
    optional.add_argument(
        "--ncpus",
        "--n-cpus",
        dest="n_cpus",
        default=1,
        type=int,
        help="Number of cpus for parallel processing of multiple files.",
    )
    add_netcdf_compression_args(parser)


def run(args):
    """Fill missing NetCDF values using nearest valid neighbours."""
    from mhm_tools.pre.fill_nearest import fill_nearest

    fill_nearest(
        input_dir=Path(args.input_dir),
        fname=args.fname,
        output_dir=Path(args.output_dir),
        mask_file=Path(args.mask_file) if args.mask_file else None,
        mask_var=args.mask_var,
        fill_value=float(args.fill_value),
        default_value=args.default_value,
        n_cpus=args.n_cpus,
        compression=get_netcdf_compression(args),
    )
