"""Regrid to L2 aligned with L0 grid from mask.nc.

Authors
-------
- Simon Lüdke
"""

from pathlib import Path


def add_args(parser):
    """Add cli arguments.

    Parameters
    ----------
    parser : argparse.ArgumentParser
        the main argument parser

    """
    optional = parser.add_argument_group("optional arguments")
    optional.add_argument(
        "--input-file", "--input", dest="input", required=True, help="Input NetCDF"
    )
    optional.add_argument(
        "--mask-file",
        "--mask",
        dest="mask",
        required=True,
        help="mask.nc defining L0 grid (must have lon/lat coords)",
    )
    optional.add_argument(
        "--output-file",
        "--output",
        dest="output",
        required=True,
        help="Output NetCDF (L2 grid)",
    )
    optional.add_argument("--l2", required=True, help="L2 resolution: e.g. 0.05, 0.1")
    optional.add_argument(
        "--method",
        default="nearest",
        choices=["nearest", "linear"],
        help="Regridding method",
    )


def run(args):
    """Regrid input NetCDF to an L2 grid derived from a mask file.

    Parameters
    ----------
    args : argparse.Namespace
        parsed command line arguments
    """
    from mhm_tools.pre.regrid import regrid

    input = Path(args.input)
    output = Path(args.output)
    mask = Path(args.mask)
    msg = ""
    for file in [input, output, mask]:
        if not file.is_file():
            msg += f"{input!s} is not a file; "
    l2 = float(args.l2)
    regrid(input=input, output=output, mask=mask, l2=l2, method=args.method)
