"""Regrid to L2 aligned with L0 grid from mask.nc"""

from pathlib import Path

from mhm_tools.pre.regrid import regrid


def add_args(parser):
    """Add cli arguments.

    Parameters
    ----------
    parser : argparse.ArgumentParser
        the main argument parser

    """
    parser.add_argument("--input", required=True, help="Input NetCDF")
    parser.add_argument("--mask", required=True, help="mask.nc defining L0 grid (must have lon/lat coords)")
    parser.add_argument("--output", required=True, help="Output NetCDF (L2 grid)")
    parser.add_argument("--l2", required=True, help="L2 resolution: e.g. 0.05, 0.1")
    parser.add_argument("--method", default="nearest", choices=["nearest", "linear", "bilinear"],
                   help="Regridding method")
def run(args):
    input = Path(args.input)

    output = Path(args.output)
    mask = Path(args.mask)
    msg = ""
    for file in [input, output, mask]:
        if not file.is_file():
            msg += f"{str(input)} is not a file; "
    l2 = float(args.l2)
    regrid(input=input, output=output, mask=mask, l2=l2, method=args.method)
