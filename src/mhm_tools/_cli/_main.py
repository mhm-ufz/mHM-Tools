"""Command line interface for mhm-tools."""
import argparse

from .. import __version__
from . import _bankfull


def _get_parser():
    parent_parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parent_parser.add_argument(
        "-V",
        "--version",
        action="version",
        version=__version__,
        help="display version information",
    )

    subparsers = parent_parser.add_subparsers(
        title="subcommands", dest="command", required=True
    )

    # all sub-parsers should be added here

    parser = subparsers.add_parser("bankfull", description=_bankfull.__doc__)
    _bankfull.add_args(parser)
    parser.set_defaults(func=_bankfull.bankfull)

    # return the parser
    return parent_parser


def main(argv=None):
    """
    Main CLI routine.

    Parameters
    ----------
    argv : list of str
        command line arguments, default is None

    Returns
    -------
        result of the called sub-argument routine
    """
    args = _get_parser().parse_args(argv)
    return args.func(args)
