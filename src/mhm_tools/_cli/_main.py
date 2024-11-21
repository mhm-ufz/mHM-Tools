"""Command line interface for mhm-tools."""

import argparse

from mhm_tools.common.logger import set_log_level

from .. import __version__
from . import (
    _bankfull,
    _create_catchment,
    _create_mhm_restart_file,
    _create_subdomain_masks,
    _latlon,
)


class Formatter(
    argparse.ArgumentDefaultsHelpFormatter, argparse.RawDescriptionHelpFormatter
):
    """Custom formatter for argparse with help and raw text."""


def add_command_from_module(subparsers, name, module):
    """
    Add a subcommand from a given module.

    Parameters
    ----------
    subparsers : subparsers
        Subparser to add the command to.
    name : str
        Name of the command to add.
    module : module
        Module containing the `add_args` and `run` functions defining the command.
    """
    desc = module.__doc__
    kwargs = {"description": desc}
    if desc:
        kwargs["help"] = desc.splitlines()[0]
    parser = subparsers.add_parser(name, formatter_class=Formatter, **kwargs)
    module.add_args(parser)
    parser.set_defaults(func=module.run)


def _get_parser():
    parent_parser = argparse.ArgumentParser(
        prog="mhm-tools",
        description=__doc__,
        formatter_class=Formatter,
    )

    parent_parser.add_argument(
        "-V",
        "--version",
        action="version",
        version=__version__,
        help="Display version information.",
    )

    sub_help = (
        "All tools are provided as sub-commands. "
        "Please refer to the respective help texts."
    )
    subparsers = parent_parser.add_subparsers(
        title="Available Tools", dest="command", required=True, description=sub_help
    )

    # all sub-parsers should be added here
    # documentation taken from docstring of respective cli module (first line summary)
    # module needs two functions: add_args and run

    add_command_from_module(subparsers, "bankfull", _bankfull)
    add_command_from_module(subparsers, "latlon", _latlon)
    add_command_from_module(subparsers, "create_catchment", _create_catchment)
    add_command_from_module(
        subparsers, "create_subdomain_masks", _create_subdomain_masks
    )
    add_command_from_module(
        subparsers, "create_mhm_restart_file", _create_mhm_restart_file
    )

    # add logging 
    parent_parser.add_argument(
        "--log_level",
        type=str, 
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        default="INFO",
        help="Set the logging level"
    )


    # return the parser
    return parent_parser


def main(argv=None):
    """
    Execute main CLI routine.

    Parameters
    ----------
    argv : list of str
        command line arguments, default is None

    Returns
    -------
        result of the called sub-argument routine
    """
    args = _get_parser().parse_args(argv)
    set_log_level(args.log_level)
    return args.func(args)
