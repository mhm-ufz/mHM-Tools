"""Command line interface for mhm-tools."""

import argparse

from mhm_tools.common.logger import configure_mhm_tools_logger

from .. import __version__
from . import (
    _2d_map,
    _bankfull,
    _calculate_pet,
    _create_catchment,
    _create_idgauges,
    _create_mhm_restart_file,
    _create_subdomain_masks,
    _crop_mhm_setup,
    _difference,
    _discharge_evaluation,
    _file_converter,
    _gridded_data_evaluation,
    _hydrograph,
    _landcover_ascii_to_nc,
    _latlon,
    _link_folder_tree,
    _long_term_mean,
    _merge,
    _mhm_run_overview,
    _prepare_mhm_forcings,
    _ratio,
    _regrid,
    _relative_difference,
    _taylor_diagram,
)


class Formatter(
    argparse.ArgumentDefaultsHelpFormatter, argparse.RawDescriptionHelpFormatter
):
    """Custom formatter for argparse with help and raw text."""


def _expand_long_option_aliases(option_strings):
    """Add dash/underscore aliases for long options."""
    expanded = []
    seen = set()
    for option in option_strings:
        if option not in seen:
            expanded.append(option)
            seen.add(option)
        if not option.startswith("--"):
            continue
        if "-" in option[2:]:
            alias = "--" + option[2:].replace("-", "_")
            if alias not in seen:
                expanded.append(alias)
                seen.add(alias)
        if "_" in option[2:]:
            alias = "--" + option[2:].replace("_", "-")
            if alias not in seen:
                expanded.append(alias)
                seen.add(alias)
    return tuple(expanded)


def _patch_actions_container(container):
    """Patch group containers to auto-add dash/underscore aliases."""
    original_add_argument = container.add_argument

    def add_argument_with_aliases(*name_or_flags, **kwargs):
        return original_add_argument(
            *_expand_long_option_aliases(name_or_flags), **kwargs
        )

    container.add_argument = add_argument_with_aliases


class AliasArgumentParser(argparse.ArgumentParser):
    """ArgumentParser that accepts both dash and underscore long options."""

    def add_argument(self, *name_or_flags, **kwargs):
        return super().add_argument(
            *_expand_long_option_aliases(name_or_flags), **kwargs
        )

    def add_argument_group(self, *args, **kwargs):
        group = super().add_argument_group(*args, **kwargs)
        _patch_actions_container(group)
        return group

    def add_mutually_exclusive_group(self, **kwargs):
        group = super().add_mutually_exclusive_group(**kwargs)
        _patch_actions_container(group)
        return group


def add_command_from_module(subparsers, name, module):
    """Add a subcommand from a given module.

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
    primary = name.replace("_", "-") if "_" in name else name
    aliases = [primary.replace("-", "_")] if "-" in primary else []
    parser = subparsers.add_parser(
        primary, aliases=aliases, formatter_class=Formatter, **kwargs
    )
    module.add_args(parser)
    parser.set_defaults(func=module.run)


def _get_parser():
    parent_parser = AliasArgumentParser(
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
        title="Available Tools",
        dest="command",
        required=True,
        description=sub_help,
        parser_class=AliasArgumentParser,
    )

    # all sub-parsers should be added here
    # documentation taken from docstring of respective cli module (first line summary)
    # module needs two functions: add_args and run

    add_command_from_module(subparsers, "bankfull", _bankfull)

    add_command_from_module(subparsers, "hydrograph", _hydrograph)
    add_command_from_module(
        subparsers, "gridded-data-evaluation", _gridded_data_evaluation
    )
    add_command_from_module(subparsers, "discharge-evaluation", _discharge_evaluation)
    add_command_from_module(subparsers, "latlon", _latlon)
    add_command_from_module(subparsers, "converter-nc-ascii", _file_converter)
    add_command_from_module(subparsers, "landcover-ascii-to-nc", _landcover_ascii_to_nc)
    add_command_from_module(subparsers, "merge-files", _merge)
    add_command_from_module(subparsers, "regrid-file", _regrid)
    add_command_from_module(subparsers, "create-catchment", _create_catchment)
    add_command_from_module(subparsers, "crop-mhm-setup", _crop_mhm_setup)
    add_command_from_module(subparsers, "prepare-mhm-forcings", _prepare_mhm_forcings)
    add_command_from_module(subparsers, "calculate-pet", _calculate_pet)
    add_command_from_module(subparsers, "create-id-gauges", _create_idgauges)
    add_command_from_module(
        subparsers, "create-subdomain-masks", _create_subdomain_masks
    )
    add_command_from_module(
        subparsers, "create-mhm-restart-file", _create_mhm_restart_file
    )
    add_command_from_module(subparsers, "long-term-mean", _long_term_mean)
    add_command_from_module(subparsers, "difference", _difference)
    add_command_from_module(subparsers, "relative-difference", _relative_difference)
    add_command_from_module(subparsers, "ratio", _ratio)
    add_command_from_module(subparsers, "taylor-diagram", _taylor_diagram)
    add_command_from_module(subparsers, "2d-map", _2d_map)
    add_command_from_module(subparsers, "link-folder-tree", _link_folder_tree)
    add_command_from_module(subparsers, "run-overview", _mhm_run_overview)

    # add logging
    # option 1 explicit log levels by name
    parent_parser.add_argument(
        "--log-level",
        "--log_level",
        type=str,
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        default=None,
        help="Set the log level explicitly.",
    )
    # option 2 regulation verbosity by -v and -q flags default is INFO
    parent_parser.add_argument(
        "--verbose", "-v", action="count", default=0, help="Increase verbosity"
    )
    parent_parser.add_argument(
        "--quiet",
        "-q",
        action="count",
        default=0,
        help="Reduce verbosity can be repeted e.g. -qq",
    )
    # handle file and terminal output
    parent_parser.add_argument(
        "--log-file", "--log_file", type=str, default=None, help="Generate a log file."
    )
    parent_parser.add_argument(
        "--log-file-level",
        "--log_file_level",
        type=str,
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        default=None,
        help="Set log level for the log file. Defaults to console log level.",
    )
    parent_parser.add_argument(
        "--no-console-output",
        "--no_console_output",
        action="store_true",
        help="Prohibit console output.",
    )

    # return the parser
    return parent_parser


def main(argv=None):
    """Execute main CLI routine.

    Parameters
    ----------
    argv : list of str
        command line arguments, default is None

    Returns
    -------
        result of the called sub-argument routine
    """
    args = _get_parser().parse_args(argv)
    configure_mhm_tools_logger(
        log_level=args.log_level,
        count_verbose=args.verbose,
        count_quiet=args.quiet,
        log_file=args.log_file,
        log_file_level=args.log_file_level,
        no_colsole_logging=args.no_console_output,
    )
    return args.func(args)
