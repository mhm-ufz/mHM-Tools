"""Click-based command line interface for mhm-tools."""

import argparse
from types import SimpleNamespace
from typing import Callable, List, Tuple

import click

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

try:
    from trogon import tui as trogon_tui
except Exception:  # pragma: no cover - optional dependency
    trogon_tui = None

_COMMAND_MODULES: List[Tuple[str, object]] = [
    ("bankfull", _bankfull),
    ("hydrograph", _hydrograph),
    ("gridded-data-evaluation", _gridded_data_evaluation),
    ("discharge-evaluation", _discharge_evaluation),
    ("latlon", _latlon),
    ("converter-nc-ascii", _file_converter),
    ("landcover-ascii-to-nc", _landcover_ascii_to_nc),
    ("merge-files", _merge),
    ("regrid-file", _regrid),
    ("create-catchment", _create_catchment),
    ("crop-mhm-setup", _crop_mhm_setup),
    ("prepare-mhm-forcings", _prepare_mhm_forcings),
    ("calculate-pet", _calculate_pet),
    ("create-id-gauges", _create_idgauges),
    ("create-subdomain-masks", _create_subdomain_masks),
    ("create-mhm-restart-file", _create_mhm_restart_file),
    ("long-term-mean", _long_term_mean),
    ("difference", _difference),
    ("relative-difference", _relative_difference),
    ("ratio", _ratio),
    ("taylor-diagram", _taylor_diagram),
    ("2d-map", _2d_map),
    ("link-folder-tree", _link_folder_tree),
    ("run-overview", _mhm_run_overview),
]
_LOG_LEVEL_CHOICES = ("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL")


class AliasGroup(click.Group):
    """Click group that supports hidden command aliases."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._aliases = {}

    def add_alias(self, alias: str, target: str) -> None:
        """Register an alias that resolves to an existing command."""
        if alias in self.commands:
            msg = f"Alias '{alias}' collides with an existing command."
            raise click.ClickException(msg)
        if target not in self.commands:
            msg = f"Cannot alias unknown command '{target}'."
            raise click.ClickException(msg)
        self._aliases[alias] = target

    def get_command(self, ctx, cmd_name):
        command = super().get_command(ctx, cmd_name)
        if command is not None:
            return command
        target = self._aliases.get(cmd_name)
        if target is None:
            return None
        return super().get_command(ctx, target)


def _validate_log_level_option(ctx, param, value):  # noqa: ARG001
    """Validate log-level style options without using click.Choice.

    Using plain strings avoids a Trogon `NoSelection` crash path for optional
    choice widgets with no explicit default selection.
    """
    if value is None:
        return None
    normalized = str(value).upper()
    if normalized not in _LOG_LEVEL_CHOICES:
        choices = ", ".join(_LOG_LEVEL_CHOICES)
        msg = f"must be one of: {choices}"
        raise click.BadParameter(msg)
    return normalized


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


def _normalize_default(action: argparse.Action):
    """Normalize parser defaults for Click option declarations."""
    default = action.default
    normalized_default = default
    if default is argparse.SUPPRESS:
        normalized_default = None
    elif isinstance(action, argparse._CountAction) and default is None:
        normalized_default = 0
    elif isinstance(action, argparse._StoreTrueAction) and default is None:
        normalized_default = False
    elif isinstance(action, argparse._StoreFalseAction) and default is None:
        normalized_default = True
    elif (isinstance(action, argparse._AppendAction) and default is None) or (
        action.nargs in ("+", "*") and default is None
    ):
        normalized_default = ()
    return normalized_default


def _convert_scalar(value, action: argparse.Action):
    """Apply parser conversion semantics for a single option value."""
    converted = value
    if action.type is not None:
        try:
            converted = action.type(value)
        except Exception as exc:
            raise click.BadParameter(str(exc)) from exc
    if action.choices is not None and converted not in action.choices:
        msg = f"must be one of {list(action.choices)}"
        raise click.BadParameter(msg)
    return converted


def _convert_callback(action: argparse.Action) -> Callable:
    """Create a Click callback that mirrors parser value conversion."""

    def _callback(ctx, param, value):  # noqa: ARG001
        if value is None:
            return None
        if isinstance(action, argparse._AppendAction):
            if value is None:
                return []
            return [_convert_scalar(item, action) for item in value]
        if action.nargs in ("+", "*"):
            return [_convert_scalar(item, action) for item in value]
        if isinstance(action.nargs, int) and action.nargs > 1:
            return [_convert_scalar(item, action) for item in value]
        return _convert_scalar(value, action)

    return _callback


def _action_to_click_option(action: argparse.Action):
    """Convert one parser action to a Click option."""
    option_strings = tuple(_expand_long_option_aliases(action.option_strings))
    if not option_strings:
        return None

    kwargs = {
        "required": action.required,
        "help": None if action.help == argparse.SUPPRESS else action.help,
        "show_default": True,
    }
    if not action.required:
        kwargs["default"] = _normalize_default(action)

    if isinstance(action, argparse._CountAction):
        kwargs["count"] = True
    elif isinstance(action, argparse._StoreTrueAction):
        kwargs["is_flag"] = True
        kwargs["flag_value"] = True
    elif isinstance(action, argparse._StoreFalseAction):
        kwargs["is_flag"] = True
        kwargs["flag_value"] = False
    else:
        if action.nargs in ("+", "*"):
            kwargs["multiple"] = True
            if kwargs.get("default") is None and not action.required:
                kwargs["default"] = ()
        elif isinstance(action.nargs, int) and action.nargs > 1:
            kwargs["nargs"] = action.nargs
        if action.type in (int, float, str, bool):
            kwargs["type"] = action.type
        elif action.type is None:
            kwargs["type"] = str
        else:
            kwargs["type"] = str
        if action.choices is not None:
            kwargs["type"] = click.Choice([str(choice) for choice in action.choices])
        kwargs["callback"] = _convert_callback(action)

    param_decls = [*list(option_strings), action.dest]
    return click.Option(param_decls=param_decls, **kwargs)


def _build_click_command(command_name: str, module):
    """Build one Click command from a parser-based command module."""
    parser = argparse.ArgumentParser(
        prog=f"mhm-tools {command_name}",
        add_help=False,
    )
    module.add_args(parser)

    params = []
    for action in parser._actions:
        if isinstance(action, argparse._HelpAction):
            continue
        option = _action_to_click_option(action)
        if option is not None:
            params.append(option)

    def _callback(**kwargs):
        args = SimpleNamespace(**kwargs)
        return module.run(args)

    return click.Command(
        name=command_name,
        callback=_callback,
        params=params,
        help=module.__doc__,
        context_settings={"help_option_names": ["-h", "--help"]},
    )


@click.group(
    cls=AliasGroup,
    context_settings={"help_option_names": ["-h", "--help"]},
)
@click.version_option(__version__, "-V", "--version", prog_name="mhm-tools")
@click.option(
    "--log-level",
    "--log_level",
    type=str,
    callback=_validate_log_level_option,
    default=None,
    help="Set the log level explicitly.",
)
@click.option("--verbose", "-v", count=True, help="Increase verbosity")
@click.option(
    "--quiet",
    "-q",
    count=True,
    help="Reduce verbosity can be repeated e.g. -qq",
)
@click.option(
    "--log-file",
    "--log_file",
    type=str,
    default=None,
    help="Generate a log file.",
)
@click.option(
    "--log-file-level",
    "--log_file_level",
    type=str,
    callback=_validate_log_level_option,
    default=None,
    help="Set log level for the log file. Defaults to console log level.",
)
@click.option(
    "--no-console-output",
    "--no_console_output",
    is_flag=True,
    default=False,
    help="Prohibit console output.",
)
def cli(
    log_level,
    verbose,
    quiet,
    log_file,
    log_file_level,
    no_console_output,
):
    """All tools are provided as sub-commands."""
    configure_mhm_tools_logger(
        log_level=log_level,
        count_verbose=verbose,
        count_quiet=quiet,
        log_file=log_file,
        log_file_level=log_file_level,
        no_colsole_logging=no_console_output,
    )


if trogon_tui is not None:
    cli = trogon_tui()(cli)
else:

    @cli.command("tui", help="Open Textual TUI (requires 'trogon').")
    def _missing_tui():
        msg = "trogon is not installed. Install with: pip install trogon"
        raise click.ClickException(msg)


for _command_name, _module in _COMMAND_MODULES:
    _cmd = _build_click_command(_command_name, _module)
    cli.add_command(_cmd, name=_command_name)
    if "-" in _command_name:
        cli.add_alias(_command_name.replace("-", "_"), _command_name)


def main(argv=None):
    """Execute the main click CLI."""
    try:
        return cli.main(args=argv, prog_name="mhm-tools", standalone_mode=False)
    except click.ClickException as exc:
        exc.show()
        return exc.exit_code
    except click.Abort:
        click.echo("Aborted!", err=True)
        return 1
