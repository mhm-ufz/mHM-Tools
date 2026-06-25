"""
Click-based command line interface for mhm-tools.

Authors
- Simon Lüdke
"""

import argparse
import difflib
import importlib
from types import SimpleNamespace
from typing import Callable, List, Optional, Tuple

import click

from mhm_tools.common.logger import configure_mhm_tools_logger

from .. import __version__

try:
    from trogon import tui as trogon_tui
except Exception:  # pragma: no cover - optional dependency
    trogon_tui = None


def _patch_trogon_command_tree_order() -> None:
    """Make Trogon's command tree respect Click insertion order."""
    if trogon_tui is None:
        return
    try:
        from rich.text import Text
        from trogon.widgets.command_tree import CommandTree
    except Exception:
        return

    def _ordered_on_mount(self):
        def build_tree(data, node):
            for cmd_name, cmd_data in data.items():
                if cmd_name == self.command_name:
                    continue
                if cmd_data.subcommands:
                    label = Text(cmd_name)
                    if cmd_data.is_group:
                        group_style = self.get_component_rich_style("group")
                        label.stylize(group_style)
                        label.append(" ")
                        label.append("group", "dim i")
                    child = node.add(label, allow_expand=False, data=cmd_data)
                    build_tree(cmd_data.subcommands, child)
                else:
                    node.add_leaf(cmd_name, data=cmd_data)
            return node

        build_tree(self.cli_metadata, self.root)
        self.root.expand_all()
        self.select_node(self.root)

    CommandTree.on_mount = _ordered_on_mount


_COMMAND_GROUPS: List[Tuple[str, str, List[Tuple[str, object]]]] = [
    (
        "setup-creation",
        "Create and prepare mHM/mRM setup files.",
        [
            ("create-catchment", "mhm_tools._cli._create_catchment"),
            ("crop-mhm-setup", "mhm_tools._cli._crop_mhm_setup"),
            ("latlon", "mhm_tools._cli._latlon"),
            ("create-header", "mhm_tools._cli._create_header"),
            ("calculate-pet", "mhm_tools._cli._calculate_pet"),
            ("prepare-mhm-forcings", "mhm_tools._cli._prepare_mhm_forcings"),
        ],
    ),
    (
        "data-processing",
        "Convert, regrid, merge, and derive gridded data products.",
        [
            ("converter-nc-ascii", "mhm_tools._cli._file_converter"),
            ("merge-files", "mhm_tools._cli._merge"),
            ("fill-nearest", "mhm_tools._cli._fill_nearest"),
            ("regrid-file", "mhm_tools._cli._regrid"),
            ("fill-nearest", "mhm_tools._cli._fill_nearest"),
            ("long-term-mean", "mhm_tools._cli._long_term_mean"),
            ("difference", "mhm_tools._cli._difference"),
            ("relative-difference", "mhm_tools._cli._relative_difference"),
            ("ratio", "mhm_tools._cli._ratio"),
            ("bankfull", "mhm_tools._cli._bankfull"),
        ],
    ),
    (
        "evaluation",
        "Evaluate simulations against observations or reference data.",
        [
            ("discharge-evaluation", "mhm_tools._cli._discharge_evaluation"),
            ("hydrograph", "mhm_tools._cli._hydrograph"),
            ("gridded-data-evaluation", "mhm_tools._cli._gridded_data_evaluation"),
            ("run-overview", "mhm_tools._cli._mhm_run_overview"),
        ],
    ),
    (
        "utilities",
        "General helper commands.",
        [
            ("link-folder-tree", "mhm_tools._cli._link_folder_tree"),
        ],
    ),
    (
        "visualization",
        "Create diagnostic plots and run summaries.",
        [
            ("2d-map", "mhm_tools._cli._2d_map"),
            ("taylor-diagram", "mhm_tools._cli._taylor_diagram"),
        ],
    ),
    (
        "mhm-v5-v6-converter",
        "Convert mHM5 setup files to mHMv6 format.",
        [
            ("landcover-ascii-to-nc", "mhm_tools._cli._landcover_ascii_to_nc"),
        ],
    ),
    (
        "legacy-tools",
        "Legacy tools that are no longer recomended or only created for very specific usecases.",
        [
            ("create-id-gauges", "mhm_tools._cli._create_idgauges"),
            ("create-subdomain-masks", "mhm_tools._cli._create_subdomain_masks"),
            ("create-mhm-restart-file", "mhm_tools._cli._create_mhm_restart_file"),
            (
                "create-mhm-restart-from-setup",
                "mhm_tools._cli._create_mhm_restart_from_setup",
            ),
        ],
    ),
]
_COMMAND_MODULES: List[Tuple[str, object]] = [
    command for _, _, commands in _COMMAND_GROUPS for command in commands
]
_LOG_LEVEL_CHOICES = ("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL")
_GROUP_ORDER = ("required arguments", "optional arguments", "flags", "options")
_ROOT_OPTION_ALIASES = {
    "--log_level": "--log-level",
    "--log_file": "--log-file",
    "--log_file_level": "--log-file-level",
    "--no_console_output": "--no-console-output",
}


def _translate_option_aliases(args, option_aliases):
    """Translate hidden option aliases to their canonical option names."""
    translated_args = []
    for arg in args:
        if not arg.startswith("--"):
            translated_args.append(arg)
            continue
        option_name, separator, option_value = arg.partition("=")
        canonical = option_aliases.get(option_name)
        if canonical is None:
            translated_args.append(arg)
            continue
        translated_args.append(
            f"{canonical}{separator}{option_value}" if separator else canonical
        )
    return translated_args


class AliasGroup(click.Group):
    """Click group that supports aliases and preserves command insertion order."""

    def __init__(self, *args, **kwargs):
        self.command_section_name = kwargs.pop("command_section_name", "Commands")
        super().__init__(*args, **kwargs)
        self._aliases = {}
        self._path_aliases = {}

    def add_alias(self, alias: str, target: str) -> None:
        """Register an alias that resolves to an existing command."""
        if alias in self.commands:
            msg = f"Alias '{alias}' collides with an existing command."
            raise click.ClickException(msg)
        if target not in self.commands:
            msg = f"Cannot alias unknown command '{target}'."
            raise click.ClickException(msg)
        self._aliases[alias] = target

    def add_path_alias(self, alias: str, target_path: Tuple[str, ...]) -> None:
        """Register an alias that expands to a nested command path."""
        if alias in self.commands:
            msg = f"Alias '{alias}' collides with an existing command."
            raise click.ClickException(msg)
        if not target_path:
            msg = f"Cannot alias '{alias}' to an empty command path."
            raise click.ClickException(msg)
        if target_path[0] not in self.commands:
            msg = f"Cannot alias '{alias}' to unknown command path '{target_path}'."
            raise click.ClickException(msg)
        self._path_aliases[alias] = target_path

    def list_commands(self, _ctx):
        """Return commands in their assigned insertion order."""
        return [
            name
            for name, command in self.commands.items()
            if not getattr(command, "hidden", False)
        ]

    def get_command(self, ctx, cmd_name):
        command = super().get_command(ctx, cmd_name)
        if command is not None:
            return command
        target = self._aliases.get(cmd_name)
        if target is None:
            return None
        return super().get_command(ctx, target)

    def resolve_command(self, ctx, args):
        """Resolve command name and provide typo suggestions."""
        if args and args[0] in self._path_aliases:
            args = [*self._path_aliases[args[0]], *args[1:]]
        try:
            return super().resolve_command(ctx, args)
        except click.UsageError as exc:
            if not args:
                raise
            unknown = args[0]
            candidates = sorted(
                set(self.commands.keys())
                | set(self._aliases.keys())
                | set(self._path_aliases.keys())
            )
            suggestions = difflib.get_close_matches(
                unknown, candidates, n=3, cutoff=0.5
            )
            if not suggestions:
                raise
            suggestion_list = ", ".join(suggestions)
            msg = f"No such command '{unknown}'. Did you mean: {suggestion_list}?"
            raise click.UsageError(msg, ctx=ctx) from exc

    def parse_args(self, ctx, args):
        """Translate hidden root option aliases before Click parses arguments."""
        return super().parse_args(
            ctx, _translate_option_aliases(args, _ROOT_OPTION_ALIASES)
        )

    def format_commands(self, ctx, formatter):
        """Render command lists using the configured section heading."""
        rows = []
        for subcommand in self.list_commands(ctx):
            command = self.get_command(ctx, subcommand)
            if command is None or command.hidden:
                continue
            rows.append((subcommand, command))

        if not rows:
            return

        limit = formatter.width - 6 - max(len(row[0]) for row in rows)
        with formatter.section(self.command_section_name):
            formatter.write_dl(
                [
                    (subcommand, command.get_short_help_str(limit))
                    for subcommand, command in rows
                ]
            )


class GroupedOption(click.Option):
    """Click option with an attached source argument-group label."""

    def __init__(self, *args, option_group: str = "options", **kwargs):
        self.option_group = option_group
        super().__init__(*args, **kwargs)


class GroupedCommand(click.Command):
    """Click command that renders options grouped by source parser groups."""

    def __init__(self, *args, option_aliases=None, **kwargs):
        self.option_aliases = option_aliases or {}
        super().__init__(*args, **kwargs)

    def parse_args(self, ctx, args):
        """Translate legacy option aliases before Click parses arguments."""
        return super().parse_args(
            ctx, _translate_option_aliases(args, self.option_aliases)
        )

    def format_options(self, ctx, formatter):
        grouped_records = {}
        for param in self.get_params(ctx):
            if not isinstance(param, click.Option):
                continue
            record = param.get_help_record(ctx)
            if record is None:
                continue
            group_name = getattr(param, "option_group", "options")
            grouped_records.setdefault(group_name, []).append(record)

        if not grouped_records:
            return

        ordered_groups = []
        lowered_keys = {key.lower(): key for key in grouped_records}
        for preferred in _GROUP_ORDER:
            key = lowered_keys.get(preferred)
            if key is not None:
                ordered_groups.append(key)
        for key in grouped_records:
            if key not in ordered_groups:
                ordered_groups.append(key)

        for group_name in ordered_groups:
            with formatter.section(group_name):
                formatter.write_dl(grouped_records[group_name])


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


def _canonical_option_strings(option_strings):
    """Return visible option declarations while leaving aliases hidden."""
    canonical = []
    first_long_option = None
    for option in option_strings:
        if option.startswith("--"):
            if first_long_option is None:
                first_long_option = option
            continue
        canonical.append(option)
    if first_long_option is not None:
        canonical.append(first_long_option)
    return tuple(canonical)


def _option_aliases(option_strings):
    """Map hidden option aliases to their visible canonical long option."""
    canonical_options = _canonical_option_strings(option_strings)
    canonical_long_options = [
        option for option in canonical_options if option.startswith("--")
    ]
    if not canonical_long_options:
        return {}
    canonical_long_option = canonical_long_options[0]
    aliases = {}
    for option in _expand_long_option_aliases(option_strings):
        if not option.startswith("--") or option in canonical_options:
            continue
        aliases[option] = canonical_long_option
    return aliases


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


def _contiguous_int_choices(choices):
    """Return integer range bounds when choices are contiguous integers."""
    try:
        int_choices = sorted(int(choice) for choice in choices)
    except (TypeError, ValueError):
        return None
    if not int_choices:
        return None
    expected = list(range(int_choices[0], int_choices[-1] + 1))
    if int_choices != expected:
        return None
    return int_choices[0], int_choices[-1]


def _action_to_click_option(action: argparse.Action, option_group: str = "options"):
    """Convert one parser action to a Click option."""
    option_strings = tuple(_canonical_option_strings(action.option_strings))
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
            if action.type is int:
                choice_range = _contiguous_int_choices(action.choices)
                if choice_range is not None:
                    kwargs["type"] = click.IntRange(*choice_range)
                else:
                    kwargs["type"] = int
            elif action.type is float:
                kwargs["type"] = float
            else:
                kwargs["type"] = click.Choice(
                    [str(choice) for choice in action.choices]
                )
        kwargs["callback"] = _convert_callback(action)

    param_decls = [*list(option_strings), action.dest]
    return GroupedOption(
        param_decls=param_decls,
        option_group=option_group,
        **kwargs,
    )


def _build_click_command(command_name: str, module, prog_path: Optional[str] = None):
    """Build one Click command from a parser-based command module."""
    parser = argparse.ArgumentParser(
        prog=prog_path or f"mhm-tools {command_name}",
        add_help=False,
    )
    module.add_args(parser)

    action_groups = {}
    for group in parser._action_groups:
        group_title = group.title or "options"
        for group_action in group._group_actions:
            action_groups[id(group_action)] = group_title

    params = []
    option_aliases = {}
    for action in parser._actions:
        if isinstance(action, argparse._HelpAction):
            continue
        option_group = action_groups.get(id(action), "options")
        option = _action_to_click_option(action, option_group=option_group)
        if option is not None:
            params.append(option)
            option_aliases.update(_option_aliases(action.option_strings))

    def _callback(**kwargs):
        args = SimpleNamespace(**kwargs)
        return module.run(args)

    return GroupedCommand(
        name=command_name,
        callback=_callback,
        params=params,
        help=module.__doc__,
        option_aliases=option_aliases,
        context_settings={"help_option_names": ["-h", "--help"]},
    )


def _add_dash_underscore_alias(group: AliasGroup, command_name: str) -> None:
    """Add dash/underscore command aliases when applicable."""
    aliases = []
    if "-" in command_name:
        aliases.append(command_name.replace("-", "_"))
    if "_" in command_name:
        aliases.append(command_name.replace("_", "-"))
    for alias in aliases:
        if alias not in group.commands:
            group.add_alias(alias, command_name)


def _add_command_with_aliases(
    group: AliasGroup,
    command: click.Command,
    command_name: str,
) -> None:
    """Register a command and its dash/underscore alias."""
    group.add_command(command, name=command_name)
    _add_dash_underscore_alias(group, command_name)


def _add_path_aliases(
    group: AliasGroup,
    command_name: str,
    target_path: Tuple[str, ...],
) -> None:
    """Register legacy top-level aliases that expand to a grouped command."""
    aliases = [command_name]
    if "-" in command_name:
        aliases.append(command_name.replace("-", "_"))
    if "_" in command_name:
        aliases.append(command_name.replace("_", "-"))
    for alias in aliases:
        group.add_path_alias(alias, target_path)


def _hide_tui_launcher(group: click.Group) -> None:
    """Hide the TUI launcher from command-group listings."""
    tui_command = group.commands.get("tui")
    if tui_command is not None:
        tui_command.hidden = True


@click.group(
    cls=AliasGroup,
    context_settings={"help_option_names": ["-h", "--help"]},
    subcommand_metavar="GROUP COMMAND [ARGS]...",
    command_section_name="Command Groups",
)
@click.version_option(__version__, "-V", "--version", prog_name="mhm-tools")
@click.option(
    "--log-level",
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
    type=str,
    default=None,
    help="Generate a log file.",
)
@click.option(
    "--log-file-level",
    type=str,
    callback=_validate_log_level_option,
    default=None,
    help="Set log level for the log file. Defaults to console log level.",
)
@click.option(
    "--no-console-output",
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


_LEGACY_COMMAND_PATHS = []
for _group_name, _group_help, _group_commands in _COMMAND_GROUPS:
    _group = AliasGroup(
        name=_group_name,
        help=_group_help,
        context_settings={"help_option_names": ["-h", "--help"]},
    )
    for _command_name, _module in _group_commands:
        # If the module is a string, import it lazily here.
        if isinstance(_module, str):
            _module_obj = importlib.import_module(_module)
        else:
            _module_obj = _module
        _cmd = _build_click_command(
            _command_name,
            _module_obj,
            prog_path=f"mhm-tools {_group_name} {_command_name}",
        )
        _add_command_with_aliases(_group, _cmd, _command_name)
        _LEGACY_COMMAND_PATHS.append((_command_name, (_group_name, _command_name)))
    _add_command_with_aliases(cli, _group, _group_name)

for _command_name, _target_path in _LEGACY_COMMAND_PATHS:
    _add_path_aliases(cli, _command_name, _target_path)


if trogon_tui is not None:
    _patch_trogon_command_tree_order()
    cli = trogon_tui()(cli)
    _hide_tui_launcher(cli)
else:

    @cli.command("tui", help="Open Textual TUI (requires 'trogon').")
    def _missing_tui():
        msg = "trogon is not installed. Install with: pip install trogon"
        raise click.ClickException(msg)

    _hide_tui_launcher(cli)


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
