"""Tests for the consolidated categorical-data formatter CLI."""

import importlib
from pathlib import Path

import pytest
from click.testing import CliRunner

from mhm_tools._cli._main import cli


@pytest.mark.parametrize(
    ("data_type", "module_name", "function_name", "extension_args", "extension"),
    [
        (
            "soil",
            "mhm_tools.pre.format_soil",
            "format_soil_data",
            ["-e", "tif"],
            "tif",
        ),
        (
            "geology",
            "mhm_tools.pre.format_geology",
            "format_geology_data",
            [],
            "nc",
        ),
    ],
)
def test_format_data_dispatches_with_shared_options(
    monkeypatch,
    data_type,
    module_name,
    function_name,
    extension_args,
    extension,
):
    """The selected formatter receives paths, extension, and CRS options."""
    captured = {}

    def fake_formatter(**kwargs):
        captured.update(kwargs)

    module = importlib.import_module(module_name)
    monkeypatch.setattr(module, function_name, fake_formatter)
    result = CliRunner().invoke(
        cli,
        [
            "data-converter",
            "format-data",
            "-t",
            data_type,
            "-i",
            "input.tif",
            "-d",
            "dem.asc",
            "-o",
            "output",
            "-l",
            "lookup.gpkg",
            "-m",
            "source",
            "-s",
            "EPSG:32632",
            "-r",
            "EPSG:32633",
            *extension_args,
        ],
    )

    assert result.exit_code == 0, result.output
    assert captured == {
        "input_file": Path("input.tif"),
        "dem_file": Path("dem.asc"),
        "output_path": Path("output"),
        "lookup_table": Path("lookup.gpkg"),
        "mapping_field": "source",
        "output_type": extension,
        "input_crs": "EPSG:32632",
        "dem_crs": "EPSG:32633",
    }


def test_format_data_help_and_alias():
    """Help exposes the new choices through both command spellings."""
    runner = CliRunner()
    result = runner.invoke(cli, ["data-converter", "format-data", "--help"])
    alias_result = runner.invoke(cli, ["data-converter", "format_data", "--help"])

    assert result.exit_code == 0, result.output
    assert alias_result.exit_code == 0, alias_result.output
    assert "-t, --type" in result.output
    assert "[soil|geology]" in result.output
    assert "-e, --extension" in result.output
    assert "[nc|asc|tif]" in result.output


@pytest.mark.parametrize(
    ("option", "value"),
    [("-t", "landcover"), ("-e", "tiff")],
)
def test_format_data_rejects_invalid_choices(option, value):
    """Unsupported data types and output extensions fail during parsing."""
    arguments = [
        "data-converter",
        "format-data",
        "-t",
        "soil",
        "-i",
        "input.tif",
        "-d",
        "dem.tif",
        "-o",
        "output",
        "-l",
        "lookup.gpkg",
        "-m",
        "source",
    ]
    if option == "-t":
        arguments[arguments.index("soil")] = value
    else:
        arguments.extend([option, value])

    result = CliRunner().invoke(cli, arguments)

    assert result.exit_code != 0
    assert "Invalid value" in result.output


def test_old_format_commands_are_not_registered():
    """Only the consolidated formatter is listed in data-converter."""
    result = CliRunner().invoke(cli, ["data-converter", "--help"])

    assert result.exit_code == 0, result.output
    assert "format-data" in result.output
    assert "format-soil-data" not in result.output
    assert "format-geology-data" not in result.output
