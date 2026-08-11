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
        (
            "lc",
            "mhm_tools.pre.format_lc_data",
            "format_lc_data",
            ["-e", "asc"],
            "asc",
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
            "-c",
            "target",
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
        "class_field": "target",
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
    assert "[soil|geology|lc]" in result.output
    assert "-i, --input-path" in result.output
    assert "-c, --class-field" in result.output
    assert "-e, --extension" in result.output
    assert "[nc|asc|tif]" in result.output


def test_format_data_accepts_input_file_as_a_legacy_alias(monkeypatch):
    """Existing callers may keep using the old --input-file spelling."""
    captured = {}

    def fake_formatter(**kwargs):
        captured.update(kwargs)

    module = importlib.import_module("mhm_tools.pre.format_lc_data")
    monkeypatch.setattr(module, "format_lc_data", fake_formatter)
    result = CliRunner().invoke(
        cli,
        [
            "data-converter",
            "format-data",
            "-t",
            "lc",
            "--input-file",
            "input.tif",
            "-d",
            "dem.tif",
            "-o",
            "output",
            "-l",
            "lookup.gpkg",
            "-m",
            "source",
            "-c",
            "target",
        ],
    )

    assert result.exit_code == 0, result.output
    assert captured["input_file"] == Path("input.tif")


def test_format_data_accepts_underscore_input_path_alias(monkeypatch):
    """The documented underscore spelling remains accepted by the CLI."""
    captured = {}

    def fake_formatter(**kwargs):
        captured.update(kwargs)

    module = importlib.import_module("mhm_tools.pre.format_lc_data")
    monkeypatch.setattr(module, "format_lc_data", fake_formatter)
    result = CliRunner().invoke(
        cli,
        [
            "data-converter",
            "format-data",
            "-t",
            "lc",
            "--input_path",
            "input.tif",
            "-d",
            "dem.tif",
            "-o",
            "output",
            "-l",
            "lookup.gpkg",
            "-m",
            "source",
            "-c",
            "target",
        ],
    )

    assert result.exit_code == 0, result.output
    assert captured["input_file"] == Path("input.tif")


def test_format_data_forwards_explicit_resampling_to_single_files(monkeypatch):
    """The shared resampling option also applies to legacy file inputs."""
    captured = {}

    def fake_formatter(**kwargs):
        captured.update(kwargs)

    module = importlib.import_module("mhm_tools.pre.format_geology")
    monkeypatch.setattr(module, "format_geology_data", fake_formatter)
    result = CliRunner().invoke(
        cli,
        [
            "data-converter",
            "format-data",
            "-t",
            "geology",
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
            "-c",
            "target",
            "--resampling",
            "mode",
        ],
    )

    assert result.exit_code == 0, result.output
    assert captured["resampling"] == "mode"


@pytest.mark.parametrize(
    ("data_type", "module_name", "function_name", "lookup_options"),
    [
        (
            "lc",
            "mhm_tools.pre.format_lc_data",
            "format_lc_periods",
            ["-l", "lookup.gpkg", "-m", "source", "-c", "target"],
        ),
        ("soil", "mhm_tools.pre.format_soil", "format_soil_horizons", []),
    ],
)
def test_format_data_dispatches_directory_manifests(
    tmp_path,
    monkeypatch,
    data_type,
    module_name,
    function_name,
    lookup_options,
):
    """A directory selects the multi-period or multi-horizon public API."""
    input_path = tmp_path / data_type
    input_path.mkdir()
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
            "--input-path",
            str(input_path),
            "-d",
            "dem.tif",
            "-o",
            "output",
            "--resampling",
            "auto",
            *lookup_options,
        ],
    )

    assert result.exit_code == 0, result.output
    assert captured["input_path"] == input_path
    assert captured["resampling"] == "auto"
    assert "input_file" not in captured
    if data_type == "soil":
        assert "lookup_table" not in captured


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
        "-c",
        "target",
    ]
    if option == "-t":
        arguments[arguments.index("soil")] = value
    else:
        arguments.extend([option, value])

    result = CliRunner().invoke(cli, arguments)

    assert result.exit_code != 0
    assert "Invalid value" in result.output


def test_format_data_requires_class_field():
    """The CLI never guesses a lookup class column."""
    result = CliRunner().invoke(
        cli,
        [
            "data-converter",
            "format-data",
            "-t",
            "lc",
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
        ],
    )

    assert result.exit_code != 0
    assert "class-field" in result.output


def test_old_format_commands_are_not_registered():
    """Only the consolidated formatter is listed in data-converter."""
    result = CliRunner().invoke(cli, ["data-converter", "--help"])

    assert result.exit_code == 0, result.output
    assert "format-data" in result.output
    assert "format-soil-data" not in result.output
    assert "format-geology-data" not in result.output
