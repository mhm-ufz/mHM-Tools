"""Tests for the format-morph-categorical, format-soilgrids, format-lai-netcdf CLIs."""

import importlib
from pathlib import Path

import pytest
from click.testing import CliRunner

from mhm_tools._cli._main import cli
from mhm_tools.common.netcdf import NetcdfCompression

GRID_ARGS = ["-d", "dem.tif", "-o", "output"]
LOOKUP_ARGS = ["-l", "lookup.gpkg", "-m", "source", "-c", "target"]


def _capture_formatter(monkeypatch, module_name, function_name):
    """Replace a formatter with a stub recording its keyword arguments.

    Parameters
    ----------
    monkeypatch : pytest.MonkeyPatch
        Fixture used to replace the formatter.
    module_name : str
        Module holding the formatter.
    function_name : str
        Name of the formatter.

    Returns
    -------
    dict
        Filled with the keyword arguments once the formatter is called.
    """
    captured = {}
    module = importlib.import_module(module_name)
    monkeypatch.setattr(module, function_name, lambda **kwargs: captured.update(kwargs))
    return captured


def _invoke(command, *arguments):
    """Run a data-converter command through the CLI.

    Parameters
    ----------
    command : str
        Name of the data-converter command.
    *arguments : str
        Command-line arguments of the command.

    Returns
    -------
    click.testing.Result
        Result holding the exit code and the output.
    """
    return CliRunner().invoke(cli, ["data-converter", command, *arguments])


@pytest.mark.parametrize(
    ("data_type", "module_name", "function_name", "extension_args", "extension"),
    [
        ("soil", "mhm_tools.pre.format_soil", "format_soil_data", ["-e", "tif"], "tif"),
        ("geology", "mhm_tools.pre.format_geology", "format_geology_data", [], "nc"),
        ("lai", "mhm_tools.pre.format_lai", "format_lai_data", [], "nc"),
        ("lc", "mhm_tools.pre.format_lc_data", "format_lc_data", ["-e", "asc"], "asc"),
    ],
)
def test_format_morph_categorical_dispatches_with_shared_options(
    monkeypatch, data_type, module_name, function_name, extension_args, extension
):
    """The selected formatter receives paths, lookup, extension and CRS options."""
    captured = _capture_formatter(monkeypatch, module_name, function_name)
    result = _invoke(
        "format-morph-categorical",
        "-t",
        data_type,
        "-i",
        "input.tif",
        "-d",
        "dem.asc",
        "-o",
        "output",
        *LOOKUP_ARGS,
        "-s",
        "EPSG:32632",
        "-r",
        "EPSG:32633",
        "--resampling",
        "mode",
        "--compression",
        "9",
        "--no-shuffle",
        "--significant-digits",
        "3",
        *extension_args,
    )

    assert result.exit_code == 0, result.output
    assert captured == {
        "input_file": Path("input.tif"),
        "compression": NetcdfCompression(
            complevel=9, shuffle=False, significant_digits=3
        ),
        "dem_file": Path("dem.asc"),
        "output_path": Path("output"),
        "lookup_table": Path("lookup.gpkg"),
        "mapping_field": "source",
        "class_field": "target",
        "output_type": extension,
        "input_crs": "EPSG:32632",
        "dem_crs": "EPSG:32633",
        "resampling": "mode",
        "fill_nodata": True,
    }


@pytest.mark.parametrize(
    ("command", "header", "example_row"),
    [
        (
            "format-morph-categorical",
            "StartDateTime,EndDateTime,FilePath",
            "2000-01-01T00:00:00,2005-07-01T12:00:00,landcover_2000.tif",
        ),
        (
            "format-soilgrids",
            "Horizon,Upper Depth,Lower Depth,Clay Layer,Sand Layer,Silt Layer,"
            "Bulk Density Layer,Bulk Density Unit",
            "1,0,100,clay1.tif,sand1.tif,silt1.tif,bd1.tif,kg/m3",
        ),
    ],
)
def test_format_help_documents_manifest_layout(command, header, example_row):
    """Help shows the manifest layout verbatim so a user can copy it."""
    result = _invoke(command, "--help")

    assert result.exit_code == 0, result.output
    assert "--significant-digits" in result.output
    assert "--no-shuffle" in result.output
    assert header in result.output
    assert example_row in result.output


@pytest.mark.parametrize(
    ("extra_args", "expected_resolution", "expected_resampling"),
    [
        ([], "long-term-mean-monthly", "bilinear"),
        (
            ["--output-temporal-resolution", "monthly", "--resampling", "nearest"],
            "monthly",
            "nearest",
        ),
    ],
)
def test_format_lai_netcdf_dispatches_with_defaults(
    monkeypatch, extra_args, expected_resolution, expected_resampling
):
    """Gridded LAI needs no lookup and falls back to the documented defaults."""
    captured = _capture_formatter(
        monkeypatch, "mhm_tools.pre.format_lai", "format_lai_netcdf_data"
    )
    result = _invoke(
        "format-lai-netcdf",
        "-i",
        "lai.nc",
        *GRID_ARGS,
        "-s",
        "EPSG:32737",
        "--significant-digits",
        "4",
        *extra_args,
    )

    assert result.exit_code == 0, result.output
    assert captured == {
        "input_file": Path("lai.nc"),
        "compression": NetcdfCompression(
            complevel=None, shuffle=None, significant_digits=4
        ),
        "dem_file": Path("dem.tif"),
        "output_path": Path("output"),
        "output_temporal_resolution": expected_resolution,
        "input_crs": "EPSG:32737",
        "dem_crs": None,
        "resampling": expected_resampling,
    }


def test_format_morph_categorical_dispatches_lc_manifest(monkeypatch):
    """A land-cover manifest goes to format_lc_periods with its lookup options."""
    captured = _capture_formatter(
        monkeypatch, "mhm_tools.pre.format_lc_data", "format_lc_periods"
    )
    result = _invoke(
        "format-morph-categorical",
        "-t",
        "lc",
        "-i",
        "periods.csv",
        *GRID_ARGS,
        *LOOKUP_ARGS,
        "--resampling",
        "auto",
        "--compression",
        "0",
    )

    assert result.exit_code == 0, result.output
    assert captured["input_file"] == Path("periods.csv")
    assert captured["lookup_table"] == Path("lookup.gpkg")
    assert captured["compression"] == NetcdfCompression(complevel=0, shuffle=None)
    assert captured["resampling"] == "auto"
    assert captured["fill_nodata"] is True


@pytest.mark.parametrize(
    ("step_args", "composition_step", "bulkdensity_step"),
    [
        ([], 5.0, 0.1),
        (["--composition-step", "2.5", "--bulkdensity-step", "0.05"], 2.5, 0.05),
    ],
)
def test_format_soilgrids_dispatches_class_intervals(
    monkeypatch, step_args, composition_step, bulkdensity_step
):
    """The soil horizon manifest and its class intervals reach the formatter."""
    captured = _capture_formatter(
        monkeypatch, "mhm_tools.pre.format_soil", "format_soil_horizons"
    )
    result = _invoke(
        "format-soilgrids", "-i", "soil.txt", *GRID_ARGS, "-e", "asc", *step_args
    )

    assert result.exit_code == 0, result.output
    assert captured["input_file"] == Path("soil.txt")
    assert captured["output_type"] == "asc"
    assert captured["resampling"] == "auto"
    assert captured["composition_step"] == composition_step
    assert captured["bulkdensity_step"] == bulkdensity_step
    assert captured["fill_nodata"] is True
    assert "lookup_table" not in captured


@pytest.mark.parametrize(
    ("command", "arguments", "message"),
    [
        (
            "format-soilgrids",
            ["-i", "soil.tif"],
            "CSV or TXT soil horizon manifest",
        ),
        (
            "format-morph-categorical",
            ["-t", "soil", "-i", "soil.txt", *LOOKUP_ARGS],
            "format-soilgrids",
        ),
        ("format-lai-netcdf", ["-i", "lai.tif"], "(.nc)"),
    ],
)
def test_format_rejects_input_of_other_command(command, arguments, message):
    """Each command refuses input meant for another one and says what it expects."""
    result = _invoke(command, *arguments, *GRID_ARGS)

    assert result.exit_code == 2
    assert message in result.output


def test_format_morph_categorical_rejects_directory_input(tmp_path):
    """The CLI accepts files only, not manifest-containing directories."""
    result = _invoke(
        "format-morph-categorical",
        "-t",
        "soil",
        "-i",
        str(tmp_path),
        *GRID_ARGS,
        *LOOKUP_ARGS,
    )

    assert result.exit_code == 2
    assert "not a directory" in result.output


def test_format_morph_categorical_requires_class_field():
    """The CLI never guesses a lookup class column."""
    result = _invoke(
        "format-morph-categorical",
        "-t",
        "lc",
        "-i",
        "input.tif",
        *GRID_ARGS,
        "-l",
        "lookup.gpkg",
        "-m",
        "source",
    )

    assert result.exit_code != 0
    assert "class-field" in result.output
