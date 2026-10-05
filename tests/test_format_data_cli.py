"""Tests for the consolidated categorical-data formatter CLI."""

import importlib
from pathlib import Path

import pytest
from click.testing import CliRunner

from mhm_tools._cli._main import cli
from mhm_tools.common.netcdf import NetcdfCompression


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
            "lai",
            "mhm_tools.pre.format_lai",
            "format_lai_data",
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
            "--resampling",
            "mode",
            "--compression",
            "9",
            "--no-shuffle",
            "--significant-digits",
            "3",
            *extension_args,
        ],
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


def test_format_data_help_documents_manifest_formats():
    """Help shows both manifest layouts so a user can prepare one."""
    result = CliRunner().invoke(cli, ["data-converter", "format-data", "--help"])

    assert result.exit_code == 0, result.output
    assert "--significant-digits" in result.output
    assert "--no-shuffle" in result.output
    # The land-cover and soil headers appear verbatim, so they can be copied.
    assert "StartDateTime,EndDateTime,FilePath" in result.output
    assert (
        "Horizon,Upper Depth,Lower Depth,Clay Layer,Sand Layer,Silt Layer,"
        "Bulk Density Layer,Bulk Density Unit" in result.output
    )
    assert "2000-01-01T00:00:00,2005-07-01T12:00:00,landcover_2000.tif" in result.output
    assert "1,0,100,clay1.tif,sand1.tif,silt1.tif,bd1.tif,kg/m3" in result.output


@pytest.mark.parametrize(
    ("temporal_args", "expected_resolution"),
    [
        ([], "long-term-mean-monthly"),
        (["--output-temporal-resolution", "monthly"], "monthly"),
    ],
)
def test_format_data_dispatches_gridded_lai_without_lookup(
    monkeypatch, temporal_args, expected_resolution
):
    """A gridded LAI NetCDF needs no lookup or input cadence option."""
    captured = {}

    def fake_formatter(**kwargs):
        captured.update(kwargs)

    module = importlib.import_module("mhm_tools.pre.format_lai")
    monkeypatch.setattr(module, "format_lai_netcdf_data", fake_formatter)
    arguments = [
        "data-converter",
        "format-data",
        "-t",
        "lai",
        "-i",
        "lai.nc",
        "-d",
        "dem.tif",
        "-o",
        "output",
        "--resampling",
        "bilinear",
    ]
    result = CliRunner().invoke(
        cli, [*arguments, *temporal_args, "--significant-digits", "4"]
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
        "dem_crs": None,
        "resampling": "bilinear",
    }


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
def test_format_data_dispatches_manifest_files(
    tmp_path,
    monkeypatch,
    data_type,
    module_name,
    function_name,
    lookup_options,
):
    """A directly supplied manifest selects the corresponding public API."""
    suffix = ".txt" if data_type == "soil" else ".csv"
    input_file = tmp_path / f"custom-{data_type}-inputs{suffix}"
    input_file.touch()
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
            "--input-file",
            str(input_file),
            "-d",
            "dem.tif",
            "-o",
            "output",
            "--resampling",
            "auto",
            "--compression",
            "0",
            *lookup_options,
        ],
    )

    assert result.exit_code == 0, result.output
    assert captured["compression"] == NetcdfCompression(complevel=0, shuffle=None)
    assert captured["input_file"] == input_file
    assert captured["resampling"] == "auto"
    assert captured["fill_nodata"] is True
    if data_type == "soil":
        assert "lookup_table" not in captured
        assert "composition_step" not in captured
        assert "bulkdensity_step" not in captured


def test_format_data_dispatches_soil_manifest_class_intervals(tmp_path, monkeypatch):
    """Soil manifest class intervals are forwarded to the formatter."""
    input_file = tmp_path / "soil.txt"
    input_file.touch()
    captured = {}

    def fake_formatter(**kwargs):
        """Capture formatter keyword arguments."""
        captured.update(kwargs)

    module = importlib.import_module("mhm_tools.pre.format_soil")
    monkeypatch.setattr(module, "format_soil_horizons", fake_formatter)
    result = CliRunner().invoke(
        cli,
        [
            "data-converter",
            "format-data",
            "-t",
            "soil",
            "-i",
            str(input_file),
            "-d",
            "dem.tif",
            "-o",
            "output",
            "--composition-step",
            "2.5",
            "--bulkdensity-step",
            "0.05",
        ],
    )

    assert result.exit_code == 0, result.output
    assert captured["composition_step"] == 2.5
    assert captured["bulkdensity_step"] == 0.05


def test_format_data_rejects_soil_class_intervals_for_raster_input():
    """Soil class intervals apply only to multi-horizon manifest input."""
    result = CliRunner().invoke(
        cli,
        [
            "data-converter",
            "format-data",
            "-t",
            "soil",
            "-i",
            "soil.tif",
            "-d",
            "dem.tif",
            "-o",
            "output",
            "--composition-step",
            "2.5",
        ],
    )

    assert result.exit_code == 2
    assert "only valid for soil manifest inputs" in result.output


def test_format_data_rejects_directory_input(tmp_path):
    """The CLI accepts files only, not manifest-containing directories."""
    input_directory = tmp_path / "inputs"
    input_directory.mkdir()

    result = CliRunner().invoke(
        cli,
        [
            "data-converter",
            "format-data",
            "-t",
            "soil",
            "--input-file",
            str(input_directory),
            "-d",
            "dem.tif",
            "-o",
            "output",
        ],
    )

    assert result.exit_code == 2
    assert "not a directory" in result.output


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
