import argparse

import click
import numpy as np
import xarray as xr
from click.testing import CliRunner

import mhm_tools.common.file_handler as fh
from mhm_tools._cli import _file_converter, _gridded_data_evaluation
from mhm_tools._cli._main import _build_click_command


class _CompressionCommand:
    @staticmethod
    def add_args(parser: argparse.ArgumentParser):
        optional = parser.add_argument_group("optional arguments")
        optional.add_argument(
            "-x",
            "--compression",
            type=int,
            choices=range(10),
            default=9,
            help="Compression level for the NetCDF file.",
        )

    @staticmethod
    def run(args):
        click.echo(args.compression)


def test_int_choices_accept_numeric_default():
    command = _build_click_command("compression", _CompressionCommand)

    result = CliRunner().invoke(command, [])

    assert result.exit_code == 0
    assert result.output == "9\n"


def test_int_choices_accept_valid_explicit_values():
    command = _build_click_command("compression", _CompressionCommand)

    result_min = CliRunner().invoke(command, ["--compression", "0"])
    result_max = CliRunner().invoke(command, ["--compression", "9"])

    assert result_min.exit_code == 0
    assert result_min.output == "0\n"
    assert result_max.exit_code == 0
    assert result_max.output == "9\n"


def test_int_choices_reject_invalid_explicit_value():
    command = _build_click_command("compression", _CompressionCommand)

    result = CliRunner().invoke(command, ["--compression", "10"])

    assert result.exit_code != 0
    assert "Invalid value for '-x' / '--compression'" in result.output


def test_gridded_data_evaluation_parses_mask_var():
    parser = argparse.ArgumentParser()
    _gridded_data_evaluation.add_args(parser)

    args = parser.parse_args(
        [
            "--input-path",
            "input.nc",
            "--output-dir",
            "out",
            "--mask-file",
            "mask.nc",
            "--mask-var",
            "mask_l2",
        ]
    )

    assert args.mask_var == "mask_l2"


def test_converter_nc_ascii_only_header_writes_requested_new_file(
    tmp_path, monkeypatch
):
    """Write only-header converter output to a requested new header file."""
    ds = xr.Dataset(
        {"var": (("lat", "lon"), np.arange(4, dtype=np.float32).reshape(2, 2))},
        coords={"lat": [51.0, 50.0], "lon": [10.0, 11.0]},
    )
    monkeypatch.setattr(fh, "get_xarray_ds_from_file", lambda *args, **kwargs: ds)
    command = _build_click_command("converter-nc-ascii", _file_converter)
    output_path = tmp_path / "header.txt"

    result = CliRunner().invoke(
        command,
        [
            "--input-file",
            "input.nc",
            "--output-file",
            str(output_path),
            "--only-header",
        ],
    )

    assert result.exit_code == 0
    assert output_path.is_file()
