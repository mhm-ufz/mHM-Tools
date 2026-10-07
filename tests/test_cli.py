import argparse
import contextlib
import re

import click
import numpy as np
import pytest
import rioxarray  # noqa: F401
import xarray as xr
from click.testing import CliRunner

import mhm_tools.common.file_handler as fh
from mhm_tools._cli import (
    _2d_map,
    _bankfull,
    _calculate_pet,
    _create_catchment,
    _create_dem_derivatives,
    _create_idgauges,
    _create_mhm_restart_from_setup,
    _create_subdomain_masks,
    _create_wmo_region_masks,
    _crop_mhm_setup,
    _difference,
    _discharge_eval_comparison,
    _discharge_evaluation,
    _file_converter,
    _fill_nearest,
    _format_data,
    _gridded_data_evaluation,
    _hydrograph,
    _landcover_ascii_to_nc,
    _latlon,
    _link_folder_tree,
    _long_term_mean,
    _merge,
    _metric_plots,
    _mhm_run_overview,
    _prepare_mhm_forcings,
    _ratio,
    _regrid,
    _relative_difference,
    _snow_evaluation,
    _taylor_diagram,
    _twsa_evaluation,
)
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


def test_converter_forwards_explicit_crs_to_geotiff_writer(monkeypatch):
    """The CLI exposes CRS assignment for otherwise CRS-less inputs."""
    ds = xr.Dataset(
        {"var": (("lat", "lon"), np.ones((2, 2), dtype=np.int32))},
        coords={"lat": [1.5, 0.5], "lon": [0.5, 1.5]},
    )
    captured = {}
    monkeypatch.setattr(fh, "get_xarray_ds_from_file", lambda *_args, **_kwargs: ds)

    def fake_write(*args, **kwargs):
        captured.update(kwargs)

    monkeypatch.setattr(fh, "write_xarray_to_file", fake_write)
    command = _build_click_command("converter-nc-ascii", _file_converter)
    result = CliRunner().invoke(
        command,
        ["-i", "input.asc", "-o", "output.tif", "-c", "EPSG:32632"],
    )

    assert result.exit_code == 0, result.output
    assert captured["crs"] == "EPSG:32632"


def test_converter_preserves_netcdf_crs_in_geotiff(tmp_path):
    """CF grid-mapping metadata is available during NetCDF conversion."""
    data = xr.DataArray(
        np.arange(4, dtype=np.int32).reshape(2, 2),
        dims=("y", "x"),
        coords={"y": [1.5, 0.5], "x": [0.5, 1.5]},
        name="classes",
    ).rio.write_crs("EPSG:32632")
    input_file = tmp_path / "classes.nc"
    output_file = tmp_path / "classes.tif"
    fh.write_xarray_to_file(data, input_file)
    command = _build_click_command("converter-nc-ascii", _file_converter)

    result = CliRunner().invoke(
        command,
        ["-i", str(input_file), "-o", str(output_file)],
    )

    assert result.exit_code == 0, result.output
    converted = fh.get_raster_data(output_file)
    try:
        assert converted.rio.crs.to_epsg() == 32632
        np.testing.assert_array_equal(converted.values, data.values)
    finally:
        converted.close()


# the same renames in difference, ratio and relative-difference
MAP_COMPARISON_OPTIONS = (
    ("--ref-dir", ("--ref-input-dir",), "ref_input_dir"),
    ("--input-dir", ("--mod-input-dir",), "mod_input_dir"),
    ("--ref-file-name", ("--reference-pattern",), "reference_pattern"),
    ("--input-file-name", ("--model-pattern",), "model_pattern"),
    ("--input-var", ("--mod-var",), "mod_var"),
)

# renamed options: module, new name shown in the help, former names kept as
# hidden aliases, destination
RENAMED_OPTIONS = (
    (_2d_map, "--input-var", ("--var",), "var"),
    (_bankfull, "--input-var", ("--var",), "var"),
    (_calculate_pet, "--resample-time-to", ("--freq",), "freq"),
    (_create_catchment, "--output-dir", ("--output-path",), "output_path"),
    (_create_catchment, "--input-var", ("--vn", "--varname"), "vn"),
    (_create_catchment, "--input-type", ("--var",), "var"),
    (_create_catchment, "--ftype", ("--ftp",), "ftp"),
    (_create_catchment, "--shape-dir", ("--shape-folder",), "shape_folder"),
    (
        _create_catchment,
        "--id-gauges-output-dir",
        ("--id-gauges-out-path",),
        "id_gauges_out_path",
    ),
    (_create_dem_derivatives, "--input-var", ("--varname",), "varname"),
    (_create_idgauges, "--output-file", ("--output",), "out_file"),
    (
        _create_mhm_restart_from_setup,
        "--input-file-name",
        ("--input-name", "--file-name"),
        "file_name",
    ),
    (_create_subdomain_masks, "--mask-file", ("--land-mask",), "land_mask"),
    (
        _create_subdomain_masks,
        "--mask-var",
        ("--land-mask-variable", "--land_mask_var"),
        "land_mask_variable",
    ),
    (_create_wmo_region_masks, "--input-var", ("--vn", "--varname"), "vn"),
    (_create_wmo_region_masks, "--input-type", ("--var",), "var"),
    (
        _crop_mhm_setup,
        "--input-file-name",
        ("--input-name", "--file-name"),
        "file_name",
    ),
    *(
        (module, new_option, former_options, dest)
        for module in (_difference, _ratio, _relative_difference)
        for new_option, former_options, dest in MAP_COMPARISON_OPTIONS
    ),
    (
        _discharge_eval_comparison,
        "--input-file-name",
        ("--file-names", "--file-pattern"),
        "file_names",
    ),
    (_discharge_eval_comparison, "--shape-dir", ("--shape-folder",), "shape_folder"),
    (_discharge_eval_comparison, "--mask-dir", ("--mask-folder",), "mask_folder"),
    (_discharge_eval_comparison, "--ref-name", ("--reference-name",), "reference_name"),
    (_discharge_evaluation, "--input-path", ("--model-data-path",), "input_path"),
    (_discharge_evaluation, "--ref-path", ("--observed-data-path",), "ref_path"),
    (
        _discharge_evaluation,
        "--input-file-name",
        ("--model-file-name",),
        "input_file_name",
    ),
    (
        _discharge_evaluation,
        "--input-var",
        ("--input-variable", "--model-variable"),
        "input_variable",
    ),
    (
        _discharge_evaluation,
        "--ref-var",
        ("--ref-variable", "--observed-variable"),
        "ref_variable",
    ),
    (_discharge_evaluation, "--facc-var", ("--facc-variable",), "facc_variable"),
    (_discharge_evaluation, "--shape-dir", ("--shape-folder",), "shape_folder"),
    (_discharge_evaluation, "--mask-dir", ("--mask-folder",), "mask_folder"),
    (
        _discharge_evaluation,
        "--n-bootstrap-years",
        ("--n-boostrap-years",),
        "n_boostrap_years",
    ),
    (
        _discharge_evaluation,
        "--gauge-optimization-method",
        ("--gauge-location-method",),
        "gauge_location_method",
    ),
    (_file_converter, "--input-var", ("--varname",), "varname"),
    (_fill_nearest, "--input-file-name", ("--fname", "--input-name"), "fname"),
    (_format_data, "--output-dir", ("--output-path",), "output_path"),
    (_format_data, "--output-extension", ("--extension",), "extension"),
    (_gridded_data_evaluation, "--input-var", ("--input-variable",), "input_variable"),
    (_gridded_data_evaluation, "--ref-var", ("--ref-variable",), "ref_variable"),
    (
        _gridded_data_evaluation,
        "--n-bootstrap-years",
        ("--n-boostrap-years",),
        "n_boostrap_years",
    ),
    (_gridded_data_evaluation, "--metric", ("--metrics",), "metric"),
    (_hydrograph, "--precipitation-file", ("--prec",), "prec"),
    (_hydrograph, "--input-name", ("--input-names", "--name"), "sim_names"),
    (_landcover_ascii_to_nc, "--output-file", ("--output",), "output"),
    (_landcover_ascii_to_nc, "--output-var", ("--varname",), "varname"),
    (_latlon, "--write-header-l0", ("--h0",), "h0"),
    (_latlon, "--write-header-l1", ("--h1",), "h1"),
    (_latlon, "--write-header-l11", ("--h11",), "h11"),
    (_latlon, "--write-header-l2", ("--h2",), "h2"),
    (_latlon, "--output-file", ("--out-file",), "out_file"),
    (_link_folder_tree, "--input-file-name", ("--file-name",), "file_name"),
    (
        _long_term_mean,
        "--input-file-name",
        ("--input-name", "--in-file"),
        "in_file",
    ),
    (
        _long_term_mean,
        "--output-file-name",
        ("--output-name", "--out-file"),
        "out_file",
    ),
    (_merge, "--input-file-name", ("--input-name",), "input_name"),
    (
        _metric_plots,
        "--input-file-name",
        ("--file-names", "--file-pattern"),
        "file_names",
    ),
    (_metric_plots, "--shape-dir", ("--shape-folder",), "shape_folder"),
    (_metric_plots, "--mask-dir", ("--mask-folder",), "mask_folder"),
    (_mhm_run_overview, "--base-dir", ("--base-path",), "base_path"),
    (
        _prepare_mhm_forcings,
        "--input-file-name",
        ("--input-name", "--in-file"),
        "in_file",
    ),
    (_prepare_mhm_forcings, "--input-var", ("--var",), "var"),
    (
        _prepare_mhm_forcings,
        "--output-file-name",
        ("--output-name", "--out-file"),
        "out_file",
    ),
    (_prepare_mhm_forcings, "--output-var", ("--out-var",), "out_var"),
    (
        _prepare_mhm_forcings,
        "--resample-time-to",
        ("--target-frequency",),
        "target_frequency",
    ),
    (_regrid, "--l2-resolution", ("--l2",), "l2"),
    (_snow_evaluation, "--input-var", ("--input-variable",), "input_variable"),
    (_snow_evaluation, "--ref-var", ("--ref-variable",), "ref_variable"),
    (
        _snow_evaluation,
        "--resample-time-to",
        ("--target-frequency",),
        "target_frequency",
    ),
    (_snow_evaluation, "--available-mem", ("--max-memory-gib",), "max_memory_gib"),
    (_taylor_diagram, "--ref-dir", ("--ref-input-dir",), "ref_input_dir"),
    (
        _taylor_diagram,
        "--ref-file-name",
        ("--reference-pattern",),
        "reference_pattern",
    ),
    (
        _taylor_diagram,
        "--input-dir",
        ("--input-dirs", "--mod-input-dirs"),
        "mod_input_dirs",
    ),
    (
        _taylor_diagram,
        "--input-file-name",
        ("--input-file-names", "--model-patterns"),
        "model_patterns",
    ),
    (_taylor_diagram, "--input-var", ("--input-vars", "--mod-vars"), "mod_vars"),
    (_taylor_diagram, "--ref-name", ("--ref-label",), "ref_label"),
    (
        _taylor_diagram,
        "--input-name",
        ("--input-names", "--mod-labels"),
        "mod_labels",
    ),
    (_twsa_evaluation, "--input-var", ("--input-variable",), "input_variable"),
    (_twsa_evaluation, "--ref-var", ("--ref-variable",), "ref_variable"),
    (_twsa_evaluation, "--input-factor", ("--input-scale",), "input_scale"),
    (_twsa_evaluation, "--ref-factor", ("--ref-scale",), "ref_scale"),
    (_twsa_evaluation, "--metric", ("--metrics",), "metrics"),
    (_twsa_evaluation, "--available-mem", ("--max-memory-gib",), "max_memory_gib"),
)
RENAMED_OPTION_IDS = [
    f"{module.__name__.rsplit('.', 1)[-1]}{new_option}"
    for module, new_option, _former_options, _dest in RENAMED_OPTIONS
]


def is_option_listed(text, option):
    """Tell whether an option name appears in a text as a whole option.

    Args:
        text (str): Text to search, such as the output of --help.
        option (str): Option name, such as '--var'.

    Returns:
        True if the option appears, not counting it as the start of a longer
        option such as '--varname-eq-in-filename'.
    """
    return re.search(rf"(?<![\w-]){re.escape(option)}(?![\w-])", text) is not None


@pytest.mark.parametrize(
    ("module", "new_option", "former_options", "dest"),
    RENAMED_OPTIONS,
    ids=RENAMED_OPTION_IDS,
)
def test_renamed_option_keeps_its_former_names(
    module, new_option, former_options, dest
):
    """Store the new and every former option name into the same destination."""
    parser = argparse.ArgumentParser()
    module.add_args(parser)
    command = _build_click_command("renamed-options", module)

    new_action = parser._option_string_actions[new_option]
    assert new_action.dest == dest
    for former_option in former_options:
        assert parser._option_string_actions[former_option] is new_action
        assert command.option_aliases[former_option] == new_option


@pytest.mark.parametrize(
    ("module", "new_option", "former_options", "dest"),
    RENAMED_OPTIONS,
    ids=RENAMED_OPTION_IDS,
)
def test_help_shows_only_the_new_option_name(
    module,
    new_option,
    former_options,
    dest,  # noqa: ARG001
):
    """List the new option name in the help and none of its former names."""
    command = _build_click_command("renamed-options", module)

    result = CliRunner().invoke(command, ["--help"])

    assert result.exit_code == 0, result.output
    assert is_option_listed(result.output, new_option)
    for former_option in former_options:
        assert not is_option_listed(result.output, former_option)


def test_discharge_evaluation_run_passes_paths_and_variables(monkeypatch):
    """Hand the data options to evaludate_discharge_data under its own names."""
    received_calls = []

    def record_evaluation(*args, **kwargs):
        """Record the arguments the evaluation is called with."""
        received_calls.append((args, kwargs))

    monkeypatch.setattr(
        "mhm_tools.post.discharge_evaluation.evaludate_discharge_data",
        record_evaluation,
    )
    parser = argparse.ArgumentParser()
    _discharge_evaluation.add_args(parser)

    _discharge_evaluation.run(
        parser.parse_args(
            [
                "--output-dir",
                "out",
                "--input-path",
                "sim.nc",
                "--ref-path",
                "obs.nc",
                "--input-file-name",
                "discharge.nc",
                "--input-var",
                "Qsim",
                "--ref-var",
                "Qobs",
            ]
        )
    )

    positional_args, keyword_args = received_calls[0]
    assert positional_args == ("sim.nc", "obs.nc")
    assert keyword_args["model_file_name"] == "discharge.nc"
    assert keyword_args["sim_variable"] == "Qsim"
    assert keyword_args["observed_variable"] == "Qobs"


@pytest.mark.parametrize(
    ("memory_args", "expected_memory_gb"),
    [
        ([], 8.0),
        (["--available-mem", "500mb"], 0.5),
        (["--max-memory-gib", "8"], 8.0),
    ],
)
@pytest.mark.parametrize(
    ("module", "evaluation_path"),
    [
        (_snow_evaluation, "mhm_tools.post.snow_evaluation.snow_evaluation"),
        (_twsa_evaluation, "mhm_tools.post.twsa_evaluation.twsa_evaluation"),
    ],
)
def test_available_mem_reaches_the_evaluation_in_gb(
    monkeypatch, module, evaluation_path, memory_args, expected_memory_gb
):
    """Hand the memory budget of either option name on in Gb, 500mb as 0.5."""
    received_arguments = {}

    def record_evaluation(**kwargs):
        """Record the arguments the evaluation is called with.

        Returns:
            An empty dict of written files.
        """
        received_arguments.update(kwargs)
        return {}

    monkeypatch.setattr(evaluation_path, record_evaluation)
    # the twsa evaluation would otherwise start a dask cluster
    monkeypatch.setattr(
        "mhm_tools.common.parallel.create_dask_cluster",
        lambda _ncpus: contextlib.nullcontext(),
    )
    parser = argparse.ArgumentParser()
    module.add_args(parser)

    module.run(
        parser.parse_args(
            ["--input-path", "in", "--ref-path", "ref", "--output-dir", "out"]
            + memory_args
        )
    )

    assert received_arguments["max_memory_gib"] == pytest.approx(expected_memory_gb)
