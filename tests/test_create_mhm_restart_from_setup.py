import argparse
from pathlib import Path

import numpy as np
import pytest
import xarray as xr

from mhm_tools._cli._create_mhm_restart_from_setup import (
    _as_list,
    _parse_fill_nearest_files,
    add_args,
    run,
)
from mhm_tools.common.resolution_handler import Resolution
from mhm_tools.pre import MHMRunner
from mhm_tools.pre.create_mhm_restart_from_setup import (
    MHMSetupTile,
    _collect_restart_files_for_tiles,
    _merge_restart_files,
    _move_restart_files,
    _restore_recreated_fill_files_from_original,
    _tile_has_active_mask_cell,
    _validate_prepared_tile_dirs,
    create_mhm_restart_from_setup,
    create_setup_tiles,
    get_crop_slices,
    merge_mhm_restart_files,
)
from mhm_tools.pre.crop_mhm_setup import LatlonFiles, crop_mhm_setup


def test_get_crop_slices_decreasing_latitude():
    lonslice, latslice = get_crop_slices(1.0, 2.0, 3.0, 4.0)

    assert (lonslice.start, lonslice.stop) == (1.0, 2.0)
    assert (latslice.start, latslice.stop) == (4.0, 3.0)


def test_create_setup_tiles_snaps_float_roundoff_bounds(tmp_path):
    resolution = 4.000000000008071 / 960

    tiles = create_setup_tiles(
        lon_min_bound=4.0,
        lon_max_bound=8.0,
        lat_min_bound=-60.0,
        lat_max_bound=-52.0,
        l1_resolution=resolution,
        l1_increment=960,
        output_path=tmp_path,
    )

    assert tiles[0].latslice.stop == -60.0
    assert tiles[1].latslice.stop == -56.0


def test_parse_fill_nearest_files_ignores_empty_whitespace_fields():
    assert _parse_fill_nearest_files(
        "soil_class_horizon_01.nc  soil_class_horizon_02.nc\tsoil_class_horizon_03.nc"
    ) == [
        "soil_class_horizon_01.nc",
        "soil_class_horizon_02.nc",
        "soil_class_horizon_03.nc",
    ]
    assert _parse_fill_nearest_files(None) is None


def test_create_mhm_restart_from_setup_cli_parses_mhm_ncpus():
    parser = argparse.ArgumentParser()
    add_args(parser)

    args = parser.parse_args(
        [
            "--input-dir",
            "input",
            "--output-dir",
            "output",
            "--mask-file",
            "mask.nc",
            "--l1-resolution",
            "1.0",
            "--n-cpus",
            "8",
            "--mhm-ncpus",
            "2",
        ]
    )

    assert args.n_cpus == 8
    assert args.mhm_ncpus == 2
    assert args.mask_file == ["mask.nc"]
    assert not args.skip_tile_creation
    assert not args.skip_mhm_run


def test_create_mhm_restart_from_setup_cli_parses_no_tile_creation():
    parser = argparse.ArgumentParser()
    add_args(parser)

    args = parser.parse_args(
        [
            "--input-dir",
            "input",
            "--output-dir",
            "output",
            "--mask-file",
            "mask.nc",
            "--l1-resolution",
            "1.0",
            "--no-tile-creation",
        ]
    )

    assert args.skip_tile_creation


def test_create_mhm_restart_from_setup_cli_parses_no_mhm_run():
    parser = argparse.ArgumentParser()
    add_args(parser)

    args = parser.parse_args(
        [
            "--input-dir",
            "input",
            "--output-dir",
            "output",
            "--mask-file",
            "mask.nc",
            "--l1-resolution",
            "1.0",
            "--no-mhm-run",
        ]
    )

    assert args.skip_mhm_run


def test_create_mhm_restart_from_setup_cli_parses_recreate_restart():
    parser = argparse.ArgumentParser()
    add_args(parser)

    args = parser.parse_args(
        [
            "--input-dir",
            "input",
            "--output-dir",
            "output",
            "--mask-file",
            "mask.nc",
            "--l1-resolution",
            "1.0",
            "--recreate-restart",
        ]
    )

    assert args.recreate_restart


def test_create_mhm_restart_from_setup_cli_parses_multiple_masks_and_parameters():
    parser = argparse.ArgumentParser()
    add_args(parser)

    args = parser.parse_args(
        [
            "--input-dir",
            "input",
            "--output-dir",
            "output",
            "--mask-file",
            "mask_a.nc",
            "mask_b.nc",
            "--l1-resolution",
            "1.0",
            "--parameter-file",
            "param_a.nml",
            "param_b.nml",
        ]
    )

    assert args.mask_file == ["mask_a.nc", "mask_b.nc"]
    assert args.parameter_files == ["param_a.nml", "param_b.nml"]


def test_create_mhm_restart_from_setup_cli_splits_quoted_multi_values():
    parser = argparse.ArgumentParser()
    add_args(parser)

    args = parser.parse_args(
        [
            "--input-dir",
            "input",
            "--output-dir",
            "output",
            "--mask-file",
            "mask_a.nc mask_b.nc",
            "--l1-resolution",
            "1.0",
            "--parameter-file",
            "param_a.nml param_b.nml",
            "--fill_nearest_file",
            "soil_1.nc soil_2.nc",
            "--n_cpus",
            "8",
            "--mhm-ncpus",
            "2",
        ]
    )

    assert args.mask_file == ["mask_a.nc mask_b.nc"]
    assert args.parameter_files == ["param_a.nml param_b.nml"]
    assert args.fill_nearest_files == "soil_1.nc soil_2.nc"
    assert _as_list(args.mask_file) == ["mask_a.nc", "mask_b.nc"]
    assert _as_list(args.parameter_files) == ["param_a.nml", "param_b.nml"]
    assert args.n_cpus == 8
    assert args.mhm_ncpus == 2


def test_create_mhm_restart_from_setup_cli_loops_masks_and_parameters(
    tmp_path, monkeypatch
):
    calls = []
    merge_calls = []

    def fake_get_coords(mask_file, mask_var):  # noqa: ARG001
        if mask_file == "mask_a.nc":
            return 0.0, 1.0, 0.0, 1.0, None
        return 1.0, 2.0, 0.0, 1.0, None

    def fake_get_xarray_ds_from_file(mask_file):
        if mask_file == "mask_a.nc":
            mask = np.array([[1.0, 0.0]])
        else:
            mask = np.array([[0.0, 1.0]])
        return xr.Dataset(
            data_vars={"mask": (("lat", "lon"), mask)},
            coords={"lat": np.array([0.5]), "lon": np.array([0.5, 1.5])},
            attrs={"mask_file": mask_file},
        )

    def fake_create_mhm_restart_from_setup(**kwargs):
        calls.append(kwargs)
        output_path = Path(kwargs["output_path"])
        return {
            "restart_files": [output_path / "slice_0_0" / "output" / "restart.nc"],
            "merged_restart_file": kwargs["merged_restart_file"],
            "tiles": [],
        }

    def fake_merge_restart_files(*args, **kwargs):
        merge_calls.append((args, kwargs))
        return xr.Dataset(attrs={})

    monkeypatch.setattr("mhm_tools.common.cli_utils.get_coords", fake_get_coords)
    monkeypatch.setattr(
        "mhm_tools.common.file_handler.get_xarray_ds_from_file",
        fake_get_xarray_ds_from_file,
    )
    monkeypatch.setattr(
        "mhm_tools.pre.create_mhm_restart_from_setup.create_mhm_restart_from_setup",
        fake_create_mhm_restart_from_setup,
    )
    monkeypatch.setattr(
        "mhm_tools.pre.create_mhm_restart_from_setup.merge_restart_files",
        fake_merge_restart_files,
    )

    args = argparse.Namespace(
        input_path="input",
        output_path=tmp_path / "out",
        mask_file=["mask_a.nc mask_b.nc"],
        l1_resolution=1.0,
        l1_increment=1,
        l11_resolution=None,
        crs=None,
        n_cpus=1,
        mhm_ncpus=None,
        crop_ncpus=1,
        file_name="*.*",
        available_mem="5",
        create_header=True,
        chunking=False,
        output_var=None,
        no_cropping=False,
        skip_tile_creation=False,
        skip_mhm_run=False,
        lat_order="decreasing",
        output_suffix=None,
        mask_all=False,
        mask_var="mask",
        mhm_packages=None,
        mhm_args="--quiet",
        parameter_files=["param_a.nml param_b.nml"],
        restart_pattern="**/mHM_restart*.nc",
        restart_output_dir=tmp_path / "restart",
        no_restart_check=False,
        recreate_restart=False,
        no_merge=False,
        merged_restart_file=tmp_path / "final.nc",
        fill_nearest_files=None,
        l0_mask_files=None,
    )

    run(args)

    assert len(calls) == 2
    assert calls[0]["output_path"] == tmp_path / "out" / "000_mask_a"
    assert calls[1]["output_path"] == tmp_path / "out" / "001_mask_b"
    assert calls[0]["restart_output_path"] == tmp_path / "restart" / "000_mask_a"
    assert calls[1]["restart_output_path"] == tmp_path / "restart" / "001_mask_b"
    assert calls[0]["skip_mhm_run"] is False
    assert calls[0]["merge"] is False
    assert calls[1]["merge"] is False
    assert calls[0]["merged_restart_file"] is None
    assert calls[1]["merged_restart_file"] is None
    assert calls[0]["mhm_args"] == "--quiet -p param_a.nml"
    assert calls[1]["mhm_args"] == "--quiet -p param_b.nml"
    assert len(merge_calls) == 1
    final_args, final_kwargs = merge_calls[0]
    assert final_args == ()
    assert final_kwargs["restart_file_paths"] == [
        tmp_path / "out" / "000_mask_a" / "slice_0_0" / "output" / "restart.nc",
        tmp_path / "out" / "001_mask_b" / "slice_0_0" / "output" / "restart.nc",
    ]
    assert final_kwargs["output_file"] == tmp_path / "final.nc"
    assert final_kwargs["lon_min"] == 0.0
    assert final_kwargs["lon_max"] == 2.0
    assert final_kwargs["mask_var"] == "mask"
    assert len(final_kwargs["mask_ds"]) == 2
    assert final_kwargs["mask_ds"][0]["mask"].values.tolist() == [[1.0, 0.0]]
    assert final_kwargs["mask_ds"][1]["mask"].values.tolist() == [[0.0, 1.0]]


def test_create_setup_tiles_uses_l1_increment():
    tiles = create_setup_tiles(
        lon_min_bound=0.0,
        lon_max_bound=2.0,
        lat_min_bound=0.0,
        lat_max_bound=2.0,
        l1_resolution=0.5,
        l1_increment=2,
        output_path="setup_tiles",
    )

    assert [tile.name for tile in tiles] == [
        "slice_0_0",
        "slice_0_1",
        "slice_1_0",
        "slice_1_1",
    ]
    assert (tiles[0].lonslice.start, tiles[0].lonslice.stop) == (0.0, 1.0)
    assert (tiles[0].latslice.start, tiles[0].latslice.stop) == (1.0, 0.0)
    assert (tiles[0].lon_min, tiles[0].lon_max) == (0.25, 0.75)
    assert (tiles[0].lat_min, tiles[0].lat_max) == (0.25, 0.75)


def test_create_mhm_restart_from_setup_crops_runs_tiles_and_finds_restarts(
    tmp_path, monkeypatch
):
    input_path = tmp_path / "input_setup"
    output_path = tmp_path / "cropped_tiles"
    input_path.mkdir()
    calls = []

    def fake_crop_mhm_setup(**kwargs):
        calls.append(kwargs)
        Path(kwargs["output_path"]).mkdir(parents=True, exist_ok=True)

    def fake_run_mhm(self, setup_path):  # noqa: ARG001
        restart_dir = Path(setup_path) / "output"
        restart_dir.mkdir(parents=True, exist_ok=True)
        (restart_dir / "mHM_restart_001.nc").write_text("restart")

    monkeypatch.setattr(
        "mhm_tools.pre.create_mhm_restart_from_setup.crop_mhm_setup",
        fake_crop_mhm_setup,
    )
    monkeypatch.setattr(MHMRunner, "run_mhm", fake_run_mhm)

    restart_files = create_mhm_restart_from_setup(
        input_path=input_path,
        output_path=output_path,
        mask_da=None,
        lon_min=0.0,
        lon_max=2.0,
        lat_min=0.0,
        lat_max=2.0,
        l1_resolution=1.0,
        l1_increment=1,
        merge=False,
    )

    assert len(calls) == 4
    assert calls[0]["input_path"] == input_path
    assert calls[0]["output_path"] == output_path / "slice_0_0"
    assert calls[0]["mask_ds"] is None
    assert "mask_da" not in calls[0]
    assert restart_files["restart_files"] == [
        output_path / "slice_0_0" / "output" / "mHM_restart_001.nc",
        output_path / "slice_0_1" / "output" / "mHM_restart_001.nc",
        output_path / "slice_1_0" / "output" / "mHM_restart_001.nc",
        output_path / "slice_1_1" / "output" / "mHM_restart_001.nc",
    ]
    for tile in ("slice_0_0", "slice_0_1", "slice_1_0", "slice_1_1"):
        assert (output_path / tile / "output").is_dir()
        assert (output_path / tile / "restart").is_dir()


def test_create_mhm_restart_from_setup_uses_status_for_merge_and_reports_failures(
    tmp_path, monkeypatch, capsys
):
    output_path = tmp_path / "cropped_tiles"
    merge_calls = []

    def fake_prepare_tiles_for_mhm(**kwargs):
        return kwargs["tiles"]

    def fake_run_mhm_for_tiles(**kwargs):
        tiles = kwargs["prepared_tiles"]
        return [
            {
                "tile": tiles[0].name,
                "restart_files": [
                    tiles[0].output_path / "output" / "mHM_restart_001.nc"
                ],
                "status": 0,
                "message": "",
            },
            {
                "tile": tiles[1].name,
                "restart_files": [
                    tiles[1].output_path / "output" / "mHM_restart_001.nc"
                ],
                "status": 1,
                "message": "mHM failed",
            },
        ]

    def fake_merge_restart_files(**kwargs):
        merge_calls.append(kwargs)
        return xr.Dataset(attrs={})

    monkeypatch.setattr(
        "mhm_tools.pre.create_mhm_restart_from_setup._prepare_tiles_for_mhm",
        fake_prepare_tiles_for_mhm,
    )
    monkeypatch.setattr(
        "mhm_tools.pre.create_mhm_restart_from_setup._run_mhm_for_tiles",
        fake_run_mhm_for_tiles,
    )
    monkeypatch.setattr(
        "mhm_tools.pre.create_mhm_restart_from_setup.merge_restart_files",
        fake_merge_restart_files,
    )

    result = create_mhm_restart_from_setup(
        input_path=tmp_path / "input_setup",
        output_path=output_path,
        mask_da=None,
        lon_min=0.0,
        lon_max=2.0,
        lat_min=0.0,
        lat_max=1.0,
        l1_resolution=1.0,
        l1_increment=1,
        merge=True,
    )

    successful_restart = output_path / "slice_0_0" / "output" / "mHM_restart_001.nc"
    failed_restart = output_path / "slice_1_0" / "output" / "mHM_restart_001.nc"
    assert result["restart_files"] == [successful_restart]
    assert failed_restart not in merge_calls[0]["restart_file_paths"]
    assert merge_calls[0]["restart_file_paths"] == [successful_restart]
    assert result["failed_tiles"] == [{"tile": "slice_1_0", "message": "mHM failed"}]
    assert result["failed_tiles_file"] == output_path / "failed_mhm_tiles.txt"
    assert "slice_1_0: mHM failed" in capsys.readouterr().out
    assert result["failed_tiles_file"].read_text() == (
        "Failed mHM tiles:\nslice_1_0: mHM failed\n"
    )


def test_create_mhm_restart_from_setup_prepares_all_tiles_before_mhm_runs(
    tmp_path, monkeypatch
):
    input_path = tmp_path / "input_setup"
    output_path = tmp_path / "cropped_tiles"
    input_path.mkdir()
    crop_calls = []
    run_calls = []

    def fake_crop_mhm_setup(**kwargs):
        crop_calls.append(Path(kwargs["output_path"]).name)
        Path(kwargs["output_path"]).mkdir(parents=True, exist_ok=True)

    def fake_run_mhm(self, setup_path):  # noqa: ARG001
        assert crop_calls == ["slice_0_0", "slice_0_1", "slice_1_0", "slice_1_1"]
        run_calls.append(Path(setup_path).name)
        restart_dir = Path(setup_path) / "output"
        restart_dir.mkdir(parents=True, exist_ok=True)
        (restart_dir / "mHM_restart_001.nc").write_text("restart")

    monkeypatch.setattr(
        "mhm_tools.pre.create_mhm_restart_from_setup.crop_mhm_setup",
        fake_crop_mhm_setup,
    )
    monkeypatch.setattr(MHMRunner, "run_mhm", fake_run_mhm)

    create_mhm_restart_from_setup(
        input_path=input_path,
        output_path=output_path,
        mask_da=None,
        lon_min=0.0,
        lon_max=2.0,
        lat_min=0.0,
        lat_max=2.0,
        l1_resolution=1.0,
        l1_increment=1,
        merge=False,
    )

    assert run_calls == ["slice_0_0", "slice_0_1", "slice_1_0", "slice_1_1"]


def test_create_mhm_restart_from_setup_can_skip_tile_creation(tmp_path, monkeypatch):
    input_path = tmp_path / "input_setup"
    output_path = tmp_path / "cropped_tiles"
    input_path.mkdir()
    run_calls = []

    for tile in ("slice_0_0", "slice_0_1", "slice_1_0", "slice_1_1"):
        (output_path / tile).mkdir(parents=True)

    def fail_crop_mhm_setup(**kwargs):  # noqa: ARG001
        raise AssertionError("crop_mhm_setup should not be called")

    def fake_run_mhm(self, setup_path):  # noqa: ARG001
        run_calls.append(Path(setup_path).name)
        restart_dir = Path(setup_path) / "output"
        restart_dir.mkdir(parents=True, exist_ok=True)
        (restart_dir / "mHM_restart_001.nc").write_text("restart")

    monkeypatch.setattr(
        "mhm_tools.pre.create_mhm_restart_from_setup.crop_mhm_setup",
        fail_crop_mhm_setup,
    )
    monkeypatch.setattr(MHMRunner, "run_mhm", fake_run_mhm)

    result = create_mhm_restart_from_setup(
        input_path=input_path,
        output_path=output_path,
        mask_da=None,
        lon_min=0.0,
        lon_max=2.0,
        lat_min=0.0,
        lat_max=2.0,
        l1_resolution=1.0,
        l1_increment=1,
        merge=False,
        skip_tile_creation=True,
    )

    assert run_calls == ["slice_0_0", "slice_0_1", "slice_1_0", "slice_1_1"]
    assert result["restart_files"] == [
        output_path / "slice_0_0" / "output" / "mHM_restart_001.nc",
        output_path / "slice_0_1" / "output" / "mHM_restart_001.nc",
        output_path / "slice_1_0" / "output" / "mHM_restart_001.nc",
        output_path / "slice_1_1" / "output" / "mHM_restart_001.nc",
    ]


def test_create_mhm_restart_from_setup_can_skip_mhm_run(tmp_path, monkeypatch):
    output_path = tmp_path / "cropped_tiles"

    for tile in ("slice_0_0", "slice_0_1", "slice_1_0", "slice_1_1"):
        restart_dir = output_path / tile / "output"
        restart_dir.mkdir(parents=True)
        (restart_dir / "mHM_restart_001.nc").write_text("restart")

    def fail_run_mhm(self, setup_path):  # noqa: ARG001
        raise AssertionError("mHM should not be run")

    monkeypatch.setattr(MHMRunner, "run_mhm", fail_run_mhm)

    result = create_mhm_restart_from_setup(
        input_path=tmp_path / "input_setup",
        output_path=output_path,
        mask_da=None,
        lon_min=0.0,
        lon_max=2.0,
        lat_min=0.0,
        lat_max=2.0,
        l1_resolution=1.0,
        l1_increment=1,
        merge=False,
        skip_tile_creation=True,
        skip_mhm_run=True,
    )

    assert result["failed_tiles"] == []
    assert result["restart_files"] == [
        output_path / "slice_0_0" / "output" / "mHM_restart_001.nc",
        output_path / "slice_0_1" / "output" / "mHM_restart_001.nc",
        output_path / "slice_1_0" / "output" / "mHM_restart_001.nc",
        output_path / "slice_1_1" / "output" / "mHM_restart_001.nc",
    ]


def test_validate_prepared_tile_dirs_parallel_lists_all_missing(tmp_path):
    tiles = create_setup_tiles(
        lon_min_bound=0.0,
        lon_max_bound=3.0,
        lat_min_bound=0.0,
        lat_max_bound=1.0,
        l1_resolution=1.0,
        l1_increment=1,
        output_path=tmp_path / "cropped_tiles",
    )
    tiles[0].output_path.mkdir(parents=True)

    with pytest.raises(FileNotFoundError) as exc_info:
        _validate_prepared_tile_dirs(tiles, n_jobs=2)

    message = str(exc_info.value)
    assert str(tiles[1].output_path) in message
    assert str(tiles[2].output_path) in message
    assert message.index(str(tiles[1].output_path)) < message.index(
        str(tiles[2].output_path)
    )


def test_validate_prepared_tile_dirs_serial_accepts_existing_tiles(tmp_path):
    tiles = create_setup_tiles(
        lon_min_bound=0.0,
        lon_max_bound=2.0,
        lat_min_bound=0.0,
        lat_max_bound=1.0,
        l1_resolution=1.0,
        l1_increment=1,
        output_path=tmp_path / "cropped_tiles",
    )
    for tile in tiles:
        tile.output_path.mkdir(parents=True)

    _validate_prepared_tile_dirs(tiles, n_jobs=1)


def test_collect_restart_files_for_tiles_parallel_preserves_order(tmp_path):
    tiles = create_setup_tiles(
        lon_min_bound=0.0,
        lon_max_bound=3.0,
        lat_min_bound=0.0,
        lat_max_bound=1.0,
        l1_resolution=1.0,
        l1_increment=1,
        output_path=tmp_path / "cropped_tiles",
    )
    for tile in tiles:
        (tile.output_path / "output").mkdir(parents=True)
    restart_0 = tiles[0].output_path / "output" / "mHM_restart_001.nc"
    restart_2 = tiles[2].output_path / "output" / "mHM_restart_001.nc"
    restart_0.write_text("restart")
    restart_2.write_text("restart")

    results = _collect_restart_files_for_tiles(
        prepared_tiles=tiles,
        restart_pattern="**/mHM_restart*.nc",
        require_restart=True,
        n_jobs=2,
    )

    assert [result["tile"] for result in results] == [tile.name for tile in tiles]
    assert results[0]["status"] == 0
    assert results[0]["restart_files"] == [restart_0]
    assert results[1]["status"] == 1
    assert results[1]["restart_files"] == []
    assert "No restart file matching" in results[1]["message"]
    assert results[2]["status"] == 0
    assert results[2]["restart_files"] == [restart_2]


def test_collect_restart_files_for_tiles_searches_restart_output_without_duplicates(
    tmp_path,
):
    setup_path = tmp_path / "cropped_tiles"
    restart_output_path = tmp_path / "restart"
    tiles = create_setup_tiles(
        lon_min_bound=0.0,
        lon_max_bound=2.0,
        lat_min_bound=0.0,
        lat_max_bound=1.0,
        l1_resolution=1.0,
        l1_increment=1,
        output_path=setup_path,
    )
    for tile in tiles:
        (tile.output_path / "output").mkdir(parents=True)

    restart_0 = tiles[0].output_path / "output" / "mHM_restart_001.nc"
    moved_restart_0 = (
        restart_output_path
        / tiles[0].output_path.relative_to(setup_path)
        / "output"
        / "mHM_restart_001.nc"
    )
    moved_restart_1 = (
        restart_output_path
        / tiles[1].output_path.relative_to(setup_path)
        / "output"
        / "mHM_restart_001.nc"
    )
    restart_0.write_text("restart in tile")
    moved_restart_0.parent.mkdir(parents=True)
    moved_restart_0.write_text("restart already moved")
    moved_restart_1.parent.mkdir(parents=True)
    moved_restart_1.write_text("restart only moved")

    results = _collect_restart_files_for_tiles(
        prepared_tiles=tiles,
        restart_pattern="**/mHM_restart*.nc",
        require_restart=True,
        n_jobs=1,
        setup_path=setup_path,
        restart_output_path=restart_output_path,
    )

    assert results[0]["status"] == 0
    assert results[0]["restart_files"] == [restart_0]
    assert results[1]["status"] == 0
    assert results[1]["restart_files"] == [moved_restart_1]


def test_create_mhm_restart_from_setup_reports_missing_restarts_separately(
    tmp_path, monkeypatch, capsys
):
    output_path = tmp_path / "cropped_tiles"
    existing_failed_file = output_path / "failed_mhm_tiles.txt"

    for tile in ("slice_0_0", "slice_1_0"):
        (output_path / tile / "output").mkdir(parents=True)
    (output_path / "slice_0_0" / "output" / "mHM_restart_001.nc").write_text("restart")
    existing_failed_file.write_text("existing mHM failure report\n")

    def fail_run_mhm(self, setup_path):  # noqa: ARG001
        raise AssertionError("mHM should not be run")

    monkeypatch.setattr(MHMRunner, "run_mhm", fail_run_mhm)

    result = create_mhm_restart_from_setup(
        input_path=tmp_path / "input_setup",
        output_path=output_path,
        mask_da=None,
        lon_min=0.0,
        lon_max=2.0,
        lat_min=0.0,
        lat_max=1.0,
        l1_resolution=1.0,
        l1_increment=1,
        merge=False,
        skip_tile_creation=True,
        skip_mhm_run=True,
        n_jobs=2,
    )

    assert result["restart_files"] == [
        output_path / "slice_0_0" / "output" / "mHM_restart_001.nc"
    ]
    assert result["failed_tiles"] == [
        {
            "tile": "slice_1_0",
            "message": (
                "No restart file matching '**/mHM_restart*.nc' was found for "
                f"slice_1_0 in {output_path / 'slice_1_0'} while mHM runs "
                "were skipped."
            ),
        }
    ]
    assert result["failed_tiles_file"] == output_path / "missing_restart_files.txt"
    assert existing_failed_file.read_text() == "existing mHM failure report\n"
    assert "Missing restart files:" in capsys.readouterr().out
    assert "slice_1_0" in result["failed_tiles_file"].read_text()


def test_create_mhm_restart_from_setup_recreates_missing_restarts(
    tmp_path, monkeypatch
):
    output_path = tmp_path / "cropped_tiles"
    repair_calls = []

    for tile in ("slice_0_0", "slice_1_0"):
        (output_path / tile / "output").mkdir(parents=True)
    existing_restart = output_path / "slice_0_0" / "output" / "mHM_restart_001.nc"
    recreated_restart = output_path / "slice_1_0" / "output" / "mHM_restart_001.nc"
    existing_restart.write_text("restart")

    def fake_restore_tile_meteo_from_original(**kwargs):
        repair_calls.append(
            (
                "restore",
                kwargs["tile"].name,
                kwargs["input_path"],
                kwargs["l1_resolution"],
                kwargs["lat_order"],
            )
        )

    def fake_write_tile_meteo_header(setup_path, l1_resolution):
        repair_calls.append(("header", Path(setup_path).name, l1_resolution))

    def fake_restore_recreated_fill_files_from_original(**kwargs):
        repair_calls.append(
            (
                "restore_fill",
                kwargs["tile"].name,
                kwargs["input_path"],
                kwargs["fill_nearest_files"],
            )
        )

    def fake_fill_recreated_restart_inputs(tile, fill_nearest_files):
        repair_calls.append(("fill", tile.name, fill_nearest_files))

    def fake_run_mhm_for_tile(**kwargs):
        tile = kwargs["tile"]
        repair_calls.append(("mhm", tile.name, kwargs["restart_pattern"]))
        recreated_restart.write_text("restart")
        return {
            "tile": tile.name,
            "restart_files": [recreated_restart],
            "status": 0,
            "message": "",
        }

    monkeypatch.setattr(
        "mhm_tools.pre.create_mhm_restart_from_setup._restore_tile_meteo_from_original",
        fake_restore_tile_meteo_from_original,
    )
    monkeypatch.setattr(
        "mhm_tools.pre.create_mhm_restart_from_setup._write_tile_meteo_header",
        fake_write_tile_meteo_header,
    )
    monkeypatch.setattr(
        "mhm_tools.pre.create_mhm_restart_from_setup._restore_recreated_fill_files_from_original",
        fake_restore_recreated_fill_files_from_original,
    )
    monkeypatch.setattr(
        "mhm_tools.pre.create_mhm_restart_from_setup._fill_recreated_restart_inputs",
        fake_fill_recreated_restart_inputs,
    )
    monkeypatch.setattr(
        "mhm_tools.pre.create_mhm_restart_from_setup._run_mhm_for_tile",
        fake_run_mhm_for_tile,
    )

    result = create_mhm_restart_from_setup(
        input_path=tmp_path / "input_setup",
        output_path=output_path,
        mask_da=None,
        lon_min=0.0,
        lon_max=2.0,
        lat_min=0.0,
        lat_max=1.0,
        l1_resolution=1.0,
        l1_increment=1,
        merge=False,
        fill_nearest_files=["soil.nc"],
        skip_tile_creation=True,
        skip_mhm_run=True,
        recreate_restart=True,
    )

    assert result["failed_tiles"] == []
    assert result["restart_files"] == [existing_restart, recreated_restart]
    assert repair_calls == [
        ("restore", "slice_1_0", tmp_path / "input_setup", 1.0, "decreasing"),
        ("header", "slice_1_0", 1.0),
        ("restore_fill", "slice_1_0", tmp_path / "input_setup", ["soil.nc"]),
        ("fill", "slice_1_0", ["soil.nc"]),
        ("mhm", "slice_1_0", "**/mHM_restart*.nc"),
    ]


def test_restore_recreated_fill_files_from_original_recrops_non_meteo_files(
    tmp_path, monkeypatch
):
    input_path = tmp_path / "input_setup"
    tile_path = tmp_path / "tiles" / "slice_0_0"
    source_soil = input_path / "input" / "morph" / "soil.nc"
    tile_soil = tile_path / "input" / "morph" / "soil.nc"
    source_meteo = input_path / "input" / "meteo" / "pre.nc"
    tile_meteo = tile_path / "input" / "meteo" / "pre.nc"
    source_soil.parent.mkdir(parents=True)
    source_meteo.parent.mkdir(parents=True)
    tile_soil.parent.mkdir(parents=True)
    tile_meteo.parent.mkdir(parents=True)
    source_soil.write_text("original soil")
    source_meteo.write_text("original meteo")
    tile_soil.write_text("damaged soil")
    tile_meteo.write_text("damaged meteo")
    crop_calls = []

    tile = MHMSetupTile(
        name="slice_0_0",
        output_path=tile_path,
        lonslice=slice(0, 1),
        latslice=slice(1, 0),
        lon_min=0.5,
        lon_max=0.5,
        lat_min=0.5,
        lat_max=0.5,
    )

    def fake_crop_mhm_setup(**kwargs):
        crop_calls.append(kwargs)
        output_file = Path(kwargs["output_path"]) / Path(kwargs["input_path"]).name
        output_file.write_text("restored")

    monkeypatch.setattr(
        "mhm_tools.pre.create_mhm_restart_from_setup.crop_mhm_setup",
        fake_crop_mhm_setup,
    )

    restored_files = _restore_recreated_fill_files_from_original(
        tile=tile,
        input_path=input_path,
        fill_nearest_files=["*.nc"],
        l1_resolution=1.0,
        l11_resolution=0.5,
        crs="EPSG:4326",
        crop_n_jobs=2,
        available_mem_gib=5,
        chunking=True,
        lat_order="increasing",
    )

    assert restored_files == [tile_soil]
    assert tile_soil.read_text() == "restored"
    assert tile_meteo.read_text() == "damaged meteo"
    assert len(crop_calls) == 1
    assert crop_calls[0]["input_path"] == source_soil
    assert crop_calls[0]["output_path"] == tile_soil.parent
    assert crop_calls[0]["lonslice"] == tile.lonslice
    assert crop_calls[0]["latslice"] == tile.latslice
    assert crop_calls[0]["resolutions"].l1 == 1.0
    assert crop_calls[0]["resolutions"].l11 == 0.5
    assert crop_calls[0]["crs"] == "EPSG:4326"
    assert crop_calls[0]["n_jobs"] == 2
    assert crop_calls[0]["chunking"] is True
    assert crop_calls[0]["lat_order"] == "increasing"


def test_fill_recreated_restart_inputs_uses_nearest_for_meteo(tmp_path, monkeypatch):
    tile = MHMSetupTile(
        name="slice_0_0",
        output_path=tmp_path / "slice_0_0",
        lonslice=slice(0, 1),
        latslice=slice(1, 0),
        lon_min=0.5,
        lon_max=0.5,
        lat_min=0.5,
        lat_max=0.5,
    )
    meteo_dir = tile.output_path / "input" / "meteo"
    meteo_dir.mkdir(parents=True)
    for meteo_name in ("pre.nc", "pet.nc", "tavg.nc"):
        (meteo_dir / meteo_name).write_text("original")
    soil_file = tile.output_path / "input" / "soil.nc"
    soil_file.parent.mkdir(parents=True, exist_ok=True)
    soil_file.write_text("soil")
    fill_calls = []

    def fake_fill_nearest(**kwargs):
        fill_calls.append(kwargs)
        staged = Path(kwargs["output_dir"]) / kwargs["fname"]
        staged.parent.mkdir(parents=True, exist_ok=True)
        staged.write_text("filled")
        return [staged]

    monkeypatch.setattr(
        "mhm_tools.pre.create_mhm_restart_from_setup.fill_nearest",
        fake_fill_nearest,
    )

    from mhm_tools.pre.create_mhm_restart_from_setup import (
        _fill_recreated_restart_inputs,
    )

    _fill_recreated_restart_inputs(tile, ["soil.nc"])

    assert [call["fname"] for call in fill_calls] == [
        "pre.nc",
        "pet.nc",
        "tavg.nc",
        "soil.nc",
    ]
    assert [call["input_dir"] for call in fill_calls] == [
        meteo_dir,
        meteo_dir,
        meteo_dir,
        soil_file.parent,
    ]
    assert all(call["mask_file"] is None for call in fill_calls)
    assert all(call["mask_var"] is None for call in fill_calls)
    assert [call["fill_value"] for call in fill_calls] == [2.2, 2.2, 2.2, 1]
    assert all("default_value" not in call for call in fill_calls)
    assert (meteo_dir / "pre.nc").read_text() == "filled"
    assert soil_file.read_text() == "filled"


def test_move_restart_files_moves_tile_mask(tmp_path):
    setup_path = tmp_path / "setup"
    restart_file = setup_path / "slice_0_0" / "output" / "mHM_restart_001.nc"
    mask_file = setup_path / "slice_0_0" / "mask_tile.nc"
    restart_file.parent.mkdir(parents=True)
    restart_file.write_text("restart")
    mask_file.write_text("mask")
    target_restart_file = (
        tmp_path / "restart" / "slice_0_0" / "output" / "mHM_restart_001.nc"
    )
    target_mask_file = tmp_path / "restart" / "slice_0_0" / "mask_tile.nc"
    target_restart_file.parent.mkdir(parents=True)
    target_restart_file.write_text("old restart")
    target_mask_file.write_text("old mask")

    moved = _move_restart_files(
        restart_files=[restart_file],
        setup_path=setup_path,
        restart_output_path=tmp_path / "restart",
    )

    assert moved == [target_restart_file]
    assert target_restart_file.read_text() == "restart"
    assert target_mask_file.read_text() == "mask"
    assert not restart_file.exists()
    assert not mask_file.exists()


def test_move_restart_files_keeps_restart_files_already_at_output_path(tmp_path):
    restart_output_path = tmp_path / "restart"
    restart_file = restart_output_path / "slice_0_0" / "output" / "mHM_restart_001.nc"
    restart_file.parent.mkdir(parents=True)
    restart_file.write_text("restart")

    moved = _move_restart_files(
        restart_files=[restart_file],
        setup_path=tmp_path / "setup",
        restart_output_path=restart_output_path,
    )

    assert moved == [restart_file]
    assert restart_file.read_text() == "restart"


def test_create_mhm_restart_from_setup_skip_tile_creation_prepares_missing_tiles(
    tmp_path, monkeypatch
):
    input_path = tmp_path / "input_setup"
    output_path = tmp_path / "cropped_tiles"
    input_path.mkdir()
    crop_calls = []

    def fake_crop_mhm_setup(**kwargs):
        crop_calls.append(kwargs)
        Path(kwargs["output_path"]).mkdir(parents=True, exist_ok=True)

    def fake_run_mhm(self, setup_path):  # noqa: ARG001
        restart_dir = Path(setup_path) / "output"
        restart_dir.mkdir(parents=True, exist_ok=True)
        (restart_dir / "mHM_restart_001.nc").write_text("restart")

    monkeypatch.setattr(
        "mhm_tools.pre.create_mhm_restart_from_setup.crop_mhm_setup",
        fake_crop_mhm_setup,
    )
    monkeypatch.setattr(MHMRunner, "run_mhm", fake_run_mhm)

    result = create_mhm_restart_from_setup(
        input_path=input_path,
        output_path=output_path,
        mask_da=None,
        lon_min=0.0,
        lon_max=1.0,
        lat_min=0.0,
        lat_max=1.0,
        l1_resolution=1.0,
        l1_increment=1,
        merge=False,
        skip_tile_creation=True,
    )

    assert [call["output_path"] for call in crop_calls] == [output_path / "slice_0_0"]
    assert result["restart_files"] == [
        output_path / "slice_0_0" / "output" / "mHM_restart_001.nc"
    ]


def test_create_mhm_restart_from_setup_uses_mask_only_to_select_tiles(
    tmp_path, monkeypatch
):
    input_path = tmp_path / "input_setup"
    output_path = tmp_path / "cropped_tiles"
    input_path.mkdir()
    calls = []

    def fake_crop_mhm_setup(**kwargs):
        calls.append(kwargs)
        Path(kwargs["output_path"]).mkdir(parents=True, exist_ok=True)

    def fake_run_mhm(self, setup_path):  # noqa: ARG001
        restart_dir = Path(setup_path) / "output"
        restart_dir.mkdir(parents=True, exist_ok=True)
        (restart_dir / "mHM_restart_001.nc").write_text("restart")

    mask_ds = xr.Dataset(
        data_vars={
            "land_mask": (
                ("lat", "lon"),
                np.array([[0, 1], [0, 0]]),
            )
        },
        coords={
            "lat": np.array([1.5, 0.5]),
            "lon": np.array([0.5, 1.5]),
        },
    )
    monkeypatch.setattr(
        "mhm_tools.pre.create_mhm_restart_from_setup.crop_mhm_setup",
        fake_crop_mhm_setup,
    )
    monkeypatch.setattr(MHMRunner, "run_mhm", fake_run_mhm)

    restart_files = create_mhm_restart_from_setup(
        input_path=input_path,
        output_path=output_path,
        mask_da=mask_ds,
        lon_min=0.0,
        lon_max=2.0,
        lat_min=0.0,
        lat_max=2.0,
        l1_resolution=1.0,
        l1_increment=1,
        merge=False,
        mask_var="land_mask",
    )

    assert len(calls) == 1
    assert calls[0]["output_path"] == output_path / "slice_1_1"
    assert calls[0]["mask_ds"] is None
    assert calls[0]["mask_all"] is False
    assert restart_files["restart_files"] == [
        output_path / "slice_1_1" / "output" / "mHM_restart_001.nc",
    ]
    with xr.open_dataset(output_path / "slice_1_1" / "mask_tile.nc") as tile_mask:
        assert list(tile_mask.data_vars) == ["land_mask"]
        assert tile_mask["land_mask"].item() == 1


def test_create_mhm_restart_from_setup_fills_files_before_mhm(tmp_path, monkeypatch):
    input_path = tmp_path / "input_setup"
    output_path = tmp_path / "cropped_tiles"
    input_path.mkdir()
    fill_calls = []

    def fake_crop_mhm_setup(**kwargs):
        tile_path = Path(kwargs["output_path"])
        tile_path.mkdir(parents=True, exist_ok=True)
        (tile_path / "dem.nc").write_text("dem")
        (tile_path / "forcing.nc").write_text("forcing")

    def fake_fill_nearest(**kwargs):
        fill_calls.append(kwargs)
        return [Path(kwargs["input_dir"]) / kwargs["fname"]]

    def fake_run_mhm(self, setup_path):  # noqa: ARG001
        assert len(fill_calls) == 1
        restart_dir = Path(setup_path) / "output"
        restart_dir.mkdir(parents=True, exist_ok=True)
        (restart_dir / "mHM_restart_001.nc").write_text("restart")

    monkeypatch.setattr(
        "mhm_tools.pre.create_mhm_restart_from_setup.crop_mhm_setup",
        fake_crop_mhm_setup,
    )
    monkeypatch.setattr(
        "mhm_tools.pre.create_mhm_restart_from_setup.fill_nearest",
        fake_fill_nearest,
    )
    monkeypatch.setattr(MHMRunner, "run_mhm", fake_run_mhm)

    create_mhm_restart_from_setup(
        input_path=input_path,
        output_path=output_path,
        mask_da=None,
        lon_min=0.0,
        lon_max=1.0,
        lat_min=0.0,
        lat_max=1.0,
        l1_resolution=1.0,
        l1_increment=1,
        merge=False,
        fill_nearest_files=["forcing.nc"],
    )

    assert fill_calls == [
        {
            "input_dir": output_path / "slice_0_0",
            "fname": "forcing.nc",
            "output_dir": output_path / "slice_0_0",
            "mask_file": output_path / "slice_0_0" / "dem.nc",
            "mask_var": None,
        }
    ]


def test_create_mhm_restart_from_setup_masks_dem_with_l0_mask_files(
    tmp_path, monkeypatch
):
    input_path = tmp_path / "input_setup"
    output_path = tmp_path / "cropped_tiles"
    input_path.mkdir()

    def fake_crop_mhm_setup(**kwargs):
        tile_path = Path(kwargs["output_path"])
        tile_path.mkdir(parents=True, exist_ok=True)
        dem = xr.Dataset(
            data_vars={
                "dem": (
                    ("lat", "lon"),
                    np.array([[10.0, 20.0], [30.0, 40.0]]),
                )
            },
            coords={
                "lat": np.array([1.5, 0.5]),
                "lon": np.array([0.5, 1.5]),
            },
        )
        l0_mask = xr.Dataset(
            data_vars={
                "mask_values": (
                    ("lat", "lon"),
                    np.array([[1.0, -9999.0], [2.0, -9999.0]]),
                    {"missing_value": -9999.0},
                )
            },
            coords={
                "lat": np.array([1.5, 0.5]),
                "lon": np.array([0.5, 1.5]),
            },
        )
        dem.to_netcdf(tile_path / "dem.nc")
        l0_mask.to_netcdf(tile_path / "l0_mask.nc")

    def fake_run_mhm(self, setup_path):  # noqa: ARG001
        setup_path = Path(setup_path)
        assert (setup_path / "output").is_dir()
        assert (setup_path / "restart").is_dir()
        with xr.open_dataset(setup_path / "dem.nc") as dem:
            assert dem["dem"].sel(lat=1.5, lon=0.5).item() == 10.0
            assert bool(np.isnan(dem["dem"].sel(lat=1.5, lon=1.5)).item())
            assert dem["dem"].sel(lat=0.5, lon=0.5).item() == 30.0
            assert bool(np.isnan(dem["dem"].sel(lat=0.5, lon=1.5)).item())
        restart_dir = setup_path / "output"
        (restart_dir / "mHM_restart_001.nc").write_text("restart")

    monkeypatch.setattr(
        "mhm_tools.pre.create_mhm_restart_from_setup.crop_mhm_setup",
        fake_crop_mhm_setup,
    )
    monkeypatch.setattr(MHMRunner, "run_mhm", fake_run_mhm)

    create_mhm_restart_from_setup(
        input_path=input_path,
        output_path=output_path,
        mask_da=None,
        lon_min=0.0,
        lon_max=2.0,
        lat_min=0.0,
        lat_max=2.0,
        l1_resolution=1.0,
        l1_increment=2,
        merge=False,
        l0_mask_files=["l0_mask.nc"],
    )


def test_create_mhm_restart_from_setup_fills_meteo_files_without_mask(
    tmp_path, monkeypatch
):
    input_path = tmp_path / "input_setup"
    output_path = tmp_path / "cropped_tiles"
    input_path.mkdir()
    fill_calls = []

    def fake_crop_mhm_setup(**kwargs):
        tile_path = Path(kwargs["output_path"])
        meteo_path = tile_path / "meteo"
        nested_meteo_path = tile_path / "nested" / "meteo"
        meteo_path.mkdir(parents=True, exist_ok=True)
        nested_meteo_path.mkdir(parents=True, exist_ok=True)
        (meteo_path / "pre.nc").write_text("forcing")
        (nested_meteo_path / "temp.nc").write_text("forcing")

    def fake_fill_nearest(**kwargs):
        fill_calls.append(kwargs)
        output_file = Path(kwargs["output_dir"]) / kwargs["fname"]
        output_file.parent.mkdir(parents=True, exist_ok=True)
        output_file.write_text("filled")
        return [output_file]

    def fake_run_mhm(self, setup_path):  # noqa: ARG001
        setup_path = Path(setup_path)
        assert (setup_path / "meteo" / "pre.nc").read_text() == "filled"
        assert (setup_path / "nested" / "meteo" / "temp.nc").read_text() == "filled"
        restart_dir = Path(setup_path) / "output"
        restart_dir.mkdir(parents=True, exist_ok=True)
        (restart_dir / "mHM_restart_001.nc").write_text("restart")

    monkeypatch.setattr(
        "mhm_tools.pre.create_mhm_restart_from_setup.crop_mhm_setup",
        fake_crop_mhm_setup,
    )
    monkeypatch.setattr(
        "mhm_tools.pre.create_mhm_restart_from_setup.fill_nearest",
        fake_fill_nearest,
    )
    monkeypatch.setattr(MHMRunner, "run_mhm", fake_run_mhm)

    create_mhm_restart_from_setup(
        input_path=input_path,
        output_path=output_path,
        mask_da=None,
        lon_min=0.0,
        lon_max=1.0,
        lat_min=0.0,
        lat_max=1.0,
        l1_resolution=1.0,
        l1_increment=1,
        merge=False,
    )

    assert fill_calls == [
        {
            "input_dir": output_path / "slice_0_0" / "meteo",
            "fname": "pre.nc",
            "output_dir": output_path / "slice_0_0" / "meteo_filled",
            "mask_file": None,
            "mask_var": None,
        },
        {
            "input_dir": output_path / "slice_0_0" / "nested" / "meteo",
            "fname": "temp.nc",
            "output_dir": output_path / "slice_0_0" / "nested" / "meteo_filled",
            "mask_file": None,
            "mask_var": None,
        },
    ]


def test_tile_selection_uses_mask_cell_overlap_at_tile_boundary():
    tile = MHMSetupTile(
        name="slice_0_0",
        output_path=Path("slice_0_0"),
        lonslice=slice(0.0, 1.0),
        latslice=slice(1.0, 0.0),
        lon_min=0.25,
        lon_max=0.75,
        lat_min=0.25,
        lat_max=0.75,
    )
    mask_da = xr.DataArray(
        np.array([[1]]),
        dims=("lat", "lon"),
        coords={"lat": [0.95], "lon": [0.95]},
    )

    assert _tile_has_active_mask_cell(tile, mask_da)


def test_tile_selection_ignores_non_positive_mask_values():
    tile = MHMSetupTile(
        name="slice_0_0",
        output_path=Path("slice_0_0"),
        lonslice=slice(0.0, 1.0),
        latslice=slice(1.0, 0.0),
        lon_min=0.25,
        lon_max=0.75,
        lat_min=0.25,
        lat_max=0.75,
    )
    mask_da = xr.DataArray(
        np.array([[-1.0, 0.0]]),
        dims=("lat", "lon"),
        coords={"lat": [0.5], "lon": [0.25, 0.75]},
    )

    assert not _tile_has_active_mask_cell(tile, mask_da)


def test_create_mhm_restart_from_setup_considers_all_positive_mask_tiles(
    tmp_path, monkeypatch
):
    input_path = tmp_path / "input_setup"
    output_path = tmp_path / "cropped_tiles"
    input_path.mkdir()
    calls = []

    def fake_crop_mhm_setup(**kwargs):
        calls.append(kwargs)
        restart_dir = Path(kwargs["output_path"]) / "output"
        restart_dir.mkdir(parents=True, exist_ok=True)

    def fake_run_mhm(self, setup_path):  # noqa: ARG001
        restart_dir = Path(setup_path) / "output"
        (restart_dir / "mHM_restart_001.nc").write_text("restart")

    monkeypatch.setattr(
        "mhm_tools.pre.create_mhm_restart_from_setup.crop_mhm_setup",
        fake_crop_mhm_setup,
    )
    monkeypatch.setattr(MHMRunner, "run_mhm", fake_run_mhm)

    mask_ds = xr.Dataset(
        data_vars={
            "land_mask": (
                ("lat", "lon"),
                np.ones((5, 5), dtype=float),
            )
        },
        coords={
            "lat": np.array([4.5, 3.5, 2.5, 1.5, 0.5]),
            "lon": np.array([0.5, 1.5, 2.5, 3.5, 4.5]),
        },
    )

    result = create_mhm_restart_from_setup(
        input_path=input_path,
        output_path=output_path,
        mask_da=mask_ds,
        lon_min=0.0,
        lon_max=5.0,
        lat_min=0.0,
        lat_max=5.0,
        l1_resolution=1.0,
        l1_increment=1,
        merge=False,
        mask_var="land_mask",
    )

    assert len(calls) == 25
    assert len(result["tiles"]) == 25


def test_crop_mhm_setup_creates_default_latlon_from_cropped_dem(tmp_path, monkeypatch):
    input_path = tmp_path / "input_setup"
    output_path = tmp_path / "cropped_setup"
    dem_file = input_path / "morph" / "dem.nc"
    dem_file.parent.mkdir(parents=True)
    dem_file.write_text("dem")
    calls = []

    def fake_crop_file(**kwargs):
        latlon_files = LatlonFiles()
        latlon_files.set_dem_output_file(
            Path(kwargs["output_path"]) / "morph" / "dem.nc"
        )
        return latlon_files

    def fake_call_create_latlon(*args, **kwargs):
        calls.append((args, kwargs))

    monkeypatch.setattr("mhm_tools.pre.crop_mhm_setup.crop_file", fake_crop_file)
    monkeypatch.setattr(
        "mhm_tools.pre.crop_mhm_setup.call_create_latlon",
        fake_call_create_latlon,
    )
    resolutions = Resolution(l1=1.0)
    resolutions.l11 = None

    crop_mhm_setup(
        mask_ds=None,
        output_path=output_path,
        input_path=input_path,
        resolutions=resolutions,
        lonslice=slice(0.0, 1.0),
        latslice=slice(1.0, 0.0),
        filename="morph/dem.nc",
    )

    assert len(calls) == 1
    args, kwargs = calls[0]
    assert args == ()
    assert kwargs["dem_output_file"] == output_path / "morph" / "dem.nc"
    assert kwargs["resolutions"] is resolutions
    assert kwargs["latlon_output_file"] == output_path / "latlon" / "latlon.nc"
    assert kwargs["meteo_header_path"] is None
    assert kwargs["lat_order"] == "decreasing"


def test_mhm_runner_uses_python_bindings():
    runner = MHMRunner()
    command = runner.get_command()

    assert command.startswith("python -c ")
    assert "import mhm" in command
    assert "mhm.model.init(" in command
    assert "mhm.model.run()" in command
    assert "mhm.model.finalize()" in command


def test_mhm_runner_uses_module_paths_before_loading_modules():
    runner = MHMRunner(
        mhm_packages=(
            "/global/apps/modulefiles python_env_mpr iomkl/2020b "
            "netCDF-Fortran/4.5.3 CMake pFUnit/4.2.2_iomkl2020b"
        ),
        mhm_args="-p FinalParam.nml",
    )
    command = runner.get_command()

    assert command.startswith("module use /global/apps/modulefiles\n")
    assert (
        "module load python_env_mpr iomkl/2020b netCDF-Fortran/4.5.3 "
        "CMake pFUnit/4.2.2_iomkl2020b\n"
    ) in command
    assert "python -c " in command
    assert 'namelist_mhm_param="FinalParam.nml"' in command


def test_merge_mhm_restart_files_reindexes_to_global_domain(tmp_path):
    restart_file = tmp_path / "tile_restart.nc"
    output_file = tmp_path / "mHM_restart_001.nc"
    ds = xr.Dataset(
        data_vars={
            "L1_state": (
                ("lat", "lon"),
                np.array([[1.0, 2.0], [3.0, 4.0]]),
            )
        },
        coords={
            "lat": np.array([1.5, 0.5]),
            "lon": np.array([0.5, 1.5]),
        },
    )
    ds.to_netcdf(restart_file)

    merge_mhm_restart_files(
        restart_files=[restart_file],
        output_file=output_file,
        lon_min_bound=0.0,
        lon_max_bound=3.0,
        lat_min_bound=0.0,
        lat_max_bound=2.0,
        l1_resolution=1.0,
    )

    with xr.open_dataset(output_file) as merged:
        assert list(merged["lon"].values) == [0.5, 1.5, 2.5]
        assert list(merged["lat"].values) == [1.5, 0.5]
        assert bool(np.isnan(merged["L1_state"].sel(lat=1.5, lon=2.5)).item())


def test_merge_restart_files_writes_cf_lat_lon_without_flipping_native_rows(
    tmp_path,
):
    west_file = tmp_path / "slice_0_0" / "output" / "mHM_restart_001.nc"
    east_file = tmp_path / "slice_1_0" / "output" / "mHM_restart_001.nc"
    west_file.parent.mkdir(parents=True)
    east_file.parent.mkdir(parents=True)
    output_file = tmp_path / "merged.nc"
    west = xr.Dataset(
        data_vars={
            "L1_fAsp": (
                ("ncols1", "nrows1"),
                np.array([[1.0, 2.0], [3.0, 4.0]]),
            ),
            "L1_Max_Canopy_Intercept": (
                ("L1_LAITimesteps", "ncols1", "nrows1"),
                np.array([[[5.0, 6.0], [7.0, 8.0]]]),
            ),
        },
        attrs={
            "xllcorner_L1": 0.0,
            "yllcorner_L1": 0.0,
            "cellsize_L1": 1.0,
            "ncols_L1": 2,
            "nrows_L1": 2,
        },
    )
    east = west.copy(deep=True)
    east.attrs["xllcorner_L1"] = 2.0
    east["L1_fAsp"] = (
        ("ncols1", "nrows1"),
        np.array([[9.0, 10.0], [11.0, 12.0]]),
    )
    west.to_netcdf(west_file)
    east.to_netcdf(east_file)

    merged = _merge_restart_files(
        restart_file_paths=[west_file, east_file],
        lon_min=0.0,
        lon_max=4.0,
        lat_min=0.0,
        lat_max=2.0,
        l1_resolution=1.0,
        output_file=output_file,
    )

    assert output_file.exists()
    assert list(merged.sizes)[:2] == ["lat", "lon"]
    assert merged.sizes["lon"] == 4
    assert merged.sizes["lat"] == 2
    assert merged["L1_fAsp"].dims == ("lat", "lon")
    assert merged["L1_maxInter"].dims == ("L1_LAITimesteps", "lat", "lon")
    assert list(merged["lon"].values) == [0.5, 1.5, 2.5, 3.5]
    assert list(merged["lat"].values) == [1.5, 0.5]
    np.testing.assert_array_equal(
        merged["L1_fAsp"].values,
        np.array([[1.0, 2.0, 9.0, 10.0], [3.0, 4.0, 11.0, 12.0]]),
    )
    np.testing.assert_array_equal(
        merged["L1_maxInter"].values,
        np.array([[[5.0, 6.0, 5.0, 6.0], [7.0, 8.0, 7.0, 8.0]]]),
    )
    assert merged.attrs["xllcorner_L1"] == 0.0
    assert merged.attrs["yllcorner_L1"] == 0.0
    assert merged.attrs["nrows_L1"] == 4
    assert merged.attrs["ncols_L1"] == 2
    assert merged["lon"].attrs["axis"] == "X"
    assert merged["lat"].attrs["axis"] == "Y"
    assert merged.attrs["coordinates"] == "lat_bnds lon_bnds"


def test_merge_restart_files_writes_merged_tile_mask(tmp_path):
    west_file = tmp_path / "slice_0_0" / "output" / "mHM_restart_001.nc"
    east_file = tmp_path / "slice_1_0" / "output" / "mHM_restart_001.nc"
    west_file.parent.mkdir(parents=True)
    east_file.parent.mkdir(parents=True)
    output_file = tmp_path / "merged.nc"
    restart = xr.Dataset(
        data_vars={
            "L1_fAsp": (
                ("ncols1", "nrows1"),
                np.ones((2, 2), dtype=float),
            ),
        },
        attrs={
            "xllcorner_L1": 0.0,
            "yllcorner_L1": 0.0,
            "cellsize_L1": 1.0,
            "ncols_L1": 2,
            "nrows_L1": 2,
        },
    )
    west = restart.copy(deep=True)
    east = restart.copy(deep=True)
    east.attrs["xllcorner_L1"] = 2.0
    west.to_netcdf(west_file)
    east.to_netcdf(east_file)
    west_mask = xr.Dataset(
        data_vars={"land_mask": (("lat", "lon"), np.array([[1, 0], [0, 1]]))},
        coords={"lat": np.array([1.5, 0.5]), "lon": np.array([0.5, 1.5])},
    )
    east_mask = xr.Dataset(
        data_vars={"land_mask": (("lat", "lon"), np.array([[0, 1], [1, 0]]))},
        coords={"lat": np.array([1.5, 0.5]), "lon": np.array([2.5, 3.5])},
    )
    west_mask.to_netcdf(west_file.parents[1] / "mask_tile.nc")
    east_mask.to_netcdf(east_file.parents[1] / "mask_tile.nc")

    merged = _merge_restart_files(
        restart_file_paths=[west_file, east_file],
        lon_min=0.0,
        lon_max=4.0,
        lat_min=0.0,
        lat_max=2.0,
        l1_resolution=1.0,
        output_file=output_file,
        mask_ds=[west_mask, east_mask],
        mask_var="land_mask",
    )

    tile_mask_file = tmp_path / "merged_tile_mask.nc"
    assert merged.attrs["merged_tile_mask_file"] == str(tile_mask_file)
    with xr.open_dataset(tile_mask_file) as tile_mask:
        assert tile_mask["land_mask"].dims == ("lat", "lon")
        np.testing.assert_array_equal(
            tile_mask["land_mask"].values,
            np.array([[1.0, 0.0, 0.0, 1.0], [0.0, 1.0, 1.0, 0.0]]),
        )


def test_merge_restart_files_places_tiles_from_yllcorner(tmp_path):
    south_file = tmp_path / "slice_0_0" / "output" / "mHM_restart_001.nc"
    north_file = tmp_path / "slice_0_1" / "output" / "mHM_restart_001.nc"
    south_file.parent.mkdir(parents=True)
    north_file.parent.mkdir(parents=True)
    output_file = tmp_path / "merged.nc"
    south = xr.Dataset(
        data_vars={"L1_fAsp": (("ncols1", "nrows1"), np.array([[1.0]]))},
        attrs={
            "xllcorner_L1": 0.0,
            "yllcorner_L1": 0.0,
            "cellsize_L1": 1.0,
            "ncols_L1": 1,
            "nrows_L1": 1,
        },
    )
    north = south.copy(deep=True)
    north.attrs["yllcorner_L1"] = 1.0
    north["L1_fAsp"] = (("ncols1", "nrows1"), np.array([[2.0]]))
    south.to_netcdf(south_file)
    north.to_netcdf(north_file)

    merged = _merge_restart_files(
        restart_file_paths=[south_file, north_file],
        lon_min=0.0,
        lon_max=1.0,
        lat_min=0.0,
        lat_max=2.0,
        l1_resolution=1.0,
        output_file=output_file,
    )

    assert list(merged["lat"].values) == [1.5, 0.5]
    np.testing.assert_array_equal(merged["L1_fAsp"].values, np.array([[2.0], [1.0]]))


def test_merge_restart_files_allows_six_soil_horizons(tmp_path):
    restart_file = tmp_path / "slice_0_0" / "output" / "mHM_restart_001.nc"
    restart_file.parent.mkdir(parents=True)
    output_file = tmp_path / "merged.nc"
    ds = xr.Dataset(
        data_vars={
            "L1_soilMoist": (
                ("L1_LandCoverPeriods", "horizon_out", "ncols1", "nrows1"),
                np.ones((1, 6, 1, 1), dtype=float),
            ),
        },
        coords={
            "L1_LandCoverPeriods": np.array([2000]),
            "horizon_out": np.arange(6),
        },
        attrs={
            "xllcorner_L1": 0.0,
            "yllcorner_L1": 0.0,
            "cellsize_L1": 1.0,
            "ncols_L1": 1,
            "nrows_L1": 1,
        },
    )
    ds.to_netcdf(restart_file)

    merged = _merge_restart_files(
        restart_file_paths=[restart_file],
        lon_min=0.0,
        lon_max=1.0,
        lat_min=0.0,
        lat_max=1.0,
        l1_resolution=1.0,
        output_file=output_file,
    )

    assert merged.sizes["L1_SoilHorizons"] == 6
    assert merged["L1_SoilHorizons_bnds"].shape == (6, 2)
    assert merged["L1_soilMoist"].dims == (
        "L1_LandCoverPeriods",
        "L1_SoilHorizons",
        "lat",
        "lon",
    )


def test_merge_mhm_restart_files_prepares_tiles_in_parallel_batches(tmp_path):
    restart_files = []
    for index in range(3):
        restart_file = tmp_path / f"tile_restart_{index}.nc"
        ds = xr.Dataset(
            data_vars={
                "L1_state": (
                    ("lat", "lon"),
                    np.array([[float(index + 1)]]),
                )
            },
            coords={
                "lat": np.array([0.5]),
                "lon": np.array([index + 0.5]),
            },
        )
        ds.to_netcdf(restart_file)
        restart_files.append(restart_file)

    output_file = tmp_path / "mHM_restart_001.nc"
    merge_mhm_restart_files(
        restart_files=restart_files,
        output_file=output_file,
        lon_min_bound=0.0,
        lon_max_bound=3.0,
        lat_min_bound=0.0,
        lat_max_bound=1.0,
        l1_resolution=1.0,
        n_jobs=2,
    )

    with xr.open_dataset(output_file) as merged:
        assert merged["L1_state"].sel(lat=0.5, lon=0.5).item() == 1.0
        assert merged["L1_state"].sel(lat=0.5, lon=1.5).item() == 2.0
        assert merged["L1_state"].sel(lat=0.5, lon=2.5).item() == 3.0


def test_merge_mhm_restart_files_tolerates_restart_coordinate_roundoff(tmp_path):
    restart_file = tmp_path / "tile_restart.nc"
    output_file = tmp_path / "mHM_restart_001.nc"
    ds = xr.Dataset(
        data_vars={
            "L1_state": (
                ("ncols1", "nrows1"),
                np.array([[1.0, 2.0], [3.0, 4.0]]),
            )
        },
        attrs={
            "xllcorner_L1": 0.3,
            "yllcorner_L1": 0.0,
            "cellsize_L1": 0.1,
            "ncols_L1": 2,
            "nrows_L1": 2,
        },
    )
    ds.to_netcdf(restart_file)

    merge_mhm_restart_files(
        restart_files=[restart_file],
        output_file=output_file,
        lon_min_bound=0.0,
        lon_max_bound=0.5,
        lat_min_bound=0.0,
        lat_max_bound=0.2,
        l1_resolution=0.1,
    )

    with xr.open_dataset(output_file) as merged:
        assert merged["L1_state"].isel(lon=3, lat=0).item() == 1.0
        assert merged["L1_state"].isel(lon=4, lat=0).item() == 2.0
        assert merged["L1_state"].isel(lon=3, lat=1).item() == 3.0
        assert merged["L1_state"].isel(lon=4, lat=1).item() == 4.0


def test_merge_mhm_restart_files_preserves_requested_lower_left_attrs(tmp_path):
    restart_file = tmp_path / "tile_restart.nc"
    output_file = tmp_path / "mHM_restart_001.nc"
    ds = xr.Dataset(
        data_vars={
            "L1_state": (
                ("ncols1", "nrows1"),
                np.array([[1.0]]),
            )
        },
        attrs={
            "xllcorner_L1": 4.0,
            "yllcorner_L1": -60.0,
            "cellsize_L1": 1.0 / 240.0,
            "ncols_L1": 1,
            "nrows_L1": 1,
        },
    )
    ds.to_netcdf(restart_file)

    merge_mhm_restart_files(
        restart_files=[restart_file],
        output_file=output_file,
        lon_min_bound=4.0,
        lon_max_bound=4.0 + 1.0 / 240.0,
        lat_min_bound=-60.0,
        lat_max_bound=-60.0 + 1.0 / 240.0,
        l1_resolution=1.0 / 240.0,
    )

    with xr.open_dataset(output_file) as merged:
        assert merged.attrs["xllcorner_L1"] == 4.0
        assert merged.attrs["yllcorner_L1"] == -60.0


def test_merge_mhm_restart_files_transforms_restart_grid_metadata(tmp_path):
    restart_file_west = tmp_path / "tile_restart_west.nc"
    restart_file_east = tmp_path / "tile_restart_east.nc"
    output_file = tmp_path / "mHM_restart_001.nc"
    west = xr.Dataset(
        data_vars={
            "L1_state": (
                ("ncols1", "nrows1"),
                np.array([[1.0], [3.0]]),
            )
        },
        attrs={
            "xllcorner_L1": 0.0,
            "yllcorner_L1": 0.0,
            "cellsize_L1": 1.0,
            "ncols_L1": 2,
            "nrows_L1": 1,
        },
    )
    east = xr.Dataset(
        data_vars={
            "L1_state": (
                ("ncols1", "nrows1"),
                np.array([[2.0], [4.0]]),
            )
        },
        attrs={
            "xllcorner_L1": 1.0,
            "yllcorner_L1": 0.0,
            "cellsize_L1": 1.0,
            "ncols_L1": 2,
            "nrows_L1": 1,
        },
    )
    west.to_netcdf(restart_file_west)
    east.to_netcdf(restart_file_east)

    merge_mhm_restart_files(
        restart_files=[restart_file_west, restart_file_east],
        output_file=output_file,
        lon_min_bound=0.0,
        lon_max_bound=2.0,
        lat_min_bound=0.0,
        lat_max_bound=2.0,
        l1_resolution=1.0,
    )

    with xr.open_dataset(output_file) as merged:
        assert list(merged["lon"].values) == [0.5, 1.5]
        assert list(merged["lat"].values) == [1.5, 0.5]
        assert merged["L1_state"].sel(lon=0.5, lat=1.5).item() == 1.0
        assert merged["L1_state"].sel(lon=1.5, lat=1.5).item() == 2.0
        assert merged["L1_state"].sel(lon=0.5, lat=0.5).item() == 3.0
        assert merged["L1_state"].sel(lon=1.5, lat=0.5).item() == 4.0
        assert merged.attrs["xllcorner_L1"] == 0.0
        assert merged.attrs["yllcorner_L1"] == 0.0
        assert merged.attrs["ncols_L1"] == 2
        assert merged.attrs["nrows_L1"] == 2


def test_merge_mhm_restart_files_masks_each_tile_with_tile_mask(tmp_path):
    tile_path = tmp_path / "slice_0_0"
    restart_file = tile_path / "output" / "mHM_restart_001.nc"
    output_file = tmp_path / "mHM_restart_001.nc"
    restart_file.parent.mkdir(parents=True)
    ds = xr.Dataset(
        data_vars={
            "L1_state": (
                ("ncols1", "nrows1"),
                np.array([[1.0, 2.0], [3.0, 4.0]]),
            )
        },
        attrs={
            "xllcorner_L1": 0.0,
            "yllcorner_L1": 0.0,
            "cellsize_L1": 1.0,
            "ncols_L1": 2,
            "nrows_L1": 2,
        },
    )
    mask = xr.Dataset(
        data_vars={
            "land_mask": (
                ("lat", "lon"),
                np.array([[1, 0], [1, 1]]),
            )
        },
        coords={
            "lat": np.array([1.5, 0.5]),
            "lon": np.array([0.5, 1.5]),
        },
    )
    ds.to_netcdf(restart_file)
    mask.to_netcdf(tile_path / "mask_tile.nc")

    merge_mhm_restart_files(
        restart_files=[restart_file],
        output_file=output_file,
        lon_min_bound=0.0,
        lon_max_bound=2.0,
        lat_min_bound=0.0,
        lat_max_bound=2.0,
        l1_resolution=1.0,
        mask_var="land_mask",
    )

    with xr.open_dataset(output_file) as merged:
        assert bool(np.isnan(merged["L1_state"].sel(lon=1.5, lat=1.5)).item())
        assert merged["L1_state"].sel(lon=0.5, lat=1.5).item() == 1.0
        assert merged["L1_state"].sel(lon=0.5, lat=0.5).item() == 3.0
        assert merged["L1_state"].sel(lon=1.5, lat=0.5).item() == 4.0


def test_merge_mhm_restart_files_masks_final_result(tmp_path):
    restart_file = tmp_path / "tile_restart.nc"
    output_file = tmp_path / "mHM_restart_001.nc"
    ds = xr.Dataset(
        data_vars={
            "L1_state": (
                ("lat", "lon"),
                np.array([[1.0, 2.0], [3.0, 4.0]]),
            ),
            "scalar_state": ((), 5.0),
        },
        coords={
            "lat": np.array([1.5, 0.5]),
            "lon": np.array([0.5, 1.5]),
        },
    )
    mask_ds = xr.Dataset(
        data_vars={
            "land_mask": (
                ("lat", "lon"),
                np.array([[1, 0], [1, 1]]),
            )
        },
        coords={
            "lat": np.array([1.5, 0.5]),
            "lon": np.array([0.5, 1.5]),
        },
    )
    ds.to_netcdf(restart_file)

    merge_mhm_restart_files(
        restart_files=[restart_file],
        output_file=output_file,
        lon_min_bound=0.0,
        lon_max_bound=2.0,
        lat_min_bound=0.0,
        lat_max_bound=2.0,
        l1_resolution=1.0,
        mask_ds=mask_ds,
        mask_var="land_mask",
    )

    with xr.open_dataset(output_file) as merged:
        assert bool(np.isnan(merged["L1_state"].sel(lat=1.5, lon=1.5)).item())
        assert merged["L1_state"].sel(lat=0.5, lon=1.5).item() == 4.0
        assert merged["scalar_state"].item() == 5.0


def test_merge_mhm_restart_files_snaps_final_mask_coordinate_roundoff(tmp_path):
    restart_file = tmp_path / "tile_restart.nc"
    output_file = tmp_path / "mHM_restart_001.nc"
    ds = xr.Dataset(
        data_vars={
            "L1_state": (
                ("lat", "lon"),
                np.array([[1.0, 2.0], [3.0, 4.0]]),
            )
        },
        coords={
            "lat": np.array([1.5, 0.5]),
            "lon": np.array([0.5, 1.5]),
        },
    )
    mask_ds = xr.Dataset(
        data_vars={
            "land_mask": (
                ("lat", "lon"),
                np.array([[1, 0], [1, 1]]),
            )
        },
        coords={
            "lat": np.array([1.5000000000000002, 0.5000000000000002]),
            "lon": np.array([0.5000000000000001, 1.5000000000000002]),
        },
    )
    ds.to_netcdf(restart_file)

    merge_mhm_restart_files(
        restart_files=[restart_file],
        output_file=output_file,
        lon_min_bound=0.0,
        lon_max_bound=2.0,
        lat_min_bound=0.0,
        lat_max_bound=2.0,
        l1_resolution=1.0,
        mask_ds=mask_ds,
        mask_var="land_mask",
    )

    with xr.open_dataset(output_file) as merged:
        assert bool(np.isnan(merged["L1_state"].sel(lat=1.5, lon=1.5)).item())
        assert merged["L1_state"].sel(lat=0.5, lon=1.5).item() == 4.0
        assert merged.attrs["nCells_L1"] == 3


def test_merge_mhm_restart_files_combines_final_masks_on_target_grid(tmp_path):
    restart_file = tmp_path / "tile_restart.nc"
    output_file = tmp_path / "mHM_restart_001.nc"
    ds = xr.Dataset(
        data_vars={
            "L1_state": (
                ("lat", "lon"),
                np.array([[1.0, 2.0], [3.0, 4.0]]),
            )
        },
        coords={
            "lat": np.array([1.5, 0.5]),
            "lon": np.array([0.5, 1.5]),
        },
    )
    mask_west = xr.Dataset(
        data_vars={
            "land_mask": (
                ("lat", "lon"),
                np.array([[1, 0], [0, 0]]),
            )
        },
        coords={
            "lat": np.array([1.5, 0.5]),
            "lon": np.array([0.5, 1.5]),
        },
    )
    mask_east = xr.Dataset(
        data_vars={
            "land_mask": (
                ("lat", "lon"),
                np.array([[0, 0], [0, 1]]),
            )
        },
        coords={
            "lat": np.array([1.5, 0.5]),
            "lon": np.array([0.5, 1.5]),
        },
    )
    ds.to_netcdf(restart_file)

    merge_mhm_restart_files(
        restart_files=[restart_file],
        output_file=output_file,
        lon_min_bound=0.0,
        lon_max_bound=2.0,
        lat_min_bound=0.0,
        lat_max_bound=2.0,
        l1_resolution=1.0,
        mask_ds=[mask_west, mask_east],
        mask_var="land_mask",
    )

    with xr.open_dataset(output_file) as merged:
        assert merged["L1_state"].sel(lat=1.5, lon=0.5).item() == 1.0
        assert bool(np.isnan(merged["L1_state"].sel(lat=1.5, lon=1.5)).item())
        assert bool(np.isnan(merged["L1_state"].sel(lat=0.5, lon=0.5)).item())
        assert merged["L1_state"].sel(lat=0.5, lon=1.5).item() == 4.0
        assert merged.attrs["nCells_L1"] == 2
