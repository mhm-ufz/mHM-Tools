"""Tests for create-catchment command-line arguments."""

import argparse

import pytest

import mhm_tools.pre
from mhm_tools._cli import _create_catchment


def create_parser():
    """Create a parser containing the create-catchment arguments.

    Returns:
        An argument parser configured for create-catchment.
    """
    parser = argparse.ArgumentParser()
    _create_catchment.add_args(parser)
    return parser


def test_distance_defaults_preserve_cell_based_behavior():
    """Leave both CLI distance forms unset for internal five-cell defaulting."""
    arguments = create_parser().parse_args(["-i", "input.nc", "-o", "output"])

    assert arguments.max_distance_cells is None
    assert arguments.max_distance_m is None
    assert arguments.use_max_error


def test_meter_distance_and_no_area_delimiter_are_parsed():
    """Parse meter distance and disabled area candidate filtering."""
    arguments = create_parser().parse_args(
        [
            "-i",
            "input.nc",
            "-o",
            "output",
            "--max-distance-m",
            "1500",
            "--no-area-delimiter",
        ]
    )

    assert arguments.max_distance_m == 1500
    assert arguments.max_distance_cells is None
    assert not arguments.use_max_error


def test_distance_cli_arguments_are_mutually_exclusive():
    """Reject simultaneous cell and meter distance arguments."""
    with pytest.raises(SystemExit):
        create_parser().parse_args(
            [
                "-i",
                "input.nc",
                "-o",
                "output",
                "--max-distance-cells",
                "2",
                "--max-distance-m",
                "1000",
            ]
        )


def test_coords_are_latlon_by_default():
    """Treat the gauge coordinates as lat/lon unless the flag is given."""
    arguments = create_parser().parse_args(["-i", "input.nc", "-o", "output"])

    assert not arguments.coords_are_not_latlon


def test_coords_are_not_latlon_flag_is_parsed():
    """Mark the gauge coordinates as projected when the flag is given."""
    arguments = create_parser().parse_args(
        ["-i", "input.nc", "-o", "output", "--coords-are-not-latlon"]
    )

    assert arguments.coords_are_not_latlon


@pytest.mark.parametrize(
    ("flags", "expected_latlon"),
    [([], True), (["--coords-are-not-latlon"], False)],
)
def test_run_passes_latlon_to_create_catchment(
    monkeypatch, tmp_path, flags, expected_latlon
):
    """Hand create_catchment latlon=False only for projected coordinates."""
    received_arguments = {}

    def record_create_catchment(**kwargs):
        """Record the arguments create_catchment is called with."""
        received_arguments.update(kwargs)

    monkeypatch.setattr(mhm_tools.pre, "create_catchment", record_create_catchment)
    arguments = create_parser().parse_args(
        ["-i", "input.nc", "-o", str(tmp_path), *flags]
    )
    _create_catchment.run(arguments)

    assert received_arguments["latlon"] is expected_latlon


@pytest.mark.parametrize(
    ("lonlatbox", "expected_l0_resolution"),
    [("5,15,45,55", None), ("5,15,45,55,0.0625", 0.0625)],
)
def test_run_crops_to_a_lonlatbox_of_four_or_five_values(
    monkeypatch, tmp_path, lonlatbox, expected_l0_resolution
):
    """Crop to the bounds of either form, taking L0 only from a fifth value."""
    received_arguments = {}

    def record_create_catchment(**kwargs):
        """Record the arguments create_catchment is called with."""
        received_arguments.update(kwargs)

    monkeypatch.setattr(mhm_tools.pre, "create_catchment", record_create_catchment)
    arguments = create_parser().parse_args(
        ["-i", "input.nc", "-o", str(tmp_path), "--lonlatbox", lonlatbox]
    )
    _create_catchment.run(arguments)

    assert received_arguments["coordinate_slices"] == {
        "lat": slice(55.0, 45.0),
        "lon": slice(5.0, 15.0),
    }
    assert received_arguments["resolutions"].l0 == expected_l0_resolution
