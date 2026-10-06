"""Tests for create-catchment command-line arguments."""

import argparse

import pytest

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
