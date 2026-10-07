import argparse

import numpy as np
import pytest
import xarray as xr

from mhm_tools.common.cli_utils import (
    add_netcdf_compression_args,
    get_coords,
    get_coords_from_mask,
    get_netcdf_compression,
    normalize_cli_sequence,
    parse_lonlatbox,
)
from mhm_tools.common.netcdf import NO_QUANTIZATION


def test_normalize_cli_sequence_splits_repeated_and_comma_separated_values():
    assert normalize_cli_sequence(["a,b", " c "]) == ["a", "b", "c"]


def test_normalize_cli_sequence_returns_none_for_none_or_empty():
    assert normalize_cli_sequence(None) is None
    assert normalize_cli_sequence([]) is None
    assert normalize_cli_sequence([" ", ","]) is None


def test_get_coords_from_mask_with_bounds(tmp_path):
    # build simple lon/lat coords with explicit bounds variables
    lon = np.array([0.0, 1.0, 2.0])
    lat = np.array([10.0, 11.0])
    lon_bnds = np.column_stack([lon - 0.5, lon + 0.5])
    lat_bnds = np.column_stack([lat - 0.5, lat + 0.5])
    mask_vals = np.ones((lat.size, lon.size), dtype=float)

    ds = xr.Dataset(
        data_vars={"mask": (("lat", "lon"), mask_vals)},
        coords={
            "lon": ("lon", lon, {"bounds": "lon_bnds"}),
            "lat": ("lat", lat, {"bounds": "lat_bnds"}),
            "lon_bnds": (("lon", "bnds"), lon_bnds),
            "lat_bnds": (("lat", "bnds"), lat_bnds),
        },
    )

    fn = tmp_path / "mask.nc"
    ds.to_netcdf(fn)

    lon_min, lon_max, lat_min, lat_max, mask_da = get_coords_from_mask(str(fn))

    assert lon_min == float(lon_bnds.min())
    assert lon_max == float(lon_bnds.max())
    assert lat_min == float(lat_bnds.min())
    assert lat_max == float(lat_bnds.max())

    # returned DataArray should equal the dataset variable
    xr.testing.assert_equal(mask_da, ds["mask"])


def test_get_coords_from_mask_snaps_bound_roundoff(tmp_path):
    resolution = 1.0 / 240.0
    lon = np.array([4.0 + resolution / 2])
    lat = np.array([-60.0 + resolution / 2])
    ds = xr.Dataset(
        data_vars={"mask": (("lat", "lon"), np.ones((1, 1), dtype=float))},
        coords={
            "lon": ("lon", lon, {"bounds": "lon_bnds"}),
            "lat": ("lat", lat, {"bounds": "lat_bnds"}),
            "lon_bnds": (
                ("lon", "bnds"),
                np.array([[4.0, 4.0 + resolution]]),
            ),
            "lat_bnds": (
                ("lat", "bnds"),
                np.array([[-60.0, -55.99999999999193]]),
            ),
        },
        attrs={"spatial_resolution": resolution},
    )

    fn = tmp_path / "mask.nc"
    ds.to_netcdf(fn)

    lon_min, lon_max, lat_min, lat_max, _mask_da = get_coords_from_mask(str(fn))

    assert lon_min == 4.0
    assert lon_max == 4.0 + resolution
    assert lat_min == -60.0
    assert lat_max == -56.0


def _compression_parser():
    parser = argparse.ArgumentParser(add_help=False)
    add_netcdf_compression_args(parser)
    return parser


def test_add_netcdf_compression_args_creates_its_own_group():
    parser = _compression_parser()

    titles = [group.title for group in parser._action_groups]
    assert "netcdf output" in titles
    group = next(g for g in parser._action_groups if g.title == "netcdf output")
    declared = {option for a in group._group_actions for option in a.option_strings}
    assert {
        "--compression",
        "--no-shuffle",
        "--significant-digits",
        "--quantize-mode",
    } <= declared


def test_add_netcdf_compression_args_reuses_a_given_group():
    parser = argparse.ArgumentParser(add_help=False)
    optional = parser.add_argument_group("optional arguments")
    add_netcdf_compression_args(optional)

    titles = [group.title for group in parser._action_groups]
    assert "netcdf output" not in titles
    declared = {option for a in optional._group_actions for option in a.option_strings}
    assert "--compression" in declared


def test_get_netcdf_compression_returns_none_without_any_option():
    # None is what makes the writers keep an input file's compression
    assert get_netcdf_compression(_compression_parser().parse_args([])) is None


def test_get_netcdf_compression_returns_none_for_a_namespace_without_the_options():
    assert get_netcdf_compression(argparse.Namespace()) is None


@pytest.mark.parametrize(
    ("argv", "complevel", "shuffle", "significant_digits"),
    [
        (["--compression", "9"], 9, None, None),
        (["--compression", "0"], 0, None, None),
        (["--significant-digits", "4"], None, None, 4),
        (["--significant-digits", "0"], None, None, NO_QUANTIZATION),
        (["--no-shuffle"], None, False, None),
    ],
)
def test_get_netcdf_compression_reads_each_option(
    argv, complevel, shuffle, significant_digits
):
    compression = get_netcdf_compression(_compression_parser().parse_args(argv))

    assert compression is not None
    assert compression.complevel == complevel
    assert compression.shuffle is shuffle
    assert compression.significant_digits == significant_digits


def test_get_netcdf_compression_carries_the_quantize_mode():
    args = _compression_parser().parse_args(
        ["--significant-digits", "3", "--quantize-mode", "GranularBitRound"]
    )
    compression = get_netcdf_compression(args)

    assert compression.quantize_mode == "GranularBitRound"
    assert compression.significant_digits == 3


@pytest.mark.parametrize(
    "argv",
    [
        ["--compression", "12"],
        ["--compression", "-1"],
        ["--quantize-mode", "nonsense"],
    ],
)
def test_add_netcdf_compression_args_rejects_invalid_values(argv):
    with pytest.raises(SystemExit):
        _compression_parser().parse_args(argv)


@pytest.mark.parametrize(
    ("lonlatbox", "expected"),
    [
        ("5,15,45,55", (5.0, 15.0, 45.0, 55.0, None)),
        ("5,15,45,55,0.0625", (5.0, 15.0, 45.0, 55.0, 0.0625)),
    ],
)
def test_parse_lonlatbox_takes_four_bounds_and_an_optional_resolution(
    lonlatbox, expected
):
    """Return the bounds, and the L0 resolution only when a fifth value is given."""
    assert parse_lonlatbox(lonlatbox) == expected


@pytest.mark.parametrize("lonlatbox", ["5,15,45", "5,15,45,55,0.0625,1"])
def test_parse_lonlatbox_rejects_other_value_counts(lonlatbox):
    """Refuse a lonlatbox with neither four nor five values, naming the form."""
    with pytest.raises(ValueError, match="lon_min,lon_max,lat_min,lat_max"):
        parse_lonlatbox(lonlatbox)


@pytest.mark.parametrize("lonlatbox", ["5,15,45,55", "5,15,45,55,0.0625"])
def test_get_coords_reads_the_same_bounds_from_four_and_five_values(lonlatbox):
    """Take the bounds of a lonlatbox with or without an L0 resolution."""
    assert get_coords(lonlatbox) == (5.0, 15.0, 45.0, 55.0, None)
