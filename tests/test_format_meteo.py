"""Tests for format-meteo: its CLI arguments, hourly to daily formatting and the time merge."""

import argparse

import click
import numpy as np
import pytest
import xarray as xr

from mhm_tools._cli import _format_meteo
from mhm_tools.common.esri_grid import write_header
from mhm_tools.common.merge import write_time_block
from mhm_tools.common.netcdf import NetcdfCompression
from mhm_tools.pre.format_meteo import MeteoFolderSpec, format_meteo_data

HEADER = {
    "ncols": 2,
    "nrows": 2,
    "xllcorner": 9.5,
    "yllcorner": 49.5,
    "cellsize": 1.0,
    "nodata_value": -9999.0,
}


def _write_hourly_precipitation(input_file, first, hours):
    """Write `hours` hourly precipitation steps in mm from `first`, each step's value its index."""
    times = np.datetime64(first) + np.arange(hours) * np.timedelta64(1, "h")
    values = np.broadcast_to(np.arange(hours, dtype="float64")[:, None, None], (hours, 2, 2))
    xr.Dataset(
        {"tp": (("time", "lat", "lon"), values.copy(), {"units": "mm"})},
        coords={"time": times, "lat": [51.0, 50.0], "lon": [10.0, 11.0]},
    ).to_netcdf(input_file)


def test_the_cli_requires_the_grid_and_an_existing_input_file(tmp_path):
    """Refuse a call without the target grid, and an input file that does not exist."""
    parser = argparse.ArgumentParser()
    _format_meteo.add_args(parser)
    arguments = [
        "-i", str(tmp_path / "tp.nc"), "-g", "*.nc", "-v", "tp", "--output-var", "pre",
        "--input-temporal-resolution", "hourly", "--output-temporal-resolution", "daily",
        "-u", "mm", "-s", "EPSG:4326", "-r", "EPSG:4326", "-o", str(tmp_path),
    ]
    with pytest.raises(SystemExit):
        parser.parse_args(arguments)
    args = parser.parse_args([*arguments, "-d", "header.txt", "-x", "9"])
    assert args.grid_file == "header.txt"
    assert args.compression == 9
    with pytest.raises(click.UsageError, match="--input-file"):
        _format_meteo.run(args)


def test_hourly_precipitation_is_summed_to_complete_days_on_the_grid(tmp_path):
    """Sum 50 hourly steps to the two complete days inside them, compressed as asked."""
    _write_hourly_precipitation(tmp_path / "tp.nc", "2000-12-31T23", 50)
    write_header(tmp_path / "header.txt", HEADER)
    spec = MeteoFolderSpec(
        "pre", tmp_path, crs="EPSG:4326", pattern="tp.nc", temporal_resolution="Hourly",
        output_temporal_resolution="Daily", aggregation="Sum", unit="mm", variable="tp",
    )
    count = format_meteo_data(
        spec, tmp_path / "out" / "pre.nc", tmp_path / "header.txt", grid_crs="EPSG:4326",
        compression=NetcdfCompression(complevel=9),
    )
    assert count == 2
    with xr.open_dataset(tmp_path / "out" / "pre.nc") as ds:
        assert ds["pre"].shape == (2, 2, 2)
        assert ds["pre"].encoding["complevel"] == 9
        np.testing.assert_allclose(ds["pre"][:, 0, 0], [sum(range(1, 25)), sum(range(25, 49))])
        np.testing.assert_array_equal(
            ds["time"].values, np.array(["2001-01-01", "2001-01-02"], dtype="datetime64[ns]")
        )


def test_time_blocks_are_merged_into_one_file(tmp_path):
    """Append a second block after the first, with its time and bounds."""
    starts = np.array(["2001-01-01", "2001-01-02", "2001-01-03"], dtype="datetime64[ns]")
    bounds = np.column_stack((starts, starts + np.timedelta64(1, "D")))

    def block(steps):
        return xr.Dataset(
            {
                "pre": (("time", "lat", "lon"), np.full((len(steps), 1, 1), float(steps[0]))),
                "time_bnds": (("time", "bnds"), bounds[steps]),
            },
            coords={"time": starts[steps], "lat": [50.0], "lon": [10.0]},
        )

    output_file = tmp_path / "pre.nc"
    handle = write_time_block(block([0]), "pre", output_file, 0)
    handle = write_time_block(block([1, 2]), "pre", output_file, 1, handle)
    handle.close()
    with xr.open_dataset(output_file) as ds:
        np.testing.assert_array_equal(ds["time"].values, starts)
        np.testing.assert_array_equal(ds["time_bnds"].values, bounds)
        np.testing.assert_allclose(ds["pre"][:, 0, 0], [0.0, 1.0, 1.0])
