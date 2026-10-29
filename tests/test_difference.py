"""Tests for the sign of difference and relative-difference."""

import numpy as np
import pandas as pd
import xarray as xr

from mhm_tools.post import difference, relative_difference


def write_single_time_field(file, value):
    """Write a 2 x 2 field of one value with a single time step.

    Args:
        file (Path): NetCDF file to write.
        value (float): Value of every cell.
    """
    xr.Dataset(
        {"et": (("time", "lat", "lon"), np.full((1, 2, 2), value))},
        coords={
            "time": pd.date_range("2000-01-01", periods=1),
            "lat": [50.5, 50.0],
            "lon": [10.0, 10.5],
        },
    ).to_netcdf(file)


def test_difference_is_model_minus_reference(tmp_path, monkeypatch):
    """Give a positive difference where the model is higher than the reference."""
    write_single_time_field(tmp_path / "ref.nc", 2.0)
    write_single_time_field(tmp_path / "mod.nc", 3.0)
    plotted = {}
    monkeypatch.setattr(
        difference, "plot_map", lambda data, **_kwargs: plotted.update(data=data)
    )

    difference.calc_diff(
        ref_input_dir=str(tmp_path),
        mod_input_dir=str(tmp_path),
        reference_pattern="ref.nc",
        model_pattern="mod.nc",
        ref_var="et",
        mod_var="et",
        plot=difference.PlotOptions(colorbar_label="", title=""),
        output=difference.OutputOptions(
            output_dir=str(tmp_path),
            output_file_png="difference.png",
            save_ncfile=False,
            output_file_nc="difference.nc",
        ),
    )

    np.testing.assert_allclose(plotted["data"].values, 1.0)


def test_relative_difference_is_positive_where_model_is_higher(tmp_path, monkeypatch):
    """Give +50 % where the model is 3 and the reference 2."""
    write_single_time_field(tmp_path / "ref.nc", 2.0)
    write_single_time_field(tmp_path / "mod.nc", 3.0)
    plotted = {}
    monkeypatch.setattr(
        relative_difference,
        "plot_map",
        lambda data, **_kwargs: plotted.update(data=data),
    )

    relative_difference.calc_rel_diff(
        ref_input_dir=str(tmp_path),
        mod_input_dir=str(tmp_path),
        reference_pattern="ref.nc",
        model_pattern="mod.nc",
        ref_var="et",
        mod_var="et",
        save_ncfile=False,
        output_file_nc="relative_difference.nc",
        colorbar_label="",
        title="",
        output_dir=str(tmp_path),
        output_file_png="relative_difference.png",
    )

    np.testing.assert_allclose(plotted["data"].values, 50.0)
