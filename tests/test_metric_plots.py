import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from mhm_tools._cli import _metric_plots
from mhm_tools._cli._main import _COMMAND_GROUPS
from mhm_tools.common import plotter
from mhm_tools.post import discharge_evaluation, metric_plots


def test_get_metric_values_by_input_recurses_and_extracts_exact_columns(tmp_path):
    """Read exact metric columns from nested CSV files."""
    first_dir = tmp_path / "first"
    second_dir = tmp_path / "second"
    (first_dir / "nested").mkdir(parents=True)
    second_dir.mkdir()
    pd.DataFrame(
        {
            "alpha": [1.0, "bad", 2.0],
            "avg_alpha": [10.0, 11.0, np.inf],
        }
    ).to_csv(first_dir / "nested" / "metrics.csv", index=False)
    pd.DataFrame({"alpha": [3.0], "avg_alpha": [12.0]}).to_csv(
        second_dir / "metrics.csv", index=False
    )

    values = metric_plots.get_metric_values_by_input(
        input_paths=[first_dir, second_dir],
        input_names=["first", "second"],
        variables=["alpha", "avg_alpha"],
    )

    assert np.allclose(values["alpha"]["first"], [1.0, 2.0])
    assert np.allclose(values["alpha"]["second"], [3.0])
    assert np.allclose(values["avg_alpha"]["first"], [10.0, 11.0])
    assert np.allclose(values["avg_alpha"]["second"], [12.0])


def test_get_metric_values_by_input_groups_wildcard_matches(tmp_path):
    """Combine all CSV files below one wildcard input as one plot group."""
    for domain, value in [("GRDC_A", 0.1), ("GRDC_B", 0.2)]:
        input_dir = tmp_path / "exp1" / domain / "eval" / "crossvalidation" / "set_01"
        input_dir.mkdir(parents=True)
        pd.DataFrame({"kge": [value]}).to_csv(input_dir / "metrics.csv", index=False)

    input_pattern = tmp_path / "exp1" / "GRDC*" / "eval" / "crossvalidation" / "set_01"
    values = metric_plots.get_metric_values_by_input(
        input_paths=[str(input_pattern)],
        variables=["kge"],
    )

    assert list(values["kge"]) == ["set_01"]
    assert np.allclose(values["kge"]["set_01"], [0.1, 0.2])


def test_get_metric_csv_files_rejects_missing_wildcard_matches(tmp_path):
    """Raise a clear error if a wildcard metric input matches no paths."""
    input_pattern = tmp_path / "exp1" / "GRDC*" / "eval"

    with pytest.raises(FileNotFoundError, match="No metric input paths match pattern"):
        metric_plots.get_metric_csv_files(str(input_pattern))


def test_get_metric_input_names_rejects_mismatch(tmp_path):
    """Reject input names that do not match input paths."""
    with pytest.raises(ValueError, match="--input-names contains 1 values"):
        metric_plots.get_metric_input_names(
            input_paths=[tmp_path / "one", tmp_path / "two"],
            input_names=["one"],
        )


def test_get_metric_input_names_handles_wildcards(tmp_path):
    """Derive stable labels from wildcard inputs unless explicit names are given."""
    first_pattern = tmp_path / "exp1" / "GRDC*" / "eval" / "crossvalidation" / "set_01"
    second_pattern = tmp_path / "exp2" / "GRDC*" / "eval" / "crossvalidation" / "set_01"

    assert metric_plots.get_metric_input_names([str(first_pattern)]) == ["set_01"]
    assert metric_plots.get_metric_input_names([str(tmp_path / "exp1" / "GRDC*")]) == [
        "exp1"
    ]
    assert metric_plots.get_metric_input_names(
        input_paths=[str(first_pattern), str(second_pattern)],
        input_names=["exp1_set_01", "exp2_set_01"],
    ) == ["exp1_set_01", "exp2_set_01"]

    with pytest.raises(ValueError, match="--input-names contains 1 values"):
        metric_plots.get_metric_input_names(
            input_paths=[str(first_pattern), str(second_pattern)],
            input_names=["set_01"],
        )


def test_metric_plots_cli_parses_list_arguments():
    """Parse metric-plots CLI list arguments."""
    parser = argparse.ArgumentParser()
    _metric_plots.add_args(parser)

    args = parser.parse_args(
        [
            "--input-paths",
            "africa",
            "europe",
            "--input-names",
            "Africa",
            "Europe",
            "--variables",
            "alpha",
            "avg_alpha",
            "--file-names",
            "*metrics.csv",
            "--shape-paths",
            "africa.shp",
            "europe.shp",
            "--geometry-match-mode",
            "auto",
            "--plot-types",
            "cdf",
            "violin",
            "--output-dir",
            "plots",
        ]
    )

    assert args.input_paths == ["africa", "europe"]
    assert args.input_names == ["Africa", "Europe"]
    assert args.variables == ["alpha", "avg_alpha"]
    assert args.file_names == "*metrics.csv"
    assert args.shape_paths == ["africa.shp", "europe.shp"]
    assert args.geometry_match_mode == "auto"
    assert args.plot_types == ["cdf", "violin"]


def test_metric_plots_command_registration_uses_new_name():
    """Register metric-plots without the old metric-cdf command."""
    visualization_commands = dict(
        next(
            commands
            for group, _, commands in _COMMAND_GROUPS
            if group == "visualization"
        )
    )

    assert "metric-plots" in visualization_commands
    assert "metric-cdf" not in visualization_commands


def test_write_metric_plots_calls_shared_cdf_plotter(tmp_path, monkeypatch):
    """Write one CDF output path per requested metric variable."""
    input_dir = tmp_path / "input"
    input_dir.mkdir()
    pd.DataFrame({"alpha": [1.0, 2.0], "kge": [0.5, 0.7]}).to_csv(
        input_dir / "metrics.csv", index=False
    )
    calls = []

    def fake_plot_metric_cdf_comparison(**kwargs):
        """Capture CDF plot arguments."""
        calls.append(kwargs)

    monkeypatch.setattr(
        metric_plots,
        "plot_metric_cdf_comparison",
        fake_plot_metric_cdf_comparison,
    )

    output_files = metric_plots.write_metric_plots(
        input_paths=[str(input_dir)],
        input_names=["run1"],
        variables=["alpha", "kge"],
        output_dir=tmp_path / "plots",
        plot_types=["cdf"],
    )

    assert output_files == [
        tmp_path / "plots" / "cdf_alpha.png",
        tmp_path / "plots" / "cdf_kge.png",
    ]
    assert [call["variable_name"] for call in calls] == ["alpha", "kge"]
    assert calls[0]["values_by_label"]["run1"].tolist() == [1.0, 2.0]


def test_write_metric_plots_calls_shared_violin_plotter(tmp_path, monkeypatch):
    """Write one violin output path per requested metric variable."""
    input_dir = tmp_path / "input"
    input_dir.mkdir()
    pd.DataFrame({"alpha": [1.0, 2.0]}).to_csv(input_dir / "metrics.csv", index=False)
    calls = []

    def fake_plot_metric_violin_comparison(**kwargs):
        """Capture violin plot arguments."""
        calls.append(kwargs)

    monkeypatch.setattr(
        metric_plots,
        "plot_metric_violin_comparison",
        fake_plot_metric_violin_comparison,
    )

    output_files = metric_plots.write_metric_plots(
        input_paths=[str(input_dir)],
        input_names=["run1"],
        variables=["alpha"],
        output_dir=tmp_path / "plots",
        plot_types=["violin"],
    )

    assert output_files == [tmp_path / "plots" / "violin_alpha.png"]
    assert calls[0]["variable_name"] == "alpha"
    assert calls[0]["values_by_label"]["run1"].tolist() == [1.0, 2.0]


def test_write_metric_plots_ignores_geometry_without_catchment_map(
    tmp_path, monkeypatch
):
    """Do not validate geometry options unless catchment maps are requested."""
    input_dir = tmp_path / "input"
    input_dir.mkdir()
    pd.DataFrame({"kge": [0.5]}).to_csv(input_dir / "metrics.csv", index=False)

    monkeypatch.setattr(metric_plots, "plot_metric_cdf_comparison", lambda **_: None)
    output_files = metric_plots.write_metric_plots(
        input_paths=[str(input_dir)],
        variables=["kge"],
        output_dir=tmp_path / "plots",
        plot_types=["cdf"],
        shape_paths=["shape.shp"],
        mask_paths=["mask.nc"],
    )

    assert output_files == [tmp_path / "plots" / "cdf_kge.png"]


def test_write_metric_catchment_maps_treats_empty_geometry_paths_as_none(
    tmp_path, monkeypatch
):
    """Treat Click empty tuple defaults like omitted geometry path options."""
    input_dir = tmp_path / "input"
    input_dir.mkdir()
    shape_folder = tmp_path / "shapes"
    shape_folder.mkdir()
    pd.DataFrame({"id": [1], "kge": [0.5]}).to_csv(
        input_dir / "metrics.csv", index=False
    )
    output_file = tmp_path / "plots" / "catchment_map_kge.png"

    def fake_write_catchment_median_maps(**kwargs):
        """Return a map path without requiring geospatial dependencies."""
        return [output_file]

    monkeypatch.setattr(
        metric_plots,
        "write_catchment_median_maps",
        fake_write_catchment_median_maps,
    )

    output_files = metric_plots.write_metric_catchment_maps(
        input_paths=[str(input_dir)],
        variables=["kge"],
        output_dir=tmp_path / "plots",
        shape_folder=shape_folder,
        shape_paths=(),
        mask_paths=(),
    )

    assert output_files == [output_file]


def test_plot_metric_violin_comparison_skips_empty_series(tmp_path, monkeypatch):
    """Plot only finite violin series and skip empty labels."""
    saved_files = []
    plotted_values = []

    def fake_violinplot(values, **kwargs):
        """Capture violin values."""
        plotted_values.extend(values)
        return {"bodies": []}

    fig, ax = plotter.plt.subplots()
    monkeypatch.setattr(ax, "violinplot", fake_violinplot)
    monkeypatch.setattr(
        fig,
        "savefig",
        lambda output_file, **kwargs: saved_files.append(output_file),
    )
    monkeypatch.setattr(plotter.plt, "subplots", lambda **kwargs: (fig, ax))

    plotter.plot_metric_violin_comparison(
        values_by_label={"empty": [np.nan, np.inf], "run1": [1.0, 2.0]},
        variable_name="alpha",
        output_file=tmp_path / "violin_alpha.png",
    )

    assert saved_files == [tmp_path / "violin_alpha.png"]
    assert len(plotted_values) == 1
    assert plotted_values[0].tolist() == [1.0, 2.0]


def test_discharge_plot_cdf_delegates_to_shared_plotter(tmp_path, monkeypatch):
    """Use the shared CDF plotter from the discharge CDF wrapper."""
    rows = []
    for region_id in range(1, 7):
        rows.append(
            {
                "id": int(f"{region_id}001"),
                "alpha": 1.0 + region_id / 10,
                "beta": 1.0,
                "gamma": 0.9,
                "kge": 0.8,
                "nse": 0.7,
            }
        )
    df = pd.DataFrame(rows)
    comparison_calls = []
    cdf_value_calls = []

    def fake_plot_metric_cdf_comparison(**kwargs):
        """Capture CDF comparison plot arguments."""
        comparison_calls.append(kwargs)

    def fake_plot_cdf_values(*args, **kwargs):
        """Capture CDF value plot arguments."""
        cdf_value_calls.append((args, kwargs))
        return np.asarray(args[1]), np.linspace(0, 1, len(args[1]))

    monkeypatch.setattr(
        discharge_evaluation,
        "plot_metric_cdf_comparison",
        fake_plot_metric_cdf_comparison,
    )
    monkeypatch.setattr(discharge_evaluation, "plot_cdf_values", fake_plot_cdf_values)
    monkeypatch.setattr(
        discharge_evaluation.plt, "savefig", lambda *args, **kwargs: None
    )

    discharge_evaluation.plot_cdf(df, Path(tmp_path))

    assert comparison_calls
    assert cdf_value_calls
    assert tmp_path / "cdf_alpha_global.png" in [
        call["output_file"] for call in comparison_calls
    ]


def test_discharge_plot_metric_violins_uses_repeated_rows(tmp_path, monkeypatch):
    """Create discharge violin plots from all finite repeated metric rows."""
    df = pd.DataFrame(
        {
            "id": [1, 1, 2],
            "kge": [0.1, 0.3, np.inf],
            "nse": [0.2, np.nan, 0.6],
        }
    )
    calls = []

    def fake_plot_metric_violin_comparison(**kwargs):
        """Capture discharge violin plot arguments."""
        calls.append(kwargs)

    monkeypatch.setattr(
        discharge_evaluation,
        "plot_metric_violin_comparison",
        fake_plot_metric_violin_comparison,
    )

    output_files = discharge_evaluation.plot_metric_violins(
        df,
        Path(tmp_path),
        variables=["kge", "nse", "alpha"],
    )

    assert output_files == [
        tmp_path / "violin_kge.png",
        tmp_path / "violin_nse.png",
    ]
    assert calls[0]["values_by_label"]["all stations"].tolist() == [0.1, 0.3]
    assert calls[1]["values_by_label"]["all stations"].tolist() == [0.2, 0.6]
