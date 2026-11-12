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
        tmp_path / "plots" / "metric_plots_overview.pdf",
        tmp_path / "plots" / "metric_summary.csv",
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

    assert output_files == [
        tmp_path / "plots" / "violin_alpha.png",
        tmp_path / "plots" / "metric_plots_overview.pdf",
        tmp_path / "plots" / "metric_summary.csv",
    ]
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

    assert output_files == [
        tmp_path / "plots" / "cdf_kge.png",
        tmp_path / "plots" / "metric_plots_overview.pdf",
        tmp_path / "plots" / "metric_summary.csv",
    ]


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


def test_get_lower_axis_limit_floors_extreme_low_values():
    """Cut off at -1 once the data reaches at or below it."""
    assert plotter._get_lower_axis_limit([-5.0, 0.2, 0.5]) == -1.0
    assert plotter._get_lower_axis_limit([-1.0, 0.5]) == -1.0


def test_get_lower_axis_limit_pads_slightly_when_min_above_floor():
    """Use a small pad below the data's own minimum when it never nears -1."""
    limit = plotter._get_lower_axis_limit([0.2, 0.3, 0.4])
    assert -1.0 < limit < 0.2


def test_plot_metric_violin_comparison_floors_kge_y_axis_at_constant_mean_bound(
    tmp_path, monkeypatch
):
    """Violin plots must cut the KGE y-axis off at -0.41 for extreme low outliers."""
    fig, ax = plotter.plt.subplots()
    monkeypatch.setattr(fig, "savefig", lambda output_file, **kwargs: None)
    monkeypatch.setattr(plotter.plt, "subplots", lambda **kwargs: (fig, ax))

    plotter.plot_metric_violin_comparison(
        values_by_label={"run1": [-5.0, 0.2, 0.5], "run2": [0.1, 0.3, 0.6]},
        variable_name="kge",
        output_file=tmp_path / "violin_kge.png",
    )

    assert ax.get_ylim()[0] == plotter.KGE_CONSTANT_MEAN_BOUND


def test_plot_metric_cdf_comparison_floors_kge_x_axis_at_constant_mean_bound(
    tmp_path, monkeypatch
):
    """CDF plots must cut the KGE x-axis off at -0.41 for extreme low outliers."""
    fig, ax = plotter.plt.subplots()
    monkeypatch.setattr(fig, "savefig", lambda output_file, **kwargs: None)
    monkeypatch.setattr(plotter.plt, "subplots", lambda **kwargs: (fig, ax))

    plotter.plot_metric_cdf_comparison(
        values_by_label={"run1": [-5.0, 0.2, 0.5], "run2": [0.1, 0.3, 0.6]},
        variable_name="kge",
        output_file=tmp_path / "cdf_kge.png",
    )

    assert ax.get_xlim()[0] == plotter.KGE_CONSTANT_MEAN_BOUND


def test_plot_metric_cdf_comparison_respects_explicit_x_limits(tmp_path, monkeypatch):
    """An explicit x_limits argument must override the -1 default floor."""
    fig, ax = plotter.plt.subplots()
    monkeypatch.setattr(fig, "savefig", lambda output_file, **kwargs: None)
    monkeypatch.setattr(plotter.plt, "subplots", lambda **kwargs: (fig, ax))

    plotter.plot_metric_cdf_comparison(
        values_by_label={"run1": [-5.0, 0.2, 0.5]},
        variable_name="kge",
        output_file=tmp_path / "cdf_kge.png",
        x_limits=(-0.5, 1.0),
    )

    assert ax.get_xlim() == (-0.5, 1.0)


def test_plot_metric_violin_comparison_respects_explicit_y_limits(
    tmp_path, monkeypatch
):
    """An explicit y_limits argument must override the -1 default floor."""
    fig, ax = plotter.plt.subplots()
    monkeypatch.setattr(fig, "savefig", lambda output_file, **kwargs: None)
    monkeypatch.setattr(plotter.plt, "subplots", lambda **kwargs: (fig, ax))

    plotter.plot_metric_violin_comparison(
        values_by_label={"run1": [-5.0, 0.2, 0.5]},
        variable_name="kge",
        output_file=tmp_path / "violin_kge.png",
        y_limits=(-0.5, 1.0),
    )

    assert ax.get_ylim() == (-0.5, 1.0)


def test_plot_metric_violin_comparison_excludes_out_of_window_values_from_kde(
    tmp_path, monkeypatch
):
    """An out-of-range outlier must not squash the in-range KDE shape.

    The violin's KDE bandwidth/shape is computed from every value handed to
    it, so a value far outside y_limits (never actually visible) would still
    flatten the visible portion if it weren't excluded first.
    """
    plotted_values = []

    def fake_violinplot(values, **kwargs):
        plotted_values.extend(values)
        return {"bodies": []}

    fig, ax = plotter.plt.subplots()
    monkeypatch.setattr(ax, "violinplot", fake_violinplot)
    monkeypatch.setattr(fig, "savefig", lambda output_file, **kwargs: None)
    monkeypatch.setattr(plotter.plt, "subplots", lambda **kwargs: (fig, ax))

    plotter.plot_metric_violin_comparison(
        values_by_label={"run1": [0.5, 0.6, 0.7, -50.0]},
        variable_name="kge",
        output_file=tmp_path / "violin_kge.png",
        y_limits=(-0.5, 1.0),
    )

    assert plotted_values[0].tolist() == [0.5, 0.6, 0.7]


def test_plot_metric_violin_comparison_label_keeps_full_stats_despite_window(
    tmp_path, monkeypatch
):
    """n= and median= in the label must reflect all finite values, not just
    the in-window subset used for the KDE shape."""
    fig, ax = plotter.plt.subplots()
    monkeypatch.setattr(ax, "violinplot", lambda values, **kwargs: {"bodies": []})
    monkeypatch.setattr(fig, "savefig", lambda output_file, **kwargs: None)
    monkeypatch.setattr(plotter.plt, "subplots", lambda **kwargs: (fig, ax))

    plotter.plot_metric_violin_comparison(
        values_by_label={"run1": [0.5, 0.6, 0.7, -50.0]},
        variable_name="kge",
        output_file=tmp_path / "violin_kge.png",
        y_limits=(-0.5, 1.0),
    )

    assert "n=4" in ax.get_xticklabels()[0].get_text()


def test_write_metric_plots_applies_axis_limits_to_cdf_and_violin(
    tmp_path, monkeypatch
):
    """axis_limits_by_variable must reach both the CDF x-axis and violin y-axis."""
    input_dir = tmp_path / "input"
    input_dir.mkdir()
    pd.DataFrame({"kge": [-5.0, 0.2, 0.5]}).to_csv(
        input_dir / "metrics.csv", index=False
    )
    cdf_calls, violin_calls = [], []
    monkeypatch.setattr(
        metric_plots,
        "plot_metric_cdf_comparison",
        lambda **kwargs: cdf_calls.append(kwargs),
    )
    monkeypatch.setattr(
        metric_plots,
        "plot_metric_violin_comparison",
        lambda **kwargs: violin_calls.append(kwargs),
    )

    metric_plots.write_metric_plots(
        input_paths=[str(input_dir)],
        input_names=["run1"],
        variables=["kge"],
        output_dir=tmp_path / "plots",
        plot_types=["cdf", "violin"],
        axis_limits_by_variable={"kge": (-0.5, 1.0)},
    )

    assert cdf_calls[0]["x_limits"] == (-0.5, 1.0)
    assert violin_calls[0]["y_limits"] == (-0.5, 1.0)


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

    def fake_plot_metric_cdf_comparison(**kwargs):
        """Capture CDF comparison plot arguments."""
        comparison_calls.append(kwargs)

    monkeypatch.setattr(
        discharge_evaluation,
        "plot_metric_cdf_comparison",
        fake_plot_metric_cdf_comparison,
    )
    monkeypatch.setattr(
        discharge_evaluation.plt, "savefig", lambda *args, **kwargs: None
    )

    discharge_evaluation.plot_cdf(df, Path(tmp_path))

    assert comparison_calls
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


def test_get_discharge_metric_values_by_variable_adds_wmo_regions():
    """All stations come first, then every WMO region with finite values."""
    df = pd.DataFrame({"id": [1001, 1002, 2001, 9001], "kge": [0.1, 0.3, 0.5, np.inf]})

    values_by_variable = discharge_evaluation.get_discharge_metric_values_by_variable(
        df, variables=["kge"]
    )

    assert list(values_by_variable["kge"]) == ["all stations", "Africa", "Asia"]
    assert values_by_variable["kge"]["Africa"].tolist() == [0.1, 0.3]


def test_get_discharge_metric_values_by_variable_skips_regions_without_values():
    """A region with only NaN for one metric gets no label for that metric."""
    df = pd.DataFrame({"id": [1001, 2001], "kge": [0.1, 0.5], "nse": [0.2, np.nan]})

    values_by_variable = discharge_evaluation.get_discharge_metric_values_by_variable(
        df, variables=["kge", "nse"]
    )

    assert "Asia" in values_by_variable["kge"]
    assert "Asia" not in values_by_variable["nse"]


def test_create_table_results_within_mean_factor_leaves_out_extreme_beta():
    """Stations with beta beyond the factor are left out, the limits and NaN stay."""
    results_df = pd.DataFrame(
        {"id": [1, 2, 3, 4, 5, 6], "beta": [1.0, 31.0, 0.03, 30.0, 1 / 30, np.nan]}
    )

    table_df, notes = discharge_evaluation.create_table_results_within_mean_factor(
        results_df, 30
    )

    assert table_df["id"].tolist() == [1, 4, 5, 6]
    assert notes == ["beta > 30: 1 of 6 stations", "beta < 1/30: 1 of 6 stations"]


def test_create_table_results_within_mean_factor_keeps_all_without_factor():
    """None keeps every station, a factor below 1 is refused."""
    results_df = pd.DataFrame({"id": [1, 2], "beta": [1.0, 100.0]})

    table_df, notes = discharge_evaluation.create_table_results_within_mean_factor(
        results_df, None
    )

    assert table_df["id"].tolist() == [1, 2]
    assert notes == []
    with pytest.raises(ValueError, match="at least 1"):
        discharge_evaluation.create_table_results_within_mean_factor(results_df, 0.5)


def test_create_table_results_within_mean_factor_counts_bootstrap_stations_once():
    """Several left-out samples of one station count as one station."""
    results_df = pd.DataFrame({"id": [1, 1, 1, 2], "beta": [40.0, 50.0, 1.0, 1.0]})

    table_df, notes = discharge_evaluation.create_table_results_within_mean_factor(
        results_df, 30
    )

    assert table_df["id"].tolist() == [1, 2]
    assert notes == ["beta > 30: 1 of 2 stations"]


def test_create_non_finite_metric_notes_counts_stations_per_metric():
    """Only metrics with NaN or inf values get a note; missing columns are skipped."""
    table_df = pd.DataFrame(
        {"id": [1, 2, 3], "kge": [0.1, np.nan, np.inf], "nse": [0.1, 0.2, 0.3]}
    )

    notes = discharge_evaluation.create_non_finite_metric_notes(
        table_df, ["kge", "nse", "gamma"]
    )

    assert notes == ["no finite kge (NaN/inf): 2 of 3 stations"]


def test_write_metric_plot_overview_pdf_passes_notes_to_the_table(
    tmp_path, monkeypatch
):
    """The notes reach the summary table pages; without notes they are None."""
    table_calls = []
    monkeypatch.setattr(
        plotter,
        "_write_metric_summary_table_pages",
        lambda **kwargs: table_calls.append(kwargs),
    )
    summary_rows = [{"realisation": "all stations", "variable": "kge", "value": 0.5}]

    plotter.write_metric_plot_overview_pdf(
        output_file=tmp_path / "with_notes.pdf",
        plot_files=[],
        summary_rows=summary_rows,
        notes=["beta > 30: 1 of 2 stations"],
    )
    plotter.write_metric_plot_overview_pdf(
        output_file=tmp_path / "without_notes.pdf",
        plot_files=[],
        summary_rows=summary_rows,
    )

    assert table_calls[0]["notes"] == ["beta > 30: 1 of 2 stations"]
    assert table_calls[1]["notes"] is None


@pytest.mark.parametrize(
    ("ratios", "expected_factor"),
    [
        (np.linspace(0.9, 1.1, 101), 1.25),
        (np.linspace(0.7, 1.4, 101), 1.5),
        (np.linspace(0.55, 1.9, 101), 2.0),
        (np.array([0.6, 1.0, 1.0]), 2.0),
        (np.concatenate([np.linspace(0.95, 1.05, 980), np.full(20, 20.0)]), 2.0),
    ],
    ids=[
        "within-1.25",
        "within-1.5",
        "within-2",
        "reciprocal-of-0.6",
        "2-percent-at-20",
    ],
)
def test_get_ratio_bounds_picks_the_narrowest_set(ratios, expected_factor):
    """The spread is measured on the log of the ratios, so 0.6 needs the factor 2."""
    bounds = plotter.get_ratio_bounds(ratios)

    np.testing.assert_allclose(bounds, plotter.RATIO_BOUNDS[expected_factor])


def test_create_discrete_colour_norm_with_ratio_bins():
    """The beta style keeps a ratio map above 0 and centred on 1, outliers extend."""
    rng = np.random.default_rng(0)
    # many dry cells with ratios around 20 once stretched the scale to -21.5..23.5
    ratios = np.concatenate([rng.uniform(0.6, 1.8, 900), rng.uniform(15, 23, 100)])

    _, _, bounds, extent, _ = plotter.create_discrete_colour_norm(
        ratios, bounds_type="data", **plotter.get_metric_plot_style("beta")
    )

    assert any(np.array_equal(bounds, edges) for edges in plotter.RATIO_BOUNDS.values())
    assert bounds[0] > 0
    assert (bounds.size - 1) % 2 == 1
    centre_bin = (bounds.size - 1) // 2
    assert bounds[centre_bin] < 1 < bounds[centre_bin + 1]
    assert extent == "max"


def test_write_metric_plot_overview_pdf_uses_given_columns_without_logging(
    tmp_path, monkeypatch
):
    """Given table columns replace the derived ones; log_tables=False logs nothing."""
    styled_columns = []
    logged_tables = []
    monkeypatch.setattr(
        plotter,
        "_style_metric_summary_table",
        lambda **kwargs: styled_columns.append(kwargs["columns"]),
    )
    monkeypatch.setattr(
        plotter,
        "_log_metric_summary_table",
        lambda *args, **_kwargs: logged_tables.append(args),
    )
    summary_rows = [{"variable": "SPAEF", "statistic": "avg_spaef", "value": 0.5}]

    output_file = plotter.write_metric_plot_overview_pdf(
        output_file=tmp_path / "overview.pdf",
        plot_files=[],
        summary_rows=summary_rows,
        table_columns=["variable", "statistic", "value"],
        log_tables=False,
    )

    assert output_file.is_file()
    assert styled_columns == [["variable", "statistic", "value"]]
    assert logged_tables == []
