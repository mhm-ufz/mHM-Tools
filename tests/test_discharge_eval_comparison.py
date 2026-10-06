import logging
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from mhm_tools.common import plotter
from mhm_tools.post import discharge_eval_comparison as dec
from mhm_tools.post import metric_plots


@pytest.fixture(autouse=True)
def _ensure_mhm_tools_logger_propagates():
    """Ensure caplog can see mhm_tools logs regardless of test order.

    Other test modules call `configure_mhm_tools_logger(...)`, which sets
    `propagate=False` and a restrictive level (e.g. ERROR) on the shared
    "mhm_tools" logger. Since it's a real singleton (not reset between test
    files/modules), whichever test ran last in the session leaves both
    settings in place - blocking WARNING records from ever reaching caplog's
    root-attached handler regardless of `caplog.set_level`.
    """
    logger = logging.getLogger("mhm_tools")
    previous_propagate = logger.propagate
    previous_level = logger.level
    logger.propagate = True
    logger.setLevel(logging.NOTSET)
    yield
    logger.propagate = previous_propagate
    logger.setLevel(previous_level)


def _write_results_csv(path, ids, kge, x=None, y=None):
    """Write a small discharge-evaluation-shaped results.csv for tests."""
    n = len(ids)
    df = pd.DataFrame(
        {
            "id": ids,
            "kge": kge,
            "x": x if x is not None else np.arange(n, dtype=float),
            "y": y if y is not None else np.arange(n, dtype=float) + 50.0,
        }
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return df


def test_prepare_discharge_metric_diff_data_computes_overlap_diff():
    """Diff only overlapping gauge ids, using reference coordinates."""
    reference_df = pd.DataFrame(
        {
            "id": [1, 2, 3],
            "kge": [0.5, 0.6, 0.7],
            "x": [10.0, 11.0, 12.0],
            "y": [50.0, 51.0, 52.0],
        }
    )
    other_df = pd.DataFrame({"id": [2, 3, 4], "kge": [0.65, 0.9, 1.0]})

    lons, lats, diff = dec._prepare_discharge_metric_diff_data(
        reference_df, other_df, "kge"
    )

    assert np.allclose(lons, [11.0, 12.0])
    assert np.allclose(lats, [51.0, 52.0])
    assert np.allclose(diff, [0.05, 0.2])


def test_prepare_discharge_metric_diff_data_requires_id_column():
    """Raise ValueError when either input lacks an 'id' column."""
    reference_df = pd.DataFrame({"id": [1], "kge": [0.5], "x": [10.0], "y": [50.0]})
    other_df = pd.DataFrame({"kge": [0.6]})

    with pytest.raises(ValueError, match="'id' column"):
        dec._prepare_discharge_metric_diff_data(reference_df, other_df, "kge")


def test_prepare_discharge_metric_diff_data_missing_variable_returns_empty(caplog):
    """Return empty arrays (with a warning) when the variable is missing."""
    reference_df = pd.DataFrame({"id": [1], "x": [10.0], "y": [50.0]})
    other_df = pd.DataFrame({"id": [1], "kge": [0.6]})

    caplog.set_level("WARNING")
    lons, lats, diff = dec._prepare_discharge_metric_diff_data(
        reference_df, other_df, "kge"
    )

    assert lons.size == 0
    assert lats.size == 0
    assert diff.size == 0
    assert "kge" in caplog.text


def test_write_discharge_metric_diff_maps_defaults_reference_to_first_input(
    monkeypatch, tmp_path
):
    """Default reference_name to the first input and skip it as an 'other' run."""
    for name, kge in [("run1", [0.5, 0.6]), ("run2", [0.6, 0.7]), ("run3", [0.4, 0.5])]:
        _write_results_csv(tmp_path / name / "results.csv", [1, 2], kge)

    calls = []
    monkeypatch.setattr(
        dec,
        "plot_discharge_metric_diff_map",
        lambda **kwargs: calls.append(kwargs) or Path("dummy.png"),
    )

    output_files = dec.write_discharge_metric_diff_maps(
        input_paths=[
            str(tmp_path / "run1"),
            str(tmp_path / "run2"),
            str(tmp_path / "run3"),
        ],
        output_dir=tmp_path / "out",
        variables=["kge"],
    )

    assert len(output_files) == 2
    reference_names = {call["reference_name"] for call in calls}
    other_names = {call["other_name"] for call in calls}
    assert reference_names == {"run1"}
    assert other_names == {"run2", "run3"}


def test_write_discharge_metric_diff_maps_honors_explicit_reference_name(
    monkeypatch, tmp_path
):
    """Use an explicitly given reference_name as the baseline."""
    for name, kge in [("run1", [0.5, 0.6]), ("run2", [0.6, 0.7])]:
        _write_results_csv(tmp_path / name / "results.csv", [1, 2], kge)

    calls = []
    monkeypatch.setattr(
        dec,
        "plot_discharge_metric_diff_map",
        lambda **kwargs: calls.append(kwargs) or Path("dummy.png"),
    )

    dec.write_discharge_metric_diff_maps(
        input_paths=[str(tmp_path / "run1"), str(tmp_path / "run2")],
        output_dir=tmp_path / "out",
        variables=["kge"],
        reference_name="run2",
    )

    assert len(calls) == 1
    assert calls[0]["reference_name"] == "run2"
    assert calls[0]["other_name"] == "run1"


def test_write_discharge_metric_diff_maps_requires_at_least_two_inputs(
    caplog, tmp_path
):
    """Skip (with a warning) when fewer than two inputs are given."""
    _write_results_csv(tmp_path / "run1" / "results.csv", [1, 2], [0.5, 0.6])

    caplog.set_level("WARNING")
    output_files = dec.write_discharge_metric_diff_maps(
        input_paths=[str(tmp_path / "run1")],
        output_dir=tmp_path / "out",
        variables=["kge"],
    )

    assert output_files == []
    assert "at least two" in caplog.text


def test_write_discharge_eval_comparison_plots_writes_cdf_and_violin(
    monkeypatch, tmp_path
):
    """End-to-end: real cdf/violin output via write_metric_plots, map-diff mocked out."""
    for name, kge in [("run1", [0.5, 0.6]), ("run2", [0.6, 0.7])]:
        _write_results_csv(tmp_path / name / "results.csv", [1, 2], kge)

    monkeypatch.setattr(dec, "write_discharge_metric_diff_maps", lambda **kwargs: [])

    output_files = dec.write_discharge_eval_comparison_plots(
        input_paths=[str(tmp_path / "run1"), str(tmp_path / "run2")],
        output_dir=tmp_path / "out",
        variables=["kge"],
        plot_types=["cdf", "violin"],
    )

    output_names = {Path(f).name for f in output_files}
    assert "cdf_kge.png" in output_names
    assert "violin_kge.png" in output_names
    assert (tmp_path / "out" / "cdf_kge.png").is_file()
    assert (tmp_path / "out" / "metric_summary.csv").is_file()


def test_write_discharge_eval_comparison_plots_defaults_to_catchment_map_with_shape_folder(
    monkeypatch, tmp_path
):
    """Default plot_types must include catchment-map once a shape_folder is given."""
    for name, kge in [("run1", [0.5, 0.6]), ("run2", [0.6, 0.7])]:
        _write_results_csv(tmp_path / name / "results.csv", [1, 2], kge)

    captured = {}

    def _fake_write_metric_plots(**kwargs):
        captured.update(kwargs)
        return []

    monkeypatch.setattr(metric_plots, "write_metric_plots", _fake_write_metric_plots)
    monkeypatch.setattr(dec, "write_discharge_metric_diff_maps", lambda **kwargs: [])
    monkeypatch.setattr(
        dec, "write_discharge_metric_diff_region_maps", lambda **kwargs: []
    )
    monkeypatch.setattr(
        dec, "write_discharge_eval_comparison_region_plots", lambda **kwargs: []
    )

    dec.write_discharge_eval_comparison_plots(
        input_paths=[str(tmp_path / "run1"), str(tmp_path / "run2")],
        output_dir=tmp_path / "out",
        variables=["kge"],
        shape_folder=str(tmp_path / "shapes"),
    )

    assert "catchment-map" in captured["plot_types"]


def test_write_discharge_eval_comparison_plots_defaults_omit_catchment_map(
    monkeypatch, tmp_path
):
    """Default plot_types must skip catchment-map without a shape/mask folder."""
    for name, kge in [("run1", [0.5, 0.6]), ("run2", [0.6, 0.7])]:
        _write_results_csv(tmp_path / name / "results.csv", [1, 2], kge)

    captured = {}

    def _fake_write_metric_plots(**kwargs):
        captured.update(kwargs)
        return []

    monkeypatch.setattr(metric_plots, "write_metric_plots", _fake_write_metric_plots)
    monkeypatch.setattr(dec, "write_discharge_metric_diff_maps", lambda **kwargs: [])
    monkeypatch.setattr(
        dec, "write_discharge_metric_diff_region_maps", lambda **kwargs: []
    )
    monkeypatch.setattr(
        dec, "write_discharge_eval_comparison_region_plots", lambda **kwargs: []
    )

    dec.write_discharge_eval_comparison_plots(
        input_paths=[str(tmp_path / "run1"), str(tmp_path / "run2")],
        output_dir=tmp_path / "out",
        variables=["kge"],
    )

    assert "catchment-map" not in captured["plot_types"]


def test_write_sanitized_results_csvs_discards_out_of_range_values(tmp_path):
    """Values outside [value_min, value_max] become NaN; other columns untouched."""
    _write_results_csv(
        tmp_path / "run1" / "results.csv",
        ids=[1, 2, 3],
        kge=[0.5, -500.0, 0.6],
        x=[10.0, 11.0, 12.0],
        y=[50.0, 51.0, 52.0],
    )

    sanitized_paths = dec._write_sanitized_results_csvs(
        input_paths=[str(tmp_path / "run1")],
        input_names=["run1"],
        file_names="results.csv",
        variables=["kge"],
        output_dir=tmp_path / "sanitized",
    )

    df = pd.read_csv(sanitized_paths[0])
    assert df["kge"].tolist()[0] == pytest.approx(0.5)
    assert np.isnan(df["kge"].tolist()[1])
    assert df["kge"].tolist()[2] == pytest.approx(0.6)
    assert df["id"].tolist() == [1, 2, 3]
    assert df["x"].tolist() == [10.0, 11.0, 12.0]


def test_write_sanitized_results_csvs_discards_kge_above_one(tmp_path, caplog):
    """kge/nse above 1 are discarded as physically impossible, even within range."""
    _write_results_csv(tmp_path / "run1" / "results.csv", ids=[1, 2], kge=[0.5, 1.3])

    caplog.set_level("WARNING")
    sanitized_paths = dec._write_sanitized_results_csvs(
        input_paths=[str(tmp_path / "run1")],
        input_names=["run1"],
        file_names="results.csv",
        variables=["kge"],
        output_dir=tmp_path / "sanitized",
    )

    df = pd.read_csv(sanitized_paths[0])
    assert df["kge"].tolist()[0] == pytest.approx(0.5)
    assert np.isnan(df["kge"].tolist()[1])
    assert "physically impossible" in caplog.text


def test_write_discharge_eval_comparison_plots_overview_pdf_includes_all_plots(
    monkeypatch, tmp_path
):
    """The overview PDF must include global and per-region PNGs, global first."""
    ids = [f"1{i:03d}" for i in range(3)] + [f"6{i:03d}" for i in range(3)]
    for name, kge in [
        ("run1", [0.5, 0.55, 0.6, 0.5, 0.55, 0.6]),
        ("run2", [0.6, 0.65, 0.7, 0.6, 0.65, 0.7]),
    ]:
        _write_results_csv(
            tmp_path / name / "results.csv",
            ids=ids,
            kge=kge,
            x=[10, 11, 12, 20, 21, 22],
            y=[10, 11, 12, 50, 51, 52],
        )

    captured = {}

    def _fake_overview_pdf(**kwargs):
        captured.update(kwargs)
        return tmp_path / "out" / "metric_plots_overview.pdf"

    monkeypatch.setattr(plotter, "write_metric_plot_overview_pdf", _fake_overview_pdf)
    monkeypatch.setattr(dec, "write_discharge_metric_diff_maps", lambda **kwargs: [])
    monkeypatch.setattr(
        dec, "write_discharge_metric_diff_region_maps", lambda **kwargs: []
    )

    output_files = dec.write_discharge_eval_comparison_plots(
        input_paths=[str(tmp_path / "run1"), str(tmp_path / "run2")],
        output_dir=tmp_path / "out",
        variables=["kge"],
        plot_types=["cdf", "violin"],
    )

    plot_files = [Path(f).name for f in captured["plot_files"]]
    global_index = plot_files.index("cdf_kge.png")
    region_index = plot_files.index("cdf_kge_region_Africa.png")
    assert global_index < region_index
    assert "violin_kge.png" in plot_files
    assert "violin_kge_region_Europe.png" in plot_files
    assert output_files.count(tmp_path / "out" / "metric_plots_overview.pdf") == 1


def test_sort_region_files_by_region_groups_all_plot_types_per_region():
    """Region files must group by region (index order), not by plot type."""
    files = [
        "cdf_kge_region_Africa.png",
        "violin_kge_region_Europe.png",
        "map_diff_kge_run2_vs_run1_region_Africa.png",
        "cdf_kge_region_Europe.png",
        "violin_kge_region_Africa.png",
        "map_diff_kge_run2_vs_run1_region_Europe.png",
    ]

    sorted_names = [f.name for f in dec._sort_region_files_by_region(files)]

    assert sorted_names == [
        "cdf_kge_region_Africa.png",
        "violin_kge_region_Africa.png",
        "map_diff_kge_run2_vs_run1_region_Africa.png",
        "cdf_kge_region_Europe.png",
        "violin_kge_region_Europe.png",
        "map_diff_kge_run2_vs_run1_region_Europe.png",
    ]


def test_write_discharge_eval_comparison_region_catchment_maps_filters_by_region(
    monkeypatch, tmp_path
):
    """Each region's catchment map call must only receive that region's rows."""
    ids = [f"1{i:03d}" for i in range(2)] + [f"6{i:03d}" for i in range(2)]
    _write_results_csv(
        tmp_path / "run1" / "results.csv", ids=ids, kge=[0.5, 0.6, 0.7, 0.8]
    )

    calls = []

    def fake_write_catchment_median_maps(**kwargs):
        calls.append(kwargs)
        return [Path("dummy.png")]

    from mhm_tools.common import catchment_maps

    monkeypatch.setattr(
        catchment_maps, "write_catchment_median_maps", fake_write_catchment_median_maps
    )

    output_files = dec.write_discharge_eval_comparison_region_catchment_maps(
        input_paths=[str(tmp_path / "run1")],
        output_dir=tmp_path / "out",
        variables=["kge"],
        shape_folder=str(tmp_path / "shapes"),
    )

    assert len(calls) == 2
    prefixes = {call["output_prefix"] for call in calls}
    assert prefixes == {"catchment_map_region_Africa", "catchment_map_region_Europe"}
    calls_by_prefix = {call["output_prefix"]: call for call in calls}
    for call in calls:
        assert len(call["metric_df"]) == 2
        assert set(call["metric_df"]["id"].astype(str)).issubset(set(ids))
    assert calls_by_prefix["catchment_map_region_Africa"]["extent"] == (
        dec._get_region_extent("Africa")
    )
    assert calls_by_prefix["catchment_map_region_Europe"]["extent"] == (
        dec._get_region_extent("Europe")
    )
    assert len(output_files) == 2


def test_write_discharge_eval_comparison_region_catchment_maps_disambiguates_multiple_inputs(
    monkeypatch, tmp_path
):
    """Output prefixes must include the input name when comparing >1 run."""
    ids = [f"1{i:03d}" for i in range(2)]
    for name, kge in [("run1", [0.5, 0.6]), ("run2", [0.6, 0.7])]:
        _write_results_csv(tmp_path / name / "results.csv", ids=ids, kge=kge)

    calls = []

    def fake_write_catchment_median_maps(**kwargs):
        calls.append(kwargs)
        return []

    from mhm_tools.common import catchment_maps

    monkeypatch.setattr(
        catchment_maps, "write_catchment_median_maps", fake_write_catchment_median_maps
    )

    dec.write_discharge_eval_comparison_region_catchment_maps(
        input_paths=[str(tmp_path / "run1"), str(tmp_path / "run2")],
        output_dir=tmp_path / "out",
        variables=["kge"],
        shape_folder=str(tmp_path / "shapes"),
    )

    prefixes = {call["output_prefix"] for call in calls}
    assert prefixes == {
        "catchment_map_region_Africa_run1",
        "catchment_map_region_Africa_run2",
    }
