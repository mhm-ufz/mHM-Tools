"""Tests for spatial metric helpers."""

import numpy as np
import pandas as pd
import pytest

from mhm_tools.common.metrics import metrics_handler, tsm


def test_filter_nan_removes_pairs():
    """NaN filtering keeps only complete simulated/observed pairs."""
    s = np.array([1.0, np.nan, 3.0])
    o = np.array([1.0, 2.0, np.nan])
    s_clean, o_clean = tsm.filter_nan(s, o)
    assert np.allclose(s_clean, np.array([1.0]))
    assert np.allclose(o_clean, np.array([1.0]))


def test_objective_functions_spearman_only():
    """Spearman-only objective calculation writes gamma."""
    s = np.array([1.0, 2.0, 3.0])
    o = np.array([1.0, 2.0, 3.0])
    res = tsm.objective_functions(s, o, metrics=["spearman"], param="test")
    assert "test-gamma" in res
    assert np.isclose(res["test-gamma"], 1.0)


def test_norm_deviation_shape_and_values():
    """Normalized deviation preserves shape and expected values."""
    data = np.array(
        [
            [[1.0, 2.0], [3.0, 4.0]],
            [[2.0, 2.0], [2.0, 2.0]],
        ]
    )
    out = tsm.norm_deviation(data)
    assert out.shape == data.shape
    mean_t0 = np.nanmean(data[0])
    expected_t0 = data[0] - mean_t0 / mean_t0
    assert np.allclose(out[0], expected_t0)


def test_calculate_objectives_for_gridded_data_keys():
    """Default gridded objectives include all TSM components."""
    m1 = np.random.RandomState(0).rand(3, 2, 2)
    m2 = np.random.RandomState(1).rand(3, 2, 2)
    res = tsm.calculate_tsm_for_gridded_data(m1, m2, "a", "b")
    for key in [
        "general-beta",
        "spatial-alpha",
        "spatial-gamma",
        "temporal-alpha",
        "temporal-gamma",
        "comb",
    ]:
        assert key in res


def test_create_results_csv_defaults_to_all(tmp_path):
    """Result CSV defaults to all accepted metrics."""
    m1 = np.random.RandomState(2).rand(3, 2, 2)
    m2 = np.random.RandomState(3).rand(3, 2, 2)

    metrics_handler.create_results_csv(
        map1=m1, map2=m2, ds1_name="input", ds2_name="ref", out_dir=tmp_path
    )

    df1 = pd.read_csv(tmp_path / "tsm.csv", index_col=0)
    assert "comb" in df1.columns
    assert "spaef" not in df1.columns
    df2 = pd.read_csv(tmp_path / "spaef.csv", index_col=0)
    assert "spaef" in df2.columns
    assert "comb" not in df2.columns
    df3 = pd.read_csv(tmp_path / "esp.csv", index_col=0)
    assert "esp" in df3.columns
    assert "comb" not in df3.columns


def test_create_results_csv_accepts_spaef(tmp_path):
    """Result CSV can write SPAEF components."""
    m1 = np.arange(12, dtype=float).reshape(3, 2, 2)
    m2 = m1.copy()

    metrics_handler.create_results_csv(
        map1=m1,
        map2=m2,
        ds1_name="input",
        ds2_name="ref",
        out_dir=tmp_path,
        metric="spaef",
    )

    df = pd.read_csv(tmp_path / "spaef.csv", index_col=0)
    assert "spaef" in df.columns
    assert "alpha" in df.columns
    assert "beta" in df.columns
    assert "gamma" in df.columns
    assert np.isclose(df.loc[0, "spaef"], 1.0)


def test_create_results_csv_accepts_esp(tmp_path):
    """Result CSV can write ESP components."""
    m1 = np.arange(12, dtype=float).reshape(3, 2, 2)
    m2 = m1.copy()

    metrics_handler.create_results_csv(
        map1=m1,
        map2=m2,
        ds1_name="input",
        ds2_name="ref",
        out_dir=tmp_path,
        metric="esp",
    )

    df = pd.read_csv(tmp_path / "esp.csv", index_col=0)
    assert "esp" in df.columns
    assert "rs" in df.columns
    assert "gamma" in df.columns
    assert "alpha" in df.columns
    assert np.isclose(df.loc[0, "esp"], 1.0)
    assert np.isclose(df.loc[0, "rs"], 1.0)
    assert np.isclose(df.loc[0, "gamma"], 1.0)
    assert np.isclose(df.loc[0, "alpha"], 0.0)


def test_create_results_csv_rejects_unknown_metric(tmp_path):
    """Unknown result metrics are rejected."""
    with pytest.raises(ValueError, match="Unsupported result metric"):
        metrics_handler.create_results_csv(
            np.ones((2, 2)),
            np.ones((2, 2)),
            "input",
            "ref",
            tmp_path / "results.csv",
            metric="unknown",
        )
