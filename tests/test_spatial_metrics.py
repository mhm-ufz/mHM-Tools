"""Tests for spatial metric helpers."""

import numpy as np
import pandas as pd
import pytest

from mhm_tools.common.metrics import metrics_handler, tsm
from mhm_tools.common.metrics.mspaef import MSPAEF
from mhm_tools.common.metrics.waspaef import WASPAEF


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
    df4 = pd.read_csv(tmp_path / "waspaef.csv", index_col=0)
    assert "waspaef" in df4.columns
    assert "comb" not in df4.columns
    df5 = pd.read_csv(tmp_path / "mspaef.csv", index_col=0)
    assert "mspaef" in df5.columns
    assert "comb" not in df5.columns


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


def test_create_results_csv_accepts_WASPAEF(tmp_path):
    """Result CSV can write WASPAEF components."""
    m1 = np.arange(12, dtype=float).reshape(3, 2, 2)
    m2 = m1.copy()

    metrics_handler.create_results_csv(
        map1=m1,
        map2=m2,
        ds1_name="input",
        ds2_name="ref",
        out_dir=tmp_path,
        metric="WASPAEF",
    )

    df = pd.read_csv(tmp_path / "waspaef.csv", index_col=0)
    assert "waspaef" in df.columns
    assert "rho" in df.columns
    assert "sigma" in df.columns
    assert "wd" in df.columns
    assert np.isclose(df.loc[0, "waspaef"], 0.0)
    assert np.isclose(df.loc[0, "rho"], 1.0)
    assert np.isclose(df.loc[0, "sigma"], 1.0)
    assert np.isclose(df.loc[0, "wd"], 0.0)


def test_waspaef_uses_original_values_for_wasserstein_distance():
    """WASPAEF WD uses sorted original values and captures additive bias."""
    m1 = np.arange(1, 10, 1)
    m2 = np.arange(2, 11, 1)

    waspaef, rho, sigma, wd = WASPAEF(m1, m2)

    assert np.isclose(wd, 1)
    assert np.isclose(rho, 1.0)
    assert np.isclose(sigma, 1.0)
    assert np.isclose(waspaef, 1)


def test_create_results_csv_accepts_mspaef(tmp_path):
    """Result CSV can write MSPAEF components."""
    m1 = np.arange(12, dtype=float).reshape(3, 2, 2)
    m2 = m1.copy()

    metrics_handler.create_results_csv(
        map1=m1,
        map2=m2,
        ds1_name="input",
        ds2_name="ref",
        out_dir=tmp_path,
        metric="mspaef",
    )

    df = pd.read_csv(tmp_path / "mspaef.csv", index_col=0)
    assert "mspaef" in df.columns
    assert "nrmse" in df.columns
    assert "sigma" in df.columns
    assert "sigma_error" in df.columns
    assert "mean_bias" in df.columns
    assert "rho" in df.columns
    assert np.isclose(df.loc[0, "mspaef"], 1.0)
    assert np.isclose(df.loc[0, "nrmse"], 0.0)
    assert np.isclose(df.loc[0, "sigma"], 1.0)
    assert np.isclose(df.loc[0, "sigma_error"], 0.0)
    assert np.isclose(df.loc[0, "mean_bias"], 0.0)
    assert np.isclose(df.loc[0, "rho"], 1.0)


def test_mspaef_uses_observed_iqr_for_bias_terms():
    """MSPAEF normalizes RMSE and mean bias by the observed IQR."""
    m1 = np.array([1.0, 2.0, 3.0])
    m2 = np.array([0.0, 1.0, 2.0])

    mspaef, nrmse, sigma, sigma_error, mean_bias, rho = MSPAEF(m1, m2)

    assert np.isclose(nrmse, 1.0)
    assert np.isclose(sigma, 1.0)
    assert np.isclose(sigma_error, 0.0)
    assert np.isclose(mean_bias, 1.0)
    assert np.isclose(rho, 1.0)
    assert np.isclose(mspaef, 0.2928932188134524)


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
