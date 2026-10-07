"""Tests for spatial metric helpers."""

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from mhm_tools.common.metrics import metrics_handler, tsm
from mhm_tools.common.metrics.mspaef import MSPAEF
from mhm_tools.common.metrics.rmse import calculate_root_mean_square_error_per_cell
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
    expected_t0 = (data[0] - mean_t0) / mean_t0
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
    assert "avg_spaef" not in df1.columns
    df2 = pd.read_csv(tmp_path / "spaef.csv", index_col=0)
    assert "avg_spaef" in df2.columns
    assert "comb" not in df2.columns
    df3 = pd.read_csv(tmp_path / "esp.csv", index_col=0)
    assert "avg_esp" in df3.columns
    assert "comb" not in df3.columns
    df4 = pd.read_csv(tmp_path / "waspaef.csv", index_col=0)
    assert "avg_waspaef" in df4.columns
    assert "comb" not in df4.columns
    df5 = pd.read_csv(tmp_path / "mspaef.csv", index_col=0)
    assert "avg_mspaef" in df5.columns
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
    assert "avg_spaef" in df.columns
    assert "avg_alpha" in df.columns
    assert "avg_beta" in df.columns
    assert "avg_gamma" in df.columns
    assert np.isclose(df.loc[0, "avg_spaef"], 1.0)


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
    assert "avg_esp" in df.columns
    assert "avg_rs" in df.columns
    assert "avg_gamma" in df.columns
    assert "avg_alpha" in df.columns
    assert np.isclose(df.loc[0, "avg_esp"], 1.0)
    assert np.isclose(df.loc[0, "avg_rs"], 1.0)
    assert np.isclose(df.loc[0, "avg_gamma"], 1.0)
    assert np.isclose(df.loc[0, "avg_alpha"], 1.0)


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
    assert "avg_waspaef" in df.columns
    assert "avg_rho" in df.columns
    assert "avg_sigma" in df.columns
    assert "avg_wd" in df.columns
    assert np.isclose(df.loc[0, "avg_waspaef"], 0.0)
    assert np.isclose(df.loc[0, "avg_rho"], 1.0)
    assert np.isclose(df.loc[0, "avg_sigma"], 1.0)
    assert np.isclose(df.loc[0, "avg_wd"], 0.0)


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
    assert "avg_mspaef" in df.columns
    assert "avg_nrmse" in df.columns
    assert "avg_sigma" in df.columns
    assert "avg_sigma_error" in df.columns
    assert "avg_mean_bias" in df.columns
    assert "avg_rho" in df.columns
    assert np.isclose(df.loc[0, "avg_mspaef"], 1.0)
    assert np.isclose(df.loc[0, "avg_nrmse"], 0.0)
    assert np.isclose(df.loc[0, "avg_sigma"], 1.0)
    assert np.isclose(df.loc[0, "avg_sigma_error"], 0.0)
    assert np.isclose(df.loc[0, "avg_mean_bias"], 0.0)
    assert np.isclose(df.loc[0, "avg_rho"], 1.0)


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


@pytest.mark.parametrize(
    ("metric", "expected"),
    [
        ("all", metrics_handler.ACCEPTED_RESULT_METRICS),
        ("none", ()),
        ("spaef", "SPAEF"),
        ("spaef, tsm", ("SPAEF", "TSM")),
        (["ESP", "tsm"], ("ESP", "TSM")),
        (None, "TSM"),
    ],
)
def test_normalize_results_metric_accepts_every_selection(metric, expected):
    """Resolve all, none, one metric and lists of metrics without failing."""
    assert metrics_handler.normalize_results_metric(metric) == expected


def test_normalize_results_metric_rejects_an_unknown_name():
    """Refuse a list holding a metric the result CSV does not offer."""
    with pytest.raises(ValueError, match="Unsupported result metric"):
        metrics_handler.normalize_results_metric("spaef, kge")


@pytest.mark.parametrize(
    ("metric", "expected_files"),
    [
        ("all", {"tsm.csv", "spaef.csv", "esp.csv", "waspaef.csv", "mspaef.csv"}),
        ("none", set()),
        ("esp", {"esp.csv"}),
        ("esp,tsm", {"esp.csv", "tsm.csv"}),
    ],
)
def test_create_results_csv_writes_one_file_per_selected_metric(
    tmp_path, metric, expected_files
):
    """Write exactly the CSV files of the selected metrics, none for none."""
    m1 = np.random.RandomState(2).rand(3, 2, 2)
    m2 = np.random.RandomState(3).rand(3, 2, 2)

    results = metrics_handler.create_results_csv(
        map1=m1,
        map2=m2,
        ds1_name="input",
        ds2_name="ref",
        out_dir=tmp_path,
        metric=metric,
    )

    assert {path.name for path in tmp_path.glob("*.csv")} == expected_files
    assert {f"{name.lower()}.csv" for name in results} == expected_files


def _storage_record(values):
    """Wrap a (time, lat, lon) array in a daily DataArray on a 2x2 grid."""
    values = np.asarray(values, dtype=float)
    times = pd.date_range("2004-01-01", periods=values.shape[0], freq="D")
    return xr.DataArray(
        values,
        dims=("time", "lat", "lon"),
        coords={"time": times, "lat": [10.0, 11.0], "lon": [20.0, 21.0]},
    )


def test_rmse_per_cell_matches_the_hand_computed_error():
    """Every cell holds the root of its own mean squared difference."""
    differences = np.array(
        [
            [[1.0, -1.0], [2.0, 0.0]],
            [[1.0, 1.0], [2.0, 0.0]],
            [[1.0, -3.0], [2.0, 4.0]],
            [[1.0, 3.0], [2.0, 0.0]],
        ]
    )
    observed = _storage_record(np.zeros((4, 2, 2)))
    simulated = _storage_record(differences)

    rmse = calculate_root_mean_square_error_per_cell(simulated, observed)

    assert rmse.dims == ("lat", "lon")
    np.testing.assert_allclose(rmse.values, np.sqrt((differences**2).mean(axis=0)))


def test_two_identical_records_score_zero():
    """A perfect match is an error of zero rather than an empty cell."""
    record = _storage_record(np.random.default_rng(0).normal(size=(5, 2, 2)))

    rmse = calculate_root_mean_square_error_per_cell(record, record)

    np.testing.assert_allclose(rmse.values, 0.0, atol=1e-12)


def test_a_step_missing_in_either_record_is_left_out():
    """A step only one record holds is dropped instead of scored as an error."""
    simulated_values = np.zeros((4, 2, 2))
    simulated_values[:, 0, 0] = [1.0, 1.0, 1.0, 9.0]
    simulated_values[:, 0, 1] = 2.0
    observed = _storage_record(np.zeros((4, 2, 2)))
    simulated = _storage_record(simulated_values)
    # the observation is missing where the simulation is far off
    observed[3, 0, 0] = np.nan

    rmse = calculate_root_mean_square_error_per_cell(simulated, observed)

    assert np.isclose(rmse.sel(lat=10.0, lon=20.0), 1.0)
    assert np.isclose(rmse.sel(lat=10.0, lon=21.0), 2.0)


def test_a_cell_with_too_few_pairs_stays_empty():
    """A cell holding fewer pairs than required is NaN, its neighbours are not."""
    observed = _storage_record(np.zeros((4, 2, 2)))
    simulated = _storage_record(np.ones((4, 2, 2)))
    observed[2:, 0, 0] = np.nan

    rmse = calculate_root_mean_square_error_per_cell(
        simulated, observed, min_valid_pairs=3
    )

    assert np.isnan(rmse.sel(lat=10.0, lon=20.0))
    assert np.isclose(rmse.sel(lat=10.0, lon=21.0), 1.0)


def test_min_valid_pairs_below_one_is_lifted_to_one():
    """A cell without a single pair stays empty instead of scoring zero."""
    observed = _storage_record(np.zeros((3, 2, 2)))
    simulated = _storage_record(np.ones((3, 2, 2)))
    observed[:, 0, 0] = np.nan

    default = calculate_root_mean_square_error_per_cell(simulated, observed)
    lifted = calculate_root_mean_square_error_per_cell(
        simulated, observed, min_valid_pairs=0
    )

    np.testing.assert_array_equal(np.isnan(default.values), np.isnan(lifted.values))
    assert np.isnan(lifted.sel(lat=10.0, lon=20.0))
    assert np.isclose(lifted.sel(lat=10.0, lon=21.0), 1.0)


def test_a_chunked_record_is_scored_lazily():
    """A dask backed record is scored without being computed."""
    record = _storage_record(np.arange(24).reshape(6, 2, 2)).chunk({"time": 2})

    rmse = calculate_root_mean_square_error_per_cell(record, record * 2)

    assert rmse.chunks is not None
