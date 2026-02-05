import numpy as np

from mhm_tools.common import spatial_metrics as sm


def test_filter_nan_removes_pairs():
    s = np.array([1.0, np.nan, 3.0])
    o = np.array([1.0, 2.0, np.nan])
    s_clean, o_clean = sm.filter_nan(s, o)
    assert np.allclose(s_clean, np.array([1.0]))
    assert np.allclose(o_clean, np.array([1.0]))


def test_objective_functions_spearman_only():
    s = np.array([1.0, 2.0, 3.0])
    o = np.array([1.0, 2.0, 3.0])
    res = sm.objective_functions(s, o, metrics=["spearman"], param="test")
    assert "test-gamma" in res
    assert np.isclose(res["test-gamma"], 1.0)


def test_norm_deviation_shape_and_values():
    data = np.array(
        [
            [[1.0, 2.0], [3.0, 4.0]],
            [[2.0, 2.0], [2.0, 2.0]],
        ]
    )
    out = sm.norm_deviation(data)
    assert out.shape == data.shape
    mean_t0 = np.nanmean(data[0])
    expected_t0 = data[0] - mean_t0 / mean_t0
    assert np.allclose(out[0], expected_t0)


def test_calculate_objectives_for_gridded_data_keys():
    m1 = np.random.RandomState(0).rand(3, 2, 2)
    m2 = np.random.RandomState(1).rand(3, 2, 2)
    res = sm.calculate_objectives_for_gridded_data(m1, m2, "a", "b")
    for key in [
        "general-beta",
        "spatial-alpha",
        "spatial-gamma",
        "temporal-alpha",
        "temporal-gamma",
        "comb",
    ]:
        assert key in res
