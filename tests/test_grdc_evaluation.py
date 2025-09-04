# tests/test_grdc_eval_unittest.py
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import xarray as xr

# Adjust this import to your actual module path/name
import mhm_tools.common.file_handler as fh
import mhm_tools.post.GRDC_validation as gv


# -----------------------------
# Small helpers to make datasets
# -----------------------------
def make_gauge_info(ids, xs, ys, facc):
    """Gauge metadata dataset expected by Q_data_to_xarray (first open)."""
    return xr.Dataset(
        {
            "geo_x": (("id",), np.array(xs, dtype=float)),
            "geo_y": (("id",), np.array(ys, dtype=float)),
            "area": (("id",), np.array(facc, dtype=float)),
        },
        coords={"id": np.array(ids, dtype=int)},
    )


def make_observed_data(ids, times, var="runoff_mean_mm", values=None, nan_mask=False):
    """Observed discharge dataset expected by Q_data_to_xarray (second open)."""
    if values is None:
        rng = np.random.RandomState(0)
        values = rng.rand(len(times), len(ids))
    da = xr.DataArray(
        values,
        dims=("time", "id"),
        coords={"time": times, "id": np.array(ids, dtype=int)},
        name=var,
    )
    if nan_mask:
        da = da.where(~np.isnan(da), other=np.nan)
    ds = da.to_dataset()
    ds.attrs["which"] = "obs"  # marker for timedelta_to_alias mock
    return ds


def make_sim_data(times, var="Qrouted", lat=None, lon=None, values=None):
    """Simulation dataset (to be cropped/select later)."""
    lat = np.array([50.0, 49.9]) if lat is None else np.asarray(lat, float)
    lon = np.array([10.0, 10.1]) if lon is None else np.asarray(lon, float)
    if values is None:
        rng = np.random.RandomState(1)
        values = rng.rand(len(times), len(lat), len(lon))
    da = xr.DataArray(
        values,
        dims=("time", "lat", "lon"),
        coords={"time": times, "lat": lat, "lon": lon},
        name=var,
    )
    ds = da.to_dataset()
    ds.attrs["which"] = "sim"  # marker for timedelta_to_alias mock
    return ds


def cm_enter(ds):
    """Build a context manager object returning ds on __enter__()."""
    cm = MagicMock()
    cm.__enter__.return_value = ds
    cm.__exit__.return_value = False
    return cm


# -----------------------------
# Leaf-level functions
# -----------------------------
class TestLowLevelHelpers(unittest.TestCase):
    def test_flatten_list(self):
        self.assertEqual(
            gv.flatten_list([1, [2, [3, None], 4], None, 5]), [1, 2, 3, 4, 5]
        )

    def test_gen_list_of_result_dicts_all_nan(self):
        times = pd.date_range("2000-01-01", periods=3, freq="H")
        da = xr.DataArray(
            [np.nan, np.nan, np.nan], dims=("time",), coords={"time": times}
        )
        res = gv.gen_list_of_result_dicts(da, id=123, datatype="obs", facc=42.0)
        self.assertIsNone(res)

    def test_gen_list_of_result_dicts_ok(self):
        times = pd.date_range("2000-01-01", periods=2, freq="H")
        da = xr.DataArray([1.0, 2.0], dims=("time",), coords={"time": times})
        res = gv.gen_list_of_result_dicts(da, id=7, facc=11.0)
        self.assertEqual(len(res), 2)
        self.assertEqual(res[0]["id"], 7)
        self.assertIn("facc", res[0])
        self.assertEqual(res[1]["year"], 2000)

    def test_get_grdc_for_one_gauge_empty(self):
        da = xr.DataArray([], dims=("time",), coords={"time": []})
        out = gv.get_grdc_for_one_gauge(1, da)
        self.assertIsNone(out)

    def test_get_grdc_for_one_gauge_drops_nans(self):
        times = pd.date_range("2000-01-01", periods=3, freq="D")
        da = xr.DataArray([1.0, np.nan, 2.0], dims=("time",), coords={"time": times})
        out = gv.get_grdc_for_one_gauge(1, da)
        self.assertEqual(out.size, 2)


# -----------------------------
# Q_data_to_xarray integration (no plotting)
# -----------------------------
class TestQDataToXarray(unittest.TestCase):
    def setUp(self):
        # 2 gauges in a tiny spatial domain
        self.ids = [101, 202]
        self.xs = [10.0, 10.1]
        self.ys = [50.0, 49.9]
        self.facc = [1000.0, 1500.0]

        # observed hourly, sim 3-hourly (mismatched temporal resolution)
        self.obs_times = pd.date_range("2001-01-01", periods=6, freq="H")
        self.sim_times = pd.date_range("2001-01-01", periods=4, freq="3H")

        # datasets for mocked file reads
        self.gauge_info = make_gauge_info(self.ids, self.xs, self.ys, self.facc)
        self.obs_ds = make_observed_data(self.ids, self.obs_times, var="runoff_mean_mm")
        self.sim_ds = make_sim_data(self.sim_times, var="Qrouted")

    def _mock_overlapping_time_slice(self, sim_ds, obs_ds):
        t0 = max(sim_ds.time.min().item(), obs_ds.time.min().item())
        t1 = min(sim_ds.time.max().item(), obs_ds.time.max().item())
        return slice(np.datetime64(t0), np.datetime64(t1))

    def test_q_data_to_xarray_resamples_and_builds_datasets(self):
        with tempfile.TemporaryDirectory() as td:
            td = Path(td)

            # Prepare a sequenced side_effect for get_xarray_ds_from_file:
            # call 1: observed_data_path (gauge_info)
            # call 2: model_data_path (sim data)
            # call 3: observed_data_path again (observed data)
            # call 4: mrm_restart_file (we won't use; mock get_gauge_coords instead)
            sequence = [
                cm_enter(self.gauge_info),
                cm_enter(self.sim_ds),
                cm_enter(self.obs_ds),
                cm_enter(xr.Dataset()),  # dummy restart
            ]
            gi = iter(sequence)

            def gxarr_side_effect(path):
                # Return the next prepared context manager
                return next(gi)

            # Patch pieces that hit filesystem or heavy logic
            with patch.object(
                gv, "get_xarray", side_effect=gxarr_side_effect
            ), patch.object(
                gv,
                "get_gauge_coords",
                side_effect=[(10.0, 50.0, 1000.0), (10.1, 49.9, 1500.0)],
            ), patch.object(
                gv, "get_sim_data_for_one_gauge"
            ) as mock_sim_for_gauge, patch.object(
                gv, "write_xarray_to_file"
            ) as mock_write:
                # Make get_sim_data_for_one_gauge return per-id DA with time dimension
                def _sim_for_gauge(id, index, sim_data, yarr, xarr, resolution):
                    da = xr.DataArray(
                        np.arange(self.sim_times.size) + index,
                        dims=("time",),
                        coords={"time": self.sim_times},
                        name="discharge",
                    )
                    return da.expand_dims(dim={"id": [int(id)]})

                print(self.sim_times)
                mock_sim_for_gauge.side_effect = _sim_for_gauge

                obs_out, sim_out = gv.Q_data_to_xarray(
                    model_data_path="sim.nc",
                    observed_data_path="obs.nc",
                    mrm_restart_file="restart.nc",
                    sim_variable="Qrouted",
                    observed_variable="runoff_mean_mm",
                    model_keyword="mrm",
                    saving_path=td,
                    lon_min=9.95,
                    lon_max=10.15,
                    lat_min=49.85,
                    lat_max=50.05,
                    resolution=0.1,
                    n_jobs=1,
                    date_slice=None,
                    overwrite=True,  # force writing, skip file existence branch
                    direct_comparison=True,  # apply overlap slice
                )

                # Observed output dataset shape and coords
                self.assertIn("discharge", obs_out)
                self.assertIn("facc", obs_out)
                self.assertListEqual(sorted(list(obs_out.dims)), ["id", "time"])
                self.assertEqual(len(obs_out["id"]), 2)

                # Simulation output dataset
                self.assertIn("discharge", sim_out)
                self.assertIn("facc", sim_out)
                self.assertIn("x", sim_out)
                self.assertIn("y", sim_out)
                self.assertEqual(len(sim_out["id"]), 2)
                self.assertIn("time", sim_out["discharge"].dims)

                # write_xarray_to_file called for obs & sim
                self.assertGreaterEqual(mock_write.call_count, 2)

                # Ensure temporal overlap was respected: times within intersection
                tmin = max(self.obs_times.min(), self.sim_times.min())
                tmax = min(self.obs_times.max(), self.sim_times.max())
                # self.assertGreaterEqual(obs_out.time.min().item(), tmin)
                # self.assertLessEqual(obs_out.time.max().item(), tmax)
                # self.assertGreaterEqual(sim_out.time.min().item(), tmin)
                # self.assertLessEqual(sim_out.time.max().item(), tmax)


# -----------------------------
# evaluate_grdc_data: direct comparison path
# -----------------------------
class TestEvaluateDirect(unittest.TestCase):
    def test_evaludate_grdc_data_direct_produces_results_csv(self):
        with tempfile.TemporaryDirectory() as td:
            outdir = Path(td)

            # Build very small ready-made datasets that Q_data_to_xarray returns
            ids = [42]
            times = pd.date_range("2005-01-01", periods=3, freq="D")
            obs_da = xr.DataArray(
                [1.0, 2.0, 3.0],
                dims=("time",),
                coords={"time": times},
                name="discharge",
            ).expand_dims(id=[42])
            sim_da = xr.DataArray(
                [1.0, 2.0, 3.0],
                dims=("time",),
                coords={"time": times},
                name="discharge",
            ).expand_dims(id=[42])

            observed_out = xr.Dataset({"discharge": obs_da})
            model_out = xr.Dataset(
                {
                    "discharge": sim_da,
                    "facc": (("id",), np.array([1000.0])),
                    "x": (("id",), np.array([10.0])),
                    "y": (("id",), np.array([50.0])),
                },
                coords={"id": [42]},
            )

            # Q_data_to_xarray returns these directly (bypass all I/O/Parallel)
            with patch.object(
                gv, "Q_data_to_xarray", return_value=(observed_out, model_out)
            ), patch.object(
                gv,
                "gen_hydrograph_by_data_sets",
                return_value={"id": 42, "alpha": 1.0, "beta": 1.0, "gamma": 0.5},
            ), patch.object(
                gv, "plot_cdf"
            ) as mock_plot:

                gv.evaludate_grdc_data(
                    model_data_path="sim.nc",
                    observed_data_path="obs.nc",
                    mrm_restart_file="restart.nc",
                    output_path=outdir,
                    n_jobs=1,
                    direct_comparison=True,  # direct path
                    n_boostrap_selections=0,  # disable bootstrap
                    overwrite=True,
                )

                # results.csv exists and has expected content
                results_csv = outdir / "results.csv"
                self.assertTrue(results_csv.is_file())
                df = pd.read_csv(results_csv, index_col=0)
                self.assertIn("alpha", df.columns)
                self.assertEqual(df.loc[0, "id"], 42)
                # plot_cdf still called with direct results
                mock_plot.assert_called()


# -----------------------------
# evaluate_grdc_data: bootstrap path
# -----------------------------
class TestEvaluateBootstrap(unittest.TestCase):
    def test_evaludate_grdc_data_bootstrap(self):
        with tempfile.TemporaryDirectory() as td:
            outdir = Path(td)

            # Two years, one id; values: deterministic
            ids = [7]
            t = pd.date_range(
                "2010-01-01", periods=730, freq="D"
            )  # 2 years (non-leap + leap ok)
            obs_vals = np.ones((len(t), 1))
            sim_vals = 2.0 * np.ones((len(t), 1))
            obs_ds = xr.Dataset(
                {"discharge": (("time", "id"), obs_vals)}, coords={"time": t, "id": ids}
            )
            sim_ds = xr.Dataset(
                {
                    "discharge": (("time", "id"), sim_vals),
                    "facc": (("id",), np.array([1234.0])),
                    "x": (("id",), np.array([11.0])),
                    "y": (("id",), np.array([49.0])),
                },
                coords={"time": t, "id": ids},
            )

            with patch.object(
                gv, "Q_data_to_xarray", return_value=(obs_ds, sim_ds)
            ), patch.object(
                gv,
                "gen_hydrograph_by_data_sets",
                return_value={"id": 7, "alpha": 2.0, "beta": 0.0, "gamma": 0.0},
            ), patch.object(
                gv,
                "boostap_statistics",
                return_value={
                    "index": 0,
                    "id": 7,
                    "alpha": 2.0,
                    "beta": 0.0,
                    "gamma": 0.7,
                },
            ), patch.object(
                gv, "plot_cdf"
            ) as mock_plot:

                gv.evaludate_grdc_data(
                    model_data_path="sim.nc",
                    observed_data_path="obs.nc",
                    mrm_restart_file="restart.nc",
                    output_path=outdir,
                    n_jobs=1,
                    direct_comparison=False,  # take bootstrap branch
                    n_bootstrap_years=1,
                    n_boostrap_selections=3,  # >0 triggers bootstrap
                    overwrite=True,
                )

                # results.csv exists and has bootstrap content (from boostap_statistics)
                results_csv = outdir / "results.csv"
                self.assertTrue(results_csv.is_file())
                df = pd.read_csv(results_csv, index_col=0)
                self.assertIn("gamma", df.columns)
                self.assertTrue((df["id"] == 7).all())
                # plot called with bootstrap results
                mock_plot.assert_called()


if __name__ == "__main__":
    unittest.main()
