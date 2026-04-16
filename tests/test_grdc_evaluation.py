# tests/test_grdc_eval_unittest.py
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import xarray as xr

# Adjust this import to your actual module path/name
import mhm_tools.post.discharge_evaluation as gv

# -----------------------------
# Small helpers to make datasets
# -----------------------------


def make_observed_data(
    ids, times, xs, ys, facc, var="runoff_mean_mm", values=None, nan_mask=False
):
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
    ds = xr.Dataset(
        {
            "geo_x": (("id",), np.array(xs, dtype=float)),
            "runoff_mean_mm": da,
            "geo_y": (("id",), np.array(ys, dtype=float)),
            "area": (("id",), np.array(facc, dtype=float)),
        },
        coords={"id": np.array(ids, dtype=int), "time": times},
    )
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
        times = pd.date_range("2000-01-01", periods=3, freq="h")
        da = xr.DataArray(
            [np.nan, np.nan, np.nan], dims=("time",), coords={"time": times}
        )
        res = gv.gen_list_of_result_dicts(da, id=123, datatype="obs", facc=42.0)
        self.assertIsNone(res)

    def test_gen_list_of_result_dicts_ok(self):
        times = pd.date_range("2000-01-01", periods=2, freq="h")
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
        self.obs_times = pd.date_range("2001-01-01", periods=6, freq="h")
        self.sim_times = pd.date_range("2001-01-01", periods=4, freq="3h")
        # datasets for mocked file reads
        self.obs_ds = make_observed_data(
            self.ids, self.obs_times, self.xs, self.ys, self.facc, var="runoff_mean_mm"
        )
        self.sim_ds = make_sim_data(self.sim_times, var="Qrouted")
        # Add id dimension so downstream selection by id is valid
        if "id" not in self.sim_ds.dims:
            self.sim_ds = self.sim_ds.expand_dims(id=np.array(self.ids, dtype=int))

    def _mock_overlapping_time_slice(self, sim_ds, obs_ds):
        t0 = max(sim_ds.time.min().item(), obs_ds.time.min().item())
        t1 = min(sim_ds.time.max().item(), obs_ds.time.max().item())
        return slice(np.datetime64(t0), np.datetime64(t1))

    def test_q_data_to_xarray_resamples_and_builds_datasets(self):
        with tempfile.TemporaryDirectory() as td:
            td = Path(td)

            # Prepare a sequenced side_effect for get_xarray_ds_from_file:
            # call 1: observed_data_path again (observed data)
            # call 2: model_data_path (sim data)
            # call 3: mrm_restart_file (we won't use; mock get_gauge_coords instead)
            sequence = [
                cm_enter(self.obs_ds),
                cm_enter(self.sim_ds),
                cm_enter(xr.Dataset()),  # dummy restart
            ]
            gi = iter(sequence)

            def gxarr_side_effect(path, **kwargs):
                # Return the next prepared context manager
                return next(gi)

            # Patch pieces that hit filesystem or heavy logic
            with patch.object(
                gv, "load_ds", side_effect=gxarr_side_effect
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
                def _sim_for_gauge(
                    id, index, sim_data, yarr, xarr, resolution, **kwargs
                ):
                    sim_times = pd.to_datetime(np.asarray(sim_data.time.values))
                    da = xr.DataArray(
                        np.arange(sim_times.size) + index,
                        dims=("time",),
                        coords={"time": sim_times},
                        name="discharge",
                    )
                    return da.expand_dims(dim={"id": [int(id)]})

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
                    write_input_data_cache=True,
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
                self.assertGreaterEqual(
                    pd.Timestamp(obs_out.time.min().values), pd.Timestamp(tmin)
                )
                self.assertLessEqual(
                    pd.Timestamp(obs_out.time.max().values), pd.Timestamp(tmax)
                )
                self.assertGreaterEqual(
                    pd.Timestamp(sim_out.time.min().values), pd.Timestamp(tmin)
                )
                self.assertLessEqual(
                    pd.Timestamp(sim_out.time.max().values), pd.Timestamp(tmax)
                )

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
            ) as mock_plot, patch.object(
                gv, "plot_map"
            ):

                gv.evaludate_discharge_data(
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
                # plot_cdf not called for small sample sizes
                mock_plot.assert_not_called()

    def test_evaludate_grdc_data_filters_by_min_overlapping_years(self):
        with tempfile.TemporaryDirectory() as td:
            outdir = Path(td)
            ids = [101, 202]
            times = pd.date_range("2000-01-01", "2003-12-01", freq="MS")

            obs_vals = np.ones((len(times), len(ids)), dtype=float)
            sim_vals = np.full((len(times), len(ids)), np.nan, dtype=float)
            years = times.year
            # Gauge 101 has overlap in 2000, 2001, 2002 (3 years).
            sim_vals[np.isin(years, [2000, 2001, 2002]), 0] = 2.0
            # Gauge 202 has overlap only in 2001 (1 year).
            sim_vals[years == 2001, 1] = 2.0

            obs_ds = xr.Dataset(
                {"discharge": (("time", "id"), obs_vals)},
                coords={"time": times, "id": np.array(ids, dtype=int)},
            )
            sim_ds = xr.Dataset(
                {
                    "discharge": (("time", "id"), sim_vals),
                    "facc": (("id",), np.array([1000.0, 1500.0])),
                    "x": (("id",), np.array([10.0, 10.1])),
                    "y": (("id",), np.array([50.0, 49.9])),
                },
                coords={"time": times, "id": np.array(ids, dtype=int)},
            )

            def _fake_hydrograph(**kwargs):
                return {
                    "id": int(kwargs["id"]),
                    "alpha": 1.0,
                    "beta": 1.0,
                    "gamma": 1.0,
                }

            with patch.object(
                gv, "Q_data_to_xarray", return_value=(obs_ds, sim_ds)
            ), patch.object(
                gv,
                "gen_hydrograph_by_data_sets",
                side_effect=_fake_hydrograph,
            ) as mock_hydro, patch.object(
                gv, "plot_cdf"
            ), patch.object(
                gv, "plot_map"
            ):
                gv.evaludate_discharge_data(
                    model_data_path="sim.nc",
                    observed_data_path="obs.nc",
                    output_path=outdir,
                    n_jobs=1,
                    direct_comparison=True,
                    n_boostrap_selections=0,
                    min_overlapping_years=2,
                    overwrite=True,
                )

                called_ids = [
                    int(call.kwargs["id"]) for call in mock_hydro.call_args_list
                ]
                self.assertListEqual(called_ids, [101])

                results_csv = outdir / "results.csv"
                self.assertTrue(results_csv.is_file())
                df = pd.read_csv(results_csv, index_col=0)
                self.assertListEqual(
                    sorted(df["id"].astype(int).unique().tolist()), [101]
                )


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
            ) as mock_plot, patch.object(
                gv, "plot_map"
            ):

                gv.evaludate_discharge_data(
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
                # plot_cdf not called for small sample sizes
                mock_plot.assert_not_called()


def cm_enter(ds):
    """Build a context manager object returning ds on __enter__()."""
    cm = MagicMock()
    cm.__enter__.return_value = ds
    cm.__exit__.return_value = False
    return cm


def make_observed_data(ids, times, xs, ys, facc, var="runoff_mean_mm", values=None):
    if values is None:
        values = np.arange(len(times) * len(ids)).reshape(len(times), len(ids))
    da = xr.DataArray(
        values,
        dims=("time", "id"),
        coords={"time": times, "id": np.array(ids, dtype=int)},
        name=var,
    )
    ds = xr.Dataset(
        {
            "geo_x": (("id",), np.array(xs, dtype=float)),
            "geo_y": (("id",), np.array(ys, dtype=float)),
            "area": (("id",), np.array(facc, dtype=float)),
            var: da,
        },
        coords={"id": np.array(ids, dtype=int), "time": times},
    )
    return ds


def make_scc_gauges(ids, xs, ys):
    return xr.Dataset(
        {
            "lon": (("station",), np.array(xs, dtype=float)),
            "lat": (("station",), np.array(ys, dtype=float)),
            "station": (("station",), np.array(ids, dtype=int)),
        }
    )


def write_discharge_file(path, gid, times, obs_vals, sim_vals):
    gid_str = f"{int(gid):010d}"
    ds = xr.Dataset(
        {
            f"Qsim_{gid_str}": (("time",), np.array(sim_vals, dtype=float)),
            f"Qobs_{gid_str}": (("time",), np.array(obs_vals, dtype=float)),
        },
        coords={"time": times},
    )
    ds.to_netcdf(path)


def assert_discharge_equal(ds_a, ds_b):
    da_a = ds_a["discharge"].sortby("id")
    da_b = ds_b["discharge"].sortby("id")
    if {"time", "id"}.issubset(set(da_a.dims)):
        da_a = da_a.transpose("time", "id")
    if {"time", "id"}.issubset(set(da_b.dims)):
        da_b = da_b.transpose("time", "id")
    xr.testing.assert_allclose(da_a, da_b)


class TestGRDCValidation2Consistency(unittest.TestCase):
    def test_q_data_to_xarray_consistent_across_inputs(self):
        ids = [1, 2]
        xs = [10.0, 10.1]
        ys = [50.0, 49.9]
        facc = [100.0, 200.0]
        times = pd.date_range("2001-01-01", periods=4, freq="D")
        date_slice = slice(times[1], times[-1])

        obs_vals = np.array([[1.0, 2.0], [1.5, 2.5], [2.0, 3.0], [2.5, 3.5]])
        sim_vals = np.array([[10.0, 20.0], [10.5, 20.5], [11.0, 21.0], [11.5, 21.5]])

        obs_ds = make_observed_data(
            ids, times, xs, ys, facc, var="runoff_mean_mm", values=obs_vals
        )
        sim_ds = xr.Dataset(
            {
                "Qrouted": (
                    ("time", "id"),
                    np.zeros((len(times), len(ids)), dtype=float),
                )
            },
            coords={
                "time": times,
                "id": np.array(ids, dtype=int),
            },
        )
        scc_ds = make_scc_gauges(ids, xs, ys)

        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            discharge_root = td / "discharge_inputs"
            discharge_root.mkdir()
            for gid, i in zip(ids, range(len(ids))):
                sub = discharge_root / f"gauge_{gid}"
                sub.mkdir()
                write_discharge_file(
                    sub / "discharge.nc",
                    gid,
                    times,
                    obs_vals[:, i],
                    sim_vals[:, i],
                )

            def load_ds_discharge(path, file_name=None):
                ds = xr.open_dataset(path)
                return cm_enter(ds)

            with patch.object(
                gv, "load_ds", side_effect=load_ds_discharge
            ), patch.object(gv, "write_xarray_to_file"):
                obs_d, sim_d = gv.Q_data_to_xarray(
                    model_data_path=discharge_root,
                    observed_data_path="obs.nc",
                    sim_variable=None,
                    observed_variable=None,
                    model_keyword="mrm",
                    saving_path=td,
                    model_file_name="discharge.nc",
                    date_slice=date_slice,
                    overwrite=True,
                    direct_comparison=False,
                )

            def load_ds_side_effect(path, file_name=None):
                path = Path(path)
                if path.name == "obs.nc":
                    return cm_enter(obs_ds)
                if path.name == "sim.nc":
                    return cm_enter(sim_ds)
                if path.name == "scc.nc":
                    return cm_enter(scc_ds)
                if path.name == "restart.nc":
                    return cm_enter(xr.Dataset())
                raise FileNotFoundError(path)

            def sim_from_nodes(*args, **kwargs):
                da = xr.DataArray(
                    sim_vals,
                    dims=("time", "id"),
                    coords={"time": times, "id": np.array(ids, dtype=int)},
                    name="discharge",
                )
                da = da.sel(time=date_slice)
                return da, np.array(xs), np.array(ys), np.array(ids, dtype=int)

            def sim_for_gauge(id, index, sim_data, yarr, xarr, resolution, **kwargs):
                da = xr.DataArray(
                    sim_vals[:, index],
                    dims=("time",),
                    coords={"time": times},
                    name="discharge",
                )
                da = da.sel(time=date_slice)
                return da.expand_dims(dim={"id": [int(id)]})

            with patch.object(
                gv, "load_ds", side_effect=load_ds_side_effect
            ), patch.object(
                gv,
                "resample_to_coarser_calendar",
                return_value=(sim_ds, obs_ds),
            ), patch.object(
                gv, "get_sim_data_for_gauges_from_nodes", side_effect=sim_from_nodes
            ), patch.object(
                gv, "write_xarray_to_file"
            ):
                obs_s, sim_s = gv.Q_data_to_xarray(
                    model_data_path="sim.nc",
                    observed_data_path="obs.nc",
                    scc_gauges_file="scc.nc",
                    sim_variable="Qrouted",
                    observed_variable="runoff_mean_mm",
                    model_keyword="mrm",
                    saving_path=td,
                    model_file_name="mrm_output.nc",
                    date_slice=date_slice,
                    overwrite=True,
                    direct_comparison=False,
                )

            with patch.object(
                gv, "load_ds", side_effect=load_ds_side_effect
            ), patch.object(
                gv,
                "resample_to_coarser_calendar",
                return_value=(sim_ds, obs_ds),
            ), patch.object(
                gv,
                "get_gauge_coords",
                side_effect=[(xs[0], ys[0], facc[0]), (xs[1], ys[1], facc[1])],
            ), patch.object(
                gv, "get_sim_data_for_one_gauge", side_effect=sim_for_gauge
            ), patch.object(
                gv, "write_xarray_to_file"
            ):
                obs_g, sim_g = gv.Q_data_to_xarray(
                    model_data_path="sim.nc",
                    observed_data_path="obs.nc",
                    mrm_restart_file="restart.nc",
                    sim_variable="Qrouted",
                    observed_variable="runoff_mean_mm",
                    model_keyword="mrm",
                    saving_path=td,
                    model_file_name="mrm_output.nc",
                    date_slice=date_slice,
                    overwrite=True,
                    direct_comparison=False,
                )

            assert_discharge_equal(obs_d, obs_s)
            assert_discharge_equal(obs_d, obs_g)
            assert_discharge_equal(sim_d, sim_s)
            assert_discharge_equal(sim_d, sim_g)


if __name__ == "__main__":
    unittest.main()
