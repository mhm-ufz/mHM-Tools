import numpy as np
import pytest
import xarray as xr

from mhm_tools.post.taylor_diagram import (
    calc_tim_mean,
    generate_taylor_diagram,
    mask_nan,
)


def _make_ds(var_name: str, values: np.ndarray) -> xr.Dataset:
    time = np.arange(values.shape[0])
    lat = np.array([10.0, 11.0])
    lon = np.array([100.0, 101.0])
    return xr.Dataset(
        {
            var_name: (
                ("time", "lat", "lon"),
                values,
            )
        },
        coords={"time": time, "lat": lat, "lon": lon},
    )


def test_calc_tim_mean_requires_time():
    da = xr.DataArray(np.ones((2, 2)), dims=("lat", "lon"))
    with pytest.raises(ValueError, match="time"):
        calc_tim_mean(da)


def test_calc_tim_mean_requires_multiple_steps():
    da = xr.DataArray(np.ones((1, 2, 2)), dims=("time", "lat", "lon"))
    with pytest.raises(ValueError, match="more than one time step"):
        calc_tim_mean(da)


def test_mask_nan_filters_consistently():
    obs = np.array([1.0, np.nan, 3.0, 4.0])
    sims = {
        "a": np.array([1.0, 2.0, np.nan, 4.0]),
        "b": np.array([1.0, 2.0, 3.0, np.nan]),
    }
    obs_clean, sims_clean = mask_nan(obs, sims)
    assert np.allclose(obs_clean, np.array([1.0]))
    assert np.allclose(sims_clean["a"], np.array([1.0]))
    assert np.allclose(sims_clean["b"], np.array([1.0]))


def test_generate_taylor_diagram_creates_output(tmp_path):
    ref_vals = np.random.RandomState(0).rand(5, 2, 2)
    mod1_vals = ref_vals * 1.1
    mod2_vals = ref_vals * 0.9

    ref_dir = tmp_path / "ref"
    mod1_dir = tmp_path / "mod1"
    mod2_dir = tmp_path / "mod2"
    ref_dir.mkdir()
    mod1_dir.mkdir()
    mod2_dir.mkdir()

    _make_ds("ref_var", ref_vals).to_netcdf(ref_dir / "ref.nc")
    _make_ds("mod_var", mod1_vals).to_netcdf(mod1_dir / "mod.nc")
    _make_ds("mod_var", mod2_vals).to_netcdf(mod2_dir / "mod.nc")

    generate_taylor_diagram(
        ref_input_dir=str(ref_dir),
        reference_pattern="ref.nc",
        ref_var="ref_var",
        ref_label="ref",
        mod_input_dirs=[str(mod1_dir), str(mod2_dir)],
        model_patterns=["mod.nc", "mod.nc"],
        mod_vars=["mod_var", "mod_var"],
        mod_labels=["m1", "m2"],
        title="",
        output_dir=str(tmp_path),
        output_file="taylor.png",
        normalize=False,
    )

    assert (tmp_path / "taylor.png").is_file()


def test_generate_taylor_diagram_normalize_zero_std_raises(tmp_path):
    ref_vals = np.zeros((5, 2, 2))
    mod_vals = np.ones((5, 2, 2))

    ref_dir = tmp_path / "ref"
    mod_dir = tmp_path / "mod"
    ref_dir.mkdir()
    mod_dir.mkdir()

    _make_ds("ref_var", ref_vals).to_netcdf(ref_dir / "ref.nc")
    _make_ds("mod_var", mod_vals).to_netcdf(mod_dir / "mod.nc")

    with pytest.raises(ValueError, match="Standard deviation"):
        generate_taylor_diagram(
            ref_input_dir=str(ref_dir),
            reference_pattern="ref.nc",
            ref_var="ref_var",
            ref_label="ref",
            mod_input_dirs=[str(mod_dir)],
            model_patterns=["mod.nc"],
            mod_vars=["mod_var"],
            mod_labels=["m1"],
            title="",
            output_dir=str(tmp_path),
            output_file="taylor.png",
            normalize=True,
        )
