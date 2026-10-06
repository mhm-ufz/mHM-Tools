"""
Unit tests for the Hydrograph class in the mhm_tools.post.hydrograph module.

Classes:
    TestHydrograph: A unittest.TestCase subclass containing tests for the Hydrograph class.

Methods
-------
    setUp: Set up the test environment for each test in the TestHydrograph class.
    test_read_area: Test the get_catchment_area method of the Hydrograph class.
    test_remove_nans: Test the remove_empty_values method of the Hydrograph class.
    test_calc_objectives: Test the calc_objectives method of the Hydrograph class.
"""

import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from mhm_tools.common.logger import configure_mhm_tools_logger
from mhm_tools.post.hydrograph import Hydrograph, get_hydrograph_from_path

HERE = Path(__file__).parent


class TestHydrograph(unittest.TestCase):
    """A test case class for testing the Hydrograph class."""

    def setUp(self):
        """Set up the test case by initializing necessary variables and loading data from a specific path."""
        configure_mhm_tools_logger(log_level="ERROR")
        self.path = HERE / "files" / "test_hydrograph"
        self.hydro = Hydrograph()
        self.hydro.load_data_from_discharge_nc(self.path)

    def test_read_area(self):
        """
        Test case for the `get_catchment_area` method in the `hydro` object.

        It verifies that the catchment area is correctly read and rounded to the specified decimal places.
        """
        self.hydro.get_catchment_area(self.path)
        assert (
            self.hydro.catchment.area == "11636"
        )  # 11636.250 rounded to 0 decimal places
        self.hydro.get_catchment_area(self.path, 3)
        assert (
            self.hydro.catchment.area == "11636.250"
        )  # 11636.250 rounded to 0 decimal places

    def test_remove_nans(self):
        """
        Test case for the remove_empty_values method.

        This test case checks if the remove_empty_values method correctly removes NaN and None values from the input arrays.
        It also verifies that the method raises a TypeError when the input arrays contain invalid data types.
        """
        q_test = [1, 2, 3, np.nan, 5, 6, np.nan, 8]
        q_test_2 = np.array([1, None, 3, 4, 5, 6, 7, 8])
        q_test_result = np.array([1, 3, 5, 6, 8])
        q_test_rem, q_test_rem_2 = self.hydro.remove_empty_values(q_test, q_test_2)
        assert np.all(q_test_rem == q_test_rem_2)
        assert np.all(q_test_rem_2 == q_test_result)
        q_test[0] = "wrong_input_type"
        with pytest.raises(TypeError):
            self.hydro.remove_empty_values(q_test, q_test_2)

    def test_calc_objectives(self):
        """
        Test the calc_objectives method of the Hydrograph class.

        This method tests the calculation of various objectives such as KGE (Kling-Gupta Efficiency) and NSE (Nash-Sutcliffe Efficiency).
        It verifies the correctness of the calculated objectives by comparing them with expected values.

        The method performs the following tests:
        - Test with equal arrays: The method calculates objectives using two identical arrays and checks if the calculated KGE and NSE are close to 1.
        - Test with offset arrays: The method calculates objectives using two arrays with an offset and checks if the calculated KGE and NSE are not close to 1.
        - Test with random arrays: The method calculates objectives using two random arrays and checks if the calculated alpha and beta values are close to the expected values.
        - Test with linearly increasing arrays: The method calculates objectives using two linearly increasing arrays and checks if the calculated slope (r) is close to 1.
        - Test with arrays of different lengths: The method checks if a ValueError is raised when two arrays of different lengths are provided.
        - Test with arrays containing nan and None values: The method checks if nan and None values are correctly removed before calculating the objectives.
        - Test with actual data: The method calculates objectives using observed and simulated discharge data and compares the calculated KGE and NSE with the expected values.

        """
        # test equal arrays
        self.hydro.sim_discharge_data_clean = None
        self.hydro.obs_discharge_data_clean = None
        self.hydro.calc_objectives(
            self.hydro.obs_discharge_data, self.hydro.obs_discharge_data
        )
        assert np.abs(self.hydro.objectives.kge - 1) < 1e-4
        assert np.abs(self.hydro.objectives.nse - 1) < 1e-4

        # test offset
        self.hydro.sim_discharge_data_clean = None
        self.hydro.obs_discharge_data_clean = None
        self.hydro.calc_objectives(
            self.hydro.obs_discharge_data, self.hydro.obs_discharge_data * 2
        )
        assert np.abs(self.hydro.objectives.kge - 1) > 1e-4
        assert np.abs(self.hydro.objectives.nse - 1) > 1e-4

        self.hydro.sim_discharge_data_clean = None
        self.hydro.obs_discharge_data_clean = None
        self.hydro.calc_objectives(
            np.random.normal(2, 0.5, 1000000), np.random.normal(1, 1, 1000000)
        )
        assert (
            np.abs(self.hydro.objectives.alpha - 2) < 1e-1
        )  # test if the relation of standard diviations is correct
        assert (
            np.abs(self.hydro.objectives.beta - 0.5) < 1e-1
        )  # test if the relation of the mean is correct

        self.hydro.sim_discharge_data_clean = None
        self.hydro.obs_discharge_data_clean = None
        self.hydro.calc_objectives(np.linspace(0, 10, 100), np.linspace(1, 11, 100))
        assert np.abs(self.hydro.objectives.gamma - 1) < 1e-4  # slope

        # test for wrong input
        # with pytest.raises(
        #     ValueError, match="The two timeseries do not have the same length."
        # ):
        #     self.hydro.calc_objectives(
        #         np.random.normal(2, 0.5, 1), np.random.normal(1, 1, 10)
        #     )

        self.hydro.sim_discharge_data_clean = None
        self.hydro.obs_discharge_data_clean = None
        # test if nan and None values are removed correctly
        q_test_2 = np.array([1, 2, 3, None, 5, 6, np.nan, 8])
        self.hydro.calc_objectives(q_test_2, np.arange(1, 9))
        assert self.hydro.objectives.nse - 1 < 1e-6
        assert self.hydro.objectives.kge - 1 < 1e-6

        # test for right result
        self.hydro.sim_discharge_data_clean = None
        self.hydro.obs_discharge_data_clean = None
        self.hydro.calc_objectives(
            self.hydro.obs_discharge_data, self.hydro.sim_discharge_data
        )
        assert (
            np.abs(self.hydro.objectives.kge - 0.74597) < 1e-5
        )  # comparison with mhm internal kge calculation
        assert (
            np.abs(self.hydro.objectives.nse - 0.76691) < 1e-5
        )  # comparison with mhm internal nse calculation


if __name__ == "__main__":
    unittest.main()


def test_calc_objectives_uses_cropped_overlap_for_kge():
    """Regression test for KGE calculation after cropping to overlapping time."""
    observation_time = np.array(
        ["2000-01-01", "2000-01-02", "2000-01-03", "2000-01-04"],
        dtype="datetime64[D]",
    )
    simulation_time = np.array(
        ["2000-01-03", "2000-01-04", "2000-01-05", "2000-01-06"],
        dtype="datetime64[D]",
    )
    observation = xr.DataArray(
        [10.0, 10.0, 2.0, 3.0],
        coords={"time": observation_time},
        dims="time",
    )
    simulation = xr.DataArray(
        [2.0, 3.0, 100.0, 100.0],
        coords={"time": simulation_time},
        dims="time",
    )
    hydrograph = Hydrograph(simulation=simulation, observation=observation)

    assert hydrograph.crop_data_to_overlapping_time()
    hydrograph.calc_objectives(
        observed=hydrograph.obs_discharge_data,
        simulated=hydrograph.sim_discharge_data,
    )

    assert np.isclose(hydrograph.objectives.kge, 1.0)


def test_get_hydrograph_from_path_single_input_creates_csv(tmp_path):
    input_path = HERE / "files" / "test_hydrograph"
    output_file = tmp_path / "hydrograph.png"

    get_hydrograph_from_path(
        input_path=input_path,
        output_file=output_file,
        show=False,
        save=False,
        title="unit test",
        plot_code="t",
        prec_path=input_path,
    )

    assert (tmp_path / "kge.csv").is_file()
    assert not output_file.exists()


def test_get_hydrograph_from_path_multi_input_creates_csv(tmp_path):
    input_path = HERE / "files" / "test_hydrograph"
    output_file = tmp_path / "hydrograph.png"

    get_hydrograph_from_path(
        input_path=[input_path, input_path],
        output_file=output_file,
        show=False,
        save=False,
        title="unit test",
        plot_code="t",
        prec_path=input_path,
    )

    assert (tmp_path / "kge.csv").is_file()


def test_get_hydrograph_from_path_output_dir_default_png(tmp_path):
    input_path = HERE / "files" / "test_hydrograph"
    output_dir = tmp_path / "outdir"

    get_hydrograph_from_path(
        input_path=input_path,
        output_file=output_dir,
        show=False,
        save=True,
        title="unit test",
        plot_code="t",
        prec_path=input_path,
    )

    assert (output_dir / "hydrograph.png").is_file()
    assert (output_dir / "kge.csv").is_file()


def test_load_precipitation_data_dir_without_pre_nc(tmp_path):
    hydro = Hydrograph(calc_stats=False)
    hydro.load_precipiation_data(tmp_path)
    assert hydro.pre is None


def _hydrograph_with_offset(units, area_km2=None):
    """Score a simulation lying 2 units above a daily observation for 10 days."""
    time = pd.date_range("1990-01-01", periods=10, freq="D")
    # the variation sums to zero, so the observed sum is 10 times 10
    observed_values = 10.0 + np.tile([-1.0, 1.0], 5)
    observed = xr.DataArray(
        observed_values, dims="time", coords={"time": time}, attrs={"units": units}
    )
    simulated = (observed + 2.0).assign_attrs(units=units)
    hydro = Hydrograph(calc_stats=True)
    hydro.set_discharge(simulation=simulated, observation=observed)
    hydro.catchment.area = area_km2
    hydro.calc_objectives(hydro.obs_discharge_data, hydro.sim_discharge_data)
    return hydro


def test_calc_objectives_gives_diff_as_a_volume():
    """Give diff in m3, the rate difference times the daily step."""
    hydro = _hydrograph_with_offset("m3 s-1")
    assert hydro.objectives.diff == pytest.approx(10 * 2.0 * 86400)
    assert hydro.objectives.rel_diff == pytest.approx(0.2)


def test_calc_objectives_turns_a_depth_rate_into_a_volume_with_the_area():
    """Turn a mm/d difference into m3 over the catchment area."""
    hydro = _hydrograph_with_offset("mm d-1", area_km2=100.0)
    # 20 mm over 100 km2
    assert hydro.objectives.diff == pytest.approx(0.02 * 100.0 * 1e6)


def test_calc_objectives_gives_no_volume_for_a_depth_rate_without_area():
    """Give NaN for a depth rate when the catchment area is unknown."""
    hydro = _hydrograph_with_offset("mm d-1")
    assert np.isnan(hydro.objectives.diff)
