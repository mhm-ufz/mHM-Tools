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
import pytest

from mhm_tools.common.logger import configure_mhm_tools_logger
from mhm_tools.post.hydrograph import Hydrograph

HERE = Path(__file__).parent


class TestHydrograph(unittest.TestCase):
    """A test case class for testing the Hydrograph class."""

    def setUp(self):
        """Set up the test case by initializing necessary variables and loading data from a specific path."""
        configure_mhm_tools_logger(log_level="ERROR")
        self.path = str(HERE / "files" / "test_hydrograph") + "/"
        self.hydro = Hydrograph()
        self.hydro.load_data_from_path(self.path)

    #    def test_file_readin(self):
    #         self.hydro.discharge_data
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
        self.hydro.calc_objectives(
            self.hydro.discharge_data["obs"], self.hydro.discharge_data["obs"]
        )
        assert np.abs(self.hydro.objectives.kge - 1) < 1e-4
        assert np.abs(self.hydro.objectives.nse - 1) < 1e-4

        # test offset
        self.hydro.calc_objectives(
            self.hydro.discharge_data["obs"], self.hydro.discharge_data["obs"] * 2
        )
        assert np.abs(self.hydro.objectives.kge - 1) > 1e-4
        assert np.abs(self.hydro.objectives.nse - 1) > 1e-4

        self.hydro.calc_objectives(
            np.random.normal(2, 0.5, 1000000), np.random.normal(1, 1, 1000000)
        )
        assert (
            np.abs(self.hydro.objectives.alpha - 2) < 1e-1
        )  # test if the relation of standard diviations is correct
        assert (
            np.abs(self.hydro.objectives.beta - 0.5) < 1e-1
        )  # test if the relation of the mean is correct

        self.hydro.calc_objectives(np.linspace(0, 10, 100), np.linspace(1, 11, 100))
        assert np.abs(self.hydro.objectives.r - 1) < 1e-4  # slope

        # test for wrong input
        with pytest.raises(
            ValueError, match="The two timeseries do not have the same length."
        ):
            self.hydro.calc_objectives(
                np.random.normal(2, 0.5, 1), np.random.normal(1, 1, 10)
            )

        # test if nan and None values are removed correctly
        q_test_2 = np.array([1, 2, 3, None, 5, 6, np.nan, 8])
        self.hydro.calc_objectives(q_test_2, np.arange(1, 9))
        assert self.hydro.objectives.nse - 1 < 1e-6
        assert self.hydro.objectives.kge - 1 < 1e-6

        # test for right result
        self.hydro.calc_objectives(
            self.hydro.discharge_data["obs"], self.hydro.discharge_data["sim"]
        )
        assert (
            np.abs(self.hydro.objectives.kge - 0.74597) < 1e-5
        )  # comparison with mhm internal kge calculation
        assert (
            np.abs(self.hydro.objectives.nse - 0.76691) < 1e-5
        )  # comparison with mhm internal nse calculation


if __name__ == "__main__":
    unittest.main()
