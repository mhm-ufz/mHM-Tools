import unittest
from pathlib import Path

import numpy as np
import xarray as xr

from mhm_tools.post.hydrograph import Hydrograph

HERE = Path(__file__).parent


class TestHydrograph(unittest.TestCase):
    def setUp(self):
        self.path = str(HERE / "output") + "/"
        self.hydro = Hydrograph("error")
        self.hydro.load_data_from_path(self.path)

    #    def test_file_readin(self):
    #         self.hydro.discharge_data

    def test_calc_objectives(self):
        # test equal arrays
        self.hydro.calc_objectives(
            self.hydro.discharge_data["obs"], self.hydro.discharge_data["obs"]
        )
        self.assertTrue(np.abs(self.hydro.objectives.kge - 1) < 1e-4)
        self.assertTrue(np.abs(self.hydro.objectives.nse - 1) < 1e-4)

        # test offset
        self.hydro.calc_objectives(
            self.hydro.discharge_data["obs"], self.hydro.discharge_data["obs"] * 2
        )
        self.assertTrue(np.abs(self.hydro.objectives.kge - 1) > 1e-4)
        self.assertTrue(np.abs(self.hydro.objectives.nse - 1) > 1e-4)

        self.hydro.calc_objectives(
            np.random.normal(2, 0.5, 1000000), np.random.normal(1, 1, 1000000)
        )
        self.assertTrue(
            np.abs(self.hydro.objectives.alpha - 2) < 1e-1
        )  # test if the relation of standard diviations is correct
        self.assertTrue(
            np.abs(self.hydro.objectives.beta - 0.5) < 1e-1
        )  # test if the relation of the mean is correct

        self.hydro.calc_objectives(np.linspace(0, 10, 100), np.linspace(1, 11, 100))
        self.assertTrue(np.abs(self.hydro.objectives.r - 1) < 1e-4)  # slope

        # test for wrong input
        with self.assertRaises(Exception) as context:
            self.hydro.calc_objectives(
                np.random.normal(2, 0.5, 1), np.random.normal(1, 1, 10)
            )
        self.assertTrue(
            "The two timeseries do not have the same length.", str(context.exception)
        )

        # test if nan and None values are removed correctly
        q_test_2 = np.array([1, 2, 3, None, 5, 6, np.nan, 8])
        self.hydro.calc_objectives(q_test_2, np.arange(1, 9))
        self.assertTrue(self.hydro.objectives.nse - 1 < 1e-6)
        self.assertTrue(self.hydro.objectives.kge - 1 < 1e-6)

        # test for right result
        self.hydro.calc_objectives(
            self.hydro.discharge_data["obs"], self.hydro.discharge_data["sim"]
        )
        print(self.hydro.objectives.nse, self.hydro.objectives.kge)
        self.assertTrue(np.abs(self.hydro.objectives.kge - 1) < 1e-6)


if __name__ == "__main__":
    unittest.main()
