import unittest
from pathlib import Path

import numpy as np
import xarray as xr

from mhm_tools.post.hydrograph import Hydrograph

HERE = Path(__file__).parent


class TestHydrograph(unittest.TestCase):
    def setUp(self):
        self.in_file = HERE / "output" / "discharge.nc"
        
    def test_calc_kge(self):
        hydro = Hydrograph('error')
        q_test = np.random.random(1000)

        # test equal arrays
        kge = hydro.calc_kling_gupta_efficiency(q_test, q_test)
        self.assertTrue(np.abs(kge['KGE']-1) < 1e-4)

        # test offset
        kge = hydro.calc_kling_gupta_efficiency(q_test, q_test+1)
        self.assertTrue(np.abs(kge['KGE']-1) > 1e-4)

        kge = hydro.calc_kling_gupta_efficiency(np.random.normal(2,0.5,1000000), np.random.normal(1,1,1000000))
        print(kge)
        self.assertTrue(np.abs(kge['alpha']-2) < 1e-1) # test if the relation of standard diviations is correct
        self.assertTrue(np.abs(kge['beta']-0.5) < 1e-1) # test if the relation of the mean is correct

        kge = hydro.calc_kling_gupta_efficiency(np.linspace(0,10,100), np.linspace(1,11,100))
        self.assertTrue(np.abs(kge['r']-1) < 1e-4) # slope
        self.assertTrue(np.abs(kge['offset']-1) < 1e-4) # offset

        # test for wrong input
        # self.assertRaises(TypeError, hydro.calc_kling_gupta_efficiency(np.random.normal(2,0.5,10000), np.random.normal(1,1,1000000)))
        


if __name__ == "__main__":
    unittest.main()
