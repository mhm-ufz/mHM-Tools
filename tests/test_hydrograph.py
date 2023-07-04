import unittest
from pathlib import Path

import numpy as np
import xarray as xr

from mhm_tools.post.hydrograph import Hydrograph

HERE = Path(__file__).parent


class TestHydrograph(unittest.TestCase):
    def setUp(self):
        self.in_file = HERE / "output" / "discharge.nc"
        
    def test_calc_objectives(self):
        hydro = Hydrograph('error')
        q_test = np.random.random(1000)

        # test equal arrays
        nse, kge = hydro.calc_objectives(q_test, q_test)
        self.assertTrue(np.abs(kge['KGE']-1) < 1e-4)
        self.assertTrue(np.abs(nse-1) < 1e-4)

        # test offset
        nse, kge = hydro.calc_objectives(q_test, q_test+1)
        self.assertTrue(np.abs(kge['KGE']-1) > 1e-4)
        self.assertTrue(np.abs(nse-1) > 1e-4)

        nse, kge = hydro.calc_objectives(np.random.normal(2,0.5,1000000), np.random.normal(1,1,1000000))
        self.assertTrue(np.abs(kge['alpha']-2) < 1e-1) # test if the relation of standard diviations is correct
        self.assertTrue(np.abs(kge['beta']-0.5) < 1e-1) # test if the relation of the mean is correct

        nse, kge = hydro.calc_objectives(np.linspace(0,10,100), np.linspace(1,11,100))
        self.assertTrue(np.abs(kge['r']-1) < 1e-4) # slope
        self.assertTrue(np.abs(kge['offset']-1) < 1e-4) # offset

        # test for wrong input
        with self.assertRaises(Exception) as context:
            hydro.calc_objectives(np.random.normal(2,0.5,1), np.random.normal(1,1,10))
        self.assertTrue('The two timeseries do not have the same length.', str(context.exception))
                
        # test if nan and None values are removed correctly
        q_test_2 = np.array([1,2,3,None,5,6,np.nan,8])
        nse, kge = hydro.calc_objectives(q_test_2, np.arange(1,9))
        self.assertTrue(nse - 1 < 1e-6)
        self.assertTrue(kge['KGE'] - 1 < 1e-6)

if __name__ == "__main__":
    unittest.main()
