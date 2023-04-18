import unittest
from pathlib import Path

import numpy as np
import xarray as xr

import mhm_tools as mt

HERE = Path(__file__).parent


class TestBankfull(unittest.TestCase):
    def setUp(self):
        self.in_file = HERE / "output" / "mRM_Fluxes_States.nc"
        self.out_file = HERE / "Q-bkfl.nc"
        self.out_file.unlink(missing_ok=True)
        self.q_ref = np.array(
            [
                [np.nan, np.nan, 0.4079, 69.2566, np.nan, np.nan],
                [0.6511, 5.4555, 67.2125, 68.4389, np.nan, np.nan],
                [0.167, 3.4875, 58.2526, 0.5125, np.nan, np.nan],
                [np.nan, 1.4072, 55.1022, 2.3058, 1.072, 0.0315],
                [np.nan, 0.8413, 51.0577, 3.9395, 2.0886, 0.2707],
                [np.nan, 0.0476, 41.8436, 37.3851, 19.3875, 3.016],
                [np.nan, np.nan, 2.4074, 14.0697, 13.5038, 5.8762],
                [np.nan, np.nan, 0.0138, 10.8045, 9.7246, 1.2178],
                [np.nan, np.nan, np.nan, np.nan, 2.1156, np.nan],
            ],
            dtype=np.float32,
        )
        self.p_ref = np.array(
            [
                [np.nan, np.nan, 3.0657, 39.9459, np.nan, np.nan],
                [3.8731, 11.2114, 39.352, 39.7094, np.nan, np.nan],
                [1.9618, 8.9639, 36.6352, 3.4362, np.nan, np.nan],
                [np.nan, 5.694, 35.6308, 7.2888, 4.9698, 0.8519],
                [np.nan, 4.4028, 34.2983, 9.5272, 6.9369, 2.4975],
                [np.nan, 1.0477, 31.0496, 29.3488, 21.135, 8.336],
                [np.nan, np.nan, 7.4477, 18.0046, 17.6388, 11.6356],
                [np.nan, np.nan, 0.5629, 15.7777, 14.9685, 5.297],
                [np.nan, np.nan, np.nan, np.nan, 6.9817, np.nan],
            ],
            dtype=np.float32,
        )

    def test_bankfull(self):
        mt.post.bankfull_discharge(self.in_file, self.out_file, peri_bkfl=True)
        self.assertTrue(self.out_file.is_file())
        ds = xr.load_dataset(self.out_file)
        self.assertIn("Q_bkfl", ds.variables)
        self.assertIn("P_bkfl", ds.variables)
        self.assertSequenceEqual(ds["Q_bkfl"].dims, ("northing", "easting"))
        self.assertSequenceEqual(ds["P_bkfl"].dims, ("northing", "easting"))
        self.assertTrue(
            np.all(np.isclose(ds["Q_bkfl"].data, self.q_ref, atol=1e-4, equal_nan=True))
        )
        self.assertTrue(
            np.all(np.isclose(ds["P_bkfl"].data, self.p_ref, atol=1e-4, equal_nan=True))
        )


if __name__ == "__main__":
    unittest.main()
