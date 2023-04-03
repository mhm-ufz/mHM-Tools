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
                [np.nan, np.nan, 0.3729, 58.051, np.nan, np.nan],
                [0.4879, 3.6077, 56.396, 57.349, np.nan, np.nan],
                [0.1256, 2.1827, 50.7687, 0.3459, np.nan, np.nan],
                [np.nan, 0.8753, 49.271, 1.6169, 0.8178, 0.0269],
                [np.nan, 0.5669, 46.6579, 2.8417, 1.5105, 0.2802],
                [np.nan, 0.0387, 41.616, 37.4089, 21.0907, 3.7146],
                [np.nan, np.nan, 2.2173, 14.3115, 14.2666, 6.2693],
                [np.nan, np.nan, 0.0138, 11.1801, 9.8372, 1.2622],
                [np.nan, np.nan, np.nan, np.nan, 2.1144, np.nan],
            ],
            dtype=np.float32,
        )
        self.p_ref = np.array(
            [
                [np.nan, np.nan, 2.9311, 36.5718, np.nan, np.nan],
                [3.3528, 9.1171, 36.0467, 36.35, np.nan, np.nan],
                [1.7014, 7.0915, 34.201, 2.8231, np.nan, np.nan],
                [np.nan, 4.4908, 33.6928, 6.1036, 4.3408, 0.7876],
                [np.nan, 3.6139, 32.7872, 8.0915, 5.8992, 2.541],
                [np.nan, 0.9447, 30.965, 29.3582, 22.0438, 9.2512],
                [np.nan, np.nan, 7.1475, 18.1587, 18.1301, 12.0185],
                [np.nan, np.nan, 0.5629, 16.0496, 15.0548, 5.3926],
                [np.nan, np.nan, np.nan, np.nan, 6.9797, np.nan],
            ],
            dtype=np.float32,
        )

    def test_bankfull(self):
        mt.post.gen_bankfull_discharge(self.in_file, self.out_file, peri_bkfl=True)
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
