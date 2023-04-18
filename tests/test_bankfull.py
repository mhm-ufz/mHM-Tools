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
                [np.nan, np.nan, 5.8775, 541.3693, np.nan, np.nan],
                [7.854, 72.6613, 566.7343, 550.3261, np.nan, np.nan],
                [1.8963, 45.3326, 498.84, 7.659, np.nan, np.nan],
                [np.nan, 21.0789, 494.9771, 28.3122, 13.4768, 0.383],
                [np.nan, 11.8728, 470.2629, 48.0984, 24.2233, 2.2134],
                [np.nan, 0.736, 410.4428, 356.59, 185.3501, 23.8731],
                [np.nan, np.nan, 41.2516, 152.2119, 131.5352, 59.0854],
                [np.nan, np.nan, 0.1082, 103.5214, 93.6477, 11.6922],
                [np.nan, np.nan, np.nan, np.nan, 17.2395, np.nan],
            ],
            dtype=np.float32,
        )
        self.p_ref = np.array(
            [
                [np.nan, np.nan, 11.6369, 111.6833, np.nan, np.nan],
                [13.452, 40.916, 114.2697, 112.6033, np.nan, np.nan],
                [6.61, 32.3181, 107.2067, 13.284, np.nan, np.nan],
                [np.nan, 22.0376, 106.7908, 25.5404, 17.6212, 2.9706],
                [np.nan, 16.5393, 104.0906, 33.2895, 23.6243, 7.1411],
                [np.nan, 4.118, 97.2451, 90.6412, 65.3488, 23.4529],
                [np.nan, np.nan, 30.8292, 59.2196, 55.0506, 36.8962],
                [np.nan, np.nan, 1.5789, 48.8378, 46.4504, 16.413],
                [np.nan, np.nan, np.nan, np.nan, 19.9298, np.nan],
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
