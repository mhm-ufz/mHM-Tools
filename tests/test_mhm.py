import unittest

import mhm_tools as mt


class TestVersion(unittest.TestCase):
    def test_creation(self):
        self.assertIsInstance(mt.__version__, str)
