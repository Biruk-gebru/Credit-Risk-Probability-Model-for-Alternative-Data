import unittest
import pandas as pd
import os
import sys

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from utils import load_data, check_missing_values

class TestUtils(unittest.TestCase):
    def setUp(self):
        self.test_file = 'test_data.csv'
        df = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, None]})
        df.to_csv(self.test_file, index=False)

    def tearDown(self):
        if os.path.exists(self.test_file):
            os.remove(self.test_file)

    def test_load_data(self):
        df = load_data(self.test_file)
        self.assertEqual(df.shape, (3, 2))

    def test_check_missing_values(self):
        df = load_data(self.test_file)
        missing = check_missing_values(df)
        self.assertEqual(missing['B'], 1)
        self.assertEqual(missing['A'], 0)

if __name__ == '__main__':
    unittest.main()
