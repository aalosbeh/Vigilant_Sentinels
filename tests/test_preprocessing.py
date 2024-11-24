import unittest
from src.preprocessing import clean_data

class TestPreprocessing(unittest.TestCase):
    def test_clean_data(self):
        df = pd.DataFrame({'col1': [1, 2, None], 'col2': [None, 2, 3]})
        cleaned_df = clean_data(df)
        self.assertFalse(cleaned_df.isnull().values.any())
