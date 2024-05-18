import unittest
import pandas as pd
import numpy as np
from tagra.preprocessing import preprocess_dataframe

class TestPreprocessing(unittest.TestCase):

    def setUp(self):
        self.df = pd.DataFrame({
            'A': [1, 2, np.nan, np.nan, 5],
            'B': ['a', 'b', 'c', 'd', 'e'],
            'C': [1.5, 2.5, 3.5, 4.5, 5.5],
            'D': [np.nan, '2021-01-01', '2021-01-02', '2021-01-03', '2021-01-04']
        })

    def test_nan_action_drop_row(self):
        result = preprocess_dataframe(self.df, nan_action='drop row')
        self.assertEqual(len(result), 2)

    def test_nan_action_drop_column(self):
        result = preprocess_dataframe(self.df, nan_action='drop column')
        self.assertNotIn('A', result.columns)
        self.assertNotIn('D', result.columns)

    def test_nan_action_infer(self):
        result = preprocess_dataframe(self.df, nan_action='infer')
        self.assertFalse(result.isnull().values.any())

    def test_numeric_scaling_standard(self):
        result = preprocess_dataframe(self.df, numeric_cols=['A', 'C'], numeric_scaling='standard')
        self.assertAlmostEqual(result['A'].mean(), 0, places=5)
        self.assertAlmostEqual(result['C'].mean(), 0, places=5)

    def test_numeric_scaling_minmax(self):
        result = preprocess_dataframe(self.df, numeric_cols=['A', 'C'], numeric_scaling='minmax')
        self.assertAlmostEqual(result['A'].max(), 1)
        self.assertAlmostEqual(result['C'].max(), 1)

    def test_categorical_encoding_one_hot(self):
        result = preprocess_dataframe(self.df, categorical_cols=['B'], categorical_encoding='one-hot')
        self.assertIn('B_a', result.columns)
        self.assertIn('B_b', result.columns)

    def test_categorical_encoding_label(self):
        result = preprocess_dataframe(self.df, categorical_cols=['B'], categorical_encoding='label')
        self.assertTrue(np.issubdtype(result['B'].dtype, np.integer))

    def test_ignore_cols(self):
        result = preprocess_dataframe(self.df, ignore_cols=['D'])
        self.assertIn('D', result.columns)

    # def test_infer_columns(self):
    #     df = pd.DataFrame({
    #         'A': [1, 2, 3, 4, 5],
    #         'B': [1.1, 2.2, 3.3, 4.4, 5.5],
    #         'C': ['a', 'b', 'c', 'd', 'e'],
    #         'D': ['2021-01-01', '2021-01-02', '2021-01-03', '2021-01-04', '2021-01-05']
    #     })
    #     result = preprocess_dataframe(df, unknown_col_action='infer')
    #     self.assertIn('A', result.select_dtypes(include=[np.number]).columns)
    #     self.assertIn('B', result.select_dtypes(include=[np.number]).columns)
    #     self.assertIn('C', result.select_dtypes(include=['object']).columns)
    #     self.assertIn('D', result.columns)  # Should be ignored

if __name__ == '__main__':
    unittest.main()
