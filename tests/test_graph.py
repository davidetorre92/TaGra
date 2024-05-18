import unittest
import pandas as pd
import networkx as nx
from tagra.graph import create_graph

class TestGraphCreation(unittest.TestCase):

    def setUp(self):
        self.df = pd.DataFrame({
            'A': [1, 2, 3, 4, 5],
            'B': [1.1, 2.2, 3.3, 4.4, 5.5],
            'C': ['a', 'b', 'c', 'd', 'e']
        })

    def test_knn_method(self):
        G = create_graph(dataframe_path=self.df, method='knn', k=2)
        self.assertEqual(len(G.nodes), len(self.df))
        self.assertTrue(all(len(list(G.neighbors(n))) >= 2 for n in G.nodes))

    def test_distance_threshold_method(self):
        G = create_graph(dataframe_path=self.df, method='distance_threshold', threshold=1.5)
        self.assertEqual(len(G.nodes), len(self.df))

    def test_similarity_method(self):
        G = create_graph(dataframe_path=self.df, method='similarity', threshold=0.99)
        self.assertEqual(len(G.nodes), len(self.df))

if __name__ == '__main__':
    unittest.main()
