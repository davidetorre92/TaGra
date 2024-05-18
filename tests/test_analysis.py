import unittest
import networkx as nx
from tagra.analysis import analyze_graph

class TestGraphAnalysis(unittest.TestCase):

    def setUp(self):
        self.G = nx.Graph()
        self.G.add_edges_from([(0, 1), (1, 2), (2, 3), (3, 4)])
        nx.set_node_attributes(self.G, {0: 'A', 1: 'A', 2: 'B', 3: 'B', 4: 'C'}, 'attribute')

    def test_degree_distribution(self):
        degree_sequence, _ = analyze_graph(self.G, verbose=False)
        self.assertEqual(len(degree_sequence), len(self.G.nodes))

    def test_closeness_centrality(self):
        _, neighbors_count = analyze_graph(self.G, verbose=False)
        self.assertIn('neighbors_count', neighbors_count.columns)

    def test_community_detection(self):
        _, neighbors_count = analyze_graph(self.G, attribute='attribute', clustering_method='hierarchical', verbose=False)
        self.assertIn('community', neighbors_count.columns)

if __name__ == '__main__':
    unittest.main()
