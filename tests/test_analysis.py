import unittest
import networkx as nx
import pandas as pd
import numpy as np
import os
import tempfile
import pickle
from datetime import datetime

from tagra.analysis import analyze_graph

class TestAnalyzeGraph(unittest.TestCase):

    def setUp(self):
        # Create a temporary directory
        self.test_dir = tempfile.TemporaryDirectory()
        self.temp_dir = self.test_dir.name

        # Create a simple graph for testing
        self.G = nx.karate_club_graph()
        for i, data in self.G.nodes(data=True):
            data['club'] = data.get('club', 'None')
        
        # Save the graph to a temporary file
        self.graph_path = os.path.join(self.temp_dir, 'test_graph.pickle')
        with open(self.graph_path, 'wb') as f:
            pickle.dump(self.G, f)

    def tearDown(self):
        # Cleanup the temporary directory
        self.test_dir.cleanup()

    def test_analyze_graph_with_attribute(self):
        # Run analyze_graph with an attribute
        analyze_graph(
            graph_path=self.graph_path,
            attribute='club',
            clustering_method='hierarchical',
            inconsistency_threshold=0.1,
            verbose=False,
            plot_graph=True,
            neigh_prob_path=os.path.join(self.temp_dir, 'neighbor_stat.dat'),
            degree_distribution_outpath=os.path.join(self.temp_dir, 'degree_distribution.png'),
            betweenness_distribution_outpath=os.path.join(self.temp_dir, 'betweenness_distribution.png'),
            prob_heatmap_path=os.path.join(self.temp_dir, 'prob_heatmap.png'),
            community_composition_outpath=os.path.join(self.temp_dir, 'community_composition.png'),
            graph_visualization_path=os.path.join(self.temp_dir, 'graph_visualization.png')
        )

        # Check if the output files were created
        self.assertTrue(os.path.exists(os.path.join(self.temp_dir, 'neighbor_stat.dat')))
        self.assertTrue(os.path.exists(os.path.join(self.temp_dir, 'degree_distribution.png')))
        self.assertTrue(os.path.exists(os.path.join(self.temp_dir, 'betweenness_distribution.png')))
        self.assertTrue(os.path.exists(os.path.join(self.temp_dir, 'prob_heatmap.png')))
        self.assertTrue(os.path.exists(os.path.join(self.temp_dir, 'community_composition.png')))
        self.assertTrue(os.path.exists(os.path.join(self.temp_dir, 'graph_visualization.png')))

if __name__ == '__main__':
    unittest.main()
