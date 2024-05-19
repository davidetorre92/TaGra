import pickle
import datetime
import networkx as nx
import pandas as pd
import numpy as np

from .utils import (
    analyze_neighborhood_attributes,
    print_neighbors_prob,
    heat_map_prob,
    plot_distribution,
    plot_community_composition,
    matplotlib_graph_visualization
)

def analyze_graph(graph_path=None, 
                  attribute=None, 
                  clustering_method='hierarchical', 
                  inconsistency_threshold=0.1, 
                  verbose=True,
                  plot_graph=False, 
                  neigh_prob_path = None,
                  degree_distribution_outpath = None,
                  betweenness_distribution_outpath = None,
                  prob_heatmap_path = None,
                  community_composition_outpath = None,
                  graph_visualization_path = None):
    if isinstance(graph_path, str):
        G = pickle.load(open(graph_path, 'rb'))
    elif isinstance(graph_path, nx.Graph):
        G = graph_path
    else:
        raise ValueError("Invalid graph_path. Must be a path to a file or a NetworkX Graph.")

    if neigh_prob_path is None:
        neigh_prob_path = f"./neighbor_stat_{datetime.datetime.now().strftime('%Y%m%d%H%M')}.dat"

    if verbose:
        print(f"--------------------------\nGraph analysis options\n--------------------------\n\n"
              f"\tOptions:\n"
              f"\tgraph_path: {graph_path}, attribute: {attribute}, \n"
              f"\tclustering_method: {clustering_method}, inconsistency_threshold: {inconsistency_threshold}, verbose: {verbose}\n\n")

    if attribute is not None:
        df_neigh = analyze_neighborhood_attributes(G, attribute_name = attribute)
        probabilities = print_neighbors_prob(df_neigh, attribute)
        for (i, j), prob in probabilities.items():
            print(f"P({j}|{i}) = {prob}")
        with open(neigh_prob_path, 'w') as fp:
            for (i, j), prob in probabilities.items():
                fp.write(f"P({j}|{i}) = {prob}")   

        heat_map_prob(probabilities, df_neigh, attribute, prob_heatmap_path)

    degree_data = {'data': [degree for _, degree in G.degree()],
                   'title': 'Degree distribution',
                   'xlabel': 'Degree',
                   'ylabel': 'Number of Nodes'}
    betweenness_data = {'data': [degree for _, degree in G.degree()],
                   'title': 'Degree distribution',
                   'xlabel': 'Degree',
                   'ylabel': 'Number of Nodes'}
    
    plot_distribution(degree_data, degree_distribution_outpath)
    plot_distribution(betweenness_data, betweenness_distribution_outpath)
    plot_community_composition(G, attribute, community_composition_outpath)
    if plot_graph:
        matplotlib_graph_visualization(G, attribute, graph_visualization_path, palette = 'viridis')
