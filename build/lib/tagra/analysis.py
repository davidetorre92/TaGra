import pickle
import datetime
import networkx as nx
import pandas as pd
import numpy as np
import os
from datetime import datetime

from .utils import (
    analyze_neighborhood_attributes,
    print_neighbors_prob,
    heat_map_prob,
    plot_distribution,
    plot_community_composition,
    matplotlib_graph_visualization
)

def analyze_graph(graph, 
                  target_attributes=None, 
                  clustering_method='hierarchical', 
                  inconsistency_threshold=0.1,
                  verbose=True,
                  output_directory=None,
                  neigh_prob_filename = None,
                  degree_distribution_filename = None,
                  betweenness_distribution_filename = None,
                  prob_heatmap_filename = None,
                  community_composition_filename = None,
                  graph_visualization_filename = None):
    
    # Output path managing
    time_str = datetime.now().strftime('%Y%m%d%H%M')
    if output_directory is None:
        output_directory = './'
    if os.path.exists(output_directory) is False:
        os.mkdir(output_directory)
        print(f"{datetime.datetime.now()}: Output directory created: {output_directory}.")

    if degree_distribution_filename is None:
        degree_distribution_outpath = None
    else:
        basename = os.path.basename(degree_distribution_filename)
        base, ext = os.path.splitext(basename)
        degree_distribution_filename = f"{base}_{time_str}{ext}"
        degree_distribution_outpath = os.path.join(output_directory, degree_distribution_filename)

    if betweenness_distribution_filename is None:
        betweenness_distribution_outpath = None
    else:
        basename = os.path.basename(betweenness_distribution_filename)
        base, ext = os.path.splitext(basename)
        betweenness_distribution_filename = f"{base}_{time_str}{ext}"
        betweenness_distribution_outpath = os.path.join(output_directory, betweenness_distribution_filename)

    if prob_heatmap_filename is None:
        prob_heatmap_outpath = None
    else:
        basename = os.path.basename(prob_heatmap_filename)
        base, ext = os.path.splitext(basename)
        prob_heatmap_filename = f"{base}_{time_str}{ext}"
        prob_heatmap_outpath = os.path.join(output_directory, prob_heatmap_filename)

    if community_composition_filename is None:
        community_composition_outpath = None
    else:
        basename = os.path.basename(community_composition_filename)
        base, ext = os.path.splitext(basename)
        community_composition_filename = f"{base}_{time_str}{ext}"
        community_composition_outpath = os.path.join(output_directory, community_composition_filename)

    if graph_visualization_filename is None:
        graph_visualization_path = None
    else:
        basename = os.path.basename(graph_visualization_filename)
        base, ext = os.path.splitext(basename)
        graph_visualization_filename = f"{base}_{time_str}{ext}"
        graph_visualization_path = os.path.join(output_directory, graph_visualization_filename)

    if neigh_prob_filename is None:
        neigh_prob_path = None
    else:
        basename = os.path.basename(neigh_prob_filename)
        base, ext = os.path.splitext(basename)
        neigh_prob_filename = f"{base}_{time_str}{ext}"
        neigh_prob_path = os.path.join(output_directory, neigh_prob_filename)

    if isinstance(graph, str):
        G = pickle.load(open(graph, 'rb'))
    elif isinstance(graph, nx.Graph):
        G = graph
    else:
        raise ValueError("Invalid graph. Must be a path to a file or a NetworkX Graph.")

    if target_attributes is not None:
        if type(target_attributes) != list:
            pass
        else:
            if len(target_attributes) > 0:
                target_attributes = str(tuple(target_attributes))

    if verbose:
        print(f"--------------------------\nGraph analysis options\n--------------------------\n\n"
              f"\tOptions:\n"
              f"\tgraph_path: {graph}, attribute: {target_attributes}, \n"
              f"\tclustering_method: {clustering_method}, inconsistency_threshold: {inconsistency_threshold}, verbose: {verbose}\n\n")

    if target_attributes is not None:
        df_neigh = analyze_neighborhood_attributes(G, target_attribute = target_attributes)
        probabilities = print_neighbors_prob(df_neigh, target_attributes)
        for (i, j), prob in probabilities.items():
            print(f"P({j}|{i}) = {prob}")
        if neigh_prob_path is not None:
            with open(neigh_prob_path, 'w') as fp:
                for (i, j), prob in probabilities.items():
                    fp.write(f"P({j}|{i}) = {prob}")   

        heat_map_prob(probabilities, df_neigh, target_attributes, prob_heatmap_outpath)

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
    plot_community_composition(G, target_attributes, community_composition_outpath)
    if graph_visualization_path is not None:
        matplotlib_graph_visualization(G, target_attributes, graph_visualization_path, palette = 'viridis')
