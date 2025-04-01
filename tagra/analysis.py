import pickle
import datetime
import networkx as nx
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from datetime import datetime
from sklearn.metrics import silhouette_score
from scipy.stats import chi2_contingency
import random

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
                  verbose=True,
                  pos=None,
                  output_directory=None,
                  neigh_prob_filename=None,
                  degree_distribution_filename=None,
                  prob_heatmap_filename=None,
                  community_filename=None,
                  graph_visualization_filename=None,
                  network_metrics_filename=None,
                  overwrite=False):
    """
    Analyzes a graph and generates various metrics and visualizations.
    
    Parameters:
    -----------
    graph : networkx.Graph or str
        The graph to analyze, or a path to a pickle file containing a graph.
    target_attributes : str or list, optional
        Target attributes for coloring and analysis.
    verbose : bool, default=True
        Whether to print detailed information.
    pos : dict, optional
        Node positions for visualization.
    output_directory : str, optional
        Directory to save output files.
    neigh_prob_filename : str, optional
        Filename for neighborhood probability statistics.
    degree_distribution_filename : str, optional
        Filename for degree distribution plot.
    prob_heatmap_filename : str, optional
        Filename for neighborhood probability heatmap.
    community_filename : str, optional
        Filename for community composition histogram.
    graph_visualization_filename : str, optional
        Filename for graph visualization.
    network_metrics_filename : str, optional
        Filename for network metrics report.
    overwrite : bool, default=False
        Whether to overwrite existing files.
        
    Returns:
    --------
    dict
        Dictionary containing computed metrics.
    """
    
    # Output path managing
    time_str = datetime.now().strftime('%Y%m%d%H%M')
    if output_directory is None:
        output_directory = './'
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)
        print(f"{datetime.now()}: Output directory created: {output_directory}.")

    # Configure output paths
    paths = {}
    for filename_param, filename_value in [
        ('degree_distribution_filename', degree_distribution_filename),
        ('prob_heatmap_filename', prob_heatmap_filename),
        ('community_filename', community_filename),
        ('graph_visualization_filename', graph_visualization_filename),
        ('neigh_prob_filename', neigh_prob_filename),
        ('network_metrics_filename', network_metrics_filename)
    ]:
        if filename_value is not None:
            if not overwrite:
                basename = os.path.basename(filename_value)
                base, ext = os.path.splitext(basename)
                filename_value = f"{base}_{time_str}{ext}"
            paths[filename_param] = os.path.join(output_directory, filename_value)
        else:
            paths[filename_param] = None

    if verbose:
        print(f"{datetime.now()}: Starting graph analysis...")
        print(f"{datetime.now()}: Configured output paths for analysis results.")

    # Load graph if path is provided
    if isinstance(graph, str):
        if verbose:
            print(f"{datetime.now()}: Loading graph from file: {graph}")
        G = pickle.load(open(graph, 'rb'))
    elif isinstance(graph, nx.Graph):
        if verbose:
            print(f"{datetime.now()}: Using provided NetworkX graph object.")
        G = graph
    else:
        raise ValueError("Invalid graph. Must be a path to a file or a NetworkX Graph.")

    # Handle target attributes
    if target_attributes is not None and isinstance(target_attributes, list) and len(target_attributes) > 0:
        target_attributes = str(tuple(target_attributes))
        if verbose:
            print(f"{datetime.now()}: Using target attributes: {target_attributes}")

    if verbose:
        print(f"--------------------------\nGraph analysis options\n--------------------------\n\n"
              f"\tOptions:\n"
              f"\tgraph: {graph}, attribute: {target_attributes}, \n"
              f"\toverwrite: {overwrite}\n\n")
        print(f"{datetime.now()}: Graph has {G.number_of_nodes()} nodes and {G.number_of_edges()} edges.")

    # Dictionary to store all computed metrics
    metrics = {}
    
    # Calculate basic graph metrics
    if verbose:
        print(f"{datetime.now()}: Calculating basic graph metrics...")
    metrics['nodes'] = G.number_of_nodes()
    metrics['edges'] = G.number_of_edges()
    metrics['density'] = nx.density(G)
    # if verbose:
    #     print(f"{datetime.now()}: Graph density: {metrics['density']:.6f}")
    
    if verbose:
        print(f"{datetime.now()}: Calculating average clustering coefficient...")
    metrics['avg_clustering'] = nx.average_clustering(G)
    # if verbose:
    #     print(f"{datetime.now()}: Average clustering coefficient: {metrics['avg_clustering']:.6f}")
    
    # Connected components analysis
    if verbose:
        print(f"{datetime.now()}: Analyzing connected components...")
    components = list(nx.connected_components(G))
    metrics['connected_components'] = len(components)
    metrics['largest_component_size'] = len(max(components, key=len))
    # if verbose:
    #     print(f"{datetime.now()}: Found {metrics['connected_components']} connected components.")
    #     print(f"{datetime.now()}: Largest component has {metrics['largest_component_size']} nodes.")
    
    # Calculate additional network metrics (always calculate these even if not saving to file)
    if verbose:
        print(f"{datetime.now()}: Calculating additional network metrics...")
    
    # Calculate assortativity
    try:
        if verbose:
            print(f"{datetime.now()}: Calculating degree assortativity coefficient...")
        metrics['assortativity'] = nx.degree_assortativity_coefficient(G)
        # if verbose:
        #     print(f"{datetime.now()}: Assortativity coefficient: {metrics['assortativity']:.6f}")
    except Exception as e:
        if verbose:
            print(f"{datetime.now()}: Could not calculate assortativity: {str(e)}")
        metrics['assortativity'] = None
                
    # Perform neighborhood analysis if target attributes provided
    if target_attributes is not None:
        if verbose:
            print(f"{datetime.now()}: Performing statistical analysis for target attribute: {target_attributes}")
        
        # Chi-square test for neighborhood attributes
        try:
            if verbose:
                print(f"{datetime.now()}: Analyzing neighborhood attributes...")
            df_neigh = analyze_neighborhood_attributes(G, target_attribute=target_attributes)
            target_values = df_neigh[f'node_{target_attributes}'].unique()
            # if verbose:
            #     print(f"{datetime.now()}: Found {len(target_values)} unique values for target attribute.")
            
            # Create contingency table
            if verbose:
                print(f"{datetime.now()}: Creating contingency table for chi-square test...")
            contingency_table = []
            for target_value in target_values:
                row = []
                nodes_with_value = df_neigh[df_neigh[f'node_{target_attributes}'] == target_value]
                for neigh_value in target_values:
                    col_name = f'n_{neigh_value}'
                    total_neighbors = nodes_with_value[col_name].sum()
                    row.append(total_neighbors)
                contingency_table.append(row)
            
            # Chi-square test
            if len(contingency_table) > 1 and all(sum(row) > 0 for row in contingency_table):
                if verbose:
                    print(f"{datetime.now()}: Performing chi-square test...")
                chi2, p_value, dof, expected = chi2_contingency(contingency_table)
                metrics['chi2_stat'] = chi2
                metrics['chi2_p_value'] = p_value
                # if verbose:
                #     print(f"{datetime.now()}: Chi-square statistic: {chi2:.6f}, p-value: {p_value:.6f}")
            else:
                if verbose:
                    print(f"{datetime.now()}: Skipping chi-square test - insufficient data.")
                metrics['chi2_stat'] = None
                metrics['chi2_p_value'] = None
                
            # Permutation test for neighborhood patterns
            if len(target_values) > 1:
                if verbose:
                    print(f"{datetime.now()}: Calculating homophily score...")
                original_probs = print_neighbors_prob(df_neigh, target_attributes)
                
                # Calculate diagonal sum (homophily measure)
                diagonal_sum = sum(original_probs.get((i, i), 0) for i in target_values)
                metrics['homophily_score'] = diagonal_sum / len(target_values)
                # if verbose:
                #     print(f"{datetime.now()}: Homophily score: {metrics['homophily_score']:.6f}")
                
                # Permutation test (compare with randomized attribute assignments)
                if verbose:
                    print(f"{datetime.now()}: Starting permutation test (this may take a moment)...")
                n_permutations = 100
                permutation_diagonals = []
                
                for i in range(n_permutations):
                    if verbose and i % 25 == 0:
                        print(f"{datetime.now()}: Permutation test progress: {i}/{n_permutations}...")
                        
                    # Create copy of graph with shuffled attributes
                    G_perm = G.copy()
                    attr_values = [G.nodes[n].get(target_attributes, 'None') for n in G.nodes()]
                    random.shuffle(attr_values)
                    
                    for i, node in enumerate(G.nodes()):
                        G_perm.nodes[node][target_attributes] = attr_values[i]
                        
                    # Calculate neighborhood probabilities
                    df_neigh_perm = analyze_neighborhood_attributes(G_perm, target_attribute=target_attributes)
                    perm_probs = print_neighbors_prob(df_neigh_perm, target_attributes)
                    
                    # Calculate diagonal sum
                    perm_diagonal = sum(perm_probs.get((i, i), 0) for i in target_values)
                    permutation_diagonals.append(perm_diagonal / len(target_values))
                
                # Calculate permutation test p-value
                p_value = sum(d >= metrics['homophily_score'] for d in permutation_diagonals) / n_permutations
                metrics['homophily_p_value'] = p_value
                # if verbose:
                #     print(f"{datetime.now()}: Permutation test p-value: {p_value:.6f}")
                
                # Z-score compared to random
                if len(permutation_diagonals) > 1:
                    if verbose:
                        print(f"{datetime.now()}: Calculating Z-score...")
                    perm_mean = np.mean(permutation_diagonals)
                    perm_std = np.std(permutation_diagonals)
                    if perm_std > 0:
                        metrics['homophily_z_score'] = (metrics['homophily_score'] - perm_mean) / perm_std
                        # if verbose:
                        #     print(f"{datetime.now()}: Homophily Z-score: {metrics['homophily_z_score']:.6f}")
                    else:
                        metrics['homophily_z_score'] = 0
                        if verbose:
                            print(f"{datetime.now()}: Homophily Z-score: 0 (zero standard deviation in permutations)")
                else:
                    metrics['homophily_z_score'] = 0
                    if verbose:
                        print(f"{datetime.now()}: Could not calculate Z-score - insufficient permutation data.")
                    
        except Exception as e:
            if verbose:
                print(f"{datetime.now()}: Error in neighborhood analysis: {str(e)}")
            metrics['chi2_stat'] = None
            metrics['chi2_p_value'] = None
            metrics['homophily_score'] = None
            metrics['homophily_p_value'] = None
            metrics['homophily_z_score'] = None
    
    # Community detection using Girvan-Newman
    try:
        if verbose:
            print(f"{datetime.now()}: Detecting communities using Girvan-Newman algorithm...")
        communities_generator = nx.algorithms.community.girvan_newman(G)
        top_level_communities = next(communities_generator)
        communities = [list(c) for c in sorted(top_level_communities, key=len, reverse=True)]
        metrics['community_count'] = len(communities)
        if verbose:
            print(f"{datetime.now()}: Found {metrics['community_count']} communities.")
        
        # Calculate modularity score if communities exist
        if metrics['community_count'] > 1:
            if verbose:
                print(f"{datetime.now()}: Calculating modularity score...")
            community_dict = {}
            for i, comm in enumerate(communities):
                for node in comm:
                    community_dict[node] = i
            metrics['modularity'] = nx.algorithms.community.modularity(G, communities)
            if verbose:
                print(f"{datetime.now()}: Modularity score: {metrics['modularity']:.6f}")
        else:
            metrics['modularity'] = 0
            if verbose:
                print(f"{datetime.now()}: Only one community found, modularity set to 0.")
            
    except Exception as e:
        if verbose:
            print(f"{datetime.now()}: Error in community detection: {str(e)}")
        metrics['community_count'] = 0
        metrics['modularity'] = None


    # Function to format metrics output (used for both file and console)
    def format_metrics_report():
        lines = []
        lines.append("Network Metrics Report")
        lines.append("=====================\n")
        lines.append(f"Generated on: {datetime.now()}\n")
        
        lines.append("Basic Metrics:")
        lines.append(f"- Nodes: {metrics['nodes']}")
        lines.append(f"- Edges: {metrics['edges']}")
        lines.append(f"- Density: {metrics['density']:.6f} (Fraction of possible connections that actually exist)")
        lines.append(f"- Average Clustering Coefficient: {metrics['avg_clustering']:.6f} (Measure of how nodes tend to cluster together)")
        lines.append(f"- Connected Components: {metrics['connected_components']} (Number of separate subgraphs)")
        lines.append(f"- Largest Component Size: {metrics['largest_component_size']} nodes ({metrics['largest_component_size']/metrics['nodes']*100:.1f}% of graph)")
        
        if 'assortativity' in metrics and metrics['assortativity'] is not None:
            lines.append(f"- Assortativity Coefficient: {metrics['assortativity']:.6f} (Tendency of nodes to connect to similar nodes by degree)")
        
        if 'community_count' in metrics:
            lines.append(f"- Community Count: {metrics['community_count']} (Detected using Girvan-Newman algorithm)")
        
        if 'modularity' in metrics and metrics['modularity'] is not None:
            lines.append(f"- Modularity Score: {metrics['modularity']:.6f} (Strength of division into communities)")
        
        if target_attributes is not None:
            lines.append("\nTarget Attribute Analysis:")
            
            if 'chi2_stat' in metrics and metrics['chi2_stat'] is not None:
                lines.append(f"- Neighborhood Pattern Chi-square: {metrics['chi2_stat']:.6f}")
                lines.append(f"  This tests whether nodes connect to neighbors with particular target attributes")
                lines.append(f"  in a non-random way. Higher values suggest stronger patterns.")
                lines.append(f"- Chi-square p-value: {metrics['chi2_p_value']:.6f}")
                if metrics['chi2_p_value'] < 0.05:
                    lines.append(f"  Significant result (p < 0.05): The pattern of connections between nodes")
                    lines.append(f"  with different target attributes is not random.")
                else:
                    lines.append(f"  Non-significant result (p >= 0.05): No strong evidence that connections")
                    lines.append(f"  follow a pattern based on target attributes.")
            
            if 'homophily_score' in metrics and metrics['homophily_score'] is not None:
                lines.append(f"- Homophily Score: {metrics['homophily_score']:.6f}")
                lines.append(f"  Measures how often nodes connect to others with the same target attribute.")
                lines.append(f"  Score of 1.0 = perfect homophily (nodes only connect to same class)")
                lines.append(f"  Score of {1.0/len(df_neigh[f'node_{target_attributes}'].unique()):.2f} = random connections")
                
            if 'homophily_p_value' in metrics and metrics['homophily_p_value'] is not None:
                lines.append(f"- Homophily Permutation Test p-value: {metrics['homophily_p_value']:.6f}")
                if metrics['homophily_p_value'] < 0.05:
                    lines.append(f"  Significant result (p < 0.05): The observed homophily is unlikely")
                    lines.append(f"  to occur by random chance.")
                else:
                    lines.append(f"  Non-significant result (p >= 0.05): The observed homophily could")
                    lines.append(f"  be explained by random chance.")
                    
            if 'homophily_z_score' in metrics and metrics['homophily_z_score'] is not None:
                lines.append(f"- Homophily Z-score: {metrics['homophily_z_score']:.6f}")
                lines.append(f"  How many standard deviations the homophily score is from random expectation.")
                if abs(metrics['homophily_z_score']) > 2:
                    lines.append(f"  |Z| > 2: Strong evidence of non-random connectivity pattern.")
        
        lines.append("\nInterpretation:")
        lines.append("- Graph Density: " + 
                    ("Very sparse graph" if metrics['density'] < 0.01 else
                    "Sparse graph" if metrics['density'] < 0.1 else
                    "Moderately connected graph" if metrics['density'] < 0.3 else
                    "Densely connected graph"))
            
        lines.append("- Community Structure: " +
                    ("Weak community structure" if 'modularity' in metrics and metrics['modularity'] is not None and metrics['modularity'] < 0.2 else
                    "Moderate community structure" if 'modularity' in metrics and metrics['modularity'] is not None and metrics['modularity'] < 0.4 else
                    "Strong community structure" if 'modularity' in metrics and metrics['modularity'] is not None else
                    "Not analyzed"))
            
        if target_attributes is not None and 'homophily_score' in metrics and metrics['homophily_score'] is not None:
            lines.append("- Class Separation: " +
                        ("Poor separation of target classes" if metrics['homophily_score'] < 0.3 else
                        "Moderate separation of target classes" if metrics['homophily_score'] < 0.9 else
                        "Strong separation of target classes"))
            
            if 'chi2_p_value' in metrics and metrics['chi2_p_value'] is not None:
                if metrics['chi2_p_value'] < 0.05:
                    lines.append("  The connection patterns by class are statistically significant.")
                else:
                    lines.append("  The connection patterns by class may be due to random chance.")
        
        return lines    
    # Write metrics to file if filename is provided
    if paths['network_metrics_filename']:
        if verbose:
            print(f"{datetime.now()}: Writing network metrics report to file...")
            
        with open(paths['network_metrics_filename'], 'w') as f:
            for line in format_metrics_report():
                f.write(line + "\n")
        
        if verbose:
            print(f"{datetime.now()}: Network metrics saved to {paths['network_metrics_filename']}")
    else:
        # Print metrics to console if no filename is provided
        if verbose:
            print(f"{datetime.now()}: Network metrics report (no output file specified):")
            print("-" * 50)
            for line in format_metrics_report():
                print(line)
            print("-" * 50)

    # Perform neighborhood analysis if target attributes provided
    if target_attributes is not None:
        if verbose:
            print(f"{datetime.now()}: Analyzing neighborhood probabilities...")
            
        df_neigh = analyze_neighborhood_attributes(G, target_attribute=target_attributes)
        probabilities = print_neighbors_prob(df_neigh, target_attributes)
        
        # Print probabilities
        if verbose:
            print(f"{datetime.now()}: Neighborhood probability results:")
            
        for (i, j), prob in probabilities.items():
            print(f"P({j}|{i}) = {prob:.4f}")
        
        # Save to file if specified
        if paths['neigh_prob_filename'] is not None:
            if verbose:
                print(f"{datetime.now()}: Saving neighborhood probabilities to file...")
                
            with open(paths['neigh_prob_filename'], 'w') as fp:
                for (i, j), prob in probabilities.items():
                    fp.write(f"P({j}|{i}) = {prob:.4f}\n")
            
            if verbose:
                print(f"{datetime.now()}: Neighborhood probabilities saved to {paths['neigh_prob_filename']}")
        
        # Create heatmap
        if paths['prob_heatmap_filename'] is not None:
            if verbose:
                print(f"{datetime.now()}: Creating probability heatmap...")
                
            heat_map_prob(probabilities, df_neigh, target_attributes, paths['prob_heatmap_filename'], verbose)
            

    # Create degree distribution plot
    if paths['degree_distribution_filename'] is not None:
        if verbose:
            print(f"{datetime.now()}: Creating degree distribution plot...")
            
        degree_data = {'data': [degree for _, degree in G.degree()],
                       'title': 'Degree distribution',
                       'xlabel': 'Degree',
                       'ylabel': 'Number of Nodes'}
        plot_distribution(degree_data, paths['degree_distribution_filename'], verbose)
            
    # Create community composition plot
    if paths['community_filename'] is not None:
        if verbose:
            print(f"{datetime.now()}: Creating community composition plot...")
            
        plot_community_composition(G, target_attributes, communities, paths['community_filename'], verbose)
        
    
    # Create graph visualization
    if paths['graph_visualization_filename'] is not None:
        if verbose:
            print(f"{datetime.now()}: Creating graph visualization...")
            
        matplotlib_graph_visualization(G, target_attributes, paths['graph_visualization_filename'], verbose, pos=pos)
            
    if verbose:
        print(f"{datetime.now()}: Graph analysis complete.")

    return metrics