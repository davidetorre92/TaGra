import datetime
import networkx as nx
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

def analyze_neighborhood_attributes(graph, target_attribute, return_probs=False):
    """
    Analyzes attributes in the neighborhoods of each node in a graph, optionally returning probabilities.

    Parameters:
    - graph (networkx.Graph): The input graph.
    - attribute_name (str): The name of the node attribute to analyze.
    - return_probs (bool): If True, returns the probability of each attribute in the neighborhood.

    Returns:
    - pd.DataFrame: A DataFrame with each row representing a node. Columns include the node's attribute,
                    degree, and either the count or probability of each attribute in its neighborhood.
    """
    
    unique_attributes = set(nx.get_node_attributes(graph, target_attribute).values())
    data = []
    for node in graph.nodes:
        neighbors = list(graph.neighbors(node))
        neighbor_attrs = [graph.nodes[n].get(target_attribute, None) for n in neighbors]

        attr_counts = {}
        attr_counts[f"node_{target_attribute}"] = graph.nodes[node].get(target_attribute, None)
        attr_counts["node_index"] = node
        attr_counts["degree"] = len(neighbors)

        for attr in unique_attributes:
            if return_probs and attr_counts["degree"] > 0:
                attr_counts[f"p_{attr}"] = neighbor_attrs.count(attr) / attr_counts["degree"]
            else:
                attr_counts[f"n_{attr}"] = neighbor_attrs.count(attr)

        data.append(attr_counts)

    cols = ["node_index", f"node_{target_attribute}", "degree"] + \
           [f"{'p' if return_probs else 'n'}_{attr}" for attr in unique_attributes]
    df = pd.DataFrame(data, columns=cols)

    return df

def print_neighbors_prob(df_neigh, label_col):
    probabilities = {}
    for label_i in df_neigh[f'node_{label_col}'].unique():
        nodes_with_label_i = df_neigh[df_neigh[f'node_{label_col}'] == label_i]
        total_degree_i = nodes_with_label_i['degree'].sum()
        for label_j in df_neigh[f'node_{label_col}'].unique():
            col_name = f'n_{label_j}'
            total_neighbors_with_label_j = nodes_with_label_i[col_name].sum()
            probabilities[(label_i, label_j)] = total_neighbors_with_label_j / total_degree_i if total_degree_i else 0

    return probabilities

def heat_map_prob(probabilities, df_neigh, label_col, prob_heatmap_path):
    labels = sorted(df_neigh[f'node_{label_col}'].unique())
    prob_matrix = pd.DataFrame(index=labels, columns=labels, data=0.0)

    for (i, j), prob in probabilities.items():
        prob_matrix.loc[i, j] = prob

    fig, ax = plt.subplots(figsize=(8, 6))
    cax = ax.imshow(prob_matrix, cmap='seismic', interpolation='nearest')
    fig.colorbar(cax)

    for i in range(len(labels)):
        for j in range(len(labels)):
            text = ax.text(j, i, f"{prob_matrix.iloc[i, j]:.2f}",
                           ha="center", va="center", color="w")

    ax.set_title('Probability Distribution Heatmap')
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels)
    ax.set_yticklabels(labels)
    plt.xlabel('Label j')
    plt.ylabel('Label i')
    if prob_heatmap_path:
        plt.savefig(prob_heatmap_path)
        print(f"{datetime.datetime.now()}: Neighbour probability data saved in {prob_heatmap_path}")

def plot_distribution(data_dict, outpath, bins = None, double_log = True):
    data = data_dict['data']
    if bins is None:
        bins = range(0, max(data))
    hist, bin_edges = np.histogram(data, bins = bins)
    fig, ax = plt.subplots()
    ax.scatter(bin_edges[:-1], hist, alpha=0.75, edgecolor='black')
    ax.set_title(data_dict['title'])
    ax.set_xlim((1,None))
    ax.set_ylim((1,None))
    ax.set_xlabel(data_dict['xlabel'])
    ax.set_ylabel(data_dict['ylabel'])
    if double_log:
        ax.set_xscale('log')
        ax.set_yscale('log')
    ax.grid()
    if outpath:
        fig.savefig(outpath)
        print(f"{datetime.datetime.now()}: {data_dict['title']} saved in {outpath}")

def plot_community_composition(G, attribute_name, outpath):
    communities_generator = nx.algorithms.community.girvan_newman(G)
    top_level_communities = next(communities_generator)
    communities = [list(c) for c in sorted(top_level_communities, key=len, reverse=True)]

    if attribute_name is not None:
        labels_per_node = [G.nodes[node][attribute_name] for node in G.nodes()]
        unique_labels = np.unique(labels_per_node)
    else:
        labels_per_node = [0 for node in G.nodes()]
        unique_labels = [0]

    community_compositions = {}
    if attribute_name is not None:
        for comm_id, community in enumerate(communities):
            labels_community = [G.nodes[node][attribute_name] for node in community]
            unique_labels, counts = np.unique(labels_community, return_counts=True)
            community_compositions[comm_id] = {label: 0 for label in unique_labels}
            for label, count in zip(unique_labels, counts):
                community_compositions[comm_id][label] = count
    else:
        for comm_id, community in enumerate(communities):
            labels_community = [0 for node in community]
            unique_labels, counts = np.unique(labels_community, return_counts=True)
            community_compositions[comm_id] = {label: 0 for label in unique_labels}
            for label, count in zip(unique_labels, counts):
                community_compositions[comm_id][label] = count

    indices = list(community_compositions.keys())
    bar_width = 0.35
    fig, ax = plt.subplots(figsize = (15,15))

    bottoms = [0] * len(indices)
    for label in unique_labels:
        values = [community_compositions[idx].get(label, 0) for idx in indices]
        ax.bar(indices, values, bar_width, label=label, bottom=bottoms)
        bottoms = [bottom + value for bottom, value in zip(bottoms, values)]

    ax.set_xticks(indices)
    ax.set_xticklabels(indices)
    ax.set_xlabel('Community ID')
    ax.set_ylabel('Counts')
    ax.set_title('Counts of outcomes by community ID')
    ax.grid()
    ax.legend()

    if outpath:
        fig.savefig(outpath)
        print(f"{datetime.datetime.now()}: Community composition saved in {outpath}")
        
def matplotlib_graph_visualization(G, attribute = None, outpath = None, palette = 'seismic', pos = None):
    plt.figure(figsize=(10, 10))
    if pos is None:
      pos = nx.spring_layout(G, seed=2112)

    node_color = []
    if attribute is not None:
        classification_attribute_name = attribute
        y = np.array([G.nodes[node][classification_attribute_name] for node in G.nodes()])
        unique = np.unique(y)
        unique_dict = {key: index for index, key in enumerate(unique)}
        colors = np.linspace(0, 1, len(unique))
        cmap = plt.get_cmap(palette, len(unique))
        color_array = cmap(colors)
        node_color = [color_array[unique_dict[key]] for key in y]
    nx.draw(G, pos, with_labels=True, node_size=50, font_size=8, node_color = node_color)
    plt.title("Graph of Relations Based on Manifold Learning Transformed Data")
    if outpath:
        plt.savefig(outpath)
        print(f'{datetime.datetime.now()}: Graph saved in {outpath}')

def measure_mixing_matrix(G, communities):
    community_edge_count = {(i, j): 0 for i in communities.keys() for j in communities.keys()}

    # Create a mapping from node to its community
    node_to_community = {}
    for community, nodes in communities.items():
        for node in nodes:
            node_to_community[node] = community

    # Check if all nodes in the graph are covered by the communities
    graph_nodes = set(G.nodes())
    community_nodes = set(node_to_community.keys())
    if graph_nodes != community_nodes:
        missing_in_communities = graph_nodes - community_nodes
        missing_in_graph = community_nodes - graph_nodes
        raise ValueError(
            f"The nodes in the communities do not match the nodes in the graph.\n"
            f"Nodes in the graph not in communities: {missing_in_communities}\n"
            f"Nodes in communities not in the graph: {missing_in_graph}"
        )

    # Iterate over all edges in the graph
    for source, target in G.edges():
        if source in node_to_community and target in node_to_community:
            c_source = node_to_community[source]
            c_target = node_to_community[target]
            if not G.is_directed():
                community_edge_count[c_source, c_target] += 1
                community_edge_count[c_target, c_source] += 1
            else:
                community_edge_count[c_source, c_target] += 1

    return community_edge_count
