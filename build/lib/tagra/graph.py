import datetime
import os
import numpy as np
import pandas as pd
import pickle
import networkx as nx
from scipy.spatial.distance import pdist, squareform
from sklearn.metrics.pairwise import cosine_similarity
import pdb

def create_graph(input_dataframe=None, preprocessed_dataframe=None,
                 inferred_columns_filename=None, numeric_columns=None,
                 output_directory=None, graph_filename=None, method='knn', k=5,
                 distance_threshold=None,
                 similarity_threshold=None, verbose=True, overwrite=False):
    if verbose:
        print(f"--------------------------\nGraph creation options\n--------------------------\n\n"
              f"\tOptions:\n"
              f"\tinput_dataframe: {input_dataframe}, preprocessed_dataframe: {preprocessed_dataframe}\n"
              f"\toutput_directory: {output_directory}, graph_filename: {graph_filename},\nmethod: {method}, distance_threshold: {distance_threshold}, similarity_threshold: {similarity_threshold}, k: {k}, verbose: {verbose}\n"
              f"\toverwrite: {overwrite}\n")

    # Output path managing
    if output_directory is None:
        output_directory = './'
    if os.path.exists(output_directory) is False:
        os.mkdir(output_directory)
        print(f"{datetime.datetime.now()}: Output directory created: {output_directory}.")
    if graph_filename is None:
        if isinstance(input_dataframe, str):
            basename = os.path.basename(input_dataframe)
            base, _ = os.path.splitext(basename)
            if overwrite:
                graph_filename = f"{base}.graphml"
            else:
                graph_filename = f"{base}_{datetime.datetime.now().strftime('%Y%m%d%H%M')}.graphml"
        else:
            if overwrite:
                graph_filename = f"graph.pickle"
            else:
                graph_filename = f"graph_{datetime.datetime.now().strftime('%Y%m%d%H%M')}.pickle"
    else:
        basename = os.path.basename(graph_filename)
        base, ext = os.path.splitext(basename)
        if ext == '.graphml':
            pass
        else:
            raise ValueError("Invalid graph_filename. Must be a .graphml file.")
        if overwrite:
            graph_filename = f"{base}.graphml"
        else:
            graph_filename = f"{base}_{datetime.datetime.now().strftime('%Y%m%d%H%M')}.graphml"

    if inferred_columns_filename is not None:
        inferred_columns_dictionary_path = os.path.join(output_directory, inferred_columns_filename)
        inferred_columns_dictionary = pickle.load(open(inferred_columns_dictionary_path, 'rb'))
        numeric_columns = inferred_columns_dictionary['numeric_columns']
    output_path = os.path.join(output_directory, graph_filename)
    print(f"{datetime.datetime.now()}: Output path for the preprocessed file: {output_path}.")

    # Load dataframe
    if isinstance(input_dataframe, str):
        df = pd.read_pickle(input_dataframe) if input_dataframe.endswith('.pickle') else pd.read_csv(input_dataframe)
    elif isinstance(input_dataframe, pd.DataFrame):
        df = input_dataframe.copy()
    else:
        raise ValueError("Invalid input_dataframe. Must be a path to a file or a pandas DataFrame.")

    if preprocessed_dataframe is None:
        df_preprocessed = df.copy()
    else:
        if isinstance(preprocessed_dataframe, str):
            df_preprocessed = pd.read_pickle(preprocessed_dataframe) if preprocessed_dataframe.endswith('.pickle') else pd.read_csv(preprocessed_dataframe)
        elif isinstance(preprocessed_dataframe, pd.DataFrame):
            df_preprocessed = preprocessed_dataframe.copy()
        else:
            raise ValueError("Invalid preprocessed_dataframe. Must be a path to a file or a pandas DataFrame.")

    if df.shape[0] != df_preprocessed.shape[0]:
        df = df.dropna().copy()
        if verbose:
            print(f"{datetime.datetime.now()}: Dropped rows with NaN values from the original dataframe due to mismatch with preprocessed dataframe.")

    G = nx.Graph()
    # Collect available indices (i.e. nodes) in the graph: in df_preprocessed we may have dropped nans!
    df = df.loc[df_preprocessed.index,:]
    # Reset indices to avoid unsorted node indices
    df = df.reset_index(drop=True)
    df_preprocessed = df_preprocessed.reset_index(drop=True)
    for i, row in df.iterrows():
        G.add_node(i, **row.to_dict())

    if numeric_columns is None:
        numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
        data = df_preprocessed.select_dtypes(include=numerics)
        numeric_columns = list(data.columns)
        values = data.values
    elif len(numeric_columns) > 0:
        values = df_preprocessed[numeric_columns].values
    else:
        raise ValueError("No numeric columns specified.")

    if values.shape[1]  == 0:
        raise ValueError("No numeric columns found in the preprocessed dataframe.")
    
    if verbose:
        print(f"{datetime.datetime.now()}: Evaluating edges from the following columns: {numeric_columns}")
    indices = [i for i in G.nodes()]
    if method == 'knn':
        dists = squareform(pdist(values, metric='euclidean'))
        for i in indices:
            knn_indices = np.argsort(dists[i])[:k + 1]
            for j in knn_indices:
                if i != j:
                    G.add_edge(i, j)
    elif method == 'distance':
        dists = squareform(pdist(values, metric='euclidean'))
        for ind_i in range(len(indices)):
            for ind_j in range(ind_i+1, len(indices)):
                i = indices[ind_i]
                j = indices[ind_j]
                if dists[i, j] <= distance_threshold:
                    G.add_edge(i, j)
    elif method == 'similarity':
        sim_matrix = cosine_similarity(values)

        for ind_i in range(len(indices)):
            for ind_j in range(ind_i+1, len(indices)):
                i = indices[ind_i]
                j = indices[ind_j]
                if sim_matrix[i, j] >= similarity_threshold:
                    G.add_edge(i, j)
    else:
        raise ValueError(f"Unsupported method: {method}")

    pickle.dump(G, open(output_path, 'wb'))
    if verbose:
        print(f"{datetime.datetime.now()}: Nodes attributes:\n" 
              f"{G.nodes[0].keys()}")
        print(f"{datetime.datetime.now()}: Saved graph to {output_path}.")

    return G
