import datetime
import os
import numpy as np
import pandas as pd
import pickle
import networkx as nx
from scipy.spatial.distance import pdist, squareform
from sklearn.metrics.pairwise import cosine_similarity

def create_graph(input_dataframe=None, preprocessed_dataframe=None, output_path=None, method='knn', threshold=0.75, k=5, verbose=True):
    if verbose:
        print(f"--------------------------\nGraph creation options\n--------------------------\n\n"
              f"\tOptions:\n"
              f"\input_dataframe: {input_dataframe}, preprocessed_dataframe: {preprocessed_dataframe}, \n"
              f"\toutput_path: {output_path}, method: {method}, threshold: {threshold}, k: {k}, verbose: {verbose}\n\n")

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
        df_preprocessed = df.dropna().copy()
        if verbose:
            print(f"{datetime.datetime.now()}: Dropped rows with NaN values from the original dataframe due to mismatch with preprocessed dataframe.")

    G = nx.Graph()

    for i, row in df.iterrows():
        G.add_node(i, **row.to_dict())

    if method == 'knn':
        numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
        values = df_preprocessed.select_dtypes(include=numerics).values
        dists = squareform(pdist(values, metric='euclidean'))
        for i in range(len(df)):
            knn_indices = np.argsort(dists[i])[:k + 1]
            for j in knn_indices:
                if i != j:
                    G.add_edge(i, j)
    elif method == 'distance_threshold':
        numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
        values = df_preprocessed.select_dtypes(include=numerics).values
        dists = squareform(pdist(values, metric='euclidean'))
        for i in range(len(df)):
            for j in range(i + 1, len(df)):
                if dists[i, j] <= threshold:
                    G.add_edge(i, j)
    elif method == 'similarity':
        numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
        values = df_preprocessed.select_dtypes(include=numerics).values
        sim_matrix = cosine_similarity(values)
        for i in range(len(df)):
            for j in range(i + 1, len(df)):
                if sim_matrix[i, j] >= threshold:
                    G.add_edge(i, j)
    else:
        raise ValueError(f"Unsupported method: {method}")

    if output_path is None:
        if isinstance(input_dataframe, str):
            base, _ = os.path.splitext(input_dataframe)
            output_path = f"{base}_graph_{datetime.datetime.now().strftime('%Y%m%d%H%M')}.pickle"
        else:
            output_path = f"./graph_{datetime.datetime.now().strftime('%Y%m%d%H%M')}.pickle"

    pickle.dump(G, open(output_path, 'wb'))
    if verbose:
        print(f"{datetime.datetime.now()}: Nodes attributes:\n" 
              f"{G.nodes[0].keys()}")
        print(f"{datetime.datetime.now()}: Saved graph to {output_path}.")

    return G
