import datetime
import os
from typing import Optional, Union, List, Dict
import numpy as np
import pandas as pd
import pickle
import networkx as nx
from scipy.spatial import cKDTree
from scipy.spatial.distance import pdist, squareform
from sklearn.metrics.pairwise import cosine_similarity


def create_graph(
    input_dataframe: Optional[Union[str, pd.DataFrame]] = None,
    preprocessed_dataframe: Optional[Union[str, pd.DataFrame]] = None,
    inferred_columns_filename: Optional[str] = None,
    numeric_columns: Optional[List[str]] = None,
    output_directory: Optional[str] = None,
    graph_filename: Optional[str] = None,
    method: str = "knn",
    k: int = 5,
    distance_threshold: Optional[float] = None,
    similarity_threshold: Optional[float] = None,
    verbose: bool = True,
    overwrite: bool = False,
) -> nx.Graph:
    """
    Creates a graph from a dataframe by connecting points based on a specified method.

    Args:
        input_dataframe: Path to the input dataframe or a pandas DataFrame.
        preprocessed_dataframe: Path to the preprocessed dataframe or a pandas DataFrame.
        inferred_columns_filename: Path to a pickle file containing inferred numeric columns.
        numeric_columns: List of numeric columns to use for graph construction.
        output_directory: Directory to save the output graph.
        graph_filename: Name of the output graph file.
        method: Method for connecting nodes ('knn', 'distance', or 'similarity').
        k: Number of nearest neighbors for the 'knn' method.
        distance_threshold: Distance threshold for the 'distance' method.
        similarity_threshold: Similarity threshold for the 'similarity' method.
        verbose: Whether to print progress messages.
        overwrite: Whether to overwrite existing files.

    Returns:
        A NetworkX graph with nodes and edges based on the specified method.

    Raises:
        ValueError: If invalid inputs are provided.
    """
    # Validate inputs
    if input_dataframe is None and preprocessed_dataframe is None:
        raise ValueError("Either input_dataframe or preprocessed_dataframe must be provided.")

    # Output path management
    output_directory = output_directory or "./"
    os.makedirs(output_directory, exist_ok=True)

    if graph_filename is None:
        base = (
            os.path.splitext(os.path.basename(input_dataframe))[0]
            if isinstance(input_dataframe, str)
            else "graph"
        )
        timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M")
        graph_filename = f"{base}.graphml" if overwrite else f"{base}_{timestamp}.graphml"
    elif not graph_filename.endswith(".graphml"):
        raise ValueError("graph_filename must end with '.graphml'.")

    output_path = os.path.join(output_directory, graph_filename)
    if verbose:
        print(f"{datetime.datetime.now()}: Output path: {output_path}.")

    # Load dataframes
    df = _load_dataframe(input_dataframe)
    df_preprocessed = _load_dataframe(preprocessed_dataframe) if preprocessed_dataframe is not None else df.copy()

    # Ensure dataframes have the same number of rows
    if df.shape[0] != df_preprocessed.shape[0]:
        df = df.dropna().copy()
        if verbose:
            print(f"{datetime.datetime.now()}: Dropped rows with NaN values from the original dataframe.")

    # Create graph and add nodes
    G = nx.Graph()
    df = df.loc[df_preprocessed.index, :].reset_index(drop=True)
    df_preprocessed = df_preprocessed.reset_index(drop=True)
    for i, row in df.iterrows():
        G.add_node(i, **row.to_dict())

    # Prepare numeric data
    if numeric_columns is None:
        numeric_columns = df_preprocessed.select_dtypes(include=["number"]).columns.tolist()
    values = df_preprocessed[numeric_columns].values

    if values.shape[1] == 0:
        raise ValueError("No numeric columns found in the preprocessed dataframe.")

    if verbose:
        print(f"{datetime.datetime.now()}: Using numeric columns: {numeric_columns}")

    # Build edges based on the specified method
    if method == "knn":
        _add_knn_edges(G, values, k)
    elif method == "distance":
        _add_distance_edges(G, values, distance_threshold)
    elif method == "similarity":
        _add_similarity_edges(G, values, similarity_threshold)
    else:
        raise ValueError(f"Unsupported method: {method}")

    # Save graph
    with open(output_path, "wb") as f:
        pickle.dump(G, f)
    if verbose:
        print(f"{datetime.datetime.now()}: Saved graph to {output_path}.")

    return G


def _load_dataframe(data: Union[str, pd.DataFrame]) -> pd.DataFrame:
    """Load a dataframe from a file or return a copy if already a DataFrame."""
    if isinstance(data, str):
        return pd.read_pickle(data) if data.endswith(".pickle") else pd.read_csv(data)
    elif isinstance(data, pd.DataFrame):
        return data.copy()
    raise ValueError("Input must be a file path or a pandas DataFrame.")


def _add_knn_edges(G: nx.Graph, values: np.ndarray, k: int) -> None:
    """Add edges based on k-nearest neighbors."""
    tree = cKDTree(values)
    for i in G.nodes():
        distances, indices = tree.query(values[i], k=k + 1)
        for j in indices[1:]:  # Skip the first index (self)
            G.add_edge(i, j)


def _add_distance_edges(G: nx.Graph, values: np.ndarray, distance_threshold: float) -> None:
    """Add edges based on a distance threshold."""
    tree = cKDTree(values)
    pairs = tree.query_pairs(distance_threshold)
    for i, j in pairs:
        G.add_edge(i, j)


def _add_similarity_edges(G: nx.Graph, values: np.ndarray, similarity_threshold: float) -> None:
    """Add edges based on a similarity threshold."""
    sim_matrix = cosine_similarity(values)
    for i, j in zip(*np.where(sim_matrix >= similarity_threshold)):
        if i != j:  # Avoid self-loops
            G.add_edge(i, j)