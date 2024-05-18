import json
import os
import sys
import argparse
import pickle
import pandas as pd
from tagra.preprocessing import preprocess_dataframe
from tagra.graph import create_graph
from tagra.analysis import analyze_graph
from tagra.config import *

def main(config_path):
    config = load_config(config_path)
    # Preprocessing
    df_preprocessed = preprocess_dataframe(
        input_dataframe=config['input_dataframe'],
        output_path=config['output_path'],
        numeric_cols=config['numeric_columns'],
        categorical_cols=config['categorical_columns'],
        unknown_col_action=config['unknown_column_action'],
        ignore_cols=config['ignore_columns'],
        threshold=config['numeric_threshold'],
        numeric_scaling=config['numeric_scaling'],
        categorical_encoding=config['categorical_encoding'],
        nan_action=config['nan_action'],
        nan_threshold=config['nan_threshold'],
        verbose=config['verbose'],
        manifold_method=config['manifold_learning'],
        manifold_dim=config['manifold_dimension']
    )

    # Save preprocessed file
    preprocessed_path = 'preprocessed_moons.csv'
    df_preprocessed.to_csv(preprocessed_path, index=False)

    # Graph Creation
    graph = create_graph(
        input_dataframe=config['input_dataframe'],
        preprocessed_dataframe=df_preprocessed,
        method=config['method'],
        k=config['k'],
        threshold=config['threshold'],
        verbose=config['verbose']
    )

    # Salva il grafo
    graph_path = 'moons_graph.pickle'
    pickle.dump(graph, open(graph_path, 'wb'))

    # Graph Analysis
    analyze_graph(
        graph_path=graph_path,
        attribute='class',
        clustering_method=config['clustering_method'],
        inconsistency_threshold=config['inconsistency_threshold'],
        verbose=config['verbose'],
        degree_distribution_outpath=config['degree_distribution_outpath'],
        betweenness_distribution_outpath=config['betweenness_distribution_outpath'],
        community_composition_outpath=config['community_composition_outpath'],
        graph_visualization_path=config['graph_visualization_path'],
        plot_graph=config['plot_graph']
    )

    print("Analysis complete.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run TaGra example with configuration file.')
    parser.add_argument('-c', '--config', type=str, required=False, default=None, help='Path to the configuration file.')
    args = parser.parse_args()

    if args.config is not None and not os.path.isfile(args.config):
        print(f"Error: The configuration file {args.config} does not exist.")
        sys.exit(1)

    main(args.config)
