import json
import os
import sys
import argparse
from datetime import datetime
import pandas as pd
from tagra.preprocessing import preprocess_dataframe
from tagra.graph import create_graph
from tagra.analysis import analyze_graph
from tagra.config import *

def main(config_path):
    start_time = datetime.now()

    config = load_config(config_path)
    # Preprocessing
    df_preprocessed = preprocess_dataframe(
        input_dataframe=config['input_dataframe'],
        output_directory=config['output_directory'],
        preprocessed_filename=config['preprocessed_filename'],
        inferred_columns_filename=config['inferred_columns_filename'],
        numeric_columns=config['numeric_columns'],
        categorical_columns=config['categorical_columns'],
        target_columns=config['target_columns'],
        unknown_column_action=config['unknown_column_action'],
        ignore_columns=config['ignore_columns'],
        numeric_threshold=config['numeric_threshold'],
        numeric_scaling=config['numeric_scaling'],
        categorical_encoding=config['categorical_encoding'],
        nan_action=config['nan_action'],
        nan_threshold=config['nan_threshold'],
        verbose=config['verbose'],
        manifold_method=config['manifold_method'],
        manifold_dim=config['manifold_dimension'],
        overwrite=config['overwrite']
    )

    # Graph Creation
    graph = create_graph(
        input_dataframe=config['input_dataframe'],
        output_directory=config['output_directory'],
        graph_filename=config['graph_filename'],
        inferred_columns_filename=config['inferred_columns_filename'],
        numeric_columns=config['numeric_columns'],
        preprocessed_dataframe=df_preprocessed,
        similarity_threshold=config['similarity_threshold'],
        distance_threshold=config['distance_threshold'],
        method=config['method'],
        k=config['k'],
        verbose=config['verbose'],
        overwrite=config['overwrite']
    )

    # Graph Analysis
    analyze_graph(
        graph,
        target_attributes=config['target_columns'],
        clustering_method=config['clustering_method'],
        inconsistency_threshold=config['inconsistency_threshold'],
        verbose=config['verbose'],
        output_directory=config['output_directory'],
        degree_distribution_filename=config['degree_distribution_filename'],
        betweenness_distribution_filename=config['betweenness_distribution_filename'],
        community_composition_filename=config['community_composition_filename'],
        graph_visualization_filename=config['graph_visualization_filename'],
        prob_heatmap_filename=config['prob_heatmap_filename'],
        overwrite=config['overwrite']
    )

    end_time = datetime.now()
    
    print(f"Analysis complete. Execution time: {end_time - start_time}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run TaGra example with configuration file.')
    parser.add_argument('-c', '--config', type=str, required=False, default=None, help='Path to the configuration file.')
    args = parser.parse_args()

    if args.config is not None and not os.path.isfile(args.config):
        print(f"Error: The configuration file {args.config} does not exist.")
        sys.exit(1)

    main(args.config)
