# TaGra Documentation

TaGra is a Python package designed for preprocessing DataFrames, creating graphs, and analyzing data. This documentation provides an overview of the package, installation instructions, and usage examples.

## Installation

To install TaGra, you can use pip:

```bash
pip install tagra
```

# Usage
## Preprocessing DataFrames
The preprocessing module allows you to preprocess DataFrames with various options.

### Example
```python
from tagra.preprocessing import preprocess_dataframe

df = preprocess_dataframe(
    input_path='data.csv',
    output_path='preprocessed_data.csv',
    numeric_cols=['A', 'C'],
    categorical_cols=['B'],
    unknown_col_action='infer',
    ignore_cols=['D'],
    threshold=0.05,
    numeric_scaling='standard',
    categorical_encoding='one-hot',
    nan_action='infer',
    nan_threshold=0.6,
    verbose=True,
    manifold_method=None,
    manifold_dim=2
)
```

## Creating Graphs
The graph module allows you to create graphs from DataFrames using various methods such as K-Nearest Neighbors (KNN), distance thresholds, and similarity.

### Example
```python
from tagra.graph import create_graph

G = create_graph(
    dataframe_path='data.csv',
    preprocessed_dataframe_path='preprocessed_data.csv',
    output_path='graph.pickle',
    method='knn',
    threshold=0.75,
    k=5,
    verbose=True
)
```

## Analyzing Graphs
The analysis module provides functionality to analyze graphs, including degree distribution, centrality measures, and community detection.

### Example
from tagra.analysis import analyze_graph

degree_sequence, neighbors_count = analyze_graph(
    graph_path='graph.pickle',
    attribute='attribute_column',
    clustering_method='hierarchical',
    inconsistency_threshold=0.1,
    verbose=True
)

## Configuration
You can configure TaGra using a JSON configuration file. The config module provides functions to load and save configurations.

### Example
from tagra.config import load_config, save_config

config = load_config('config.json')
save_config(config, 'config.json')

# Modules
## preprocessing
This module handles the preprocessing of DataFrames. It includes options for handling missing values, scaling numeric data, encoding categorical data, and applying manifold learning techniques.

### Functions:

preprocess_dataframe: Preprocess a DataFrame with specified options.

## graph
This module creates graphs from DataFrames. It supports various methods for defining edges between nodes, including KNN, distance thresholds, and similarity.

### Functions:

- create_graph: Create a graph from a DataFrame with specified options.

## analysis
This module provides tools for analyzing graphs. It includes functions to calculate degree distribution, centrality measures, and to perform community detection.

### Functions:

- analyze_graph: Analyze a graph with specified options.

## config
This module manages configuration settings for the package. It can load and save configurations from/to JSON files.

### Functions:

- load_config: Load configuration settings from a JSON file.
- save_config: Save configuration settings to a JSON file.

# Contributing
If you wish to contribute to the project, feel free to fork the repository and create a pull request. Contributions are always welcome!

# License
This project is licensed under the MIT License. See the LICENSE file for more details.

# Contact
For any questions or issues, please contact dtorre[at]luiss[dot]it

