# TaGra (Table to Graph)

TaGra is a comprehensive Python library designed to simplify the preprocessing of data, the construction of graphs from data tables, and the analysis of those graphs. It provides automated tools for handling missing data, scaling, encoding, and graph construction, making it easier for data scientists and researchers to focus on their analysis and results.

## Features

- **Automated Data Preprocessing**: Clean and transform your data with options for handling missing values, scaling, encoding, and more.
- **Graph Construction**: Build graphs from data tables using various methods such as K-Nearest Neighbors (KNN), distance thresholds, and similarity measures.
- **Graph Analysis**: Perform statistical analysis on graphs, including degree distribution, centrality measures, community detection, and neighbor analysis.
- **Visualization**: Generate insightful visualizations to understand your data and graph structures.

## Installation

To install TaGra, simply use pip:

```sh
pip install tagra
```
## Quickstart
```sh
python3 examples/examples_usage.py -c examples/example_config.json
```
You can edit the option in ```examples/example_config.json``` and adapt them as you wish.
The default option will produce a prepreocessing and a graph based on the ```moons``` dataset (SciKit Learn).

# Usage
## Data Preprocessing

```python
from tagra.preprocessing import preprocess_dataframe

# Example usage
df = preprocess_dataframe(
    input_path='data.csv',
    output_path=None,
    numeric_columns=[],
    categorical_columns=[],
    unknown_column_action='infer',
    ignore_columns=[],
    numeric_threshold=0.05,
    numeric_scaling='standard',
    categorical_encoding='one-hot',
    nan_action='infer',
    nan_threshold=0.6,
    verbose=True,
    manifold_learning=None,
    manifold_dimension=2
)
```

## Graph construction
from tagra.graph import construct_graph

### Example usage
graph = construct_graph(
    data_path='data.csv',
    preprocessed_data_path=None,
    method='knn',
    k=5,
    output_path=None,
    verbose=True
)

## Graph Analysis
```python
from tagra.analysis import analyze_graph

# Example usage
results = analyze_graph(
    graph_path='graph.pickle',
    degree_distribution=True,
    centrality_measures=['closeness', 'giant_connected_component'],
    community_detection='hierarchical',
    neighbor_analysis=True,
    attribute=None
)
```

# Detailed Documentation
For a comprehensive guide on how to use each feature of TaGra, including detailed parameter descriptions and additional examples, please refer to the full documentation.

# Contributing
We welcome contributions from the community. If you would like to contribute, please read our Contributing Guide for more information on how to get started.

# License
This project is licensed under the MIT License. See the LICENSE file for more details.

# Support
If you have any questions or need help, please feel free to open an issue on our GitHub repository.

---
Thank you for using TaGra! We hope it makes your data preprocessing and graph analysis tasks easier and more efficient.

