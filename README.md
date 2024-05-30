# TaGra (Table to Graph)

TaGra is a comprehensive Python library designed to simplify the preprocessing of data, the construction of graphs from data tables, and the analysis of those graphs. It provides automated tools for handling missing data, scaling, encoding, manifold learning techniques and graph construction.

## Scope of TaGra

TaGra achieves three primary objectives:

1. **Automatic Data Preprocessing**: TaGra automates the preprocessing of tabular data, handling missing values, scaling numeric features, encoding categorical variables and manifold learning techniques based on user-defined configurations.

2. **Graph Creation**: TaGra offers three distinct methods to create graphs from the data:
   - **K-Nearest Neighbors (KNN)**: Constructs a graph by connecting each node to its k-nearest neighbors based on Euclidean distance.
   - **Distance Threshold (Radius Graph)**: Connects nodes if the Euclidean distance between them is less than a specified threshold.
   - **Similarity Graph**: Adds an edge between nodes if their cosine similarity exceeds a given threshold.

When creating the graph, each row together with all its features is mapped to a node and an arc between two rows is created using the methods described above.

3. **Basic Graph Analysis**: TaGra provides functions to analyze the generated graphs, including degree distribution, betweenness centrality distribution, and community composition analysis. More will come soon!

## Configuration File

The configuration file is a JSON file that contains all the settings required for preprocessing and graph creation. Below are the key settings:

- `input_dataframe`: Path to the input DataFrame. Supported extensions are csv, xlsx, pickle, json, parquet, hdf, h5.
- `output_directory`: Path to the folder where the results will be collected. If not specified, the current directory is used.
- `preprocessed_filename`: Filename of the preprocessed DataFrame. If not specified, a default name pattern is used.
- `graph_filename`: Filename of the graph file. If not specified, a default name pattern is used.
- `numeric_columns`: List of numeric columns.
- `categorical_columns`: List of categorical columns.
- `target_columns`: List of target columns used for graph coloring and neighborhood statistics.
- `ignore_columns`: List of columns to ignore during preprocessing.
- `unknown_column_action`: Action for unspecified columns. Options: 'infer' or 'ignore'.
- `numeric_threshold`: Threshold for inferring numeric columns.
- `numeric_scaling`: Scaling mode for numeric columns. Options: 'standard' or 'minmax'.
- `categorical_encoding`: Encoding for categorical columns. Options: 'one-hot' or 'label'.
- `nan_action`: Action for NaN values. Options: 'drop row', 'drop column', or 'infer'.
- `nan_threshold`: Threshold for dropping columns based on NaN ratio.
- `verbose`: Flag for detailed output.
- `manifold_method`: Method for manifold learning. Options: 'Isomap' or null.
- `manifold_dimension`: Number of dimensions for manifold learning output.
- `method`: Method to infer the graph. Options: 'knn', 'distance_threshold', or 'similarity'.
- `k`: Number of neighbors for 'knn' method.
- `distance_threshold`: Distance threshold for 'distance_threshold' method.
- `similarity_threshold`: Similarity threshold for 'similarity' method.
- `clustering_method`: Method for clustering analysis. (TODO)
- `inconsistency_threshold`: Threshold for inconsistency in clustering. (TODO)
- `neigh_prob_path`: Filename for neighborhood statistics.
- `prob_heatmap_filename`: Filename for heatmap of neighborhood statistics.
- `degree_distribution_filename`: Filename for degree distribution plot.
- `betweenness_distribution_filename`: Filename for betweenness centrality distribution plot.
- `community_composition_filename`: Filename for community composition histogram.
- `graph_visualization_filename`: Filename for graph visualization. If null, graph is not plotted.

## Functions

### Preprocessing

TaGra provides automatic data preprocessing that includes:

- Handling missing values based on user-defined settings.
- Scaling numeric features using standard or min-max scaling.
- Encoding categorical variables using one-hot or label encoding.
- Inferring the type of unspecified columns based on a threshold.

### Graph Creation

TaGra supports three methods for creating graphs from preprocessed data:

1. **K-Nearest Neighbors (KNN)**:
   - Connects each node to its k-nearest neighbors.
   - Requires the parameter `k` to specify the number of neighbors.

2. **Distance Threshold (Radius Graph)**:
   - Connects nodes if their Euclidean distance is below a specified threshold.
   - Requires the parameter `distance_threshold`.

3. **Similarity Graph**:
   - Adds an edge between nodes if their cosine similarity is above a specified threshold.
   - Requires the parameter `similarity_threshold`.

### Graph Analysis

TaGra includes basic graph analysis functions:

- **Degree Distribution**: Plots the degree distribution of the graph.
- **Betweenness Centrality Distribution**: Plots the betweenness centrality distribution.
- **Community Composition**: Analyzes and plots the composition of communities within the graph.

## Installation

To install TaGra, simply use pip:

```sh
pip install tagra
```
## Quickstart
```sh
python3 examples/example_usage.py -c examples/example_config.json
```
You can edit the option in ```examples/example_config.json``` and adapt them as you wish.
The default option will produce a prepreocessing and a graph based on the ```moons``` dataset (SciKit Learn).

# Usage
## Configuration File

The configuration file is a JSON file that contains all the settings required for preprocessing and graph creation. Below are the key settings:

- `input_dataframe`: DataFrame path. Supported extensions are: csv, xlsx, pickle, json, parquet, hdf, h5. In the case of a .csv file, the presence of the header will be deduced in the preprocessing part.
- `output_directory`: Path to the folder where the results will be collected. If not specified, the path from where the executable was launched will be used. If the folder does not exist, it will be created.
- `preprocessed_filename`: Filename of the preprocessed dataframe. If not specified, a name with this pattern is created: `{basename}_{timestamp}.{ext}` where `{basename}` is the name of the `input_dataframe`, `{timestamp}` is a string in the format ‘%Y%m%d%H%M’ and `{ext}` is the file extension. The supported extensions are the same as for `input_dataframe`.
- `graph_filename`: Filename of the graph file. If not specified, a name with the same pattern as before is created. Supported extension: .graphml.
- `numeric_columns`: A list containing the numeric columns.
- `categorical_columns`: A list containing the categorical columns.
- `target_columns`: A list containing the "target" variable, used only to color the graph and to evaluate the statistics on the neighborhood in the resulting graph.
- `ignore_columns`: A list containing the columns to be ignored in the preprocessing.
- `unknown_column_action`: An action to deal with columns that have not been specified. Available options: 'infer' (infer how to deal with those columns) or 'ignore' (ignore the columns).
- `numeric_threshold`: A threshold to establish if a column is categorical or numeric when `unknown_column_action` is set to 'infer'. If the ratio of unique instances in a column to the total number of rows exceeds this threshold, then the column will be added to the `numeric_columns`; otherwise, it will be added to the `categorical_columns`.
- `numeric_scaling`: Scaling mode for `numeric_columns`. Available options: 'standard' (Standard Scaler) or 'minmax' (MinMax Scaler).
- `categorical_encoding`: Encoding for `categorical_columns`. Available options: 'one-hot' (One-Hot-Encoding) or 'label' (Label Encoding).
- `nan_action`: An action to deal with NaN values. Options: 'drop row', 'drop column' or 'infer' (fills with the average).
- `nan_threshold`: If `nan_action` is 'drop column', the column will be dropped if the ratio of NaNs in the column to the total number of rows is greater than this value.
- `verbose`: A flag to print detailed output.
- `manifold_method`: Method to apply manifold learning on `numeric_columns`. Options are 'Isomap' or null (avoid manifold learning).
- `manifold_dimension`: If `manifold_method` is not null, the number of dimensions of the output space of the manifold learning method.
- `method`: Method to infer the graph. Available options: 'knn' (make a graph with the k-nearest neighbors based on Euclidean distance), 'distance_threshold' (put an edge between nodes if their Euclidean distance is less than `distance_threshold`), 'similarity' (add an edge between two nodes if their cosine similarity is more than `similarity_threshold`).
- `k`: Number of neighbors if method is 'knn'.
- `distance_threshold`: Distance threshold; if the Euclidean distance between two rows is less than the threshold, add an edge between the rows.
- `similarity_threshold`: Similarity threshold; if the cosine similarity between two rows is greater than the threshold, add an edge between the rows.
- `clustering_method`: TODO!
- `inconsistency_threshold`: TODO!
- `neigh_prob_path`: Filename containing the statistics on the neighbors.
- `prob_heatmap_filename`: Filename of the heatmap containing the statistics on the neighbors.
- `degree_distribution_filename`: Filename with the log-log degree distribution plot.
- `betweenness_distribution_filename`: Filename with the log-log betweenness plot.
- `community_composition_filename`: Filename with the community distribution histogram.
- `graph_visualization_filename`: Path to the file where the graph visualization will be saved. If null, the graph will not be plotted.

## Data Preprocessing

```python
from tagra.preprocessing import preprocess_dataframe

# Example usage

df_preprocessed = preprocess_dataframe(
    input_dataframe='moons.csv',
    output_directory='./results/',
    preprocessed_filename=None,
    numeric_cols=[],
    categorical_cols=[],
    target_cols='class',
    unknown_col_action='infer',
    ignore_cols=[],
    numeric_threshold=0.05,
    numeric_scaling='standard',
    categorical_encoding='one-hot',
    nan_action='infer',
    nan_threshold=0.5,
    verbose=True,
    manifold_method='Isomap',
    manifold_dim=2
)

```

It will produce a preprocessed dataframe of moons.csv in the results/ directory with name moons_{timestap}.csv, where time stamp is a string in the format ‘%Y%m%d%H%M’ with the current time.

### Arguments

- `input_dataframe` (str or pd.DataFrame): Path to the input DataFrame or the DataFrame itself. Supported extensions for file paths are csv, xlsx, pickle, json, parquet, hdf, h5.

- `output_directory` (str): Path to the folder where the results will be saved. If not specified, the current directory is used.

- `preprocessed_filename` (str): Filename of the preprocessed DataFrame. If not specified, a name with the pattern `{basename}_{timestamp}.{ext}` is created, where `{basename}` is the name of the input DataFrame, `{timestamp}` is a string in the format ‘%Y%m%d%H%M’, and `{ext}` is the file extension.

- `numeric_cols` (list): List of columns to be treated as numeric. If not specified, numeric columns are inferred based on the `numeric_threshold`.

- `categorical_cols` (list): List of columns to be treated as categorical. If not specified, categorical columns are inferred based on the `numeric_threshold`.

- `target_cols` (list or str): List of target columns used for coloring the graph and evaluating neighborhood statistics. Can be a single column name as a string.

- `unknown_col_action` (str): Action to deal with columns not specified in `numeric_cols`, `categorical_cols`, or `ignore_cols`. Available options are 'infer' (to infer column type) and 'ignore' (to ignore the columns).

- `ignore_cols` (list): List of columns to be ignored during preprocessing.

- `numeric_threshold` (float): Threshold for inferring numeric columns. If the ratio of unique instances in a column to the total number of rows exceeds this threshold, the column is treated as numeric.

- `numeric_scaling` (str): Scaling method for numeric columns. Available options are 'standard' (Standard Scaler) and 'minmax' (MinMax Scaler).

- `categorical_encoding` (str): Encoding method for categorical columns. Available options are 'one-hot' (One-Hot Encoding) and 'label' (Label Encoding).

- `nan_action` (str): Action to deal with NaN values. Available options are 'drop row', 'drop column', and 'infer' (fill with the mean for numeric columns and mode for categorical columns).

- `nan_threshold` (float): Threshold for dropping columns based on the ratio of NaN values. If `nan_action` is 'drop column', columns with a NaN ratio greater than this value are dropped.

- `verbose` (bool): Flag to print detailed output during processing. Useful for debugging and tracking the preprocessing steps.

- `manifold_method` (str): Method for manifold learning on numeric columns. Options are 'Isomap' and None (to avoid manifold learning).

- `manifold_dim` (int): Number of dimensions for the output space of the manifold learning method. Applicable if `manifold_method` is not None.

## Graph construction
from tagra.graph import construct_graph

### Example usage
```python
graph = construct_graph(
    input_dataframe='moons.csv',
    output_directory='./results/,
    graph_filename=None,
    preprocessed_dataframe=df_preprocessed,
    method='knn',
    k=5,
    verbose=True
)
```
This example uses the `construct_graph` function to generate a graph with the distances from the preprocessed DataFrame `df_preprocessed` derived from 'moons.csv'. `moons.csv` will be used to add the features to the nodes. The name of the graph will be `graph_{timestamp}.graph` and will be saved in the directory './results'

### Arguments

- **input_dataframe** (`str` or `pd.DataFrame`, default=None):
  Path to the input DataFrame or the DataFrame itself. Supported file formats include csv, xlsx, pickle, json, parquet, hdf, h5.

- **preprocessed_dataframe** (`pd.DataFrame`, default=None):
  The DataFrame that has been preprocessed. If this is provided, the `input_dataframe` will be ignored.

- **output_directory** (`str`, default=None):
  Path to the directory where the resulting graph will be saved. If not specified, the current working directory is used.

- **graph_filename** (`str`, default=None):
  The name of the output graph file. If not specified, a default name with the pattern `{basename}_{timestamp}.graphml` is used.

- **method** (`str`, default='knn'):
  The method to construct the graph. Available options are:
  - 'knn': Creates a graph based on k-nearest neighbors.
  - 'distance_threshold': Connects nodes if the Euclidean distance between them is less than `distance_threshold`.
  - 'similarity': Connects nodes if the cosine similarity between them is greater than `similarity_threshold`.

- **distance_threshold** (`float`, default=0.75):
  The threshold for the Euclidean distance method. Nodes within this distance will be connected.

- **similarity_threshold** (`float`, default=0.95):
  The threshold for the similarity method. Nodes with a cosine similarity greater than this value will be connected.

- **k** (`int`, default=5):
  The number of nearest neighbors to connect in the 'knn' method.

- **verbose** (`bool`, default=True):
  A flag to print detailed output during the graph construction process. Useful for debugging and tracking progress.


## Graph Analysis

Simple graph analysis.

### Example usage
```python
from tagra.analysis import analyze_graph

# Example usage
results = analyze_graph(
    graph, 
    target_attributes='class', 
    verbose=True,
    output_directory=None,
    neigh_prob_filename = None,
    degree_distribution_filename = None,
    betweenness_distribution_filename = None,
    prob_heatmap_filename = None,
    community_composition_filename = None,
    graph_visualization_filename = None
)

```
`graph` will be analized and the a basic analysis on the graph will be performed.
During the runtime it is evaluated the probability of finding a node of target_attribute $i$ in the neighborhood of a node of target_attribute $j$ in the format:
P(i|j) = ...
 

### Arguments

- **graph** (`networkx.Graph`): The input graph to be analyzed.

- **target_attributes** (`str` or `list`, default=None): Attribute(s) to be analyzed. Used for coloring the graph and analyzing neighborhood statistics.

- **clustering_method** (`str`, default='hierarchical'): Method for clustering analysis. Currently supports 'hierarchical'.

- **inconsistency_threshold** (`float`, default=0.1): Threshold for the inconsistency in hierarchical clustering.

- **verbose** (`bool`, default=True): Flag to print detailed output during the analysis process.

- **output_directory** (`str`, default=None): Path to the directory where the results will be saved. If not specified, the current working directory is used.

- **neigh_prob_filename** (`str`, default=None): Filename for saving the neighborhood probability statistics. If not specified, a default name is generated.

- **degree_distribution_filename** (`str`, default=None): Filename for saving the degree distribution plot. If not specified, a default name is generated.

- **betweenness_distribution_filename** (`str`, default=None): Filename for saving the betweenness centrality distribution plot. If not specified, a default name is generated.

- **prob_heatmap_filename** (`str`, default=None): Filename for saving the heatmap of neighborhood probabilities. If not specified, a default name is generated.

- **community_composition_filename** (`str`, default=None): Filename for saving the community composition histogram. If not specified, a default name is generated.

- **graph_visualization_filename** (`str`, default=None): Filename for saving the graph visualization. If not specified, the graph will not be plotted.


# Contributing
We welcome contributions from the community. If you would like to contribute, please read our Contributing Guide for more information on how to get started.

# License
This project is licensed under the MIT License. See the LICENSE file for more details.

# Support
If you have any questions or need help, please feel free to open an issue on our GitHub repository.

---
Thank you for using TaGra! We hope it makes your data preprocessing and graph analysis tasks easier and more efficient.

