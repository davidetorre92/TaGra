import datetime
import os
import numpy as np
import pandas as pd
import pickle
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder, LabelEncoder
import pdb
def preprocess_dataframe(input_dataframe=None, 
                         output_directory="results/", 
                         preprocessed_filename=None,
                         inferred_columns_filename=None, 
                         numeric_columns=[],
                         categorical_columns=[], 
                         target_columns=[], 
                         unknown_column_action='infer',
                         ignore_columns=[], 
                         numeric_threshold=0.05,
                         numeric_scaling='standard', 
                         categorical_encoding='one-hot',
                         nan_action='infer', 
                         nan_threshold=0.5,
                         verbose=True, 
                         manifold_method='UMAP', 
                         manifold_dim=2,
                         overwrite=False):

    if verbose:
        print(f"--------------------------\nPreprocessing options\n--------------------------\n\n"
        f"\tOptions:\n"
        f"\tinput_path: {input_dataframe}, output_directory: {output_directory}, preprocessed_filename: {preprocessed_filename}\n"
        f"\tnumeric_columns: {numeric_columns}, categorical_columns: {categorical_columns}, target_columns: {target_columns}, \n"
        f"\tunknown_column_action: {unknown_column_action}, ignore_columns: {ignore_columns}, \n"
        f"\tnumeric_threshold: {numeric_threshold}, numeric_scaling: {numeric_scaling}, \n"
        f"\tcategorical_encoding: {categorical_encoding}, nan_action: {nan_action}, \n"
        f"\tnan_threshold: {nan_threshold}, verbose: {verbose}, \n"
        f"\tmanifold_method: {manifold_method}, manifold_dim: {manifold_dim}\n")

    # Output path managing
    if output_directory is None:
        output_directory = './'
    if os.path.exists(output_directory) is False:
        os.mkdir(output_directory)
        print(f"{datetime.datetime.now()}: Output directory created: {output_directory}.")
    if preprocessed_filename is None:
        if isinstance(input_dataframe, str):
            basename = os.path.basename(input_dataframe)
            base, ext = os.path.splitext(basename)
            if overwrite:
                preprocessed_filename = f"{base}_preprocessed{ext}"
            else:    
                preprocessed_filename = f"{base}_preprocessed_{datetime.datetime.now().strftime('%Y%m%d%H%M')}{ext}"
        else:
            if overwrite:
                preprocessed_filename = f"preprocessed.pickle"
            else:
                preprocessed_filename = f"preprocessed_{datetime.datetime.now().strftime('%Y%m%d%H%M')}.pickle"
    
    output_path = os.path.join(output_directory, preprocessed_filename)
    if verbose: print(f"{datetime.datetime.now()}: Output path for the preprocessed file: {output_path}.")
    if inferred_columns_filename is not None:
        if inferred_columns_filename.endswith('.pickle') is False:
            raise ValueError("Invalid inferred_columns_filename. Must be a path to a pickle file.")
        else:
            inferred_columns_dictionary_path = os.path.join(output_directory, inferred_columns_filename)
            if verbose: print(f"{datetime.datetime.now()}: Inferred columns dictionary path: {inferred_columns_dictionary_path}.")

    # Load dataframe
    if isinstance(input_dataframe, str):
        if input_dataframe.endswith('.csv'):
            # read the first row of the CSV to determine if the first column is an index
            peek_df = pd.read_csv(input_dataframe, nrows=1)
            # check if the first column looks like an index (e.g., unnamed or follows a specific pattern)
            if peek_df.columns[0].startswith('Unnamed') or peek_df.columns[0].isdigit():
                df = pd.read_csv(input_dataframe, index_col=0)
            else:
                df = pd.read_csv(input_dataframe)
        elif input_dataframe.endswith('.xlsx'):
            df = pd.read_excel(input_dataframe, index_col=None)
        elif input_dataframe.endswith('.pickle'):
            df = pd.read_pickle(input_dataframe)
        elif input_dataframe.endswith('.json'):
            df = pd.read_json(input_dataframe)
        elif input_dataframe.endswith('.parquet'):
            df = pd.read_parquet(input_dataframe)
        elif input_dataframe.endswith('.hdf') or input_dataframe.endswith('.h5'):
            df = pd.read_hdf(input_dataframe)
        else:
            # Suggesting action to the user
            supported_formats = ", ".join(["CSV", "Excel (.xlsx)", "Pickle", "JSON", "Parquet", "HDF5 (.hdf, .h5)"])
            raise ValueError(f"The file format is not supported. Please convert your file to one of the following supported formats: {supported_formats}.")
    elif isinstance(input_dataframe, pd.DataFrame):
        df = input_dataframe.copy()
    else:
        raise ValueError("Invalid input_path. Must be a path to a file or a pandas DataFrame.")

    # Checking columns
    ## Checking target_columns
    if target_columns is not None:
        if type(target_columns) != list:
            if target_columns in df.columns is False:
                raise ValueError(f"Target column {target_columns} not found.") 
            target_columns = [target_columns] # We need them to be lists. 
        else:
            for target_col in target_columns:
                if target_col in df.columns is False:
                    raise ValueError(f"Target column {target_col} not found.") 
    else:
        target_columns = []

    ## Checking numeric_columns
    if numeric_columns is not None:
        if type(numeric_columns) != list:
            if numeric_columns in df.columns is False:
                raise ValueError(f"Numeric column {numeric_columns} not found.") 
            numeric_columns = [numeric_columns] # We need them to be lists. 
        else:
            for numeric_col in numeric_columns:
                if numeric_col in df.columns is False:
                    raise ValueError(f"Numeric column {numeric_col} not found.") 
    else:
        numeric_columns = []

    ## Checking categorical_columns
    if categorical_columns is not None:
        if type(categorical_columns) != list:
            if categorical_columns in df.columns is False:
                raise ValueError(f"Categorical column {categorical_columns} not found.") 
            categorical_columns = [categorical_columns] # We need them to be lists. 
        else:
            for categorical_col in categorical_columns:
                if categorical_col in df.columns is False:
                    raise ValueError(f"Categorical column {categorical_col} not found.") 
    else:
        categorical_columns = []

    ## Checking ignore_columns
    if ignore_columns is not None:
        if type(ignore_columns) != list:
            if ignore_columns in df.columns is False:
                raise ValueError(f"Ignore column {ignore_columns} not found.") 
            ignore_columns = [ignore_columns] # We need them to be lists. 
        else:
            for ignore_col in ignore_columns:
                if ignore_col in df.columns is False:
                    raise ValueError(f"Ignore column {ignore_col} not found.") 
    else:
        ignore_columns = []

    # Targets should not be preprocessed
    ignore_columns += target_columns
    # Unknown columns inference
    if unknown_column_action == 'infer':
        for col in df.columns:
            if col not in numeric_columns and col not in categorical_columns and col not in ignore_columns:
                if df[col].dtype in [np.float64, np.float32, np.int64, np.int32]:
                    numeric_columns.append(col)
                    if verbose:
                        print(f"{datetime.datetime.now()}: Column '{col}' added to numeric columns by inference.")
                elif df[col].dtype == 'bool' or np.issubdtype(df[col].dtype, np.datetime64):
                    ignore_columns.append(col)
                    if verbose:
                        print(f"{datetime.datetime.now()}: Column '{col}' added to ignored columns by inference.")
                elif df[col].dtype == 'object':
                    categorical_columns.append(col)
                    if verbose:
                        print(f"{datetime.datetime.now()}: Column '{col}' added to categorical column columns by inference.")      
                else:
                    unique_ratio = len(df[col].unique()) / len(df[col])
                    if unique_ratio > numeric_threshold:
                        numeric_columns.append(col)
                        if verbose:
                            print(f"{datetime.datetime.now()}: Column '{col}' added to numeric columns by unique ratio inference.")
                    else:
                        categorical_columns.append(col)
                        if verbose:
                            print(f"{datetime.datetime.now()}: Column '{col}' added to categorical columns by unique ratio inference.")
    elif unknown_column_action == 'ignore':
        ignore_columns += [col for col in df.columns if col not in numeric_columns and col not in categorical_columns and col not in ignore_columns]
    else: raise ValueError(f"unknown_column_action {unknown_column_action} not supported. Aborting...")
    if verbose:
        print(f"--------------------------\nDataframe short report\n--------------------------\n\n")
        print(f"{df.shape[0]} rows and {df.shape[1]} columns")
        print(f"column list: {list(df.columns)}")
        print(f"nans:\n{df.isna().sum()}")

    # Set target columns to be only one colum
    target_col_name = tuple(target_columns) if len(target_columns) > 1 else (target_columns[0] if len(target_columns) == 1 else '')
    if len(target_columns) > 1:
        df[target_col_name] = df[target_columns].apply(tuple, axis=1)
        df = df.drop(columns=target_columns)
    
    if len(target_columns) != 0:
        unique_targets = np.unique(df[target_col_name].values)
        N_col = df.shape[0]
        print(f"Target class proportions")
        for target in unique_targets:
            n_target = df[df[target_col_name] == target].shape[0]
            print(f"\t{target}: {n_target / N_col * 100}%")
    print(f"--------------------------\nEnd of the report.")

    # NaNs
    if nan_action == 'drop row':
        df.dropna(inplace=True)
        if verbose:
            print(f"{datetime.datetime.now()}: Dropped rows with NaN values.")
    elif nan_action == 'drop column':
        df.dropna(axis=1, thresh=int(nan_threshold * df.shape[0]), inplace=True)
        if verbose:
            print(f"{datetime.datetime.now()}: Dropped columns with NaN values above threshold.")
    elif nan_action == 'infer':
        for col in numeric_columns:
            df[col] = df[col].fillna(df[col].mean())
            if verbose:
                print(f"{datetime.datetime.now()}: Filled NaN values in numeric column '{col}' with mean.")
        for col in categorical_columns:
            df[col] = df[col].fillna(df[col].mode()[0])
            if verbose:
                print(f"{datetime.datetime.now()}: Filled NaN values in categorical column '{col}' with mode.")
        if verbose:
            print(f"{datetime.datetime.now()}: Filled NaN values with column means.")

    # Preprocessing numerical cols
    if numeric_scaling == 'standard':
        scaler = StandardScaler()
    elif numeric_scaling == 'minmax':
        scaler = MinMaxScaler()
    df[numeric_columns] = scaler.fit_transform(df[numeric_columns])
    if verbose:
        print(f"{datetime.datetime.now()}: Scaled numeric columns using {numeric_scaling} scaling.")

    # Preprocessing cat cols
    if categorical_encoding == 'one-hot':
        df = pd.get_dummies(df, columns=categorical_columns)
    elif categorical_encoding == 'label':
        encoder = LabelEncoder()
        for col in categorical_columns:
            df[col] = encoder.fit_transform(df[col])
    if verbose:
        print(f"{datetime.datetime.now()}: Encoded categorical columns using {categorical_encoding} encoding.")

    # Manifold learning
    manifold_positions = None
    if manifold_method:
        manifold_dim = 2
        if len(numeric_columns) < manifold_dim:
            if verbose:
                print(f"{datetime.datetime.now()}: manifold_dim is larger than number of numeric columns. Skipping...")
        else:
            # Initialize manifold method
            if manifold_method == 'Isomap':
                from sklearn.manifold import Isomap
                manifold = Isomap(n_components=manifold_dim)
            elif manifold_method == 'TSNE':
                from sklearn.manifold import TSNE
                manifold = TSNE(n_components=manifold_dim)
            elif manifold_method == 'UMAP':
                from umap.umap_ import UMAP
                manifold = UMAP(n_components=manifold_dim, 
                            random_state=42,  # For reproducibility
                            n_neighbors=15,    # Default=15, adjust based on data size
                            min_dist=0.1)      # Default=0.1, controls cluster tightness
            else:
                raise ValueError(f"Unsupported manifold method: {manifold_method}. "
                                f"Choose from ['Isomap', 'TSNE', 'UMAP']")

            # Common processing for all manifold methods
            manifold_numeric_columns = [f'manifold_{i}' for i in range(manifold_dim)]
            
            # Fit-transform and preserve original index
            manifold_transform = manifold.fit_transform(df[numeric_columns])
            df_manifold = df.copy()
            df_manifold = df_manifold.drop(columns=numeric_columns)
            df_manifold[manifold_numeric_columns] = manifold_transform
            manifold_positions = manifold_transform  # For visualization coordinates
            
            if verbose:
                print(f"{datetime.datetime.now()}: Applied {manifold_method} with settings: "
                    f"n_components={manifold_dim}, "
                    f"n_neighbors={manifold.n_neighbors if hasattr(manifold, 'n_neighbors') else 'N/A'}")

    # Save columns category
    inferred_columns_dictionary = {}
    if inferred_columns_filename is not None:
        inferred_columns_dictionary["numeric_columns"] = numeric_columns
        inferred_columns_dictionary["categorical_columns"] = categorical_columns
        inferred_columns_dictionary["ignore_columns"] = ignore_columns
        inferred_columns_dictionary["target_columns"] = target_columns
        with open(inferred_columns_dictionary_path, 'wb') as file:
            pickle.dump(inferred_columns_dictionary, file)
        if verbose:
            print(f"{datetime.datetime.now()}: Saved inferred columns dictionary to {inferred_columns_dictionary_path}.")

    # Save
    if output_path.endswith('.pickle'):
        df.to_pickle(output_path)
    elif output_path.endswith('.csv'):
        df.to_csv(output_path, index=False)
    elif output_path.endswith('.xlsx'):
        df.to_excel(output_path, index=False)
    elif output_path.endswith('.json'):
        df.to_json(output_path, index=False)
    elif output_path.endswith('.parquet'):
        df.to_parquet(output_path, index=False)
    elif output_path.endswith('.hdf') or output_path.endswith('.h5'):
        df.to_hdf(output_path, index=False)
    if verbose:
        print(f"{datetime.datetime.now()}: Saved preprocessed DataFrame to {output_path}.")

    return df, manifold_positions