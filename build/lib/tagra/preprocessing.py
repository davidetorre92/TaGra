import datetime
import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder, LabelEncoder
from sklearn.manifold import Isomap

def preprocess_dataframe(input_dataframe=None, output_path=None, 
                         numeric_cols=[], categorical_cols=[], 
                         unknown_col_action='infer', ignore_cols=[], 
                         threshold=0.05, numeric_scaling='standard', 
                         categorical_encoding='one-hot', nan_action='infer', 
                         nan_threshold=1, verbose=True, 
                         manifold_method=None, manifold_dim=2):

    if verbose:
        print(f"--------------------------\nPreprocessing options\n--------------------------\n\n"
              f"\tOptions:\n"
              f"\tinput_path: {input_dataframe}, output_path: {output_path}, \n"
              f"\tnumeric_cols: {numeric_cols}, categorical_cols: {categorical_cols}, \n"
              f"\tunknown_col_action: {unknown_col_action}, ignore_cols: {ignore_cols}, \n"
              f"\tthreshold: {threshold}, numeric_scaling: {numeric_scaling}, \n"
              f"\tcategorical_encoding: {categorical_encoding}, nan_action: {nan_action}, \n"
              f"\tnan_threshold: {nan_threshold}, verbose: {verbose}, \n"
              f"\tmanifold_method: {manifold_method}, manifold_dim: {manifold_dim}\n\n")

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

    # Unknown columns inference
    for col in df.columns:
        if col not in numeric_cols and col not in categorical_cols and col not in ignore_cols:
            if df[col].dtype in [np.float64, np.float32, np.int64, np.int32]:
                numeric_cols.append(col)
                if verbose:
                    print(f"{datetime.datetime.now()}: Column '{col}' added to numeric columns by inference.")
            elif df[col].dtype == 'bool' or np.issubdtype(df[col].dtype, np.datetime64):
                ignore_cols.append(col)
                if verbose:
                    print(f"{datetime.datetime.now()}: Column '{col}' added to ignored columns by inference.")
            elif df[col].dtype == 'object':
                categorical_cols.append(col)
                if verbose:
                    print(f"{datetime.datetime.now()}: Column '{col}' added to categorical column columns by inference.")      
            else:
                unique_ratio = len(df[col].unique()) / len(df[col])
                if unique_ratio > threshold:
                    numeric_cols.append(col)
                    if verbose:
                        print(f"{datetime.datetime.now()}: Column '{col}' added to numeric columns by unique ratio inference.")
                else:
                    categorical_cols.append(col)
                    if verbose:
                        print(f"{datetime.datetime.now()}: Column '{col}' added to categorical columns by unique ratio inference.")

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
        for col in numeric_cols:
            df[col] = df[col].fillna(df[col].mean())
            if verbose:
                print(f"{datetime.datetime.now()}: Filled NaN values in numeric column '{col}' with mean.")
        for col in categorical_cols:
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
    df[numeric_cols] = scaler.fit_transform(df[numeric_cols])
    if verbose:
        print(f"{datetime.datetime.now()}: Scaled numeric columns using {numeric_scaling} scaling.")

    # Preprocessing cat cols
    if categorical_encoding == 'one-hot':
        df = pd.get_dummies(df, columns=categorical_cols)
    elif categorical_encoding == 'label':
        encoder = LabelEncoder()
        for col in categorical_cols:
            df[col] = encoder.fit_transform(df[col])
    if verbose:
        print(f"{datetime.datetime.now()}: Encoded categorical columns using {categorical_encoding} encoding.")

    # Manifold learning
    if manifold_method:
        if manifold_method == 'Isomap':
            manifold = Isomap(n_components=manifold_dim)
        else:
            raise ValueError(f"Unsupported manifold learning method: {manifold_method}")
        df[numeric_cols] = manifold.fit_transform(df[numeric_cols])
        if verbose:
            print(f"{datetime.datetime.now()}: Applied {manifold_method} manifold learning.")

    # Output path managing
    if output_path is None:
        if isinstance(input_dataframe, str):
            base, ext = os.path.splitext(input_dataframe)
            output_path = f"{base}_preprocessed_{datetime.datetime.now().strftime('%Y%m%d%H%M')}{ext}"
        else:
            output_path = f"./preprocessed_{datetime.datetime.now().strftime('%Y%m%d%H%M')}.pickle"

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

    return df