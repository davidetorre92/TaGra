import json
import os

default_config = {
    "nan_action": "infer",
    "nan_threshold": 0.6,
    "numeric_scaling": "standard",
    "categorical_encoding": "one-hot",
    "manifold_method": None,
    "manifold_dim": 2,
    "verbose": True,
    "method": "knn",
    "threshold": 0.75,
    "k": 5,
    "clustering_method": "hierarchical",
    "inconsistency_threshold": 0.1
}

def load_config(config_path="config.json"):
    
    if type(config_path) is str:
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                config = json.load(f)
            if config["verbose"]:
                print(f"Loaded configuration from {config_path}.")
            return config
    else:
        config = default_config
        if config["verbose"]:
            print("Using default configuration.")
    return config

def save_config(config, config_path="config.json"):
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=4)
    if config["verbose"]:
        print(f"Configuration saved to {config_path}.")
