import unittest
import json
import os
from tagra.config import load_config, save_config

class TestConfig(unittest.TestCase):

    def setUp(self):
        self.config_path = "test_config.json"
        self.config = {
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

    def tearDown(self):
        if os.path.exists(self.config_path):
            os.remove(self.config_path)

    def test_load_default_config(self):
        config = load_config(config_path="non_existent_config.json")
        self.assertEqual(config["nan_action"], "infer")

    def test_save_config(self):
        save_config(self.config, config_path=self.config_path)
        with open(self.config_path, 'r') as f:
            loaded_config = json.load(f)
        self.assertEqual(loaded_config["nan_action"], "infer")

    def test_load_saved_config(self):
        save_config(self.config, config_path=self.config_path)
        config = load_config(config_path=self.config_path)
        self.assertEqual(config["nan_action"], "infer")

if __name__ == '__main__':
    unittest.main()
