import pickle
import re

import config

def load_dataset():
    with open(config.dataset_path, "rb") as f:
        return pickle.load(f)