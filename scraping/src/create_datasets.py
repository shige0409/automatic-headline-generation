import pandas as pd
import pickle
import config

df = pd.read_csv(config.preprocessd_csv_path)

with open("../data/datasets.bin", "wb") as f:
    pickle.dump(df[["context_", "title_"]].values , f)
