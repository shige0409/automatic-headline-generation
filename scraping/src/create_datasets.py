import pandas as pd
import pickle
import config

df = pd.read_csv(config.preprocessd_csv_path)

with open("../data/datasets.bin", "wb") as f:
    pickle.dump(df[["content_", "title_", "main_category", "tweet_share"]].values , f)
