import pandas as pd
import pickle
import config

df = pd.read_csv(config.preprocessd_csv_path)

with open(config.article_bin_path, "wb") as f:
    pickle.dump(df[["context_", "title_", "main_category", "tweet_share"]].values , f)
