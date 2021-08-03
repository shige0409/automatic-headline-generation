import numpy as np
import pandas as pd

import utils
import config

# pickle
article_infos = utils.load_article_bin()
# dataframe
df = pd.DataFrame(article_infos, columns=["url", "date", "title", "context", "category", "keyword", "tweet_share", "quote"])
# 記事が消去されていたデータを除去
df = df.query("title != 'None Title'").copy()
# tweet_shareを数値化
df = df.astype({"tweet_share": np.int64})
# 文書分類用のカテゴリ抽出
df["main_category"] = df.category.apply(utils.exclude_main_category)
# カテゴリが少ない(100件ない)データを除去
main_category_count_df = df.groupby("main_category").url.count()
use_main_category = main_category_count_df.index[main_category_count_df > 100].to_list()
df = df[df.main_category.isin(use_main_category)]
# カテゴリが付与されていないデータを除去
df = df.query("main_category != 'None'").copy()

# 記事本文の前処理
df["context_"] = df.context.apply(lambda x: utils.preprocess_context(x, is_nn_tokenize=True))
# 記事見出しの前処理
df["title_"] = df.title.apply(lambda x: utils.preprocess_context(x, is_nn_tokenize=True))

# 書き出し => 学習に使うデータ
df.to_csv(config.preprocessd_csv_path, index=None)
print(config.preprocessd_csv_path, "に書き出し完了")
