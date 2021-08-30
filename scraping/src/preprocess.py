import pickle

import numpy as np
import pandas as pd

import utils
import config

# pickle
article_infos = utils.load_article_bin()
# dataframe
df = pd.DataFrame(article_infos, columns=["url", "date", "title", "content", "category", "keyword", "tweet_share", "quote"])
# 重複削除
df.drop_duplicates(inplace=True)
# 記事が消去されていたデータを除去
df = df.query("content != 'None Title' and title != 'None Title'").copy()
# 記事の文字数が明らかに短いのを除去
df = df.query("content.str.len() > 10", engine="python").copy()
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
df["content_"] = df.content.apply(lambda x: utils.preprocess_text(x, is_nn_tokenize=True))
# 記事見出しの前処理
df["title_"] = df.title.apply(lambda x: utils.preprocess_text(x, is_nn_tokenize=True))

# 記事に数字や英語が含まれている文字の割合を抽出
df["content_alpha_ratio"] = df.content_.apply(utils.count_is_alpha)
df["content_num_ratio"] = df.content_.apply(utils.count_is_num)
# 合計で25%以内だけを使う
df = df.loc[(df.content_alpha_ratio + df.content_num_ratio) <= 0.25]

# 書き出し => 学習に使うデータ
df.to_csv(config.preprocessd_csv_path, index=None)
print(config.preprocessd_csv_path, "に書き出し完了")
# 学習用のデータセット作成

with open(config.dataset_path, "wb") as f:
    pickle.dump(df[["content_", "title_", "main_category", "tweet_share"]].values , f)
print(config.dataset_path, "に書き出し完了")
