# 関数
import time
import platform
import pickle
import re

import numpy as np
import MeCab
import mojimoji

from selenium import webdriver
from selenium.webdriver.chrome.options import Options

import config
import module


# Macなら
if platform.system() == "Darwin":
    m = MeCab.Tagger(
        "-Owakati -d /opt/homebrew/lib/mecab/dic/mecab-ipadic-neologd")
# Ubuntuならmecab使わない
else:
    m = None


# scraping
def rand_sleep(length=1000, is_random=True):
    if is_random:
        # 10秒〜30秒の間でランダムにスリープ
        time.sleep(np.random.randint(10, 22))
    else:
        time.sleep(length)


def load_article_bin():
    try:
        with open(config.article_bin_path, 'rb') as f:
            return pickle.load(f)
    except:
        return []


def save_article_bin(data):
    with open(config.article_bin_path, 'wb') as f:
        pickle.dump(data, f)


def get_cli_webdriver(is_selenium):
    if is_selenium:
        return module.WebDriver()
    else:
        return module.RequestDriver()

# preprocess utils


def exclude_main_category(x):
    try:
        return x.split("<SEP>")[1]
    except:
        return "None"


def preprocess_text(x, is_nn_tokenize=False):
    # 空白除去
    t = re.sub("\s", "", x)
    # 全角の空白除去
    t = re.sub("\u3000", "", t)
    # カナ以外を半角に
    t = mojimoji.zen_to_han(t, kana=False)
    # カナだけを全角に
    t = mojimoji.han_to_zen(t, digit=False, ascii=False)
    # (〇〇)を丸ごと除去
    t = re.sub("\(.*?\)", "", t)

    if is_nn_tokenize:
        return t
    else:
        return m.parse(t)


def count_is_alpha(t):
    try:
        x = re.sub(r"[a-zA-Z]", "<E>", t)
        return x.count("<E>") / len(t)
    except:
        return -.01


def count_is_num(t):
    try:
        x = re.sub(r"\d", "<N>", t)
        return x.count("<N>") / len(t)
    except:
        return -.01
