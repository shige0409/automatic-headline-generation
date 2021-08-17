# 関数
from os import utime
import time
import pickle
import re

import numpy as np
import MeCab
import mojimoji

import requests
from requests.models import RequestEncodingMixin
from selenium import webdriver
from selenium.webdriver.chrome.options import Options

import config

# gloval variable
# mecabインスタンス
m = MeCab.Tagger("-Owakati -d /opt/homebrew/lib/mecab/dic/mecab-ipadic-neologd")

class RequestDriver():
    def __init__(self) -> None:
        self.page_source = None
    def get(self, url):
        response = requests.get(
            url,
            headers=config.headers
            )
        self.page_source = response.content



# scraping
def rand_sleep():
    # 10秒〜30秒の間でランダムにスリープ
    time.sleep(np.random.randint(10, 22))


def load_article_bin():
    try:
        with open(config.article_bin_path, 'rb') as f:
            return pickle.load(f)
    except:
        return []

def save_article_bin(data):
    with open(config.article_bin_path, 'wb') as f:
        pickle.dump(data , f)

def get_cli_webdriver(is_selenium):
    if is_selenium:
        op = Options()
        op.add_argument("enable-automation")
        op.add_argument("--no-sandbox")
        op.add_argument("--disable-infobars")
        op.add_argument('--disable-extensions')
        op.add_argument("--disable-dev-shm-usage")
        op.add_argument("--disable-browser-side-navigation")
        op.add_argument('--ignore-certificate-errors')
        op.add_argument('--ignore-ssl-errors')
        op.add_argument("--disable-gpu")
        op.add_argument("--disable-extensions")
        op.add_argument("--proxy-server='direct://'")
        op.add_argument("--proxy-bypass-list=*")
        op.add_argument("--start-maximized")
        op.add_argument("--headless")
        prefs = {"profile.default_content_setting_values.notifications" : 2}
        op.add_experimental_option("prefs",prefs)
        return webdriver.Chrome(options=op)
    else:
        return RequestDriver()
    

# preprocess utils 
def exclude_main_category(x):
    try:
        return x.split("<SEP>")[1]
    except:
        return "None"

def preprocess_text(x, is_nn_tokenize = False):
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
