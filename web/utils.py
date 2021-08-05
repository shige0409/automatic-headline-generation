import pickle
from models import config

from transformers import BertJapaneseTokenizer
from models.transformer import Transformer

def load_tokenizer():
    with open("./data/keras_tokenizer.bin", "rb") as f:
        kt = pickle.load(f)
    bt = BertJapaneseTokenizer.from_pretrained("cl-tohoku/bert-base-japanese-whole-word-masking")
    
    return bt, kt

