import pickle
import re
import mojimoji

# tokenizer
from transformers import BertJapaneseTokenizer
from tensorflow.keras import preprocessing as pp

bert_tokenizer = BertJapaneseTokenizer.from_pretrained("cl-tohoku/bert-base-japanese-whole-word-masking")
with open("./data/keras_tokenizer.bin", 'rb') as f:
    keras_tokenizer =  pickle.load(f)


def preprocess_text(text):
    t = re.sub(r"\s", "", text)
    t = re.sub("\u3000", "", t)
    # t = re.sub(r",", "、", x)
    # カナ以外を半角に
    t = mojimoji.zen_to_han(t, kana=False)
    # 数字以外を全角に
    t = mojimoji.han_to_zen(t, digit=False, ascii=False)
    t = " ".join(bert_tokenizer.tokenize(t))
    enc_t = keras_tokenizer.texts_to_sequences([t])
    enc_t = pp.sequence.pad_sequences(enc_t, maxlen=200, padding="post", truncating="post")
    return enc_t

print(preprocess_text("安田大サーカス・クロちゃん、新型コロナ感染団長安田とは「別ルートでの感染と考えられます」"))