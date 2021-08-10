import pickle
import re
import mojimoji

import numpy as np

from models.config import *
from models.transformer import Transformer, create_masks

from tensorflow import keras
from tensorflow.keras import preprocessing as pp

from transformers import BertJapaneseTokenizer

def load_tokenizer():
    with open("./data/keras_tokenizer.bin", "rb") as f:
        kt = pickle.load(f)
    bt = BertJapaneseTokenizer.from_pretrained("cl-tohoku/bert-base-japanese-whole-word-masking")
    
    return bt, kt

def load_transformer():
    tf_transformer = Transformer(
        num_layers=NUM_LAYERS, d_model=D_MODEL, num_heads=NUM_HEADS,
        dff=DDF,input_vocab_size=VOCAB_SIZE, target_vocab_size=VOCAB_SIZE,
		pe_input=ENC_SEQ_LEN, pe_target=DEC_SEQ_LEN
    )
    encoder_inputs = keras.Input(shape=(ENC_SEQ_LEN,), dtype="int32", name="encoder_inputs")
    decoder_inputs = keras.Input(shape=(DEC_SEQ_LEN,), dtype="int32", name="decoder_inputs")
    enc_padding_mask, combined_mask, dec_padding_mask = keras.layers.Lambda(create_masks)([encoder_inputs, decoder_inputs])
    decoder_outputs = tf_transformer(encoder_inputs, decoder_inputs, True, enc_padding_mask, combined_mask, dec_padding_mask)
    train_model = keras.Model([encoder_inputs, decoder_inputs], decoder_outputs)
    train_model.load_weights("data/train_model_weight_2.4.0.h5")
    return train_model


def preprocess_context(x):
    t = re.sub(r"\s", "", x)
    t = re.sub("\u3000", "", t)
    # t = re.sub(r",", "、", x)
    # カナ以外を半角に
    t = mojimoji.zen_to_han(t, kana=False)
    # 数字以外を全角に
    t = mojimoji.han_to_zen(t, digit=False, ascii=False)
    return t

def generate_title(text, models, beam_width = 3):
    bt, kt, model = models
    decoder_sentence = "[BOS]"
    input_sentence = " ".join(bt.tokenize(text))
    encoder_sentence_ = kt.texts_to_sequences([input_sentence])
    encoder_sentence_ = pp.sequence.pad_sequences(encoder_sentence_, maxlen=ENC_SEQ_LEN, padding="post", truncating="post")
    for i in range(50):
        decoder_sentence_ = kt.texts_to_sequences([decoder_sentence])
        decoder_sentence_ = pp.sequence.pad_sequences(decoder_sentence_, maxlen=DEC_SEQ_LEN, padding="post", truncating="post")
        pred = model((encoder_sentence_, decoder_sentence_))
        sampled_tkn_idx = np.argmax(pred[0,i,:])
        sampled_tkn = kt.index_word[sampled_tkn_idx]
        decoder_sentence =  decoder_sentence + " " + sampled_tkn
        if sampled_tkn == "[EOS]":
            break
    return "".join(re.sub("#", "", decoder_sentence).split(" ")[1:-1])
