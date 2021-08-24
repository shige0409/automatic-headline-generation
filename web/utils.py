import pickle
import random
import re
import mojimoji

import numpy as np
from tensorflow.python.framework import config

from models.config import *
from models.transformer import *

from tensorflow import keras
from tensorflow import nn
from tensorflow.keras import preprocessing as pp

from transformers import BertJapaneseTokenizer

def load_tokenizer():
    with open("./data/keras_tokenizer.bin", "rb") as f:
        kt = pickle.load(f)
    bt = BertJapaneseTokenizer.from_pretrained("cl-tohoku/bert-base-japanese-whole-word-masking")
    
    return bt, kt

def load_model():
    encoder_inputs = keras.Input(shape=(ENC_SEQ_LEN,), dtype="int32", name="encoder_inputs")
    decoder_inputs = keras.Input(shape=(DEC_SEQ_LEN,), dtype="int32", name="decoder_inputs")
    enc_padding_mask, combined_mask, dec_padding_mask = layers.Lambda(create_masks)([encoder_inputs, decoder_inputs])
    # model
    pos_emb = PositionalEncoding(d_model=D_MODEL, input_vocab_size=VOCAB_SIZE)
    tf_encoder = Encoder(
        num_layers=NUM_LAYERS, d_model=D_MODEL, num_heads=NUM_HEADS, dff=DDF,
        input_vocab_size=VOCAB_SIZE, category_size=CATEGORY_SIZE, rate=0.2)
    tf_decoder = Decoder(NUM_LAYERS, D_MODEL, NUM_HEADS, DDF, VOCAB_SIZE, rate=0.1)
    # feed_foward
    enc_output, enc_output_cls = tf_encoder(pos_emb(encoder_inputs), enc_padding_mask)
    dec_output = tf_decoder(pos_emb(decoder_inputs), enc_output, combined_mask, dec_padding_mask)
    # global model
    trainer = keras.Model([encoder_inputs, decoder_inputs], [dec_output, enc_output_cls], name="s2s")
    trainer.load_weights("./data/trainer_weight_multi.h5")
    return trainer

def load_sample():
    with open("../scraping/data/article_infos.bin", "rb") as f:
        data = pickle.load(f)
    return random.choice(data)[3]


def preprocess_context(x):
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
    return t

def generate_title(text, models, beam_width = 3):
    bt, kt, model = models
    pre_log_prob = np.array([[0.]]*beam_width)
    decoder_sentence = ["[BOS]"] * beam_width
    input_sentence = [" ".join(bt.tokenize(text))] * beam_width
    encoder_sentence_ = kt.texts_to_sequences(input_sentence)
    encoder_sentence_ = pp.sequence.pad_sequences(encoder_sentence_, maxlen=ENC_SEQ_LEN, padding="post", truncating="post")
    for idx in range(50):
        decoder_sentence_ = kt.texts_to_sequences(decoder_sentence)
        decoder_sentence_ = pp.sequence.pad_sequences(decoder_sentence_, maxlen=DEC_SEQ_LEN, padding="post", truncating="post")
        pred, pred_label = model((encoder_sentence_, decoder_sentence_))
        log_prob_pred = np.log(nn.softmax(pred, axis=2))
        log_prob_pred = log_prob_pred[:,idx,:]
        log_prob_pred = pre_log_prob + log_prob_pred
        # 最初の生成だけ
        if idx == 0:
            log_prob_pred = log_prob_pred[idx:idx+1]
        else:
            pass
        max_log_prob_idx_with_beam = np.argsort(-log_prob_pred.reshape(-1))[:beam_width]
        max_log_prob_idx_with_beam = [divmod(idx, VOCAB_SIZE) for idx in max_log_prob_idx_with_beam]
        log_prob = [log_prob_pred[beam_idx, vocab_idx] for beam_idx, vocab_idx in max_log_prob_idx_with_beam]
        pre_log_prob += np.array(log_prob)[:,np.newaxis]
        add_words = [kt.index_word[vocab_idx] for _, vocab_idx in max_log_prob_idx_with_beam]

        decoder_sentence = [t1 + " " + t2 for t1, t2 in zip(decoder_sentence, add_words)]
        if "[EOS]" in add_words:
            break
    return decoder_sentence, label_dict[np.argmax(pred_label[0])]


def clean_beam_title(titles):
    clean_titles = ["".join(re.sub("#", "", t).split(" ")[1:-1]) for t in titles]
    return clean_titles
