import config
import utils

from sklearn.model_selection import train_test_split
from transformer import Transformer, create_masks
from transformers import BertJapaneseTokenizer
from tensorflow.keras import layers, preprocessing as pp
import tensorflow as tf
from tensorflow import keras

# tokenizer
bert_tokenizer = BertJapaneseTokenizer.from_pretrained("cl-tohoku/bert-base-japanese-whole-word-masking")
keras_tokenizer = pp.text.Tokenizer(filters="", lower=False)
# datasets
datasets = utils.load_dataset()
# tokenize
content = [" ".join(bert_tokenizer.tokenize(d[0])) for d in datasets]
title = ["[CLS] " + " ".join(bert_tokenizer.tokenize(d[1])) + " [SEP]" for d in datasets]
# tokenizer fit
keras_tokenizer.fit_on_texts(content + title)
# 数値化 ex. [1,2,4,6,10]
X = keras_tokenizer.texts_to_sequences(content)
y = keras_tokenizer.texts_to_sequences(title)
# 文の長さをPADで揃える ex. [1,2,4,6,10,0,0,0,0]
X = pp.sequence.pad_sequences(X, maxlen=config.ENC_SEQ_LEN, padding="post", truncating="post")
y = pp.sequence.pad_sequences(y, maxlen=config.DEC_SEQ_LEN+1, padding="post", truncating="post")
# train test split
X_train, X_val ,y_train, y_val = train_test_split(X, y, test_size=0.1, random_state=41)
# 
enc_inp_trn, enc_inp_val = X_train, X_val
dec_inp_trn, dec_inp_val = y_train[:,:-1], y_val[:,:-1]
dec_out_trn, dec_out_val = y_train[:,1:], y_val[:,1:]

# model archetect
tf_transformer = Transformer(
    num_layers=2, d_model=128, num_heads=2, dff=128,
    input_vocab_size=len(keras_tokenizer.word_index), target_vocab_size=len(keras_tokenizer.word_index),
    pe_input=config.ENC_SEQ_LEN, pe_target=config.DEC_SEQ_LEN)

encoder_inputs = keras.Input(shape=(config.ENC_SEQ_LEN,), dtype="int32", name="encoder_inputs")
decoder_inputs = keras.Input(shape=(config.DEC_SEQ_LEN,), dtype="int32", name="decoder_inputs")
enc_padding_mask, combined_mask, dec_padding_mask = layers.Lambda(create_masks)([encoder_inputs, decoder_inputs])
decoder_outputs = tf_transformer(encoder_inputs, decoder_inputs, True, enc_padding_mask, combined_mask, dec_padding_mask)
train_model = keras.Model([encoder_inputs, decoder_inputs], decoder_outputs)

loss_object = tf.keras.losses.SparseCategoricalCrossentropy(
    from_logits=True, reduction='none')
def loss_function(real, pred):
  mask = tf.math.logical_not(tf.math.equal(real, 0))
  loss_ = loss_object(real, pred)

  mask = tf.cast(mask, dtype=loss_.dtype)
  loss_ *= mask

  return tf.reduce_mean(loss_)
  
train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(
    name='ACC')

# fit
train_model.compile("adam", loss=loss_function, metrics=train_accuracy)
train_model.fit(
    x=(enc_inp_trn, dec_inp_trn), y=dec_out_trn, 
    validation_data=((enc_inp_val, dec_inp_val), dec_out_val),
    batch_size=config.BATCH_SIZE, epochs=config.EPOCHS,
    #callbacks=[GenerateTitleCallback(enc_inp_val, dec_inp_val)]
    )