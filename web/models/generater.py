from .config import *
from .transformer import *
from tensorflow import keras

def load_transformer():
    # tf_transformer = Transformer(
    #     num_layers=NUM_LAYERS, d_model=D_MODEL, num_heads=NUM_HEADS,
    #     dff=DDF,input_vocab_size=VOCAB_SIZE, target_vocab_size=VOCAB_SIZE,
	# 	pe_input=ENC_SEQ_LEN, pe_target=DEC_SEQ_LEN
    # )
    # encoder_inputs = keras.Input(shape=(ENC_SEQ_LEN,), dtype="int32", name="encoder_inputs")
    # decoder_inputs = keras.Input(shape=(DEC_SEQ_LEN,), dtype="int32", name="decoder_inputs")
    # enc_padding_mask, combined_mask, dec_padding_mask = keras.layers.Lambda(create_masks)([encoder_inputs, decoder_inputs])
    # decoder_outputs = tf_transformer(encoder_inputs, decoder_inputs, True, enc_padding_mask, combined_mask, dec_padding_mask)
    # train_model = keras.Model([encoder_inputs, decoder_inputs], decoder_outputs)
    # train_model.load_weights("data/train_model_weight_2.4.0.h5")
    # return train_model
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
    trainer.load_weights("data/train_model_weight_2.4.0.h5")
    return trainer

# if __name__
# # load_model()
# model = load_transformer()
# print(model.summary())