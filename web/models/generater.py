from .config import *
from .transformer import Transformer, create_masks
from tensorflow import keras

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

def load_model():
    return keras.models.load_model("weights/train_model_weight")

# load_model()
model = load_transformer()
print(model.summary())