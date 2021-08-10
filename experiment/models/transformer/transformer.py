import numpy as np

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

import config

def create_masks(inputs_):
	def create_padding_mask(seq):
		seq = tf.cast(tf.math.equal(seq, 0), tf.float32)

		# アテンション・ロジットにパディングを追加するため
		# さらに次元を追加する
		return seq[:, tf.newaxis, tf.newaxis, :]

	def create_look_ahead_mask(size):
		mask = 1 - tf.linalg.band_part(tf.ones((size, size)), -1, 0)
		return mask  # (seq_len, seq_len)

	inp, tar = inputs_
	# Encoderパディング・マスク
	enc_padding_mask = create_padding_mask(inp)

	# デコーダーの 2つ目のアテンション・ブロックで使用
	# このパディング・マスクはエンコーダーの出力をマスクするのに使用
	dec_padding_mask = create_padding_mask(inp)

	# デコーダーの 1つ目のアテンション・ブロックで使用
	# デコーダーが受け取った入力のパディングと将来のトークンをマスクするのに使用
	look_ahead_mask = create_look_ahead_mask(tf.shape(tar)[1])
	dec_target_padding_mask = create_padding_mask(tar)
	combined_mask = tf.maximum(dec_target_padding_mask, look_ahead_mask)

	return enc_padding_mask, combined_mask, dec_padding_mask

def positional_encoding(position, d_model):
	def get_angles(pos, i, d_model):
		angle_rates = 1 / np.power(10000, (2 * (i//2)) / np.float32(d_model))
		return pos * angle_rates
	angle_rads = get_angles(np.arange(position)[:, np.newaxis],
							np.arange(d_model)[np.newaxis, :],
							d_model)

	# 配列中の偶数インデックスにはsinを適用; 2i
	angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])

	# 配列中の奇数インデックスにはcosを適用; 2i+1
	angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])

	pos_encoding = angle_rads[np.newaxis, ...]

	return tf.cast(pos_encoding, dtype=tf.float32)

def scaled_dot_product_attention(q, k, v, mask):
	"""アテンションの重みの計算
	q, k, vは最初の次元が一致していること
	k, vは最後から2番めの次元が一致していること
	マスクは型（パディングかルックアヘッドか）によって異なるshapeを持つが、
	加算の際にブロードキャスト可能であること
	引数：
		q: query shape == (..., seq_len_q, depth)
		k: key shape == (..., seq_len_k, depth)
		v: value shape == (..., seq_len_v, depth_v)
		mask: (..., seq_len_q, seq_len_k) にブロードキャスト可能な
			shapeを持つ浮動小数点テンソル。既定値はNone

	戻り値：
		出力、アテンションの重み
	"""

	matmul_qk = tf.matmul(q, k, transpose_b=True)  # (..., seq_len_q, seq_len_k)

	# matmul_qkをスケール
	dk = tf.cast(tf.shape(k)[-1], tf.float32)
	scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)

	# マスクをスケール済みテンソルに加算
	if mask is not None:
		scaled_attention_logits += (mask * -1e9)  

	# softmax は最後の軸(seq_len_k)について
	# 合計が1となるように正規化
	attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)  # (..., seq_len_q, seq_len_k)

	output = tf.matmul(attention_weights, v)  # (..., seq_len_q, depth_v)

	return output, attention_weights

def point_wise_feed_forward_network(d_model, dff):
	return tf.keras.Sequential([
		layers.Dense(dff, activation='relu'),  # (batch_size, seq_len, dff)
		layers.Dense(d_model)  # (batch_size, seq_len, d_model)
	])


# layer
class MultiHeadAttention(layers.Layer):
	def __init__(self, d_model, num_heads):
		super(MultiHeadAttention, self).__init__()
		self.num_heads = num_heads
		self.d_model = d_model

		assert d_model % self.num_heads == 0

		self.depth = d_model // self.num_heads

		self.wq = layers.Dense(d_model)
		self.wk = layers.Dense(d_model)
		self.wv = layers.Dense(d_model)

		self.dense = layers.Dense(d_model)

	def split_heads(self, x, batch_size):
		"""最後の次元を(num_heads, depth)に分割。
		結果をshapeが(batch_size, num_heads, seq_len, depth)となるようにリシェイプする。
		"""
		x = tf.reshape(x, (batch_size, -1, self.num_heads, self.depth))
		return tf.transpose(x, perm=[0, 2, 1, 3])

	def call(self, v, k, q, mask):
		batch_size = tf.shape(q)[0]

		q = self.wq(q)  # (batch_size, seq_len, d_model)
		k = self.wk(k)  # (batch_size, seq_len, d_model)
		v = self.wv(v)  # (batch_size, seq_len, d_model)

		q = self.split_heads(q, batch_size)  # (batch_size, num_heads, seq_len_q, depth)
		k = self.split_heads(k, batch_size)  # (batch_size, num_heads, seq_len_k, depth)
		v = self.split_heads(v, batch_size)  # (batch_size, num_heads, seq_len_v, depth)

		# scaled_attention.shape == (batch_size, num_heads, seq_len_q, depth)
		# attention_weights.shape == (batch_size, num_heads, seq_len_q, seq_len_k)
		scaled_attention, attention_weights = scaled_dot_product_attention(
			q, k, v, mask)

		scaled_attention = tf.transpose(scaled_attention, perm=[0, 2, 1, 3])  # (batch_size, seq_len_q, num_heads, depth)

		concat_attention = tf.reshape(scaled_attention, 
									(batch_size, -1, self.d_model))  # (batch_size, seq_len_q, d_model)

		output = self.dense(concat_attention)  # (batch_size, seq_len_q, d_model)

		return output, attention_weights

class EncoderLayer(layers.Layer):
	def __init__(self, d_model, num_heads, dff, rate=0.1):
		super(EncoderLayer, self).__init__()

		self.mha = MultiHeadAttention(d_model, num_heads)
		self.ffn = point_wise_feed_forward_network(d_model, dff)

		self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
		self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)

		self.dropout1 = layers.Dropout(rate)
		self.dropout2 = layers.Dropout(rate)

	def call(self, x, training, mask):
		attn_output, _ = self.mha(x, x, x, mask)  # (batch_size, input_seq_len, d_model)
		attn_output = self.dropout1(attn_output, training=training)
		out1 = self.layernorm1(x + attn_output)  # (batch_size, input_seq_len, d_model)

		ffn_output = self.ffn(out1)  # (batch_size, input_seq_len, d_model)
		ffn_output = self.dropout2(ffn_output, training=training)
		out2 = self.layernorm2(out1 + ffn_output)  # (batch_size, input_seq_len, d_model)

		return out2

class DecoderLayer(layers.Layer):
	def __init__(self, d_model, num_heads, dff, rate=0.1):
		super(DecoderLayer, self).__init__()

		self.mha1 = MultiHeadAttention(d_model, num_heads)
		self.mha2 = MultiHeadAttention(d_model, num_heads)

		self.ffn = point_wise_feed_forward_network(d_model, dff)

		self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
		self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
		self.layernorm3 = layers.LayerNormalization(epsilon=1e-6)

		self.dropout1 = layers.Dropout(rate)
		self.dropout2 = layers.Dropout(rate)
		self.dropout3 = layers.Dropout(rate)


	def call(self, x, enc_output, training, look_ahead_mask, padding_mask):
		# enc_output.shape == (batch_size, input_seq_len, d_model)
		attn1, attn_weights_block1 = self.mha1(x, x, x, look_ahead_mask)  # (batch_size, target_seq_len, d_model)
		attn1 = self.dropout1(attn1, training=training)
		out1 = self.layernorm1(attn1 + x)

		attn2, attn_weights_block2 = self.mha2(
			enc_output, enc_output, out1, padding_mask)  # (batch_size, target_seq_len, d_model)
		attn2 = self.dropout2(attn2, training=training)
		out2 = self.layernorm2(attn2 + out1)  # (batch_size, target_seq_len, d_model)

		ffn_output = self.ffn(out2)  # (batch_size, target_seq_len, d_model)
		ffn_output = self.dropout3(ffn_output, training=training)
		out3 = self.layernorm3(ffn_output + out2)  # (batch_size, target_seq_len, d_model)

		return out3, attn_weights_block1, attn_weights_block2

class Encoder(layers.Layer):
	def __init__(self, num_layers, d_model, num_heads, dff, input_vocab_size,
				maximum_position_encoding, rate=0.1):
		super(Encoder, self).__init__()

		self.d_model = d_model
		self.num_layers = num_layers

		self.embedding = layers.Embedding(input_vocab_size, d_model)
		self.pos_encoding = positional_encoding(maximum_position_encoding, 
												self.d_model)


		self.enc_layers = [EncoderLayer(d_model, num_heads, dff, rate) 
						for _ in range(num_layers)]

		self.dropout = layers.Dropout(rate)

	def call(self, x, training, mask):

		seq_len = tf.shape(x)[1]

		# 埋め込みと位置エンコーディングを合算する
		x = self.embedding(x)  # (batch_size, input_seq_len, d_model)
		x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
		x += self.pos_encoding[:, :seq_len, :]

		x = self.dropout(x, training=training)

		for i in range(self.num_layers):
			x = self.enc_layers[i](x, training, mask)

		return x  # (batch_size, input_seq_len, d_model)

class Decoder(layers.Layer):
	def __init__(self, num_layers, d_model, num_heads, dff, target_vocab_size,
				maximum_position_encoding, rate=0.1):
		super(Decoder, self).__init__()

		self.d_model = d_model
		self.num_layers = num_layers

		self.embedding = layers.Embedding(target_vocab_size, d_model)
		self.pos_encoding = positional_encoding(maximum_position_encoding, d_model)

		self.dec_layers = [DecoderLayer(d_model, num_heads, dff, rate) 
						for _ in range(num_layers)]
		self.dropout = layers.Dropout(rate)

	def call(self, x, enc_output, training, 
			look_ahead_mask, padding_mask):

		seq_len = tf.shape(x)[1]
		attention_weights = {}

		x = self.embedding(x)  # (batch_size, target_seq_len, d_model)
		x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
		x += self.pos_encoding[:, :seq_len, :]

		x = self.dropout(x, training=training)

		for i in range(self.num_layers):
			x, block1, block2 = self.dec_layers[i](x, enc_output, training,
													look_ahead_mask, padding_mask)

		attention_weights['decoder_layer{}_block1'.format(i+1)] = block1
		attention_weights['decoder_layer{}_block2'.format(i+1)] = block2

		# x.shape == (batch_size, target_seq_len, d_model)
		return x, attention_weights

class Transformer(tf.keras.Model):
	def __init__(self, num_layers, d_model, num_heads, dff, input_vocab_size, 
				target_vocab_size, pe_input, pe_target, rate=0.1):
		super(Transformer, self).__init__()

		self.encoder = Encoder(num_layers, d_model, num_heads, dff, 
							input_vocab_size, pe_input, rate)

		self.decoder = Decoder(num_layers, d_model, num_heads, dff, 
							target_vocab_size, pe_target, rate)

		self.final_layer = layers.Dense(target_vocab_size)


	def call(self, inp, tar, training, enc_padding_mask, 
			look_ahead_mask, dec_padding_mask):

		enc_output = self.encoder(inp, training, enc_padding_mask)  # (batch_size, inp_seq_len, d_model)

		# dec_output.shape == (batch_size, tar_seq_len, d_model)
		dec_output, attention_weights = self.decoder(
			tar, enc_output, training, look_ahead_mask, dec_padding_mask)

		final_output = self.final_layer(dec_output)  # (batch_size, tar_seq_len, target_vocab_size)

		return final_output#, attention_weights


if __name__ == "__main__":
	tf_transformer = Transformer(
		num_layers=2, d_model=128, num_heads=2, dff=128,
		input_vocab_size=10000, target_vocab_size=10000,
		pe_input=config.ENC_SEQ_LEN, pe_target=config.DEC_SEQ_LEN)
	print(tf_transformer)