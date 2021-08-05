import utils
import re
from models import config

from models.generater import load_transformer
from tensorflow.keras import preprocessing as pp
import numpy as np

bert_tokenizer, keras_tokenizer = utils.load_tokenizer()
model = load_transformer()

news_sample = '''比嘉愛未（35）主演の連ドラ「推しの王子様」（フジテレビ系木曜夜10時）。乙女ゲームを手がけるベンチャー企業の女社長・泉美が、自身の理想通りにつくった推しキャラのケント様にそっくりな航（渡辺圭祐＝27）に出会うことで始まる物語だが、当初主演予定だった深田恭子（38）が適応障害で休業することを公表し、降板。急きょ代役で泉美を演じることになったのが、比嘉だ。
比嘉愛未「2行のセリフ」が言えなかった駆け出し時代
「最近の代役といえば、NHK大河『麒麟がくる』でしょう。沢尻エリカの代役で、川口春奈が帰蝶を演じましたが、ふたを開けてみると、〈川口でよかった〉の声がほとんどだった。『推しの王子様』の比嘉も、初回を見る限り、そういう評価になりそうな予感がします。仕事ができる女社長役って、深田よりむしろ比嘉の方がハマる気がしますもんね」（ドラマ制作会社スタッフ）'''
news_sample = re.sub(r"\s", "", news_sample)
news_sample = re.sub("\u3000", "", news_sample)

def decode_title(input_sentence):
  decoder_sentence = "[BOS]"
  input_sentence = " ".join(bert_tokenizer.tokenize(input_sentence))
  encoder_sentence_ = keras_tokenizer.texts_to_sequences([input_sentence])
  encoder_sentence_ = pp.sequence.pad_sequences(encoder_sentence_, maxlen=config.ENC_SEQ_LEN, padding="post", truncating="post")
  for i in range(50):
    decoder_sentence_ = keras_tokenizer.texts_to_sequences([decoder_sentence])
    decoder_sentence_ = pp.sequence.pad_sequences(decoder_sentence_, maxlen=config.DEC_SEQ_LEN, padding="post", truncating="post")
    pred = model((encoder_sentence_, decoder_sentence_))
    sampled_tkn_idx = np.argmax(pred[0,i,:])
    sampled_tkn = keras_tokenizer.index_word[sampled_tkn_idx]
    decoder_sentence =  decoder_sentence + " " + sampled_tkn
    if sampled_tkn == "[EOS]":
      break
  return "".join(re.sub("#", "", decoder_sentence).split(" ")[1:-1])

print(decode_title(news_sample))