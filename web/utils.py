import pickle
import random
import re
import mojimoji

import numpy as np
import plotly.graph_objects as go

from models.config import *
from models.transformer import *

from tensorflow import keras
from tensorflow import nn
from tensorflow.keras import preprocessing as pp

from transformers import BertJapaneseTokenizer


def load_tokenizer():
    with open("./data/keras_tokenizer.bin", "rb") as f:
        kt = pickle.load(f)
    bt = BertJapaneseTokenizer.from_pretrained(
        "cl-tohoku/bert-base-japanese-whole-word-masking")

    return bt, kt


def load_model():
    encoder_inputs = keras.Input(
        shape=(ENC_SEQ_LEN,), dtype="int32", name="encoder_inputs")
    decoder_inputs = keras.Input(
        shape=(DEC_SEQ_LEN,), dtype="int32", name="decoder_inputs")
    enc_padding_mask, combined_mask, dec_padding_mask = layers.Lambda(
        create_masks)([encoder_inputs, decoder_inputs])
    # model
    pos_emb = PositionalEncoding(d_model=D_MODEL, input_vocab_size=VOCAB_SIZE)
    tf_encoder = Encoder(
        num_layers=NUM_LAYERS, d_model=D_MODEL, num_heads=NUM_HEADS, dff=DDF,
        input_vocab_size=VOCAB_SIZE, category_size=CATEGORY_SIZE, rate=0.2)
    tf_decoder = Decoder(NUM_LAYERS, D_MODEL, NUM_HEADS,
                         DDF, VOCAB_SIZE, rate=0.1)
    # feed_foward
    enc_output, enc_output_cls = tf_encoder(
        pos_emb(encoder_inputs), enc_padding_mask)
    dec_output = tf_decoder(pos_emb(decoder_inputs),
                            enc_output, combined_mask, dec_padding_mask)
    # global model
    trainer = keras.Model([encoder_inputs, decoder_inputs], [
                          dec_output, enc_output_cls], name="s2s")
    trainer.load_weights("./data/trainer_weight_multi.h5")
    return trainer


def load_sample():
    data = '''新型コロナウイルスに感染し、重度の肺炎で入院中のタレント野々村真の容体が、酸素吸入と治療薬の点滴投与で安定状態にあることが16日、分かった。月曜レギュラーを務めているフジテレビ系情報番組「バイキング」が伝えた。野々村は今月5日に入院し、重度の肺炎と診断された。14日に、妻でタレントの野々村俊恵と初めて電話で会話し、かすれた声でやりとりしたという。番組によると、9日から毎分6リットルの酸素吸入が始まり、一時は毎分10リットルに近づく時もあった。番組に出演した昭和大医学部客員教授の二木芳夫氏は「10リットルはかなり重症」とした。現在は治療薬レムデシベルの点滴投与を始め、症状に波はあるものの、中等症のまま安定した状態にあるという。MCの坂上忍は、俊恵さんから「けさメールをいただいた」とし、自宅療養期間の実態を明らかにした。それによると、酸素飽和度が90に落ちて救急車を呼んだものの、到着を待っている間に93になり、到着した時は96に回復したため、救急隊は戻っていったという。「隊員の方は、状態を見て『やばい』と感じたんだろうと思った。しかし、その場にいない保健所の方のマニュアルの指示しか聞くことができず、何度も『申し訳ない』と言って戻っていかれた」と、自宅療養当時の状況が伝えられた。
YouTubeで、ホームレスらの命を軽視した「激辛」発言をアップしたタレントが炎上した。しかし一般の芸能人と違い、炎上し、注目を集めれば集めるほどそれがアクセス数となって収入が伸びるのがネットの世界である。こうした構造上、ネット上のコンテンツにジャーナリズムが根付き、健全化するのは難しいのではないか。ネットで活躍してきたコラムニストだからこそ思うこととは……。受験勉強の延長線上で恋愛や人生を「攻略」する発想の限界メンタリストDaiGoが、自身のYouTubeチャンネルで生活保護受給者やホームレスの命を軽視した発言をしたとして炎上、波紋を呼んだ。「激辛」と付記した動画で「生活保護の人たちに食わせる金があるんだったら、猫を救ってほしいと僕は思う」「ホームレスの命はどうでもいい」「言っちゃ悪いけど、いないほうがよくない?」「じゃまだしさ、プラスになんないしさ、くさいしさ、治安悪くなるしさ」との発言に、優生思想的である、ヘイトクライムを誘発しかねない、などの指摘が殺到。生活保護に対する偏見や悪感情の深まりを懸念した厚生労働省が「生活保護は国民の権利です」と公式Twitterで発信する事態となった。「メンタリスト」との耳新しい呼び名で2010年代のテレビに現れ、数々の自己啓発書を出し、恋愛や職場などにおける人間関係で「相手の心を自由に操る」方法を伝授する、優れて器用なインテリタレントとして重宝され、活躍を続けてきたDaiGo。早期からニコニコ動画やYouTubeの発信プラットフォームとしての価値に気づき、テレビでの知名度を生かしてダントツの登録者数、動画再生数を打ち立て、巨額のサブスク・広告収入を手にするなど、いまどきの“withテレビ"配信シフト成功タレントの代表格である。だが、他人の心理にも世知にも長けてスマートな今年34歳の彼が、大学を出てから瞬く間に世間へ認知されたこの10年ほどの間に、世に向けて何を提供してきたのかをあらためて考えると、それは実に2010年代らしい、ネット発信にピタリと寄り添ったコンテンツだったことに気づく。線の細い受験エリートが、受験勉強の延長線上で恋愛や、面倒で複雑な人間関係や、いちいち自分たちを陥れようと意地悪くハードモードを仕掛けてきているとしか思えない人生を“攻略"してきたスキル系、自己啓発系発想の限界~そしていくばくかの傲慢~をそこには感じるのだ。続きはこちら
東京五輪空手女子組手61キロ級代表の植草歩選手が、選手村で撮影した「アビイ・ロード」風の記念写真を公開した。「まさに空手道」の声も植草選手は2021年8月11日、ツイッターとインスタグラムで、ハートマークの絵文字とともに、ほかの空手日本代表選手らと撮影した選手村でのショットを公開した。写真は夜の選手村の横断歩道で撮影されたものになっており、向かって左側から右側にかけて、背の低い選手から背の高い選手が、背の順で並んでいるというもの。全員笑顔できっちりと右腕をまっすぐ伸ばし、左足を踏み出しているという調和の取れたショットで、ビートルズのアルバム「アビイ・ロード」を彷彿とさせる写真になっていた。このショットに、植草選手の元には、「美しい写真です」「まさに空手道!」「仲良さそうで癒された」という声が集まっていた。
ホンチョンギ」でアン・ヒョソプが赤い目をした神秘的なイケメンに変身した。韓国で8月30日に放送がスタートするSBS新月火ドラマ「ホンチョンギ」は9日、アン・ヒョソプの撮影初日のスチールカットを公開した。公開された写真の中でアン・ヒョソプは、近づきがたい雰囲気で視線を奪う。気品が感じられる姿、物思いにふけったような表情が、なかなか近付けないイケメンのオーラを放つ。きれいな韓服姿で頭には冠を被った彼の“時代劇ビジュアル"が注目を集めた。何より平凡な人々とは違う、赤い目が彼を神秘的な存在にする。赤く輝く神秘的な瞳は不思議な雰囲気を醸し出し、見る人の好奇心を刺激する。劇中でアン・ヒョソプは、星座を読む書雲観で働くハラムを演じる。ハラムは幼い頃、雨乞祭を行っていた途中、事件に巻き込まれ目が見えなくなる人物だ。アン・ヒョソプは、赤い目のハラムの物語をドラマチックに描き、ドラマへの没入感を高めることが期待される。“ファンタジーロマンス時代劇"をビジュアル、演技で完璧に表現したアン・ヒョソプの姿が期待を高める。「ホンチョンギ」の制作陣は「撮影が進むほどより深く劇中人物に入り込むアン・ヒョソプを見ながら、制作陣も映像の中で完成する彼の演技を期待している。ハラムの神秘的な魅力がアン・ヒョソプに出会ってさらに強烈に表現されたと思う。『ホンチョンギ』でもう一度飛躍するアン・ヒョソプの新しい姿を楽しみにしてほしい」と伝えた。「ホンチョンギ」は、神霊な力を持つ女性画工ホンチョンギと、空の星座を読む赤い目の男ハラムが描く、ファンタジーロマンス時代劇だ。韓国で30日午後10時に放送がスタートする。
食楽web今年はいつから発売される?とSNS界隈で待ち望む声がみられた『セブンイレブン』の「ラムネわらび」が7月19日より登場しています。今年のコンビニスイーツはセブンイレブン初め、“わらび餅"アレンジのデザートが目白押しで大人気ですね。1個108円こちらは2017年より、幻のスイーツとして人気を呼んだ一品なのです。屋台でお馴染みの炭酸飲料『ラムネ』をぷるぷる食感の“わらび餅"にした商品です。早速、「ラムネわらび」を食べてみた!ワンハンドで手を汚さずに食べられるのも魅力的で、炭酸のシュワッとした清涼感がバッチリ再現されています。もちもちで、ぷるんっと弾力のある食感がクセになりますよ。1個目は常温のまま食べてみましたが、これは冷蔵庫でしっかり冷やして食べるのがおすすめ!よりラムネらしく、舌の上で炭酸の弾ける刺激やラムネの爽やかな風味を堪能できました。ちなみに冷凍庫でキンキンに冷やしたラムネわらびをカットして、レモンサワーやサイダーと合わせるのも一興。見た目も涼しげですし、遊び心あふれるドリンクを楽しめます。1個108円なので、まさに大人の駄菓子といった感じ!自宅で縁日に出かけた時のワクワク感を味わえました。人気商品なので、気になる方はぜひお近くのセブンイレブンへ。''''
    data = data.split("\n")
    # with open("./data/dev_datasets.bin", "rb") as f:
    #     data = pickle.load(f)
    return random.choice(data)


def encode(model, inputs, mask):
    pass


def decode(model, inputs, mask):
    pass


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


def generate_title(text, models, beam_width=3):
    bt, kt, model = models
    pre_log_prob = np.array([[0.]]*beam_width)
    decoder_sentence = ["[CLS]"] * beam_width
    input_sentence = [" ".join(bt.tokenize(text))] * beam_width
    encoder_sentence_ = kt.texts_to_sequences(input_sentence)
    encoder_sentence_ = pp.sequence.pad_sequences(
        encoder_sentence_, maxlen=ENC_SEQ_LEN, padding="post", truncating="post")
    for idx in range(50):
        decoder_sentence_ = kt.texts_to_sequences(decoder_sentence)
        decoder_sentence_ = pp.sequence.pad_sequences(
            decoder_sentence_, maxlen=DEC_SEQ_LEN, padding="post", truncating="post")
        pred, pred_label = model((encoder_sentence_, decoder_sentence_))
        log_prob_pred = np.log(nn.softmax(pred, axis=2))
        log_prob_pred = log_prob_pred[:, idx, :]
        log_prob_pred = pre_log_prob + log_prob_pred
        # 最初の生成だけ
        if idx == 0:
            log_prob_pred = log_prob_pred[idx:idx+1]
        else:
            pass
        max_log_prob_idx_with_beam = np.argsort(
            -log_prob_pred.reshape(-1))[:beam_width]
        max_log_prob_idx_with_beam = [
            divmod(idx, VOCAB_SIZE) for idx in max_log_prob_idx_with_beam]
        log_prob = [log_prob_pred[beam_idx, vocab_idx]
                    for beam_idx, vocab_idx in max_log_prob_idx_with_beam]
        pre_log_prob += np.array(log_prob)[:, np.newaxis]
        add_words = [kt.index_word[vocab_idx]
                     for _, vocab_idx in max_log_prob_idx_with_beam]

        decoder_sentence = [t1 + " " + t2 for t1,
                            t2 in zip(decoder_sentence, add_words)]
        if "[SEP]" in add_words:
            break
    # 画像書き出し
    fig = go.Figure(
        data=[go.Pie(labels=label_dict, values=pred_label[0].numpy())])
    fig.write_image("./static/images/test.png")
    return decoder_sentence


def clean_beam_title(titles):
    clean_titles = ["".join(re.sub("#", "", t).split(" ")[1:-1])
                    for t in titles]
    return clean_titles
