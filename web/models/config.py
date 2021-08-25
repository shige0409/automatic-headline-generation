ENC_SEQ_LEN = 1400
DEC_SEQ_LEN = 50

NUM_LAYERS = 2
D_MODEL = 200
NUM_HEADS = 4
DDF = D_MODEL*2
VOCAB_SIZE = 29007

BATCH_SIZE = 32
EPOCHS = 10

CATEGORY_SIZE = 11

dataset_path = "scraping/data/datasets.bin"
model_path = "web/data"

label_dict = ['IT 経済', 'グルメ', 'スポーツ', 'ファッション・ビューティ', 'ライフスタイル', 'ライフ総合', '国内',
'恋愛', '海外', '芸能', '車']