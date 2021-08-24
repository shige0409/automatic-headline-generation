from tensorflow.python import util
import utils

from flask import *

bert_tokenizer, keras_tokenizer = utils.load_tokenizer()
model = utils.load_model()

app = Flask(__name__)

@app.route('/')
def home():
    if request.method == "GET":
        sample = utils.load_sample()
        return render_template(
            "index.html",
            sample=sample
            )
    else:
        return "ERROR"


@app.route('/generate', methods=["POST"])
def generate():
    if request.method == "POST":
        article = str(request.form["article"])
        clean_article = utils.preprocess_context(article)
        title, label = utils.generate_title(
            clean_article,
            [bert_tokenizer, keras_tokenizer, model],
            beam_width=int(request.form["beamwidth"]))
        title = utils.clean_beam_title(title)
        return render_template(
            "generate.html",
            article = article,
            title=title,
            label=label)
    else:
        return "ERROR"


## おまじない
if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=8888, threaded=True)