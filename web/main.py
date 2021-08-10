import utils

from flask import *
from models.generater import load_transformer

bert_tokenizer, keras_tokenizer = utils.load_tokenizer()
tf_tranformer = load_transformer()

app = Flask(__name__)

@app.route('/')
def odd_even():
    if request.method == "GET":
        return render_template("index.html")
    else:
        return "ERROR"


@app.route('/generate', methods=["POST"])
def generate():
    if request.method == "POST":
        article = str(request.form["article"])
        clean_article = utils.preprocess_context(article)
        title = utils.generate_title(
            clean_article,
            [bert_tokenizer, keras_tokenizer, tf_tranformer])
        return render_template("generate.html", article = article, title=title)
    else:
        return "ERROR"


## おまじない
if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=8888, threaded=True)