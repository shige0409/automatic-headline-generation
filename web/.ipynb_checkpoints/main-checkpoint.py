from flask import *
app = Flask(__name__)

@app.route('/')
def odd_even():
    if request.method == "GET":
        return render_template("index.html")
    else:
        return "ERROR"


@app.route('/generate', methods=["POST"])
def generate_title():
    if request.method == "POST":
        article = str(request.form["article"])
        title = article[3:10]
        return render_template("generate.html", article = article, title=title)
    else:
        return "ERROR"


## おまじない
if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=8888, threaded=True)