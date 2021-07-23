import flask
from flask.helpers import send_from_directory
from flask import render_template, request
from models.fake_model import fake_model
from read_csv import retrieve_id

app = flask.Flask(__name__)
app.config["DEBUG"] = True


SELECTED_ARTICLES = []
SELECTED_MODEL = fake_model


@app.route("/", methods=["GET", "POST"])
def home():
    global SELECTED_ARTICLES
    if request.method == "POST":
        articles = SELECTED_MODEL(request.form.to_dict()["query"])
        SELECTED_ARTICLES = [retrieve_id(id) for id in articles]
    return render_template("index.html", articles=SELECTED_ARTICLES)


app.run()
